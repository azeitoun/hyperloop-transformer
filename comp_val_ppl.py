import sys
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict,
    StateDictOptions,
)

import os
import glob
import math
from tqdm import tqdm

from transformers import AutoTokenizer
from packed_dataset import PackedTokenDataset
from train import forward_pass

from filelock import FileLock

from litgpt_model import GPT
from litgpt_config import Config


def get_num_params(model):
    total = sum(p.numel() for p in model.parameters())
    non_embed = sum(p.numel() for n, p in model.named_parameters()
                    if "wte" not in n)
    return total, non_embed


def main(args):
    torch.multiprocessing.set_start_method('spawn')
    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    torch.cuda.set_device(device)
    use_fsdp = world_size > 1

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tok_path)
    with open_dict(args):
        args.vocab_size = len(tokenizer)

    # Validation dataset
    val_dataset = PackedTokenDataset(
        dataset_name=args.dataset_name,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        split="validation",
        seed=args.seed,
        rank=rank,
        world_size=world_size,
        use_locally_cached_dataset=args.use_locally_cached_dataset,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.micro_batch_size, num_workers=0)
    val_steps = math.ceil(args.max_val_tokens / (args.micro_batch_size * args.max_seq_len)) if args.max_val_tokens else None

    # Resolve mix_rs_every_l_layers from legacy flags if not explicitly set
    if args.n_res_streams is not None:
        layers_per_loop = args.last_recur_layer - args.first_recur_layer + 1
        total_looped_layers = layers_per_loop * args.num_loops
        l = None
        if args.mix_rs_every_l_layers is not None:
            l = args.mix_rs_every_l_layers
        elif args.mix_rs_after_loops:
            l = layers_per_loop
        elif args.mix_rs_after_layers:
            l = 1

        with open_dict(args):
            args.mix_rs_every_l_layers = l

        assert args.mix_rs_every_l_layers is not None, "n_res_streams requires one of mix_rs_after_loops, mix_rs_after_layers, or mix_rs_every_l_layers to be set!"
        assert total_looped_layers % args.mix_rs_every_l_layers == 0, f"Total looped layers ({total_looped_layers}) must be divisible by mix_rs_every_l_layers ({args.mix_rs_every_l_layers})!"

    # Model
    cfg = Config(name='Llama-2-7b-hf',
        hf_config={'name': 'Llama-2-7b-hf', 'org': 'meta-llama'},
        scale_embeddings=False,
        attention_scores_scalar=None,
        block_size=args.max_seq_len,
        sliding_window_size=None,
        sliding_window_layer_placing=None,
        vocab_size=args.vocab_size,
        padding_multiple=64,
        padded_vocab_size=None, # Find the padded size automatically
        n_layer=args.n_layers,
        n_head=args.n_heads,
        head_size=None, # Find the head size automatically
        n_embd=args.d_model,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        lm_head_bias=False,
        n_query_groups=None,
        shared_attention_norm=False,
        norm_class_name='RMSNorm',
        post_attention_norm=False,
        post_mlp_norm=False,
        norm_eps=1e-05,
        mlp_class_name='LLaMAMLP',
        gelu_approximate='none',
        # Relationship between d_model and intermediate_size appears to be black magic
        intermediate_size=args.d_ff, 
        rope_condense_ratio=1,
        rope_base=10000,
        rope_adjustments=None,
        n_expert=0,
        n_expert_per_token=0,
        attention_logit_softcapping=None,
        final_logit_softcapping=None,
        first_recur_layer=args.first_recur_layer,
        last_recur_layer=args.last_recur_layer,
        max_depth=args.last_recur_layer - args.first_recur_layer + 1,
        num_loops=args.num_loops,
        looping_strategy=args.looping_strategy,
        lora_rank=args.lora_rank,
        lora_attn_rank=args.lora_attn_rank,
        lora_alpha=args.lora_alpha,
        lora_attn_alpha=args.lora_attn_alpha,
        n_res_streams=args.n_res_streams,
        mix_rs_after_loops=args.mix_rs_after_loops,
        mix_rs_after_layers=args.mix_rs_after_layers,
        mix_rs_every_l_layers=args.mix_rs_every_l_layers,
        rs_mixing_strat=args.rs_mixing_strat,
        premap_mixing_strat=args.premap_mixing_strat,
        postmap_mixing_strat=args.postmap_mixing_strat,
        duplicate_init_res_proj=args.duplicate_init_res_proj,
        init_res_proj_rank=args.init_res_proj_rank,
        average_fin_res_proj=args.average_fin_res_proj,
        fin_res_proj_rank=args.fin_res_proj_rank,
        add_premap_pos_embeds=args.add_premap_pos_embeds,
        add_postmap_pos_embeds=args.add_postmap_pos_embeds,
        use_mhc=args.use_mhc,
        )

    with open_dict(args):
        args.padded_vocab_size = cfg.padded_vocab_size

    if rank == 0:
        print(cfg)

    model = GPT(cfg).to(device, dtype=torch.bfloat16)

    total_params, non_embed_params = get_num_params(model)
    if rank == 0:
        print(f"Model with:\n{total_params} parameters,\n{non_embed_params} non-embed parameters")

    # FSDP2 wrapping
    if use_fsdp:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
        )
        for i in range(len(model.transformer.h)):
            fully_shard(model.transformer.h[i], mp_policy=mp_policy)
        fully_shard(model, mp_policy=mp_policy)

    # Load checkpoint
    if args.eval_checkpoint_path is not None:
        checkpoint_path = args.eval_checkpoint_path
        if not os.path.exists(checkpoint_path):
            if rank == 0:
                print(f"Checkpoint does not exist: {checkpoint_path}")
            torch.distributed.destroy_process_group()
            return
    else:
        checkpoint_dir = os.path.join(args.save_dir, args.run_id)
        if not os.path.exists(checkpoint_dir):
            if rank == 0:
                print(f"Run directory does not exist: {checkpoint_dir}")
            torch.distributed.destroy_process_group()
            return

        checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "*.tar")))
        if len(checkpoints) == 0:
            if rank == 0:
                print(f"No checkpoints found in {checkpoint_dir}")
            torch.distributed.destroy_process_group()
            return
        checkpoint_path = checkpoints[-1]

    if rank == 0:
        print(f"Evaluating checkpoint: {checkpoint_path}")
    latest_checkpoint = torch.load(checkpoint_path, map_location=device)

    if use_fsdp:
        set_model_state_dict(model, latest_checkpoint['model_state_dict'],
                             options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True))
    else:
        model_state = {k.replace("_orig_mod.", ""): v for k, v in latest_checkpoint['model_state_dict'].items()}
        model.load_state_dict(model_state)

    # torch.compile after checkpoint load
    model = torch.compile(model)

    # Evaluation
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for step_idx, inputs in enumerate(tqdm(val_loader, disable=(rank != 0))):
            if val_steps is not None and step_idx >= val_steps:
                break
            tok_seqs = inputs['tok_seqs'].to(device)
            ret_dict = forward_pass(model, tok_seqs, args)
            outputs = ret_dict["outputs"][:, :-1, :].reshape(-1, args.padded_vocab_size)
            labels = tok_seqs[:, 1:].reshape(-1)
            loss = loss_fn(outputs, labels)
            total_loss += loss.sum().item()
            total_tokens += labels.numel()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')

    if rank == 0:
        print(f"Validation loss: {avg_loss:.4f}")
        print(f"Perplexity: {perplexity:.4f}")
        print(f"Total tokens: {total_tokens}")

        if args.eval_out_file_path is not None:
            lock = FileLock(args.eval_out_file_path + ".lock")
            with lock:
                with open(args.eval_out_file_path, "a") as f:
                    f.write(f"\n================================================\n")
                    f.write(f"{checkpoint_path}\n")
                    f.write(f"loss: {avg_loss:.4f}\n")
                    f.write(f"perplexity: {perplexity:.4f}\n")
                    f.write(f"total_tokens: {total_tokens}\n")

    torch.distributed.destroy_process_group()


def load_config_with_overrides(args):
    """
    Load a Hydra config from a file and override with command-line arguments.

    Example usage:
        python script.py --config config.yaml learning_rate=0.001 batch_size=32
    """
    config_path = None
    overrides = []

    i = 0
    while i < len(args):
        if args[i] == "--config":
            if i + 1 < len(args):
                config_path = args[i + 1]
                i += 2
            else:
                raise ValueError("--config requires a path argument")
        else:
            overrides.append(args[i])
            i += 1

    if not config_path:
        config_path = "./exp_configs/default_config.yaml"

    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config_dir = str(config_path.parent)
    config_name = config_path.stem

    print(f"Loading config from: {config_path}")
    print(f"Overrides: {overrides}")

    # Load default config first, then merge the user-provided config on top
    # so that fields missing from old run configs fall back to defaults.
    # CLI overrides are applied last so they work even for keys absent from the saved config.
    default_cfg = OmegaConf.load("./exp_configs/default_config.yaml")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        user_cfg = compose(config_name=config_name)

    cfg = OmegaConf.merge(default_cfg, user_cfg)

    # Apply CLI overrides on the merged config
    cli_cfg = OmegaConf.from_dotlist(overrides)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    print(f"\nFinal config:\n{OmegaConf.to_yaml(cfg)}")

    return cfg

if __name__ == "__main__":
    args = sys.argv[1:]
    args = load_config_with_overrides(args)
    main(args)
