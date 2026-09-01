import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict, get_optimizer_state_dict,
    set_model_state_dict, set_optimizer_state_dict,
    StateDictOptions,
)

import os
import inspect
import wandb
import glob

import math
from tqdm import tqdm


from transformers import AutoTokenizer
from packed_dataset import PackedTokenDataset

from litgpt_model import GPT
from litgpt_config import Config

# learning rate decay scheduler (cosine with warmup)
# This is simpler to reason about than two sequential lr schedulers
def get_lr(it,
           warmup_iters,
           lr_decay_iters,
           max_lr):
    min_lr = max_lr/10 # as per chinchilla
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return max_lr * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (max_lr - min_lr)

def get_custom_lr(it,
                  warmup_iters,
                  total_iters,
                  max_lr,
                  cutoff_points):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return max_lr * it / warmup_iters
    # 2) Otherwise get lr according to piecewise linear custom schedule
    lr_mult = get_piecewise_linear_mult(it, total_iters, cutoff_points)
    return lr_mult * max_lr

def get_piecewise_linear_mult(it, total_iters, cutoff_points):
    train_progress = it/total_iters
    if train_progress >= cutoff_points[-1][0]:
        return cutoff_points[-1][1]
    if train_progress <= cutoff_points[0][0]:
        return cutoff_points[0][1]

    lower_bound_idx = 0
    while lower_bound_idx < len(cutoff_points) - 1 and train_progress >= cutoff_points[lower_bound_idx + 1][0]:
        lower_bound_idx += 1

    x1, y1 = cutoff_points[lower_bound_idx][0], cutoff_points[lower_bound_idx][1]
    x2, y2 = cutoff_points[lower_bound_idx + 1][0], cutoff_points[lower_bound_idx + 1][1]

    mult = y1 + (y2 - y1)*(train_progress - x1)/(x2 - x1)
    return mult

def get_num_params(model):
    """
    Return the number of parameters in the model.
    For non-embedding count (default), the position embeddings get subtracted.
    The token embeddings would too, except due to the parameter sharing these
    params are actually used as weights in the final layer, so we include them.
    """
    n_params = sum(p.numel() for p in model.parameters())
    non_embed_params = n_params - model.transformer.wte.weight.numel()
    return n_params, non_embed_params

def configure_optimizers(model, weight_decay, learning_rate, betas, device_type,
                         only_include_params=None,
                         exclude_params=None,
                         rank=0,
                         use_fsdp=False):
    # start with all of the candidate parameters
    if only_include_params is not None:
        param_dict = {pn: p for pn, p in model.named_parameters() if pn in only_include_params}
    elif exclude_params is not None:
        param_dict = {pn: p for pn, p in model.named_parameters() if pn not in exclude_params}
    else:
        param_dict = {pn: p for pn, p in model.named_parameters()}

    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    if rank == 0:
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer: fused for single-GPU, foreach for FSDP
    if use_fsdp:
        extra_args = dict(foreach=True)
    else:
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=1e-6, **extra_args)
    if rank == 0:
        print(f"AdamW mode: {'foreach' if use_fsdp else 'fused' if extra_args.get('fused') else 'default'}")

    return optimizer



def forward_pass(model, tok_seqs, args, return_hidden_states=False):
    seq_len = tok_seqs.size(1)

    attn_mask = torch.ones((seq_len, seq_len), device=tok_seqs.device, dtype=torch.long).tril(diagonal=0)

    ret_dict = model(tok_seqs,
                        mask=attn_mask,
                        num_loops=args.num_loops,
                        first_recur_layer=args.first_recur_layer,
                        last_recur_layer=args.last_recur_layer,
                        return_hidden_states=return_hidden_states,
                        )

    return ret_dict

@hydra.main(version_base=None, config_path="./exp_configs", config_name="default_config")
def main(args: DictConfig):
    # Attention Argument handling
    if args.d_head is None:
        args.d_head = args.d_model // args.n_heads
    if args.n_q_heads is None:
        args.n_q_heads = args.n_heads
    if args.n_kv_heads is None:
        args.n_kv_heads = args.n_heads
    if args.last_recur_layer is None:
        args.last_recur_layer = args.n_layers - 1

    assert args.init_scheme in ["gpt2", "pytorch"], "Unsupported init scheme!"

    if args.use_mhc:
        assert args.n_res_streams is not None, "use_mhc requires n_res_streams to be set!"

    if args.n_res_streams is not None:
        assert not (args.mix_rs_after_layers and args.mix_rs_after_loops), "Cannot set both mix_rs_after_layers and mix_rs_after_loops!"
        assert args.mix_rs_every_l_layers is None or not (args.mix_rs_after_layers or args.mix_rs_after_loops), "Cannot set mix_rs_every_l_layers along with legacy mix_rs_after_layers/mix_rs_after_loops flags!"
        assert args.rs_mixing_strat in ["cayley", "sinkhorn", "identity", "linear", "diagonal_gate", "stream_indep_gate"], "Unsupported residual stream mixing strat!"
        assert args.premap_mixing_strat in ["default", "linear"], "Unsupported premap mixing strat!"
        assert args.postmap_mixing_strat in ["default", "linear"], "Unsupported postmap mixing strat!"
        # Resolve mix_rs_every_l_layers from legacy flags if not explicitly set
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

    torch.multiprocessing.set_start_method('spawn') # to allow using cuda with multiprocessing
    # DDP boilerplate
    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    torch.cuda.set_device(device)  # required for FSDP2 device mesh inference
    use_fsdp = world_size > 1

    assert args.global_batch_size % world_size == 0, \
        f"global_batch_size ({args.global_batch_size}) must be divisible by world_size ({world_size})"
    per_rank_batch_size = args.global_batch_size // world_size

    # Initialize wandb
    if rank == 0:
        if not args.debug:
            wandb.init(config=OmegaConf.to_container(args, resolve=True),
                       project=args.wandb_proj_name,
                       resume="allow",
                       id=args.run_id,
                       name=args.exp_name)

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tok_path)
    if rank == 0:
        print(tokenizer.eos_token)
        print(tokenizer.eos_token_id)
        print(len(tokenizer))
        print("Loaded tokenizer!")

    with open_dict(args):
        args.vocab_size = len(tokenizer)
        if rank == 0:
            print(args.vocab_size)

    # Load the dataset
    assert args.dataset_name is not None, "dataset_name must be specified"
    assert args.max_train_tokens is not None, \
        "max_train_tokens must be specified when using streaming datasets"
    global_steps = math.ceil(args.max_train_tokens / (args.global_batch_size * args.max_seq_len))

    train_dataset = PackedTokenDataset(
        dataset_name=args.dataset_name,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        split="train",
        seed=args.seed,
        rank=rank,
        world_size=world_size,
        use_locally_cached_dataset=args.use_locally_cached_dataset,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=per_rank_batch_size,
        num_workers=0,
    )

    # Slightly modified version of the Llama-2-7b config
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

    # Seed before construction/init so all ranks draw identical weights
    torch.manual_seed(args.seed)
    model = GPT(cfg)
    model.init_weights(args.init_scheme)
    # Model needs to be moved onto new device for device-specific optimizations
    model = model.to(device, dtype=torch.bfloat16)

    total_params, non_embed_params = get_num_params(model)
    if rank == 0:
        print(f"Training a model with:\n{total_params} parameters,\n{non_embed_params} non-embed parameters")

    checkpoint_dir = os.path.join(args.save_dir, args.run_id)

    if use_fsdp:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
        )
        for i in range(len(model.transformer.h)):
            fully_shard(model.transformer.h[i], mp_policy=mp_policy)
        fully_shard(model, mp_policy=mp_policy)

    optimizer = configure_optimizers(model,
                                     weight_decay=args.weight_decay,
                                     learning_rate=args.learning_rate,
                                     betas=(args.beta1, args.beta2),
                                     device_type='cuda',
                                     rank=rank,
                                     use_fsdp=use_fsdp)


    # global_steps is computed in the dataset loading block above
    if args.warmup_iters is not None:
        warmup_iters = args.warmup_iters
    else:
        warmup_iters = math.floor(args.warmup_ratio*global_steps)
    lr_decay_iters = global_steps

    loss_fn = nn.CrossEntropyLoss(reduction='none')

    model.train()
    optimizer.zero_grad(set_to_none=True)

    save_every_steps = args.save_every_batches*args.global_batch_size

    # Resume training if possible
    last_saved_step_count = 0
    has_dataset_state = False
    checkpoint_dir = os.path.join(args.save_dir, args.run_id)
    os.makedirs(checkpoint_dir, exist_ok=True)

    cfg_path = os.path.join(checkpoint_dir, "train_config.yaml")
    if not os.path.exists(cfg_path):
        with open(cfg_path, "w") as cfg_file:
            OmegaConf.save(config=args, f=cfg_file)

    if os.path.exists(checkpoint_dir):
        checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "*.tar")))
        if len(checkpoints) > 0:
            if rank == 0:
                print(f"Resuming from checkpoint: {checkpoints[-1]}")
            latest_checkpoint = torch.load(checkpoints[-1], map_location=device)
            last_saved_step_count = latest_checkpoint['step_count']

            if use_fsdp:
                set_model_state_dict(model, latest_checkpoint['model_state_dict'],
                                     options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True))
                set_optimizer_state_dict(model, optimizer, latest_checkpoint['optimizer_state_dict'],
                                         options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True))
            else:
                model.load_state_dict(latest_checkpoint['model_state_dict'])
                optimizer.load_state_dict(latest_checkpoint['optimizer_state_dict'])

            # Restore dataset iterator state if available (avoids slow skip loop)
            if args.save_dataset_state and args.dataset_name is not None:
                dataset_states = latest_checkpoint.get('dataset_states')
                if (dataset_states is not None
                        and len(dataset_states) == world_size
                        and dataset_states[rank] is not None):
                    train_dataset.load_checkpoint_state(dataset_states[rank])
                    has_dataset_state = True
                    if rank == 0:
                        print(f"Dataset state found in checkpoint, skipping data fast-forward")
                elif dataset_states is not None and len(dataset_states) != world_size and rank == 0:
                    print(f"[WARNING] Dataset state has {len(dataset_states)} entries but world_size={world_size}, falling back to skip loop")

    model = torch.compile(model)

    # When dataset state is available, skip the data-iteration loop entirely
    if has_dataset_state:
        step_count = last_saved_step_count
        start_step_idx = last_saved_step_count // args.global_batch_size
    else:
        step_count = 0
        start_step_idx = 0

    train_iter = iter(train_loader)

    for global_step_idx in tqdm(range(start_step_idx, global_steps),
                                initial=start_step_idx, total=global_steps,
                                disable=(rank != 0)):
        try:
            inputs = next(train_iter)
        except StopIteration:
            if rank == 0:
                print(f"[WARNING] Dataset exhausted at step {global_step_idx} (step_count={step_count}). Restarting iterator.")
            train_iter = iter(train_loader)
            inputs = next(train_iter)

        # Fall back to skip loop for old checkpoints without dataset state
        if not has_dataset_state and step_count < last_saved_step_count:
            step_count += args.global_batch_size
            # Periodic barrier to keep NCCL heartbeat alive during long skips
            if use_fsdp and global_step_idx % 1000 == 0:
                torch.distributed.barrier()
            continue

        global_tok_seqs = inputs['tok_seqs'].to(device)

        global_labels = global_tok_seqs[..., 1:].reshape(-1).contiguous()
        global_num_train_toks = global_labels.numel()


        total_llm_loss = 0
        grad_norm = None

        for micro_batch_start_idx in range(0, global_tok_seqs.size(0), args.micro_batch_size):
            tok_seqs = global_tok_seqs[micro_batch_start_idx:(micro_batch_start_idx + args.micro_batch_size), :]

            ret_dict = forward_pass(model, tok_seqs, args)

            outputs = ret_dict["outputs"][:, :-1, :]
            labels = tok_seqs.clone()[..., 1:] 
            batch_size, seq_len = tok_seqs.size()

            llm_loss = loss_fn(outputs.reshape(-1, args.padded_vocab_size), labels.reshape(-1))

            # Update lr according to lr schedule:
            if args.custom_lr_sched is not None:
                lr = get_custom_lr(it = (step_count//args.global_batch_size),
                                   warmup_iters = warmup_iters,
                                   total_iters=global_steps,
                                   max_lr=args.learning_rate,
                                   cutoff_points=args.custom_lr_sched)
            else: 
                lr = get_lr(it = (step_count//args.global_batch_size),
                   warmup_iters = warmup_iters,
                   lr_decay_iters = lr_decay_iters,
                   max_lr=args.learning_rate)
            if args.debug and rank == 0:
                print(f"lr: {lr}")

            # Set learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            tokenwise_logprobs = -llm_loss.reshape(batch_size, -1)
            sentencewise_logprobs = torch.sum(tokenwise_logprobs, dim=-1) # log(p(x))

            llm_loss = -torch.sum(sentencewise_logprobs)/global_num_train_toks # -log(p(x))

            total_llm_loss += llm_loss.item()

            if use_fsdp:
                is_last_micro_batch = (micro_batch_start_idx + args.micro_batch_size >= global_tok_seqs.size(0))
                model.set_requires_gradient_sync(is_last_micro_batch)
            llm_loss.backward()


            if args.debug and rank == 0:
                print(f"loss: {total_llm_loss}")

            step_count += batch_size * world_size

        assert step_count % args.global_batch_size == 0, \
            f"step_count ({step_count}) is not aligned to global_batch_size ({args.global_batch_size})"
        if args.max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


        # Log every batch
        if rank == 0:
            if not args.debug:
                wandb.log({"train_lm_loss": total_llm_loss}, step=step_count)
                wandb.log({"lr": lr}, step=step_count)

                if grad_norm is not None:
                    wandb.log({"grad_norm": grad_norm}, step=step_count)

        # Save to disk every n batches (outside rank guard — FSDP gather is collective)
        if step_count % save_every_steps == 0:
            # Gather dataset states from all ranks (collective when FSDP)
            if args.save_dataset_state and args.dataset_name is not None:
                if use_fsdp:
                    all_dataset_states = [None] * world_size
                    torch.distributed.all_gather_object(all_dataset_states, train_dataset.get_checkpoint_state())
                else:
                    all_dataset_states = [train_dataset.get_checkpoint_state()]
            else:
                all_dataset_states = None

            if use_fsdp:
                sd_options = StateDictOptions(full_state_dict=True, cpu_offload=True)
                model_state = get_model_state_dict(model, options=sd_options)
                optim_state = get_optimizer_state_dict(model, optimizer, options=sd_options)
                checkpoint_data = {
                    "step_count": step_count,
                    "model_state_dict": model_state,
                    "optimizer_state_dict": optim_state,
                    "dataset_states": all_dataset_states,
                }
                if rank == 0:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(checkpoint_data, os.path.join(checkpoint_dir, f"{step_count:010}.tar"))
                    if args.max_num_checkpoints is not None:
                        all_checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "*.tar")))
                        if len(all_checkpoints) > args.max_num_checkpoints:
                            for chkpt in all_checkpoints[:-args.max_num_checkpoints]:
                                os.remove(chkpt)
            elif rank == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_data = {
                            "step_count": step_count,
                            "model_state_dict": {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()},
                            "optimizer_state_dict": optimizer.state_dict(),
                            "dataset_states": all_dataset_states,
                            }
                torch.save(checkpoint_data, os.path.join(checkpoint_dir, f"{step_count:010}.tar"))
                if args.max_num_checkpoints is not None:
                    all_checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "*.tar")))
                    if len(all_checkpoints) > args.max_num_checkpoints:
                        for chkpt in all_checkpoints[:-args.max_num_checkpoints]:
                            os.remove(chkpt)

    # Final save
    if args.save_dataset_state and args.dataset_name is not None:
        if use_fsdp:
            all_dataset_states = [None] * world_size
            torch.distributed.all_gather_object(all_dataset_states, train_dataset.get_checkpoint_state())
        else:
            all_dataset_states = [train_dataset.get_checkpoint_state()]
    else:
        all_dataset_states = None

    if use_fsdp:
        sd_options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        model_state = get_model_state_dict(model, options=sd_options)
        optim_state = get_optimizer_state_dict(model, optimizer, options=sd_options)
        checkpoint_data = {
            "step_count": step_count,
            "model_state_dict": model_state,
            "optimizer_state_dict": optim_state,
            "dataset_states": all_dataset_states,
        }
        if rank == 0:
            torch.save(checkpoint_data, os.path.join(checkpoint_dir, f"{step_count:010}_final.tar"))

    elif rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_data = {
                    "step_count": step_count,
                    "model_state_dict": {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "dataset_states": all_dataset_states,
                    }
        torch.save(checkpoint_data, os.path.join(checkpoint_dir, f"{step_count:010}_final.tar"))

        # Remove old checkpoints
        if args.max_num_checkpoints is not None:
            all_checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "*.tar")))
            if len(all_checkpoints) > args.max_num_checkpoints:
                for chkpt in all_checkpoints[:-args.max_num_checkpoints]:
                    os.remove(chkpt)

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
