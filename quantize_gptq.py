import sys
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import glob
from tqdm import tqdm

from transformers import AutoTokenizer
from packed_dataset import PackedTokenDataset
from train import forward_pass

from litgpt_model import GPT
from litgpt_config import Config


class GPTQQuantizer:
    """GPTQ quantizer for a single nn.Linear layer.

    Implements the GPTQ algorithm (Frantar et al., 2022) which quantizes
    weight columns sequentially, using the inverse Hessian to optimally
    adjust remaining columns and minimize output error.
    """

    def __init__(self, layer, bits=4, group_size=128):
        self.layer = layer
        self.bits = bits
        self.group_size = group_size if group_size > 0 else layer.weight.shape[1]
        self.dev = layer.weight.device

        self.rows = layer.weight.shape[0]   # out_features
        self.columns = layer.weight.shape[1]  # in_features

        self.H = torch.zeros((self.columns, self.columns), device=self.dev, dtype=torch.float32)
        self.n_samples = 0

    def add_batch(self, inp):
        """Accumulate Hessian from input activations. inp: [..., in_features]"""
        inp = inp.reshape(-1, inp.shape[-1]).float()
        self.H += inp.T @ inp
        self.n_samples += inp.shape[0]

    def quantize(self):
        """Run GPTQ. Modifies layer.weight.data in-place.
        Returns average per-row quantization loss.
        """
        W = self.layer.weight.data.clone().float()
        H = self.H / self.n_samples

        # Zero out dead columns (never activated during calibration)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        # Damping for numerical stability
        damp = 0.01 * torch.diag(H).mean()
        diag_idx = torch.arange(self.columns, device=self.dev)
        H[diag_idx, diag_idx] += damp

        # Inverse Hessian via Cholesky
        try:
            H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
        except RuntimeError:
            # Extra damping if Cholesky fails
            H[diag_idx, diag_idx] += 0.1 * torch.diag(H).mean()
            try:
                H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
            except RuntimeError:
                H_inv = torch.linalg.pinv(H)

        try:
            Hinv_chol = torch.linalg.cholesky(H_inv, upper=True)
        except RuntimeError:
            # If H_inv is not PD (e.g. from pinv), add small diagonal
            H_inv[diag_idx, diag_idx] += 1e-6
            Hinv_chol = torch.linalg.cholesky(H_inv, upper=True)

        Q = torch.zeros_like(W)
        Losses = torch.zeros_like(W)
        qmax = (1 << self.bits) - 1
        blocksize = 128

        cur_scale = None
        cur_zero = None

        for col_start in range(0, self.columns, blocksize):
            col_end = min(col_start + blocksize, self.columns)
            count = col_end - col_start

            W1 = W[:, col_start:col_end].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv_chol[col_start:col_end, col_start:col_end]

            for i in range(count):
                col_idx = col_start + i
                w = W1[:, i]
                d = Hinv1[i, i]

                # Compute per-group quantization parameters at group boundaries
                if col_idx % self.group_size == 0:
                    group_end = min(col_idx + self.group_size, self.columns)
                    W_group = W[:, col_idx:group_end]
                    w_min = W_group.min(dim=1).values
                    w_max = W_group.max(dim=1).values
                    cur_scale = (w_max - w_min).clamp(min=1e-5) / qmax
                    cur_zero = torch.round(-w_min / cur_scale).clamp(0, qmax)

                # Quantize and dequantize
                q = torch.clamp(torch.round(w / cur_scale + cur_zero), 0, qmax)
                q = cur_scale * (q - cur_zero)

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                # Error correction: adjust remaining columns
                err = (w - q) / d
                Err1[:, i] = err
                W1[:, i + 1:] -= err.unsqueeze(1) * Hinv1[i, i + 1:].unsqueeze(0)

            Q[:, col_start:col_end] = Q1
            Losses[:, col_start:col_end] = Losses1 / 2

            # Propagate error to columns beyond this block
            W[:, col_end:] -= Err1 @ Hinv_chol[col_start:col_end, col_end:]

        self.layer.weight.data = Q.to(self.layer.weight.dtype)
        return Losses.sum().item() / self.rows

    def free(self):
        """Free Hessian memory."""
        del self.H
        self.H = None


def get_num_params(model):
    total = sum(p.numel() for p in model.parameters())
    non_embed = sum(p.numel() for n, p in model.named_parameters()
                    if "wte" not in n)
    return total, non_embed


def main(args):
    # Single-GPU setup (GPTQ modifies weights in-place sequentially)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tok_path)
    with open_dict(args):
        args.vocab_size = len(tokenizer)

    # Calibration dataset (use training split to avoid biasing validation metrics)
    cal_dataset = PackedTokenDataset(
        dataset_name=args.dataset_name,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        split="train",
        seed=args.seed,
        rank=0,
        world_size=1,
        use_locally_cached_dataset=args.use_locally_cached_dataset,
    )
    cal_loader = DataLoader(cal_dataset, batch_size=args.micro_batch_size, num_workers=0)

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
        padded_vocab_size=None,
        n_layer=args.n_layers,
        n_head=args.n_heads,
        head_size=None,
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

    print(cfg)

    model = GPT(cfg).to(device, dtype=torch.bfloat16)

    total_params, non_embed_params = get_num_params(model)
    print(f"Model with:\n{total_params} parameters,\n{non_embed_params} non-embed parameters")

    # Load checkpoint (single GPU, no FSDP wrapping, no torch.compile)
    if args.eval_checkpoint_path is not None:
        checkpoint_path = args.eval_checkpoint_path
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint does not exist: {checkpoint_path}")
            return
    else:
        checkpoint_dir = os.path.join(args.save_dir, args.run_id)
        if not os.path.exists(checkpoint_dir):
            print(f"Run directory does not exist: {checkpoint_dir}")
            return

        checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "*.tar")))
        if len(checkpoints) == 0:
            print(f"No checkpoints found in {checkpoint_dir}")
            return
        checkpoint_path = checkpoints[-1]

    print(f"Loading checkpoint: {checkpoint_path}")
    latest_checkpoint = torch.load(checkpoint_path, map_location=device)

    model_state = {k.replace("_orig_mod.", ""): v for k, v in latest_checkpoint['model_state_dict'].items()}
    model.load_state_dict(model_state)

    model.eval()

    # === GPTQ Quantization ===
    bits = args.gptq_bits
    group_size = args.gptq_group_size
    n_batches = args.gptq_n_batches

    # Create quantizers and register hooks on all linear layers (except lm_head).
    # For looped models, hooks on the looped blocks' linear layers fire num_loops
    # times per forward pass, so the Hessian naturally aggregates inputs across
    # all loop iterations. This produces quantization that accounts for the full
    # distribution of inputs the shared weights see during inference.
    HC_PATTERNS = ('init_res_proj', 'fin_res_proj', 'pre_proj', 'res_proj', 'post_proj', 'mhc_')
    skip_hc = args.gptq_skip_hc

    quantizers = {}
    hooks = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or name == 'lm_head':
            continue
        if skip_hc and any(p in name for p in HC_PATTERNS):
            continue
        q = GPTQQuantizer(module, bits, group_size)
        quantizers[name] = q
        hooks.append(module.register_forward_hook(
            lambda mod, inp, out, qz=q: qz.add_batch(inp[0].data)
        ))

    print(f"\nGPTQ Configuration:")
    print(f"  Bits: {bits}")
    print(f"  Group size: {group_size}")
    print(f"  Calibration batches: {n_batches}")
    print(f"  Skip hyperconnection weights: {skip_hc}")
    print(f"  Layers to quantize: {len(quantizers)}")

    if args.num_loops > 1:
        layers_per_loop = args.last_recur_layer - args.first_recur_layer + 1
        print(f"  Looped model: {layers_per_loop} layers x {args.num_loops} loops")
        print(f"  Hessians aggregate inputs across all {args.num_loops} loop iterations")

    # Calibration: run forward passes to collect input statistics
    print(f"\nCollecting calibration data...")
    with torch.no_grad():
        for step_idx, inputs in enumerate(tqdm(cal_loader, desc="Calibrating")):
            if step_idx >= n_batches:
                break
            tok_seqs = inputs['tok_seqs'].to(device)
            forward_pass(model, tok_seqs, args)

    for h in hooks:
        h.remove()

    # Quantize each layer using collected Hessians
    print(f"\nQuantizing {len(quantizers)} layers...")
    total_loss = 0
    for name, q in tqdm(quantizers.items(), desc="Quantizing"):
        if q.n_samples == 0:
            print(f"  {name}: SKIPPED (no calibration inputs)")
            continue
        loss = q.quantize()
        total_loss += loss
        print(f"  {name}: error={loss:.6f} (n_inputs={q.n_samples})")
        q.free()

    print(f"\nTotal quantization error: {total_loss:.6f}")
    print(f"Average per-layer error: {total_loss / len(quantizers):.6f}")

    # Save quantized model
    output_dir = args.gptq_output_dir
    if output_dir is None:
        output_dir = os.path.join(args.save_dir, args.run_id, "quantized")
    os.makedirs(output_dir, exist_ok=True)

    hc_suffix = "_full_hc_prec" if skip_hc else ""
    output_path = os.path.join(output_dir, f"quantized_w{bits}_g{group_size}{hc_suffix}.tar")
    torch.save({
        'model_state_dict': model.state_dict(),
        'gptq_config': {
            'bits': bits,
            'group_size': group_size,
            'n_calibration_batches': n_batches,
            'source_checkpoint': checkpoint_path,
        },
    }, output_path)
    print(f"\nSaved quantized model to {output_path}")


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
