# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
# Modifications copyright 2026 Abbas Zeitoun.
# This file has been modified from the original litgpt source.

"""Full definition of a decoder-only transformer-based language model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""
import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from typing_extensions import Self

from litgpt_config import Config


def _normal_init_(root: nn.Module, std: float) -> None:
    """Draw all Linear/Embedding weights in the subtree from N(0, std) and zero
    the biases, leaving modules with purpose-built initializations untouched."""
    skip_modules = (
        LoRALayer,  # lora_B must stay zero so LoRA deltas start at 0
        LinearPreMapping, PreMapping, LinearPostMapping, PostMapping,
        CayleyResMapping, SinkhornResMapping, LinearResMapping,
        DiagonalGateResMapping, StreamIndepGateResMapping,
    )

    def visit(module: nn.Module) -> None:
        if isinstance(module, skip_modules):
            return
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if isinstance(module, nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)
        for child in module.children():
            visit(child)

    visit(root)


class GPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias)
        use_mlp_lora = (config.lora_rank is not None)
        use_attn_lora = (config.lora_attn_rank is not None)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config, block_idx) for block_idx in range(config.first_recur_layer)]
                                + [Block(config, block_idx, use_lora=use_mlp_lora, use_attn_lora=use_attn_lora, use_mhc=config.use_mhc) for block_idx in range(config.first_recur_layer, config.last_recur_layer + 1)]
                                + [Block(config, block_idx) for block_idx in range(config.last_recur_layer + 1, config.n_layer)]),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        if config.n_res_streams is not None:
            if config.duplicate_init_res_proj:
                self.init_res_proj = None
            elif config.init_res_proj_rank is not None:
                self.init_res_proj = LowRankLinear(config.n_embd, config.n_embd*config.n_res_streams, config.init_res_proj_rank)
            else:
                self.init_res_proj = nn.Linear(config.n_embd, config.n_embd*config.n_res_streams)

            if config.average_fin_res_proj:
                self.fin_res_proj = None
            elif config.fin_res_proj_rank is not None:
                self.fin_res_proj = LowRankLinear(config.n_embd*config.n_res_streams, config.n_embd, config.fin_res_proj_rank)
            else:
                self.fin_res_proj = nn.Linear(config.n_embd*config.n_res_streams, config.n_embd)
                
            # mHC mappings are stored within the transformer Blocks themselves
            if not config.use_mhc:
                total_looped_layers = (config.last_recur_layer - config.first_recur_layer + 1) * config.num_loops
                if config.mix_rs_every_l_layers is not None:
                    n_res_projs = total_looped_layers // config.mix_rs_every_l_layers
                else:
                    n_res_projs = 0

                # Premappings
                if config.premap_mixing_strat == "linear":
                    self.pre_proj = nn.ModuleList(LinearPreMapping(config.n_embd, config.n_res_streams) for _ in range(n_res_projs))
                else: # if config.premap_mixing_strat == "default"
                    self.pre_proj = nn.ModuleList(PreMapping(config.n_embd, config.n_res_streams) for _ in range(n_res_projs))


                # Resmappings
                if config.rs_mixing_strat == "cayley":
                    self.res_proj = nn.ModuleList(CayleyResMapping(config.n_embd, config.n_res_streams) for _ in range(n_res_projs))
                elif config.rs_mixing_strat == "sinkhorn":
                    self.res_proj = nn.ModuleList(SinkhornResMapping(config.n_embd, config.n_res_streams) for _ in range(n_res_projs))
                elif config.rs_mixing_strat == "linear":
                    self.res_proj = nn.ModuleList(LinearResMapping(config.n_embd, config.n_res_streams) for _ in range(n_res_projs))
                elif config.rs_mixing_strat == "diagonal_gate":
                    self.res_proj = nn.ModuleList(DiagonalGateResMapping(config.n_embd, config.n_res_streams) for _ in range(n_res_projs))
                elif config.rs_mixing_strat == "stream_indep_gate":
                    self.res_proj = nn.ModuleList(StreamIndepGateResMapping(config.n_embd, config.n_res_streams) for _ in range(n_res_projs))
                else: # if config.rs_mixing_strat == "identity"
                    self.res_proj = nn.ModuleList(IdentityResMapping() for _ in range(n_res_projs))


                # Postmappings
                if config.postmap_mixing_strat == "linear":
                    self.post_proj = nn.ModuleList(LinearPostMapping(config.n_embd, config.n_res_streams) for _ in range(n_res_projs))
                else: # if config.postmap_mixing_strat == "default"
                    self.post_proj = nn.ModuleList(PostMapping(config.n_embd, config.n_res_streams) for _ in range(n_res_projs))


                # Add (learned) position embeddings
                if config.add_premap_pos_embeds:
                    self.premap_pos_embeds = nn.Embedding(n_res_projs, config.n_embd)
                if config.add_postmap_pos_embeds:
                    self.postmap_pos_embeds = nn.Embedding(n_res_projs, config.n_embd)

        self.max_depth = self.config.max_depth
        self.max_seq_length = self.config.block_size
        self.mask_cache: Optional[torch.Tensor] = None

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        """
        When doing inference, the sequences used might be shorter than the model's context length.
        This allows setting a smaller number to avoid allocating unused memory
        """
        if value > self.config.block_size:
            raise ValueError(f"Cannot attend to {value}, block size is only {self.config.block_size}."
                             " This is likely because the input text exceeds the supported context length of this model.")
        self._max_seq_length = value
        if not hasattr(self, "cos"):
            # first call
            cos, sin = self.rope_cache()
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        # override
        elif value != self.cos.size(0):
            self.cos, self.sin = self.rope_cache(device=self.cos.device)
        # the mask and kv cache size will get updated on `set_kv_cache`. we cannot update it here because we don't know
        # if the kv cache is expected

    def reset_parameters(self) -> None:
        # Trigger resetting the rope-cache
        self.cos, self.sin = self.rope_cache(device=self.cos.device)

    def init_weights(self, scheme: str = "gpt2") -> None:
        if scheme == "pytorch":
            # Keep the PyTorch default (and module-specific custom) initializations
            return
        if scheme != "gpt2":
            raise ValueError(f"Unsupported init scheme: {scheme!r}")

        # GPT-2 scheme: all weights are drawn from N(0, 0.02), except the output
        # projections that write to the residual stream (attn.proj, mlp.proj),
        # whose std is damped by the total unrolled depth so the residual-stream
        # variance stays bounded at init. Looped layers write to the stream once
        # per loop, hence the unrolled (runtime) depth rather than n_layer.
        cfg = self.config
        if cfg.first_recur_layer is not None and cfg.last_recur_layer is not None and cfg.num_loops is not None:
            layers_per_loop = cfg.last_recur_layer - cfg.first_recur_layer + 1
            total_unrolled_depth = cfg.first_recur_layer + layers_per_loop * cfg.num_loops + (cfg.n_layer - 1 - cfg.last_recur_layer)
        else:
            total_unrolled_depth = cfg.n_layer
        residual_std = 0.02 / math.sqrt(2 * total_unrolled_depth)

        _normal_init_(self, std=0.02)
        for block in self.transformer.h:
            _normal_init_(block.attn.proj, std=residual_std)
            _normal_init_(block.mlp.proj, std=residual_std)

    def forward(self, idx: torch.Tensor,
                input_pos: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                num_loops: int = 1,
                first_recur_layer: int = 0,
                last_recur_layer: int = None,
                return_hidden_states: bool = False) -> torch.Tensor:

        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

        if input_pos is not None:  # use the kv cache
            cos = batched_index_select(self.cos, 0, input_pos)
            sin = batched_index_select(self.sin, 0, input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = batched_index_select(self.mask_cache, 2, input_pos)
            if mask.dim() > 4:
                # the mask cache has a batch dim of 1 in addition to the one
                # we get if input_pos has a batch dimension
                mask = mask.squeeze(1)
        else:
            cos = self.cos[:T]
            sin = self.sin[:T]

        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        if self.config.scale_embeddings:
            x = x * torch.tensor(self.config.n_embd**0.5, dtype=x.dtype)

        if mask is not None:
            mask = mask.to(torch.bool).unsqueeze(0).unsqueeze(0)

        if return_hidden_states:
            all_hidden_states = []

        for block in self.transformer.h[:first_recur_layer]:
            x = block(x, cos, sin, mask, input_pos)
            if return_hidden_states:
                all_hidden_states.append(x)

        if self.config.n_res_streams is not None:
            if self.config.duplicate_init_res_proj:
                x_res = x.unsqueeze(2).expand(-1, -1, self.config.n_res_streams, -1)
            else:
                x_res = self.init_res_proj(x) # [batch_size, seq_len, n_streams*hid_size]
                x_res = x_res.reshape(x_res.size(0), x_res.size(1), self.config.n_res_streams, -1) # [batch_size, seq_len, n_streams, hid_size]

        if self.config.looping_strategy == "cycle":
            use_mhc = self.config.use_mhc and self.config.n_res_streams is not None
            use_rs_maps = self.config.n_res_streams is not None and self.config.mix_rs_every_l_layers is not None

            if use_mhc:
                for loop_idx in range(num_loops):
                    for rec_block_idx, block in enumerate(self.transformer.h[first_recur_layer:(last_recur_layer+1)]):
                        if self.config.lora_rank is not None or self.config.lora_attn_rank is not None:
                            x_res = block(x_res, cos, sin, mask, input_pos,
                                          lora_layer_idx=loop_idx, mhc_layer_idx=loop_idx)
                        else:
                            x_res = block(x_res, cos, sin, mask, input_pos,
                                          mhc_layer_idx=loop_idx)
                        if return_hidden_states:
                            all_hidden_states.append(x_res)
            else:
                for loop_idx in range(num_loops):
                    for rec_block_idx, block in enumerate(self.transformer.h[first_recur_layer:(last_recur_layer+1)]):
                        rec_layer_idx = loop_idx*(last_recur_layer - first_recur_layer + 1) + rec_block_idx

                        if use_rs_maps and rec_layer_idx % self.config.mix_rs_every_l_layers == 0:
                            mapping_idx = rec_layer_idx // self.config.mix_rs_every_l_layers
                            x = self.pre_proj[mapping_idx](x_res)

                            if self.config.add_premap_pos_embeds:
                                x = x + self.premap_pos_embeds(torch.tensor(mapping_idx, device=x.device))

                        if self.config.lora_rank is not None or self.config.lora_attn_rank is not None:
                            x = block(x, cos, sin, mask, input_pos,
                                      lora_layer_idx=loop_idx)
                        else:
                            x = block(x, cos, sin, mask, input_pos)

                        if return_hidden_states:
                            all_hidden_states.append(x)

                        if use_rs_maps and (rec_layer_idx + 1) % self.config.mix_rs_every_l_layers == 0:
                            mapping_idx = rec_layer_idx // self.config.mix_rs_every_l_layers
                            if self.config.add_postmap_pos_embeds:
                                x = x + self.postmap_pos_embeds(torch.tensor(mapping_idx, device=x.device))

                            x_res_mapped = self.res_proj[mapping_idx](x_res)
                            x_res = x_res_mapped + self.post_proj[mapping_idx](x_res, x)

        elif self.config.looping_strategy == "sequence":
            for rec_block_idx, block in enumerate(self.transformer.h[first_recur_layer:(last_recur_layer+1)]):
                for loop_idx in range(num_loops):
                    # rec_layer_idx = loop_idx*(last_recur_layer - first_recur_layer + 1) + rec_block_idx
                    rec_layer_idx = rec_block_idx*num_loops + loop_idx
                    if self.config.lora_rank is not None or self.config.lora_attn_rank is not None:
                        x = block(x, cos, sin, mask, input_pos,
                                  lora_layer_idx=loop_idx)
                    else:
                        x = block(x, cos, sin, mask, input_pos)

                    if return_hidden_states:
                        all_hidden_states.append(x)

        if self.config.n_res_streams is not None:
            if self.config.average_fin_res_proj:
                x = x_res.mean(dim=2)
            else:
                x = self.fin_res_proj(x_res.reshape(x_res.size(0), x_res.size(1), -1))

        for block in self.transformer.h[(last_recur_layer+1):]:
            x = block(x, cos, sin, mask, input_pos)
            if return_hidden_states:
                all_hidden_states.append(x)

        x = self.transformer.ln_f(x) # (b, t, hid_size)
        x = self.lm_head(x)  # (b, t, vocab_size)
        if self.config.final_logit_softcapping is not None:
            x = torch.tanh(x / self.config.final_logit_softcapping) * self.config.final_logit_softcapping

        ret_dict = {
                "outputs": x,
                }

        if return_hidden_states:
            ret_dict["all_hidden_states"] = all_hidden_states

        return ret_dict

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def rope_cache(self, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.config.rope_adjustments is None:
            extra_config = None

        else:
            adjusted_params_required = ["factor", "low_freq_factor", "high_freq_factor", "original_max_seq_len"]
            params_present = [param in self.config.rope_adjustments for param in adjusted_params_required]
            num_params_present = sum(params_present)

            if num_params_present == 0:
                extra_config = None  # uses standard RoPE
            elif num_params_present == 4:
                # These parameters should always be used together so that we don't interfere with standard RoPE
                extra_config = {
                    "original_max_seq_len": self.config.rope_adjustments["original_max_seq_len"],
                    "factor": self.config.rope_adjustments["factor"],
                    "low_freq_factor": self.config.rope_adjustments["low_freq_factor"],
                    "high_freq_factor": self.config.rope_adjustments["high_freq_factor"],
                }
            else:
                # Some but not all parameters are specified; raise an error
                missing_params = [param for param, present in zip(adjusted_params_required, params_present) if not present]
                raise ValueError(
                    f"The following adjusted RoPE parameters are missing in rope_adjustments: {', '.join(missing_params)}. "
                    "All adjusted RoPE parameters must be specified together."
                )

        return build_rope_cache(
            seq_len=self.max_seq_length,
            n_elem=self.config.rope_n_elem,
            device=device,
            condense_ratio=self.config.rope_condense_ratio,
            base=self.config.rope_base,
            extra_config=extra_config,
        )

    def set_kv_cache(
        self,
        batch_size: int,
        max_seq_length: Optional[int] = None,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if rope_cache_length is None:
            rope_cache_length = self.cos.size(-1)

        if max_seq_length is None:
            max_seq_length = self.max_seq_length

        # initialize the kv cache for all blocks
        for block in self.transformer.h:
            block.attn.kv_cache = block.attn.build_kv_cache(
                batch_size, max_seq_length, rope_cache_length, device, dtype,
            )

        if self.mask_cache is None or self.mask_cache.size(3) != max_seq_length:
            # passing `attn_mask` to SDPA disables the flash implementation. since we only need the mask
            # for the kv-cache support (only during inference), we only create it in that situation
            self.mask_cache = build_mask_cache(max_seq_length, device)

    def clear_kv_cache(self) -> None:
        self.mask_cache = None
        for block in self.transformer.h:
            block.attn.kv_cache = None

def cayley_transform(A: torch.Tensor) -> torch.Tensor:
    orig_dtype = A.dtype
    A = A.float()
    A_skew = (A - A.transpose(-2, -1)) / 2
    n = A.shape[-1]
    eye = torch.eye(n, device=A.device, dtype=A.dtype)
    Q = torch.linalg.solve((eye + A_skew).transpose(-2, -1), (eye - A_skew).transpose(-2, -1)).transpose(-2, -1)
    return Q.to(orig_dtype)

class LowRankLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))

class LinearPreMapping(nn.Module):
    def __init__(self, hidden_dim: int, expansion_rate: int = 4, eps: float = 1e-20):
        super().__init__()
        self.expansion_rate = expansion_rate
        self.eps = eps
        self.phi = nn.Linear(expansion_rate * hidden_dim, expansion_rate, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, n, c = x.shape
        x_flat = x.reshape(batch * seq, n * c)
        rms = torch.sqrt(torch.mean(x_flat ** 2, dim=-1, keepdim=True) + self.eps)
        H_pre = self.phi(x_flat / rms).reshape(batch, seq, n)
        h_in = (H_pre.unsqueeze(-1) * x).sum(dim=2)
        return h_in

class PreMapping(nn.Module):
    def __init__(self, hidden_dim: int, expansion_rate: int = 4, alpha_init: float = 0.01, eps: float = 1e-20):
        super().__init__()
        self.expansion_rate = expansion_rate
        self.eps = eps
        self.phi = nn.Linear(expansion_rate * hidden_dim, expansion_rate, bias=False)
        self.alpha = nn.Parameter(torch.tensor([alpha_init]))
        self.bias = nn.Parameter(torch.full((expansion_rate,), 1.0 / expansion_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, n, c = x.shape
        x_flat = x.reshape(batch * seq, n * c)
        rms = torch.sqrt(torch.mean(x_flat ** 2, dim=-1, keepdim=True) + self.eps)
        # Different from original HC formulation
        # Uses all res streams to determine weighting of each stream for each token
        H_pre = torch.sigmoid(self.alpha * self.phi(x_flat / rms) + self.bias).reshape(batch, seq, n)
        h_in = (H_pre.unsqueeze(-1) * x).sum(dim=2)
        return h_in


class LinearPostMapping(nn.Module):
    def __init__(self, hidden_dim: int, expansion_rate: int = 4, eps: float = 1e-20):
        super().__init__()
        self.expansion_rate = expansion_rate
        self.eps = eps
        self.phi = nn.Linear(expansion_rate * hidden_dim, expansion_rate, bias=False)

    def forward(self, x: torch.Tensor, layer_output: torch.Tensor) -> torch.Tensor:
        batch, seq, n, c = x.shape
        x_flat = x.reshape(batch * seq, n * c)
        rms = torch.sqrt(torch.mean(x_flat ** 2, dim=-1, keepdim=True) + self.eps)
        H_post = self.phi(x_flat / rms).reshape(batch, seq, n)
        h_post = H_post.unsqueeze(-1) * layer_output.unsqueeze(2)
        return h_post

class PostMapping(nn.Module):
    def __init__(self, hidden_dim: int, expansion_rate: int = 4, alpha_init: float = 0.01, eps: float = 1e-20):
        super().__init__()
        self.expansion_rate = expansion_rate
        self.eps = eps
        self.phi = nn.Linear(expansion_rate * hidden_dim, expansion_rate, bias=False)
        self.alpha = nn.Parameter(torch.tensor([alpha_init]))
        self.bias = nn.Parameter(torch.zeros(expansion_rate))

    def forward(self, x: torch.Tensor, layer_output: torch.Tensor) -> torch.Tensor:
        batch, seq, n, c = x.shape
        x_flat = x.reshape(batch * seq, n * c)
        rms = torch.sqrt(torch.mean(x_flat ** 2, dim=-1, keepdim=True) + self.eps)
        # TODO: the 2.0 follows the mHC paper, but it's unclear why it's needed
        H_post = 2.0 * torch.sigmoid(self.alpha * self.phi(x_flat / rms) + self.bias).reshape(batch, seq, n)
        h_post = H_post.unsqueeze(-1) * layer_output.unsqueeze(2)
        return h_post

class CayleyResMapping(nn.Module):
    def __init__(self, hidden_dim: int, expansion_rate: int = 4, alpha_init: float = 0.01, eps: float = 1e-20):
        super().__init__()
        self.expansion_rate = expansion_rate
        self.eps = eps
        self.phi = nn.Linear(expansion_rate * hidden_dim, expansion_rate ** 2, bias=False)
        self.alpha = nn.Parameter(torch.tensor([alpha_init]))
        self.bias = nn.Parameter(torch.zeros(expansion_rate, expansion_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, n, c = x.shape
        x_flat = x.reshape(batch * seq, n * c)
        rms = torch.sqrt(torch.mean(x_flat ** 2, dim=-1, keepdim=True) + self.eps)
        dynamic = self.phi(x_flat / rms).reshape(batch * seq, n, n)
        H_res = cayley_transform(self.alpha * dynamic + self.bias).reshape(batch, seq, n, n)
        h_res = H_res @ x
        return h_res

class SinkhornResMapping(nn.Module):
    def __init__(self, hidden_dim: int, expansion_rate: int = 4, alpha_init: float = 0.01, num_iters: int = 20, eps: float = 1e-20):
        super().__init__()
        self.expansion_rate = expansion_rate
        self.num_iters = num_iters
        self.eps = eps
        self.phi = nn.Linear(expansion_rate * hidden_dim, expansion_rate ** 2, bias=False)
        self.alpha = nn.Parameter(torch.tensor([alpha_init]))
        self.bias = nn.Parameter(torch.zeros(expansion_rate, expansion_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, n, c = x.shape
        x_flat = x.reshape(batch * seq, n * c)
        rms = torch.sqrt(torch.mean(x_flat ** 2, dim=-1, keepdim=True) + self.eps)
        dynamic = self.phi(x_flat / rms).reshape(batch * seq, n, n)
        M = torch.exp(self.alpha * dynamic + self.bias)
        for _ in range(self.num_iters):
            M = M / (M.sum(dim=-1, keepdim=True) + self.eps)
            M = M / (M.sum(dim=-2, keepdim=True) + self.eps)
        H_res = M.reshape(batch, seq, n, n)
        h_res = H_res @ x
        return h_res

class LinearResMapping(nn.Module):
    def __init__(self, hidden_dim: int, expansion_rate: int = 4, eps: float = 1e-20):
        super().__init__()
        self.eps = eps
        self.linear = nn.Linear(expansion_rate*hidden_dim, expansion_rate*hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, n, c = x.shape
        x_flat = x.reshape(batch * seq, n * c)
        rms = torch.sqrt(torch.mean(x_flat ** 2, dim=-1, keepdim=True) + self.eps)
        x = self.linear(x_flat/rms)
        x = x.reshape(batch, seq, n, c)
        return x

class DiagonalGateResMapping(nn.Module):
    def __init__(self, hidden_dim: int, expansion_rate: int = 4, alpha_init: float = 0.01, eps: float = 1e-20):
        super().__init__()
        self.expansion_rate = expansion_rate
        self.eps = eps
        self.phi = nn.Linear(expansion_rate * hidden_dim, expansion_rate, bias=False)
        self.alpha = nn.Parameter(torch.tensor([alpha_init]))
        self.bias = nn.Parameter(torch.ones(expansion_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, n, c = x.shape
        x_flat = x.reshape(batch * seq, n * c)
        rms = torch.sqrt(torch.mean(x_flat ** 2, dim=-1, keepdim=True) + self.eps)
        gate = torch.sigmoid(self.alpha * self.phi(x_flat / rms) + self.bias).reshape(batch, seq, n, 1)
        return gate * x

class StreamIndepGateResMapping(nn.Module):
    def __init__(self, hidden_dim: int, expansion_rate: int = 4, alpha_init: float = 0.01, eps: float = 1e-20):
        super().__init__()
        self.eps = eps
        self.phi = nn.Linear(hidden_dim, 1, bias=False)
        self.alpha = nn.Parameter(torch.tensor([alpha_init]))
        self.bias = nn.Parameter(torch.ones(expansion_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, n, c = x.shape
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        gate = torch.sigmoid(self.alpha * self.phi(x / rms).squeeze(-1) + self.bias).unsqueeze(-1)
        return gate * x

class IdentityResMapping(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

class Block(nn.Module):
    def __init__(self, config: Config,
                 block_idx: int,
                 use_lora: bool = False,
                 use_attn_lora: bool = False,
                 use_mhc: bool = False) -> None:
        super().__init__()
        if not config.parallel_residual and config.shared_attention_norm:
            raise NotImplementedError(
                "No checkpoint amongst the ones we support uses this configuration"
                " (non-parallel residual and shared attention norm)."
            )

        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        if use_attn_lora:
            self.attn = LoRACausalSelfAttention(config, block_idx)
        else:
            self.attn = EfficientCausalSelfAttention(config, block_idx)
        self.post_attention_norm = (
            config.norm_class(config.n_embd, eps=config.norm_eps) if config.post_attention_norm else nn.Identity()
        )
        self.norm_2 = None if config.shared_attention_norm else config.norm_class(config.n_embd, eps=config.norm_eps)

        if use_lora:
            self.mlp = LoRALLaMAMLP(config)
        else:
            # Why do we need this level of indirection?
            self.mlp = config.mlp_class(config)
        self.post_mlp_norm = (
            config.norm_class(config.n_embd, eps=config.norm_eps) if config.post_mlp_norm else nn.Identity()
        )

        self.config = config

        if use_mhc and config.n_res_streams is not None:
            n = config.n_res_streams
            num_loops = config.num_loops

            def _make_pre():
                if config.premap_mixing_strat == "linear":
                    return LinearPreMapping(config.n_embd, n)
                else:  # "default"
                    return PreMapping(config.n_embd, n)

            def _make_res():
                if config.rs_mixing_strat == "diagonal_gate":
                    return DiagonalGateResMapping(config.n_embd, n)
                elif config.rs_mixing_strat == "stream_indep_gate":
                    return StreamIndepGateResMapping(config.n_embd, n)
                elif config.rs_mixing_strat == "cayley":
                    return CayleyResMapping(config.n_embd, n)
                elif config.rs_mixing_strat == "sinkhorn":
                    return SinkhornResMapping(config.n_embd, n)
                elif config.rs_mixing_strat == "linear":
                    return LinearResMapping(config.n_embd, n)
                else:  # "identity"
                    return IdentityResMapping()

            def _make_post():
                if config.postmap_mixing_strat == "linear":
                    return LinearPostMapping(config.n_embd, n)
                else:  # "default"
                    return PostMapping(config.n_embd, n)

            self.mhc_pre_attn = nn.ModuleList(_make_pre() for _ in range(num_loops))
            self.mhc_res_attn = nn.ModuleList(_make_res() for _ in range(num_loops))
            self.mhc_post_attn = nn.ModuleList(_make_post() for _ in range(num_loops))
            self.mhc_pre_mlp = nn.ModuleList(_make_pre() for _ in range(num_loops))
            self.mhc_res_mlp = nn.ModuleList(_make_res() for _ in range(num_loops))
            self.mhc_post_mlp = nn.ModuleList(_make_post() for _ in range(num_loops))

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        lora_layer_idx: Optional[int] = None,
        mhc_layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Non-parallel residual       Parallel residual
           ┌─ x                     ┌─ x ──────────────────┐             Note: if `shared_attention_norm` is True,
           │  ↓                     │  ↓                   ↓                   the output from `norm_1` is reused
           │  norm_1                │  norm_1  ───────►    norm_2
           │  ↓                     │  ↓                   ↓
           │  attn                  │  attn                MLP
           │  ↓                     │  ↓                   ↓
           |  post_attn_norm        |  post_attn_norm      post_mlp_norm
           |  ↓                     |  ↓                   ↓
        ┌─ └► +                     └► + ◄─────────────────┘
        |     ↓
        │     norm_2
        │     ↓
        │     MLP
        │     ↓
        |     post_mlp_norm
        |     ↓
        └───► +
        """

        if mhc_layer_idx is not None:
            # MHC mode: x is x_res of shape (B, T, n_res_streams, C)
            x_res = x

            # --- Attention sub-block ---
            x = self.mhc_pre_attn[mhc_layer_idx](x_res)
            x_normed = self.norm_1(x)
            if self.config.lora_attn_rank is not None and lora_layer_idx is not None:
                x = self.attn(x_normed, cos, sin, mask, input_pos, lora_layer_idx=lora_layer_idx)
            else:
                x = self.attn(x_normed, cos, sin, mask, input_pos)
            x = self.post_attention_norm(x)
            x_res_mapped = self.mhc_res_attn[mhc_layer_idx](x_res)
            x_res = x_res_mapped + self.mhc_post_attn[mhc_layer_idx](x_res, x)

            # --- MLP sub-block ---
            x = self.mhc_pre_mlp[mhc_layer_idx](x_res)
            if self.config.lora_rank is not None and lora_layer_idx is not None:
                x = self.post_mlp_norm(self.mlp(self.norm_2(x), lora_layer_idx))
            else:
                x = self.post_mlp_norm(self.mlp(self.norm_2(x)))
            x_res_mapped = self.mhc_res_mlp[mhc_layer_idx](x_res)
            x_res = x_res_mapped + self.mhc_post_mlp[mhc_layer_idx](x_res, x)
            return x_res

        x_normed = self.norm_1(x)
        if self.config.lora_attn_rank is not None and lora_layer_idx is not None:
            attention_output = self.attn(x_normed, cos, sin, mask, input_pos, lora_layer_idx=lora_layer_idx)
        else:
            attention_output = self.attn(x_normed, cos, sin, mask, input_pos)

        attention_output = self.post_attention_norm(attention_output)

        if self.config.parallel_residual:
            x_normed = x_normed if self.config.shared_attention_norm else self.norm_2(x)
            if self.config.lora_rank is not None and lora_layer_idx is not None:
                x = self.mlp(x_normed, lora_layer_idx) + attention_output + x
            else:
                x = self.mlp(x_normed) + attention_output + x
        else:
            x = attention_output + x
            if self.config.lora_rank is not None and lora_layer_idx is not None:
                x = self.post_mlp_norm(self.mlp(self.norm_2(x), lora_layer_idx)) + x
            else:
                x = self.post_mlp_norm(self.mlp(self.norm_2(x))) + x

        return x


class EfficientCausalSelfAttention(nn.Module):
    def __init__(self, config: Config, block_idx: int) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)
        # output projection
        # if `head_size` is explicitly specified in the config, `n_emd` might not be equal to `head_size * n_head`
        self.proj = nn.Linear(config.head_size * config.n_head, config.n_embd, bias=config.bias)
        # disabled by default
        self.kv_cache: Optional[KVCache] = None
        self.apply_sliding_window_attention = (
            config.sliding_window_size is not None and
            block_idx % config.sliding_window_layer_placing == 0
        )

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x)

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size)
        qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)

        # maybe repeat k and v if for the non multi-head attention cases
        # training: flash attention requires it
        # inference: multi-query would require a full kv cache so avoid it to limit its memory usage
        if self.config.n_query_groups != self.config.n_head and (input_pos is None or self.config.n_query_groups != 1):
            k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
            v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)

        q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
        k = k.reshape(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
        v = v.reshape(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)

        # Only the needed slice of cos and sin is passed to forward()
        # Other vectors that need to be rotated have already been rotated
        # in the KV cache
        q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
        k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
        k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)

        if input_pos is not None:
            if not isinstance(self.kv_cache, KVCache):
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            k, v = self.kv_cache(input_pos, k, v)

        if self.apply_sliding_window_attention:
            """
                  Global Window              Sliding window             Sliding window
                  attention mask      +            bias          =      attention mask
            ┌────────────────────────┐  ┌───────────────────────┐  ┌─────────────────────────┐
            │ True False False False │  │ True  True  True True │  │ True  False False False │
            │ True True  False False │  │ True  True  True True │  │ True  True  False False │
            │ True True  True  False │  │ False True  True True │  │ False True  True  False │
            │ True True  True  True  │  │ False False True True │  │ False False True  True  │
            └────────────────────────┘  └───────────────────────┘  └─────────────────────────┘
            """
            if mask is None:
                mask = torch.ones(T, T, dtype=q.dtype, device=q.device).triu(diagonal=1)
                mask.masked_fill_(mask.bool(), float("-inf"))
            sliding_window_bias = torch.ones_like(mask).tril(diagonal=-self.config.sliding_window_size)
            sliding_window_bias.masked_fill_(sliding_window_bias.bool(), float("-inf"))
            mask += sliding_window_bias

        y = self.scaled_dot_product_attention(q, k, v, mask)

        y = y.reshape(B, T, self.config.head_size * self.config.n_head)  # re-assemble all head outputs side by side

        # output projection
        return self.proj(y)

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.config.attention_scores_scalar or self.config.head_size)

        # with softcapping we cannot use SDPA
        if self.config.attention_logit_softcapping is not None:
            scale = 1.0 / math.sqrt(self.config.attention_scores_scalar or self.config.head_size)
            scores = q @ k.mT * scale
            scores = (
                torch.tanh(scores / self.config.attention_logit_softcapping) * self.config.attention_logit_softcapping
            )
            if mask is None:
                mask = torch.ones(q.size(2), q.size(2), dtype=q.dtype, device=q.device).triu(diagonal=1)
                mask.masked_fill_(mask.bool(), torch.finfo(q.dtype).min)
            scores = scores + mask
            scores = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float).to(dtype=q.dtype)
            y = scores @ v
        else:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
            )
        return y.transpose(1, 2)

    def build_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "KVCache":
        heads = 1 if self.config.n_query_groups == 1 else self.config.n_head
        v_shape = (batch_size, heads, max_seq_length, self.config.head_size)
        if rope_cache_length is None:
            if self.config.rotary_percentage != 1.0:
                raise TypeError("Please pass the `rope_cache_length=gpt.cos.size(-1)` value")
            k_shape = v_shape
        else:
            k_shape = (
                batch_size,
                heads,
                max_seq_length,
                rope_cache_length + self.config.head_size - self.config.rope_n_elem,
            )
        return KVCache(k_shape, v_shape, device=device, dtype=dtype)

class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer that can be composed within other LoRA layers.
    
    LoRA decomposes weight updates into low-rank matrices:
    W' = W + BA, where B is (out_features × rank) and A is (rank × in_features)
    """
    def __init__(self, in_features, out_features, rank=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.scaling = alpha / rank

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize B to zero so initially LoRA has no effect
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.lora_B(self.dropout(self.lora_A(x))) * self.scaling

class LoRACausalSelfAttention(nn.Module):
    def __init__(self, config: Config, block_idx: int) -> None:
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.attn_q = nn.Linear(config.n_embd, config.n_head*config.head_size, bias=config.bias)
        self.attn_k = nn.Linear(config.n_embd, config.n_query_groups*config.head_size, bias=config.bias)
        self.attn_v = nn.Linear(config.n_embd, config.n_query_groups*config.head_size, bias=config.bias)
        # output projection
        # if `head_size` is explicitly specified in the config, `n_emd` might not be equal to `head_size * n_head`
        self.proj = nn.Linear(config.head_size * config.n_head, config.n_embd, bias=config.bias)
        # disabled by default
        self.kv_cache: Optional[KVCache] = None
        self.apply_sliding_window_attention = (
            config.sliding_window_size is not None and
            block_idx % config.sliding_window_layer_placing == 0
        )

        # Add LoRA projections for every possible loop
        attn_alpha = config.lora_attn_alpha if config.lora_attn_alpha is not None else config.lora_attn_rank
        self.lora_q = nn.ModuleList(LoRALayer(in_features=config.n_embd,
                                              out_features=config.n_head*config.head_size,
                                              rank=config.lora_attn_rank,
                                              alpha=attn_alpha) for _ in range(config.num_loops))

        self.lora_k = nn.ModuleList(LoRALayer(in_features=config.n_embd,
                                              out_features=config.n_query_groups*config.head_size,
                                              rank=config.lora_attn_rank,
                                              alpha=attn_alpha) for _ in range(config.num_loops))

        self.lora_v = nn.ModuleList(LoRALayer(in_features=config.n_embd,
                                              out_features=config.n_query_groups*config.head_size,
                                              rank=config.lora_attn_rank,
                                              alpha=attn_alpha) for _ in range(config.num_loops))

        self.lora_proj = nn.ModuleList(LoRALayer(in_features=config.n_head*config.head_size,
                                                 out_features=config.n_embd,
                                                 rank=config.lora_attn_rank,
                                                 alpha=attn_alpha) for _ in range(config.num_loops))
        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        lora_layer_idx: int = 0) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        q = self.attn_q(x) + self.lora_q[lora_layer_idx](x) # B, T, n_head*head_size
        k = self.attn_k(x) + self.lora_k[lora_layer_idx](x) # B, T, n_query_groups*head_size
        v = self.attn_v(x) + self.lora_v[lora_layer_idx](x) # B, T, n_query_groups*head_size

        q = q.view(B, T, self.config.n_head, self.config.head_size)  # (B, T, nh_q, hs)
        k = k.view(B, T, self.config.n_query_groups, self.config.head_size)  # (B, T, nh_k, hs)
        v = v.view(B, T, self.config.n_query_groups, self.config.head_size)  # (B, T, nh_v, hs)

        q = q.permute(0, 2, 1, 3)  # (B, nh_q, T, hs)
        k = k.permute(0, 2, 1, 3)  # (B, nh_k, T, hs)
        v = v.permute(0, 2, 1, 3)  # (B, nh_v, T, hs)

        # Expand K/V for GQA: repeat each group's head to match the number of query heads
        if self.config.n_query_groups != self.config.n_head and (input_pos is None or self.config.n_query_groups != 1):
            q_per_kv = self.config.n_head // self.config.n_query_groups
            k = k.repeat_interleave(q_per_kv, dim=1)
            v = v.repeat_interleave(q_per_kv, dim=1)

        # Only the needed slice of cos and sin is passed to forward()
        # Other vectors that need to be rotated have already been rotated
        # in the KV cache
        q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
        k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
        k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)

        if input_pos is not None:
            if not isinstance(self.kv_cache, KVCache):
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            k, v = self.kv_cache(input_pos, k, v)

        if self.apply_sliding_window_attention:
            """
                  Global Window              Sliding window             Sliding window
                  attention mask      +            bias          =      attention mask
            ┌────────────────────────┐  ┌───────────────────────┐  ┌─────────────────────────┐
            │ True False False False │  │ True  True  True True │  │ True  False False False │
            │ True True  False False │  │ True  True  True True │  │ True  True  False False │
            │ True True  True  False │  │ False True  True True │  │ False True  True  False │
            │ True True  True  True  │  │ False False True True │  │ False False True  True  │
            └────────────────────────┘  └───────────────────────┘  └─────────────────────────┘
            """
            if mask is None:
                mask = torch.ones(T, T, dtype=q.dtype, device=q.device).triu(diagonal=1)
                mask.masked_fill_(mask.bool(), float("-inf"))
            sliding_window_bias = torch.ones_like(mask).tril(diagonal=-self.config.sliding_window_size)
            sliding_window_bias.masked_fill_(sliding_window_bias.bool(), float("-inf"))
            mask += sliding_window_bias

        y = self.scaled_dot_product_attention(q, k, v, mask)

        y = y.reshape(B, T, self.config.head_size * self.config.n_head)  # re-assemble all head outputs side by side

        # output projection
        return self.proj(y) + self.lora_proj[lora_layer_idx](y)

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.config.attention_scores_scalar or self.config.head_size)

        # with softcapping we cannot use SDPA
        if self.config.attention_logit_softcapping is not None:
            scale = 1.0 / math.sqrt(self.config.attention_scores_scalar or self.config.head_size)
            scores = q @ k.mT * scale
            scores = (
                torch.tanh(scores / self.config.attention_logit_softcapping) * self.config.attention_logit_softcapping
            )
            if mask is None:
                mask = torch.ones(q.size(2), q.size(2), dtype=q.dtype, device=q.device).triu(diagonal=1)
                mask.masked_fill_(mask.bool(), torch.finfo(q.dtype).min)
            scores = scores + mask
            scores = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float).to(dtype=q.dtype)
            y = scores @ v
        else:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
            )
        return y.transpose(1, 2)

    def build_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "KVCache":
        heads = 1 if self.config.n_query_groups == 1 else self.config.n_head
        v_shape = (batch_size, heads, max_seq_length, self.config.head_size)
        if rope_cache_length is None:
            if self.config.rotary_percentage != 1.0:
                raise TypeError("Please pass the `rope_cache_length=gpt.cos.size(-1)` value")
            k_shape = v_shape
        else:
            k_shape = (
                batch_size,
                heads,
                max_seq_length,
                rope_cache_length + self.config.head_size - self.config.rope_n_elem,
            )
        return KVCache(k_shape, v_shape, device=device, dtype=dtype)


class GptNeoxMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.gelu(x, approximate=self.config.gelu_approximate)
        return self.proj(x)


class LoRALLaMAMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.fc_2 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

        # Add LoRA layers
        mlp_alpha = config.lora_alpha if config.lora_alpha is not None else config.lora_rank
        self.lora_fc1 = nn.ModuleList(LoRALayer(in_features=config.n_embd,
                                                 out_features=config.intermediate_size,
                                                 rank=config.lora_rank,
                                                 alpha=mlp_alpha) for _ in range(config.num_loops))

        self.lora_fc2 = nn.ModuleList(LoRALayer(in_features=config.n_embd,
                                                 out_features=config.intermediate_size,
                                                 rank=config.lora_rank,
                                                 alpha=mlp_alpha) for _ in range(config.num_loops))

        self.lora_proj = nn.ModuleList(LoRALayer(in_features=config.intermediate_size,
                                                 out_features=config.n_embd,
                                                 rank=config.lora_rank,
                                                 alpha=mlp_alpha) for _ in range(config.num_loops))

        self.config = config

    def forward(self,
                x: torch.Tensor,
                lora_layer_idx: int = 0) -> torch.Tensor:
        x_fc_1 = self.fc_1(x) + self.lora_fc1[lora_layer_idx](x)
        x_fc_2 = self.fc_2(x) + self.lora_fc2[lora_layer_idx](x)
        x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        return self.proj(x) + self.lora_proj[lora_layer_idx](x)

class LLaMAMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.fc_2 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        return self.proj(x)


class GemmaMLP(LLaMAMLP):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.gelu(x_fc_1, approximate=self.config.gelu_approximate) * x_fc_2
        return self.proj(x)


class LLaMAMoE(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.gate = nn.Linear(config.n_embd, config.n_expert, bias=False)
        self.experts = nn.ModuleList(LLaMAMLP(config) for _ in range(config.n_expert))

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Derived from: https://github.com/mistralai/mistral-src/blob/b46d6/moe_one_file_ref.py#L203-L219
        See also figure 1 in https://arxiv.org/abs/2211.15841
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        x = x.view(-1, C)  # (B*T, C)
        router = self.gate(x)  # (B*T, n_expert)
        probs, indices = torch.topk(router, self.config.n_expert_per_token)  # (B*T, n_expert_per_token)
        probs = probs.softmax(dim=1, dtype=torch.float).to(dtype=x.dtype)
        masks = indices.unsqueeze(-1) == torch.arange(self.config.n_expert, device=x.device)
        masks = masks.permute(2, 0, 1)  # (n_expert, B*T, n_expert_per_token)
        y = torch.zeros_like(x)  # (B*T, C)
        for mask, expert in zip(masks, self.experts):
            token_idx, expert_idx = torch.where(mask)
            y[token_idx] += probs[token_idx, expert_idx, None] * expert(x[token_idx])
        return y.view(B, T, C)


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    device: Optional[torch.device] = None,
    base: int = 10000,
    condense_ratio: int = 1,
    extra_config: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Enhanced Transformer with Rotary Position Embedding.

    Args:
        seq_len (int): Sequence length.
        n_elem (int): Number of elements (head dimension).
        device (torch.device, optional): Device for tensor allocations.
        base (int, optional): Base for computing inverse frequencies.
        condense_ratio (int, optional): Ratio to condense the position indices.
        extra_config (dict, optional): Configuration parameters for frequency adjustments (used by Llama 3.1 and 3.2)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Cosine and sine caches for RoPE.
    """

    # Compute the inverse frequencies theta
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))

    if extra_config is not None:
        orig_context_len = extra_config["original_max_seq_len"]
        factor = extra_config["factor"]
        low_freq_factor = extra_config["low_freq_factor"]
        high_freq_factor = extra_config["high_freq_factor"]

        wavelen = 2 * torch.pi / theta
        ratio = orig_context_len / wavelen
        smooth_factor = (ratio - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smooth_factor = torch.clamp(smooth_factor, min=0.0, max=1.0)

        # Compute adjusted_theta without masked indexing
        adjusted_theta = (1 - smooth_factor) * (theta / factor) + smooth_factor * theta
        theta = adjusted_theta

    # Create position indices `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)

    return torch.cos(idx_theta), torch.sin(idx_theta)


def batched_index_select(t, dim, idx):
    """index_select for batched index and unbatched t"""
    if idx.dim() == 1:
        return torch.index_select(t, dim, idx)

    *batch_shape, idx_size = idx.shape
    res = torch.index_select(t, dim, idx.reshape(-1))  # flat index
    # split out single batch idx
    res = res.view(*t.shape[:dim], -1, idx_size, *t.shape[dim + 1 :])
    # move batch dim to front, this is np.rollaxis(res, dim, 0) for tensors
    dims = [dim] + list(range(res.dim()))
    del dims[dim + 1]
    res = res.permute(dims)
    # unflatten batch dims
    res = res.view(*batch_shape, *res.shape[1:])
    return res


def batched_index_copy_(t, dim, idx, val):
    """Index copy for batched t, idx, val"""

    if t.device.type == "mps":
        # Normalize negative dimensions
        if dim < 0:
            dim = t.dim() + dim
        if idx.dim() == 1:
            idx_shape = [1] * val.dim()
            idx_shape[dim] = -1
            idx_expanded = idx.view(*idx_shape)
            idx_expanded = idx_expanded.expand_as(val)
            t.scatter_(dim, idx_expanded, val)
            return t

        elif idx.dim() == 2:
            assert dim != 0, "Cannot index the batch dimension"
            batch_size = idx.size(0)
            idx_size = idx.size(1)
            assert batch_size == t.size(0) == val.size(0)

            idx_shape = [batch_size] + [1] * (val.dim() - 1)
            idx_shape[dim] = idx_size
            idx_expanded = idx.view(*idx_shape)
            idx_expanded = idx_expanded.expand_as(val)

            t.scatter_(dim, idx_expanded, val)
            return t
        else:
            raise NotImplementedError(f"idx.dim() == {idx.dim()} not supported")

    else:
        if idx.dim() == 1:
            return t.index_copy_(dim, idx, val)

        assert idx.dim() == 2, f"multiple batch dims not yet {idx.shape=}"
        assert dim != 0, f"cannot index batch dim {dim=}"
        batch_size, idx_size = idx.shape
        assert batch_size == t.size(0)
        assert batch_size == val.size(0)

        # if we can view the batch and indexed dimensions together, we could
        # do index trickery. This is, sadly, not the case for kvcache so we
        # fall back to for loop
        for i in range(batch_size):
            unbatched_dim = dim if dim < 0 else dim - 1
            t[i].index_copy_(unbatched_dim, idx[i], val[i])
        return t


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    # xclone = x.clone()
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    if cos.dim() > 1:
        # batch dimensions must align
        # sin/cos are (B, T, hs) so we unsqueeze -3 for nh
        # we count from back because all of apply_rope does
        cos = cos.unsqueeze(-3)
        sin = sin.unsqueeze(-3)

    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)


class KVCache(nn.Module):
    def __init__(
        self,
        k_shape: Tuple[int, int, int, int],
        v_shape: Tuple[int, int, int, int],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.register_buffer("k", torch.zeros(k_shape, device=device, dtype=dtype), persistent=False)
        self.register_buffer("v", torch.zeros(v_shape, device=device, dtype=dtype), persistent=False)

    def forward(self, input_pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # move the buffer to the activation dtype for when AMP is used
        self.k = self.k.to(k.dtype)
        self.v = self.v.to(v.dtype)
        # update the cache
        n = k.size(0)
        k = batched_index_copy_(self.k[:n, ...], -2, input_pos, k)
        v = batched_index_copy_(self.v[:n, ...], -2, input_pos, v)
        return k, v

    def reset_parameters(self) -> None:
        torch.nn.init.zeros_(self.k)
        torch.nn.init.zeros_(self.v)


def build_mask_cache(max_seq_length: int, device: Optional[torch.device] = None) -> torch.Tensor:
    ones = torch.ones((max_seq_length, max_seq_length), device=device, dtype=torch.bool)
    return torch.tril(ones).unsqueeze(0).unsqueeze(0)


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-6, add_unit_offset: bool = False) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim
        self.add_unit_offset = add_unit_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        # NOTE: the original RMSNorm paper implementation is not equivalent
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        weight = (1 + self.weight) if self.add_unit_offset else self.weight
        return (x_normed * weight.float()).to(dtype=dtype)

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)
