"""Evaluate a recurrent transformer checkpoint on lm-evaluation-harness benchmarks.

Usage:
  Single-GPU:
    python eval_lm_harness.py --config path/to/config.yaml save_dir=path run_id=id lm_eval_tasks=hellaswag lm_eval_limit=10

  Multi-GPU (FSDP2):
    torchrun --nproc_per_node=2 eval_lm_harness.py --config path/to/config.yaml save_dir=path run_id=id lm_eval_tasks=hellaswag
"""

import sys
import os
import glob
import json

import torch
import torch.nn.functional as F

from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict,
    StateDictOptions,
)

from omegaconf import open_dict
from transformers import AutoTokenizer

import lm_eval
from lm_eval.api.model import LM

from train import forward_pass

from litgpt_model import GPT
from litgpt_config import Config


class HarnessedLM(LM):
    def __init__(self, args, device, rank, world_size, batch_size=1):
        super().__init__()
        self._rank = rank
        self._world_size = world_size
        self._device = device
        self._batch_size = batch_size

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.tok_path)
        with open_dict(args):
            args.vocab_size = len(self.tokenizer)

        # Resolve mix_rs_every_l_layers from legacy flags
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

            assert args.mix_rs_every_l_layers is not None, \
                "n_res_streams requires one of mix_rs_after_loops, mix_rs_after_layers, or mix_rs_every_l_layers to be set!"
            assert total_looped_layers % args.mix_rs_every_l_layers == 0, \
                f"Total looped layers ({total_looped_layers}) must be divisible by mix_rs_every_l_layers ({args.mix_rs_every_l_layers})!"

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

        self.args = args
        self.max_seq_len = args.max_seq_len

        model = GPT(cfg).to(device, dtype=torch.bfloat16)

        # FSDP2 wrapping
        use_fsdp = world_size > 1
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
            self.checkpoint_path = args.eval_checkpoint_path
            assert os.path.exists(self.checkpoint_path), f"Checkpoint does not exist: {self.checkpoint_path}"
        else:
            checkpoint_dir = os.path.join(args.save_dir, args.run_id)
            assert os.path.exists(checkpoint_dir), f"Run directory does not exist: {checkpoint_dir}"

            checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "*.tar")))
            assert len(checkpoints) > 0, f"No checkpoints found in {checkpoint_dir}"

            self.checkpoint_path = checkpoints[-1]
        if rank == 0:
            print(f"Evaluating checkpoint: {self.checkpoint_path}")
        latest_checkpoint = torch.load(self.checkpoint_path, map_location=device)

        if use_fsdp:
            set_model_state_dict(model, latest_checkpoint['model_state_dict'],
                                 options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True))
        else:
            model_state = {k.replace("_orig_mod.", ""): v for k, v in latest_checkpoint['model_state_dict'].items()}
            model.load_state_dict(model_state)

        # torch.compile after checkpoint load
        model = torch.compile(model)

        model.eval()
        self.model = model

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def device(self):
        return self._device

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self.max_seq_len

    def tok_encode(self, string):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_logits(self, token_ids):
        """Run the model on a single sequence and return logits (1, seq_len, vocab_size)."""
        tok_seqs = torch.tensor([token_ids], dtype=torch.long, device=self._device)
        with torch.no_grad():
            ret_dict = forward_pass(self.model, tok_seqs, self.args)
        return ret_dict["outputs"]

    def loglikelihood(self, requests):
        results = []
        for request in requests:
            context, continuation = request.args

            ctx_tokens = self.tok_encode(context)
            all_tokens = self.tok_encode(context + continuation)

            # Truncate from the left to max_seq_len
            if len(all_tokens) > self.max_seq_len:
                # Ensure at least 1 context token remains
                excess = len(all_tokens) - self.max_seq_len
                all_tokens = all_tokens[excess:]
                ctx_len = max(len(ctx_tokens) - excess, 1)
            else:
                ctx_len = len(ctx_tokens)

            cont_len = len(all_tokens) - ctx_len

            logits = self._model_logits(all_tokens)  # (1, seq_len, vocab)
            logits = logits[0].float()  # (seq_len, vocab), cast for numerical stability

            # Log-softmax over vocabulary
            log_probs = F.log_softmax(logits, dim=-1)

            # Extract log-probs for the continuation tokens
            # Token at position i predicts token at position i+1
            cont_start = ctx_len - 1  # logits at this position predict first cont token
            cont_log_probs = []
            is_greedy = True
            for i in range(cont_len):
                pos = cont_start + i
                target_id = all_tokens[ctx_len + i]
                lp = log_probs[pos, target_id].item()
                cont_log_probs.append(lp)
                if logits[pos].argmax().item() != target_id:
                    is_greedy = False

            total_lp = sum(cont_log_probs)
            results.append((total_lp, is_greedy))

        return results

    def loglikelihood_rolling(self, requests):
        results = []
        for request in requests:
            (string,) = request.args
            tokens = self.tok_encode(string)

            # Chunk into max_seq_len windows with stride
            total_lp = 0.0
            offset = 0
            while offset < len(tokens):
                chunk = tokens[offset:offset + self.max_seq_len]
                logits = self._model_logits(chunk)  # (1, chunk_len, vocab)
                logits = logits[0].float()  # cast for numerical stability
                log_probs = F.log_softmax(logits, dim=-1)

                # Score each token given its prefix (skip first token — no context)
                start = 0 if offset > 0 else 1
                for i in range(start, len(chunk)):
                    lp = log_probs[i - 1, chunk[i]].item()
                    total_lp += lp

                offset += self.max_seq_len

            results.append(total_lp)

        return results

    def generate_until(self, requests):
        raise NotImplementedError("generate_until is not yet implemented")

    # -- Distributed methods for lm-eval request sharding --

    def all_gather(self, tensor):
        if self._world_size == 1:
            return tensor
        gathered = [torch.zeros_like(tensor) for _ in range(self._world_size)]
        torch.distributed.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=0)

    def gather_object(self, obj, dst=0):
        if self._world_size == 1:
            return [obj]
        output = [None] * self._world_size if self._rank == dst else None
        torch.distributed.gather_object(obj, output, dst=dst)
        return output

    def barrier(self):
        if self._world_size > 1:
            torch.distributed.barrier()


def main():
    from comp_val_ppl import load_config_with_overrides

    args = load_config_with_overrides(sys.argv[1:])

    # Distributed setup
    torch.multiprocessing.set_start_method('spawn')
    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    torch.cuda.set_device(device)

    # Parse eval-specific args
    tasks_str = args.get("lm_eval_tasks", None)
    assert tasks_str is not None, "lm_eval_tasks must be set (comma-separated task names)"
    tasks = [t.strip() for t in tasks_str.split(",")]

    num_fewshot = args.get("lm_eval_num_fewshot", 0)
    limit = args.get("lm_eval_limit", None)
    batch_size = args.get("lm_eval_batch_size", 1)

    if rank == 0:
        print(f"Tasks: {tasks}")
        print(f"Num fewshot: {num_fewshot}")
        print(f"Limit: {limit}")
        print(f"Batch size: {batch_size}")

    # Build model wrapper
    lm = HarnessedLM(args, device, rank, world_size, batch_size=batch_size)

    # Run evaluation
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,
    )

    if rank == 0:
        # Print results table
        print("\n" + "=" * 60)
        print("lm-eval-harness Results")
        print("=" * 60)
        for task_name, task_results in results["results"].items():
            print(f"\n{task_name}:")
            for metric, value in task_results.items():
                if metric != "alias":
                    print(f"  {metric}: {value}")

        # Write results to file if eval_out_file_path is set
        if args.eval_out_file_path is not None:
            from filelock import FileLock
            row = {"checkpoint": lm.checkpoint_path, "run_id": args.run_id}
            for task_name, task_results in results["results"].items():
                for metric, value in task_results.items():
                    if metric != "alias":
                        row[f"{task_name}/{metric}"] = value
            lock = FileLock(args.eval_out_file_path + ".lock")
            with lock:
                with open(args.eval_out_file_path, "a") as f:
                    f.write(json.dumps(row) + "\n")

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
