# Hyperloop Transformer
Official implementation of the Hyperloop Transformer architecture as described in [Hyperloop Transformers](https://arxiv.org/abs/2604.21254).

## Prerequisites
Make sure you are logged in to wandb and huggingface from the command line:
```bash
wandb login
huggingface-cli login
```

To set up the virtual environment:
```bash
uv venv uvenvs/hyperloop-transformers --python 3.13
source uvenvs/hyperloop-transformers/bin/activate
uv pip install -r requirements.txt
```

## Pretraining
By default the `train.py` script loads the `exp_configs/default_config.yaml` config and overrides any parameters if they are also passed as command-line arguments in hydra format. The script automatically switches between FSDP and single-GPU training based on the number of detected GPUs and will automatically resume from the most recently saved checkpoint in the checkpoint save directory. The training dataset can either be streamed directly from huggingface, or if `use_locally_cached_dataset=True`, downloaded and cached locally. The rest of the arguments are documented in `exp_configs/default_config.yaml`.

NOTE: the codebase defaults to the updated "gpt2" initialization, corresponding to the results in Appendix D in the paper. To replicate the experimental setup used in the main body of the paper, use `init_scheme="pytorch"`.

### Example Runs
Train a vanilla Transformer (a vanilla Transformer is just a Looped Transformer with a single loop):
```bash
torchrun --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=1 \
    --nproc-per-node=2 \
    train.py n_layers=16 \
        dataset_name="fineweb-edu" \
        max_train_tokens=12500000000 \
        tok_path="meta-llama/Llama-2-7b-hf" \
        max_seq_len=2048 \
        global_batch_size=256 \
        micro_batch_size=32 \
        first_recur_layer=0 \
        last_recur_layer=1 \
        num_loops=1 \
        warmup_iters=1000 \
        max_grad_norm=1.0 \
        n_res_streams=null \
        mix_rs_after_loops=False \
        mix_rs_after_layers=False \
        use_locally_cached_dataset=True \
        save_every_batches=200 \
        exp_name='Vanilla Baseline' \
        run_id='vanilla_baseline_v1'
```

Train a vanilla Looped Transformer:
```bash
torchrun --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=1 \
    --nproc-per-node=2 \
    train.py n_layers=8 \
        dataset_name="fineweb-edu" \
        max_train_tokens=12500000000 \
        tok_path="meta-llama/Llama-2-7b-hf" \
        max_seq_len=2048 \
        global_batch_size=256 \
        micro_batch_size=32 \
        first_recur_layer=2 \
        last_recur_layer=5 \
        num_loops=3 \
        warmup_iters=1000 \
        max_grad_norm=1.0 \
        n_res_streams=null \
        mix_rs_after_loops=False \
        mix_rs_after_layers=False \
        use_locally_cached_dataset=True \
        save_every_batches=200 \
        exp_name='Looped Baseline' \
        run_id='looped_baseline_v1'
```

Train an mHC Transformer:
```bash
torchrun --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=1 \
    --nproc-per-node=2 \
    train.py n_layers=16 \
        dataset_name="fineweb-edu" \
        max_train_tokens=12500000000 \
        tok_path="meta-llama/Llama-2-7b-hf" \
        max_seq_len=2048 \
        global_batch_size=256 \
        micro_batch_size=16 \
        first_recur_layer=0 \
        last_recur_layer=15 \
        num_loops=1 \
        warmup_iters=1000 \
        max_grad_norm=1.0 \
        n_res_streams=4 \
        mix_rs_after_loops=False \
        mix_rs_after_layers=True \
        rs_mixing_strat="sinkhorn" \
        premap_mixing_strat="default" \
        postmap_mixing_strat="default" \
        save_every_batches=200 \
        duplicate_init_res_proj=True \
        average_fin_res_proj=True \
        use_locally_cached_dataset=True \
        use_mhc=True \
        exp_name='mHC Baseline' \
        run_id='mhc_baseline_v1'
```

Train a Hyperloop Transformer (Ours):
```bash
torchrun --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=1 \
    --nproc-per-node=2 \
    train.py n_layers=8 \
        dataset_name="fineweb-edu" \
        max_train_tokens=12500000000 \
        tok_path="meta-llama/Llama-2-7b-hf" \
        max_seq_len=2048 \
        global_batch_size=256 \
        micro_batch_size=32 \
        first_recur_layer=2 \
        last_recur_layer=5 \
        num_loops=3 \
        warmup_iters=1000 \
        max_grad_norm=1.0 \
        n_res_streams=4 \
        mix_rs_after_loops=True \
        mix_rs_after_layers=False \
        rs_mixing_strat="diagonal_gate" \
        premap_mixing_strat="default" \
        postmap_mixing_strat="default" \
        save_every_batches=200 \
        add_postmap_pos_embeds=True \
        duplicate_init_res_proj=True \
        average_fin_res_proj=True \
        use_locally_cached_dataset=True \
        exp_name='Hyperloop Transformer' \
        run_id='hyperloop_transformer_v1'
```

## Evaluation
The evaluation scripts load the model config from a saved checkpoint directory along with the most recent checkpoint from that directory. A specific checkpoint path can also be specified to load a different checkpoint if needed. Results are also written to `eval_out_file_path` if it is set.

### Compute Validation Perplexity
```bash
exp_name="hyperloop_transformer_v1"

torchrun --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=1 \
    --nproc-per-node=1 \
    comp_val_ppl.py --config "./checkpoints/${exp_name}/train_config.yaml" \
        micro_batch_size=32 \
        max_val_tokens=500000000 \
        eval_out_file_path="./val_ppl_results.txt"
```

### Compute Downstream Task Accuracies
```bash
exp_name="hyperloop_transformer_v1"

torchrun --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=1 \
    --nproc-per-node=1 \
    eval_lm_harness.py --config "./checkpoints/${exp_name}/train_config.yaml" \
        lm_eval_tasks="copa,hellaswag,lambada,openbookqa,piqa,race,sciq,winogrande,arc_easy,arc_challenge" \
        eval_out_file_path="./lm_eval_results.txt"
```

## Quantize with GPTQ
`quantize_gptq.py` computes quantized weights but converts them back to (rounded) BF16 before saving. The resulting models can be evaluated using the same scripts above.
```bash
exp_name="hyperloop_transformer_v1"

torchrun --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=1 \
    --nproc-per-node=1 \
    quantize_gptq.py --config "./checkpoints/${exp_name}/train_config.yaml" \
        micro_batch_size=32 \
        gptq_bits=4 \
        gptq_group_size=128 \
        gptq_n_batches=32 \
        gptq_skip_hc=False
```


## Citation
```bibtex
@inproceedings{zeitoun2026hyperloop,
  title     = {Hyperloop Transformers},
  author    = {Abbas Zeitoun and Lucas Torroba-Hennigen and Yoon Kim},
  booktitle = {Third Conference on Language Modeling},
  year      = {2026},
  url       = {https://openreview.net/forum?id=WiSctoWmm1}
}
```
