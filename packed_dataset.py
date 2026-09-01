import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node


DATASET_CONFIGS = {
    "slimpajama": {
        "path": "cerebras/SlimPajama-627B",
        "text_field": "text",
        "train_split": "train",
        "val_split": "validation",
    },
    "fineweb-edu": {
        "path": "HuggingFaceFW/fineweb-edu",
        "name": "sample-350BT",     # pre-shuffled subset
        "text_field": "text",
        "train_split": "train",
        "val_split": None,          # no dedicated val split
        "val_docs": 600_000,        # ~500M tokens carved from train
    },
}


class PackedTokenDataset(IterableDataset):
    """Streaming dataset that tokenizes and packs text into fixed-length chunks."""

    def __init__(self, dataset_name, tokenizer, max_seq_len, split="train",
                 seed=42, rank=0, world_size=1, use_locally_cached_dataset=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split = split
        self.seed = seed
        self.rank = rank
        self.world_size = world_size

        ds_cfg = DATASET_CONFIGS[dataset_name]
        self.hf_path = ds_cfg["path"]
        self.text_field = ds_cfg["text_field"]
        if split == "train":
            self.ds_split = ds_cfg["train_split"]
        else:
            self.ds_split = ds_cfg.get("val_split") or ds_cfg["train_split"]

        self.val_docs = ds_cfg.get("val_docs")
        self.hf_name = ds_cfg.get("name")

        # Download raw parquet files once; subsequent loads use the HF hub cache
        if use_locally_cached_dataset:
            from huggingface_hub import snapshot_download
            print(f"[INFO] Downloading dataset {self.hf_path} to local cache. This may take a while on first run...")
            self._local_dataset_path = snapshot_download(
                repo_id=self.hf_path,
                repo_type="dataset",
                cache_dir="./hf_streaming_cache/",
                max_workers=32,
            )
            print(f"[INFO] Dataset cached at {self._local_dataset_path}")
        else:
            self._local_dataset_path = None

        # State tracking for checkpoint-based resumption
        self._ds = self._create_hf_dataset()
        self._token_buffer = []
        self._state_restored = None     # True/False after load_checkpoint_state, None if not called

    def get_checkpoint_state(self):
        """Return current iterator state for checkpointing."""
        return {
            "hf_dataset_state": self._ds.state_dict(),
            "token_buffer": list(self._token_buffer),
        }

    def load_checkpoint_state(self, state):
        """Load saved state for resumption. Call before creating the iterator."""
        self._ds.load_state_dict(state["hf_dataset_state"])
        self._token_buffer = list(state["token_buffer"])
        self._state_restored = True
        if self.rank == 0:
            print("[INFO] Restored dataset iterator state from checkpoint")

    def _create_hf_dataset(self):
        """Create the HF streaming dataset with all transforms applied."""
        if self._local_dataset_path is not None:
            ds = load_dataset(self._local_dataset_path, name=self.hf_name,
                              split=self.ds_split, streaming=True)
        else:
            ds = load_dataset(self.hf_path, name=self.hf_name, split=self.ds_split,
                              streaming=True, cache_dir="./hf_streaming_cache/")

        # Shuffle before skip/take to preserve shard-level shuffling.
        # Both splits use the same seed so skip/take carve the same boundary.
        ds = ds.shuffle(seed=self.seed, buffer_size=1_000_000)

        # Carve out validation from train when no dedicated val split exists
        if self.val_docs is not None:
            if self.split == "train":
                ds = ds.skip(self.val_docs)
            else:
                ds = ds.take(self.val_docs)

        # Shard across DDP ranks
        ds = split_dataset_by_node(ds, rank=self.rank, world_size=self.world_size)
        return ds

    def __iter__(self):
        total_tokens_yielded = 0

        # Drain any complete chunks already in the restored token buffer
        while len(self._token_buffer) >= self.max_seq_len:
            chunk = self._token_buffer[:self.max_seq_len]
            self._token_buffer = self._token_buffer[self.max_seq_len:]
            total_tokens_yielded += len(chunk)
            yield {"tok_seqs": torch.tensor(chunk, dtype=torch.long)}

        for example in self._ds:
            text = example[self.text_field]
            tokens = self.tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]

            # Add EOS between documents to prevent cross-document artifacts
            if self.tokenizer.eos_token_id is not None:
                tokens.append(self.tokenizer.eos_token_id)

            self._token_buffer.extend(tokens)

            # Yield complete chunks
            while len(self._token_buffer) >= self.max_seq_len:
                chunk = self._token_buffer[:self.max_seq_len]
                self._token_buffer = self._token_buffer[self.max_seq_len:]
                total_tokens_yielded += len(chunk)
                yield {"tok_seqs": torch.tensor(chunk, dtype=torch.long)}

        if self.rank == 0:
            print(f"[INFO] Dataset iterator exhausted after yielding {total_tokens_yielded:,} tokens (rank {self.rank})")
