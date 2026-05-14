from torch.utils.data import Dataset
import numpy as np
import torch


class ChessDataset(Dataset):
    """Supervised dataset: hard policy labels + value targets (Stockfish evals).

    The policy label is an int (the human-played move's action index). At
    __getitem__ time we expand it into a soft (4672,) one-hot vector with
    optional label-smoothing, so downstream loss code can use the same
    soft-target CE for both supervised and self-play data.

    If `legal_masks_packed` is provided (bit-packed (N, 584) uint8 from
    `np.packbits`), label smoothing spreads the smoothing mass uniformly
    over the position's *legal* moves only, not all 4672 action indices.
    Without it, smoothing falls back to uniform-over-4672 (the older,
    slightly suboptimal behaviour kept for backwards-compat with shards
    that predate the legal-mask field).
    """

    def __init__(self, boards, values, policy, label_smoothing: float = 0.0,
                 action_space: int = 4672, legal_masks_packed=None):
        self.boards = boards
        self.values = values
        self.policy = policy
        self.K = action_space
        self.smoothing = float(label_smoothing)
        self.legal_masks_packed = legal_masks_packed   # (N, K/8) uint8 or None

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        board = self.boards[idx]
        if board.dtype == torch.uint8:
            board = board.float() / 255.0
        value = self.values[idx]
        label = int(self.policy[idx].item())

        if self.legal_masks_packed is not None and self.smoothing > 0:
            packed = self.legal_masks_packed[idx].numpy()
            mask = np.unpackbits(packed, count=self.K).astype(bool)
            n_legal = int(mask.sum())
            soft = torch.zeros(self.K, dtype=torch.float32)
            per_legal = self.smoothing / n_legal
            soft[torch.from_numpy(mask)] = per_legal
            soft[label] = 1.0 - self.smoothing + per_legal
        else:
            soft = torch.full((self.K,), self.smoothing / self.K, dtype=torch.float32)
            soft[label] = 1.0 - self.smoothing + self.smoothing / self.K
        return board, value, soft


class LazyShardDataset(Dataset):
    """One shard's positions, loaded lazily via mmap.

    Designed to be wrapped in `ConcatDataset([LazyShardDataset(...), ...])` so
    that the supervised training set can span hundreds of shards without ever
    materialising them all in RAM. Each instance carries only the shard path
    and a small int64 array of local indices (which positions in this shard
    belong to its split, e.g. train or val). The shard itself is mmap'd on
    first `__getitem__` and its tensor storages stay on disk -- the OS pages
    in only what's read.

    With DataLoader workers, each worker process re-loads the shard on its
    first access (via `__getstate__` clearing the cached handle on pickle).
    Across-epoch reuse benefits from OS page cache; persistent_workers=True
    on the DataLoader keeps the worker's mmap binding alive between epochs.

    Returns the same (board, value, soft_policy) tuple as ChessDataset, with
    legal-mask-aware label smoothing when the shard has the `legal_masks_packed`
    field.
    """

    def __init__(self, shard_path, local_indices, label_smoothing: float = 0.0,
                 action_space: int = 4672):
        self.shard_path = str(shard_path)
        self.local_indices = np.asarray(local_indices, dtype=np.int64)
        self.smoothing = float(label_smoothing)
        self.K = action_space
        self._d = None

    def _ensure_loaded(self):
        if self._d is None:
            self._d = torch.load(self.shard_path, map_location="cpu",
                                  weights_only=False, mmap=True)
        return self._d

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_d"] = None   # don't try to pickle mmap'd tensors across workers
        return s

    def __len__(self):
        return len(self.local_indices)

    def __getitem__(self, idx):
        d = self._ensure_loaded()
        i = int(self.local_indices[idx])
        board = d["boards"][i]
        if board.dtype == torch.uint8:
            board = board.float() / 255.0
        value = d["evals"][i]
        label = int(d["moves"][i].item())

        masks_packed = d.get("legal_masks_packed")
        if masks_packed is not None and self.smoothing > 0:
            packed = masks_packed[i].numpy()
            mask = np.unpackbits(packed, count=self.K).astype(bool)
            n_legal = int(mask.sum())
            soft = torch.zeros(self.K, dtype=torch.float32)
            per_legal = self.smoothing / n_legal
            soft[torch.from_numpy(mask)] = per_legal
            soft[label] = 1.0 - self.smoothing + per_legal
        else:
            soft = torch.full((self.K,), self.smoothing / self.K, dtype=torch.float32)
            soft[label] = 1.0 - self.smoothing + self.smoothing / self.K
        return board, value, soft


class SelfPlayDataset(Dataset):
    """Self-play dataset: soft policy targets (MCTS visit distribution) +
    value targets (actual game outcomes from each player's perspective)."""

    def __init__(self, boards, values, pis):
        self.boards = boards
        self.values = values
        self.pis = pis

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        board = self.boards[idx]
        if board.dtype == torch.uint8:
            board = board.float() / 255.0
        return board, self.values[idx], self.pis[idx]
