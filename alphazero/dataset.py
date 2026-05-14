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
