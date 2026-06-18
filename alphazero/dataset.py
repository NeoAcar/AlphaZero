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
        # Boards are returned in their stored dtype (uint8 for compact shards).
        # The uint8 -> float/255 rescale is done in bulk on the GPU in the
        # training loop so the host->device transfer is 4x smaller. See train.py.
        board = self.boards[idx]
        value = self.values[idx]
        label = int(self.policy[idx].item())

        soft = None
        if self.legal_masks_packed is not None and self.smoothing > 0:
            packed = self.legal_masks_packed[idx].numpy()
            assert len(packed) * 8 >= self.K, (
                f"packed legal mask too short: {len(packed) * 8} bits < K={self.K}"
            )
            mask = np.unpackbits(packed, count=self.K).astype(bool)
            n_legal = int(mask.sum())
            if n_legal > 0:
                soft = torch.zeros(self.K, dtype=torch.float32)
                per_legal = self.smoothing / n_legal
                soft[torch.from_numpy(mask)] = per_legal
                soft[label] = 1.0 - self.smoothing + per_legal
            # else: corrupted mask with no legal moves -> uniform-over-K fallback below.
        if soft is None:
            soft = torch.full((self.K,), self.smoothing / self.K, dtype=torch.float32)
            soft[label] = 1.0 - self.smoothing + self.smoothing / self.K
        # Supervised positions always contribute to policy loss -> is_high_sim=1.
        return board, value, soft, torch.tensor(1.0, dtype=torch.float32)


class SelfPlayDataset(Dataset):
    """Self-play dataset: soft policy targets (MCTS visit distribution) +
    value targets (actual game outcomes).

    Supports two storage formats:

    * **Sparse pi (new fragmented self-play layout)**: keys ``pi_indices``
      (N, MAX_LEGAL) int16 with -1 padding and ``pi_values`` (N, MAX_LEGAL)
      float16. ``is_high_sim`` (N,) uint8 flag controls which positions
      contribute to policy loss (PCR). Dense pi is reconstructed on the fly.
    * **Dense pi (legacy format)**: ``pis`` (N, 4672) float32 plus optional
      ``is_high_sim``. Old-style monolithic selfplay.pt files.
    """

    def __init__(self, boards, values, pis=None, *,
                 pi_indices=None, pi_values=None, is_high_sim=None,
                 action_space: int = 4672):
        self.boards = boards
        self.values = values
        self.pis = pis                          # dense (legacy)
        self.pi_indices = pi_indices            # sparse (new)
        self.pi_values = pi_values
        self.K = action_space
        # None -> __getitem__ defaults hi to 1.0 (treat every position as
        # high-sim; matches pre-PCR behaviour).
        self.is_high_sim = is_high_sim
        if pis is None and pi_indices is None:
            raise ValueError("SelfPlayDataset needs either pis or pi_indices")

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        # Boards returned in stored dtype (uint8); GPU-side rescale in train.py.
        board = self.boards[idx]

        if self.pi_indices is not None:
            # Sparse → dense reconstruction.
            idxs = self.pi_indices[idx]                        # int16 (MAX_LEGAL,)
            vals = self.pi_values[idx]                         # float16 (MAX_LEGAL,)
            soft = torch.zeros(self.K, dtype=torch.float32)
            valid = idxs >= 0
            if valid.any():
                soft.scatter_(0, idxs[valid].long(), vals[valid].float())
        else:
            soft = self.pis[idx]

        if self.is_high_sim is not None:
            hi = self.is_high_sim[idx].float()
        else:
            hi = torch.tensor(1.0, dtype=torch.float32)
        return board, self.values[idx], soft, hi
