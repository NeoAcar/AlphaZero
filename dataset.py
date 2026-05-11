from torch.utils.data import Dataset
import torch


class ChessDataset(Dataset):
    """Supervised dataset: hard policy labels + value targets (Stockfish evals).

    The policy label is an int (the human-played move's action index). At
    __getitem__ time we expand it into a soft (4672,) one-hot vector with
    optional label-smoothing, so downstream loss code can use the same
    soft-target CE for both supervised and self-play data.
    """

    def __init__(self, boards, values, policy, label_smoothing: float = 0.0,
                 action_space: int = 4672):
        self.boards = boards
        self.values = values
        self.policy = policy
        self.K = action_space
        self.smoothing = float(label_smoothing)

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        board = self.boards[idx]
        value = self.values[idx]
        label = int(self.policy[idx].item())
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
        return self.boards[idx], self.values[idx], self.pis[idx]
