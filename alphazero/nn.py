import torch
import torch.nn as nn
import torch.nn.functional as F


def value_to_scalar(value_t: torch.Tensor, mode: str = "expected") -> torch.Tensor:
    """Normalise either tanh-scalar (B,1) or WDL-logits (B,3) value output
    to a scalar per sample. Returns shape (B,).

    mode controls how WDL collapses to a scalar:
      "expected"  -> P(W) - P(L)  in [-1, +1]   (expected score, default)
      "win_only"  -> P(W)         in [0, +1]    (just win probability;
                                                 treats draw == loss)

    Scalar (B,1) outputs are returned unchanged regardless of mode.
    """
    if value_t.shape[-1] == 3:
        wdl = torch.softmax(value_t, dim=-1)
        if mode == "expected":
            return wdl[..., 0] - wdl[..., 2]
        if mode == "win_only":
            return wdl[..., 0]
        raise ValueError(f"unknown value_scalar mode: {mode!r}")
    return value_t.squeeze(-1)


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.startBlock = nn.Sequential(
            nn.Conv2d(19, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.backBone = nn.Sequential(*[ResBlock() for _ in range(19)])

        self.policyHead = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 73, kernel_size=1, padding=0, bias=True),
            nn.Flatten(),
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.startBlock(x)
        x = self.backBone(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return value, policy

    def forward_policy(self, x):
        x = self.startBlock(x)
        x = self.backBone(x)
        return self.policyHead(x)


class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = F.relu(out)
        return out


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: per-channel gating from global pooled stats.

    Squeeze: global avg pool → (B, C). Excitation: two-layer FC bottleneck
    (C → C/r → C) with SiLU + sigmoid gate. Scale: multiply the input
    feature map by the (B, C, 1, 1) gate.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, channels, bias=False)

    def forward(self, x):
        b, c, _, _ = x.shape
        s = x.mean(dim=(2, 3))                 # (B, C) global avg pool
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s)).view(b, c, 1, 1)
        return x * s


class SEResBlock(nn.Module):
    def __init__(self, channels: int = 256, reduction: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels, reduction)

    def forward(self, x):
        residual = x
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + residual
        out = F.silu(out)
        return out


class SEResNet(nn.Module):
    """ResNet variant: SE channel attention in every block + SiLU everywhere."""

    def __init__(self, channels: int = 256, n_blocks: int = 19, reduction: int = 16):
        super().__init__()
        self.startBlock = nn.Sequential(
            nn.Conv2d(19, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )

        self.backBone = nn.Sequential(
            *[SEResBlock(channels, reduction) for _ in range(n_blocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, 73, kernel_size=1, padding=0, bias=True),
            nn.Flatten(),
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.startBlock(x)
        x = self.backBone(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return value, policy

    def forward_policy(self, x):
        x = self.startBlock(x)
        x = self.backBone(x)
        return self.policyHead(x)


class SEResNetWDL(nn.Module):
    """SEResNet variant with a 3-output WDL value head (Win/Draw/Loss logits).

    Same body + SE blocks + policy head as SEResNet. Only the value head
    differs: outputs raw (B, 3) logits, no activation. Apply softmax at the
    call site (loss uses cross-entropy; MCTS converts via P(W) - P(L) to get
    a scalar in [-1, +1]).
    """

    def __init__(self, channels: int = 256, n_blocks: int = 19, reduction: int = 16):
        super().__init__()
        self.startBlock = nn.Sequential(
            nn.Conv2d(19, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )

        self.backBone = nn.Sequential(
            *[SEResBlock(channels, reduction) for _ in range(n_blocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, 73, kernel_size=1, padding=0, bias=True),
            nn.Flatten(),
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.SiLU(),
            nn.Linear(256, 3),     # WDL logits; softmax applied externally
        )

    def forward(self, x):
        x = self.startBlock(x)
        x = self.backBone(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return value, policy

    def forward_policy(self, x):
        x = self.startBlock(x)
        x = self.backBone(x)
        return self.policyHead(x)
