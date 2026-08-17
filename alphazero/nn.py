import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Input plane formats. 19 is the current-board representation stored by the
# supervised Stockfish shards. 119 is the optional AlphaZero 8-frame history
# format (8 × 14 + 7 constants). Model constructors take ``in_channels``
# explicitly; checkpoint consumers should detect it from the first-conv weight
# shape (see detect_in_channels below).
INPUT_PLANES_LEGACY = 19
INPUT_PLANES_HISTORY = 119


def detect_in_channels(state_dict) -> int:
    """Read the first-conv input-channels from a model state_dict (or the
    inner ``model_state_dict`` of a full checkpoint dict). Used so callers
    can load 19- or 119-plane checkpoints without knowing in advance."""
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    # The paper-faithful Chessformer-GAB uses a token projection over the 12
    # piece planes and receives the seven rule planes separately through FiLM.
    # Its checkpoint therefore has no convolutional ``startBlock`` stem.
    gab_key = "token_projection.weight"
    if gab_key in state_dict:
        return int(state_dict[gab_key].shape[1]) + 7
    # Conditional SEResNet consumes only the twelve spatial piece planes in
    # its convolutional stem and reads the seven constant planes as a vector.
    if "metadata_encoder.0.weight" in state_dict:
        return INPUT_PLANES_LEGACY
    key = "startBlock.0.weight"  # shared name across ResNet/SEResNet/SEResNetWDL
    if key not in state_dict:
        raise KeyError(
            f"checkpoint missing {key!r}; cannot detect in_channels. "
            f"Keys present: {list(state_dict.keys())[:5]}..."
        )
    return int(state_dict[key].shape[1])


def value_to_scalar(value_t: torch.Tensor, mode: str = "expected") -> torch.Tensor:
    """Normalise either tanh-scalar (B,1) or WDL-logits (B,3) value output
    to a scalar per sample. Returns shape (B,).

    mode controls how WDL collapses to a scalar:
      "expected"  -> P(W) - P(L)  in [-1, +1]   (expected score, default)
      "win_only"  -> P(W)         in [0, +1]    (just win probability;
                                                 treats draw == loss)

    Scalar (B,1) outputs are returned unchanged regardless of mode.
    """
    scalar, _ = value_scalar_and_wdl(value_t, mode)
    return scalar


def value_scalar_and_wdl(value_t: torch.Tensor, mode: str = "expected"):
    """Like value_to_scalar but also returns the WDL probabilities so callers
    that need both (MCTS caches raw_nn_wdl for dashboards) don't softmax twice.

    Returns (scalar (B,), wdl_probs (B,3) | None). wdl_probs is None for plain
    tanh-scalar heads.
    """
    if value_t.shape[-1] == 3:
        wdl = torch.softmax(value_t, dim=-1)
        if mode == "expected":
            scalar = wdl[..., 0] - wdl[..., 2]
        elif mode == "win_only":
            scalar = wdl[..., 0]
        else:
            raise ValueError(f"unknown value_scalar mode: {mode!r}")
        return scalar, wdl
    return value_t.squeeze(-1), None


class ResNet(nn.Module):
    def __init__(self, in_channels: int = INPUT_PLANES_HISTORY):
        super().__init__()
        self.in_channels = in_channels
        self.startBlock = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
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

    def __init__(self, channels: int = 256, n_blocks: int = 19, reduction: int = 16,
                 in_channels: int = INPUT_PLANES_HISTORY):
        super().__init__()
        self.in_channels = in_channels
        self.startBlock = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
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


class SEResNetWDL(nn.Module):
    """SEResNet variant with a 3-output WDL value head (Win/Draw/Loss logits).

    Same body + SE blocks + policy head as SEResNet. Only the value head
    differs: outputs raw (B, 3) logits, no activation. Apply softmax at the
    call site (loss uses cross-entropy; MCTS converts via P(W) - P(L) to get
    a scalar in [-1, +1]).
    """

    def __init__(self, channels: int = 256, n_blocks: int = 19, reduction: int = 16,
                 in_channels: int = INPUT_PLANES_HISTORY):
        super().__init__()
        self.in_channels = in_channels
        self.startBlock = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
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


class ConditionalLeelaSE(nn.Module):
    """Leela-style gamma/beta SE conditioned on global rule metadata."""

    def __init__(self, channels: int, metadata_dim: int, reduction: int = 4):
        super().__init__()
        if channels % reduction:
            raise ValueError("channels must be divisible by se reduction")
        hidden = channels // reduction
        self.fc1 = nn.Linear(channels + metadata_dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, 2 * channels, bias=False)

    def forward(self, x: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        summary = x.mean(dim=(2, 3))
        hidden = F.silu(self.fc1(torch.cat((summary, metadata), dim=1)))
        gamma, beta = self.fc2(hidden).chunk(2, dim=1)
        gamma = torch.sigmoid(gamma).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class ConditionalSEResBlock(nn.Module):
    """Residual CNN block with metadata-conditioned Leela gamma/beta SE."""

    def __init__(self, channels: int, metadata_dim: int, se_reduction: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = ConditionalLeelaSE(channels, metadata_dim, se_reduction)

    def forward(self, x: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.silu(self.bn1(self.conv1(x)))
        x = self.se(self.bn2(self.conv2(x)), metadata)
        return F.silu(x + residual)


class ConditionalSEResNetWDL(nn.Module):
    """CNN student that separates board geometry from global chess metadata.

    The existing shards remain ``(B,19,8,8)``. Piece planes 0:12 feed the
    spatial convolutional tower; the seven constant planes are read once at
    ``[0, 0]`` and injected into every Leela-style gamma/beta SE unit.
    """

    def __init__(self, channels: int = 120, n_blocks: int = 11,
                 se_reduction: int = 4, metadata_dim: int = 32,
                 in_channels: int = INPUT_PLANES_LEGACY,
                 moves_left_loss: str = "huber"):
        super().__init__()
        if in_channels != INPUT_PLANES_LEGACY:
            raise ValueError("ConditionalSEResNetWDL requires the 19-plane input")
        if channels <= 0 or n_blocks <= 0 or metadata_dim <= 0:
            raise ValueError("channels, n_blocks and metadata_dim must be positive")
        if channels % se_reduction:
            raise ValueError("channels must be divisible by se_reduction")
        self.in_channels = in_channels
        self.channels = channels
        self.moves_left_loss = moves_left_loss
        self.startBlock = nn.Sequential(
            nn.Conv2d(12, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )
        self.metadata_encoder = nn.Sequential(
            nn.Linear(7, metadata_dim),
            nn.SiLU(),
        )
        self.backBone = nn.ModuleList([
            ConditionalSEResBlock(channels, metadata_dim, se_reduction)
            for _ in range(n_blocks)
        ])
        self.policyHead = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, 73, kernel_size=1, bias=True),
            nn.Flatten(),
        )
        self.valueHead = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.SiLU(),
            nn.Linear(256, 3),
        )
        self.movesLeftHead = make_moves_left_head(moves_left_loss, channels)
        self.register_buffer("static_valid", _build_attention_policy_action_map()[4])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1:] != (INPUT_PLANES_LEGACY, 8, 8):
            raise ValueError(f"expected (B,19,8,8), got {tuple(x.shape)}")
        metadata = self.metadata_encoder(x[:, 12:, 0, 0])
        features = self.startBlock(x[:, :12])
        for block in self.backBone:
            features = block(features, metadata)
        return features

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        features = self.encode(x)
        policy = self.policyHead(features)
        value = self.valueHead(features)
        if not return_aux:
            return value, policy
        tokens = features.flatten(2).transpose(1, 2)
        if self.moves_left_loss == "negbinom":
            mu, alpha = self.movesLeftHead(tokens)
            return value, policy, {"moves_left_mu": mu, "moves_left_alpha": alpha}
        return value, policy, {"moves_left_scaled": self.movesLeftHead(tokens)}

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


# ---------------------------------------------------------------------------
# ChessFormer: a 64-square transformer with chess-relative attention.
#
# This family deliberately keeps the existing network classes untouched. It is
# trained independently by train_chessformer.py; train.py and UCI keep their
# existing model-selection behaviour.
# ---------------------------------------------------------------------------


class ChessRelativeSelfAttention(nn.Module):
    """Self-attention with a learned representation of chessboard geometry.

    Every ordered square pair receives Q/K/V relation vectors selected by its
    two-dimensional displacement (delta-file, delta-rank).  On an 8x8 board
    each axis has 15 possible displacements, so the table contains only 225
    entries while still distinguishing ranks, files, diagonals and knight-like
    offsets.  This is the full Shaw-style relation used by Chessformer rather
    than a scalar relative-position bias.
    """

    BOARD_SIZE = 8
    RELATIVE_SPAN = 2 * BOARD_SIZE - 1

    def __init__(self, embed_dim: int = 256, num_heads: int = 8,
                 dropout: float = 0.0):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

        n_relations = self.RELATIVE_SPAN ** 2
        relation_shape = (n_relations, num_heads, self.head_dim)
        self.relative_q = nn.Parameter(torch.empty(relation_shape))
        self.relative_k = nn.Parameter(torch.empty(relation_shape))
        self.relative_v = nn.Parameter(torch.empty(relation_shape))

        # relative_index[i, j] identifies the displacement from query square i
        # to key square j. Tokens use the same row-major order as flatten(2).
        ranks, files = torch.meshgrid(
            torch.arange(self.BOARD_SIZE),
            torch.arange(self.BOARD_SIZE),
            indexing="ij",
        )
        coords = torch.stack((ranks.flatten(), files.flatten()), dim=-1)
        delta = coords[None, :, :] - coords[:, None, :]
        delta_rank = delta[..., 0] + self.BOARD_SIZE - 1
        delta_file = delta[..., 1] + self.BOARD_SIZE - 1
        relative_index = delta_rank * self.RELATIVE_SPAN + delta_file
        self.register_buffer("relative_index", relative_index.long())

        nn.init.normal_(self.relative_q, std=0.02)
        nn.init.normal_(self.relative_k, std=0.02)
        nn.init.normal_(self.relative_v, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        if tokens != self.BOARD_SIZE ** 2:
            raise ValueError(f"expected 64 square tokens, got {tokens}")

        qkv = self.qkv(x).reshape(
            batch, tokens, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        # (H, query-square, key-square, head-dim)
        rel_q = self.relative_q[self.relative_index].permute(2, 0, 1, 3)
        rel_k = self.relative_k[self.relative_index].permute(2, 0, 1, 3)
        rel_v = self.relative_v[self.relative_index].permute(2, 0, 1, 3)

        # (q_i + relative_q_ij) dot (k_j + relative_k_ij), expanded so
        # we never materialise a large (B,H,64,64,D) broadcast tensor.
        logits = torch.einsum("bhid,bhjd->bhij", q, k)
        logits = logits + torch.einsum("bhid,hijd->bhij", q, rel_k)
        logits = logits + torch.einsum("hijd,bhjd->bhij", rel_q, k)
        logits = logits + (rel_q * rel_k).sum(dim=-1).unsqueeze(0)
        attention = self.attn_dropout(F.softmax(logits * self.scale, dim=-1))

        content = torch.einsum("bhij,bhjd->bhid", attention, v)
        relation = torch.einsum("bhij,hijd->bhid", attention, rel_v)
        out = (content + relation).transpose(1, 2).reshape(batch, tokens, channels)
        return self.out_proj(out)


class SwiGLU(nn.Module):
    """Transformer feed-forward layer with a gated SiLU branch."""

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, 2 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(F.silu(gate) * value)


class MetadataFiLM(nn.Module):
    """Condition square features on global rule/game metadata.

    The final projection starts at zero, making FiLM an identity operation at
    initialisation; the model learns where colour, castling rights, move count
    and the rule-50 counter should alter its computation.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(embed_dim, 2 * embed_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(metadata).chunk(2, dim=-1)
        return x * (1.0 + torch.tanh(scale).unsqueeze(1)) + shift.unsqueeze(1)


class ChessFormerBlock(nn.Module):
    """Local board mixing + global chess-relative attention + SwiGLU.

    The depthwise convolution gives the network a cheap local inductive bias
    for pawn structures and adjacent pieces. Relative attention handles rays,
    pins and other long-range relationships in a single layer. LayerScale keeps
    the three residual branches stable when training from scratch.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8,
                 hidden_dim: int = 512, dropout: float = 0.0,
                 layer_scale_init: float = 1e-2):
        super().__init__()
        self.local_norm = nn.LayerNorm(embed_dim)
        self.local_depthwise = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=3, padding=1,
            groups=embed_dim, bias=False,
        )
        self.local_pointwise = nn.Conv2d(embed_dim, embed_dim, kernel_size=1,
                                         bias=False)

        self.attention_norm = nn.LayerNorm(embed_dim)
        self.metadata_film = MetadataFiLM(embed_dim)
        self.attention = ChessRelativeSelfAttention(embed_dim, num_heads, dropout)

        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = SwiGLU(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.local_scale = nn.Parameter(
            torch.full((embed_dim,), layer_scale_init)
        )
        self.attention_scale = nn.Parameter(
            torch.full((embed_dim,), layer_scale_init)
        )
        self.ffn_scale = nn.Parameter(
            torch.full((embed_dim,), layer_scale_init)
        )

    def forward(self, x: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape

        local = self.local_norm(x).transpose(1, 2).reshape(batch, channels, 8, 8)
        local = self.local_pointwise(F.silu(self.local_depthwise(local)))
        local = local.flatten(2).transpose(1, 2)
        x = x + self.dropout(local) * self.local_scale

        global_features = self.metadata_film(self.attention_norm(x), metadata)
        x = x + self.dropout(self.attention(global_features)) * self.attention_scale
        x = x + self.dropout(self.ffn(self.ffn_norm(x))) * self.ffn_scale
        return x


def _build_attention_policy_action_map():
    """Map the repository's 73x64 action layout to from/to square pairs.

    Geometrically impossible entries are marked invalid. Position-dependent
    illegality (occupied rays, check evasions, etc.) remains the job of the
    existing legal mask.
    """

    action_from = torch.zeros(4672, dtype=torch.long)
    action_to = torch.zeros(4672, dtype=torch.long)
    action_type = torch.zeros(4672, dtype=torch.long)
    promotion_piece = torch.full((4672,), -1, dtype=torch.long)
    valid = torch.zeros(4672, dtype=torch.bool)
    knight_moves = (
        (2, 1), (1, 2), (-1, 2), (-2, 1),
        (2, -1), (1, -2), (-1, -2), (-2, -1),
    )

    for move_type in range(73):
        for start in range(64):
            action = move_type * 64 + start
            start_file = start % 8
            start_rank = start // 8
            action_from[action] = start
            action_type[action] = move_type

            if move_type < 14:
                file_delta = (move_type % 7 + 1) * (1 if move_type < 7 else -1)
                rank_delta = 0
            elif move_type < 28:
                file_delta = 0
                rank_delta = (move_type % 7 + 1) * (
                    1 if move_type < 21 else -1
                )
            elif move_type < 56:
                distance = move_type % 7 + 1
                file_delta = distance * (
                    1 if move_type < 35 or 42 <= move_type < 49 else -1
                )
                rank_delta = distance * (1 if move_type < 42 else -1)
            elif move_type < 64:
                file_delta, rank_delta = knight_moves[move_type - 56]
            else:
                # Canonical inputs always place the player-to-move's pawns on
                # rank 7 before promotion, moving toward rank 8.
                if start_rank != 6:
                    continue
                promotion_offset = move_type - 64
                promotion_piece[action] = promotion_offset // 3  # N, B, R
                file_delta = promotion_offset % 3 - 1
                rank_delta = 1

            end_file = start_file + file_delta
            end_rank = start_rank + rank_delta
            if 0 <= end_file < 8 and 0 <= end_rank < 8:
                action_to[action] = end_rank * 8 + end_file
                valid[action] = True

    return action_from, action_to, action_type, promotion_piece, valid


class AttentionPolicyHead(nn.Module):
    """Score moves as relationships between their source and target squares."""

    def __init__(self, embed_dim: int = 256, policy_dim: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.from_projection = nn.Linear(embed_dim, policy_dim, bias=False)
        self.to_projection = nn.Linear(embed_dim, policy_dim, bias=False)
        self.promotion_projection = nn.Linear(embed_dim, 3, bias=True)
        self.move_type_bias = nn.Parameter(torch.zeros(73))
        self.scale = policy_dim ** -0.5

        mapping = _build_attention_policy_action_map()
        self.register_buffer("action_from", mapping[0])
        self.register_buffer("action_to", mapping[1])
        self.register_buffer("action_type", mapping[2])
        self.register_buffer("promotion_piece", mapping[3])
        self.register_buffer("static_valid", mapping[4])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        from_features = self.from_projection(x)
        to_features = self.to_projection(x)
        pair_logits = torch.einsum(
            "bid,bjd->bij", from_features, to_features
        ) * self.scale

        logits = pair_logits[:, self.action_from, self.action_to]
        logits = logits + self.move_type_bias[self.action_type]

        # Underpromotion types share a from/to traversal, so predict the choice
        # of knight/bishop/rook from the source-square representation.
        promotion_mask = self.promotion_piece >= 0
        promotion_logits = self.promotion_projection(x)
        # Gather one promotion score for every action and zero the ordinary
        # moves. This is equivalent to assigning only the underpromotion slice,
        # but avoids index_put_: in-place advanced indexing cannot be captured
        # safely by torch.compile's CUDA Graph path during inference.
        promotion_piece = self.promotion_piece.clamp_min(0)
        promotion_bonus = promotion_logits[
            :, self.action_from, promotion_piece
        ] * promotion_mask.to(dtype=logits.dtype).unsqueeze(0)
        logits = logits + promotion_bonus

        # Keep this finite: soft-target CE evaluates target * log_probability,
        # and zero * -inf would otherwise produce NaNs for invalid actions.
        return logits.masked_fill(~self.static_valid.unsqueeze(0), -1e4)


class LearnedAttentionPool(nn.Module):
    """Pool 64 square tokens with a learned query instead of crushing channels."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.query = nn.Parameter(torch.empty(embed_dim))
        nn.init.normal_(self.query, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        weights = torch.einsum("bnd,d->bn", x, self.query)
        weights = F.softmax(weights / math.sqrt(x.size(-1)), dim=-1)
        return torch.einsum("bn,bnd->bd", weights, x)


class NegativeBinomialMovesLeftHead(nn.Module):
    """Predict a negative-binomial distribution over remaining plies.

    ``mu`` is the expected number of remaining plies and ``alpha`` controls
    over-dispersion under the NB2 parameterisation:

        variance = mu + alpha * mu**2

    The ChessFormer body already supplies globally contextualised square
    tokens, so an attention pool plus a small MLP is cheaper than reshaping to
    a feature map and applying a separate convolution/flatten tower.
    """

    def __init__(self, embed_dim: int, hidden_dim: int | None = None,
                 min_mu: float = 1e-3, min_alpha: float = 1e-3):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim // 2
        self.min_mu = min_mu
        self.min_alpha = min_alpha
        self.pool = LearnedAttentionPool(embed_dim)
        self.shared = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Mish(),
        )
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.alpha_head = nn.Linear(hidden_dim, 1)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(self.pool(tokens))
        mu = F.softplus(self.mu_head(features)) + self.min_mu
        alpha = F.softplus(self.alpha_head(features)) + self.min_alpha
        return mu, alpha


class ScaledHuberMovesLeftHead(nn.Module):
    """Predict remaining plies on Leela's ``plies / 20`` regression scale."""

    def __init__(self, embed_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim // 2
        self.pool = LearnedAttentionPool(embed_dim)
        self.shared = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Mish(),
        )
        # Intentionally unconstrained: SmoothL1 regression needs no
        # distributional parameter or artificial positive-output transform.
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.output(self.shared(self.pool(tokens)))


def make_moves_left_head(loss: str, embed_dim: int) -> nn.Module:
    if loss == "negbinom":
        return NegativeBinomialMovesLeftHead(embed_dim)
    if loss == "huber":
        return ScaledHuberMovesLeftHead(embed_dim)
    raise ValueError(f"unsupported moves-left loss: {loss!r}")


def negative_binomial_nll_loss(target: torch.Tensor, mu: torch.Tensor,
                               alpha: torch.Tensor,
                               reduction: str = "mean") -> torch.Tensor:
    """Negative-binomial NLL using mean/over-dispersion (NB2) parameters.

    ``target`` must contain raw non-negative remaining-ply counts, not a log or
    scaled target. Computation is forced to float32 for stable ``lgamma`` under
    mixed-precision training. With ``r = 1 / alpha`` this is equivalent to:

        NB(total_count=r, mean=mu)

    but avoids the overhead and parameter-convention ambiguity of constructing
    a ``torch.distributions.NegativeBinomial`` object for every batch.
    """

    if mu.shape != alpha.shape:
        raise ValueError(
            f"mu and alpha must have identical shapes, got {mu.shape} and {alpha.shape}"
        )
    if target.numel() != mu.numel():
        raise ValueError(
            f"target and prediction sizes differ: {target.numel()} vs {mu.numel()}"
        )
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(f"unsupported reduction: {reduction!r}")

    # Preserve device while leaving autocast: lgamma and the subtraction of
    # nearby large terms are substantially safer in fp32 than fp16/bfloat16.
    target = target.reshape_as(mu).to(dtype=torch.float32)
    mu = mu.to(dtype=torch.float32).clamp_min(1e-8)
    alpha = alpha.to(dtype=torch.float32).clamp_min(1e-8)

    r = alpha.reciprocal()
    alpha_mu = alpha * mu
    log_normalizer = torch.log1p(alpha_mu)
    log_prob = (
        torch.lgamma(target + r)
        - torch.lgamma(r)
        - torch.lgamma(target + 1.0)
        - r * log_normalizer
        + torch.xlogy(target, alpha_mu)
        - target * log_normalizer
    )
    loss = -log_prob

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


class ChessFormerWDL(nn.Module):
    """Chess transformer with WDL, moves-left and attention policy.

    Default ``forward`` remains compatible with the existing inference API and
    returns ``(wdl_logits, policy_logits)``. Training can request the auxiliary
    predictions with ``return_aux=True`` and receives a third dictionary:

        NB: {"moves_left_mu": (B, 1), "moves_left_alpha": (B, 1)}
        Huber: {"moves_left_scaled": (B, 1)}

    The NB target is raw remaining plies. Huber uses the Leela-style target
    ``remaining_plies / 20``. Both use the existing ``positions_per_game``
    boundaries; no new labelled data is required.
    """

    def __init__(self, in_channels: int = INPUT_PLANES_LEGACY,
                 embed_dim: int = 384, n_blocks: int = 10,
                 num_heads: int = 12, hidden_dim: int = 768,
                 policy_dim: int = 192, dropout: float = 0.0,
                 moves_left_loss: str = "negbinom"):
        super().__init__()
        if in_channels < 7:
            raise ValueError("ChessFormerWDL expects piece/history planes plus 7 metadata planes")

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.moves_left_loss = moves_left_loss

        # Sequential naming preserves the current checkpoint input-plane
        # detection convention: startBlock.0.weight.shape[1] == in_channels.
        self.startBlock = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
        )
        self.square_embedding = nn.Parameter(torch.empty(64, embed_dim))
        self.metadata_projection = nn.Sequential(
            nn.Linear(7, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        nn.init.normal_(self.square_embedding, std=0.02)

        self.backBone = nn.ModuleList([
            ChessFormerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            for _ in range(n_blocks)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

        self.policyHead = AttentionPolicyHead(embed_dim, policy_dim)

        self.value_pool = LearnedAttentionPool(embed_dim)
        self.valueHead = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, 3),
        )
        self.movesLeftHead = make_moves_left_head(moves_left_loss, embed_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels or x.shape[-2:] != (8, 8):
            raise ValueError(
                f"expected (B,{self.in_channels},8,8), got {tuple(x.shape)}"
            )

        # Both the 19- and 119-plane formats store their seven constant planes
        # last. Reading one square avoids feeding 64 duplicate copies to the
        # metadata MLP; the 1x1 square projection still sees the complete input.
        metadata = self.metadata_projection(x[:, -7:, 0, 0])
        tokens = self.startBlock(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.square_embedding.unsqueeze(0) + metadata.unsqueeze(1)

        for block in self.backBone:
            tokens = block(tokens, metadata)
        return self.final_norm(tokens)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        tokens = self.encode(x)
        policy = self.policyHead(tokens)
        value_features = self.value_pool(tokens)
        value = self.valueHead(value_features)

        if not return_aux:
            return value, policy

        if self.moves_left_loss == "negbinom":
            moves_left_mu, moves_left_alpha = self.movesLeftHead(tokens)
            aux = {
                "moves_left_mu": moves_left_mu,
                "moves_left_alpha": moves_left_alpha,
            }
        else:
            aux = {"moves_left_scaled": self.movesLeftHead(tokens)}
        return value, policy, aux

    def parameter_count(self, exclude_input_projection: bool = False) -> int:
        """Return trainable parameters, optionally excluding the input stem."""
        total = sum(parameter.numel() for parameter in self.parameters())
        if exclude_input_projection:
            total -= sum(parameter.numel() for parameter in self.startBlock.parameters())
        return total


# ---------------------------------------------------------------------------
# Paper-faithful 5M Chessformer with our state FiLM + moves-left auxiliary.
#
# The body below tracks CSSLab's released Maia-3 5M implementation: eight
# post-norm encoder blocks, RMSNorm, 8 heads of dimension 32, GELU 2x MLP,
# no QKV biases, zero dropout, and the shared-template pooled GAB generator.
# Deliberate, documented deviations are only (1) rule-plane FiLM in place of
# Maia's two Elo embeddings and (2) our negative-binomial moves-left head.
# ---------------------------------------------------------------------------


class PaperGABSelfAttention(nn.Module):
    """Maia-3 5M Geometric Attention Bias (GAB) attention layer.

    The GAB generator is per block, while ``gab_weight`` is shared by every
    block exactly as in the released Chessformer code. With ``per_square=0``
    the paper's efficient 5M variant mean-pools the square tokens before
    generating its dynamic 64x64 attention biases.
    """

    def __init__(self, embed_dim: int, num_heads: int,
                 gab_weight: nn.Parameter, gab_gen_size: int = 64,
                 gab_intermediate_dim: int = 64):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.gab_gen_size = gab_gen_size
        # Paper 5M: PyTorch MHA with all Q/K/V/output biases disabled.
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
            bias=False,
        )
        self.sm2 = nn.Linear(embed_dim, gab_intermediate_dim)
        self.ln1 = nn.LayerNorm(gab_intermediate_dim)
        self.sm3 = nn.Linear(gab_intermediate_dim, num_heads * gab_gen_size)
        self.ln2 = nn.LayerNorm(num_heads * gab_gen_size)
        self.gab_weight = gab_weight

    def _gab_bias(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 64:
            raise ValueError(f"GAB expects exactly 64 square tokens, got {x.shape[1]}")
        batch = x.shape[0]
        # This is the paper's 5M pooled-GAB branch (gab_per_square_dim=0).
        y = x.mean(dim=1)
        y = F.gelu(self.sm2(y))
        y = self.ln1(y)
        y = F.gelu(self.sm3(y))
        y = self.ln2(y).view(batch, self.num_heads, self.gab_gen_size)
        bias = torch.einsum("bhi,oi->bho", y, self.gab_weight)
        return bias.view(batch, self.num_heads, 64, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MHA-equivalent attention expressed through PyTorch SDPA.

        Keeping the published ``nn.MultiheadAttention`` parameter layout
        preserves existing checkpoints exactly. Splitting its packed QKV
        projection ourselves avoids PyTorch 2.6 Inductor's broken
        ``MultiheadAttention + per-head float mask`` compilation path and lets
        SDPA select the best CUDA attention kernel directly.
        """
        batch, sequence, embed_dim = x.shape
        head_dim = embed_dim // self.num_heads
        qkv = F.linear(x, self.attention.in_proj_weight)
        query, key, value = qkv.chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch, sequence, self.num_heads, head_dim
            ).transpose(1, 2)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)
        bias = self._gab_bias(x)
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=bias, dropout_p=0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch, sequence, embed_dim
        )
        return F.linear(attended, self.attention.out_proj.weight)


class PaperChessformerFiLMBlock(nn.Module):
    """Released 5M encoder block plus identity-initialised metadata FiLM."""

    def __init__(self, embed_dim: int, num_heads: int,
                 gab_weight: nn.Parameter, mlp_dim: int = 512):
        super().__init__()
        self.metadata_film = MetadataFiLM(embed_dim)
        self.self_attn = PaperGABSelfAttention(
            embed_dim, num_heads, gab_weight,
            gab_gen_size=64, gab_intermediate_dim=64,
        )
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        # Paper code selects RMSNorm for every released 5M/23M/79M model.
        self.norm1 = nn.RMSNorm(embed_dim)
        self.norm2 = nn.RMSNorm(embed_dim)

    def forward(self, x: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        # FiLM is the sole body-level departure. It starts as identity, so the
        # unconditioned initial model is exactly the paper block.
        attended = self.self_attn(self.metadata_film(x, metadata))
        x = self.norm1(x + attended)                 # paper: post-norm
        feedforward = self.linear2(F.gelu(self.linear1(x)))
        return self.norm2(x + feedforward)            # paper: post-norm


class PaperChessformerPolicyHead(nn.Module):
    """Paper source-destination policy, remapped to this repo's 4672 actions.

    Chessformer natively emits 4096 from/to logits plus 256 promotion logits
    (4352 UCI-style actions). Our stored targets and legal masks use the
    AlphaZero 73x64 layout. Ordinary moves are therefore gathered from the
    identical 64x64 source-destination matrix; the three underpromotion
    channels receive the paper's destination-conditioned N/B/R logits. This
    is only an output-index adaptation, not a different policy computation.
    """

    def __init__(self, embed_dim: int = 256, policy_dim: int = 256):
        super().__init__()
        self.from_projection = nn.Linear(embed_dim, policy_dim, bias=False)
        self.to_projection = nn.Linear(embed_dim, policy_dim, bias=False)
        # Paper order is Q, R, B, N. Q promotions use the base traversal in
        # our 73-plane action encoding; N/B/R occupy its dedicated planes.
        self.promotion_projection = nn.Linear(policy_dim, 4, bias=False)
        # Tensor scales avoid a PyTorch 2.6 Inductor matcher bug which mistakes
        # linear-output * Python-float patterns for weight-only quantization.
        # They are architectural constants, so they need not enter checkpoints.
        self.register_buffer(
            "scale", torch.tensor(policy_dim ** -0.5), persistent=False
        )
        self.register_buffer(
            "promotion_scale", torch.tensor(policy_dim ** 0.5), persistent=False
        )

        mapping = _build_attention_policy_action_map()
        self.register_buffer("action_from", mapping[0])
        self.register_buffer("action_to", mapping[1])
        self.register_buffer("promotion_piece", mapping[3])
        self.register_buffer("static_valid", mapping[4])
        paper_promotion = torch.tensor([3, 2, 1], dtype=torch.long)  # N,B,R -> paper Q,R,B,N
        underpromotion_index = torch.zeros(4672, dtype=torch.long)
        underpromotion_mask = mapping[3] >= 0
        underpromotion_index[underpromotion_mask] = paper_promotion[
            mapping[3][underpromotion_mask]
        ]
        self.register_buffer("underpromotion_index", underpromotion_index)
        # Kept out of checkpoints: it is entirely determined by the fixed
        # 4672-action map. A boolean buffer lets compiled inference apply the
        # promotion additions without a per-forward dtype cast.
        self.register_buffer("underpromotion_mask", underpromotion_mask,
                             persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        source = self.from_projection(x)
        destination = self.to_projection(x)
        pair_logits = torch.einsum("bid,bjd->bij", source, destination) * self.scale
        logits = pair_logits[:, self.action_from, self.action_to]

        # The reference code obtains promotion biases from destination features
        # and rescales them by sqrt(head_dim) before adding to the base score.
        promo = self.promotion_projection(destination) * self.promotion_scale
        promo_bonus = promo[:, self.action_to, self.underpromotion_index]
        promo_bonus = torch.where(
            self.underpromotion_mask.unsqueeze(0), promo_bonus,
            torch.zeros_like(promo_bonus),
        )
        logits = logits + promo_bonus
        return logits.masked_fill(~self.static_valid.unsqueeze(0), -1e4)


class Chessformer5MFiLMWDL(nn.Module):
    """Maia-3/Chessformer 5M body for our 19-plane supervised pipeline.

    Input is fixed to the existing 19-plane representation. The first twelve
    planes become the paper's square-token features; the final seven rule/game
    planes are encoded once and supplied to each block through FiLM. The model
    retains the paper WDL and source-destination policy heads, plus our
    selectable remaining-plies auxiliary head.
    """

    def __init__(self, in_channels: int = INPUT_PLANES_LEGACY,
                 embed_dim: int = 256, n_blocks: int = 8,
                 num_heads: int = 8, mlp_ratio: float = 2.0,
                 policy_dim: int = 256, moves_left_loss: str = "negbinom"):
        super().__init__()
        if in_channels != INPUT_PLANES_LEGACY:
            raise ValueError(
                "Chessformer5MFiLMWDL is the paper 5M current-board variant "
                f"and requires {INPUT_PLANES_LEGACY} planes, got {in_channels}"
            )
        if embed_dim != 256 or n_blocks != 8 or num_heads != 8 or mlp_ratio != 2.0 \
                or policy_dim != 256:
            raise ValueError(
                "this class intentionally fixes the published 5M body: "
                "embed_dim=256, n_blocks=8, num_heads=8, mlp_ratio=2, policy_dim=256"
            )
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.moves_left_loss = moves_left_loss
        self.token_projection = nn.Linear(12, embed_dim)
        self.metadata_projection = nn.Sequential(
            nn.Linear(7, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # Shared GAB template matrix: 64*64 square-pairs x 64 generated bases.
        self.gab_shared_weight = nn.Parameter(torch.empty(64 * 64, 64))
        nn.init.xavier_normal_(self.gab_shared_weight)
        self.backBone = nn.ModuleList([
            PaperChessformerFiLMBlock(
                embed_dim, num_heads, self.gab_shared_weight,
                mlp_dim=int(embed_dim * mlp_ratio),
            )
            for _ in range(n_blocks)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)       # paper encoder final norm
        self.value_norm = nn.LayerNorm(embed_dim)       # paper's last_ln after mean pool
        self.value_hidden = nn.Linear(embed_dim, embed_dim)
        self.value_out = nn.Linear(embed_dim, 3)
        self.policyHead = PaperChessformerPolicyHead(embed_dim, policy_dim)
        self.movesLeftHead = make_moves_left_head(moves_left_loss, embed_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1:] != (self.in_channels, 8, 8):
            raise ValueError(
                f"expected (B,{self.in_channels},8,8), got {tuple(x.shape)}"
            )
        # 12 piece planes are square features (paper tokenization). The seven
        # constant rule planes are global state used only by our FiLM addition.
        square_features = x[:, :12].flatten(2).transpose(1, 2)
        metadata = self.metadata_projection(x[:, 12:, 0, 0])
        tokens = self.token_projection(square_features)
        for block in self.backBone:
            tokens = block(tokens, metadata)
        return self.final_norm(tokens)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        tokens = self.encode(x)
        policy = self.policyHead(tokens)
        pooled = self.value_norm(tokens.mean(dim=1))
        value = self.value_out(F.relu(self.value_hidden(pooled)))
        if not return_aux:
            return value, policy
        if self.moves_left_loss == "negbinom":
            moves_left_mu, moves_left_alpha = self.movesLeftHead(tokens)
            return value, policy, {
                "moves_left_mu": moves_left_mu,
                "moves_left_alpha": moves_left_alpha,
            }
        return value, policy, {"moves_left_scaled": self.movesLeftHead(tokens)}

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
