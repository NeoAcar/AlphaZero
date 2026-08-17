"""Train a 19-plane ChessFormer from scratch on the Stockfish shards.

This is deliberately independent from train.py. It trains only on supervised
``gen_sf_data.py`` shards and never constructs 119-plane history inputs.

The three objectives are:

* played-move policy cross entropy (optionally smoothed over legal moves),
* Stockfish soft W/D/L cross entropy,
* negative-binomial NLL for raw remaining plies.

Example:

    python train_chessformer.py \
        --shards-dir /content/AlphaZero/data/sf_shards_v2 \
        --epochs 10 --batch-size 256 --precision bf16
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import wandb  # type: ignore
except ImportError:
    wandb = None  # type: ignore

from alphazero.nn import (
    INPUT_PLANES_LEGACY,
    Chessformer5MFiLMWDL,
    ChessFormerWDL,
    ConditionalSEResNetWDL,
    negative_binomial_nll_loss,
)


ACTION_SPACE = 4672
PACKED_ACTION_SPACE = ACTION_SPACE // 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="19-plane supervised ChessFormer training (AdamW + SWA)"
    )
    parser.add_argument("--shards-dir", required=True,
                        help="gen_sf_data.py shard directory (searched recursively)")
    parser.add_argument("--max-shards", type=int, default=None,
                        help="load only the first N shards; useful for a smoke test")
    parser.add_argument("--checkpoint-dir", default="./checkpoints/chessformer_19")
    parser.add_argument("--resume", default=None,
                        help="resume a full checkpoint produced by this script")
    parser.add_argument(
        "--architecture", choices=("hybrid", "paper_gab_5m", "conditional_se"), default="hybrid",
        help="hybrid = existing 22M local-CNN ChessFormer; paper_gab_5m = "
             "paper 5M GAB Chessformer with our FiLM/moves-left additions; "
             "conditional_se = metadata-conditioned Leela-style CNN student",
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=137)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-val-batches", type=int, default=0,
                        help="0 evaluates all validation batches")

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"),
                        default="constant",
                        help="constant keeps LR fixed; cosine retains the old warmup/decay recipe")
    parser.add_argument("--warmup-fraction", type=float, default=0.03,
                        help="only used with --lr-schedule cosine")
    parser.add_argument("--swa-start-fraction", type=float, default=0.85)
    parser.add_argument("--swa-lr-ratio", type=float, default=0.05,
                        help="final/SWA learning rate divided by peak LR; cosine only")
    parser.add_argument("--swa-update-every", type=int, default=100,
                        help="average weights every N optimizer updates in SWA phase")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"),
                        default="bf16")

    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--wdl-weight", type=float, default=1.0)
    parser.add_argument("--moves-left-weight", type=float, default=0.05)
    parser.add_argument("--moves-left-loss", choices=("huber", "negbinom"),
                        default="huber",
                        help="huber regresses remaining plies / 20; negbinom uses mu/alpha NLL")
    parser.add_argument("--label-smoothing", type=float, default=0.05)

    # Defaults target roughly the old 22.7M-parameter model without merely
    # widening/deepening a residual tower.
    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--policy-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--se-channels", type=int, default=120,
                        help="conditional_se only: convolution width")
    parser.add_argument("--se-blocks", type=int, default=11,
                        help="conditional_se only: residual block count")
    parser.add_argument("--se-reduction", type=int, default=4,
                        help="conditional_se only: Leela SE bottleneck ratio")
    parser.add_argument("--metadata-dim", type=int, default=32,
                        help="conditional_se only: shared rule-metadata embedding size")

    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.grad_accum <= 0:
        parser.error("epochs, batch-size and grad-accum must be positive")
    if args.num_workers < 0:
        parser.error("num-workers cannot be negative")
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("val-fraction must be between 0 and 1")
    if not 0.0 <= args.label_smoothing < 1.0:
        parser.error("label-smoothing must be in [0, 1)")
    if not 0.0 < args.swa_start_fraction < 1.0:
        parser.error("swa-start-fraction must be in (0, 1)")
    if args.lr_schedule == "cosine" \
            and not 0.0 <= args.warmup_fraction < args.swa_start_fraction:
        parser.error("cosine needs 0 <= warmup-fraction < swa-start-fraction")
    if not 0.0 < args.swa_lr_ratio <= 1.0:
        parser.error("swa-lr-ratio must be in (0, 1]")
    if args.swa_update_every <= 0:
        parser.error("swa-update-every must be positive")
    if args.architecture == "conditional_se":
        if args.se_channels <= 0 or args.se_blocks <= 0 or args.metadata_dim <= 0:
            parser.error("conditional_se width, blocks and metadata-dim must be positive")
        if args.se_reduction <= 0 or args.se_channels % args.se_reduction:
            parser.error("se-channels must be divisible by se-reduction")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ShardedGameDataset(Dataset):
    """A zero-copy game split over tensors already loaded from each shard.

    ``segments`` has one row per selected game: (shard, local_start, length).
    We retain the shard tensors instead of concatenating and boolean-copying
    38+ GB of boards into separate train/validation tensors.
    """

    def __init__(self, shards: list[dict[str, torch.Tensor]],
                 segments: np.ndarray, use_legal_masks: bool):
        if segments.ndim != 2 or segments.shape[1] != 3:
            raise ValueError(f"segments must have shape (G,3), got {segments.shape}")
        if len(segments) == 0:
            raise ValueError("dataset split contains no games")
        self.shards = shards
        self.segments = np.asarray(segments, dtype=np.int64)
        self.cumulative_positions = np.cumsum(self.segments[:, 2], dtype=np.int64)
        self.use_legal_masks = bool(use_legal_masks)
        self.empty_mask = torch.empty(0, dtype=torch.uint8)

    @property
    def game_lengths(self) -> np.ndarray:
        return self.segments[:, 2]

    def __len__(self) -> int:
        return int(self.cumulative_positions[-1])

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        game_index = int(np.searchsorted(
            self.cumulative_positions, index, side="right"
        ))
        game_begin = (
            0 if game_index == 0 else int(self.cumulative_positions[game_index - 1])
        )
        offset = index - game_begin
        shard_index, local_start, game_length = self.segments[game_index]
        local_index = int(local_start + offset)
        shard = self.shards[int(shard_index)]

        packed_mask = (
            shard["legal_masks_packed"][local_index]
            if self.use_legal_masks else self.empty_mask
        )
        # Targets for a length-L stored game are L, L-1, ..., 1.
        remaining_plies = int(game_length - offset)
        return (
            shard["boards"][local_index],
            shard["wdls"][local_index],
            shard["moves"][local_index],
            packed_mask,
            remaining_plies,
        )


def _tensor_shape(tensor: Any) -> tuple[int, ...]:
    return tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else ()


def load_sharded_split(args: argparse.Namespace) -> tuple[
        ShardedGameDataset, ShardedGameDataset, dict[str, Any]]:
    paths = sorted(Path(args.shards_dir).rglob("shard_*.pt"))
    if args.max_shards is not None:
        paths = paths[:args.max_shards]
    if not paths:
        raise RuntimeError(f"No shard_*.pt files found under {args.shards_dir}")

    print(f"Loading {len(paths)} shard(s) without concatenating board tensors...")
    shards: list[dict[str, torch.Tensor]] = []
    ppg_by_shard: list[np.ndarray] = []
    all_have_masks = True
    empty_games_skipped = 0

    for shard_index, path in enumerate(paths):
        raw = torch.load(path, map_location="cpu", weights_only=False)
        missing = {"boards", "moves", "wdls", "positions_per_game"} - raw.keys()
        if missing:
            raise RuntimeError(f"{path} is missing fields: {sorted(missing)}")

        boards = raw["boards"]
        moves = raw["moves"]
        wdls = raw["wdls"]
        ppg = np.asarray(raw["positions_per_game"], dtype=np.int64).reshape(-1)
        masks = raw.get("legal_masks_packed")

        n_positions = len(boards)
        if _tensor_shape(boards)[1:] != (INPUT_PLANES_LEGACY, 8, 8):
            raise RuntimeError(
                f"{path}: expected boards (N,19,8,8), got {_tensor_shape(boards)}"
            )
        if _tensor_shape(wdls) != (n_positions, 3):
            raise RuntimeError(
                f"{path}: expected wdls ({n_positions},3), got {_tensor_shape(wdls)}"
            )
        if _tensor_shape(moves) != (n_positions,):
            raise RuntimeError(
                f"{path}: expected moves ({n_positions},), got {_tensor_shape(moves)}"
            )
        if len(ppg) == 0 or np.any(ppg < 0) or int(ppg.sum()) != n_positions:
            raise RuntimeError(
                f"{path}: invalid positions_per_game (sum={int(ppg.sum())}, "
                f"positions={n_positions})"
            )
        # gen_sf_data.py records a game boundary even when a PGN has no
        # mainline moves. Such games contribute zero samples and therefore
        # must not enter the train/validation game split or moves-left moment
        # calculation. Removing them does not change any position offset:
        # their length is exactly zero.
        shard_empty_games = int(np.count_nonzero(ppg == 0))
        if shard_empty_games:
            empty_games_skipped += shard_empty_games
            ppg = ppg[ppg > 0]
        if masks is None:
            all_have_masks = False
        elif _tensor_shape(masks) != (n_positions, PACKED_ACTION_SPACE):
            raise RuntimeError(
                f"{path}: expected legal mask ({n_positions},{PACKED_ACTION_SPACE}), "
                f"got {_tensor_shape(masks)}"
            )

        shard: dict[str, torch.Tensor] = {
            "boards": boards,
            "moves": moves,
            "wdls": wdls,
        }
        if masks is not None:
            shard["legal_masks_packed"] = masks
        shards.append(shard)
        ppg_by_shard.append(ppg)
        empty_note = (
            f", {shard_empty_games:,} empty skipped" if shard_empty_games else ""
        )
        print(f"  [{shard_index + 1:02d}/{len(paths):02d}] {path.name}: "
              f"{n_positions:,} positions, {len(ppg):,} usable games{empty_note}")

    game_shard = np.concatenate([
        np.full(len(ppg), i, dtype=np.int64) for i, ppg in enumerate(ppg_by_shard)
    ])
    game_start = np.concatenate([
        np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(ppg[:-1])))
        for ppg in ppg_by_shard
    ])
    game_length = np.concatenate(ppg_by_shard)
    n_games = len(game_length)
    if n_games < 2:
        raise RuntimeError("at least two games are needed for a train/val split")

    rng = np.random.default_rng(args.split_seed)
    n_val = min(n_games - 1, max(1, round(n_games * args.val_fraction)))
    val_game = np.zeros(n_games, dtype=bool)
    val_game[rng.permutation(n_games)[:n_val]] = True
    all_segments = np.column_stack((game_shard, game_start, game_length))
    train_segments = all_segments[~val_game]
    val_segments = all_segments[val_game]

    train = ShardedGameDataset(shards, train_segments, all_have_masks)
    val = ShardedGameDataset(shards, val_segments, all_have_masks)
    stats = {
        "shards": len(paths),
        "games": n_games,
        "train_games": len(train_segments),
        "val_games": len(val_segments),
        "train_positions": len(train),
        "val_positions": len(val),
        "legal_masks": all_have_masks,
        "empty_games_skipped": empty_games_skipped,
    }
    print(
        f"Split: {stats['train_games']:,} train / {stats['val_games']:,} val games; "
        f"{len(train):,} train / {len(val):,} val positions"
    )
    if empty_games_skipped:
        print(f"Ignored {empty_games_skipped:,} empty PGN games (0 positions).")
    if not all_have_masks:
        print("WARNING: at least one shard has no legal_masks_packed; label "
              "smoothing will use the model's static geometric move mask.")
    return train, val, stats


def make_loader(dataset: Dataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(dataset, **kwargs)


def unpack_legal_masks(packed: torch.Tensor) -> torch.Tensor | None:
    """Unpack np.packbits masks batchwise on the accelerator (MSB first)."""
    if packed.numel() == 0:
        return None
    shifts = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    return ((packed.unsqueeze(-1) >> shifts) & 1).flatten(1).bool()


def policy_cross_entropy(logits: torch.Tensor, moves: torch.Tensor,
                         packed_masks: torch.Tensor, smoothing: float,
                         static_valid: torch.Tensor) -> torch.Tensor:
    """Hard CE plus legal-move smoothing without a dense target tensor."""
    log_prob = F.log_softmax(logits.float(), dim=1)
    hard = -log_prob.gather(1, moves[:, None]).squeeze(1)
    if smoothing <= 0.0:
        return hard.mean()

    legal = unpack_legal_masks(packed_masks)
    if legal is None:
        legal = static_valid.unsqueeze(0).expand(logits.size(0), -1)
    legal_count = legal.sum(dim=1)
    fallback = legal_count == 0
    if fallback.any():
        legal = legal.clone()
        legal[fallback] = static_valid
        legal_count = legal.sum(dim=1)
    smooth = -(log_prob * legal).sum(dim=1) / legal_count.clamp_min(1)
    return ((1.0 - smoothing) * hard + smoothing * smooth).mean()


def soft_wdl_cross_entropy(logits: torch.Tensor,
                           targets: torch.Tensor) -> torch.Tensor:
    return -(targets.float() * F.log_softmax(logits.float(), dim=1)).sum(dim=1).mean()


def model_configuration(args: argparse.Namespace,
                        resume_checkpoint: dict[str, Any] | None) -> dict[str, Any]:
    if resume_checkpoint is not None and "model_config" in resume_checkpoint:
        config = dict(resume_checkpoint["model_config"])
        if int(config.get("in_channels", -1)) != INPUT_PLANES_LEGACY:
            raise RuntimeError("resume checkpoint is not a 19-plane ChessFormer")
        return config
    if args.architecture == "paper_gab_5m":
        # Constructor deliberately fixes every body hyperparameter to the
        # released Maia-3 5M recipe. No accidental width/depth deviations.
        return {
            "in_channels": INPUT_PLANES_LEGACY,
            "moves_left_loss": args.moves_left_loss,
        }
    if args.architecture == "conditional_se":
        return {
            "in_channels": INPUT_PLANES_LEGACY,
            "channels": args.se_channels,
            "n_blocks": args.se_blocks,
            "se_reduction": args.se_reduction,
            "metadata_dim": args.metadata_dim,
            "moves_left_loss": args.moves_left_loss,
        }
    return {
        "in_channels": INPUT_PLANES_LEGACY,
        "embed_dim": args.embed_dim,
        "n_blocks": args.blocks,
        "num_heads": args.heads,
        "hidden_dim": args.hidden_dim,
        "policy_dim": args.policy_dim,
        "dropout": args.dropout,
        "moves_left_loss": args.moves_left_loss,
    }


def build_model(args: argparse.Namespace, model_config: dict[str, Any]) -> nn.Module:
    if args.architecture == "paper_gab_5m":
        return Chessformer5MFiLMWDL(**model_config)
    if args.architecture == "conditional_se":
        return ConditionalSEResNetWDL(**model_config)
    return ChessFormerWDL(**model_config)


def adamw_parameter_groups(model: torch.nn.Module, weight_decay: float):
    """Apply decay to matrix/kernel weights, not biases/norm/LayerScale vectors."""
    decay, no_decay = [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def make_lr_scheduler(optimizer: torch.optim.Optimizer, total_steps: int,
                      lr_schedule: str, warmup_fraction: float,
                      swa_start_fraction: float, swa_lr_ratio: float):
    warmup_steps = round(total_steps * warmup_fraction)
    minimum_swa_step = warmup_steps + 1 if lr_schedule == "cosine" else 1
    swa_start_step = max(minimum_swa_step, round(total_steps * swa_start_fraction))

    def multiplier(step: int) -> float:
        if lr_schedule == "constant":
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        if step < swa_start_step:
            denominator = max(1, swa_start_step - warmup_steps)
            progress = (step - warmup_steps) / denominator
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return swa_lr_ratio + (1.0 - swa_lr_ratio) * cosine
        return swa_lr_ratio

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    return scheduler, swa_start_step


def move_batch(batch, device: torch.device):
    boards, wdls, moves, masks, remaining = batch
    boards = boards.to(device, non_blocking=True)
    if boards.dtype == torch.uint8:
        boards = boards.float().div_(255.0)
    else:
        boards = boards.float()
    return (
        boards,
        wdls.to(device, non_blocking=True),
        moves.to(device, non_blocking=True).long(),
        masks.to(device, non_blocking=True),
        remaining.to(device, non_blocking=True).float(),
    )


def batch_losses(model: torch.nn.Module, batch, device: torch.device,
                 args: argparse.Namespace, static_valid: torch.Tensor,
                 autocast_enabled: bool, autocast_dtype: torch.dtype):
    boards, wdls, moves, masks, remaining = move_batch(batch, device)
    with torch.autocast(
        device_type=device.type,
        dtype=autocast_dtype,
        enabled=autocast_enabled,
    ):
        wdl_logits, policy_logits, aux = model(boards, return_aux=True)
        policy = policy_cross_entropy(
            policy_logits, moves, masks, args.label_smoothing, static_valid
        )
        wdl = soft_wdl_cross_entropy(wdl_logits, wdls)
        if args.moves_left_loss == "negbinom":
            moves_left = negative_binomial_nll_loss(
                remaining, aux["moves_left_mu"], aux["moves_left_alpha"]
            )
        else:
            moves_left = F.smooth_l1_loss(
                aux["moves_left_scaled"].float().squeeze(1), remaining / 20.0
            )
        total = (
            args.policy_weight * policy
            + args.wdl_weight * wdl
            + args.moves_left_weight * moves_left
        )

    # Diagnostic metrics are detached so they do not enlarge the autograd
    # graph. Expected WDL score (P(W)-P(L)) gives us an MSE that is close to
    # the old scalar value-head metric, while Brier score measures calibration
    # of the complete soft W/D/L distribution.
    with torch.no_grad():
        policy_prediction = policy_logits.detach().argmax(dim=1)
        wdl_probabilities = F.softmax(wdl_logits.detach().float(), dim=1)
        wdl_targets = wdls.float()
        predicted_score = wdl_probabilities[:, 0] - wdl_probabilities[:, 2]
        target_score = wdl_targets[:, 0] - wdl_targets[:, 2]
        value_error = predicted_score - target_score

        if args.moves_left_loss == "negbinom":
            prediction = aux["moves_left_mu"].detach().squeeze(1).float()
            alpha = aux["moves_left_alpha"].detach().squeeze(1).float()
            predicted_variance = prediction + alpha * prediction.square()
        else:
            prediction = aux["moves_left_scaled"].detach().squeeze(1).float() * 20.0
            alpha = torch.zeros_like(prediction)
            predicted_variance = torch.zeros_like(prediction)
        moves_left_error = prediction - remaining

    metrics = {
        "loss": total,
        "policy_loss": policy,
        "wdl_loss": wdl,
        "moves_left_loss": moves_left,
        "policy_correct": (policy_prediction == moves).sum(),
        "wdl_correct": (
            wdl_probabilities.argmax(dim=1) == wdl_targets.argmax(dim=1)
        ).sum(),
        "value_expected_sq_error": value_error.square().sum(),
        "value_expected_abs_error": value_error.abs().sum(),
        "wdl_brier_sum": (wdl_probabilities - wdl_targets).square().sum(),
        "moves_left_abs_error": moves_left_error.abs().sum(),
        "moves_left_sq_error": moves_left_error.square().sum(),
        "moves_left_target_sum": remaining.sum(),
        "moves_left_predicted_variance_sum": predicted_variance.sum(),
        "batch_size": boards.size(0),
        "mu_sum": prediction.sum(),
        "alpha_sum": alpha.sum(),
    }
    return metrics


LOSS_METRIC_KEYS = ("loss", "policy_loss", "wdl_loss", "moves_left_loss")
SUM_METRIC_KEYS = (
    "policy_correct",
    "wdl_correct",
    "value_expected_sq_error",
    "value_expected_abs_error",
    "wdl_brier_sum",
    "moves_left_abs_error",
    "moves_left_sq_error",
    "moves_left_target_sum",
    "moves_left_predicted_variance_sum",
    "mu_sum",
    "alpha_sum",
)


def empty_metric_sums() -> dict[str, float]:
    return {key: 0.0 for key in (*LOSS_METRIC_KEYS, *SUM_METRIC_KEYS)}


def accumulate_metrics(sums: dict[str, float], metrics: dict[str, Any]) -> int:
    batch_size = int(metrics["batch_size"])
    # Transfer every scalar in one tiny device->host copy. Calling .item()
    # separately for each metric would introduce many CUDA synchronizations
    # per batch and could make richer logging measurably slower.
    keys = (*LOSS_METRIC_KEYS, *SUM_METRIC_KEYS)
    values = torch.stack([
        metrics[key].detach().float() for key in keys
    ]).cpu().tolist()
    for key, value in zip(LOSS_METRIC_KEYS, values[:len(LOSS_METRIC_KEYS)]):
        sums[key] += value * batch_size
    for key, value in zip(SUM_METRIC_KEYS, values[len(LOSS_METRIC_KEYS):]):
        sums[key] += value
    return batch_size


def summarize_metrics(sums: dict[str, float], seen: int) -> dict[str, float]:
    if seen <= 0:
        raise RuntimeError("cannot summarize zero examples")
    return {
        "loss": sums["loss"] / seen,
        "policy_loss": sums["policy_loss"] / seen,
        "wdl_loss": sums["wdl_loss"] / seen,
        "moves_left_loss": sums["moves_left_loss"] / seen,
        "policy_accuracy": sums["policy_correct"] / seen,
        "wdl_accuracy": sums["wdl_correct"] / seen,
        "value_expected_mse": sums["value_expected_sq_error"] / seen,
        "value_expected_mae": sums["value_expected_abs_error"] / seen,
        "wdl_brier": sums["wdl_brier_sum"] / seen,
        "moves_left_mae": sums["moves_left_abs_error"] / seen,
        "moves_left_rmse": math.sqrt(sums["moves_left_sq_error"] / seen),
        "moves_left_target_mean": sums["moves_left_target_sum"] / seen,
        "moves_left_mu": sums["mu_sum"] / seen,
        "moves_left_alpha": sums["alpha_sum"] / seen,
        "moves_left_predicted_variance": (
            sums["moves_left_predicted_variance_sum"] / seen
        ),
    }


def wandb_metric_dict(prefix: str,
                      summary: dict[str, float]) -> dict[str, float]:
    """Namespace summaries and display accuracies in the old run's 0-100 scale."""
    result = {}
    for key, value in summary.items():
        if key.endswith("accuracy"):
            value *= 100.0
        result[f"{prefix}/{key}"] = value
    return result


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device,
             args: argparse.Namespace, static_valid: torch.Tensor,
             autocast_enabled: bool, autocast_dtype: torch.dtype) -> dict[str, float]:
    model.eval()
    sums = empty_metric_sums()
    seen = 0
    progress = tqdm(loader, desc="validation", leave=False)
    for batch_index, batch in enumerate(progress):
        if args.max_val_batches and batch_index >= args.max_val_batches:
            break
        metrics = batch_losses(
            model, batch, device, args, static_valid,
            autocast_enabled, autocast_dtype,
        )
        seen += accumulate_metrics(sums, metrics)

    if seen == 0:
        raise RuntimeError("validation loader produced no batches")
    return summarize_metrics(sums, seen)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(model: nn.Module, optimizer, scheduler, scaler,
                       swa_model: AveragedModel, epoch: int, global_step: int,
                       best_val: float, args: argparse.Namespace,
                       model_config: dict[str, Any], data_stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_type": type(model).__name__,
        "input_planes": INPUT_PLANES_LEGACY,
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "swa_state_dict": swa_model.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val,
        "args": vars(args),
        "data_stats": data_stats,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"Device: {device}; input: fixed {INPUT_PLANES_LEGACY} planes")

    if args.precision == "bf16" and device.type == "cuda" \
            and not torch.cuda.is_bf16_supported():
        print("WARNING: GPU has no native bf16; falling back to fp16")
        args.precision = "fp16"
    autocast_enabled = device.type == "cuda" and args.precision != "fp32"
    autocast_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and args.precision == "fp16"
    )

    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)

    train_dataset, val_dataset, data_stats = load_sharded_split(args)
    train_loader = make_loader(train_dataset, args, shuffle=True)
    val_loader = make_loader(val_dataset, args, shuffle=False)

    model_config = model_configuration(args, resume_checkpoint)
    if resume_checkpoint is not None:
        saved_arch = resume_checkpoint.get("args", {}).get("architecture", "hybrid")
        if saved_arch != args.architecture:
            raise RuntimeError(
                f"resume architecture mismatch: checkpoint={saved_arch!r}, "
                f"requested={args.architecture!r}"
            )
        saved_moves_left_loss = model_config.get("moves_left_loss", "negbinom")
        if saved_moves_left_loss != args.moves_left_loss:
            raise RuntimeError(
                "resume moves-left-loss mismatch: "
                f"checkpoint={saved_moves_left_loss!r}, requested={args.moves_left_loss!r}. "
                "A different auxiliary head requires a fresh run."
            )
    model = build_model(args, model_config)
    parameter_count = model.parameter_count()
    print(f"Model parameters: {parameter_count:,} ({parameter_count / 1e6:.3f}M)")

    model.to(device)
    static_valid = getattr(model.policyHead, "static_valid", None)
    if static_valid is None:
        static_valid = model.static_valid
    static_valid = static_valid.to(device)
    optimizer = torch.optim.AdamW(
        adamw_parameter_groups(model, args.weight_decay),
        lr=args.learning_rate,
        betas=(0.9, args.beta2),
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = updates_per_epoch * args.epochs
    scheduler, swa_start_step = make_lr_scheduler(
        optimizer, total_steps, args.lr_schedule, args.warmup_fraction,
        args.swa_start_fraction, args.swa_lr_ratio,
    )
    # The default buffer behaviour copies (rather than averages) integer/bool
    # policy maps such as action_from and static_valid.
    swa_model = AveragedModel(model).to(device)

    start_epoch = 0
    global_step = 0
    best_val = math.inf
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])
        if "swa_state_dict" in resume_checkpoint:
            swa_model.load_state_dict(resume_checkpoint["swa_state_dict"])
        start_epoch = int(resume_checkpoint.get("epoch", -1)) + 1
        global_step = int(resume_checkpoint.get("global_step", 0))
        best_val = float(resume_checkpoint.get("best_val_loss", math.inf))
        print(f"Resumed epoch={start_epoch}, optimizer_step={global_step:,}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (checkpoint_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "args": vars(args),
            "model_config": model_config,
            "parameters": parameter_count,
            "data_stats": data_stats,
        }, handle, indent=2)

    wandb_run = None
    if args.wandb_project:
        if wandb is None:
            raise RuntimeError("--wandb-project was given but wandb is not installed")
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={**vars(args), **model_config, "parameters": parameter_count},
        )

    schedule_description = (
        f"constant lr={args.learning_rate:g}"
        if args.lr_schedule == "constant"
        else f"cosine peak_lr={args.learning_rate:g}, final_lr="
             f"{args.learning_rate * args.swa_lr_ratio:g}"
    )
    print(
        f"AdamW {schedule_description}, decay={args.weight_decay:g}; "
        f"{total_steps:,} optimizer updates; SWA begins at update "
        f"{swa_start_step:,} ({args.swa_start_fraction:.0%})"
    )
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        n_batches = len(train_loader)
        running_sums = empty_metric_sums()
        running_examples = 0
        interval_examples = 0
        interval_started = time.perf_counter()
        epoch_started = interval_started
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for batch_index, batch in enumerate(progress):
            window_start = (batch_index // args.grad_accum) * args.grad_accum
            window_end = min(window_start + args.grad_accum, n_batches)
            accumulation_divisor = window_end - window_start

            metrics = batch_losses(
                model, batch, device, args, static_valid,
                autocast_enabled, autocast_dtype,
            )
            loss = metrics["loss"]
            scaler.scale(loss / accumulation_divisor).backward()
            batch_size = metrics["batch_size"]
            running_examples += accumulate_metrics(running_sums, metrics)
            interval_examples += batch_size

            update_now = batch_index + 1 == window_end
            if not update_now:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

            if global_step >= swa_start_step \
                    and (global_step - swa_start_step) % args.swa_update_every == 0:
                swa_model.update_parameters(model)

            if global_step % args.log_every == 0:
                batch_sums = empty_metric_sums()
                batch_seen = accumulate_metrics(batch_sums, metrics)
                batch_summary = summarize_metrics(batch_sums, batch_seen)
                interval_elapsed = max(time.perf_counter() - interval_started, 1e-9)
                log = {
                    **wandb_metric_dict("train", batch_summary),
                    # Compatibility aliases make direct plots against train.py
                    # runs possible without renaming old panels.
                    "train/policy_ce": batch_summary["policy_loss"],
                    "train/value_mse": batch_summary["value_expected_mse"],
                    "train/accuracy": 100.0 * batch_summary["policy_accuracy"],
                    f"train/moves_left_{args.moves_left_loss}": batch_summary["moves_left_loss"],
                    "train/grad_norm": float(grad_norm),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/samples_per_second": interval_examples / interval_elapsed,
                    "optimizer_step": global_step,
                    "epoch": epoch + 1,
                }
                if device.type == "cuda":
                    gib = 1024 ** 3
                    log.update({
                        "system/gpu_memory_allocated_gib": (
                            torch.cuda.memory_allocated(device) / gib
                        ),
                        "system/gpu_memory_reserved_gib": (
                            torch.cuda.memory_reserved(device) / gib
                        ),
                        "system/gpu_peak_memory_allocated_gib": (
                            torch.cuda.max_memory_allocated(device) / gib
                        ),
                    })
                progress.set_postfix(
                    loss=f"{log['train/loss']:.3f}",
                    policy=f"{log['train/policy_loss']:.3f}",
                    mlh=f"{log['train/moves_left_loss']:.3f}",
                    lr=f"{log['train/lr']:.2e}",
                )
                if wandb_run is not None:
                    wandb.log(log, step=global_step)
                interval_examples = 0
                interval_started = time.perf_counter()

        val = evaluate(
            model, val_loader, device, args, static_valid,
            autocast_enabled, autocast_dtype,
        )
        train_epoch = summarize_metrics(running_sums, running_examples)
        epoch_elapsed = max(time.perf_counter() - epoch_started, 1e-9)
        print(
            f"epoch {epoch + 1}: train={train_epoch['loss']:.4f}, "
            f"val={val['loss']:.4f}, policy_acc={val['policy_accuracy']:.2%}, "
            f"wdl_acc={val['wdl_accuracy']:.2%}, "
            f"value_mse={val['value_expected_mse']:.4f}, "
            f"moves_left_mae={val['moves_left_mae']:.3f}"
        )

        improved = val["loss"] < best_val
        if improved:
            best_val = val["loss"]
        payload = checkpoint_payload(
            model, optimizer, scheduler, scaler, swa_model,
            epoch, global_step, best_val, args, model_config,
            data_stats,
        )
        atomic_torch_save(payload, checkpoint_dir / "model_last.pth")
        if improved:
            atomic_torch_save(payload, checkpoint_dir / "model_best.pth")

        if wandb_run is not None:
            epoch_log = {
                **wandb_metric_dict("epoch_train", train_epoch),
                **wandb_metric_dict("val", val),
                # Exact names used by the previous train.py run.
                "epoch/train_loss": train_epoch["loss"],
                "epoch/train_policy_ce": train_epoch["policy_loss"],
                "epoch/train_value_mse": train_epoch["value_expected_mse"],
                "epoch/train_accuracy": 100.0 * train_epoch["policy_accuracy"],
                "epoch/val_loss": val["loss"],
                "epoch/val_policy_ce": val["policy_loss"],
                "epoch/val_value_mse": val["value_expected_mse"],
                "epoch/val_accuracy": 100.0 * val["policy_accuracy"],
                "epoch/seconds": epoch_elapsed,
                "epoch/samples_per_second": running_examples / epoch_elapsed,
                "epoch": epoch + 1,
                "epoch/n": epoch + 1,
            }
            wandb.log(epoch_log, step=global_step)

    if int(swa_model.n_averaged.item()) == 0:
        print("SWA phase had no scheduled sample; averaging final base weights once.")
        swa_model.update_parameters(model)
    swa_val = evaluate(
        swa_model, val_loader, device, args, static_valid,
        autocast_enabled, autocast_dtype,
    )
    swa_payload = {
        "model_type": type(model).__name__,
        "input_planes": INPUT_PLANES_LEGACY,
        "model_config": model_config,
        "model_state_dict": swa_model.module.state_dict(),
        "swa_n_averaged": int(swa_model.n_averaged.item()),
        "validation": swa_val,
        "args": vars(args),
        "data_stats": data_stats,
    }
    atomic_torch_save(swa_payload, checkpoint_dir / "model_swa.pth")
    print(
        f"SWA ({swa_payload['swa_n_averaged']} samples): "
        f"val={swa_val['loss']:.4f}, policy_acc={swa_val['policy_accuracy']:.2%}, "
        f"moves_left_mae={swa_val['moves_left_mae']:.3f}"
    )
    print(f"Saved final SWA model to {checkpoint_dir / 'model_swa.pth'}")

    if wandb_run is not None:
        wandb.log({f"swa/{key}": value for key, value in swa_val.items()},
                  step=global_step)
        wandb.finish()


if __name__ == "__main__":
    main()
