import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore

try:
    import wandb  # type: ignore
except ImportError:
    wandb = None  # type: ignore

import torch.nn.functional as F
from torch.utils.data import ConcatDataset

from alphazero.dataset import ChessDataset, SelfPlayDataset
from alphazero.nn import SEResNet, SEResNetWDL


def parse_args() -> dict:
    p = argparse.ArgumentParser(description="AlphaZero supervised pre-training on Lichess + Stockfish")
    p.add_argument("--config", default="train_config.json",
                   help="JSON config file. Any --flag below overrides its value.")
    p.add_argument("--shards-dir", help="directory of shard_NNNN.pt files from gen_sf_data.py")
    p.add_argument("--max-shards", type=int, help="when using --shards-dir, cap how many shards to load")
    p.add_argument("--epochs", type=int, help="number of training epochs")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--l2-weight", type=float, help="L2 weight decay")
    p.add_argument("--optimizer", choices=["sgd", "adamw"],
                   help="sgd = SGD + Nesterov momentum (AlphaZero / LC0 style, default). "
                        "adamw = AdamW (convenient but slightly worse minima at scale).")
    p.add_argument("--momentum", type=float,
                   help="SGD momentum coefficient (ignored for adamw); default 0.9")
    p.add_argument("--log-step", type=int, help="log every N iterations")
    p.add_argument("--label-smoothing", type=float,
                   help="cross-entropy label smoothing factor (e.g. 0.1)")
    p.add_argument("--checkpoint-dir", help="directory for model checkpoints")
    p.add_argument("--log-dir", help="directory for plain-text logs")
    p.add_argument("--value-head", choices=["scalar", "wdl"],
                   help="value head + target type. 'scalar': SEResNet + MSE on evals "
                        "(default). 'wdl': SEResNetWDL + soft-CE on wdls (N,3) targets.")
    p.add_argument("--vals-per-epoch", type=int, default=1,
                   help="how many validation passes to run per epoch (default: 1, "
                        "evenly spaced; the last one lands at the end of the epoch)")
    p.add_argument("--wandb-project", help="wandb project name; if unset, wandb is disabled")
    p.add_argument("--wandb-group", help="wandb group (for grouping runs from runner.py)")
    p.add_argument("--wandb-name", help="wandb run name")
    p.add_argument("--resume", help="path to a .pth checkpoint to resume from")
    p.add_argument("--val-fraction", type=float,
                   help="fraction of games held out for validation (split by game, not position)")
    p.add_argument("--split-seed", type=int,
                   help="RNG seed for train/val game split")
    p.add_argument("--self-play-data", action="append", default=None,
                   help="path to a self-play .pt produced by selfplay.py; can be passed multiple times")
    p.add_argument("--data-mix", choices=["supervised", "self_play", "both"],
                   help="which data sources to use (default: supervised if no self-play-data, "
                        "both if any --self-play-data is given)")
    cli = p.parse_args()

    with open(cli.config) as fp:
        cfg = json.load(fp)

    # Defaults for keys the original config didn't have.
    cfg.setdefault("label_smoothing", 0.0)
    cfg.setdefault("checkpoint_dir", "./checkpoints")
    cfg.setdefault("log_dir", "./logs")
    cfg.setdefault("resume", None)
    cfg.setdefault("val_fraction", 0.1)
    cfg.setdefault("split_seed", 137)
    cfg.setdefault("self_play_data", None)
    cfg.setdefault("data_mix", None)
    cfg.setdefault("wandb_project", None)
    cfg.setdefault("wandb_group", None)
    cfg.setdefault("wandb_name", None)
    cfg.setdefault("shards_dir", None)
    cfg.setdefault("max_shards", None)
    cfg.setdefault("value_head", "scalar")
    cfg.setdefault("vals_per_epoch", 1)
    cfg.setdefault("optimizer", "sgd")
    cfg.setdefault("momentum", 0.9)

    # CLI overrides (only when explicitly given).
    overrides = {
        "epochs": cli.epochs,
        "batch_size": cli.batch_size,
        "learning_rate": cli.learning_rate,
        "l2_weight": cli.l2_weight,
        "log_step": cli.log_step,
        "label_smoothing": cli.label_smoothing,
        "checkpoint_dir": cli.checkpoint_dir,
        "log_dir": cli.log_dir,
        "resume": cli.resume,
        "val_fraction": cli.val_fraction,
        "split_seed": cli.split_seed,
        "self_play_data": cli.self_play_data,
        "data_mix": cli.data_mix,
        "wandb_project": cli.wandb_project,
        "wandb_group": cli.wandb_group,
        "wandb_name": cli.wandb_name,
        "shards_dir": cli.shards_dir,
        "max_shards": cli.max_shards,
        "value_head": cli.value_head,
        "vals_per_epoch": cli.vals_per_epoch,
        "optimizer": cli.optimizer,
        "momentum": cli.momentum,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


class Train:
    def __init__(self, args: dict) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args

        self.lr = args["learning_rate"]
        self.l2_weight = args["l2_weight"]
        self.log_step = args["log_step"]
        self.epochs = args["epochs"]
        self.batch_size = args["batch_size"]
        self.label_smoothing = args["label_smoothing"]
        self.checkpoint_dir = args["checkpoint_dir"]
        self.log_dir = args["log_dir"]
        self.val_fraction = args["val_fraction"]
        self.split_seed = args["split_seed"]
        self.self_play_paths = args["self_play_data"] or []
        self.wandb_project = args.get("wandb_project")
        self.wandb_group = args.get("wandb_group")
        self.wandb_name = args.get("wandb_name")
        self.shards_dir = args.get("shards_dir")
        self.max_shards = args.get("max_shards")
        self.value_head = args.get("value_head", "scalar") or "scalar"
        if self.value_head not in ("scalar", "wdl"):
            raise ValueError(f"value_head must be 'scalar' or 'wdl', got {self.value_head!r}")
        self.vals_per_epoch = int(args.get("vals_per_epoch", 1) or 1)
        # Resolve data_mix default
        mix = args.get("data_mix")
        if mix is None:
            mix = "both" if self.self_play_paths else "supervised"
        self.data_mix = mix

        model_cls = SEResNetWDL if self.value_head == "wdl" else SEResNet
        self.model = model_cls().to(self.device)
        print(f"Model: {model_cls.__name__} (value_head={self.value_head})")

        opt_name = str(args.get("optimizer", "sgd")).lower()
        if opt_name == "sgd":
            momentum = float(args.get("momentum", 0.9))
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=momentum,
                nesterov=True,
                weight_decay=self.l2_weight,
            )
            print(f"Optimizer: SGD (Nesterov, momentum={momentum}, lr={self.lr}, "
                  f"weight_decay={self.l2_weight})")
            if self.lr < 1e-3:
                print(f"  WARNING: lr={self.lr} looks tuned for Adam. SGD usually wants "
                      f"~1e-2 → step down (e.g. 1e-2 → 1e-3 → 1e-4).")
        elif opt_name == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=self.lr, weight_decay=self.l2_weight,
            )
            print(f"Optimizer: AdamW (lr={self.lr}, weight_decay={self.l2_weight})")
        else:
            raise ValueError(f"unknown optimizer: {opt_name!r}")

        if args.get("resume"):
            print(f"Resuming from {args['resume']}")
            ckpt = torch.load(args["resume"], map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                try:
                    self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except (ValueError, KeyError) as e:
                    # Optimizer mismatch (e.g. AdamW ckpt → SGD now). Start
                    # optimizer state fresh; model weights are still inherited.
                    print(f"  optimizer state incompatible ({e}); starting "
                          f"optimizer state from scratch")

    def data_preparation(self):
        """Build train and val datasets according to self.data_mix.

        Returns a dict with:
          "train": ConcatDataset / single Dataset for training
          "val":   Dataset of held-out supervised positions (or None if
                   data_mix == "self_play" only)
        """
        train_datasets = []
        val_dataset = None

        if self.data_mix in ("supervised", "both"):
            if not self.shards_dir:
                raise RuntimeError(
                    "Supervised data requires --shards-dir pointing at gen_sf_data.py output."
                )
            sup_train, sup_val = self._load_sharded_split()
            train_datasets.append(sup_train)
            val_dataset = sup_val

        if self.data_mix in ("self_play", "both"):
            for path in self.self_play_paths:
                print(f"Loading self-play data: {path}")
                sp = torch.load(path, map_location="cpu", weights_only=False)
                ds = SelfPlayDataset(sp["boards"], sp["values"], sp["pis"])
                print(f"  {len(ds)} positions  (meta: {sp.get('meta', {})})")
                train_datasets.append(ds)

        if not train_datasets:
            raise RuntimeError("No training data sources configured.")

        train_dataset = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)
        print(f"Combined training set: {len(train_dataset)} positions "
              f"from {len(train_datasets)} source(s).")
        return {"train": train_dataset, "val": val_dataset}

    def _load_sharded_split(self) -> tuple[ChessDataset, ChessDataset]:
        """Load shard_NNNN.pt files from --shards-dir, concatenate, split by game.

        Each shard contains:
          boards (N,19,8,8) float32
          moves (N,) long
          evals (N,1) float32 in [-1, 1]
          positions_per_game (G,) int64
        We concatenate across shards, then split GAMES (not positions) into
        train/val using --val-fraction and --split-seed, so positions from a
        single game stay together.
        """
        from pathlib import Path

        # Match both flat layout (shard_*.pt at top) and multi-worker layout
        # (worker_*/shard_*.pt). rglob handles both.
        shard_paths = sorted(Path(self.shards_dir).rglob("shard_*.pt"))
        if not shard_paths:
            raise RuntimeError(f"No shard_*.pt files found in {self.shards_dir}")
        if self.max_shards is not None:
            shard_paths = shard_paths[: self.max_shards]
        print(f"Loading {len(shard_paths)} shard(s) from {self.shards_dir}")

        all_boards = []
        all_moves = []
        all_values = []      # holds either evals (scalar) or wdls (3-vec) depending on value_head
        all_masks = []
        all_ppg = []
        any_missing_masks = False
        wdl_key = "wdls"
        for sp in shard_paths:
            d = torch.load(sp, map_location="cpu", weights_only=False)
            all_boards.append(d["boards"])
            all_moves.append(d["moves"])
            if self.value_head == "wdl":
                if wdl_key not in d:
                    raise RuntimeError(
                        f"Shard {sp.name} lacks '{wdl_key}' field. Regenerate shards "
                        f"with gen_sf_data.py (recent versions store wdls), or train "
                        f"with --value-head scalar."
                    )
                all_values.append(d[wdl_key])
            else:
                all_values.append(d["evals"])
            if "legal_masks_packed" in d:
                all_masks.append(d["legal_masks_packed"])
            else:
                any_missing_masks = True
            all_ppg.append(d["positions_per_game"])
            print(f"  {sp.name}: {len(d['boards'])} positions, {len(d['positions_per_game'])} games")

        # Cat one tensor at a time and free the source list immediately, so we
        # never hold {per-shard list} + {cat result} in memory simultaneously.
        # On big runs (17M+ positions, ~30 GB of boards alone) the lazy doubling
        # during cat + the splitting copies otherwise breach Colab's 85 GB cap.
        boards = torch.cat(all_boards, dim=0);  del all_boards
        moves  = torch.cat(all_moves,  dim=0);  del all_moves
        values = torch.cat(all_values, dim=0);  del all_values
        if self.value_head == "scalar":
            values = torch.clamp(values, -1.0, 1.0)
        if any_missing_masks:
            print("  WARN: some shards lack 'legal_masks_packed' -- "
                  "label smoothing will spread over all 4672 indices for this run.")
            legal_masks_packed = None
            del all_masks
        else:
            legal_masks_packed = torch.cat(all_masks, dim=0) if all_masks else None
            del all_masks
        positions_per_game = np.concatenate(all_ppg);  del all_ppg
        n_games = len(positions_per_game)
        n_positions = int(positions_per_game.sum())
        assert n_positions == len(boards), \
            f"Position count mismatch: ppg sum = {n_positions}, boards = {len(boards)}"

        rng = np.random.RandomState(self.split_seed)
        shuffled = rng.permutation(n_games)
        n_val_games = max(1, int(round(n_games * self.val_fraction)))
        val_game_idx = np.zeros(n_games, dtype=bool)
        val_game_idx[shuffled[:n_val_games]] = True

        per_pos_is_val = np.repeat(val_game_idx, positions_per_game)
        val_mask = torch.from_numpy(per_pos_is_val)
        train_mask = ~val_mask

        n_train_games = n_games - n_val_games
        print(f"Sharded supervised: {n_train_games} train / {n_val_games} val games "
              f"({int(train_mask.sum())} / {int(val_mask.sum())} positions). "
              f"value_head={self.value_head}, target shape={tuple(values.shape[1:])}")

        # Split each tensor train/val, then drop the source immediately. Doing
        # all four boolean-indexes inline (the previous code) kept `boards`
        # alive for both the train and val copies, peaking at 3x the boards
        # tensor size. Pattern below peaks at 2x.
        train_boards = boards[train_mask]; val_boards = boards[val_mask]; del boards
        train_moves  = moves[train_mask];  val_moves  = moves[val_mask];  del moves
        train_values = values[train_mask]; val_values = values[val_mask]; del values
        if legal_masks_packed is not None:
            train_masks_packed = legal_masks_packed[train_mask]
            val_masks_packed   = legal_masks_packed[val_mask]
            del legal_masks_packed
        else:
            train_masks_packed = val_masks_packed = None

        train_ds = ChessDataset(train_boards, train_values, train_moves,
                                label_smoothing=self.label_smoothing,
                                legal_masks_packed=train_masks_packed)
        val_ds = ChessDataset(val_boards, val_values, val_moves,
                              label_smoothing=self.label_smoothing,
                              legal_masks_packed=val_masks_packed)
        return train_ds, val_ds

    @staticmethod
    def _soft_ce(logits: torch.Tensor, soft_target: torch.Tensor) -> torch.Tensor:
        return -(soft_target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

    def _value_loss(self, outputs: torch.Tensor, targets: torch.Tensor,
                    criterion_mse: nn.MSELoss) -> torch.Tensor:
        """MSE on tanh-scalar (B,1) targets, soft-CE on WDL (B,3) probability targets."""
        if self.value_head == "wdl":
            return self._soft_ce(outputs, targets)
        return criterion_mse(outputs, targets)

    @torch.no_grad()
    def evaluate(self, val_loader, criterion_mse):
        self.model.eval()
        running_loss = 0.0
        running_mse_loss = 0.0
        running_ce_loss = 0.0
        correct = 0
        total = 1
        for data, labels_value, labels_ce in val_loader:
            data = data.to(self.device)
            labels_value = labels_value.to(self.device)
            labels_ce = labels_ce.to(self.device)
            outputs_value, outputs_ce = self.model(data)
            loss_value = self._value_loss(outputs_value, labels_value, criterion_mse)
            loss_ce = self._soft_ce(outputs_ce, labels_ce)
            running_loss += (loss_value + loss_ce).item()
            running_mse_loss += loss_value.item()
            running_ce_loss += loss_ce.item()
            predicted = outputs_ce.argmax(dim=1)
            target = labels_ce.argmax(dim=1)
            total += labels_ce.size(0)
            correct += (predicted == target).sum().item()
        n = len(val_loader)
        return {
            "loss": running_loss / n,
            "mse": running_mse_loss / n,
            "ce": running_ce_loss / n,
            "acc": 100.0 * correct / total,
        }

    def train(self, train_dataset, val_dataset) -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        use_wandb = wandb is not None and self.wandb_project is not None
        if use_wandb:
            wandb.init(
                project=self.wandb_project,
                group=self.wandb_group,
                name=self.wandb_name,
                job_type="train",
                config={
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "lr": self.lr,
                    "l2_weight": self.l2_weight,
                    "label_smoothing": self.label_smoothing,
                    "data_mix": self.data_mix,
                    "resume": self.args.get("resume"),
                },
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
        )
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False,
            )

        criterion_mse = nn.MSELoss()

        val_combined_history = []
        val_ce_history = []
        val_mse_history = []
        latest_val_dict: dict | None = None
        iters = 0

        def run_validation(label: str, epoch_one_idx: int, fraction: float) -> dict | None:
            """Run one val pass, log to wandb, save best-by-metric checkpoints.
            `fraction` is how far through the current epoch we are (1.0 = end)."""
            if val_loader is None:
                return None
            v = self.evaluate(val_loader, criterion_mse)
            val_combined_history.append(v["loss"])
            val_ce_history.append(v["ce"])
            val_mse_history.append(v["mse"])
            if use_wandb:
                wandb.log({
                    "val/loss": v["loss"],
                    "val/value_mse": v["mse"],
                    "val/policy_ce": v["ce"],
                    "val/accuracy": v["acc"],
                    "val/epoch_progress": epoch_one_idx - 1 + fraction,
                    "val/global_step": iters,
                })
            # Snapshot the model under best-by-metric (uses the same payload schema
            # as the end-of-epoch save -- safe because we're in a contiguous train).
            payload = {
                "epoch": epoch_one_idx,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": v["loss"], "val_mse": v["mse"],
                "val_ce": v["ce"], "val_acc": v["acc"],
                "label_smoothing": self.label_smoothing,
                "val_label": label,
            }
            if v["loss"] == min(val_combined_history):
                torch.save(payload, os.path.join(self.checkpoint_dir, "model_best_combined.pth"))
                print(f"  -> [{label}] new best combined val_loss {v['loss']:.4f}")
            if v["ce"] == min(val_ce_history):
                torch.save(payload, os.path.join(self.checkpoint_dir, "model_best_policy.pth"))
                print(f"  -> [{label}] new best val CE {v['ce']:.4f}")
            if v["mse"] == min(val_mse_history):
                torch.save(payload, os.path.join(self.checkpoint_dir, "model_best_value.pth"))
                print(f"  -> [{label}] new best val MSE {v['mse']:.4f}")
            # Restore train mode for the loop to continue.
            self.model.train()
            return v

        for epoch in range(self.epochs):
            lr = self.optimizer.param_groups[0]["lr"]
            self.model.train()
            running_loss = 0.0
            running_mse_loss = 0.0
            running_ce_loss = 0.0
            correct = 0
            total = 1

            n_batches = len(train_loader)
            # Evenly spaced val triggers within the epoch. The Nth trigger lands
            # on the last batch (so end-of-epoch val is implicit). vals_per_epoch=1
            # reproduces the old behaviour exactly.
            vpe = max(1, self.vals_per_epoch)
            val_trigger_batches = set(
                max(1, (i + 1) * n_batches // vpe) - 1 for i in range(vpe)
            )

            for batch_idx, (data, labels_value, labels_ce) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            ):
                data = data.to(self.device)
                labels_value = labels_value.to(self.device)
                labels_ce = labels_ce.to(self.device)
                self.optimizer.zero_grad()

                outputs_value, outputs_ce = self.model(data)
                loss_value = self._value_loss(outputs_value, labels_value, criterion_mse)
                loss_ce = self._soft_ce(outputs_ce, labels_ce)
                loss = loss_value + loss_ce
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                running_mse_loss += loss_value.item()
                running_ce_loss += loss_ce.item()

                predicted = outputs_ce.argmax(dim=1)
                ce_target_label = labels_ce.argmax(dim=1)
                total += labels_ce.size(0)
                correct += (predicted == ce_target_label).sum().item()

                if iters % self.log_step == 0:
                    if use_wandb:
                        wandb.log({
                            "train/loss": loss.item(),
                            "train/value_loss": loss_value.item(),
                            "train/policy_ce": loss_ce.item(),
                            "train/accuracy": 100.0 * correct / total,
                            "train/lr": lr,
                            "train/step": iters,
                        })
                iters += 1

                if batch_idx in val_trigger_batches and val_loader is not None:
                    fraction = (batch_idx + 1) / n_batches
                    label = f"epoch{epoch+1}_val{sorted(val_trigger_batches).index(batch_idx)+1}of{vpe}"
                    val_intra = run_validation(label, epoch + 1, fraction)
                    if val_intra is not None:
                        latest_val_dict = val_intra
                        print(f"  [{label} @ {fraction*100:.0f}%] "
                              f"val: loss {val_intra['loss']:.4f} "
                              f"mse {val_intra['mse']:.4f} "
                              f"ce {val_intra['ce']:.4f} "
                              f"acc {val_intra['acc']:.2f}%")

            train_loss = running_loss / n_batches
            train_mse = running_mse_loss / n_batches
            train_ce = running_ce_loss / n_batches
            train_acc = 100.0 * correct / total

            # The end-of-epoch val was already run by the trigger landing on
            # the final batch; `latest_val_dict` holds its full result.
            val = latest_val_dict if (val_loader is not None and latest_val_dict) else None

            if use_wandb:
                ep_log = {
                    "epoch/train_loss": train_loss,
                    "epoch/train_value_mse": train_mse,
                    "epoch/train_policy_ce": train_ce,
                    "epoch/train_accuracy": train_acc,
                    "epoch/n": epoch + 1,
                }
                if val is not None:
                    ep_log.update({
                        "epoch/val_loss": val["loss"],
                        "epoch/val_value_mse": val["mse"],
                        "epoch/val_policy_ce": val["ce"],
                    })
                wandb.log(ep_log)

            train_part = (f"train: loss {train_loss:.4f} mse {train_mse:.4f} "
                          f"ce {train_ce:.4f} acc {train_acc:.2f}%")
            if val is not None:
                val_part = (f"val: loss {val['loss']:.4f} mse {val['mse']:.4f} "
                            f"ce {val['ce']:.4f} acc {val['acc']:.2f}%")
                log_line = f"Epoch [{epoch + 1}/{self.epochs}], lr: {lr:g} | {train_part} | {val_part}"
            else:
                log_line = f"Epoch [{epoch + 1}/{self.epochs}], lr: {lr:g} | {train_part} | (no val)"
            print(log_line)
            with open(os.path.join(self.log_dir, "log.txt"), "a") as fh:
                fh.write(log_line + "\n")

            payload = {
                "epoch": epoch + 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_loss": train_loss,
                "train_mse": train_mse,
                "train_ce": train_ce,
                "train_acc": train_acc,
                "val_loss": val["loss"] if val else None,
                "val_mse": val["mse"] if val else None,
                "val_ce": val["ce"] if val else None,
                "val_acc": val["acc"] if val else None,
                "label_smoothing": self.label_smoothing,
            }
            last_path = os.path.join(self.checkpoint_dir, "model_last.pth")
            torch.save(payload, last_path)
            # Note: best-by-val-metric checkpoints are saved inside
            # run_validation(), which is called by every trigger (including the
            # end-of-epoch trigger), so no separate save needed here.

        if use_wandb:
            wandb.finish()

        analytics_path = os.path.join(self.checkpoint_dir, "training_analytics.pth")
        torch.save({
            "val_combined_history": val_combined_history,
            "val_ce_history": val_ce_history,
            "val_mse_history": val_mse_history,
        }, analytics_path)
        print(f"Training analytics saved at {analytics_path}")
        print("Training finished!")

    def main(self):
        data = self.data_preparation()
        self.train(data["train"], data["val"])


if __name__ == "__main__":
    cfg = parse_args()
    print("Training config:")
    print(json.dumps(cfg, indent=2))
    Train(cfg).main()
