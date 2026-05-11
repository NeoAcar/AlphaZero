import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # type: ignore

import torch.nn.functional as F
from torch.utils.data import ConcatDataset

import optimized_functions as f
from dataset import ChessDataset, SelfPlayDataset
from resnet import ResNet


def parse_args() -> dict:
    p = argparse.ArgumentParser(description="AlphaZero supervised pre-training on Lichess + Stockfish")
    p.add_argument("--config", default="train_config.json",
                   help="JSON config file. Any --flag below overrides its value.")
    p.add_argument("--games-path", help="path to .pgn file")
    p.add_argument("--evals-path", help="path to .npy stockfish-eval file (per-position)")
    p.add_argument("--max-games", type=int, help="number of games to load from the pgn")
    p.add_argument("--epochs", type=int, help="number of training epochs")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--l2-weight", type=float, help="L2 weight decay")
    p.add_argument("--log-step", type=int, help="log every N iterations")
    p.add_argument("--label-smoothing", type=float,
                   help="cross-entropy label smoothing factor (e.g. 0.1)")
    p.add_argument("--checkpoint-dir", help="directory for model checkpoints")
    p.add_argument("--log-dir", help="directory for tensorboard logs")
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

    # CLI overrides (only when explicitly given).
    overrides = {
        "games_path": cli.games_path,
        "evals_path": cli.evals_path,
        "max_games": cli.max_games,
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
        self.games_path = args["games_path"]
        self.evals_path = args["evals_path"]
        self.max_games = args["max_games"]
        self.label_smoothing = args["label_smoothing"]
        self.checkpoint_dir = args["checkpoint_dir"]
        self.log_dir = args["log_dir"]
        self.val_fraction = args["val_fraction"]
        self.split_seed = args["split_seed"]
        self.self_play_paths = args["self_play_data"] or []
        # Resolve data_mix default
        mix = args.get("data_mix")
        if mix is None:
            mix = "both" if self.self_play_paths else "supervised"
        self.data_mix = mix

        self.model = ResNet().to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.l2_weight)

        if args.get("resume"):
            print(f"Resuming from {args['resume']}")
            ckpt = torch.load(args["resume"], map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

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
            sup_train, sup_val = self._load_supervised_split()
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

    def _load_supervised_split(self) -> tuple[ChessDataset, ChessDataset]:
        games = f.load_pgn(self.games_path, self.max_games)
        boards, moves, _ = f.create_nn_input(games)
        evals = np.load(self.evals_path)
        evals = np.clip(np.asarray(evals, dtype=np.float32), -1.0, 1.0).reshape(-1, 1)
        evals = torch.tensor(evals, dtype=torch.float32)

        positions_per_game = np.array(
            [sum(1 for _ in g.mainline_moves()) for g in games], dtype=np.int64
        )
        cumsum = np.cumsum(positions_per_game)
        n_evals = len(evals)
        games_covered = int(np.searchsorted(cumsum, n_evals, side="right"))
        if games_covered > 0 and cumsum[games_covered - 1] > n_evals:
            games_covered -= 1
        n_effective = int(cumsum[games_covered - 1]) if games_covered > 0 else 0
        if n_effective == 0:
            raise RuntimeError("Eval file is too short to cover even one game.")
        if n_effective < len(boards):
            print(f"Eval file covers {games_covered} games ({n_effective} positions); "
                  f"dropping {len(games) - games_covered} games / "
                  f"{len(boards) - n_effective} positions from the tail.")
        boards = boards[:n_effective]
        moves = moves[:n_effective]
        evals = evals[:n_effective]
        positions_per_game = positions_per_game[:games_covered]

        rng = np.random.RandomState(self.split_seed)
        shuffled = rng.permutation(games_covered)
        n_val_games = max(1, int(round(games_covered * self.val_fraction)))
        val_game_idx = np.zeros(games_covered, dtype=bool)
        val_game_idx[shuffled[:n_val_games]] = True

        per_pos_is_val = np.repeat(val_game_idx, positions_per_game)
        val_mask = torch.from_numpy(per_pos_is_val)
        train_mask = ~val_mask

        n_train_games = games_covered - n_val_games
        print(f"Supervised: {n_train_games} train games / {n_val_games} val games "
              f"({int(train_mask.sum())} / {int(val_mask.sum())} positions).")

        train_ds = ChessDataset(boards[train_mask], evals[train_mask], moves[train_mask],
                                label_smoothing=self.label_smoothing)
        val_ds = ChessDataset(boards[val_mask], evals[val_mask], moves[val_mask],
                              label_smoothing=self.label_smoothing)
        return train_ds, val_ds

    @staticmethod
    def _soft_ce(logits: torch.Tensor, soft_target: torch.Tensor) -> torch.Tensor:
        return -(soft_target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

    @torch.no_grad()
    def evaluate(self, val_loader, criterion_mse):
        self.model.eval()
        running_loss = 0.0
        running_mse_loss = 0.0
        running_ce_loss = 0.0
        correct = 0
        total = 1
        for data, labels_mse, labels_ce in val_loader:
            data = data.to(self.device)
            labels_mse = labels_mse.to(self.device)
            labels_ce = labels_ce.to(self.device)
            outputs_mse, outputs_ce = self.model(data)
            loss_mse = criterion_mse(outputs_mse, labels_mse)
            loss_ce = self._soft_ce(outputs_ce, labels_ce)
            running_loss += (loss_mse + loss_ce).item()
            running_mse_loss += loss_mse.item()
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
        writer = SummaryWriter(log_dir=self.log_dir, flush_secs=1)

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
        iters = 0
        for epoch in range(self.epochs):
            lr = self.optimizer.param_groups[0]["lr"]
            self.model.train()
            running_loss = 0.0
            running_mse_loss = 0.0
            running_ce_loss = 0.0
            correct = 0
            total = 1

            for data, labels_mse, labels_ce in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}"):
                data = data.to(self.device)
                labels_mse = labels_mse.to(self.device)
                labels_ce = labels_ce.to(self.device)
                self.optimizer.zero_grad()

                outputs_mse, outputs_ce = self.model(data)
                loss_mse = criterion_mse(outputs_mse, labels_mse)
                loss_ce = self._soft_ce(outputs_ce, labels_ce)
                loss = loss_mse + loss_ce
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                running_mse_loss += loss_mse.item()
                running_ce_loss += loss_ce.item()

                predicted = outputs_ce.argmax(dim=1)
                ce_target_label = labels_ce.argmax(dim=1)
                total += labels_ce.size(0)
                correct += (predicted == ce_target_label).sum().item()

                if iters % self.log_step == 0:
                    writer.add_scalar("Loss/iter", loss.item(), iters)
                    writer.add_scalar("MSE/iter", loss_mse.item(), iters)
                    writer.add_scalar("CE/iter", loss_ce.item(), iters)
                    writer.add_scalar("Accuracy/iter", 100.0 * correct / total, iters)
                    writer.add_scalar("LearningRate", lr, iters)
                iters += 1

            n_batches = len(train_loader)
            train_loss = running_loss / n_batches
            train_mse = running_mse_loss / n_batches
            train_ce = running_ce_loss / n_batches
            train_acc = 100.0 * correct / total

            val = self.evaluate(val_loader, criterion_mse) if val_loader is not None else None

            writer.add_scalar("Loss/train_epoch", train_loss, epoch + 1)
            writer.add_scalar("MSE/train_epoch", train_mse, epoch + 1)
            writer.add_scalar("CE/train_epoch", train_ce, epoch + 1)
            writer.add_scalar("Accuracy/train_epoch", train_acc, epoch + 1)
            if val is not None:
                val_combined_history.append(val["loss"])
                val_ce_history.append(val["ce"])
                val_mse_history.append(val["mse"])
                writer.add_scalar("Loss/val_epoch", val["loss"], epoch + 1)
                writer.add_scalar("MSE/val_epoch", val["mse"], epoch + 1)
                writer.add_scalar("CE/val_epoch", val["ce"], epoch + 1)
                writer.add_scalar("Accuracy/val_epoch", val["acc"], epoch + 1)

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

            if val is not None:
                if val_combined_history[-1] == min(val_combined_history):
                    best_path = os.path.join(self.checkpoint_dir, "model_best_combined.pth")
                    torch.save(payload, best_path)
                    print(f"  -> new best combined val_loss, saved {best_path}")
                if val_ce_history[-1] == min(val_ce_history):
                    best_path = os.path.join(self.checkpoint_dir, "model_best_policy.pth")
                    torch.save(payload, best_path)
                    print(f"  -> new best val CE, saved {best_path}")
                if val_mse_history[-1] == min(val_mse_history):
                    best_path = os.path.join(self.checkpoint_dir, "model_best_value.pth")
                    torch.save(payload, best_path)
                    print(f"  -> new best val MSE, saved {best_path}")

        writer.flush()
        writer.close()

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
