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

import optimized_functions as f
from dataset import ChessDataset
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

        self.model = ResNet().to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.l2_weight)

        if args.get("resume"):
            print(f"Resuming from {args['resume']}")
            ckpt = torch.load(args["resume"], map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    def data_preparation(self):
        """Load PGN + evals, split by game into train/val, return six tensors."""
        games = f.load_pgn(self.games_path, self.max_games)
        boards, moves, results = f.create_nn_input(games)
        evals = np.load(self.evals_path)
        evals = np.clip(np.asarray(evals, dtype=np.float32), -1.0, 1.0).reshape(-1, 1)
        evals = torch.tensor(evals, dtype=torch.float32)

        # Count positions per loaded game so we can split by game (not by
        # position -- adjacent plies within a game are highly correlated and
        # leak information across a position-level split).
        positions_per_game = np.array(
            [sum(1 for _ in g.mainline_moves()) for g in games], dtype=np.int64
        )
        cumsum = np.cumsum(positions_per_game)

        # The eval file may cover fewer positions than the loaded games.
        # Drop any partial-game tail to keep game-level alignment intact.
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

        # Pick val games (seeded random subset of game indices).
        rng = np.random.RandomState(self.split_seed)
        shuffled = rng.permutation(games_covered)
        n_val_games = max(1, int(round(games_covered * self.val_fraction)))
        val_game_idx = np.zeros(games_covered, dtype=bool)
        val_game_idx[shuffled[:n_val_games]] = True

        # Expand the per-game mask to a per-position boolean mask.
        per_pos_is_val = np.repeat(val_game_idx, positions_per_game)
        assert len(per_pos_is_val) == n_effective
        val_mask = torch.from_numpy(per_pos_is_val)
        train_mask = ~val_mask

        n_train_games = games_covered - n_val_games
        n_train_pos = int(train_mask.sum())
        n_val_pos = int(val_mask.sum())
        print(f"Train: {n_train_games} games, {n_train_pos} positions. "
              f"Val: {n_val_games} games, {n_val_pos} positions.")

        return {
            "train": (boards[train_mask], evals[train_mask], moves[train_mask]),
            "val": (boards[val_mask], evals[val_mask], moves[val_mask]),
        }

    @torch.no_grad()
    def evaluate(self, val_loader, criterion_mse, criterion_ce):
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
            loss_ce = criterion_ce(outputs_ce, labels_ce)
            running_loss += (loss_mse + loss_ce).item()
            running_mse_loss += loss_mse.item()
            running_ce_loss += loss_ce.item()
            _, predicted = torch.max(outputs_ce.data, 1)
            total += labels_ce.size(0)
            correct += (predicted == labels_ce).sum().item()
        n = len(val_loader)
        return {
            "loss": running_loss / n,
            "mse": running_mse_loss / n,
            "ce": running_ce_loss / n,
            "acc": 100.0 * correct / total,
        }

    def train(self, train_data, val_data) -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=self.log_dir, flush_secs=1)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr

        train_boards, train_evals, train_moves = train_data
        val_boards, val_evals, val_moves = val_data
        train_loader = DataLoader(
            ChessDataset(train_boards, train_evals, train_moves),
            batch_size=self.batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            ChessDataset(val_boards, val_evals, val_moves),
            batch_size=self.batch_size, shuffle=False,
        )

        criterion_mse = nn.MSELoss()
        criterion_ce = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

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
                loss_ce = criterion_ce(outputs_ce, labels_ce)
                loss = loss_mse + loss_ce
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                running_mse_loss += loss_mse.item()
                running_ce_loss += loss_ce.item()

                _, predicted = torch.max(outputs_ce.data, 1)
                total += labels_ce.size(0)
                correct += (predicted == labels_ce).sum().item()

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

            val = self.evaluate(val_loader, criterion_mse, criterion_ce)
            val_combined_history.append(val["loss"])
            val_ce_history.append(val["ce"])
            val_mse_history.append(val["mse"])

            writer.add_scalar("Loss/train_epoch", train_loss, epoch + 1)
            writer.add_scalar("MSE/train_epoch", train_mse, epoch + 1)
            writer.add_scalar("CE/train_epoch", train_ce, epoch + 1)
            writer.add_scalar("Accuracy/train_epoch", train_acc, epoch + 1)
            writer.add_scalar("Loss/val_epoch", val["loss"], epoch + 1)
            writer.add_scalar("MSE/val_epoch", val["mse"], epoch + 1)
            writer.add_scalar("CE/val_epoch", val["ce"], epoch + 1)
            writer.add_scalar("Accuracy/val_epoch", val["acc"], epoch + 1)

            log_line = (
                f"Epoch [{epoch + 1}/{self.epochs}], lr: {lr:g} | "
                f"train: loss {train_loss:.4f} mse {train_mse:.4f} ce {train_ce:.4f} acc {train_acc:.2f}% | "
                f"val: loss {val['loss']:.4f} mse {val['mse']:.4f} ce {val['ce']:.4f} acc {val['acc']:.2f}%"
            )
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
                "val_loss": val["loss"],
                "val_mse": val["mse"],
                "val_ce": val["ce"],
                "val_acc": val["acc"],
                "label_smoothing": self.label_smoothing,
            }
            last_path = os.path.join(self.checkpoint_dir, "model_last.pth")
            torch.save(payload, last_path)

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
        split = self.data_preparation()
        self.train(split["train"], split["val"])


if __name__ == "__main__":
    cfg = parse_args()
    print("Training config:")
    print(json.dumps(cfg, indent=2))
    Train(cfg).main()
