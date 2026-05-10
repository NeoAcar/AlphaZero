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
    cli = p.parse_args()

    with open(cli.config) as fp:
        cfg = json.load(fp)

    # Defaults for keys the original config didn't have.
    cfg.setdefault("label_smoothing", 0.0)
    cfg.setdefault("checkpoint_dir", "./checkpoints")
    cfg.setdefault("log_dir", "./logs")
    cfg.setdefault("resume", None)

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

        self.model = ResNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2_weight)

        if args.get("resume"):
            print(f"Resuming from {args['resume']}")
            ckpt = torch.load(args["resume"], map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    def data_preparation(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        games = f.load_pgn(self.games_path, self.max_games)
        boards, moves, results = f.create_nn_input(games)
        evals = np.load(self.evals_path)
        # Clip mate-encoded outliers (just past +/-1) to the value head's
        # tanh range so MSE has zero attainable error on mate positions.
        evals = np.clip(np.asarray(evals, dtype=np.float32), -1.0, 1.0).reshape(-1, 1)
        evals = torch.tensor(evals, dtype=torch.float32)
        # The eval file may be longer than `boards` (per-position evals cover
        # more games than max_games). Truncate to alignment.
        evals = evals[: len(boards)]
        return boards, evals, moves

    def train(self, boards: torch.Tensor, evals: torch.Tensor, moves: torch.Tensor) -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=self.log_dir, flush_secs=1)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr

        train_dataset = ChessDataset(boards, evals, moves)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        criterion_mse = nn.MSELoss()
        criterion_ce = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        train_loss_history = []
        iters = 0
        best_epoch = 0  # last epoch where loss improved
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
            epoch_loss = running_loss / n_batches
            epoch_mse = running_mse_loss / n_batches
            epoch_ce = running_ce_loss / n_batches
            epoch_accuracy = 100.0 * correct / total
            train_loss_history.append(epoch_loss)

            writer.add_scalar("Loss/epoch", epoch_loss, epoch + 1)
            writer.add_scalar("MSE/epoch", epoch_mse, epoch + 1)
            writer.add_scalar("CE/epoch", epoch_ce, epoch + 1)
            writer.add_scalar("Accuracy/epoch", epoch_accuracy, epoch + 1)

            log_line = (
                f"Epoch [{epoch + 1}/{self.epochs}], "
                f"Loss: {epoch_loss:.4f}, lr: {lr:g}, "
                f"MSE: {epoch_mse:.4f}, CE: {epoch_ce:.4f}, Acc: {epoch_accuracy:.2f}%"
            )
            print(log_line)
            with open(os.path.join(self.log_dir, "log.txt"), "a") as fh:
                fh.write(log_line + "\n")

            if train_loss_history[-1] == min(train_loss_history):
                # New best: save checkpoint, retire older ones.
                for old in os.listdir(self.checkpoint_dir):
                    if old.startswith("model_epoch_") and old.endswith(".pth"):
                        os.remove(os.path.join(self.checkpoint_dir, old))
                best_epoch = epoch + 1
                ckpt_path = os.path.join(self.checkpoint_dir, f"model_epoch_{best_epoch}.pth")
                torch.save({
                    "epoch": best_epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": epoch_loss,
                    "mse_loss": epoch_mse,
                    "ce_loss": epoch_ce,
                    "accuracy": epoch_accuracy,
                    "label_smoothing": self.label_smoothing,
                }, ckpt_path)
                print(f"  -> new best, saved {ckpt_path}")
            else:
                # Reload best, decay LR by 0.3.
                if best_epoch:
                    ckpt_path = os.path.join(self.checkpoint_dir, f"model_epoch_{best_epoch}.pth")
                    ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                    self.model.load_state_dict(ckpt["model_state_dict"])
                    self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                new_lr = lr * 0.3
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = new_lr
                print(f"  -> no improvement, lr {lr:g} -> {new_lr:g}, reloaded best")

        writer.flush()
        writer.close()

        analytics_path = os.path.join(self.checkpoint_dir, "training_analytics.pth")
        torch.save({"train_loss_history": train_loss_history}, analytics_path)
        print(f"Training analytics saved at {analytics_path}")
        print("Training finished!")

    def main(self):
        boards, evals, moves = self.data_preparation()
        print(f"Prepared {len(boards)} positions for training.")
        self.train(boards, evals, moves)


if __name__ == "__main__":
    cfg = parse_args()
    print("Training config:")
    print(json.dumps(cfg, indent=2))
    Train(cfg).main()
