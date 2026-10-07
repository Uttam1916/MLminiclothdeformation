"""
Model Training and Optimization Module.

Faithfully implements the training procedure described in Xue & Wu (2021):
- Optimizer: Adam (lr = 1e-4)
- Loss: Mean Squared Error (MSE) on reconstructed vertex offsets
- Batch size: 32
- Early stopping & model checkpointing based on validation loss
"""

import os
import time
import json
import torch
import torch.nn as nn
from typing import Dict, List, Optional


def get_device() -> torch.device:
    """Returns CUDA device if working kernel exists, otherwise CPU."""
    if torch.cuda.is_available():
        try:
            a = torch.randn(2, 2, device="cuda")
            b = torch.randn(2, 2, device="cuda")
            _ = torch.matmul(a, b)
            torch.cuda.synchronize()
            return torch.device("cuda")
        except Exception as e:
            print(f"CUDA device detected but compute kernels incompatible ({e}). Falling back to CPU.")
            return torch.device("cpu")
    return torch.device("cpu")


class Trainer:
    """
    Manages model training, evaluation on validation set, checkpointing, and metric logging.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "checkpoints",
        model_name: str = "model",
    ):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=weight_decay,
        )

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "epoch_times": [],
        }
        self.best_val_loss = float("inf")
        self.best_epoch = 0

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for poses, offsets, _ in self.train_loader:
            poses = poses.to(self.device)
            offsets = offsets.to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(poses)
            loss = self.criterion(preds, offsets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def evaluate_val(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for poses, offsets, _ in self.val_loader:
                poses = poses.to(self.device)
                offsets = offsets.to(self.device)

                preds = self.model(poses)
                loss = self.criterion(preds, offsets)

                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def fit(
        self,
        epochs: int = 1000,
        patience: int = 200,
        log_interval: int = 100,
    ) -> Dict[str, List[float]]:
        start_time = time.time()
        print(f"[{self.model_name}] Starting training on {self.device} for {epochs} epochs...")

        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch()
            val_loss = self.evaluate_val()
            epoch_time = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["epoch_times"].append(epoch_time)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                epochs_without_improvement = 0
                self.save_checkpoint(is_best=True)
            else:
                epochs_without_improvement += 1

            if epoch % log_interval == 0 or epoch == 1 or epoch == epochs:
                print(
                    f"[{self.model_name}] Epoch {epoch:4d}/{epochs} | "
                    f"Train Loss: {train_loss:.6e} | Val Loss: {val_loss:.6e} | "
                    f"Best Val: {self.best_val_loss:.6e} (ep {self.best_epoch}) | "
                    f"Time: {epoch_time:.3f}s"
                )

            if epochs_without_improvement >= patience and epoch >= 300:
                print(f"[{self.model_name}] Early stopping triggered at epoch {epoch} (patience={patience})")
                break

        total_time = time.time() - start_time
        print(
            f"[{self.model_name}] Training finished in {total_time:.2f}s. "
            f"Best Val Loss: {self.best_val_loss:.6e} at Epoch {self.best_epoch}"
        )

        # Save training history
        history_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_history.json")
        with open(history_path, "w") as f:
            json.dump(
                {
                    "train_loss": self.history["train_loss"],
                    "val_loss": self.history["val_loss"],
                    "best_val_loss": self.best_val_loss,
                    "best_epoch": self.best_epoch,
                    "total_train_time_sec": total_time,
                },
                f,
                indent=2,
            )

        # Load best model weights
        self.load_checkpoint(os.path.join(self.checkpoint_dir, f"{self.model_name}_best.pt"))
        return self.history

    def save_checkpoint(self, is_best: bool = True):
        filename = f"{self.model_name}_best.pt" if is_best else f"{self.model_name}_latest.pt"
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "epoch": self.best_epoch,
                "model_state_dict": self.model.state_dict(),
                "best_val_loss": self.best_val_loss,
                "history": self.history,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
