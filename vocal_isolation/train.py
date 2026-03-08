import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from .config import Config
from .model import UNet


class CombinedLoss(nn.Module):
    """L1 + spectral convergence loss."""

    def __init__(self, alpha: float = 1.0, beta: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1 = nn.L1Loss()

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1_loss = self.l1(predicted, target)

        # Spectral convergence: Frobenius norm ratio
        sc_loss = torch.norm(target - predicted, p="fro") / (torch.norm(target, p="fro") + 1e-8)

        return self.alpha * l1_loss + self.beta * sc_loss


class Trainer:
    def __init__(self, model: UNet, config: Config):
        self.config = config
        self.device = config.train.resolve_device()
        self.model = model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.train.lr,
            weight_decay=config.train.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5,
        )
        self.criterion = CombinedLoss()
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.checkpoint_dir = Path(config.train.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=config.train.log_dir)
        except ImportError:
            self.writer = None

    def _train_epoch(self, loader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"Train Epoch {epoch + 1}", leave=False)
        for batch in pbar:
            mix_mag = batch["mix_mag"].to(self.device)
            vocals_mag = batch["vocals_mag"].to(self.device)

            mask = self.model(mix_mag)
            predicted_vocals = mask * mix_mag
            loss = self.criterion(predicted_vocals, vocals_mag)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.train.grad_clip
            )
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(loader)

    @torch.no_grad()
    def _validate_epoch(self, loader, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"Valid Epoch {epoch + 1}", leave=False)
        for batch in pbar:
            mix_mag = batch["mix_mag"].to(self.device)
            vocals_mag = batch["vocals_mag"].to(self.device)

            mask = self.model(mix_mag)
            predicted_vocals = mask * mix_mag
            loss = self.criterion(predicted_vocals, vocals_mag)

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(loader)

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }
        path = self.checkpoint_dir / "last.pt"
        torch.save(state, path)
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(state, best_path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt["epoch"]

    def fit(self, train_loader, valid_loader) -> dict:
        history = {"train_loss": [], "valid_loss": []}

        for epoch in range(self.config.train.epochs):
            train_loss = self._train_epoch(train_loader, epoch)
            val_loss = self._validate_epoch(valid_loader, epoch)

            self.scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["valid_loss"].append(val_loss)

            if self.writer:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/valid", val_loss, epoch)
                self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self._save_checkpoint(epoch, val_loss, is_best)

            print(
                f"Epoch {epoch + 1}/{self.config.train.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Best: {self.best_val_loss:.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )

            if self.patience_counter >= self.config.train.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if self.writer:
            self.writer.close()
        return history
