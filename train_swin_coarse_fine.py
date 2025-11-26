# ============================================================
# train_swin_unetr_coarse_fine.py
# ============================================================
import os
import csv
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from monai.networks.nets import SwinUNETR
from data_loader import get_dataloaders


# ---------------------- Fixed Seed ----------------------
torch.manual_seed(2025)
np.random.seed(2025)


# ============================================================
# Loss functions
# ============================================================
class WeightedMSELoss(nn.Module):
    def __init__(self, epsilon=1e-4, gamma=20.0):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = gamma

    def forward(self, pred, target):
        pred = torch.clamp(pred, -1.0, 1.0)
        weight = 1.0 + self.gamma * (target.abs() > self.epsilon).float()
        return (weight * (pred - target) ** 2).mean()


class SoftDiceAbsLoss(nn.Module):
    """Dice based on |p| and |g| (no sigmoid)."""
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.eps = epsilon

    def forward(self, pred, target):
        pred_abs = pred.abs()
        target_abs = target.abs()
        num = 2 * (pred_abs * target_abs).sum()
        den = pred_abs.sum() + target_abs.sum() + self.eps
        return 1 - num / den


# ============================================================
# Evaluation metrics (same as UNet version)
# ============================================================
def evaluate_metrics(pred, target, epsilon=1e-4):
    with torch.no_grad():
        pred = torch.clamp(pred, -1.0, 1.0)
        abs_error = (pred - target).abs()
        squared_error = (pred - target) ** 2

        mae = abs_error.mean().item()
        rmse = torch.sqrt(squared_error.mean()).item()

        change_mask = (target.abs() > epsilon)
        change_mae = abs_error[change_mask].mean().item() if change_mask.any() else 0.0

        pos_mask = (target > epsilon)
        pos_mae = abs_error[pos_mask].mean().item() if pos_mask.any() else 0.0

        neg_mask = (target < -epsilon)
        neg_mae = abs_error[neg_mask].mean().item() if neg_mask.any() else 0.0

        abs_mae = (pred.abs() - target.abs()).abs().mean().item()

        return {
            "Loss": squared_error.mean().item(),
            "MAE": mae,
            "RMSE": rmse,
            "Change_MAE": change_mae,
            "Positive_MAE": pos_mae,
            "Negative_MAE": neg_mae,
            "Abs_MAE": abs_mae
        }


# ============================================================
# SwinUNETR (tanh output)
# ============================================================
class SwinUNETR_Tanh(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, feature_size=24):
        super().__init__()
        self.model = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=False
        )
        self.activation = nn.Tanh()

    def forward(self, x):
        return self.activation(self.model(x))


# ============================================================
# Training one stage (coarse or fine)
# ============================================================
def train_stage(stage, model, train_loader, val_loader,
                optimizer, loss_fn, csv_path, model_prefix,
                num_epochs=100, patience=10, device="cuda"):

    header = [
        "Epoch",
        "Train_Loss", "Train_MAE", "Train_RMSE", "Train_Change_MAE",
        "Val_Loss", "Val_MAE", "Val_RMSE", "Val_Change_MAE",
        "Val_Positive_MAE", "Val_Negative_MAE", "Val_Abs_MAE"
    ]

    rows = []
    best_change_mae = float("inf")
    best_epoch = -1
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        tqdm.write(f"[{stage}] Epoch {epoch:03d}")

        # -------- Training --------
        model.train()
        train_logs = []
        for inputs, targets in tqdm(train_loader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)

            loss = loss_fn(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_logs.append(evaluate_metrics(preds, targets))

        train_avg = {k: np.mean([x[k] for x in train_logs]) for k in train_logs[0]}

        # -------- Validation --------
        model.eval()
        val_logs = []
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                preds = model(inputs)
                val_logs.append(evaluate_metrics(preds, targets))

        val_avg = {k: np.mean([x[k] for x in val_logs]) for k in val_logs[0]}

        # Save logs
        row = [
            epoch,
            train_avg["Loss"], train_avg["MAE"], train_avg["RMSE"], train_avg["Change_MAE"],
            val_avg["Loss"], val_avg["MAE"], val_avg["RMSE"], val_avg["Change_MAE"],
            val_avg["Positive_MAE"], val_avg["Negative_MAE"], val_avg["Abs_MAE"],
        ]
        rows.append(row)

        tqdm.write(
            f"Train Loss={train_avg['Loss']:.4f}, "
            f"Val Loss={val_avg['Loss']:.4f}, Val cMAE={val_avg['Change_MAE']:.4f}"
        )

        # Early stopping  (monitor Change_MAE)
        if val_avg["Change_MAE"] < best_change_mae:
            best_change_mae = val_avg["Change_MAE"]
            best_epoch = epoch
            no_improve = 0

            torch.save(model.state_dict(),
                       f"{model_prefix}_best_Val_ChangeMAE.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                tqdm.write(f"Early stopping triggered at epoch {epoch:03d}")
                break

    # Save CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    tqdm.write(f"[{stage}] Best epoch={best_epoch}, Val_ChangeMAE={best_change_mae:.6f}\n")
    return best_change_mae, best_epoch


# ============================================================
# Main pipeline
# ============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tqdm.write(f"Running on device: {device}")

    save_root = "./swin_coarse_fine"
    model_dir = os.path.join(save_root, "models")
    metrics_dir = os.path.join(save_root, "metrics")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    train_loader, val_loader, _ = get_dataloaders(
        data_root="all_data",
        batch_size_train=4,
        batch_size_val=2,
        augment_snr50=True,
        snr50_ratio=0.2,
    )

    # ----- Coarse stage hyperparameter search -----
    coarse_combos = [(2, 10), (2, 20), (5, 10), (5, 20), (10, 10), (10, 20)]
    coarse_results = []

    for gamma, alpha in coarse_combos:
        model = SwinUNETR_Tanh().to(device)
        wmse = WeightedMSELoss(gamma=gamma)
        dice = SoftDiceAbsLoss()
        loss_fn = lambda p, t: wmse(p, t) + alpha * dice(p, t)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        csv_path = f"{metrics_dir}/coarse_gamma{gamma}_alpha{alpha}.csv"
        model_prefix = f"{model_dir}/coarse_gamma{gamma}_alpha{alpha}"

        best_cmae, best_epoch = train_stage(
            stage=f"COARSE γ={gamma}, α={alpha}",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            csv_path=csv_path,
            model_prefix=model_prefix,
            num_epochs=100,
            patience=10,
            device=device
        )

        coarse_results.append((gamma, alpha, best_cmae, best_epoch))

    # ----- Fine stage -----
    fine_gamma = 15

    for gamma, alpha, _, _ in coarse_results:
        model = SwinUNETR_Tanh().to(device)

        ckpt = f"{model_dir}/coarse_gamma{gamma}_alpha{alpha}_best_Val_ChangeMAE.pt"
        model.load_state_dict(torch.load(ckpt, map_location=device))

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = WeightedMSELoss(gamma=fine_gamma)

        csv_path = f"{metrics_dir}/fine_from_coarse_gamma{gamma}_alpha{alpha}_gamma15.csv"
        model_prefix = f"{model_dir}/fine_from_coarse_gamma{gamma}_alpha{alpha}_gamma15"

        train_stage(
            stage=f"FINE from (γ={gamma}, α={alpha}), γ={fine_gamma}",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            csv_path=csv_path,
            model_prefix=model_prefix,
            num_epochs=100,
            patience=10,
            device=device
        )

    tqdm.write("\n✅ All training finished. Files saved under ./swin_coarse_fine/\n")


if __name__ == "__main__":
    main()
