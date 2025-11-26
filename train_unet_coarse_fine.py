# ============================================================
# train_unet_coarse_fine.py  (Final Version)
# ============================================================
import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from data_loader import get_dataloaders
from Unet import UNet3D


# ============================================================
# 1. Weighted MSE Loss
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


# ============================================================
# 2. SoftDice_abs Loss (no sigmoid)
# ============================================================
class SoftDiceAbsLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.eps = epsilon

    def forward(self, pred, target):
        pred_abs = pred.abs()
        target_abs = target.abs()
        num = 2 * (pred_abs * target_abs).sum()
        den = pred_abs.sum() + target_abs.sum() + self.eps
        dice = 1 - num / den
        return dice


# ============================================================
# 3. Evaluation metrics
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
            'Loss': squared_error.mean().item(),
            'MAE': mae,
            'RMSE': rmse,
            'Change_MAE': change_mae,
            'Positive_MAE': pos_mae,
            'Negative_MAE': neg_mae,
            'Abs_MAE': abs_mae
        }


# ============================================================
# 4. Utilities
# ============================================================
def initialize_weights(model, init_type='kaiming'):
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            if init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_metrics_csv(filepath, header, rows):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def pretty_epoch_log(stage, epoch, tr, va):
    return (
        f"[{stage}] Epoch {epoch:03d} | "
        f"Train: Loss {tr['Loss']:.4f}  MAE {tr['MAE']:.4f}  RMSE {tr['RMSE']:.4f}  cMAE {tr['Change_MAE']:.4f} | "
        f"Val:   Loss {va['Loss']:.4f}  MAE {va['MAE']:.4f}  RMSE {va['RMSE']:.4f}  cMAE {va['Change_MAE']:.4f}"
    )


# ============================================================
# 5. Core training function
# ============================================================
def train_stage(stage, model, train_loader, val_loader, optimizer, loss_fn, device,
                num_epochs, csv_path, model_path_prefix, epsilon=1e-4, patience=10):
    all_rows = []
    header = [
        "Epoch",
        "Train_Loss", "Train_MAE", "Train_RMSE", "Train_Change_MAE",
        "Val_Loss", "Val_MAE", "Val_RMSE", "Val_Change_MAE",
        "Val_Positive_MAE", "Val_Negative_MAE", "Val_Abs_MAE"
    ]

    best = {"Change_MAE": float("inf"), "MAE": float("inf"), "RMSE": float("inf")}
    best_epoch = -1
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        tr_metrics = {k: [] for k in ['Loss', 'MAE', 'RMSE', 'Change_MAE']}

        for inputs, targets in tqdm(train_loader, desc=f"[{stage}] Epoch {epoch:03d} [Train]", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            m = evaluate_metrics(preds, targets, epsilon)
            for k in tr_metrics:
                tr_metrics[k].append(m[k])

        train_log = {k: float(np.mean(v)) for k, v in tr_metrics.items()}

        # Validation
        model.eval()
        val_metrics = {k: [] for k in
                       ['Loss', 'MAE', 'RMSE', 'Change_MAE', 'Positive_MAE', 'Negative_MAE', 'Abs_MAE']}
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"[{stage}] Epoch {epoch:03d} [Valid]", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                preds = model(inputs)
                m = evaluate_metrics(preds, targets, epsilon)
                for k in val_metrics:
                    val_metrics[k].append(m[k])

        val_log = {k: float(np.mean(v)) for k, v in val_metrics.items()}

        tqdm.write(pretty_epoch_log(stage, epoch, train_log, val_log))

        all_rows.append([
            epoch,
            train_log['Loss'], train_log['MAE'], train_log['RMSE'], train_log['Change_MAE'],
            val_log['Loss'], val_log['MAE'], val_log['RMSE'], val_log['Change_MAE'],
            val_log['Positive_MAE'], val_log['Negative_MAE'], val_log['Abs_MAE']
        ])

        # Early stopping on Val_Change_MAE
        if val_log['Change_MAE'] < best["Change_MAE"]:
            best = {"Change_MAE": val_log['Change_MAE'],
                    "MAE": val_log['MAE'],
                    "RMSE": val_log['RMSE']}
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), model_path_prefix + "_best_Val_ChangeMAE.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                tqdm.write(f"[{stage}] Early stopping at epoch {epoch:03d}")
                break

    save_metrics_csv(csv_path, header, all_rows)
    return best, best_epoch


# ============================================================
# 6. Coarse-to-Fine training loop
# ============================================================
def main():
    set_seed(2025)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tqdm.write(f"Device: {device}")

    # Load data
    train_loader, val_loader, _ = get_dataloaders(
        data_root="all_data",
        batch_size_train=8,
        batch_size_val=2,
        augment_snr50=True,
        snr50_ratio=0.2,
    )

    # Folders (same directory)
    base_dir = "./corse_fine_Unet"
    metric_dir = os.path.join(base_dir, "metrics")
    model_dir = os.path.join(base_dir, "models")
    os.makedirs(metric_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    lr = 1e-3
    num_epochs = 100
    patience = 10

    coarse_combos = [(2, 10), (2, 20), (5, 10), (5, 20), (10, 10), (10, 20)]
    fine_gamma = 20

    coarse_summary = []
    fine_summary = []

    # ---------------------- Coarse Stage ----------------------
    tqdm.write("=== Coarse Stage ===")
    for gamma, alpha in coarse_combos:
        model = UNet3D(in_channels=2, out_channels=1)
        initialize_weights(model, init_type='kaiming')
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        wmse = WeightedMSELoss(gamma=gamma)
        dice = SoftDiceAbsLoss()
        loss_fn = lambda p, t: wmse(p, t) + alpha * dice(p, t)

        csv_path = os.path.join(metric_dir, f"coarse_gamma{gamma}_alpha{alpha}.csv")
        model_path_prefix = os.path.join(model_dir, f"coarse_gamma{gamma}_alpha{alpha}")

        tqdm.write(f"Training Coarse combo: γ={gamma}, α={alpha}")
        best, best_epoch = train_stage(
            stage=f"Coarse(γ={gamma},α={alpha})",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            num_epochs=num_epochs,
            csv_path=csv_path,
            model_path_prefix=model_path_prefix,
            patience=patience
        )
        coarse_summary.append((gamma, alpha, best_epoch, best))

    # ---------------------- Fine Stage ----------------------
    tqdm.write("\n=== Fine Stage ===")
    for gamma, alpha, _, _ in coarse_summary:
        model = UNet3D(in_channels=2, out_channels=1).to(device)
        ckpt = os.path.join(model_dir, f"coarse_gamma{gamma}_alpha{alpha}_best_Val_ChangeMAE.pt")
        model.load_state_dict(torch.load(ckpt, map_location=device))

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = WeightedMSELoss(gamma=fine_gamma)

        csv_path = os.path.join(metric_dir, f"fine_from_coarse_gamma{gamma}_alpha{alpha}_gamma{fine_gamma}.csv")
        model_path_prefix = os.path.join(model_dir, f"fine_from_coarse_gamma{gamma}_alpha{alpha}_gamma{fine_gamma}")

        tqdm.write(f"Fine-tuning from coarse(γ={gamma},α={alpha}) with γ={fine_gamma}")
        best, best_epoch = train_stage(
            stage=f"Fine(from γ={gamma},α={alpha})",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            num_epochs=num_epochs,
            csv_path=csv_path,
            model_path_prefix=model_path_prefix,
            patience=patience
        )
        fine_summary.append((gamma, alpha, best_epoch, best))

    # ============================================================
    # 7. Print final best results
    # ============================================================
    line = "=" * 85
    print("\n" + line)
    print(" FINAL VALIDATION RESULTS (Best Epoch per Model) ".center(85))
    print(line)
    print(f"{'Stage':<10}{'γ':>4}{'α':>6}{'Epoch':>8}{'Val_cMAE':>14}{'Val_MAE':>14}{'Val_RMSE':>14}")
    print("-" * 85)

    for gamma, alpha, epoch, best in coarse_summary:
        print(f"{'Coarse':<10}{gamma:>4}{alpha:>6}{epoch:>8}{best['Change_MAE']:>14.6f}{best['MAE']:>14.6f}{best['RMSE']:>14.6f}")

    for gamma, alpha, epoch, best in fine_summary:
        print(f"{'Fine':<10}{gamma:>4}{alpha:>6}{epoch:>8}{best['Change_MAE']:>14.6f}{best['MAE']:>14.6f}{best['RMSE']:>14.6f}")

    print(line)
    print("All models and metrics saved under: ./corse_fine_Unet/")
    print(line + "\n")


if __name__ == "__main__":
    main()
