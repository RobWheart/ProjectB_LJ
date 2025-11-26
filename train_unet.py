# ============================================================
# trainUnet/train_unet_standard.py  (Patience adjusted version)
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


class WeightedMSELoss(nn.Module):
    """MSE + 区域加权"""
    def __init__(self, epsilon=1e-4, gamma=20.0):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = gamma

    def forward(self, pred, target):
        pred = torch.clamp(pred, -1.0, 1.0)
        weight = 1.0 + self.gamma * (target.abs() > self.epsilon).float()
        loss = weight * (pred - target) ** 2
        return loss.mean()


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


def save_metrics_csv(filepath, header, rows):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_lr(lr: float) -> str:
    return f"{lr:.0e}"


def pretty_epoch_log(epoch, tr, va):
    return (
        f"Epoch {epoch:03d} | "
        f"Train: Loss {tr['Loss']:.4f}  MAE {tr['MAE']:.4f}  RMSE {tr['RMSE']:.4f}  cMAE {tr['Change_MAE']:.4f} | "
        f"Val:   Loss {va['Loss']:.4f}  MAE {va['MAE']:.4f}  RMSE {va['RMSE']:.4f}  cMAE {va['Change_MAE']:.4f}"
    )


def train_one_combo(lr, gamma, device='cuda', num_epochs=150, epsilon=1e-4):
    set_seed(2025)

    train_loader, val_loader, _ = get_dataloaders(
        data_root="all_data",
        batch_size_train=8,
        batch_size_val=2,
        augment_snr50=True,
        snr50_ratio=0.2,
    )

    model = UNet3D(in_channels=2, out_channels=1)
    initialize_weights(model, init_type='kaiming')
    model = model.to(device)

    criterion = WeightedMSELoss(epsilon=epsilon, gamma=gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    metric_dir = "trainUnet/metrics"
    model_dir = "trainUnet/models"
    os.makedirs(metric_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    all_rows = []
    header = [
        "Epoch",
        "Train_Loss", "Train_MAE", "Train_RMSE", "Train_Change_MAE",
        "Val_Loss", "Val_MAE", "Val_RMSE", "Val_Change_MAE",
        "Val_Positive_MAE", "Val_Negative_MAE", "Val_Abs_MAE"
    ]

    best = {k: {"value": float('inf'), "epoch": -1} for k in ["Val_MAE", "Val_Change_MAE", "Val_RMSE"]}
    lr_tag = format_lr(lr)
    csv_path = os.path.join(metric_dir, f"metrics_lr{lr_tag}_gamma{gamma}.csv")

    best_cmae_for_early_stop = float('inf')
    no_improve_epochs = 0

    for epoch in range(1, num_epochs + 1):
        # 动态 patience：前 100 epoch patience = 10，后 50 epoch patience = 5
        current_patience = 10 if epoch <= 100 else 5

        model.train()
        train_losses, train_maes, train_rmses, train_cmaes = [], [], [], []
        pbar = tqdm(train_loader, desc=f"[lr={lr_tag} γ={gamma}] Epoch {epoch:03d} [Train]", leave=False)

        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            m = evaluate_metrics(preds, targets, epsilon=epsilon)
            train_losses.append(loss.item())
            train_maes.append(m['MAE'])
            train_rmses.append(m['RMSE'])
            train_cmaes.append(m['Change_MAE'])

        train_log = {
            "Loss": float(np.mean(train_losses)),
            "MAE": float(np.mean(train_maes)),
            "RMSE": float(np.mean(train_rmses)),
            "Change_MAE": float(np.mean(train_cmaes)),
        }

        model.eval()
        agg = {k: [] for k in ['Loss','MAE','RMSE','Change_MAE','Positive_MAE','Negative_MAE','Abs_MAE']}
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"[lr={lr_tag} γ={gamma}] Epoch {epoch:03d} [Valid]", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                preds = model(inputs)
                mv = evaluate_metrics(preds, targets, epsilon=epsilon)
                for k in agg:
                    agg[k].append(mv[k])
        val_log = {k: float(np.mean(agg[k])) for k in agg}

        tqdm.write(pretty_epoch_log(epoch, train_log, val_log))

        all_rows.append([
            epoch,
            train_log['Loss'], train_log['MAE'], train_log['RMSE'], train_log['Change_MAE'],
            val_log['Loss'], val_log['MAE'], val_log['RMSE'], val_log['Change_MAE'],
            val_log['Positive_MAE'], val_log['Negative_MAE'], val_log['Abs_MAE']
        ])

        def save_best(metric_key, value):
            if value < best[metric_key]["value"]:
                best[metric_key] = {"value": value, "epoch": epoch}
                save_path = os.path.join(model_dir, f"{metric_key}_best_lr{lr_tag}_gamma{gamma}.pt")
                torch.save(model.state_dict(), save_path)
        save_best("Val_MAE", val_log['MAE'])
        save_best("Val_Change_MAE", val_log['Change_MAE'])
        save_best("Val_RMSE", val_log['RMSE'])

        # Early stopping with dynamic patience
        if val_log['Change_MAE'] < best_cmae_for_early_stop:
            best_cmae_for_early_stop = val_log['Change_MAE']
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        if no_improve_epochs >= current_patience:
            tqdm.write(f"[lr={lr_tag} γ={gamma}] Early stopping at epoch {epoch:03d}")
            break

    save_metrics_csv(csv_path, header, all_rows)
    return {"lr": lr, "gamma": gamma, "best": best, "csv": csv_path}


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    lrs = [0.005, 0.001, 0.0005, 0.0001]
    gammas = [10, 15, 20, 25]

    results = []
    tqdm.write("=== Standard 3D UNet (tanh output) Training ===")
    tqdm.write(f"Device: {device}")

    for lr in lrs:
        for gamma in gammas:
            tqdm.write(f"\n--- Training combo: lr={lr:.0e}, gamma={gamma} ---")
            combo_res = train_one_combo(
                lr=lr, gamma=gamma, device=device, num_epochs=150
            )
            results.append(combo_res)

    global_best = {k: {"value": float('inf'), "lr": None, "gamma": None, "epoch": None}
                   for k in ["Val_MAE", "Val_Change_MAE", "Val_RMSE"]}

    for r in results:
        lr, gamma = r["lr"], r["gamma"]
        for metric in global_best.keys():
            v, ep = r["best"][metric]["value"], r["best"][metric]["epoch"]
            if v < global_best[metric]["value"]:
                global_best[metric] = {"value": v, "lr": lr, "gamma": gamma, "epoch": ep}

    line = "=" * 70
    print("\n" + line)
    print("Global Best (Validation)".center(70))
    print(line)
    print(f"{'Metric':<18}{'Best Value':>14}    {'lr':>8}    {'gamma':>6}    {'epoch':>6}")
    print("-" * 70)
    for metric, gb in global_best.items():
        print(f"{metric:<18}{gb['value']:>14.6f}    {gb['lr']:.0e:>8}    {gb['gamma']:>6d}    {gb['epoch']:>6d}")
    print(line)
    print("Metrics saved under: trainUnet/metrics/")
    print("Models saved under:  trainUnet/models/")
    print(line + "\n")


if __name__ == '__main__':
    main()
