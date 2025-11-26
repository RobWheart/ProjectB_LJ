# ============================================================
# train_swin_unetr_tanh_fixed.py  (Dynamic patience version)
# ============================================================
import os
import csv
import itertools
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from monai.networks.nets import SwinUNETR
from data_loader import get_dataloaders

# ---------------------- 随机种子 ----------------------
torch.manual_seed(2025)
np.random.seed(2025)


# ============================================================
# Weighted MSE Loss
# ============================================================
class WeightedMSELoss(nn.Module):
    def __init__(self, epsilon=1e-4, gamma=20.0):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = gamma

    def forward(self, pred, target):
        weight = 1.0 + self.gamma * (target.abs() > self.epsilon).float()
        loss = weight * (pred - target) ** 2
        return loss.mean()


# ============================================================
# 评估指标
# ============================================================
def evaluate_metrics(pred, target, epsilon=1e-4):
    with torch.no_grad():
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
# SwinUNETR + tanh 输出
# ============================================================
class SwinUNETR_Tanh(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, feature_size=24, use_checkpoint=False):
        super().__init__()
        self.model = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint
        )

    def forward(self, x):
        return torch.tanh(self.model(x))


# ============================================================
# 单组训练逻辑
# ============================================================
def train_one_combo(lr, gamma, device="cuda", num_epochs=150):
    print(f"\n=== Training Swin UNETR (feature_size=24, lr={lr}, gamma={gamma}) ===")

    train_loader, val_loader, _ = get_dataloaders(
        data_root="all_data",
        batch_size_train=4,
        batch_size_val=2,
        augment_snr50=True,
        snr50_ratio=0.2
    )

    model = SwinUNETR_Tanh(in_channels=2, out_channels=1, feature_size=24).to(device)
    criterion = WeightedMSELoss(epsilon=1e-4, gamma=gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    metric_dir = "swin_unetr/metrics"
    model_dir = "swin_unetr/models"
    os.makedirs(metric_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    csv_path = os.path.join(metric_dir, f"metrics_lr{lr:.0e}_gamma{gamma}.csv")

    header = [
        "Epoch",
        "Train_Loss", "Train_MAE", "Train_RMSE", "Train_Change_MAE", "Train_Positive_MAE", "Train_Negative_MAE", "Train_Abs_MAE",
        "Val_Loss", "Val_MAE", "Val_RMSE", "Val_Change_MAE", "Val_Positive_MAE", "Val_Negative_MAE", "Val_Abs_MAE"
    ]
    all_rows = []

    best = {k: {"value": float("inf"), "epoch": -1} for k in header[8:]}
    best_cmae_for_stop = float("inf")
    no_improve = 0

    for epoch in range(1, num_epochs + 1):

        # 动态 patience 机制
        current_patience = 10 if epoch <= 100 else 5

        model.train()
        train_metrics = {k: [] for k in header[1:8]}

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch:03d} [Train]", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            m = evaluate_metrics(preds, targets)
            for k in train_metrics:
                train_metrics[k].append(m[k.replace("Train_", "")])

        train_avg = {k: np.mean(v) for k, v in train_metrics.items()}

        model.eval()
        val_metrics = {k: [] for k in header[8:]}

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch:03d} [Valid]", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                preds = model(inputs)
                m = evaluate_metrics(preds, targets)
                for k in val_metrics:
                    val_metrics[k].append(m[k.replace("Val_", "")])

        val_avg = {k: np.mean(v) for k, v in val_metrics.items()}

        print(f"\nEpoch {epoch:03d}")
        print("Train:", ", ".join([f"{k}={v:.4f}" for k, v in train_avg.items()]))
        print("Val:  ", ", ".join([f"{k}={v:.4f}" for k, v in val_avg.items()]))

        # 保存最优模型
        for metric in val_avg:
            value = val_avg[metric]
            if value < best[metric]["value"]:
                best[metric] = {"value": value, "epoch": epoch}
                save_path = os.path.join(model_dir, f"{metric}_best_lr{lr:.0e}_gamma{gamma}.pt")
                torch.save(model.state_dict(), save_path)

        # Early stopping based on Change_MAE
        if val_avg["Val_Change_MAE"] < best_cmae_for_stop:
            best_cmae_for_stop = val_avg["Val_Change_MAE"]
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= current_patience:
            print(f"\nEarly stopping at epoch {epoch:03d}  (patience={current_patience})")
            break

        row = [epoch] + [train_avg[k] for k in header[1:8]] + [val_avg[k] for k in header[8:]]
        all_rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_rows)

    print(f"\nTraining finished for lr={lr:.0e}, gamma={gamma}")
    for k, v in best.items():
        print(f"  {k}: epoch {v['epoch']}, value={v['value']:.6f}")

    return {"lr": lr, "gamma": gamma, "best": best}


# ============================================================
# 主程序
# ============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    lrs = [0.005, 0.001, 0.0005, 0.0001]
    gammas = [10, 15, 20, 25]
    results = []

    for lr, gamma in itertools.product(lrs, gammas):
        res = train_one_combo(lr=lr, gamma=gamma, device=device)
        results.append(res)

    global_best = {k: {"value": float("inf"), "lr": None, "gamma": None, "epoch": None}
                   for k in ["Val_Loss", "Val_MAE", "Val_RMSE", "Val_Change_MAE",
                             "Val_Positive_MAE", "Val_Negative_MAE", "Val_Abs_MAE"]}

    for r in results:
        lr, gamma = r["lr"], r["gamma"]
        for metric in global_best:
            v, ep = r["best"][metric]["value"], r["best"][metric]["epoch"]
            if v < global_best[metric]["value"]:
                global_best[metric] = {"value": v, "lr": lr, "gamma": gamma, "epoch": ep}

    print("\n" + "=" * 75)
    print("Global Best Validation Results".center(75))
    print("=" * 75)
    print(f"{'Metric':<20}{'Best Value':>14}   {'lr':>8}   {'gamma':>6}   {'epoch':>6}")
    print("-" * 75)
    for metric, gb in global_best.items():
        print(f"{metric:<20}{gb['value']:>14.6f}   {gb['lr']:.0e:>8}   {gb['gamma']:>6d}   {gb['epoch']:>6d}")
    print("=" * 75)
    print("Metrics saved at swin_unetr/metrics/")
    print("Models saved at swin_unetr/models/")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
