# ============================================================
# test_swin_unetr.py
# ============================================================
import torch
import numpy as np
from tqdm import tqdm
from monai.networks.nets import SwinUNETR
from data_loader import get_dataloaders


# SwinUNETR + tanh (保持和训练一致)
import torch.nn as nn
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


# 和训练脚本一致的指标计算函数
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

        return mae, rmse, change_mae, pos_mae, neg_mae, abs_mae


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on device:", device)

    # --------------------------------------------------------
    # 加载测试集
    # --------------------------------------------------------
    _, _, test_loader = get_dataloaders(
        data_root="all_data",
        batch_size_train=4,
        batch_size_val=2,
        augment_snr50=False,
    )

    # --------------------------------------------------------
    # 加载模型和权重
    # --------------------------------------------------------
    ckpt = "swin_coarse_fine/models/fine_from_coarse_gamma10_alpha20_gamma15_best_Val_ChangeMAE.pt"
    print(f"Loading checkpoint: {ckpt}")

    model = SwinUNETR_Tanh().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # --------------------------------------------------------
    # 评估
    # --------------------------------------------------------
    all_mae = []
    all_rmse = []
    all_change_mae = []
    all_pos_mae = []
    all_neg_mae = []
    all_abs_mae = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            preds = model(inputs)

            mae, rmse, cmae, pmae, nmae, absmae = evaluate_metrics(preds, targets)

            all_mae.append(mae)
            all_rmse.append(rmse)
            all_change_mae.append(cmae)
            all_pos_mae.append(pmae)
            all_neg_mae.append(nmae)
            all_abs_mae.append(absmae)

    print("\n========== Test Results ==========")
    print(f"Test MAE = {np.mean(all_mae):.6f}")
    print(f"Test RMSE = {np.mean(all_rmse):.6f}")
    print(f"Test Change MAE = {np.mean(all_change_mae):.6f}")
    print(f"Test Positive MAE = {np.mean(all_pos_mae):.6f}")
    print(f"Test Negative MAE = {np.mean(all_neg_mae):.6f}")
    print(f"Test Abs MAE = {np.mean(all_abs_mae):.6f}")
    print("==================================\n")


if __name__ == "__main__":
    main()
