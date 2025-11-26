# ============================================================
# dataset_loader.py (normalized input version)
# ============================================================
import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib

# ----------------------- 随机种子 -----------------------
def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """保证多线程加载数据时随机性可复现"""
    seed = 2025 + worker_id
    np.random.seed(seed)
    random.seed(seed)


# ----------------------- 工具函数 -----------------------
def get_file_triplets(base_dir):
    """
    遍历四个子目录，返回所有 (before, after, label) 文件路径三元组
    """
    subdirs = [
        ("method1_sphere", "expand"),
        ("method1_sphere", "shrink"),
        ("method2_seg", "expand"),
        ("method2_seg", "shrink"),
    ]
    triplets = []

    for method, mode in subdirs:
        folder = os.path.join(base_dir, method, mode)
        pattern = os.path.join(folder, f"M*_{mode}_*_before_snr100.nii.gz")
        before_files = sorted(glob.glob(pattern))
        for b_path in before_files:
            name_core = os.path.basename(b_path).replace("_before_snr100.nii.gz", "")
            a_path = os.path.join(folder, f"{name_core}_after_snr100.nii.gz")
            l_path = os.path.join(folder, f"{name_core}_label.nii.gz")
            if os.path.exists(a_path) and os.path.exists(l_path):
                triplets.append((b_path, a_path, l_path))
    return sorted(triplets)


def split_dataset(all_triplets):
    """按照编号顺序 150/25/25 划分 train/val/test"""
    train, val, test = [], [], []
    for folder_triplets in [all_triplets[i:i + 200] for i in range(0, len(all_triplets), 200)]:
        train.extend(folder_triplets[:150])
        val.extend(folder_triplets[150:175])
        test.extend(folder_triplets[175:200])
    return train, val, test


# ----------------------- Dataset 类 -----------------------
class MRIDiffDataset(Dataset):
    def __init__(self, triplets, augment=False, snr50_ratio=0.5):
        """
        triplets: [(before_path, after_path, label_path), ...]
        augment: 若 True，则随机使用 snr50 替代 snr100 版本
        snr50_ratio: 使用 snr50 版本的概率 (0~1)
        """
        self.triplets = triplets
        self.augment = augment
        self.snr50_ratio = snr50_ratio

    def __len__(self):
        return len(self.triplets)

    def _load_nii(self, path):
        img = nib.load(path).get_fdata().astype(np.float32)
        return img

    def __getitem__(self, idx):
        before_path, after_path, label_path = self.triplets[idx]

        if self.augment:
            # 按概率替换成 snr50 版本
            if random.random() < self.snr50_ratio:
                before_path = before_path.replace("snr100", "snr50")
                after_path = after_path.replace("snr100", "snr50")

        before = self._load_nii(before_path)
        after = self._load_nii(after_path)
        label = self._load_nii(label_path)

        # 拼接 before / after 为 2 通道
        inputs = np.stack([before, after], axis=0)  # [2, D, H, W]

        # 归一化输入灰度到 [0, 1]（原图范围是 0~255）
        inputs = inputs / 255.0

        # 标签保持 [-1, 1] 不变
        label = label.astype(np.float32)

        return torch.from_numpy(inputs), torch.from_numpy(label).unsqueeze(0)


# ----------------------- get_dataloaders() -----------------------
def get_dataloaders(
    data_root="all_data",
    batch_size_train=4,
    batch_size_val=2,
    augment_snr50=True,
    snr50_ratio=0.5,
):
    """
    返回 train_loader, val_loader, test_loader
    """
    set_seed(2025)
    all_triplets = get_file_triplets(data_root)
    train_triplets, val_triplets, test_triplets = split_dataset(all_triplets)

    train_ds = MRIDiffDataset(train_triplets, augment=augment_snr50, snr50_ratio=snr50_ratio)
    val_ds = MRIDiffDataset(val_triplets, augment=False)
    test_ds = MRIDiffDataset(test_triplets, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=4,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=2,
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=2,
        worker_init_fn=seed_worker,
    )

    print(f"[INFO] Loaded dataset from {data_root}")
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val:   {len(val_ds)} samples")
    print(f"  Test:  {len(test_ds)} samples")
    print(f"  SNR50 augment ratio: {snr50_ratio if augment_snr50 else 0.0}")

    return train_loader, val_loader, test_loader


# ----------------------- 直接运行测试 -----------------------
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root="all_data",
        batch_size_train=2,
        batch_size_val=1,
        augment_snr50=True,
        snr50_ratio=0.3,
    )

    x, y = next(iter(train_loader))
    print(f"Input shape: {x.shape}, Label shape: {y.shape}")
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]  Label range: [{y.min():.3f}, {y.max():.3f}]")
