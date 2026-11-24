"""Dataset, vòng lặp train/eval, EarlyStopping và ghi log cho các kiến trúc trong models.py."""
import csv
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.transforms import v2 as T

from data_utils import EMOTIONS

log = logging.getLogger("train")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Thiết bị: %s", device)
    return device


def load_processed(processed_dir):
    """Đọc các file .npz do notebook 01 tạo ra. Trả về (splits, stats)."""
    d = Path(processed_dir)
    splits = {}
    for name in ["train", "valid", "test"]:
        z = np.load(d / f"{name}.npz")
        splits[name] = (z["x"], z["y"])
    s = np.load(d / "stats.npz")
    return splits, {"mean": float(s["mean"]), "std": float(s["std"])}


def stratified_subset(y, n, seed=42):
    """Lấy khoảng n chỉ số, giữ nguyên tỉ lệ từng lớp như tập gốc. n=None -> toàn bộ."""
    if n is None or n >= len(y):
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        cls_idx = np.where(y == c)[0]
        k = max(1, round(n * len(cls_idx) / len(y)))
        idx.append(rng.choice(cls_idx, k, replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return idx


def balanced_sampler(y):
    """Tạo sampler để mỗi lớp có xác suất được lấy mẫu gần như nhau."""
    y = np.asarray(y, dtype=np.int64)
    class_counts = np.bincount(y, minlength=len(EMOTIONS))
    missing = [EMOTIONS[i] for i, count in enumerate(class_counts) if count == 0]
    if missing:
        raise ValueError(f"Tập huấn luyện thiếu lớp: {', '.join(missing)}")
    sample_weights = 1.0 / class_counts[y]
    return WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(y),
        replacement=True,
    )


class FERDataset(Dataset):
    """Ảnh uint8 (N, 1, 48, 48) -> float [0, 1] -> augment (chỉ tập train) -> chuẩn hóa."""

    def __init__(self, x, y, mean, std, augment=False):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y).long()
        self.mean, self.std = mean, std
        # Tương ứng với tham số của Keras trong báo cáo: flip ngang, xoay 10 độ,
        # dịch 0.1, zoom 0.1. RandomAffine không có fill_mode='nearest' nên dùng fill=0.
        self.aug = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ]) if augment else None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = self.x[i].float() / 255.0
        if self.aug is not None:
            img = self.aug(img)
        return (img - self.mean) / self.std, self.y[i]


class EarlyStopping:
    """Dừng khi val_loss không giảm sau `patience` epoch liên tiếp."""

    def __init__(self, patience=10):
        self.patience = patience
        self.best = float("inf")
        self.counter = 0

    def step(self, val_loss):
        """Trả về True nếu val_loss vừa cải thiện."""
        if val_loss < self.best:
            self.best, self.counter = val_loss, 0
            return True
        self.counter += 1
        return False

    @property
    def should_stop(self):
        return self.counter >= self.patience


def run_epoch(model, loader, device, criterion, optimizer=None, max_batches=None, clip=None):
    """Chạy một lượt qua loader. Có optimizer thì là train, không có thì là eval.
    clip: ngưỡng cắt gradient (chỉ dùng khi train). Trả về (loss, accuracy, số batch)."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen, n_batches = 0.0, 0, 0, 0
    with torch.enable_grad() if training else torch.no_grad():
        for i, (xb, yb) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                if clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            seen += len(yb)
            n_batches += 1
    return total_loss / seen, correct / seen, n_batches


def fit(name, model, train_loader, valid_loader, device, epochs, lr,
        run_id, logs_dir, ckpt_dir, patience=10, clip=1.0):
    """Train một mô hình, lưu checkpoint có val_loss thấp nhất và log từng epoch ra CSV."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stopper = EarlyStopping(patience=patience)

    logs_dir, ckpt_dir = Path(logs_dir), Path(ckpt_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_path = logs_dir / f"{name}_{run_id}.csv"
    ckpt_path = ckpt_dir / f"{name}_{run_id}_best.pt"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "seconds"])

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc, _ = run_epoch(model, train_loader, device, criterion, optimizer, clip=clip)
        va_loss, va_acc, _ = run_epoch(model, valid_loader, device, criterion)
        sec = time.perf_counter() - t0

        if stopper.step(va_loss):
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_loss": va_loss, "val_acc": va_acc}, ckpt_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, f"{tr_loss:.4f}", f"{tr_acc:.4f}",
                                    f"{va_loss:.4f}", f"{va_acc:.4f}", f"{sec:.1f}"])
        log.info("%s | epoch %d/%d | train_loss %.4f acc %.4f | val_loss %.4f acc %.4f | %.0f s",
                 name, epoch, epochs, tr_loss, tr_acc, va_loss, va_acc, sec)

        if stopper.should_stop:
            log.info("%s | EarlyStopping tại epoch %d (patience=%d)", name, epoch, patience)
            break
    log.info("%s | checkpoint tốt nhất: %s (val_loss %.4f)", name, ckpt_path.name, stopper.best)
    return ckpt_path, csv_path