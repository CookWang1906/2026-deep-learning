from pathlib import Path

import numpy as np
import pandas as pd

# Thứ tự nhãn 0-6 theo mô tả của Kaggle (khác thứ tự bảng chữ cái của code thầy)
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
IMG_SIZE = 48
# Tên giá trị trong cột Usage của icml_face_data.csv -> tên tập dùng trong dự án
SPLITS = {"Training": "train", "PublicTest": "valid", "PrivateTest": "test"}

def _parse_pixels(pixel_strings):
    # Chuyển chuỗi '0 12 255 ...' thành mảng uint8 shape (N, 1, 48, 48).
    flat = np.array([np.array(s.split(), dtype=np.uint8) for s in pixel_strings])
    return flat.reshape(-1, 1, IMG_SIZE, IMG_SIZE)

def _read_csv(csv_path):
    # Đọc CSV và xóa khoảng trắng thừa trong tên cột và giá trị.
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if "pixels" in df.columns:
        df["pixels"] = df["pixels"].str.strip()
    return df

def load_fer2013(csv_path):
    # Đọc icml_face_data.csv, trả về (X uint8, y int64, usage str).
    df = _read_csv(csv_path)
    X = _parse_pixels(df["pixels"])
    y = df["emotion"].to_numpy(dtype=np.int64)
    usage = df["Usage"].str.strip().to_numpy()
    return X, y, usage

def split_arrays(X, y, usage):
    # Tách thành dict {'train': (x, y), 'valid': (x, y), 'test': (x, y)}.
    return {name: (X[usage == raw], y[usage == raw]) for raw, name in SPLITS.items()}

def load_kaggle_test(csv_path):
    # Đọc test.csv (chỉ có cột pixels, không có nhãn). Trả về uint8 (N, 1, 48, 48).
    return _parse_pixels(_read_csv(csv_path)["pixels"])

def compute_mean_std(x_train):
    # Mean và std toàn cục, đơn vị [0, 1]. Chỉ gọi với tập Training.
    x = x_train.astype(np.float32) / 255.0
    return float(x.mean()), float(x.std())

def save_processed(splits, mean, std, out_dir):
    # Lưu từng tập thành .npz (uint8) và thống kê chuẩn hóa thành stats.npz.
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (x, y) in splits.items():
        np.savez_compressed(out_dir / f"{name}.npz", x=x, y=y)
    np.savez(out_dir / "stats.npz", mean=mean, std=std)
