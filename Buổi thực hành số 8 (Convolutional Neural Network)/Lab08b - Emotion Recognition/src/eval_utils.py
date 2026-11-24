"""Hàm tiện ích đánh giá và trực quan hóa kết quả (Ngày 2 và Ngày 3)."""
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from train_utils import FERDataset

# Thứ tự nhãn theo Kaggle FER2013
CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


@torch.no_grad()
def predict(model, x, y, mean, std, device, batch_size=128):
    """Trả về (xác suất (N,7), nhãn dự đoán (N,)) trên toàn bộ x."""
    model.eval()
    loader = DataLoader(FERDataset(x, y, mean, std, augment=False),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    probs = []
    for xb, _ in loader:
        probs.append(torch.softmax(model(xb.to(device)), dim=1).cpu().numpy())
    probs = np.concatenate(probs)
    return probs, probs.argmax(axis=1)


def plot_histories(history_csvs, path=None):
    """history_csvs: dict {tên: đường dẫn CSV}. Vẽ loss và accuracy (nét liền = valid, nét đứt = train)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, p in history_csvs.items():
        h = pd.read_csv(p)
        (line,) = axes[0].plot(h["epoch"], h["val_loss"], label=f"{name} (valid)")
        axes[0].plot(h["epoch"], h["train_loss"], "--", alpha=0.5, color=line.get_color())
        (line,) = axes[1].plot(h["epoch"], h["val_acc"], label=f"{name} (valid)")
        axes[1].plot(h["epoch"], h["train_acc"], "--", alpha=0.5, color=line.get_color())
    axes[0].set_title("Loss theo epoch")
    axes[1].set_title("Accuracy theo epoch")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion(cm, class_names, title, path=None, normalize=False):
    cm = np.asarray(cm, dtype=float)
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ticks = range(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(title)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


def show_images(x, y_true, y_pred, probs, idx, title, ncols=6, path=None):
    """Lưới ảnh với nhãn thật / nhãn đoán (xanh = đúng, đỏ = sai)."""
    n = len(idx)
    if n == 0:
        print("Không có ảnh để hiển thị.")
        return
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, i in zip(axes, idx):
        ax.imshow(x[i, 0], cmap="gray", vmin=0, vmax=255)
        ok = y_true[i] == y_pred[i]
        ax.set_title(f"Thật: {CLASS_NAMES[y_true[i]]}\nĐoán: {CLASS_NAMES[y_pred[i]]} "
                     f"({probs[i, y_pred[i]]:.0%})",
                     fontsize=8, color="green" if ok else "red")
    fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


def predict_image_file(model, image_path, mean, std, device, detect_face=True):
    """Dự đoán cảm xúc từ một file ảnh bất kỳ. Cần opencv-python.
    Trả về (ảnh xám gốc, ảnh mặt 48x48, khung mặt hoặc None, xác suất 7 lớp)."""
    import cv2

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")
    face, box = img, None
    if detect_face:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        if len(faces):
            x0, y0, w, h = max(faces, key=lambda f: f[2] * f[3])
            face, box = img[y0:y0 + h, x0:x0 + w], (x0, y0, w, h)
    face48 = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(face48).float().div(255.0)
    t = ((t - mean) / std)[None, None].to(device)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1)[0].cpu().numpy()
    return img, face48, box, probs
