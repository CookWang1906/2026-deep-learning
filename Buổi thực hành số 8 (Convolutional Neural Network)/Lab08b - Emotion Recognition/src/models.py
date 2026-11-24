"""Bốn kiến trúc CNN theo báo cáo của thầy: small, medium, medium_01, large.

Mỗi khối gồm các lớp Conv3x3 (padding=1) [+ BatchNorm] + ReLU, sau đó MaxPool2x2 và Dropout(0.2).
Phần phân loại: Flatten -> Linear(fc_units) -> ReLU -> Dropout(0.5) -> Linear(7).
Mô hình trả về logits, không có Softmax (CrossEntropyLoss đã tích hợp sẵn).

Khác biệt so với báo cáo gốc:
- use_bn=True thêm BatchNorm sau mỗi Conv, giúp mạng sâu không bị "đứng yên" ở mức đoán theo
  tần suất lớp khi huấn luyện với ít dữ liệu. Đặt use_bn=False để quay về kiến trúc gốc.
- Trọng số khởi tạo Glorot uniform, bias = 0 (mặc định của Keras).
"""
import torch.nn as nn

# Mỗi phần tử là một khối, số trong khối là số kênh của từng lớp Conv
MODEL_CONFIGS = {
    "small":     {"blocks": [[64, 64], [128, 128, 256, 256]],
                  "fc_units": 1024},
    "medium":    {"blocks": [[64, 64], [128, 128, 128], [256, 256, 256]],
                  "fc_units": 1024},
    "medium_01": {"blocks": [[64, 64, 64], [128, 128, 128, 128], [256, 256, 256, 256]],
                  "fc_units": 1024},
    "large":     {"blocks": [[64, 64, 64, 64], [128, 128, 128, 128],
                             [256, 256, 256, 256], [512, 512, 512, 512]],
                  "fc_units": 2048},
}
MODEL_NAMES = list(MODEL_CONFIGS)


def _init_keras_style(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class EmotionCNN(nn.Module):
    def __init__(self, blocks, fc_units, num_classes=7, in_channels=1, img_size=48, use_bn=True):
        super().__init__()
        layers, ch, size = [], in_channels, img_size
        for convs in blocks:
            for out_ch in convs:
                layers.append(nn.Conv2d(ch, out_ch, kernel_size=3, padding=1))
                if use_bn:
                    layers.append(nn.BatchNorm2d(out_ch))
                layers.append(nn.ReLU(inplace=True))
                ch = out_ch
            layers += [nn.MaxPool2d(2), nn.Dropout(0.2)]
            size //= 2
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch * size * size, fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(fc_units, num_classes),
        )
        self.apply(_init_keras_style)

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(name, use_bn=True):
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Không có kiến trúc '{name}'. Chọn một trong {MODEL_NAMES}")
    return EmotionCNN(**MODEL_CONFIGS[name], use_bn=use_bn)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)