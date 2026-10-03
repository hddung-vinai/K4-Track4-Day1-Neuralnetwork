# model.py — MLP cho bài toán 7 lớp, shape cố định (README mục 3, GUIDE "Quy định kiến trúc"):
#   x (B, 54) -> Linear(54, h1) -> ReLU -> [Dropout] -> Linear(h1, h2) -> ReLU -> [Dropout]
#             -> ... -> Linear(h_last, 7) -> logits (B, 7)
# - Lớp cuối ra logit thô, KHÔNG softmax (softmax nằm trong hàm mất mát).
# - Dropout chỉ đặt sau ReLU của lớp ẩn; mọi Linear có bias; không BatchNorm, không residual.
from __future__ import annotations

import torch
import torch.nn as nn

EXPECTED_PARAMS = {
    (256, 128): 47_879,        # M-base  (baseline)
    (512, 256): 161_287,       # M-wide
    (256, 128, 64): 55_687,    # M-deep
}
INITS = ('zeros', 'normal', 'xavier', 'he', 'default')


class MLP(nn.Module):
    # hidden: số nơ-ron các lớp ẩn; dropout: xác suất TẮT q (= p của nn.Dropout); init: xem init_weights
    def __init__(self, hidden=(256, 128), dropout: float = 0.0, init: str = 'he',
                 in_features: int = 54, num_classes: int = 7):
        super().__init__()
        layers = []
        d = in_features
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, num_classes))
        self.net = nn.Sequential(*layers)
        self.hidden = tuple(hidden)
        init_weights(self, init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, 54) -> (B, 7) logits


def init_weights(model: nn.Module, init: str) -> None:
    # Khởi tạo W của MỌI nn.Linear (kể cả lớp ra), bias = 0.
    #   zeros  : W = 0
    #   normal : W ~ N(0, 0.01^2)
    #   xavier : xavier_normal_, Var = 2/(n_in + n_out)  (công thức của PyTorch, không phải 1/n_in của slide)
    #   he     : kaiming_normal_(nonlinearity='relu'), Var = 2/n_in (fan_in)
    #   default: giữ khởi tạo mặc định của nn.Linear (Kaiming-uniform với a=sqrt(5), KHÔNG phải He)
    if init not in INITS:
        raise ValueError(f'init phải thuộc {INITS}, nhận {init!r}')
    if init == 'default':
        return
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if init == 'zeros':
                nn.init.zeros_(m.weight)
            elif init == 'normal':
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight)
            elif init == 'he':
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def activation_stats(model: nn.Module, x: torch.Tensor) -> list[float]:
    # Độ lệch chuẩn kích hoạt SAU MỖI ReLU của lớp ẩn, cuối cùng là std của logits.
    # Ví dụ M-base trả về [std(ReLU1), std(ReLU2), std(logits)].
    was_training = model.training
    model.eval()
    stds = []
    h = x
    for layer in model.net:
        h = layer(h)
        if isinstance(layer, nn.ReLU):
            stds.append(h.std().item())
    stds.append(h.std().item())   # logits
    model.train(was_training)
    return stds
