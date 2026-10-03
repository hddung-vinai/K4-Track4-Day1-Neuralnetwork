# optimizer.py — chọn bộ tối ưu, lập lịch lr và cắt gradient (dùng torch.optim, clip_grad_norm_).
#   SGD            : w <- w - lr * g
#   SGD + momentum : v <- mu * v + g ;  w <- w - lr * v          (dạng PyTorch)
#   Adam           : m <- b1 m + (1-b1) g ; v <- b2 v + (1-b2) g^2 ; w <- w - lr * m_hat / (sqrt(v_hat) + eps)
#   AdamW          : như Adam nhưng suy giảm trọng số tách riêng: w <- w - lr * wd * w - lr * m_hat / (sqrt(v_hat) + eps)
from __future__ import annotations

import math

import torch

OPTIMIZERS = ('sgd', 'sgd_momentum', 'adam', 'adamw')
# tên hiển thị trong experiments.xlsx (đúng danh sách của mẫu)
OPT_DISPLAY = {'sgd': 'SGD', 'sgd_momentum': 'SGD+momentum', 'adam': 'Adam', 'adamw': 'AdamW'}


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    if name not in OPTIMIZERS:
        raise ValueError(f'optimizer phải thuộc {OPTIMIZERS}, nhận {name!r}')
    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == 'sgd_momentum':
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == 'adam':
        # weight_decay của Adam = L2 cộng vào gradient (bị chia cho sqrt(v_hat))
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    # AdamW: suy giảm trọng số tách riêng khỏi bước thích nghi
    return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def build_scheduler(optimizer, name: str | None, total_steps: int, **kwargs):
    # None -> không lập lịch. 'cosine' -> CosineAnnealingLR theo TỪNG BƯỚC (gọi .step() sau mỗi lô)
    if name is None:
        return None
    if name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps,
                                                          eta_min=kwargs.get('eta_min', 0.0))
    raise ValueError(f'scheduler không hỗ trợ: {name!r}')


def clip_gradients(params, max_norm: float | None) -> torch.Tensor:
    # Cắt gradient theo chuẩn L2 TOÀN CỤC và trả về chuẩn TRƯỚC KHI cắt (tensor 0-d, chưa .item()
    # để không ép GPU đồng bộ ở mỗi bước). max_norm=None: chỉ đo, không cắt (max_norm = inf).
    # FP16 + GradScaler: phải scaler.unscale_(optimizer) TRƯỚC khi gọi hàm này.
    limit = math.inf if max_norm is None else float(max_norm)
    return torch.nn.utils.clip_grad_norm_(params, limit)
