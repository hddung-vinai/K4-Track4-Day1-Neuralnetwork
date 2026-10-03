# data.py — nạp train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên device, chia lô.
# Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).
#   X : float32 (N, 54) — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
#   y : int64   (N,)    — nhãn 0..6
# Tập eval CHỈ dùng để chấm điểm cuối: không tham gia chuẩn hoá, chọn cấu hình hay dừng sớm.
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10   # số cột liên tục cần chuẩn hoá (cột 0..9)
N_FEATURES = 54
N_CLASSES = 7


def load_split(processed_dir: str = 'data/processed'):
    # Trả về X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    tr = np.load(f'{processed_dir}/train.npz')
    ev = np.load(f'{processed_dir}/eval.npz')
    X_train_full, y_train_full = tr['X'], tr['y']
    X_eval, y_eval, eval_row_id = ev['X'], ev['y'], ev['row_id']
    for X, y in ((X_train_full, y_train_full), (X_eval, y_eval)):
        assert X.ndim == 2 and X.shape[1] == N_FEATURES and X.dtype == np.float32
        assert y.shape == (len(X),) and y.dtype == np.int64
        assert y.min() >= 0 and y.max() <= N_CLASSES - 1
    assert len(eval_row_id) == len(X_eval)
    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    # Tách validation TỪ train, phân tầng theo nhãn; cùng seed cho mọi thí nghiệm
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed)
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    # mean/std của 10 cột số, CHỈ trên phần train còn lại (tính trên val/eval là rò rỉ thông tin)
    mean = X_tr[:, :N_NUMERIC].mean(axis=0, dtype=np.float64)
    std = X_tr[:, :N_NUMERIC].std(axis=0, dtype=np.float64)
    std[std == 0] = 1.0   # tránh chia cho 0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X, mean, std):
    # Bản sao của X: 10 cột đầu -> (x - mean) / std; 44 cột nhị phân giữ nguyên
    X = X.copy()
    X[:, :N_NUMERIC] = (X[:, :N_NUMERIC] - mean) / std
    return X


def prepare_data(device, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = 'data/processed', verbose: bool = True) -> dict:
    # Gộp các bước và đưa TOÀN BỘ dữ liệu lên device một lần (không dùng DataLoader)
    X_full, y_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_full, y_full, val_fraction, seed)
    mean, std = fit_standardizer(X_tr)
    X_tr, X_val, X_eval = (apply_standardizer(a, mean, std) for a in (X_tr, X_val, X_eval))

    def to_t(a, dt):
        return torch.as_tensor(a, dtype=dt, device=device)

    data = dict(
        X_tr=to_t(X_tr, torch.float32), y_tr=to_t(y_tr, torch.int64),
        X_val=to_t(X_val, torch.float32), y_val=to_t(y_val, torch.int64),
        X_eval=to_t(X_eval, torch.float32), y_eval=to_t(y_eval, torch.int64),
        eval_row_id=eval_row_id, mean=mean, std=std,
    )
    if verbose:
        majority = int(np.bincount(y_tr, minlength=N_CLASSES).argmax())
        print(f'train_full={len(X_full)}  ->  X_tr={len(X_tr)}  X_val={len(X_val)}  |  X_eval={len(X_eval)}')
        print('tỉ lệ lớp (%):')
        for name, y in (('train', y_tr), ('val', y_val), ('eval', y_eval)):
            p = 100 * np.bincount(y, minlength=N_CLASSES) / len(y)
            print(f'  {name:5s}', ' '.join(f'{v:6.2f}' for v in p))
        print(f'lớp đa số (theo train) = {majority};  acc luôn-đoán-lớp-đa-số trên val = {(y_val == majority).mean():.4f}')
        m = X_tr[:, :N_NUMERIC].mean(0)
        s = X_tr[:, :N_NUMERIC].std(0)
        print(f'10 cột số trên X_tr sau chuẩn hoá: |mean| max = {np.abs(m).max():.2e}, std trong [{s.min():.4f}, {s.max():.4f}]')
    return data


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    # Thay cho DataLoader: xáo bằng randperm mỗi epoch rồi cắt lô.
    # Lô cuối nhỏ hơn batch_size vẫn được dùng (không bỏ mẫu nào).
    n = len(X)
    if shuffle:
        perm = torch.randperm(n, generator=generator, device=X.device)
    else:
        perm = torch.arange(n, device=X.device)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
