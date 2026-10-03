# train.py — đặt seed, đánh giá, vòng huấn luyện run_experiment(cfg, data), dự đoán và ghi file nộp.
# Mọi thí nghiệm chỉ là đổi dict cfg rồi gọi lại run_experiment (GUIDE, Part 2).
# Chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
from __future__ import annotations

import json
import math
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import iterate_batches, N_CLASSES
from model import MLP, EXPECTED_PARAMS, count_params, activation_stats
from optimizer import build_optimizer, build_scheduler, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). lr được chọn bằng VAL trong notebook (Part 2) rồi ghi đè.
DEFAULT_CFG = dict(
    exp_id='base-s1', group='baseline', description='Baseline M-base',
    loss='ce',                 # 'ce' | 'mse'
    optimizer='sgd_momentum',  # 'sgd' | 'sgd_momentum' | 'adam' | 'adamw'
    lr=None,                   # chọn bằng val trong notebook
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init='he',
    clip_norm=None,            # None = không clip
    precision='fp32',          # 'fp32' | 'fp16' | 'bf16'
    scheduler=None,            # None | 'cosine'
    seed=1,
    notes='',
)
TRAIN_SUBSET = 50_000   # train loss đo ở eval() trên một tập con CỐ ĐỊNH của train (giống nhau cho mọi thí nghiệm)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scores_from_confusion(cm: np.ndarray):
    # Giống scripts/evaluate.py: hàng = nhãn thật, cột = dự đoán; F1 = 0 nếu mẫu số = 0
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return prec, rec, f1


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    return float(scores_from_confusion(cm)[2].mean())


def compute_loss(logits, y, loss_name: str, reduction: str = 'mean'):
    # 'ce' : cross-entropy trên logit thô và nhãn int64.
    # 'mse': nn.MSELoss giữa logit và one-hot(y); 'mean' = trung bình trên MỌI phần tử (B*7), không có hệ số 1/2.
    #        reduction='sum' trả về tổng đã chia 7 (tức tổng theo mẫu) để evaluate chia N ra đúng thang 'mean'.
    logits = logits.float()
    if loss_name == 'ce':
        return F.cross_entropy(logits, y, reduction=reduction)
    if loss_name == 'mse':
        target = F.one_hot(y, N_CLASSES).float()
        if reduction == 'sum':
            return F.mse_loss(logits, target, reduction='sum') / N_CLASSES
        return F.mse_loss(logits, target, reduction=reduction)
    raise ValueError(f'loss không hỗ trợ: {loss_name!r}')


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    model.eval()
    out = [model(X[i:i + batch_size]).argmax(dim=1) for i in range(0, len(X), batch_size)]
    return torch.cat(out)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = 'ce', batch_size: int = 8192) -> dict:
    # loss, acc, macro_f1 ở chế độ eval() (dropout tắt), FP32
    model.eval()
    total = torch.zeros((), device=X.device, dtype=torch.float64)
    preds = []
    for i in range(0, len(X), batch_size):
        logits = model(X[i:i + batch_size])
        total += compute_loss(logits, y[i:i + batch_size], loss_name, reduction='sum').double()
        preds.append(logits.argmax(dim=1))
    pred = torch.cat(preds)
    cm = torch.bincount(y * N_CLASSES + pred, minlength=N_CLASSES * N_CLASSES)
    cm = cm.reshape(N_CLASSES, N_CLASSES).cpu().numpy()
    return dict(loss=float(total.item() / len(X)), acc=float(np.trace(cm) / cm.sum()),
                macro_f1=macro_f1_from_confusion(cm), cm=cm)


def _finite_stats(values: np.ndarray):
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return math.nan, math.nan
    return float(v.mean()), float(v.max())


def run_experiment(cfg: dict, data: dict, verbose: bool = True) -> dict:
    # Huấn luyện một cấu hình, trả về {cfg, history, summary, best_state}.
    # Chỉ dùng X_tr / X_val. X_eval KHÔNG được dùng ở đây (không chọn epoch/cấu hình bằng eval).
    cfg = {**DEFAULT_CFG, **cfg}
    cfg['hidden'] = tuple(cfg['hidden'])
    assert cfg['lr'] is not None, 'cần đặt lr'
    X_tr, y_tr, X_val, y_val = data['X_tr'], data['y_tr'], data['X_val'], data['y_val']
    device = X_tr.device
    is_cuda = device.type == 'cuda'

    # ---- 0. model, optimizer, (scaler)
    set_seed(cfg['seed'])
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()
    model = MLP(cfg['hidden'], cfg['dropout'], cfg['init']).to(device)
    n_params = count_params(model)
    assert n_params == EXPECTED_PARAMS[cfg['hidden']], f'số tham số {n_params} sai quy định'
    opt = build_optimizer(cfg['optimizer'], model.parameters(), cfg['lr'],
                          weight_decay=cfg['weight_decay'], momentum=cfg['momentum'])
    steps_per_epoch = math.ceil(len(X_tr) / cfg['batch'])
    sched = build_scheduler(opt, cfg['scheduler'], steps_per_epoch * cfg['epochs'])
    amp_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16}.get(cfg['precision'])
    use_amp = amp_dtype is not None
    # GradScaler chỉ cho FP16 (BF16 có cùng dải số mũ với FP32 nên không cần nhân loss)
    scaler = torch.amp.GradScaler(device.type, enabled=(cfg['precision'] == 'fp16'))
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg['seed'])
    sub_idx = torch.randperm(len(X_tr), generator=torch.Generator().manual_seed(0))[:TRAIN_SUBSET].to(device)
    X_sub, y_sub = X_tr[sub_idx], y_tr[sub_idx]

    # ---- 1. kiểm tra bước 0 (TRƯỚC cập nhật đầu tiên)
    act_std0 = activation_stats(model, X_val[:4096])
    step0_loss = evaluate(model, X_val, y_val, cfg['loss'])['loss']

    keys = ('epoch', 'train_loss', 'val_loss', 'val_acc', 'val_macro_f1', 'grad_norm', 'grad_norm_max',
            'clip_frac', 'nonfinite_grad_steps', 'lr', 'epoch_time_s')
    hist = {k: [] for k in keys}
    all_norms = []
    best = dict(val_loss=math.inf, epoch=None, state=None, acc=math.nan, f1=math.nan)
    diverged = False
    if verbose:
        print(f"[{cfg['exp_id']}] step0 val loss = {step0_loss:.4f}  (ln7 = {math.log(7):.4f})")

    # ---- 2. vòng huấn luyện
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        norms = []
        for step, (xb, yb) in enumerate(iterate_batches(X_tr, y_tr, cfg['batch'], gen)):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(xb)   # chỉ bọc forward + loss trong autocast
            loss = compute_loss(logits, yb, cfg['loss'])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)   # FP16: đưa gradient về thang thật TRƯỚC khi đo/cắt (no-op nếu scaler tắt)
            norms.append(clip_gradients(model.parameters(), cfg['clip_norm']))   # chuẩn TRƯỚC khi cắt
            scaler.step(opt)       # FP16: tự bỏ qua bước có inf/NaN; ngược lại = opt.step()
            scaler.update()
            if sched is not None:
                sched.step()
            # phát hiện phân kỳ (đồng bộ GPU 100 bước một lần cho rẻ); FP16 có thể tràn tạm thời nên bỏ qua
            if step % 100 == 0 and cfg['precision'] != 'fp16' and not torch.isfinite(loss).item():
                diverged = True
                break
        if is_cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        nrm = torch.stack(norms).float().cpu().numpy()
        all_norms.append(nrm)
        tr_m = evaluate(model, X_sub, y_sub, cfg['loss'])
        va_m = evaluate(model, X_val, y_val, cfg['loss'])
        g_mean, g_max = _finite_stats(nrm)
        clip_frac = float((nrm > cfg['clip_norm']).mean()) if cfg['clip_norm'] is not None else 0.0
        for k, v in zip(keys, (epoch, tr_m['loss'], va_m['loss'], va_m['acc'], va_m['macro_f1'], g_mean, g_max,
                               clip_frac, int((~np.isfinite(nrm)).sum()), opt.param_groups[0]['lr'], dt)):
            hist[k].append(v)
        if not (math.isfinite(tr_m['loss']) and math.isfinite(va_m['loss'])):
            diverged = True
        if not diverged and va_m['loss'] < best['val_loss']:
            best.update(val_loss=va_m['loss'], epoch=epoch, acc=va_m['acc'], f1=va_m['macro_f1'],
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if verbose:
            print(f"  ep {epoch:2d}  train {tr_m['loss']:.4f}  val {va_m['loss']:.4f}  acc {va_m['acc']:.4f}  "
                  f"F1 {va_m['macro_f1']:.4f}  |g| {g_mean:.3f} (max {g_max:.2f})  {dt:.2f}s")
        if diverged:
            print(f"  -> PHÂN KỲ (NaN/inf) ở epoch {epoch}; dừng sớm")
            break

    # ---- 3. tóm tắt (metric lấy ở best_epoch = epoch có val_loss thấp nhất)
    norms_all = np.concatenate(all_norms) if all_norms else np.array([math.nan])
    fin = norms_all[np.isfinite(norms_all)]
    pct = (lambda q: float(np.percentile(fin, q))) if len(fin) else (lambda q: math.nan)
    peak_mem = None
    if is_cuda:
        peak_mem = (torch.cuda.max_memory_allocated() - mem_before) / 2 ** 20
    summary = dict(
        step0_loss=step0_loss,
        best_val_loss=best['val_loss'] if best['epoch'] else math.nan,
        best_epoch=best['epoch'],
        final_train_loss=hist['train_loss'][-1] if hist['train_loss'] else math.nan,
        final_val_loss=hist['val_loss'][-1] if hist['val_loss'] else math.nan,
        val_acc=best['acc'], val_macro_f1=best['f1'],
        time_per_epoch_s=float(np.mean(hist['epoch_time_s'])) if hist['epoch_time_s'] else math.nan,
        peak_mem_MB=peak_mem, diverged=diverged,
        n_params=n_params, n_steps=int(len(norms_all)) if all_norms else 0, steps_per_epoch=steps_per_epoch,
        act_std_step0=act_std0,
        grad_norm_p50=pct(50), grad_norm_p90=pct(90), grad_norm_p99=pct(99),
        grad_norm_max=float(fin.max()) if len(fin) else math.nan,
        clip_frac_overall=float((norms_all > cfg['clip_norm']).mean()) if cfg['clip_norm'] is not None else 0.0,
    )
    if verbose:
        print(f"  => best_epoch {summary['best_epoch']}  best_val_loss {summary['best_val_loss']:.4f}  "
              f"val_acc {summary['val_acc']:.4f}  val_macro_f1 {summary['val_macro_f1']:.4f}  "
              f"{summary['time_per_epoch_s']:.2f}s/epoch  diverged={diverged}")
    return dict(cfg=cfg, history=hist, summary=summary, best_state=best['state'])


def write_predictions(row_id, preds, path: str) -> None:
    # CSV tiêu đề row_id,pred; đủ mọi dòng eval, mỗi row_id đúng một lần, pred 0..6
    row_id = np.asarray(row_id)
    preds = np.asarray(preds).astype(np.int64)
    assert len(row_id) == len(preds) and len(np.unique(row_id)) == len(row_id)
    assert preds.min() >= 0 and preds.max() <= N_CLASSES - 1
    pd.DataFrame({'row_id': row_id, 'pred': preds}).to_csv(path, index=False)


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> np.ndarray:
    # Dùng cho baseline và cấu hình cuối: nạp best_state (epoch val_loss thấp nhất), dự đoán TOÀN BỘ eval
    cfg = {**DEFAULT_CFG, **cfg}
    model = MLP(tuple(cfg['hidden']), cfg['dropout'], cfg['init']).to(data['X_eval'].device)
    model.load_state_dict(result['best_state'])
    preds = predict(model, data['X_eval']).cpu().numpy()   # FP32, eval mode (không dropout)
    write_predictions(data['eval_row_id'], preds, pred_path)
    print(f'đã ghi {pred_path} ({len(preds)} dòng)')
    return preds


def run_evaluate_script(repo_root: str, pred_path: str, out_path: str) -> dict:
    # Chạy ĐÚNG scripts/evaluate.py của giảng viên, in kết quả, trả về nội dung JSON
    import os
    cmd = [sys.executable, 'scripts/evaluate.py', '--pred', os.path.abspath(pred_path),
           '--out', os.path.abspath(out_path)]
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError('evaluate.py báo lỗi')
    with open(out_path, encoding='utf-8') as f:
        return json.load(f)
