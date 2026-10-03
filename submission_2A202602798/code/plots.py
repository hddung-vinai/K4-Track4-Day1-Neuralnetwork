# plots.py — ảnh từng thí nghiệm figures/<exp_id>.png và ảnh chồng figures/compare_<nhóm>.png
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt

LABELS = {'train_loss': 'train loss (eval mode)', 'val_loss': 'val loss', 'val_acc': 'val accuracy',
          'val_macro_f1': 'val macro-F1', 'grad_norm': 'grad_norm TB/epoch (trước clip)',
          'grad_norm_max': 'grad_norm max/epoch (trước clip)', 'epoch_time_s': 'thời gian/epoch (s)'}


def cfg_text(cfg: dict) -> str:
    hidden = '-'.join(str(h) for h in cfg['hidden'])
    clip = 'none' if cfg['clip_norm'] is None else cfg['clip_norm']
    parts = [f"{cfg['loss'].upper()}", f"{cfg['optimizer']} lr={cfg['lr']:g}", f"wd={cfg['weight_decay']:g}",
             f"batch={cfg['batch']}", f"ep={cfg['epochs']}", f"M={hidden}", f"drop={cfg['dropout']:g}",
             f"clip={clip}", f"{cfg['precision']}", f"init={cfg['init']}", f"seed={cfg['seed']}"]
    if cfg.get('scheduler'):
        parts.append(f"sched={cfg['scheduler']}")
    return ', '.join(parts)


def plot_run(result: dict, path: str, show: bool = False) -> None:
    # 3 ô: (1) train/val loss, (2) val acc + macro-F1, (3) grad_norm trước clip (TB và max mỗi epoch)
    cfg, h, s = result['cfg'], result['history'], result['summary']
    ep = h['epoch']
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    ax = axes[0]
    ax.plot(ep, h['train_loss'], 'o-', ms=3, label='train loss (eval mode, 50k mẫu cố định)')
    ax.plot(ep, h['val_loss'], 's-', ms=3, label='val loss')
    ax.set_title(f"Loss ({cfg['loss'].upper()})")
    ax.set_ylabel('loss')
    ax = axes[1]
    ax.plot(ep, h['val_acc'], 'o-', ms=3, label='val accuracy')
    ax.plot(ep, h['val_macro_f1'], 's-', ms=3, label='val macro-F1')
    ax.axhline(0.4876, color='gray', ls=':', lw=1, label='đoán lớp đa số (acc 0.4876)')
    ax.set_title('Val accuracy / macro-F1')
    ax.set_ylabel('giá trị')
    ax = axes[2]
    ax.plot(ep, h['grad_norm'], 'o-', ms=3, label='grad_norm trung bình/epoch')
    ax.plot(ep, h['grad_norm_max'], '^--', ms=3, alpha=0.7, label='grad_norm max/epoch')
    if cfg['clip_norm'] is not None:
        ax.axhline(cfg['clip_norm'], color='red', ls='--', lw=1, label=f"ngưỡng clip c={cfg['clip_norm']:g}")
    ax.set_yscale('log')
    ax.set_title('Chuẩn gradient toàn cục (đo TRƯỚC clip)')
    ax.set_ylabel('||g||_2 (log)')
    for ax in axes:
        ax.set_xlabel('epoch')
        if s.get('best_epoch'):
            ax.axvline(s['best_epoch'], color='green', ls=':', lw=1, label=f"best epoch = {s['best_epoch']}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    status = '  [PHÂN KỲ]' if s.get('diverged') else ''
    fig.suptitle(f"{cfg['exp_id']}{status}  |  {cfg_text(cfg)}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def plot_compare(results: list[dict], metric, path: str, title: str = '', show: bool = True) -> None:
    # Vẽ chồng một hay nhiều chỉ số (metric: str hoặc list[str]) của nhiều thí nghiệm, mỗi exp_id một đường
    metrics = [metric] if isinstance(metric, str) else list(metric)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.6 * len(metrics), 4.4), squeeze=False)
    colors = matplotlib.colormaps['tab10']
    for j, m in enumerate(metrics):
        ax = axes[0][j]
        for i, r in enumerate(results):
            ax.plot(r['history']['epoch'], r['history'][m], 'o-', ms=2.5, color=colors(i % 10),
                    label=r['cfg']['exp_id'])
        if m.startswith('grad_norm'):
            ax.set_yscale('log')
        ax.set_title(LABELS.get(m, m))
        ax.set_xlabel('epoch')
        ax.set_ylabel(m)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
