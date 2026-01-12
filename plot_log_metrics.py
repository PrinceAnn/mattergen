#!/usr/bin/env python3
"""Plot convergence curves from training logs via regex parsing.

Supports logs that contain lines like:
  epoch=1 val_loss=0.0344 val_mae=0.1395 best_val_mae=0.1395

Usage:
  python plot_log_metrics.py mp20_distill_gemnet_formation_energy_per_atom.log
  python plot_log_metrics.py run1.log run2.log --out convergence.png

"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regex-parse metrics from logs and plot line charts")
    p.add_argument("logs", nargs="+", help="One or more log files")
    p.add_argument("--out", type=str, default="convergence.png", help="Output image path (.png recommended)")
    p.add_argument(
        "--metrics",
        type=str,
        default="val_loss,val_mae,best_val_mae",
        help="Comma-separated metrics to plot if present",
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title (default: inferred from first log name)",
    )
    p.add_argument(
        "--max-epoch",
        type=int,
        default=None,
        help="Only plot epochs <= this value (e.g., 100)",
    )
    p.add_argument(
        "--epoch-regex",
        type=str,
        default=r"\bepoch\s*=\s*(\d+)\b",
        help="Regex with one capture group for epoch integer",
    )
    p.add_argument(
        "--kv-regex",
        type=str,
        default=r"\b([A-Za-z][A-Za-z0-9_]*)\s*=\s*([+-]?(?:\d+\.\d+|\d+)(?:[eE][+-]?\d+)?)\b",
        help="Regex with two capture groups: key and float value",
    )
    p.add_argument("--dpi", type=int, default=160)
    return p.parse_args()


def parse_log_text(text: str, epoch_re: re.Pattern, kv_re: re.Pattern) -> Dict[int, Dict[str, float]]:
    """Return metrics indexed by epoch."""
    out: Dict[int, Dict[str, float]] = {}

    for line in text.splitlines():
        m_epoch = epoch_re.search(line)
        if not m_epoch:
            continue
        try:
            epoch = int(m_epoch.group(1))
        except Exception:
            continue

        kvs = kv_re.findall(line)
        if not kvs:
            continue

        d = out.setdefault(epoch, {})
        for k, v in kvs:
            if k == "epoch":
                continue
            try:
                d[k] = float(v)
            except Exception:
                continue

    return out


def to_series(
    metrics_by_epoch: Dict[int, Dict[str, float]],
    metric: str,
    max_epoch: int | None = None,
) -> Tuple[List[int], List[float]]:
    epochs = sorted(metrics_by_epoch.keys())
    xs: List[int] = []
    ys: List[float] = []
    for e in epochs:
        if max_epoch is not None and e > max_epoch:
            continue
        d = metrics_by_epoch[e]
        if metric in d:
            xs.append(e)
            ys.append(d[metric])
    return xs, ys


def main() -> None:
    args = _parse_args()

    epoch_re = re.compile(args.epoch_regex)
    kv_re = re.compile(args.kv_regex)

    log_paths = [Path(p) for p in args.logs]
    for p in log_paths:
        if not p.exists():
            raise FileNotFoundError(p)

    metrics_wanted = [m.strip() for m in str(args.metrics).split(",") if m.strip()]
    if not metrics_wanted:
        raise ValueError("--metrics must be non-empty")

    # Lazy import so environments without plotting can still parse.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    any_plotted = False

    # If multiple logs are provided, we plot each metric for each log.
    single_metric = len(metrics_wanted) == 1
    for log_path in log_paths:
        text = log_path.read_text(errors="ignore")
        by_epoch = parse_log_text(text, epoch_re=epoch_re, kv_re=kv_re)
        if not by_epoch:
            raise RuntimeError(f"No epoch lines matched in {log_path}. Try adjusting --epoch-regex/--kv-regex")

        for metric in metrics_wanted:
            xs, ys = to_series(by_epoch, metric, max_epoch=args.max_epoch)
            if not xs:
                continue
            if len(log_paths) > 1 and single_metric:
                label = log_path.stem
            else:
                label = f"{log_path.stem}:{metric}" if len(log_paths) > 1 else metric
            ax.plot(xs, ys, linewidth=1.8, label=label)
            any_plotted = True

    if not any_plotted:
        available: Dict[str, int] = {}
        # peek into the first file
        first = log_paths[0].read_text(errors="ignore")
        by_epoch = parse_log_text(first, epoch_re=epoch_re, kv_re=kv_re)
        for d in by_epoch.values():
            for k in d.keys():
                available[k] = available.get(k, 0) + 1
        keys = ", ".join(sorted(available.keys()))
        raise RuntimeError(f"None of requested metrics found. Available keys include: {keys}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metrics_wanted[0] if single_metric else "Metric")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(loc="best", fontsize=9)

    title = args.title
    if title is None:
        title = log_paths[0].stem if len(log_paths) == 1 else "Convergence"
    ax.set_title(title)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(args.dpi))
    print(f"Wrote plot to {out_path}")


if __name__ == "__main__":
    main()
