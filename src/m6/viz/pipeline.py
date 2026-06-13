"""Animated GIF renderer for M6 pipeline visualisation.

Reads the scoring outputs (headline + per-fold RPS) and produces a
dark-theme animated GIF showing model comparison and CV progression.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from m6.config import REPO_ROOT
from m6.evaluation import PUBLISHED_RPS as PUB_RPS
from m6.logging import logger

_COLORS = {
    "historical": "#58a6ff",
    "gaussian": "#f778ba",
    "naive": "#d29922",
}
_BG = "#0d1117"
_TEXT = "#f0f6fc"
_MUTED = "#8b949e"
_GRID = "#21262d"
_AXIS = "#30363d"
_GREEN = "#3fb950"
_NAIVE_RPS = 0.16


def _ease(t: float) -> float:
    """Smoothstep easing."""
    return t * t * (3 - 2 * t)


def render_gif(
    headline: pd.DataFrame,
    per_fold: pd.DataFrame,
    out_path: Path,
    *,
    fps: int = 12,
    duration: float = 8.0,
    width: int = 800,
    height: int = 450,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    models = headline.sort_values("rps")["model"].tolist()
    model_colors = [_COLORS.get(m, _MUTED) for m in models]

    pf = per_fold.copy()
    pf["cutoff_dt"] = pd.to_datetime(pf["cutoff"])
    cutoffs = sorted(pf["cutoff_dt"].unique())
    cutoff_labels = [c.strftime("%b") for c in cutoffs]

    hl = headline.set_index("model")

    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor=_BG)

    fig.text(
        0.5,
        0.94,
        "M6 Forecasting Uncertainty",
        color=_TEXT,
        fontsize=15,
        weight=700,
        ha="center",
        va="center",
    )
    fig.text(
        0.5,
        0.89,
        "Probabilistic Quintile Forecasting \u00b7 100 Financial Assets",
        color=_MUTED,
        fontsize=9,
        ha="center",
        va="center",
    )

    # Left axes: RPS bar chart
    ax_bar = fig.add_axes((0.07, 0.30, 0.40, 0.52))
    ax_bar.set_facecolor(_BG)
    ax_bar.set_title("RPS by Model (lower is better)", color=_MUTED, fontsize=10, loc="left", pad=6)
    ax_bar.tick_params(colors=_MUTED, labelsize=8)
    for spine in ax_bar.spines.values():
        spine.set_color(_AXIS)
    ax_bar.grid(True, axis="y", color=_GRID, alpha=0.6, linestyle="--", linewidth=0.6)
    ax_bar.set_xlim(-0.6, len(models) - 0.4)
    rps_vals = [hl.loc[m, "rps"] for m in models]
    rps_max = max([*rps_vals, _NAIVE_RPS]) * 1.25
    ax_bar.set_ylim(0, rps_max)

    ax_bar.set_xticks(range(len(models)))
    ax_bar.set_xticklabels([m.capitalize() for m in models])
    bars = []
    for i, _c in enumerate(model_colors):
        b = ax_bar.bar(i, 0, color=_c, width=0.55, edgecolor=_c, linewidth=0.5, alpha=0.9)
        bars.append(b)

    naive_line = ax_bar.axhline(_NAIVE_RPS, color=_MUTED, linestyle="--", linewidth=1.2, visible=False)
    naive_label = ax_bar.text(
        len(models) - 0.55,
        _NAIVE_RPS * 1.03,
        "Naive baseline 0.16000",
        color=_MUTED,
        fontsize=7,
        ha="right",
        va="bottom",
        visible=False,
    )

    val_labels = []
    for i, (m, _c) in enumerate(zip(models, model_colors, strict=True)):
        v = hl.loc[m, "rps"]
        lbl = ax_bar.text(
            i,
            0,
            f"{v:.5f}",
            color=_c,
            fontsize=8,
            ha="center",
            va="bottom",
            weight=600,
            visible=False,
        )
        val_labels.append(lbl)

    acc_labels = []
    for i, m in enumerate(models):
        acc = hl.loc[m, "accuracy"]
        lbl = ax_bar.text(
            i,
            -rps_max * 0.04,
            f"Acc: {acc:.1%}",
            color=_MUTED,
            fontsize=6.5,
            ha="center",
            va="top",
            visible=False,
        )
        acc_labels.append(lbl)

    # Right axes: Per-fold line chart
    ax_line = fig.add_axes((0.57, 0.30, 0.38, 0.52))
    ax_line.set_facecolor(_BG)
    ax_line.set_title("RPS Across CV Folds", color=_MUTED, fontsize=10, loc="left", pad=6)
    ax_line.tick_params(colors=_MUTED, labelsize=8)
    for spine in ax_line.spines.values():
        spine.set_color(_AXIS)
    ax_line.grid(True, color=_GRID, alpha=0.6, linestyle="--", linewidth=0.6)
    ax_line.set_xlim(-0.1, len(cutoffs) - 0.9)
    y_min = max(0.14, pf["rps"].min() * 0.98)
    y_max = pf["rps"].max() * 1.05
    ax_line.set_ylim(y_min, y_max)
    ax_line.set_xticks(range(len(cutoffs)))
    ax_line.set_xticklabels(cutoff_labels)

    lines = []
    for m, _c in zip(models, model_colors, strict=True):
        (ln,) = ax_line.plot([], [], color=_c, linewidth=2.0, marker="o", markersize=5, label=m.capitalize())
        lines.append(ln)

    naive_line_r = ax_line.axhline(_NAIVE_RPS, color=_MUTED, linestyle="--", linewidth=1.0, visible=False)
    ax_line.legend(
        loc="upper right",
        frameon=False,
        fontsize=7,
        labelcolor=[_COLORS.get(m, _MUTED) for m in models],
    )

    # Bottom annotations
    anno = fig.text(
        0.07,
        0.12,
        "",
        color=_GREEN,
        fontsize=10,
        weight=600,
        family="monospace",
        visible=False,
    )
    sub_anno = fig.text(
        0.07,
        0.07,
        "",
        color=_MUTED,
        fontsize=8,
        family="monospace",
        visible=False,
    )
    bench_text = fig.text(
        0.57,
        0.12,
        "",
        color=_MUTED,
        fontsize=7.5,
        family="monospace",
        visible=False,
        va="top",
    )

    n_frames = max(round(fps * duration), 4)

    def update(frame: int):
        p = frame / (n_frames - 1) if n_frames > 1 else 1.0

        # Phase 2 (0.15-0.50): Bars build
        bar_frac = _ease(max(0.0, min(1.0, (p - 0.15) / 0.35)))
        n_bars_show = max(round(len(models) * bar_frac), 0)
        for i, b in enumerate(bars):
            h = rps_vals[i] if i < n_bars_show else 0
            b[0].set_height(h)

        if bar_frac > 0:
            naive_line.set_visible(True)
            naive_label.set_visible(True)

        for i, (_m, lbl) in enumerate(zip(models, val_labels, strict=True)):
            if i < n_bars_show and bar_frac > 0.8:
                lbl.set_visible(True)
                lbl.set_y(rps_vals[i] * 1.03)
            else:
                lbl.set_visible(False)
            if i < n_bars_show and bar_frac > 0.95:
                acc_labels[i].set_visible(True)
            else:
                acc_labels[i].set_visible(False)

        # Phase 3 (0.50-0.80): Per-fold lines draw
        line_frac = _ease(max(0.0, min(1.0, (p - 0.50) / 0.30)))
        n_cutoffs_show = max(round(len(cutoffs) * line_frac), 0)

        for i, m in enumerate(models):
            mdf = pf[pf["model"] == m].sort_values("cutoff_dt")
            vals = mdf["rps"].to_numpy()
            xs = np.arange(len(vals))
            n_show = min(n_cutoffs_show, len(vals))
            lines[i].set_data(xs[:n_show], vals[:n_show])

        if line_frac > 0:
            naive_line_r.set_visible(True)

        # Phase 4 (0.75-1.00): Annotations
        anno_frac = _ease(max(0.0, min(1.0, (p - 0.75) / 0.15)))
        if anno_frac > 0:
            best = models[0]
            best_rps = hl.loc[best, "rps"]
            if best_rps < _NAIVE_RPS:
                anno.set_text(f"\u2713 {best.capitalize()} beats naive benchmark")
            else:
                anno.set_text("No model beats naive benchmark")
            anno.set_visible(True)
            anno.set_alpha(min(1.0, anno_frac * 2))

            acc_best = hl.loc[best, "accuracy"]
            sub_anno.set_text(
                f"Best: {best.capitalize()} \u00b7 RPS {best_rps:.5f} \u00b7 "
                f"Accuracy {acc_best:.1%} \u00b7 "
                f"+{((acc_best - 0.2) / 0.2 * 100):.0f}% vs random"
            )
            sub_anno.set_visible(True)
            sub_anno.set_alpha(min(1.0, anno_frac * 2))

        bench_frac = _ease(max(0.0, min(1.0, (p - 0.82) / 0.12)))
        if bench_frac > 0:
            pub = PUB_RPS
            lines_txt = [
                "Published benchmarks:",
                f"  Dan (1st)      {pub['Dan (1st)']:.5f}",
                f"  FinQBoost (2nd) {pub['FinQBoost (2nd)']:.5f}",
                f"  SebastianR (3rd) {pub['SebastianR (3rd)']:.5f}",
            ]
            best_rps_val = hl.loc[models[0], "rps"]
            lines_txt.append("")
            lines_txt.append(f"  {models[0].capitalize()} (ours)  {best_rps_val:.5f}")
            delta = best_rps_val - pub["Dan (1st)"]
            sign = "+" if delta > 0 else ""
            lines_txt.append(f"  \u0394 to 1st: {sign}{delta:.5f}")
            bench_text.set_text("\n".join(lines_txt))
            bench_text.set_visible(True)
            bench_text.set_alpha(min(1.0, bench_frac * 2))

        return (
            bars
            + lines
            + [naive_line, naive_label, naive_line_r]
            + val_labels
            + acc_labels
            + [anno, sub_anno, bench_text]
        )

    anim = FuncAnimation(fig, update, frames=n_frames, interval=int(1000 / fps), blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    anim.save(str(out_path), writer=writer, dpi=100)
    plt.close(fig)
    return out_path


def render_pipeline_viz(
    *,
    headline_path: Path | None = None,
    per_fold_path: Path | None = None,
    out_dir: Path | None = None,
    gif: bool = True,
    gif_fps: int = 12,
    gif_duration: float = 8.0,
) -> dict[str, Path]:
    metrics_dir = REPO_ROOT / "reports" / "metrics"
    headline_path = headline_path or metrics_dir / "headline.csv"
    per_fold_path = per_fold_path or metrics_dir / "per_fold.parquet"

    if not headline_path.exists():
        raise FileNotFoundError(f"Headline scores not found: {headline_path}")
    if not per_fold_path.exists():
        raise FileNotFoundError(f"Per-fold scores not found: {per_fold_path}")

    headline = pd.read_csv(headline_path)
    per_fold = pd.read_parquet(per_fold_path)

    out_dir = out_dir or REPO_ROOT / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    if gif:
        gif_path = out_dir / "pipeline.gif"
        render_gif(headline, per_fold, gif_path, fps=gif_fps, duration=gif_duration)
        logger.info(f"viz: wrote {gif_path} ({gif_path.stat().st_size / 1024:.1f} KB)")
        paths["gif"] = gif_path

    return paths
