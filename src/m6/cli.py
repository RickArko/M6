"""Typer CLI: ``m6 download | prep | cv | score | forecast``."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import typer

from m6.config import SETTINGS, set_global_seed
from m6.evaluation import PUBLISHED_RPS as PUB_RPS
from m6.logging import logger

app = typer.Typer(add_completion=False, help="M6 forecasting uncertainty toolkit.")

_CV_KEY_COLS = ("unique_id", "ds", "cutoff", "y", "y_return")


def _load_cv_files(model_names: list[str], artifacts_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    if not model_names:
        raise typer.BadParameter("Pass at least one --model.")
    frames: list[tuple[str, pd.DataFrame]] = []
    for m in model_names:
        path = artifacts_dir / f"cv_{m}.parquet"
        if not path.exists():
            raise typer.BadParameter(f"CV artifact not found: {path}")
        df = pd.read_parquet(path)
        df["ds"] = pd.to_datetime(df["ds"])
        df["cutoff"] = pd.to_datetime(df["cutoff"])
        frames.append((m, df))

    base = frames[0][1][list(_CV_KEY_COLS)].copy()
    prob_cols: list[str] = []
    for _, df in frames:
        for c in df.columns:
            if c in _CV_KEY_COLS or c in prob_cols:
                continue
            prob_cols.append(c)
    merged = base
    for _, df in frames:
        extras = [c for c in df.columns if c not in _CV_KEY_COLS]
        merged = merged.merge(
            df[["unique_id", "ds", "cutoff", *extras]],
            on=["unique_id", "ds", "cutoff"],
            how="inner",
        )
    if merged.empty:
        raise typer.BadParameter(
            "Merged CV frame is empty — the CV files don't share (unique_id, ds, cutoff) keys."
        )
    model_names_found = sorted(set(c.rsplit("_p", 1)[0] for c in prob_cols if "_p" in c))
    return merged, model_names_found


@app.command()
def download() -> None:
    """Download M6 price data from Yahoo Finance."""
    from m6.data import load_prices

    SETTINGS.ensure_dirs()
    t0 = time.time()
    df = load_prices(cache_dir=SETTINGS.raw_dir)
    logger.info(f"Downloaded {df['unique_id'].nunique():,d} assets in {time.time() - t0:.1f}s.")


@app.command()
def prep(
    out: Path = typer.Option(None, help="Output parquet path (default: data/processed/long.parquet)."),
) -> None:
    """Build the long-format training frame with forward returns and quintiles."""
    from m6.data import build_long_frame

    set_global_seed()
    SETTINGS.ensure_dirs()
    t0 = time.time()
    df = build_long_frame(cache_dir=SETTINGS.raw_dir)
    out_path = out or (SETTINGS.processed_dir / "long.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info(f"Wrote {out_path} ({len(df):,d} rows) in {time.time() - t0:.1f}s.")


@app.command()
def cv(
    model: str = typer.Argument("naive", help="One of: naive, historical, gaussian, ensemble."),
    horizon: int = typer.Option(SETTINGS.horizon),
    n_windows: int = typer.Option(SETTINGS.n_windows),
    long_path: Path = typer.Option(None, help="Path to processed long parquet."),
) -> None:
    """Run reproducible rolling-origin cross-validation."""
    from m6.cv import ensemble_cv, gaussian_cv, historical_cv, naive_cv
    from m6.evaluation import rps_for_models

    long_path = long_path or SETTINGS.processed_dir / "long.parquet"
    df = pd.read_parquet(long_path)
    df["ds"] = pd.to_datetime(df["ds"])

    model_map = {
        "naive": naive_cv,
        "historical": historical_cv,
        "gaussian": gaussian_cv,
        "ensemble": ensemble_cv,
    }
    cv_fn = model_map.get(model)
    if cv_fn is None:
        raise typer.BadParameter(
            f"Unknown model: {model!r}. Use 'naive', 'historical', 'gaussian', or 'ensemble'."
        )

    cv_df = cv_fn(df, h=horizon, n_windows=n_windows)
    truth = cv_df[["unique_id", "ds", "y"]].copy()
    scores = rps_for_models(truth, cv_df)
    logger.info(f"RPS by model:\n{scores.to_string()}")

    out = SETTINGS.artifacts_dir / f"cv_{model}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv_df.to_parquet(out, index=False)
    logger.info(f"Wrote {out} ({len(cv_df):,d} rows)")

    # Compare to published benchmarks
    logger.info("Published M6 benchmarks:")
    for name, rps_val in PUB_RPS.items():
        delta = ""
        for m in scores.index:
            delta = f"  (yours {m}: {scores[m]:.5f}, Δ={scores[m] - rps_val:+.5f})"
        logger.info(f"  {name}: {rps_val:.5f}{delta}")


@app.command()
def score(
    models: list[str] = typer.Option(
        ...,
        "--model",
        "-m",
        help="Artifact base name (reads artifacts/cv_<name>.parquet). Repeat for multiple.",
    ),
    out: Path = typer.Option(Path("reports"), help="Output directory for figures + report."),
    long_path: Path = typer.Option(None, help="Path to processed long parquet."),
    run_id: str = typer.Option("latest", help="Run id stamped into the report header."),
    bootstrap_iter: int = typer.Option(1000, help="Bootstrap resamples for significance matrix."),
    no_report: bool = typer.Option(False, help="Skip report stitching; only write metrics + figures."),
) -> None:
    """Score CV artifacts; emit metrics, figures, and a Markdown + HTML report."""
    from m6.evaluation import compute_components
    from m6.scoring import (
        ScoringInputs,
        bias_variance_decomposition,
        headline_scores,
        paired_bootstrap_pvalues,
        per_fold_scores,
        per_segment_scores,
    )

    set_global_seed()
    long_path = long_path or SETTINGS.processed_dir / "long.parquet"
    if not long_path.exists():
        raise typer.BadParameter(f"Training long-frame not found at {long_path}; run `m6 prep` first.")

    logger.info(f"score: loading {long_path}")
    train = pd.read_parquet(long_path)
    train["ds"] = pd.to_datetime(train["ds"])

    cv_df, model_names = _load_cv_files(models, SETTINGS.artifacts_dir)
    logger.info(
        f"score: merged {len(models)} CV file(s) → {len(cv_df):,d} rows "
        f"× {len(model_names)} models: {model_names}"
    )

    components = compute_components(train)
    inp = ScoringInputs(
        cv_df=cv_df,
        train=train,
        components=components,
        models=model_names,
    )

    logger.info("score: computing metrics …")
    headline = headline_scores(inp)
    per_fold = per_fold_scores(inp)
    segment_frames = {cut: per_segment_scores(inp, cut) for cut in ("category",) if cut in cv_df.columns}
    bv = bias_variance_decomposition(inp)
    pvalues = (
        paired_bootstrap_pvalues(inp, n_iter=bootstrap_iter, seed=SETTINGS.seed)
        if len(model_names) > 1
        else None
    )

    out.mkdir(parents=True, exist_ok=True)
    metrics_dir = out / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    headline.to_csv(metrics_dir / "headline.csv", index=False)
    headline.to_parquet(metrics_dir / "headline.parquet", index=False)
    per_fold.to_parquet(metrics_dir / "per_fold.parquet", index=False)
    for cut, df_s in segment_frames.items():
        if not df_s.empty:
            df_s.to_parquet(metrics_dir / f"per_segment_{cut}.parquet", index=False)
    bv.to_parquet(metrics_dir / "bias_variance.parquet", index=False)
    if pvalues is not None:
        pvalues.to_parquet(metrics_dir / "pvalues.parquet")
    logger.info(f"score: wrote metrics → {metrics_dir}")

    # Compare with published benchmarks
    logger.info("Comparison to published M6 benchmarks (RPS, lower is better):")
    logger.info("  Naive benchmark:      0.16000")
    for _, row in headline.iterrows():
        rps_val = row["rps"]
        beat_naive = "✓" if rps_val < 0.16 else "✗"
        logger.info(f"  {row['model']:20s}: {rps_val:.5f}  ({beat_naive} beats naive)")

    if no_report:
        logger.info("score: --no-report set; skipping report.")
        return

    # Simple markdown report
    md = [
        f"# M6 Scoring Report — {run_id}",
        "",
        "## Headline Scores (RPS, lower is better)",
        "",
        "| Model | RPS | Accuracy | Log Loss | Brier |",
        "|-------|-----|----------|----------|-------|",
    ]
    for _, row in headline.iterrows():
        md.append(
            f"| {row['model']} | {row['rps']:.5f} | {row['accuracy']:.4f} | "
            f"{row['log_loss']:.4f} | {row['brier']:.4f} |"
        )
    md.extend(
        [
            "",
            "### Published Benchmarks",
            "",
        ]
    )
    for name, rps_val in PUB_RPS.items():
        md.append(f"- **{name}**: {rps_val:.5f}")
    md.append("")
    md.append("_Generated by `m6 score`._")
    md.append("")

    report_md = out / "report.md"
    report_md.write_text("\n".join(md) + "\n")
    logger.info(f"score: wrote {report_md}")


@app.command()
def viz(
    out_dir: Path = typer.Option(None, help="Output directory (default: assets/)."),
    gif: bool = typer.Option(True, "--gif/--no-gif", help="Render animated GIF."),
    gif_fps: int = typer.Option(12, "--gif-fps", help="GIF frame rate."),
    gif_duration: float = typer.Option(8.0, "--gif-duration", help="GIF loop duration (seconds)."),
) -> None:
    """Render pipeline visualisation (animated GIF)."""
    from m6.viz import render_pipeline_viz

    paths = render_pipeline_viz(
        out_dir=out_dir,
        gif=gif,
        gif_fps=gif_fps,
        gif_duration=gif_duration,
    )
    for kind, p in paths.items():
        logger.info(f"viz: {kind} -> {p}")


@app.command()
def forecast(
    model: str = typer.Argument("naive", help="One of: naive, historical, gaussian."),
    horizon: int = typer.Option(SETTINGS.horizon),
    long_path: Path = typer.Option(None),
) -> None:
    """Train on all available data and emit a future forecast."""
    from m6.models.gaussian import predict_gaussian
    from m6.models.historical import predict_historical
    from m6.models.naive import predict_naive

    long_path = long_path or SETTINGS.processed_dir / "long.parquet"
    logger.info(f"forecast {model}: loading {long_path}")
    df = pd.read_parquet(long_path)
    df["ds"] = pd.to_datetime(df["ds"])

    cutoff = df["ds"].max()

    model_map = {
        "naive": predict_naive,
        "historical": predict_historical,
        "gaussian": predict_gaussian,
    }
    fn = model_map.get(model)
    if fn is None:
        raise typer.BadParameter(f"Unknown model: {model!r}.")

    out_df = fn(df, cutoff, horizon)
    out = SETTINGS.forecasts_dir / f"forecast_{model}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)
    logger.info(f"Wrote {out} ({len(out_df):,d} rows).")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
