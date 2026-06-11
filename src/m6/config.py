"""Centralised paths, seeds, and run-time settings.

Read from environment variables (with `.env` support) and exposed
as a frozen dataclass so every module gets the same view.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv(override=False)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    return int(raw) if raw is not None and raw != "" else default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    return float(raw) if raw is not None and raw != "" else default


def _env_path(key: str, default: Path) -> Path:
    raw = os.getenv(key)
    return Path(raw).expanduser() if raw else default


@dataclass(frozen=True)
class Settings:
    seed: int = field(default_factory=lambda: _env_int("M6_SEED", 42))
    horizon: int = field(default_factory=lambda: _env_int("M6_HORIZON", 20))
    n_windows: int = field(default_factory=lambda: _env_int("M6_N_WINDOWS", 6))
    n_assets: int = field(default_factory=lambda: _env_int("M6_N_ASSETS", -1))
    data_dir: Path = field(default_factory=lambda: _env_path("M6_DATA_DIR", REPO_ROOT / "data"))
    start_date: str = field(default_factory=lambda: os.getenv("M6_START_DATE", "2015-01-01"))
    end_date: str = field(default_factory=lambda: os.getenv("M6_END_DATE", "2023-02-28"))
    covariance_shrinkage: float = field(default_factory=lambda: _env_float("M6_COV_SHRINKAGE", 0.3))
    n_monte_carlo: int = field(default_factory=lambda: _env_int("M6_N_MC", 100_000))
    ewma_halflife: int = field(default_factory=lambda: _env_int("M6_EWMA_HALFLIFE", 63))
    mom_window: int = field(default_factory=lambda: _env_int("M6_MOM_WINDOW", 20))
    mom_strength: float = field(default_factory=lambda: _env_float("M6_MOM_STRENGTH", 0.05))
    tilt_strength: float = field(default_factory=lambda: _env_float("M6_TILT_STRENGTH", 0.03))
    calib_strength: float = field(default_factory=lambda: _env_float("M6_CALIB_STRENGTH", 0.15))

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "m6"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def artifacts_dir(self) -> Path:
        return REPO_ROOT / "artifacts"

    @property
    def forecasts_dir(self) -> Path:
        return REPO_ROOT / "forecasts"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.processed_dir, self.artifacts_dir, self.forecasts_dir):
            p.mkdir(parents=True, exist_ok=True)


SETTINGS = Settings()


def set_global_seed(seed: int | None = None) -> int:
    s = SETTINGS.seed if seed is None else seed
    random.seed(s)
    np.random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    return s
