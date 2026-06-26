"""Probability forecast models for the M6 competition.

Each model implements ``predict(cv_df, cutoff, h)`` that returns
a DataFrame with columns: unique_id, ds, p1, p2, p3, p4, p5
where p1-p5 sum to 1 per row.
"""

from __future__ import annotations

from m6.models.adaptive import predict_adaptive
from m6.models.gaussian import predict_gaussian
from m6.models.historical import predict_historical
from m6.models.naive import predict_naive

__all__ = [
    "predict_adaptive",
    "predict_gaussian",
    "predict_historical",
    "predict_naive",
]
