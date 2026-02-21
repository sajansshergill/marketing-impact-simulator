from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple
import pandas as pd
import statsmodels.api as sm


@dataclass
class MMMSpec:
    target: str = "revenue"

    # Use saturated transforms for robustness
    media_terms: List[str] = None
    extra_terms: List[str] = None

    def __post_init__(self):
        if self.media_terms is None:
            self.media_terms = [
                "tv_spend_sat",
                "search_spend_sat",
                "social_spend_sat",
                "display_spend_sat",
            ]
        if self.extra_terms is None:
            self.extra_terms = [
                "brand_awareness_lag1",
                "focus_score",
                "category_leader",
                "seasonality",
                "holiday_spike",
            ]


def fit_ols_mmm(df: pd.DataFrame, spec: MMMSpec):
    X_cols = spec.media_terms + spec.extra_terms
    X = df[X_cols].copy()
    X = sm.add_constant(X)
    y = df[spec.target].astype(float)

    model = sm.OLS(y, X).fit(cov_type="HC3")  # robust SE
    return model, X_cols


def predict(model, df: pd.DataFrame, x_cols: List[str]) -> pd.Series:
    X = sm.add_constant(df[x_cols].copy(), has_constant="add")
    return model.predict(X)


def contribution_decomposition(model, df: pd.DataFrame, x_cols: List[str]) -> pd.DataFrame:
    """
    Returns per-row contributions for each term in x_cols plus intercept.
    Contribution = beta_j * x_j
    """
    params = model.params
    X = sm.add_constant(df[x_cols].copy(), has_constant="add")

    contrib = pd.DataFrame(index=df.index)
    for col in X.columns:
        contrib[col] = params[col] * X[col]
    contrib["predicted"] = contrib.sum(axis=1)
    return contrib