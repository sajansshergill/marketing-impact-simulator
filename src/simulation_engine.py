from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from .preprocessing import build_features, TransformConfig
from .mmm_model import predict


@dataclass
class Scenario:
    """
    Counterfactual levers that map to "22 laws" style decisions.
    """
    focus_delta: float = 0.0          # e.g., +0.15 = more focused
    leader_override: int | None = None # 1 or 0
    budget_shift: Dict[str, float] = None
    # budget_shift example: {"tv_spend": -0.10, "search_spend": +0.10} (10% reallocation)

    def __post_init__(self):
        if self.budget_shift is None:
            self.budget_shift = {}


def apply_scenario(df_raw: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
    df = df_raw.copy()

    # Focus shift (Law of Focus)
    df["focus_score"] = np.clip(df["focus_score"] + scenario.focus_delta, 0.0, 1.0)

    # Leadership / challenger (Law of Leadership)
    if scenario.leader_override is not None:
        df["category_leader"] = int(scenario.leader_override)

    # Budget reallocations
    for ch, pct in scenario.budget_shift.items():
        if ch not in df.columns:
            raise ValueError(f"Unknown channel: {ch}")
        df[ch] = np.clip(df[ch] * (1.0 + pct), 0.0, None)

    return df


def simulate_revenue_uplift(
    df_raw: pd.DataFrame,
    model,
    x_cols: List[str],
    transform_cfg: TransformConfig,
    scenario: Scenario
) -> pd.DataFrame:
    """
    Returns baseline vs scenario predictions and uplift.
    """
    base_feat = build_features(df_raw, transform_cfg)
    base_pred = predict(model, base_feat, x_cols)

    scen_raw = apply_scenario(df_raw, scenario)
    scen_feat = build_features(scen_raw, transform_cfg)
    scen_pred = predict(model, scen_feat, x_cols)

    out = pd.DataFrame({
        "week": df_raw["week"],
        "baseline_pred": base_pred,
        "scenario_pred": scen_pred,
    })
    out["uplift"] = out["scenario_pred"] - out["baseline_pred"]
    out["uplift_pct"] = np.where(out["baseline_pred"] != 0, out["uplift"] / out["baseline_pred"], 0.0)
    return out