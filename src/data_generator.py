from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd


@dataclass
class GenConfig:
    n_weeks: int = 156  # 3 years weekly
    seed: int = 7
    base_revenue: float = 200_000.0

    # Channel spend ranges
    tv_min: float = 10_000.0
    tv_max: float = 80_000.0
    search_min: float = 5_000.0
    search_max: float = 50_000.0
    social_min: float = 5_000.0
    social_max: float = 40_000.0
    display_min: float = 2_000.0
    display_max: float = 25_000.0

    # Strategic indicators
    category_leader_prob: float = 0.5
    focus_mean: float = 0.6
    focus_std: float = 0.15

    # Ground-truth effects (for synthetic realism)
    tv_beta: float = 1.8
    search_beta: float = 2.4
    social_beta: float = 1.6
    display_beta: float = 1.1
    brand_awareness_beta: float = 1200.0
    focus_beta: float = 50_000.0
    leader_beta: float = 35_000.0

    noise_std: float = 25_000.0


def _seasonality(week_idx: np.ndarray) -> np.ndarray:
    # yearly seasonality + small quarterly wobble
    yearly = np.sin(2 * np.pi * week_idx / 52.0)
    quarterly = 0.5 * np.sin(2 * np.pi * week_idx / 13.0)
    return yearly + quarterly


def _holidays(n: int, rng: np.random.Generator) -> np.ndarray:
    # Random "holiday spikes" (e.g., promo weeks)
    spikes = np.zeros(n)
    spike_weeks = rng.choice(np.arange(n), size=max(3, n // 26), replace=False)
    spikes[spike_weeks] = rng.uniform(0.8, 1.5, size=len(spike_weeks))
    return spikes


def generate_synthetic_mmm_data(cfg: GenConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    weeks = pd.date_range("2023-01-01", periods=cfg.n_weeks, freq="W-SUN")
    t = np.arange(cfg.n_weeks)

    # Spend (random but with mild autocorrelation)
    def ar_spend(minv: float, maxv: float, rho: float = 0.6) -> np.ndarray:
        x = rng.uniform(minv, maxv, size=cfg.n_weeks)
        for i in range(1, cfg.n_weeks):
            x[i] = rho * x[i - 1] + (1 - rho) * x[i]
        return x

    tv = ar_spend(cfg.tv_min, cfg.tv_max)
    search = ar_spend(cfg.search_min, cfg.search_max)
    social = ar_spend(cfg.social_min, cfg.social_max)
    display = ar_spend(cfg.display_min, cfg.display_max)

    # Strategy indicators
    category_leader = int(rng.uniform(0, 1) < cfg.category_leader_prob)
    focus_score = np.clip(rng.normal(cfg.focus_mean, cfg.focus_std, size=cfg.n_weeks), 0.0, 1.0)

    # Controls
    seas = _seasonality(t)
    holiday = _holidays(cfg.n_weeks, rng)

    # Brand awareness is partially driven by upper-funnel spend & focus
    # (This is synthetic "perception" proxy)
    brand_awareness = (
        40 + 0.00025 * tv + 0.00015 * social + 8 * focus_score + 5 * seas + 10 * holiday
        + rng.normal(0, 2.5, size=cfg.n_weeks)
    )
    brand_awareness = np.clip(brand_awareness, 0, None)

    df = pd.DataFrame(
        {
            "week": weeks,
            "tv_spend": tv,
            "search_spend": search,
            "social_spend": social,
            "display_spend": display,
            "brand_awareness": brand_awareness,
            "focus_score": focus_score,
            "category_leader": category_leader,
            "seasonality": seas,
            "holiday_spike": holiday,
        }
    )

    # Revenue ground truth (nonlinear-ish, but keep it estimable)
    # We apply diminishing returns via sqrt on spend and add the strategic effects.
    revenue = (
        cfg.base_revenue
        + cfg.tv_beta * np.sqrt(tv)
        + cfg.search_beta * np.sqrt(search)
        + cfg.social_beta * np.sqrt(social)
        + cfg.display_beta * np.sqrt(display)
        + cfg.brand_awareness_beta * brand_awareness
        + cfg.focus_beta * focus_score
        + cfg.leader_beta * category_leader
        + 15_000.0 * df["seasonality"].to_numpy()
        + 20_000.0 * df["holiday_spike"].to_numpy()
        + rng.normal(0, cfg.noise_std, size=cfg.n_weeks)
    )

    df["revenue"] = np.clip(revenue, 0, None)
    return df


def save_dataset(df: pd.DataFrame, out_csv: str = "data/processed/mmm_synth.csv") -> None:
    import os

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    cfg = GenConfig()
    df = generate_synthetic_mmm_data(cfg)
    save_dataset(df)
    print(df.head())
    print(f"Saved -> data/processed/mmm_synth.csv  (rows={len(df)})")