from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from .adstock import adstock
from .saturation import hill_saturation


@dataclass
class TransformConfig:
    adstock_decay: Dict[str, float] = None
    sat_alpha: Dict[str, float] = None
    sat_gamma: Dict[str, float] = None

    def __post_init__(self):
        if self.adstock_decay is None:
            self.adstock_decay = {
                "tv_spend": 0.65,
                "search_spend": 0.30,
                "social_spend": 0.45,
                "display_spend": 0.40,
            }
        if self.sat_alpha is None:
            self.sat_alpha = {
                "tv_spend": 50_000.0,
                "search_spend": 25_000.0,
                "social_spend": 20_000.0,
                "display_spend": 12_000.0,
            }
        if self.sat_gamma is None:
            self.sat_gamma = {k: 1.0 for k in self.adstock_decay.keys()}


def build_features(df: pd.DataFrame, cfg: TransformConfig) -> pd.DataFrame:
    out = df.copy()

    # Basic cleaning
    out = out.sort_values("week").reset_index(drop=True)

    # Lag brand metric to represent "Law of Perspective"
    out["brand_awareness_lag1"] = out["brand_awareness"].shift(1).bfill()

    # Adstock + saturation for channels
    for ch, decay in cfg.adstock_decay.items():
        a = adstock(out[ch].to_numpy(), decay=decay)
        s = hill_saturation(a, alpha=cfg.sat_alpha[ch], gamma=cfg.sat_gamma[ch])
        out[f"{ch}_adstock"] = a
        out[f"{ch}_sat"] = s

    return out


def train_test_split_time(df: pd.DataFrame, test_weeks: int = 26) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if test_weeks <= 0 or test_weeks >= len(df):
        raise ValueError("test_weeks must be between 1 and len(df)-1")
    train = df.iloc[:-test_weeks].copy()
    test = df.iloc[-test_weeks:].copy()
    return train, test