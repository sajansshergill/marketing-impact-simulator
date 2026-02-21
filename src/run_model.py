from __future__ import annotations

import os
import pandas as pd

from .data_generator import GenConfig, generate_synthetic_mmm_data, save_dataset
from .preprocessing import TransformConfig, build_features, train_test_split_time
from .mmm_model import MMMSpec, fit_ols_mmm, predict, contribution_decomposition


def main():
    # 1) Generate or load data
    csv_path = "data/processed/mmm_synth.csv"
    if not os.path.exists(csv_path):
        df_raw = generate_synthetic_mmm_data(GenConfig())
        save_dataset(df_raw, csv_path)
    else:
        df_raw = pd.read_csv(csv_path, parse_dates=["week"])

    # 2) Feature engineering
    tcfg = TransformConfig()
    df = build_features(df_raw, tcfg)

    # 3) Train/Test split (time-based)
    train, test = train_test_split_time(df, test_weeks=26)

    # 4) Fit OLS MMM
    spec = MMMSpec()
    model, x_cols = fit_ols_mmm(train, spec)

    # 5) Evaluate
    test_pred = predict(model, test, x_cols)
    test_out = test[["week", "revenue"]].copy()
    test_out["pred"] = test_pred
    test_out["error"] = test_out["revenue"] - test_out["pred"]
    mae = test_out["error"].abs().mean()

    # 6) Decomposition on full data
    contrib = contribution_decomposition(model, df, x_cols)
    contrib_out = pd.concat([df[["week", "revenue"]], contrib], axis=1)

    os.makedirs("outputs", exist_ok=True)
    test_out.to_csv("outputs/test_predictions.csv", index=False)
    contrib_out.to_csv("outputs/contributions.csv", index=False)

    print(model.summary())
    print(f"\nSaved: outputs/test_predictions.csv, outputs/contributions.csv")
    print(f"Test MAE (last 26 weeks): {mae:,.0f}")


if __name__ == "__main__":
    main()