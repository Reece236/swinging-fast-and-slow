from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sfs_py.data_prep import load_baseballsavant_csv, add_basic_fields, recreate_squared_up, remove_partial_swings, build_intention_dataset
from sfs_py.intention_model import fit_intention_model, summarize_random_slopes
from sfs_py.causal_model import fit_causal_model


def main(data_dir: str, models_dir: str, out_dir: str) -> None:
    data_dir_p = Path(data_dir)
    models_dir_p = Path(models_dir)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_baseballsavant_csv(str(data_dir_p / "baseballsavant.csv"))

    # Build intention dataset
    intent = build_intention_dataset(df)

    # Fit intention models (can be slow)
    fit_bs = fit_intention_model(intent, swing_metric="bat_speed", draws=800, tune=600, chains=2)
    fit_sl = fit_intention_model(intent, swing_metric="swing_length", draws=800, tune=600, chains=2)

    # Summarize approaches
    approach_bs = summarize_random_slopes(fit_bs)
    approach_sl = summarize_random_slopes(fit_sl)
    approach = approach_bs.merge(approach_sl, on="batter_side_id", how="inner")

    approach.to_csv(out_dir_p / "approach_summary.csv", index=False)

    # Load model predictions saved in R pipeline (if available)
    # For full parity, you would re-train predpitchscore and xgboost models in Python;
    # here we assume the models/predictions are provided as in the repo README.
    pred_pitch = pd.read_csv(models_dir_p / "pred_outcome_pitch.csv") if (models_dir_p / "pred_outcome_pitch.csv").exists() else None
    if pred_pitch is None:
        print("Missing models/pred_outcome_pitch.csv. Skipping causal model fitting.")
        return

    # Merge approach into a causal dataset similar to scripts/estimate_models.R
    pred_pitch = add_basic_fields(pred_pitch)
    pred_pitch = pred_pitch.merge(approach, on="batter_side_id", how="inner")

    # Fit causal models
    causal = fit_causal_model(pred_pitch)
    # Statsmodels results can be saved via pickle if needed
    print(causal.fit_contact.summary())
    print(causal.fit_fair.summary())
    print(causal.fit_hit.summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/workspace/data")
    parser.add_argument("--models_dir", default="/workspace/models")
    parser.add_argument("--out_dir", default="/workspace/python/out")
    args = parser.parse_args()
    main(args.data_dir, args.models_dir, args.out_dir)

