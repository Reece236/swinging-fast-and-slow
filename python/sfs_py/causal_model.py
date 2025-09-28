from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm


def log_odds(p: np.ndarray | pd.Series) -> np.ndarray:
    return np.log(np.clip(p, 1e-8, 1 - 1e-8) / np.clip(1 - p, 1e-8, 1.0))


@dataclass
class CausalModel:
    fit_contact: sm.GLM
    fit_fair: sm.GLM
    fit_hit: sm.OLS


def fit_causal_model(data_causal: pd.DataFrame) -> CausalModel:
    """Fit GLMs with offsets to estimate approach effects.

    Requires columns: strikes, strikes_bat_speed, strikes_swing_length, prob_contact, prob_fair,
    is_contact, is_fair, hit_pred, pred_hit
    """
    for c in [
        "strikes",
        "strikes_bat_speed",
        "strikes_swing_length",
        "prob_contact",
        "prob_fair",
        "is_contact",
        "is_fair",
        "hit_pred",
        "pred_hit",
    ]:
        if c not in data_causal.columns:
            raise ValueError(f"Missing required column: {c}")

    df = data_causal.copy()
    df["approach_bat_speed"] = df["strikes"] * df["strikes_bat_speed"]
    df["approach_swing_length"] = df["strikes"] * df["strikes_swing_length"]

    # Contact model
    X = sm.add_constant(df[["approach_bat_speed", "approach_swing_length"]])
    y = df["is_contact"].astype(int)
    offset = log_odds(df["prob_contact"]).to_numpy()
    glm_binom = sm.GLM(y, X, family=sm.families.Binomial(), offset=offset)
    fit_contact = glm_binom.fit()

    # Fair given contact
    df_fair = df[df["is_contact"]].copy()
    Xf = sm.add_constant(df_fair[["approach_bat_speed", "approach_swing_length"]])
    yf = df_fair["is_fair"].astype(int)
    of = log_odds(df_fair["prob_fair"]).to_numpy()
    glm_binom_f = sm.GLM(yf, Xf, family=sm.families.Binomial(), offset=of)
    fit_fair = glm_binom_f.fit()

    # Hit value given fair
    df_hit = df[df["is_fair"]].copy()
    Xh = sm.add_constant(df_hit[["approach_bat_speed", "approach_swing_length"]])
    yh = df_hit["hit_pred"].astype(float)
    oh = df_hit["pred_hit"].astype(float)  # linear offset
    ols = sm.OLS(yh - oh, Xh)  # move offset to LHS
    fit_hit = ols.fit()

    return CausalModel(fit_contact=fit_contact, fit_fair=fit_fair, fit_hit=fit_hit)
