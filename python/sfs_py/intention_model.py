from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm


@dataclass
class IntentionModelFit:
    model: pm.Model
    idata: az.InferenceData
    swing_metric: Literal["bat_speed", "swing_length"]
    batter_levels: np.ndarray
    pitcher_levels: np.ndarray


def _build_design_matrices(intent: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    features = intent[["balls", "strikes", "plate_x_ref", "plate_z", "batter_side_id", "pitcher_id"]].copy()
    # Encodings
    batter_levels = features["batter_side_id"].astype("category").cat.categories.to_numpy()
    pitcher_levels = features["pitcher_id"].astype("category").cat.categories.to_numpy()
    batter_idx = features["batter_side_id"].astype("category").cat.codes.to_numpy()
    pitcher_idx = features["pitcher_id"].astype("category").cat.codes.to_numpy()
    X = features[["balls", "strikes", "plate_x_ref", "plate_z"]].to_numpy().astype(float)
    return features, batter_idx, pitcher_idx, batter_levels, pitcher_levels, X


def fit_intention_model(intent: pd.DataFrame, swing_metric: Literal["bat_speed", "swing_length"] = "bat_speed", draws: int = 2000, tune: int = 1500, chains: int = 4, target_accept: float = 0.9) -> IntentionModelFit:
    """Fit a hierarchical skew-normal intention model analogous to the BRMS specification.

    Formula parity (approx):
      swing_metric ~ balls + strikes + plate_x_ref + plate_z
                    + (1 | pitcher_id)
                    + (1 + strikes + plate_x_ref + plate_z | batter_side_id)
      sigma ~ 1
      alpha ~ 1 + (1 | batter_side_id)
    """
    metric = swing_metric
    if metric not in ("bat_speed", "swing_length"):
        raise ValueError("swing_metric must be 'bat_speed' or 'swing_length'")

    # Filter NA in response
    intent = intent.dropna(subset=[metric]).copy()
    features, batter_idx, pitcher_idx, batter_levels, pitcher_levels, X = _build_design_matrices(intent)
    y = intent[metric].to_numpy().astype(float)

    n_obs = y.shape[0]
    n_batter = batter_levels.shape[0]
    n_pitcher = pitcher_levels.shape[0]

    with pm.Model(coords={
        "obs": np.arange(n_obs),
        "slope_name": ["strikes", "plate_x_ref", "plate_z"],
        "fix_name": ["balls", "strikes", "plate_x_ref", "plate_z"],
        "batter": batter_levels,
        "pitcher": pitcher_levels,
    }) as model:
        # Fixed effects
        beta_fix = pm.Normal("beta_fix", mu=0.0, sigma=5.0, dims=("fix_name",))

        # Batter random intercept and random slopes (strikes, plate_x_ref, plate_z)
        sigma_bi = pm.HalfNormal("sigma_bi", sigma=2.0)
        u_bi_raw = pm.Normal("u_bi_raw", mu=0.0, sigma=1.0, dims=("batter",))
        u_bi = pm.Deterministic("u_bi", sigma_bi * u_bi_raw, dims=("batter",))

        sigma_bs = pm.HalfNormal("sigma_bs", sigma=2.0, dims=("slope_name",))
        u_bs_raw = pm.Normal("u_bs_raw", mu=0.0, sigma=1.0, dims=("slope_name", "batter"))
        u_bs = pm.Deterministic("u_bs", sigma_bs[:, None] * u_bs_raw, dims=("slope_name", "batter"))

        # Pitcher random intercept
        sigma_pi = pm.HalfNormal("sigma_pi", sigma=2.0)
        u_pi_raw = pm.Normal("u_pi_raw", mu=0.0, sigma=1.0, dims=("pitcher",))
        u_pi = pm.Deterministic("u_pi", sigma_pi * u_pi_raw, dims=("pitcher",))

        # Scale and skewness
        sigma = pm.HalfNormal("sigma", sigma=5.0)
        alpha_mu = pm.Normal("alpha_mu", mu=0.0, sigma=2.0)
        alpha_sd = pm.HalfNormal("alpha_sd", sigma=1.0)
        alpha_b_raw = pm.Normal("alpha_b_raw", mu=0.0, sigma=1.0, dims=("batter",))
        alpha_b = pm.Deterministic("alpha_b", alpha_mu + alpha_sd * alpha_b_raw, dims=("batter",))

        # Linear predictor
        balls = X[:, 0]
        strikes = X[:, 1]
        plate_x_ref = X[:, 2]
        plate_z = X[:, 3]

        # Fixed part
        mu_fix = (beta_fix[0] * balls + beta_fix[1] * strikes + beta_fix[2] * plate_x_ref + beta_fix[3] * plate_z)

        # Random contributions
        mu = (
            mu_fix
            + u_bi[batter_idx]
            + u_pi[pitcher_idx]
            + u_bs[0, batter_idx] * strikes
            + u_bs[1, batter_idx] * plate_x_ref
            + u_bs[2, batter_idx] * plate_z
        )

        pm.SkewNormal("y", mu=mu, sigma=sigma, alpha=alpha_b[batter_idx], observed=y, dims=("obs",))

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            progressbar=False,
        )

    return IntentionModelFit(model=model, idata=idata, swing_metric=metric, batter_levels=batter_levels, pitcher_levels=pitcher_levels)


def summarize_random_slopes(intent_fit: IntentionModelFit) -> pd.DataFrame:
    """Extract posterior means of batter random slopes for strikes.

    Returns a DataFrame with columns: batter_side_id, strikes_<metric>
    """
    posterior = intent_fit.idata.posterior
    # u_bs dims: chain, draw, slope_name, batter
    u_bs = posterior["u_bs"].mean(dim=("chain", "draw"))
    strikes_by_batter = u_bs.sel(slope_name="strikes").to_numpy()
    out = pd.DataFrame({
        "batter_side_id": intent_fit.batter_levels,
        f"strikes_{intent_fit.swing_metric}": strikes_by_batter,
    })
    return out
