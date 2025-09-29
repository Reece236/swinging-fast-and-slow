from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .causal_model import CausalModel


def adjust_outcome_for_approach(
    approach: pd.DataFrame,
    pred_outcome_pitch: pd.DataFrame,
    causal_model: CausalModel,
) -> pd.DataFrame:
    """Return approach-adjusted prob_contact, prob_fair, pred_hit for each pitch x approach.

    approach columns: strikes_bat_speed, strikes_swing_length
    pred_outcome_pitch columns: strikes, prob_contact, prob_fair, pred_hit
    """
    # Cross product
    left = pred_outcome_pitch.assign(join=1)
    right = approach.assign(join=1)
    expanded = left.merge(right, on="join").drop(columns=["join"]) 
    expanded["approach_bat_speed"] = expanded["strikes"] * expanded["strikes_bat_speed"]
    expanded["approach_swing_length"] = expanded["strikes"] * expanded["strikes_swing_length"]

    # Predict using fitted GLMs
    # Contact
    Xc = np.column_stack(
        [
            np.ones(len(expanded)),
            expanded["approach_bat_speed"].to_numpy(),
            expanded["approach_swing_length"].to_numpy(),
        ]
    )
    expanded["prob_contact"] = causal_model.fit_contact.predict(Xc)

    # Fair given contact
    Xf = Xc  # same design
    expanded["prob_fair"] = causal_model.fit_fair.predict(Xf)

    # Hit value adjustment (linear with offset)
    expanded["pred_hit"] = causal_model.fit_hit.predict(Xc) + expanded["pred_hit"].to_numpy()
    return expanded


def summarize_pitch_outcome(pred_outcome_pitch: pd.DataFrame) -> pd.DataFrame:
    """Summarize multinomial pitch outcomes using predicted conditionals.

    Expects columns: prob_swing, prob_hbp, prob_strike, prob_contact, prob_fair, pred_hit
    """
    g = pred_outcome_pitch.groupby(list(set(pred_outcome_pitch.columns) & {"strikes_bat_speed", "strikes_swing_length", "balls", "strikes"}), dropna=False, as_index=False)
    out = g.apply(
        lambda x: pd.Series(
            {
                "mean_prob_hbp": np.mean((1 - x["prob_swing"]) * x["prob_hbp"]),
                "mean_prob_ball": np.mean((1 - x["prob_swing"]) * (1 - x["prob_hbp"]) * (1 - x["prob_strike"])),
                "mean_prob_strike": np.mean((1 - x["prob_swing"]) * (1 - x["prob_hbp"]) * x["prob_strike"] + x["prob_swing"] * (1 - x["prob_contact"])) ,
                "mean_prob_foul": np.mean(x["prob_swing"] * x["prob_contact"] * (1 - x["prob_fair"])) ,
                "mean_prob_fair": np.mean(x["prob_swing"] * x["prob_contact"] * x["prob_fair"]) ,
                "mean_pred_hit": np.average(x["pred_hit"], weights=(x["prob_swing"] * x["prob_contact"] * x["prob_fair"]).replace(0, np.nan)) ,
            }
        )
    ).reset_index(drop=True)
    return out


def _terminal_transition_table(prob_outcome_by_count: pd.DataFrame) -> pd.DataFrame:
    existing_group_vars = [c for c in prob_outcome_by_count.columns if c not in {"balls", "strikes", "prob", "outcome"}]

    grid = (
        pd.MultiIndex.from_product([range(0, 4), range(0, 3), ["Nonterminal"], prob_outcome_by_count["outcome"].unique()], names=["pre_balls", "pre_strikes", "pre_state", "outcome"])
        .to_frame(index=False)
    )
    grid["post_balls"] = grid["pre_balls"] + (grid["outcome"] == "ball").astype(int)
    grid["post_strikes"] = grid["pre_strikes"] + (grid["outcome"] == "strike").astype(int) + ((grid["pre_strikes"] < 2) & (grid["outcome"] == "foul")).astype(int)
    grid["post_state"] = np.select(
        [
            grid["outcome"] == "hbp",
            grid["outcome"] == "fair",
            grid["post_balls"] == 4,
            grid["post_strikes"] == 3,
        ],
        [
            "Hit By Pitch",
            "Fair Ball",
            "Walk",
            "Strikeout",
        ],
        default="Nonterminal",
    )
    return grid


def calculate_pred_outcome_pa(
    pred_outcome_by_count: pd.DataFrame,
    linear_weight: pd.DataFrame,
    nonterminal_prob_threshold: float = 1e-6,
) -> pd.DataFrame:
    existing_group_vars = [
        c
        for c in pred_outcome_by_count.columns
        if c
        not in {
            "balls",
            "strikes",
            "mean_prob_hbp",
            "mean_prob_ball",
            "mean_prob_strike",
            "mean_prob_foul",
            "mean_prob_fair",
            "mean_pred_hit",
        }
    ]

    pred_hit_by_count = pred_outcome_by_count[existing_group_vars + ["balls", "strikes", "mean_pred_hit"]].copy()

    prob_outcome_by_count = (
        pred_outcome_by_count.melt(
            id_vars=existing_group_vars + ["balls", "strikes"],
            value_vars=[
                "mean_prob_hbp",
                "mean_prob_ball",
                "mean_prob_strike",
                "mean_prob_foul",
                "mean_prob_fair",
            ],
            var_name="name",
            value_name="prob",
        )
        .assign(outcome=lambda d: d["name"].str.replace("mean_prob_", "", regex=False))
        .drop(columns=["name"])
        .rename(columns={"balls": "pre_balls", "strikes": "pre_strikes"})
    )

    trans = _terminal_transition_table(prob_outcome_by_count)
    prob_transition = (
        prob_outcome_by_count.merge(
            trans, on=["pre_balls", "pre_strikes", "outcome"], how="left"
        )
        .groupby(existing_group_vars + ["pre_balls", "pre_strikes", "pre_state", "post_balls", "post_strikes", "post_state"], as_index=False)["prob"].sum()
    )

    prob_terminal = prob_transition.copy()
    while True:
        mask = prob_terminal["post_state"] == "Nonterminal"
        if mask.empty or (prob_terminal.loc[mask, "prob"].max() <= nonterminal_prob_threshold):
            break
        next_step = (
            prob_terminal.merge(
                prob_transition,
                left_on=existing_group_vars + ["post_balls", "post_strikes", "post_state"],
                right_on=existing_group_vars + ["pre_balls", "pre_strikes", "pre_state"],
                how="left",
                suffixes=("_1", "_2"),
            )
            .assign(
                post_balls=lambda d: d["post_balls_2"].fillna(d["post_balls_1"]),
                post_strikes=lambda d: d["post_strikes_2"].fillna(d["post_strikes_1"]),
                post_state=lambda d: d["post_state_2"].fillna(d["post_state_1"]),
                prob_2=lambda d: d["prob_2"].fillna(1.0),
            )
        )
        keep_cols = existing_group_vars + [
            "pre_balls_1",
            "pre_strikes_1",
            "pre_state_1",
            "post_balls",
            "post_strikes",
            "post_state",
            "prob_1",
            "prob_2",
        ]
        next_step = next_step[keep_cols]
        next_step = next_step.rename(columns={
            "pre_balls_1": "pre_balls",
            "pre_strikes_1": "pre_strikes",
            "pre_state_1": "pre_state",
            "prob_1": "prob_1",
            "prob_2": "prob_2",
        })
        prob_terminal = (
            next_step.assign(prob=lambda d: d["prob_1"] * d["prob_2"]).groupby(existing_group_vars + ["pre_balls", "pre_strikes", "pre_state", "post_balls", "post_strikes", "post_state"], as_index=False)["prob"].sum()
        )

    # Aggregate terminal states from 0-0 only
    terminal = prob_terminal[
        (prob_terminal["pre_balls"] == 0)
        & (prob_terminal["pre_strikes"] == 0)
        & (prob_terminal["post_state"] != "Nonterminal")
    ].copy()
    terminal = terminal.rename(columns={"post_balls": "balls", "post_strikes": "strikes"})
    terminal["event"] = terminal["post_state"]

    # Merge pred_hit for Fair Ball states (balls, strikes present for fair only)
    pa = terminal.merge(
        pred_hit_by_count,
        on=existing_group_vars + ["balls", "strikes"],
        how="left",
    ).merge(linear_weight, on="event", how="left")

    def _choose_value(row):
        if row["event"] == "Fair Ball" and not pd.isna(row["mean_pred_hit"]):
            return row["mean_pred_hit"]
        return row["linear_weight"]

    pa["value"] = pa.apply(_choose_value, axis=1)
    runs = pa.groupby(existing_group_vars, as_index=False).apply(lambda x: pd.Series({"runs": np.average(x["value"], weights=x["prob"])})).reset_index(drop=True)
    return runs


def evaluate_approach(
    approach: pd.DataFrame,
    pred_outcome_pitch: pd.DataFrame,
    causal_model: CausalModel,
    linear_weight: pd.DataFrame,
) -> pd.DataFrame:
    adjusted = adjust_outcome_for_approach(
        approach=approach,
        pred_outcome_pitch=pred_outcome_pitch,
        causal_model=causal_model,
    )
    by_count = (
        adjusted.groupby(["strikes_bat_speed", "strikes_swing_length", "balls", "strikes"], as_index=False)
        .apply(lambda x: pd.Series({
            "mean_prob_hbp": np.mean((1 - x["prob_swing"]) * x["prob_hbp"]),
            "mean_prob_ball": np.mean((1 - x["prob_swing"]) * (1 - x["prob_hbp"]) * (1 - x["prob_strike"])) ,
            "mean_prob_strike": np.mean((1 - x["prob_swing"]) * (1 - x["prob_hbp"]) * x["prob_strike"] + x["prob_swing"] * (1 - x["prob_contact"])) ,
            "mean_prob_foul": np.mean(x["prob_swing"] * x["prob_contact"] * (1 - x["prob_fair"])) ,
            "mean_prob_fair": np.mean(x["prob_swing"] * x["prob_contact"] * x["prob_fair"]) ,
            "mean_pred_hit": np.average(x["pred_hit"], weights=(x["prob_swing"] * x["prob_contact"] * x["prob_fair"]).replace(0, np.nan)) ,
        }))
        .reset_index(drop=True)
    )
    runs = calculate_pred_outcome_pa(by_count, linear_weight=linear_weight)
    return runs

