# ================================
# src/ch2026_logloss.py
#
# Log-loss-oriented validation, calibration, and submission prediction.
#
# Functions
#   - run_logloss_experiments(train_frame: pd.DataFrame, train_x: pd.DataFrame, sample_frame: pd.DataFrame, sample_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] : Select calibrated probabilities and diagnostics.
# ================================

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ch2026_features import TARGET_COLUMNS
from src.ch2026_experiments import _final_probability_for_strategy, _rolling_masks, _score_target
from src.ch2026_modeling import _fit_predict_subject_strategy_tuned


def _binary_log_loss(y_true: np.ndarray, probability: np.ndarray) -> float:
    """Compute binary log loss with clipping for numeric stability."""

    y_array = np.asarray(y_true, dtype=float)
    p_array = np.clip(np.asarray(probability, dtype=float), 1e-4, 1.0 - 1e-4)
    return float(-(y_array * np.log(p_array) + (1.0 - y_array) * np.log(1.0 - p_array)).mean())


def _model_name_from_strategy(strategy: str) -> str:
    """Infer the model registry name used by a probability strategy."""

    if strategy == "rule_subject_strategy_tuned":
        return "rule"
    if strategy.startswith("model_"):
        return strategy.removeprefix("model_")
    if strategy.startswith("meta_rule_"):
        return strategy.removeprefix("meta_rule_")
    if strategy.startswith("blend_"):
        return strategy.removeprefix("blend_").rsplit("_rule_", 1)[0]
    if strategy == "history_blend":
        return "history"
    raise ValueError(f"Unknown strategy: {strategy}")


def _history_probability(frame: pd.DataFrame, target: str, prior: float) -> np.ndarray:
    """Build a conservative subject-history probability for one target."""

    prefix = f"{target.lower()}_history"
    last_3 = frame[f"{prefix}_last_3_mean"].fillna(prior).to_numpy(dtype=float)
    last_7 = frame[f"{prefix}_last_7_mean"].fillna(prior).to_numpy(dtype=float)
    dow = frame[f"{prefix}_dow_mean"].fillna(prior).to_numpy(dtype=float)
    return 0.4 * last_7 + 0.3 * last_3 + 0.2 * dow + 0.1 * prior


def _score_logloss_candidates(train_frame: pd.DataFrame, train_x: pd.DataFrame) -> pd.DataFrame:
    """Score model and history candidates with prior shrinkage on rolling temporal folds."""

    rows: list[dict[str, float | str]] = []
    shrink_alphas = [0.0, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0]
    for fold_name, valid_mask in _rolling_masks(train_frame):
        for target in TARGET_COLUMNS:
            fit_frame = train_frame.loc[~valid_mask]
            valid_frame = train_frame.loc[valid_mask]
            y_true = valid_frame[target].to_numpy(dtype=int)
            prior = float(fit_frame[target].mean())

            history_probability = _history_probability(valid_frame, target, prior)
            rows.append(
                {
                    "fold": fold_name,
                    "target": target,
                    "submission_strategy": "history_blend",
                    "model_name": "history",
                    "shrink_alpha": 1.0,
                    "logloss": _binary_log_loss(y_true, history_probability),
                }
            )

            _, probabilities = _score_target(train_frame, train_x, target, valid_mask)
            for strategy, probability in probabilities.items():
                for alpha in shrink_alphas:
                    calibrated = prior + alpha * (np.asarray(probability, dtype=float) - prior)
                    rows.append(
                        {
                            "fold": fold_name,
                            "target": target,
                            "submission_strategy": strategy,
                            "model_name": _model_name_from_strategy(strategy),
                            "shrink_alpha": alpha,
                            "logloss": _binary_log_loss(y_true, calibrated),
                        }
                    )

    return pd.DataFrame(rows)


def _select_logloss_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Select the lowest mean log loss per target across temporal folds."""

    grouped = (
        scores.groupby(["target", "submission_strategy", "model_name", "shrink_alpha"], as_index=False)
        .agg(
            mean_logloss=("logloss", "mean"),
            std_logloss=("logloss", "std"),
            max_logloss=("logloss", "max"),
            folds=("fold", "nunique"),
        )
        .fillna({"std_logloss": 0.0})
    )
    grouped["selection_score"] = grouped["mean_logloss"] + 0.25 * grouped["std_logloss"] + 0.05 * grouped["max_logloss"]
    return (
        grouped.sort_values(
            ["target", "selection_score", "mean_logloss", "max_logloss"],
            ascending=[True, True, True, True],
        )
        .groupby("target", as_index=False)
        .first()
    )


def run_logloss_experiments(
    train_frame: pd.DataFrame,
    train_x: pd.DataFrame,
    sample_frame: pd.DataFrame,
    sample_x: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select calibrated probabilities and return submission predictions plus diagnostics."""

    scores = _score_logloss_candidates(train_frame, train_x)
    selected = _select_logloss_scores(scores)
    rule_predictions = _fit_predict_subject_strategy_tuned(train_frame, sample_frame)

    predictions = pd.DataFrame(index=sample_x.index)
    for _, row in selected.iterrows():
        target = str(row["target"])
        prior = float(train_frame[target].mean())
        strategy = str(row["submission_strategy"])
        if strategy == "history_blend":
            probability = _history_probability(sample_frame, target, prior)
        else:
            probability = _final_probability_for_strategy(
                strategy,
                str(row["model_name"]),
                target,
                train_frame,
                train_x,
                sample_frame,
                sample_x,
                rule_predictions,
            )
        alpha = float(row["shrink_alpha"])
        predictions[target] = prior + alpha * (np.asarray(probability, dtype=float) - prior)

    return predictions.astype(float), scores, selected
