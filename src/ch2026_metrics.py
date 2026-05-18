# ================================
# src/ch2026_metrics.py
#
# Orchestration for the CH2026 seven-metric baseline pipeline.
#
# Functions
#   - run_pipeline(data_dir: Path, output_dir: Path, metric: str = "f1", install_check: bool = False) -> None : Train models and write outputs.
# ================================

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ch2026_chained import run_chained_logloss_pipeline
from src.ch2026_features import KEY_COLUMNS, TARGET_COLUMNS, build_sensor_features, make_model_frame, prepare_feature_matrices
from src.ch2026_experiments import dependency_status, run_f1_experiments
from src.ch2026_logloss import run_logloss_experiments
from src.ch2026_modeling import cross_validate, select_submission_predictions


def _write_submission(sample: pd.DataFrame, predictions: pd.DataFrame, path: Path, as_int: bool = True) -> pd.DataFrame:
    """Write one submission file in sample key order."""

    submission = sample[KEY_COLUMNS].copy()
    for target in TARGET_COLUMNS:
        if as_int:
            submission[target] = predictions[target].to_numpy(dtype=int)
        else:
            submission[target] = predictions[target].clip(1e-4, 1.0 - 1e-4).to_numpy(dtype=float)
    submission.to_csv(path, index=False)
    return submission


def run_pipeline(data_dir: Path, output_dir: Path, metric: str = "f1", install_check: bool = False) -> None:
    """Train the CH2026 baseline models and write validation/submission outputs."""

    if install_check:
        status = dependency_status()
        print("Dependency status: " + ", ".join(f"{name}={'ok' if available else 'missing'}" for name, available in status.items()))

    train_path = data_dir / "ch2026_metrics_train.csv"
    sample_path = data_dir / "ch2026_submission_sample.csv"
    items_dir = data_dir / "ch2025_data_items"

    train = pd.read_csv(train_path)
    sample = pd.read_csv(sample_path)
    sensor_features = build_sensor_features(items_dir)
    train_frame, sample_frame = make_model_frame(train, sample, sensor_features)
    train_x, sample_x = prepare_feature_matrices(train_frame, sample_frame)

    cv_scores = cross_validate(train_frame, train_x)
    legacy_predictions, legacy_strategy_scores = select_submission_predictions(train_frame, train_x, sample_frame, sample_x, cv_scores)

    output_dir.mkdir(parents=True, exist_ok=True)
    cv_scores.to_csv(output_dir / "ch2026_cv_scores.csv", index=False)

    summary = cv_scores.groupby(["strategy", "target"])[["accuracy", "f1", "positive_rate"]].mean().reset_index()
    summary.to_csv(output_dir / "ch2026_cv_summary.csv", index=False)

    if metric != "f1":
        predictions = legacy_predictions
        strategy_scores = legacy_strategy_scores
        selected = (
            strategy_scores.sort_values(["target", "accuracy", "f1"], ascending=[True, False, False])
            .groupby("target", as_index=False)
            .first()
        )
        _write_submission(sample, predictions, output_dir / "ch2026_submission.csv")
        strategy_scores.to_csv(output_dir / "ch2026_strategy_scores.csv", index=False)
        selected.to_csv(output_dir / "ch2026_selected_strategies.csv", index=False)
    else:
        (
            predictions,
            aggressive_predictions,
            generalized_predictions,
            probability_predictions,
            generalized_probability_predictions,
            model_scores,
            selected,
            generalized_selected,
        ) = run_f1_experiments(train_frame, train_x, sample_frame, sample_x)
        logloss_predictions, logloss_scores, logloss_selected = run_logloss_experiments(train_frame, train_x, sample_frame, sample_x)
        _, chained_scores = run_chained_logloss_pipeline(data_dir, output_dir)
        strategy_scores = model_scores.loc[model_scores["fold"] == "last_quarter_by_subject"].copy()
        _write_submission(sample, predictions, output_dir / "ch2026_submission.csv")
        _write_submission(sample, predictions, output_dir / "ch2026_submission_f1_best.csv")
        _write_submission(sample, aggressive_predictions, output_dir / "ch2026_submission_f1_aggressive.csv")
        _write_submission(sample, generalized_predictions, output_dir / "ch2026_submission_f1_generalized.csv")
        _write_submission(sample, probability_predictions, output_dir / "ch2026_submission_logloss_best.csv", as_int=False)
        _write_submission(sample, generalized_probability_predictions, output_dir / "ch2026_submission_logloss_generalized.csv", as_int=False)
        _write_submission(sample, logloss_predictions, output_dir / "ch2026_submission_logloss_cv.csv", as_int=False)
        _write_submission(sample, legacy_predictions, output_dir / "ch2026_submission_legacy.csv")
        model_scores.to_csv(output_dir / "ch2026_model_scores.csv", index=False)
        model_scores.to_csv(output_dir / "ch2026_generalization_scores.csv", index=False)
        logloss_scores.to_csv(output_dir / "ch2026_logloss_scores.csv", index=False)
        logloss_selected.to_csv(output_dir / "ch2026_logloss_selected_strategies.csv", index=False)
        chained_scores.to_csv(output_dir / "ch2026_chained_scores.csv", index=False)
        strategy_scores.to_csv(output_dir / "ch2026_strategy_scores.csv", index=False)
        selected.to_csv(output_dir / "ch2026_selected_strategies.csv", index=False)
        generalized_selected.to_csv(output_dir / "ch2026_generalized_strategies.csv", index=False)

    print(summary.to_string(index=False))
    leaderboard = (
        strategy_scores.sort_values(["target", "f1", "accuracy", "positive_rate"], ascending=[True, False, False, True])
        .groupby("target")
        .head(3)
    )
    print("\nTop submission strategy candidates")
    leaderboard_columns = [column for column in ["target", "submission_strategy", "threshold", "accuracy", "f1", "positive_rate"] if column in leaderboard.columns]
    print(leaderboard[leaderboard_columns].to_string(index=False))
    print("\nSelected submission strategies")
    display_columns = [column for column in ["target", "submission_strategy", "threshold", "accuracy", "f1", "positive_rate"] if column in selected.columns]
    print(selected[display_columns].to_string(index=False))
    if metric == "f1":
        print("\nGeneralized submission strategies")
        generalized_columns = [
            column
            for column in [
                "target",
                "submission_strategy",
                "threshold",
                "mean_f1",
                "std_f1",
                "min_f1",
                "mean_accuracy",
                "generalization_score",
            ]
            if column in generalized_selected.columns
        ]
        print(generalized_selected[generalized_columns].to_string(index=False))
        print("\nLog-loss submission strategies")
        logloss_columns = [
            column
            for column in [
                "target",
                "submission_strategy",
                "shrink_alpha",
                "mean_logloss",
                "std_logloss",
                "max_logloss",
                "selection_score",
            ]
            if column in logloss_selected.columns
        ]
        print(logloss_selected[logloss_columns].to_string(index=False))
        print("\nChained log-loss scores")
        print(chained_scores.to_string(index=False))
    print(f"Wrote {output_dir / 'ch2026_submission.csv'}")
