# ================================
# src/ch2026_metrics.py
#
# Orchestration for the CH2026 seven-metric baseline pipeline.
#
# Functions
#   - run_pipeline(data_dir: Path, output_dir: Path, metric: str = "f1", install_check: bool = False) -> None : Train models and write outputs.
#     Supported metrics: f1, accuracy, xgb-variants, lstm, lstm-targetwise, sequence-variants,
#                        registry-ensembles, oof-ensemble, anchor-stack, conservative-blend,
#                        feature-diagnosis, raw-cnn
# ================================

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ch2026_anchor_stack import run_anchor_stack_pipeline
from src.ch2026_conservative_blend import run_conservative_blend_pipeline
from src.ch2026_feature_diagnosis import run_feature_diagnosis
from src.ch2026_rawcnn import run_rawcnn_pipeline
from src.ch2026_chained import run_chained_logloss_pipeline
from src.ch2026_chained import run_xgb_variant_experiments
from src.ch2026_ensemble import run_oof_ensemble_pipeline
from src.ch2026_features import KEY_COLUMNS, TARGET_COLUMNS, build_sensor_features, make_model_frame, prepare_feature_matrices
from src.ch2026_experiments import dependency_status, run_f1_experiments
from src.ch2026_logloss import export_registry_probability_candidates, run_logloss_experiments
from src.ch2026_modeling import cross_validate, select_submission_predictions
from src.ch2026_sequence import run_lstm_sequence_pipeline, run_sequence_variant_experiments, write_lstm_targetwise_blend


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

    if metric == "xgb-variants":
        output_dir.mkdir(parents=True, exist_ok=True)
        scores = run_xgb_variant_experiments(data_dir, output_dir)
        print(scores.groupby("variant")["logloss"].mean().sort_values().to_string())
        return

    if metric == "lstm":
        output_dir.mkdir(parents=True, exist_ok=True)
        _, scores = run_lstm_sequence_pipeline(data_dir, output_dir)
        print(scores.to_string(index=False))
        print(f"Wrote {output_dir / 'ch2026_submission_lstm.csv'}")
        return

    if metric == "sequence-variants":
        output_dir.mkdir(parents=True, exist_ok=True)
        scores = run_sequence_variant_experiments(data_dir, output_dir)
        print(scores.groupby("model_type")["mean_logloss"].mean().sort_values().to_string())
        _, selected = write_lstm_targetwise_blend(output_dir)
        print("\nTargetwise sequence/tree selection")
        print(selected.to_string(index=False))
        print(f"Wrote {output_dir / 'ch2026_submission_lstm_targetwise.csv'}")
        return

    if metric == "lstm-targetwise":
        output_dir.mkdir(parents=True, exist_ok=True)
        submission_path = output_dir / "ch2026_submission_lstm.csv"
        score_path = output_dir / "ch2026_sequence_scores.csv"
        if not submission_path.exists() or not score_path.exists():
            _, scores = run_lstm_sequence_pipeline(data_dir, output_dir)
            print(scores.to_string(index=False))
        _, selected = write_lstm_targetwise_blend(output_dir)
        print(selected.to_string(index=False))
        print(f"Wrote {output_dir / 'ch2026_submission_lstm_targetwise.csv'}")
        return

    if metric == "anchor-stack":
        output_dir.mkdir(parents=True, exist_ok=True)
        _, scores = run_anchor_stack_pipeline(data_dir, output_dir)
        print(scores.sort_values("logloss").head(25).to_string(index=False))
        print(f"Wrote {output_dir / 'ch2026_submission_anchor_stack_targetwise.csv'}")
        return

    if metric == "conservative-blend":
        output_dir.mkdir(parents=True, exist_ok=True)
        diagnostic = run_conservative_blend_pipeline(data_dir, output_dir)
        print(diagnostic.to_string(index=False))
        print(f"Wrote conservative blend submissions to {output_dir}/")
        return

    if metric == "feature-diagnosis":
        output_dir.mkdir(parents=True, exist_ok=True)
        run_feature_diagnosis(output_dir)
        print(f"Wrote {output_dir / 'ch2026_feature_diagnosis.csv'}")
        return

    if metric == "raw-cnn":
        output_dir.mkdir(parents=True, exist_ok=True)
        _, scores = run_rawcnn_pipeline(data_dir, output_dir)
        print(scores.to_string(index=False))
        print(f"Wrote {output_dir / 'ch2026_submission_rawcnn.csv'}")
        return

    sensor_features = build_sensor_features(items_dir)
    train_frame, sample_frame = make_model_frame(train, sample, sensor_features)
    train_x, sample_x = prepare_feature_matrices(train_frame, sample_frame)

    if metric == "registry-ensembles":
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = export_registry_probability_candidates(train_frame, train_x, sample_frame, sample_x, output_dir)
        print(manifest.groupby("source")["mean_logloss"].mean().sort_values().head(20).to_string())
        _, selected = write_lstm_targetwise_blend(output_dir)
        print("\nTargetwise registry/sequence/tree selection")
        print(selected.to_string(index=False))
        print(f"Wrote {output_dir / 'ch2026_submission_lstm_targetwise.csv'}")
        return

    if metric == "oof-ensemble":
        output_dir.mkdir(parents=True, exist_ok=True)
        _, selection, scores = run_oof_ensemble_pipeline(data_dir, output_dir)
        print(scores.groupby(["source", "fold_strategy"])["logloss"].mean().sort_values().head(20).to_string())
        print("\nOOF weighted ensemble selection")
        print(selection.to_string(index=False))
        print(f"Wrote {output_dir / 'ch2026_submission_oof_ensemble.csv'}")
        return

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
