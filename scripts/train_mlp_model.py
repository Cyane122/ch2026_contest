from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

from train_baseline_model import (  # noqa: E402
    KEYS,
    SUBMISSIONS,
    TABLES,
    TARGETS,
    binary_logloss,
    build_features,
    clip_proba,
    feature_columns,
    load_rows,
    make_design,
    target_history_features,
    validation_splits,
)

REPORTS = ROOT / "reports"


def candidate_mlps(random_state: int = 42) -> dict[str, object]:
    return {
        "mlp_tiny": make_pipeline(
            SimpleImputer(strategy="median"),
            VarianceThreshold(),
            SelectKBest(f_classif, k=128),
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(32,),
                activation="relu",
                solver="adam",
                alpha=0.10,
                learning_rate_init=0.001,
                max_iter=1200,
                early_stopping=True,
                validation_fraction=0.25,
                n_iter_no_change=40,
                random_state=random_state,
            ),
        ),
        "mlp_small": make_pipeline(
            SimpleImputer(strategy="median"),
            VarianceThreshold(),
            SelectKBest(f_classif, k=256),
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 16),
                activation="relu",
                solver="adam",
                alpha=0.08,
                learning_rate_init=0.0008,
                max_iter=1500,
                early_stopping=True,
                validation_fraction=0.25,
                n_iter_no_change=50,
                random_state=random_state,
            ),
        ),
        "mlp_wide_regularized": make_pipeline(
            SimpleImputer(strategy="median"),
            VarianceThreshold(),
            SelectKBest(f_classif, k=384),
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(96, 24),
                activation="relu",
                solver="adam",
                alpha=0.20,
                learning_rate_init=0.0006,
                max_iter=1500,
                early_stopping=True,
                validation_fraction=0.25,
                n_iter_no_change=50,
                random_state=random_state,
            ),
        ),
    }


def best_blend(y_true: pd.Series, model_p: np.ndarray, prior_p: np.ndarray) -> tuple[float, float]:
    best_alpha, best_score = 1.0, binary_logloss(y_true, model_p)
    for alpha in np.linspace(0, 1, 41):
        pred = alpha * model_p + (1.0 - alpha) * prior_p
        score = binary_logloss(y_true, pred)
        if score < best_score:
            best_alpha, best_score = float(alpha), float(score)
    return best_alpha, best_score


def prior_candidates(
    train: pd.DataFrame,
    sample: pd.DataFrame,
    fit_mask: pd.Series,
    valid_mask: pd.Series,
    target: str,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_fit = train.loc[fit_mask, target]
    global_prior = float(y_fit.mean())
    subject_prior = train.loc[fit_mask].groupby("subject_id")[target].mean()
    valid_prior = train.loc[valid_mask, "subject_id"].map(subject_prior).fillna(global_prior).to_numpy()
    sample_prior = sample["subject_id"].map(train.groupby("subject_id")[target].mean()).fillna(float(train[target].mean())).to_numpy()

    fit_history = target_history_features(train.loc[fit_mask], train.loc[fit_mask], target)
    valid_history = target_history_features(train.loc[valid_mask], train.loc[fit_mask], target)
    all_history = target_history_features(train, train, target)
    sample_history = target_history_features(sample, train, target)

    valid_neighbor = valid_history[f"{target}_neighbor_mean"].to_numpy(dtype=float)
    valid_neighbor = np.where(np.isnan(valid_neighbor), valid_prior, valid_neighbor)
    sample_neighbor = sample_history[f"{target}_neighbor_mean"].to_numpy(dtype=float)
    sample_neighbor = np.where(np.isnan(sample_neighbor), sample_prior, sample_neighbor)

    priors = {
        "subject_prior": (valid_prior, sample_prior),
        "neighbor_prior": (valid_neighbor, sample_neighbor),
        "subject_neighbor_prior": (
            0.5 * valid_prior + 0.5 * valid_neighbor,
            0.5 * sample_prior + 0.5 * sample_neighbor,
        ),
    }
    return priors, fit_history, valid_history, all_history, sample_history


def evaluate_mlp_split(
    split_name: str,
    valid_mask: pd.Series,
    train: pd.DataFrame,
    sample: pd.DataFrame,
    train_x: pd.DataFrame,
    sample_x: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_mask = ~valid_mask
    submission = sample[KEYS].copy()
    rows = []

    for target in TARGETS:
        y_fit = train.loc[fit_mask, target]
        y_valid = train.loc[valid_mask, target]
        priors, fit_history, valid_history, all_history, sample_history = prior_candidates(
            train, sample, fit_mask, valid_mask, target
        )

        best = None
        best_sample_pred = None
        for prior_name, (valid_p, sample_p) in priors.items():
            score = binary_logloss(y_valid, valid_p)
            if best is None or score < best["valid_log_loss"]:
                best = {
                    "target": target,
                    "model": prior_name,
                    "alpha": 0.0,
                    "valid_log_loss": score,
                    "valid_model_log_loss": np.nan,
                    "split": split_name,
                    "valid_rows": int(valid_mask.sum()),
                }
                best_sample_pred = sample_p.copy()

        x_fit = pd.concat([train_x.loc[fit_mask, feature_cols].reset_index(drop=True), fit_history.reset_index(drop=True)], axis=1)
        x_valid = pd.concat([train_x.loc[valid_mask, feature_cols].reset_index(drop=True), valid_history.reset_index(drop=True)], axis=1)
        x_all = pd.concat([train_x[feature_cols].reset_index(drop=True), all_history.reset_index(drop=True)], axis=1)
        x_sample = pd.concat([sample_x[feature_cols].reset_index(drop=True), sample_history.reset_index(drop=True)], axis=1)

        for model_name, model in candidate_mlps(random_state=777).items():
            model.fit(x_fit, y_fit)
            valid_model_p = model.predict_proba(x_valid)[:, 1]
            model_score = binary_logloss(y_valid, valid_model_p)
            for prior_name, (valid_prior, sample_prior) in priors.items():
                alpha, blend_score = best_blend(y_valid, valid_model_p, valid_prior)
                if blend_score < best["valid_log_loss"]:
                    final_model = candidate_mlps(random_state=1777)[model_name]
                    final_model.fit(x_all, train[target])
                    sample_model_p = final_model.predict_proba(x_sample)[:, 1]
                    best_sample_pred = alpha * sample_model_p + (1.0 - alpha) * sample_prior
                    best = {
                        "target": target,
                        "model": f"{model_name}+{prior_name}",
                        "alpha": alpha,
                        "valid_log_loss": blend_score,
                        "valid_model_log_loss": model_score,
                        "split": split_name,
                        "valid_rows": int(valid_mask.sum()),
                    }

        submission[target] = clip_proba(best_sample_pred)
        rows.append(best)

    scores = pd.DataFrame(rows)
    scores.loc[len(scores)] = {
        "target": "average",
        "model": "selected",
        "alpha": np.nan,
        "valid_log_loss": scores["valid_log_loss"].mean(),
        "valid_model_log_loss": scores["valid_model_log_loss"].mean(),
        "split": split_name,
        "valid_rows": int(valid_mask.sum()),
    }
    return scores, submission


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)

    train, sample, all_rows = load_rows()
    features = build_features(all_rows)
    train_x, sample_x = make_design(train, sample, features)
    cols = feature_columns(train_x)

    split_scores = []
    split_submissions = {}
    for split_name, mask in validation_splits(train).items():
        scores, submission = evaluate_mlp_split(split_name, mask, train, sample, train_x, sample_x, cols)
        split_scores.append(scores)
        split_submissions[split_name] = submission
        submission.to_csv(SUBMISSIONS / f"mlp_{split_name}_submission.csv", index=False)

    all_scores = pd.concat(split_scores, ignore_index=True)
    all_scores.to_csv(TABLES / "mlp_validation_scores_by_split.csv", index=False)

    ensemble_weights = {
        "interleaved_offset1": 0.25,
        "interleaved_offset2": 0.25,
        "subject_hole0": 0.10,
        "subject_hole1": 0.15,
        "subject_hole2": 0.10,
        "late7": 0.15,
    }
    ensemble = sample[KEYS].copy()
    for target in TARGETS:
        values = np.zeros(len(sample), dtype=float)
        for split, weight in ensemble_weights.items():
            values += weight * split_submissions[split][target].to_numpy(dtype=float)
        ensemble[target] = clip_proba(values)
    ensemble.to_csv(SUBMISSIONS / "mlp_robust_ensemble_submission.csv", index=False)

    meta = {
        "model_family": "sklearn MLPClassifier tabular neural network",
        "feature_count": len(cols),
        "split_average_log_loss": {
            row["split"]: float(row["valid_log_loss"])
            for row in all_scores[all_scores["target"] == "average"].to_dict("records")
        },
        "submission_path": str(SUBMISSIONS / "mlp_robust_ensemble_submission.csv"),
    }
    (REPORTS / "mlp_model_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(all_scores[all_scores["target"] == "average"].to_string(index=False))
    print(f"Wrote {SUBMISSIONS / 'mlp_robust_ensemble_submission.csv'}")


if __name__ == "__main__":
    main()
