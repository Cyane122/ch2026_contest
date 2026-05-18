# ================================
# src/ch2026_experiments.py
#
# F1-oriented model, rule, and blend experiments for CH2026 metrics.
#
# Functions
#   - dependency_status() -> dict[str, bool] : Report optional ML package availability.
#   - run_f1_experiments(train_frame: pd.DataFrame, train_x: pd.DataFrame, sample_frame: pd.DataFrame, sample_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] : Validate candidates and predict submission rows.
# ================================

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from importlib.util import find_spec

import numpy as np
import pandas as pd

from src.ch2026_features import TARGET_COLUMNS
from src.ch2026_modeling import (
    _binary_scores,
    _candidate_strategy_names,
    _fit_predict_subject_strategy_tuned,
    _predict_rule,
)

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


@dataclass(frozen=True)
class CandidateScore:
    """Validation score and prediction metadata for one target candidate."""

    target: str
    submission_strategy: str
    model_name: str
    threshold: float
    accuracy: float
    f1: float
    positive_rate: float
    prior_gap: float


def dependency_status() -> dict[str, bool]:
    """Report optional package availability for the experiment registry."""

    return {
        "scikit-learn": find_spec("sklearn") is not None,
        "lightgbm": find_spec("lightgbm") is not None,
        "catboost": find_spec("catboost") is not None,
    }


def _temporal_mask(train_frame: pd.DataFrame, quantile: float = 0.75) -> pd.Series:
    """Return the latest temporal validation block for each subject."""

    lifelog_date = pd.to_datetime(train_frame["lifelog_date"])
    mask = pd.Series(False, index=train_frame.index)
    for _, subject_dates in lifelog_date.groupby(train_frame["subject_id"]):
        cutoff = subject_dates.quantile(quantile)
        mask.loc[subject_dates.index] = subject_dates >= cutoff
    return mask


def _rolling_masks(train_frame: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    """Return auxiliary rolling-origin masks for diagnostics."""

    return [
        ("last_quarter_by_subject", _temporal_mask(train_frame, 0.75)),
        ("last_third_by_subject", _temporal_mask(train_frame, 2.0 / 3.0)),
        ("last_fifth_by_subject", _temporal_mask(train_frame, 0.80)),
    ]


def _best_threshold(y_true: np.ndarray, probabilities: np.ndarray, prior: float) -> tuple[float, dict[str, float]]:
    """Tune a decision threshold on validation F1."""

    best_threshold = 0.5
    best_scores = {"accuracy": -1.0, "f1": -1.0, "positive_rate": 0.0, "prior_gap": np.inf}
    for threshold in np.round(np.arange(0.05, 1.0, 0.05), 2):
        predictions = (probabilities >= threshold).astype(int)
        scores = _binary_scores(y_true, predictions)
        scores["prior_gap"] = abs(scores["positive_rate"] - prior)
        score_key = (scores["f1"], scores["accuracy"], -scores["prior_gap"])
        best_key = (best_scores["f1"], best_scores["accuracy"], -best_scores["prior_gap"])
        if score_key > best_key:
            best_threshold = float(threshold)
            best_scores = scores
    return best_threshold, best_scores


def _constant_probability(y: pd.Series, size: int) -> np.ndarray | None:
    """Return a constant probability vector when training labels have one class."""

    if y.nunique() == 1:
        return np.full(size, float(y.iloc[0]), dtype=float)
    return None


def _sklearn_registry() -> dict[str, Callable[[], object]]:
    """Build sklearn model factories when sklearn is installed."""

    if find_spec("sklearn") is None:
        return {}

    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC, SVC

    registry: dict[str, Callable[[], object]] = {
        "logistic_regression": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", C=0.5, max_iter=3000, random_state=2026),
        ),
        "linear_svc": lambda: make_pipeline(StandardScaler(), LinearSVC(class_weight="balanced", C=0.2, max_iter=5000, random_state=2026)),
        "svc_rbf": lambda: make_pipeline(
            StandardScaler(),
            SVC(C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=2026),
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=120,
            max_depth=5,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=2026,
            n_jobs=1,
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=160,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=2026,
            n_jobs=1,
        ),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_iter=80,
            learning_rate=0.05,
            max_leaf_nodes=7,
            l2_regularization=0.2,
            random_state=2026,
        ),
        "mlp": lambda: make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(24,), alpha=0.03, max_iter=300, random_state=2026),
        ),
    }

    if find_spec("lightgbm") is not None:
        from lightgbm import LGBMClassifier

        registry["lightgbm"] = lambda: LGBMClassifier(
            n_estimators=70,
            learning_rate=0.03,
            num_leaves=7,
            min_child_samples=8,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            random_state=2026,
            verbosity=-1,
        )

    if find_spec("catboost") is not None:
        from catboost import CatBoostClassifier

        registry["catboost"] = lambda: CatBoostClassifier(
            iterations=70,
            depth=3,
            learning_rate=0.04,
            l2_leaf_reg=8.0,
            loss_function="Logloss",
            verbose=False,
            random_seed=2026,
        )

    return registry


def _predict_model_probability(model: object, x: pd.DataFrame) -> np.ndarray:
    """Predict probabilities from sklearn-like estimators."""

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)
        return np.asarray(probabilities[:, 1], dtype=float)
    decision = np.asarray(model.decision_function(x), dtype=float)
    decision = np.clip(decision, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-decision))


def _fit_model_probability(model_name: str, train_x: pd.DataFrame, y: pd.Series, predict_x: pd.DataFrame) -> np.ndarray:
    """Fit one registry model and return probabilities for the prediction matrix."""

    constant = _constant_probability(y, len(predict_x))
    if constant is not None:
        return constant
    model = _sklearn_registry()[model_name]()
    model.fit(train_x, y)
    return _predict_model_probability(model, predict_x)


def _rule_validation_probability(train_frame: pd.DataFrame, target: str, valid_mask: pd.Series) -> np.ndarray:
    """Predict validation rows with subject-level tuned label-history rules."""

    fit_frame = train_frame.loc[~valid_mask]
    valid_frame = train_frame.loc[valid_mask].assign(dayofweek=pd.to_datetime(train_frame.loc[valid_mask, "lifelog_date"]).dt.dayofweek)
    probabilities = pd.Series(1.0, index=valid_frame.index)
    for subject, valid_subject in valid_frame.groupby("subject_id", sort=True):
        fit_subject_all = fit_frame.loc[fit_frame["subject_id"] == subject]
        if fit_subject_all.empty:
            probabilities.loc[valid_subject.index] = float(fit_frame[target].mean() >= 0.5)
            continue
        fit_subject_all = fit_subject_all.assign(_date=pd.to_datetime(fit_subject_all["lifelog_date"]))
        cutoff = fit_subject_all["_date"].quantile(0.75)
        inner_fit = fit_subject_all.loc[fit_subject_all["_date"] < cutoff].drop(columns=["_date"])
        inner_valid = fit_subject_all.loc[fit_subject_all["_date"] >= cutoff].drop(columns=["_date"])
        inner_valid = inner_valid.assign(dayofweek=pd.to_datetime(inner_valid["lifelog_date"]).dt.dayofweek)
        best_strategy = "rolling_last_14"
        best_key = (-1.0, -1.0)
        if not inner_fit.empty and not inner_valid.empty:
            for strategy_name in _candidate_strategy_names():
                candidate = _predict_rule(strategy_name, inner_fit, inner_valid, target)
                scores = _binary_scores(inner_valid[target].to_numpy(dtype=int), candidate)
                key = (scores["f1"], scores["accuracy"])
                if key > best_key:
                    best_key = key
                    best_strategy = strategy_name
        prediction = _predict_rule(best_strategy, fit_subject_all.drop(columns=["_date"]), valid_subject, target)
        probabilities.loc[valid_subject.index] = prediction.astype(float)
    return probabilities.loc[train_frame.loc[valid_mask].index].to_numpy(dtype=float)


def _add_score(
    rows: list[CandidateScore],
    target: str,
    strategy: str,
    model_name: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    prior: float,
) -> float:
    """Tune and append one candidate score."""

    threshold, scores = _best_threshold(y_true, probabilities, prior)
    rows.append(
        CandidateScore(
            target=target,
            submission_strategy=strategy,
            model_name=model_name,
            threshold=threshold,
            accuracy=scores["accuracy"],
            f1=scores["f1"],
            positive_rate=scores["positive_rate"],
            prior_gap=scores["prior_gap"],
        )
    )
    return threshold


def _score_target(
    train_frame: pd.DataFrame,
    train_x: pd.DataFrame,
    target: str,
    valid_mask: pd.Series,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Validate all rule, model, and blend candidates for one target."""

    rows: list[CandidateScore] = []
    fit_x = train_x.loc[~valid_mask]
    valid_x = train_x.loc[valid_mask]
    fit_y = train_frame.loc[~valid_mask, target].astype(int)
    valid_y = train_frame.loc[valid_mask, target].to_numpy(dtype=int)
    prior = float(fit_y.mean())

    valid_probabilities: dict[str, np.ndarray] = {}
    rule_probability = _rule_validation_probability(train_frame, target, valid_mask)
    valid_probabilities["rule_subject_strategy_tuned"] = rule_probability
    _add_score(rows, target, "rule_subject_strategy_tuned", "rule", valid_y, rule_probability, prior)

    for model_name in _sklearn_registry():
        try:
            probability = _fit_model_probability(model_name, fit_x, fit_y, valid_x)
        except Exception as exc:  # noqa: BLE001 - experiments should continue when an optional model fails.
            print(f"Skipped {model_name} for {target}: {exc}")
            continue
        valid_probabilities[f"model_{model_name}"] = probability
        _add_score(rows, target, f"model_{model_name}", model_name, valid_y, probability, prior)
        for alpha in (0.25, 0.5, 0.75):
            blend = alpha * probability + (1.0 - alpha) * rule_probability
            strategy = f"blend_{model_name}_rule_{alpha:g}"
            valid_probabilities[strategy] = blend
            _add_score(rows, target, strategy, model_name, valid_y, blend, prior)

    if find_spec("sklearn") is not None and _sklearn_registry():
        meta_x = valid_x.copy()
        meta_x["rule_probability"] = rule_probability
        for model_name in ("extra_trees", "lightgbm", "catboost", "hist_gradient_boosting"):
            if model_name not in _sklearn_registry():
                continue
            try:
                fit_meta_x = fit_x.copy()
                fit_meta_frame = train_frame.loc[~valid_mask].copy()
                rule_column = f"{target.lower()}_history_last_value"
                fit_meta_x["rule_probability"] = fit_meta_frame[rule_column].fillna(prior).to_numpy(dtype=float)
                probability = _fit_model_probability(model_name, fit_meta_x, fit_y, meta_x)
            except Exception as exc:  # noqa: BLE001
                print(f"Skipped meta {model_name} for {target}: {exc}")
                continue
            strategy = f"meta_rule_{model_name}"
            valid_probabilities[strategy] = probability
            _add_score(rows, target, strategy, model_name, valid_y, probability, prior)

    return pd.DataFrame([row.__dict__ for row in rows]), valid_probabilities


def _select_scores(strategy_scores: pd.DataFrame) -> pd.DataFrame:
    """Select the best F1 candidate per target with deterministic tie-breakers."""

    return (
        strategy_scores.sort_values(
            ["target", "f1", "accuracy", "prior_gap"],
            ascending=[True, False, False, True],
        )
        .groupby("target", as_index=False)
        .first()
    )


def _select_generalized_scores(strategy_scores: pd.DataFrame) -> pd.DataFrame:
    """Select candidates with high mean F1 and lower split-to-split variance."""

    grouped = (
        strategy_scores.groupby(["target", "submission_strategy", "model_name"], as_index=False)
        .agg(
            mean_f1=("f1", "mean"),
            std_f1=("f1", "std"),
            min_f1=("f1", "min"),
            mean_accuracy=("accuracy", "mean"),
            mean_prior_gap=("prior_gap", "mean"),
            median_threshold=("threshold", "median"),
            folds=("fold", "nunique"),
        )
        .fillna({"std_f1": 0.0})
    )
    grouped["generalization_score"] = grouped["mean_f1"] - 0.5 * grouped["std_f1"] + 0.1 * grouped["min_f1"]
    selected = (
        grouped.sort_values(
            ["target", "generalization_score", "mean_f1", "min_f1", "mean_accuracy", "mean_prior_gap"],
            ascending=[True, False, False, False, False, True],
        )
        .groupby("target", as_index=False)
        .first()
    )
    selected["threshold"] = selected["median_threshold"]
    return selected


def _final_probability_for_strategy(
    strategy: str,
    model_name: str,
    target: str,
    train_frame: pd.DataFrame,
    train_x: pd.DataFrame,
    sample_frame: pd.DataFrame,
    sample_x: pd.DataFrame,
    rule_predictions: pd.DataFrame,
) -> np.ndarray:
    """Fit the selected candidate on all training rows and predict sample probabilities."""

    rule_probability = rule_predictions[target].to_numpy(dtype=float)
    if strategy == "rule_subject_strategy_tuned":
        return rule_probability
    if strategy.startswith("model_"):
        return _fit_model_probability(model_name, train_x, train_frame[target].astype(int), sample_x)
    if strategy.startswith("blend_"):
        model_probability = _fit_model_probability(model_name, train_x, train_frame[target].astype(int), sample_x)
        alpha = float(strategy.rsplit("_", 1)[1])
        return alpha * model_probability + (1.0 - alpha) * rule_probability
    if strategy.startswith("meta_rule_"):
        meta_train_x = train_x.copy()
        rule_column = f"{target.lower()}_history_last_value"
        meta_train_x["rule_probability"] = train_frame[rule_column].fillna(train_frame[target].mean()).to_numpy(dtype=float)
        meta_sample_x = sample_x.copy()
        meta_sample_x["rule_probability"] = rule_probability
        return _fit_model_probability(model_name, meta_train_x, train_frame[target].astype(int), meta_sample_x)
    raise ValueError(f"Unknown selected strategy: {strategy}")


def run_f1_experiments(
    train_frame: pd.DataFrame,
    train_x: pd.DataFrame,
    sample_frame: pd.DataFrame,
    sample_x: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate F1 candidates and return binary/probability submissions, scores, and selected tables."""

    score_frames = []
    for fold_name, valid_mask in _rolling_masks(train_frame):
        for target in TARGET_COLUMNS:
            target_scores, _ = _score_target(train_frame, train_x, target, valid_mask)
            target_scores["fold"] = fold_name
            score_frames.append(target_scores)

    diagnostics = pd.concat(score_frames, ignore_index=True)
    strategy_scores = diagnostics.loc[diagnostics["fold"] == "last_quarter_by_subject"].copy()
    selected = _select_scores(strategy_scores)
    generalized_selected = _select_generalized_scores(diagnostics)

    rule_predictions = _fit_predict_subject_strategy_tuned(train_frame, sample_frame)
    best_predictions = pd.DataFrame(index=sample_x.index)
    aggressive_predictions = pd.DataFrame(index=sample_x.index)
    probability_predictions = pd.DataFrame(index=sample_x.index)
    for _, row in selected.iterrows():
        target = str(row["target"])
        probability = _final_probability_for_strategy(
            str(row["submission_strategy"]),
            str(row["model_name"]),
            target,
            train_frame,
            train_x,
            sample_frame,
            sample_x,
            rule_predictions,
        )
        probability_predictions[target] = probability
        best_predictions[target] = (probability >= float(row["threshold"])).astype(int)
        aggressive_predictions[target] = (probability >= max(0.05, float(row["threshold"]) - 0.1)).astype(int)

    generalized_predictions = pd.DataFrame(index=sample_x.index)
    generalized_probability_predictions = pd.DataFrame(index=sample_x.index)
    for _, row in generalized_selected.iterrows():
        target = str(row["target"])
        probability = _final_probability_for_strategy(
            str(row["submission_strategy"]),
            str(row["model_name"]),
            target,
            train_frame,
            train_x,
            sample_frame,
            sample_x,
            rule_predictions,
        )
        generalized_probability_predictions[target] = probability
        generalized_predictions[target] = (probability >= float(row["threshold"])).astype(int)

    return (
        best_predictions.astype(int),
        aggressive_predictions.astype(int),
        generalized_predictions.astype(int),
        probability_predictions.astype(float),
        generalized_probability_predictions.astype(float),
        diagnostics,
        selected,
        generalized_selected,
    )
