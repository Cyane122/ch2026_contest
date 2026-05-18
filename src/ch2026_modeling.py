# ================================
# src/ch2026_modeling.py
#
# Numpy-based modeling and validation for CH2026 binary metrics.
#
# Classes
#   - LogisticModel : Trained logistic regression state.
#
# Functions
#   - cross_validate(train_frame: pd.DataFrame, train_x: pd.DataFrame) -> pd.DataFrame : Run validation.
#   - select_submission_predictions(train_frame: pd.DataFrame, train_x: pd.DataFrame, sample_frame: pd.DataFrame, sample_x: pd.DataFrame, cv_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] : Select and predict all targets.
# ================================

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.ch2026_features import TARGET_COLUMNS


@dataclass
class LogisticModel:
    """Trained logistic regression parameters and preprocessing state."""

    weights: np.ndarray
    bias: float
    mean: np.ndarray
    scale: np.ndarray
    constant: float | None = None


def _fit_logistic_regression(x: pd.DataFrame, y: pd.Series, iterations: int = 1000) -> LogisticModel:
    """Fit a small balanced logistic regression model with numpy gradient descent."""

    y_array = y.to_numpy(dtype=float)
    if y_array.min() == y_array.max():
        return LogisticModel(
            weights=np.zeros(x.shape[1], dtype=float),
            bias=0.0,
            mean=np.zeros(x.shape[1], dtype=float),
            scale=np.ones(x.shape[1], dtype=float),
            constant=float(y_array[0]),
        )

    x_array = x.to_numpy(dtype=float)
    mean = x_array.mean(axis=0)
    scale = x_array.std(axis=0)
    scale[scale == 0] = 1.0
    x_scaled = np.clip((x_array - mean) / scale, -8.0, 8.0)

    weights = np.zeros(x_scaled.shape[1], dtype=float)
    positive_rate = y_array.mean()
    bias = float(np.log(positive_rate / (1.0 - positive_rate)))
    class_weights = np.where(y_array == 1.0, 0.5 / positive_rate, 0.5 / (1.0 - positive_rate))

    for step in range(iterations):
        logits = np.clip(x_scaled @ weights + bias, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        error = (probabilities - y_array) * class_weights
        learning_rate = 0.08 / (1.0 + step / 250.0)
        weights -= learning_rate * ((x_scaled.T @ error) / len(y_array) + 0.01 * weights)
        bias -= learning_rate * float(error.mean())

    return LogisticModel(weights=weights, bias=bias, mean=mean, scale=scale)


def _predict_probability(model: LogisticModel, x: pd.DataFrame) -> np.ndarray:
    """Predict class-one probabilities."""

    if model.constant is not None:
        return np.full(len(x), model.constant, dtype=float)
    x_array = x.to_numpy(dtype=float)
    x_scaled = np.clip((x_array - model.mean) / model.scale, -8.0, 8.0)
    logits = np.clip(x_scaled @ model.weights + model.bias, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _binary_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute accuracy, F1, and positive prediction rate."""

    true_positive = float(((y_true == 1) & (y_pred == 1)).sum())
    false_positive = float(((y_true == 0) & (y_pred == 1)).sum())
    false_negative = float(((y_true == 1) & (y_pred == 0)).sum())
    precision = true_positive / max(true_positive + false_positive, 1.0)
    recall = true_positive / max(true_positive + false_negative, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "f1": f1,
        "positive_rate": float(y_pred.mean()),
    }


def _validate_split(
    train_frame: pd.DataFrame,
    train_x: pd.DataFrame,
    target: str,
    valid_mask: pd.Series,
    fold_name: str,
    strategy: str,
) -> dict[str, float | str]:
    """Fit one validation split and return scores."""

    model = _fit_logistic_regression(train_x.loc[~valid_mask], train_frame.loc[~valid_mask, target])
    probabilities = _predict_probability(model, train_x.loc[valid_mask])
    predictions = (probabilities >= 0.5).astype(int)
    scores = _binary_scores(train_frame.loc[valid_mask, target].to_numpy(dtype=int), predictions)
    return {"target": target, "strategy": strategy, "fold": fold_name, **scores}


def cross_validate(train_frame: pd.DataFrame, train_x: pd.DataFrame) -> pd.DataFrame:
    """Run leave-one-subject-out and per-subject temporal validation."""

    rows: list[dict[str, float | str]] = []
    lifelog_date = pd.to_datetime(train_frame["lifelog_date"])
    for target in TARGET_COLUMNS:
        for subject in sorted(train_frame["subject_id"].unique()):
            subject_mask = train_frame["subject_id"] == subject
            rows.append(_validate_split(train_frame, train_x, target, subject_mask, subject, "leave_subject_out"))

        temporal_mask = pd.Series(False, index=train_frame.index)
        for subject, subject_dates in lifelog_date.groupby(train_frame["subject_id"]):
            cutoff = subject_dates.quantile(0.75)
            temporal_mask.loc[subject_dates.index] = subject_dates >= cutoff
        rows.append(_validate_split(train_frame, train_x, target, temporal_mask, "last_quarter_by_subject", "temporal"))

    return pd.DataFrame(rows)


def _temporal_mask(train_frame: pd.DataFrame) -> pd.Series:
    """Return the validation mask for each subject's latest quarter."""

    lifelog_date = pd.to_datetime(train_frame["lifelog_date"])
    mask = pd.Series(False, index=train_frame.index)
    for _, subject_dates in lifelog_date.groupby(train_frame["subject_id"]):
        cutoff = subject_dates.quantile(0.75)
        mask.loc[subject_dates.index] = subject_dates >= cutoff
    return mask


def _baseline_temporal_scores(train_frame: pd.DataFrame) -> pd.DataFrame:
    """Score prior baselines on the temporal split."""

    rows: list[dict[str, float | str]] = []
    valid_mask = _temporal_mask(train_frame)
    fit_frame = train_frame.loc[~valid_mask]
    valid_frame = train_frame.loc[valid_mask]
    fit_dow = pd.to_datetime(fit_frame["lifelog_date"]).dt.dayofweek
    valid_dow = pd.to_datetime(valid_frame["lifelog_date"]).dt.dayofweek
    for target in TARGET_COLUMNS:
        global_value = int(fit_frame[target].mean() >= 0.5)
        global_pred = np.full(len(valid_frame), global_value, dtype=int)
        rows.append(
            {
                "target": target,
                "submission_strategy": "global_majority",
                **_binary_scores(valid_frame[target].to_numpy(dtype=int), global_pred),
            }
        )

        subject_means = fit_frame.groupby("subject_id")[target].mean()
        fallback = fit_frame[target].mean()
        subject_pred = (
            valid_frame["subject_id"]
            .map(subject_means)
            .fillna(fallback)
            .ge(0.5)
            .astype(int)
            .to_numpy()
        )
        rows.append(
            {
                "target": target,
                "submission_strategy": "subject_majority",
                **_binary_scores(valid_frame[target].to_numpy(dtype=int), subject_pred),
            }
        )

        subject_dow_pred = []
        subject_dow_truth = []
        for subject, valid_subject in valid_frame.assign(dayofweek=valid_dow).groupby("subject_id", sort=True):
            fit_subject = fit_frame.loc[fit_frame["subject_id"] == subject].assign(dayofweek=fit_dow.loc[fit_frame["subject_id"] == subject])
            fallback_subject = fit_subject[target].mean() if not fit_subject.empty else fallback
            dow_means = fit_subject.groupby("dayofweek")[target].mean()
            subject_dow_pred.append(
                valid_subject["dayofweek"].map(dow_means).fillna(fallback_subject).ge(0.5).astype(int).to_numpy()
            )
            subject_dow_truth.append(valid_subject[target].to_numpy(dtype=int))
        rows.append(
            {
                "target": target,
                "submission_strategy": "subject_dow_majority",
                **_binary_scores(np.concatenate(subject_dow_truth), np.concatenate(subject_dow_pred)),
            }
        )

        global_dow_means = fit_frame.assign(dayofweek=fit_dow).groupby("dayofweek")[target].mean()
        global_dow_pred = valid_dow.map(global_dow_means).fillna(fallback).ge(0.5).astype(int).to_numpy()
        rows.append(
            {
                "target": target,
                "submission_strategy": "global_dow_majority",
                **_binary_scores(valid_frame[target].to_numpy(dtype=int), global_dow_pred),
            }
        )

        for alpha in (0.25, 0.5, 0.75):
            blended = alpha * subject_pred + (1.0 - alpha) * global_value
            rows.append(
                {
                    "target": target,
                    "submission_strategy": f"blend_subject_global_{alpha:g}",
                    **_binary_scores(valid_frame[target].to_numpy(dtype=int), (blended >= 0.5).astype(int)),
                }
            )

        for window in (1, 3, 5, 7, 14):
            rolling_predictions: list[np.ndarray] = []
            rolling_truth: list[np.ndarray] = []
            for subject, valid_subject in valid_frame.groupby("subject_id", sort=True):
                fit_subject = fit_frame.loc[fit_frame["subject_id"] == subject].sort_values("lifelog_date")
                if fit_subject.empty:
                    prior = fallback
                else:
                    prior = fit_subject[target].tail(window).mean()
                rolling_predictions.append(np.full(len(valid_subject), int(prior >= 0.5), dtype=int))
                rolling_truth.append(valid_subject[target].to_numpy(dtype=int))
            y_true = np.concatenate(rolling_truth)
            y_pred = np.concatenate(rolling_predictions)
            rows.append(
                {
                    "target": target,
                    "submission_strategy": f"rolling_last_{window}",
                    **_binary_scores(y_true, y_pred),
                }
            )

        tuned_truth: list[np.ndarray] = []
        tuned_predictions: list[np.ndarray] = []
        for subject, valid_subject in valid_frame.assign(dayofweek=valid_dow).groupby("subject_id", sort=True):
            fit_subject = fit_frame.loc[fit_frame["subject_id"] == subject].assign(
                dayofweek=fit_dow.loc[fit_frame["subject_id"] == subject]
            )
            best_predictions = None
            best_score = (-1.0, -1.0)
            for strategy_name in _candidate_strategy_names():
                candidate = _predict_rule(strategy_name, fit_subject, valid_subject, target)
                accuracy, f1, _ = _binary_scores(valid_subject[target].to_numpy(dtype=int), candidate).values()
                if (accuracy, f1) > best_score:
                    best_score = (accuracy, f1)
                    best_predictions = candidate
            tuned_truth.append(valid_subject[target].to_numpy(dtype=int))
            tuned_predictions.append(best_predictions)
        rows.append(
            {
                "target": target,
                "submission_strategy": "subject_strategy_tuned",
                **_binary_scores(np.concatenate(tuned_truth), np.concatenate(tuned_predictions)),
            }
        )
    return pd.DataFrame(rows)


def _logistic_temporal_scores(cv_scores: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal logistic scores in the same schema as baseline scores."""

    scores = cv_scores.loc[cv_scores["strategy"] == "temporal", ["target", "accuracy", "f1", "positive_rate"]].copy()
    scores["submission_strategy"] = "logistic"
    return scores


def _fit_predict_logistic(train_frame: pd.DataFrame, train_x: pd.DataFrame, sample_x: pd.DataFrame) -> pd.DataFrame:
    """Fit one model per target and return binary submission predictions."""

    predictions = pd.DataFrame(index=sample_x.index)
    for target in TARGET_COLUMNS:
        model = _fit_logistic_regression(train_x, train_frame[target])
        probabilities = _predict_probability(model, sample_x)
        predictions[target] = (probabilities >= 0.5).astype(int)
    return predictions


def _fit_predict_global_majority(train_frame: pd.DataFrame, sample_x: pd.DataFrame) -> pd.DataFrame:
    """Predict each target with its train-set majority class."""

    predictions = pd.DataFrame(index=sample_x.index)
    for target in TARGET_COLUMNS:
        predictions[target] = int(train_frame[target].mean() >= 0.5)
    return predictions


def _fit_predict_subject_majority(train_frame: pd.DataFrame, sample_frame: pd.DataFrame) -> pd.DataFrame:
    """Predict each target with each subject's train-set majority class."""

    predictions = pd.DataFrame(index=sample_frame.index)
    for target in TARGET_COLUMNS:
        subject_means = train_frame.groupby("subject_id")[target].mean()
        fallback = train_frame[target].mean()
        predictions[target] = sample_frame["subject_id"].map(subject_means).fillna(fallback).ge(0.5).astype(int).to_numpy()
    return predictions


def _fit_predict_subject_dow_majority(train_frame: pd.DataFrame, sample_frame: pd.DataFrame) -> pd.DataFrame:
    """Predict each target with subject-specific day-of-week priors."""

    predictions = pd.DataFrame(index=sample_frame.index)
    train_with_dow = train_frame.assign(dayofweek=pd.to_datetime(train_frame["lifelog_date"]).dt.dayofweek)
    sample_dow = pd.to_datetime(sample_frame["lifelog_date"]).dt.dayofweek
    for target in TARGET_COLUMNS:
        fallback = train_frame[target].mean()
        subject_fallback = train_frame.groupby("subject_id")[target].mean()
        priors = train_with_dow.groupby(["subject_id", "dayofweek"])[target].mean()
        values = []
        for row_index, row in sample_frame.iterrows():
            subject = row["subject_id"]
            dayofweek = int(sample_dow.loc[row_index])
            prior = priors.get((subject, dayofweek), subject_fallback.get(subject, fallback))
            values.append(int(prior >= 0.5))
        predictions[target] = values
    return predictions


def _fit_predict_global_dow_majority(train_frame: pd.DataFrame, sample_frame: pd.DataFrame) -> pd.DataFrame:
    """Predict each target with global day-of-week priors."""

    predictions = pd.DataFrame(index=sample_frame.index)
    train_dow = pd.to_datetime(train_frame["lifelog_date"]).dt.dayofweek
    sample_dow = pd.to_datetime(sample_frame["lifelog_date"]).dt.dayofweek
    for target in TARGET_COLUMNS:
        fallback = train_frame[target].mean()
        priors = train_frame.assign(dayofweek=train_dow).groupby("dayofweek")[target].mean()
        predictions[target] = sample_dow.map(priors).fillna(fallback).ge(0.5).astype(int).to_numpy()
    return predictions


def _fit_predict_subject_global_blend(train_frame: pd.DataFrame, sample_frame: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Predict each target with a weighted subject/global prior."""

    predictions = pd.DataFrame(index=sample_frame.index)
    for target in TARGET_COLUMNS:
        global_prior = train_frame[target].mean()
        subject_priors = train_frame.groupby("subject_id")[target].mean()
        blended = alpha * sample_frame["subject_id"].map(subject_priors).fillna(global_prior) + (1.0 - alpha) * global_prior
        predictions[target] = blended.ge(0.5).astype(int).to_numpy()
    return predictions


def _candidate_strategy_names() -> list[str]:
    """Return simple label-history rules for aggressive subject-level tuning."""

    windows = [1, 2, 3, 5, 7, 10, 14, 21, 28]
    cycles = [3, 5, 7, 10, 14, 21, 28]
    return (
        ["all_0", "all_1", "subject_dow_majority", "last_same_dow"]
        + [f"rolling_last_{window}" for window in windows]
        + [f"cycle_last_{window}" for window in cycles]
        + [f"repeat_lag_{window}" for window in windows]
    )


def _predict_rule(strategy_name: str, fit_frame: pd.DataFrame, predict_frame: pd.DataFrame, target: str) -> np.ndarray:
    """Apply one label-history rule to a subject slice."""

    if strategy_name == "all_0":
        return np.zeros(len(predict_frame), dtype=int)
    if strategy_name == "all_1":
        return np.ones(len(predict_frame), dtype=int)
    if fit_frame.empty:
        return np.ones(len(predict_frame), dtype=int)

    fit_ordered = fit_frame.sort_values("lifelog_date")
    if strategy_name == "subject_dow_majority":
        fit_dow = fit_ordered.copy()
        if "dayofweek" not in fit_dow.columns:
            fit_dow["dayofweek"] = pd.to_datetime(fit_dow["lifelog_date"]).dt.dayofweek
        predict_dow = predict_frame["dayofweek"] if "dayofweek" in predict_frame.columns else pd.to_datetime(predict_frame["lifelog_date"]).dt.dayofweek
        priors = fit_dow.groupby("dayofweek")[target].mean()
        return predict_dow.map(priors).fillna(fit_ordered[target].mean()).ge(0.5).astype(int).to_numpy()
    if strategy_name == "last_same_dow":
        fit_dow = fit_ordered.copy()
        if "dayofweek" not in fit_dow.columns:
            fit_dow["dayofweek"] = pd.to_datetime(fit_dow["lifelog_date"]).dt.dayofweek
        predict_dow = predict_frame["dayofweek"] if "dayofweek" in predict_frame.columns else pd.to_datetime(predict_frame["lifelog_date"]).dt.dayofweek
        values = []
        for dayofweek in predict_dow:
            matches = fit_dow.loc[fit_dow["dayofweek"] == dayofweek, target]
            values.append(int(matches.iloc[-1] if len(matches) else fit_ordered[target].iloc[-1]))
        return np.asarray(values, dtype=int)
    if strategy_name.startswith("rolling_last_"):
        window = int(strategy_name.rsplit("_", 1)[1])
        return np.full(len(predict_frame), int(fit_ordered[target].tail(window).mean() >= 0.5), dtype=int)
    if strategy_name.startswith("cycle_last_"):
        window = int(strategy_name.rsplit("_", 1)[1])
        values = fit_ordered[target].tail(window).to_numpy(dtype=int)
        return np.asarray([values[index % len(values)] for index in range(len(predict_frame))], dtype=int)
    if strategy_name.startswith("repeat_lag_"):
        lag = int(strategy_name.rsplit("_", 1)[1])
        value = int(fit_ordered[target].iloc[-lag] if len(fit_ordered) >= lag else fit_ordered[target].iloc[-1])
        return np.full(len(predict_frame), value, dtype=int)
    raise ValueError(f"Unknown strategy: {strategy_name}")


def _fit_predict_subject_strategy_tuned(train_frame: pd.DataFrame, sample_frame: pd.DataFrame) -> pd.DataFrame:
    """Tune a rule on each subject's latest train quarter and apply it to sample rows."""

    predictions = pd.DataFrame(index=sample_frame.index)
    train_with_date = train_frame.assign(_date=pd.to_datetime(train_frame["lifelog_date"]))
    sample_with_dow = sample_frame.assign(dayofweek=pd.to_datetime(sample_frame["lifelog_date"]).dt.dayofweek)
    for target in TARGET_COLUMNS:
        predictions[target] = 1
        for subject, subject_train in train_with_date.groupby("subject_id", sort=True):
            sample_mask = sample_frame["subject_id"] == subject
            if not sample_mask.any():
                continue
            cutoff = subject_train["_date"].quantile(0.75)
            fit_subject = subject_train.loc[subject_train["_date"] < cutoff].drop(columns=["_date"])
            valid_subject = subject_train.loc[subject_train["_date"] >= cutoff].drop(columns=["_date"])
            valid_subject = valid_subject.assign(dayofweek=pd.to_datetime(valid_subject["lifelog_date"]).dt.dayofweek)
            if fit_subject.empty or valid_subject.empty:
                fit_subject = subject_train.drop(columns=["_date"])
                best_strategy = "rolling_last_14"
            else:
                best_strategy = "all_1"
                best_score = (-1.0, -1.0)
                for strategy_name in _candidate_strategy_names():
                    candidate = _predict_rule(strategy_name, fit_subject, valid_subject, target)
                    scores = _binary_scores(valid_subject[target].to_numpy(dtype=int), candidate)
                    score_pair = (scores["accuracy"], scores["f1"])
                    if score_pair > best_score:
                        best_score = score_pair
                        best_strategy = strategy_name
                fit_subject = subject_train.drop(columns=["_date"])
            predictions.loc[sample_mask, target] = _predict_rule(
                best_strategy,
                fit_subject,
                sample_with_dow.loc[sample_mask],
                target,
            )
    return predictions.astype(int)


def _fit_predict_rolling_majority(train_frame: pd.DataFrame, sample_frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Predict each target with each subject's latest train labels."""

    predictions = pd.DataFrame(index=sample_frame.index)
    ordered = train_frame.sort_values("lifelog_date")
    for target in TARGET_COLUMNS:
        fallback = ordered[target].mean()
        subject_priors = ordered.groupby("subject_id")[target].apply(lambda values: values.tail(window).mean())
        predictions[target] = sample_frame["subject_id"].map(subject_priors).fillna(fallback).ge(0.5).astype(int).to_numpy()
    return predictions


def select_submission_predictions(
    train_frame: pd.DataFrame,
    train_x: pd.DataFrame,
    sample_frame: pd.DataFrame,
    sample_x: pd.DataFrame,
    cv_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the best temporal-F1 strategy per target and predict submission rows."""

    strategy_scores = pd.concat(
        [_logistic_temporal_scores(cv_scores), _baseline_temporal_scores(train_frame)],
        ignore_index=True,
    )
    selected = (
        strategy_scores.sort_values(["target", "accuracy", "f1"], ascending=[True, False, False])
        .groupby("target", as_index=False)
        .first()
    )

    prediction_sets = {
        "logistic": _fit_predict_logistic(train_frame, train_x, sample_x),
        "global_majority": _fit_predict_global_majority(train_frame, sample_x),
        "subject_majority": _fit_predict_subject_majority(train_frame, sample_frame),
        "subject_dow_majority": _fit_predict_subject_dow_majority(train_frame, sample_frame),
        "global_dow_majority": _fit_predict_global_dow_majority(train_frame, sample_frame),
        "blend_subject_global_0.25": _fit_predict_subject_global_blend(train_frame, sample_frame, 0.25),
        "blend_subject_global_0.5": _fit_predict_subject_global_blend(train_frame, sample_frame, 0.5),
        "blend_subject_global_0.75": _fit_predict_subject_global_blend(train_frame, sample_frame, 0.75),
        "subject_strategy_tuned": _fit_predict_subject_strategy_tuned(train_frame, sample_frame),
        "rolling_last_1": _fit_predict_rolling_majority(train_frame, sample_frame, 1),
        "rolling_last_3": _fit_predict_rolling_majority(train_frame, sample_frame, 3),
        "rolling_last_5": _fit_predict_rolling_majority(train_frame, sample_frame, 5),
        "rolling_last_7": _fit_predict_rolling_majority(train_frame, sample_frame, 7),
        "rolling_last_14": _fit_predict_rolling_majority(train_frame, sample_frame, 14),
    }
    predictions = pd.DataFrame(index=sample_x.index)
    for _, row in selected.iterrows():
        target = str(row["target"])
        strategy = str(row["submission_strategy"])
        predictions[target] = prediction_sets[strategy][target].to_numpy(dtype=int)
    return predictions, strategy_scores
