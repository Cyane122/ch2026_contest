# ================================
# src/ch2026_chained.py
#
# Subject-hole CV, six-hour-shift features, and S-to-Q chained log-loss models.
#
# Functions
#   - run_chained_logloss_pipeline(data_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame] : Train chained models and write predictions.
#   - run_xgb_variant_experiments(data_dir: Path, output_dir: Path) -> pd.DataFrame : Generate XGBoost-heavy chained variants.
# ================================

from __future__ import annotations

import ast
import os
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

from src.ch2026_features import KEY_COLUMNS, TARGET_COLUMNS

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def _flatten_hr_row(value: object) -> float:
    """Parse a list-like heart-rate cell and return the positive-value mean."""

    try:
        values = value if isinstance(value, (list, tuple, np.ndarray)) else ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return np.nan
    valid = [item for item in values if isinstance(item, (int, float)) and item > 0]
    return float(np.mean(valid)) if valid else np.nan


def _shifted_date(timestamp: pd.Series) -> pd.Series:
    """Align late-night sensor events to the prior sleep/lifelog day."""

    return (pd.to_datetime(timestamp) - pd.Timedelta(hours=6)).dt.date.astype(str)


def make_subject_hole_folds(train: pd.DataFrame, n_folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split each subject timeline into alternating held-out date holes."""

    all_indices = train.index.to_numpy()
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    block_count = max(n_folds * 2, 4)
    subject_chunks: dict[str, list[np.ndarray]] = {}

    for subject_id, group in train.sort_values(["subject_id", "sleep_date"]).groupby("subject_id", sort=False):
        indices = group.index.to_numpy()
        subject_chunks[str(subject_id)] = [chunk for chunk in np.array_split(indices, block_count) if len(chunk)]

    for fold_id in range(n_folds):
        valid_parts = []
        for chunks in subject_chunks.values():
            for hole_id in (fold_id, fold_id + n_folds):
                if hole_id < len(chunks):
                    valid_parts.append(chunks[hole_id])
        if not valid_parts:
            continue
        valid_idx = np.concatenate(valid_parts)
        train_idx = np.setdiff1d(all_indices, valid_idx, assume_unique=False)
        if len(train_idx) and len(valid_idx):
            folds.append((train_idx, valid_idx))

    return folds


def build_dual_feature_space(items_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Build six-hour-shifted absolute and subject-relative sensor features."""

    activity = pd.read_parquet(items_dir / "ch2025_mActivity.parquet")
    activity["dt"] = pd.to_datetime(activity["timestamp"])
    activity["hour"] = activity["dt"].dt.hour
    activity["date"] = _shifted_date(activity["timestamp"])
    activity_day = (
        activity.loc[activity["hour"].isin(range(6, 20))]
        .groupby(["subject_id", "date"])["m_activity"]
        .sum()
        .reset_index()
        .rename(columns={"m_activity": "act_day_sum"})
    )
    activity_night = (
        activity.loc[~activity["hour"].isin(range(6, 20))]
        .groupby(["subject_id", "date"])["m_activity"]
        .sum()
        .reset_index()
        .rename(columns={"m_activity": "act_night_sum"})
    )
    activity_daily = activity_day.merge(activity_night, on=["subject_id", "date"], how="outer")

    pedometer = pd.read_parquet(items_dir / "ch2025_wPedo.parquet")
    pedometer["date"] = _shifted_date(pedometer["timestamp"])
    pedometer_daily = (
        pedometer.groupby(["subject_id", "date"])[["step", "burned_calories"]]
        .sum()
        .reset_index()
        .rename(columns={"step": "total_steps", "burned_calories": "total_calories"})
    )

    screen = pd.read_parquet(items_dir / "ch2025_mScreenStatus.parquet")
    screen["dt"] = pd.to_datetime(screen["timestamp"])
    screen["hour"] = screen["dt"].dt.hour
    screen["date"] = _shifted_date(screen["timestamp"])
    screen_presleep = (
        screen.loc[screen["hour"].isin([20, 21, 22, 23])]
        .groupby(["subject_id", "date"])["m_screen_use"]
        .mean()
        .reset_index()
        .rename(columns={"m_screen_use": "screen_presleep_ratio"})
    )
    screen_sleep = (
        screen.loc[screen["hour"].isin([0, 1, 2, 3, 4, 5, 6])]
        .groupby(["subject_id", "date"])["m_screen_use"]
        .mean()
        .reset_index()
        .rename(columns={"m_screen_use": "screen_sleep_ratio"})
    )
    screen_daily = screen_presleep.merge(screen_sleep, on=["subject_id", "date"], how="outer")

    heart_rate = pd.read_parquet(items_dir / "ch2025_wHr.parquet")
    heart_rate["hr_val"] = heart_rate["heart_rate"].apply(_flatten_hr_row)
    heart_rate["dt"] = pd.to_datetime(heart_rate["timestamp"])
    heart_rate["hour"] = heart_rate["dt"].dt.hour
    heart_rate["date"] = _shifted_date(heart_rate["timestamp"])
    heart_rate_sleep = (
        heart_rate.loc[heart_rate["hour"].isin([0, 1, 2, 3, 4, 5, 6])]
        .groupby(["subject_id", "date"])["hr_val"]
        .mean()
        .reset_index()
        .rename(columns={"hr_val": "hr_sleep_mean"})
    )

    light = pd.read_parquet(items_dir / "ch2025_mLight.parquet")
    light["date"] = _shifted_date(light["timestamp"])
    light_daily = light.groupby(["subject_id", "date"])["m_light"].mean().reset_index().rename(columns={"m_light": "light_mean"})

    features = activity_daily.merge(pedometer_daily, on=["subject_id", "date"], how="outer")
    features = features.merge(screen_daily, on=["subject_id", "date"], how="outer")
    features = features.merge(heart_rate_sleep, on=["subject_id", "date"], how="outer")
    features = features.merge(light_daily, on=["subject_id", "date"], how="outer")

    absolute_columns = [column for column in features.columns if column not in {"subject_id", "date"}]
    for column in absolute_columns:
        subject_stats = features.groupby("subject_id")[column].agg(["mean", "std"]).reset_index()
        subject_stats.columns = ["subject_id", f"{column}_subject_mean", f"{column}_subject_std"]
        features = features.merge(subject_stats, on="subject_id", how="left")
        features[f"{column}_zscore"] = (
            (features[column] - features[f"{column}_subject_mean"])
            / (features[f"{column}_subject_std"] + 1e-8)
        )
        features = features.drop(columns=[f"{column}_subject_mean", f"{column}_subject_std"])

    features["dayofweek"] = pd.to_datetime(features["date"]).dt.dayofweek
    return features, absolute_columns


def _prepare_chained_frames(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Read labels/sample files and attach chained model features."""

    train = pd.read_csv(data_dir / "ch2026_metrics_train.csv")
    sample = pd.read_csv(data_dir / "ch2026_submission_sample.csv")
    features, absolute_columns = build_dual_feature_space(data_dir / "ch2025_data_items")

    for frame in (train, sample, features):
        frame["subject_id"] = frame["subject_id"].astype(str)
    train["lifelog_date"] = train["lifelog_date"].astype(str)
    sample["lifelog_date"] = sample["lifelog_date"].astype(str)
    features["date"] = features["date"].astype(str)

    train_frame = train.merge(features, left_on=["subject_id", "lifelog_date"], right_on=["subject_id", "date"], how="left")
    sample_frame = sample.merge(features, left_on=["subject_id", "lifelog_date"], right_on=["subject_id", "date"], how="left")
    train_frame = train_frame.drop(columns=["date"])
    sample_frame = sample_frame.drop(columns=["date"])

    encoder = LabelEncoder()
    encoder.fit(pd.concat([train_frame["subject_id"], sample_frame["subject_id"]]).unique())
    for frame in (train_frame, sample_frame):
        frame["subject_encoded"] = encoder.transform(frame["subject_id"]).astype(int)

    zscore_columns = [column for column in train_frame.columns if column.endswith("_zscore")]
    numeric_columns = sorted(set(absolute_columns + zscore_columns + ["dayofweek"]))
    for column in numeric_columns:
        median = train_frame[column].median()
        fill_value = float(median) if pd.notna(median) else 0.0
        train_frame[column] = train_frame[column].fillna(fill_value)
        sample_frame[column] = sample_frame[column].fillna(fill_value)

    stage1_features = absolute_columns + ["dayofweek", "subject_encoded"]
    stage2_features = absolute_columns + zscore_columns + ["dayofweek", "subject_encoded"]
    return train, sample, train_frame, sample_frame, stage1_features, stage2_features


def _make_model_pair(random_state: int, model_mode: str = "blend") -> list[tuple[str, object, float]]:
    """Return the model blend used for each fold."""

    lgbm = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=15,
        subsample=0.65,
        colsample_bytree=0.65,
        min_child_samples=15,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=1,
        verbose=-1,
    )
    if model_mode == "lgbm":
        return [("lightgbm", lgbm, 1.0)]

    if find_spec("xgboost") is not None:
        from xgboost import XGBClassifier

        secondary = XGBClassifier(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.65,
            colsample_bytree=0.65,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=random_state,
            eval_metric="logloss",
            enable_categorical=False,
            n_jobs=1,
        )
        if model_mode == "xgb":
            return [("xgboost", secondary, 1.0)]
        if model_mode == "lgbm_xgb_50_50":
            return [("lightgbm", lgbm, 0.5), ("xgboost", secondary, 0.5)]
        if model_mode == "lgbm_xgb_30_70":
            return [("lightgbm", lgbm, 0.3), ("xgboost", secondary, 0.7)]
        return [("lightgbm", lgbm, 0.7), ("xgboost", secondary, 0.3)]

    secondary = ExtraTreesClassifier(
        n_estimators=180,
        max_depth=5,
        min_samples_leaf=4,
        random_state=random_state,
        n_jobs=1,
    )
    if model_mode == "extra_trees":
        return [("extra_trees", secondary, 1.0)]
    return [("lightgbm", lgbm, 0.7), ("extra_trees", secondary, 0.3)]


def _fit_predict_blend(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    valid_x: pd.DataFrame,
    test_x: pd.DataFrame,
    random_state: int,
    model_mode: str = "blend",
) -> tuple[np.ndarray, np.ndarray, str]:
    """Fit one fold blend and return validation/test probabilities."""

    valid_probability = np.zeros(len(valid_x), dtype=float)
    test_probability = np.zeros(len(test_x), dtype=float)
    model_names = []
    for model_name, model, weight in _make_model_pair(random_state, model_mode):
        model.fit(train_x, train_y)
        valid_probability += weight * model.predict_proba(valid_x)[:, 1]
        test_probability += weight * model.predict_proba(test_x)[:, 1]
        model_names.append(model_name)
    return valid_probability, test_probability, "+".join(model_names)


def run_chained_logloss_pipeline(
    data_dir: Path,
    output_dir: Path,
    output_name: str = "ch2026_submission_chained.csv",
    score_name: str = "ch2026_chained_scores.csv",
    use_chaining: bool = True,
    model_mode: str = "blend",
    q_model_mode: str | None = None,
    s_model_mode: str | None = None,
    clip: float = 1e-4,
    n_folds: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train the chained log-loss pipeline and write a probability submission."""

    train, sample, train_frame, sample_frame, stage1_features, stage2_features = _prepare_chained_frames(data_dir)
    folds = make_subject_hole_folds(train_frame, n_folds=n_folds)
    if not folds:
        raise ValueError("No subject-hole CV folds were created.")
    q_mode = q_model_mode or model_mode
    s_mode = s_model_mode or model_mode

    scores: list[dict[str, float | str]] = []
    predictions = pd.DataFrame(index=sample_frame.index)
    s_targets = ["S1", "S2", "S3", "S4"]
    q_targets = ["Q1", "Q2", "Q3"]

    for target in s_targets:
        y = train_frame[target].fillna(0).astype(int).to_numpy()
        oof_probability = np.zeros(len(train_frame), dtype=float)
        test_probability = np.zeros(len(sample_frame), dtype=float)
        target_losses = []
        model_blend = ""
        for fold_id, (train_idx, valid_idx) in enumerate(folds):
            valid_probability, fold_test_probability, model_blend = _fit_predict_blend(
                train_frame[stage1_features].iloc[train_idx],
                y[train_idx],
                train_frame[stage1_features].iloc[valid_idx],
                sample_frame[stage1_features],
                random_state=4200 + fold_id,
                model_mode=s_mode,
            )
            oof_probability[valid_idx] = valid_probability
            test_probability += fold_test_probability / len(folds)
            target_losses.append(log_loss(y[valid_idx], np.clip(valid_probability, 1e-4, 1.0 - 1e-4)))
        train_frame[f"predicted_{target}_prob"] = oof_probability
        sample_frame[f"predicted_{target}_prob"] = test_probability
        predictions[target] = test_probability
        scores.append({"target": target, "stage": "S", "model_blend": model_blend, "logloss": float(np.mean(target_losses))})

    stage2_extended_features = stage2_features + [f"predicted_{target}_prob" for target in s_targets] if use_chaining else stage2_features
    for target in q_targets:
        y = train_frame[target].fillna(0).astype(int).to_numpy()
        test_probability = np.zeros(len(sample_frame), dtype=float)
        target_losses = []
        model_blend = ""
        for fold_id, (train_idx, valid_idx) in enumerate(folds):
            valid_probability, fold_test_probability, model_blend = _fit_predict_blend(
                train_frame[stage2_extended_features].iloc[train_idx],
                y[train_idx],
                train_frame[stage2_extended_features].iloc[valid_idx],
                sample_frame[stage2_extended_features],
                random_state=5200 + fold_id,
                model_mode=q_mode,
            )
            test_probability += fold_test_probability / len(folds)
            target_losses.append(log_loss(y[valid_idx], np.clip(valid_probability, 1e-4, 1.0 - 1e-4)))
        predictions[target] = test_probability
        scores.append({"target": target, "stage": "Q", "model_blend": model_blend, "logloss": float(np.mean(target_losses))})

    submission = sample[KEY_COLUMNS].copy()
    for target in TARGET_COLUMNS:
        submission[target] = predictions[target].clip(clip, 1.0 - clip).to_numpy(dtype=float)

    output_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_dir / output_name, index=False)
    score_frame = pd.DataFrame(scores)
    score_frame.to_csv(output_dir / score_name, index=False)
    return submission, score_frame


def run_xgb_variant_experiments(data_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Generate XGBoost-heavy chained variants and write their CV scores."""

    variants = [
        {
            "name": "xgb_only",
            "kwargs": {"model_mode": "xgb"},
        },
        {
            "name": "lgbm_xgb_50_50",
            "kwargs": {"model_mode": "lgbm_xgb_50_50"},
        },
        {
            "name": "lgbm_xgb_30_70",
            "kwargs": {"model_mode": "lgbm_xgb_30_70"},
        },
        {
            "name": "q_xgb_s_blend",
            "kwargs": {"s_model_mode": "blend", "q_model_mode": "xgb"},
        },
        {
            "name": "q_70_s_blend",
            "kwargs": {"s_model_mode": "blend", "q_model_mode": "lgbm_xgb_30_70"},
        },
        {
            "name": "hole7_blend",
            "kwargs": {"model_mode": "blend", "n_folds": 7},
        },
    ]
    score_frames = []
    for variant in variants:
        name = str(variant["name"])
        _, scores = run_chained_logloss_pipeline(
            data_dir,
            output_dir,
            output_name=f"ch2026_submission_{name}.csv",
            score_name=f"ch2026_{name}_scores.csv",
            **variant["kwargs"],
        )
        scores["variant"] = name
        score_frames.append(scores)
    summary = pd.concat(score_frames, ignore_index=True)
    summary.to_csv(output_dir / "ch2026_xgb_variant_scores.csv", index=False)
    return summary
