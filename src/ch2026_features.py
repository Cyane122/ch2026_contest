# ================================
# src/ch2026_features.py
#
# Feature engineering for the CH2026 metrics prediction task.
#
# Functions
#   - build_sensor_features(items_dir: Path) -> pd.DataFrame : Aggregate parquet sensor tables.
#   - make_model_frame(train: pd.DataFrame, sample: pd.DataFrame, sensor_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] : Attach features to rows.
#   - prepare_feature_matrices(train_frame: pd.DataFrame, sample_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] : Build numeric model matrices.
# ================================

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMNS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
KEY_COLUMNS = ["subject_id", "sleep_date", "lifelog_date"]


def _time_period(hour: pd.Series) -> pd.Series:
    """Map timestamps to coarse day periods for sensor aggregation."""

    labels = np.select(
        [hour < 6, hour < 12, hour < 18],
        ["night", "morning", "day"],
        default="evening",
    )
    return pd.Series(labels, index=hour.index)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten pandas multi-index aggregate columns."""

    df.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in df.columns
    ]
    return df


def _numeric_row_stats(values: pd.Series, prefix: str) -> pd.DataFrame:
    """Convert a column of numeric lists into row-level statistics."""

    def to_array(value: object) -> np.ndarray:
        """Return a finite float array for a list-like cell."""

        if value is None:
            return np.array([], dtype=float)
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return np.array([], dtype=float)
        return array[np.isfinite(array)]

    arrays = values.map(to_array)
    return pd.DataFrame(
        {
            f"{prefix}_len": arrays.map(len),
            f"{prefix}_mean": arrays.map(lambda item: float(item.mean()) if len(item) else np.nan),
            f"{prefix}_std": arrays.map(lambda item: float(item.std()) if len(item) else np.nan),
            f"{prefix}_min": arrays.map(lambda item: float(item.min()) if len(item) else np.nan),
            f"{prefix}_max": arrays.map(lambda item: float(item.max()) if len(item) else np.nan),
        },
        index=values.index,
    )


def _aggregate_numeric(
    df: pd.DataFrame,
    group_columns: list[str],
    value_columns: list[str],
    prefix: str,
) -> pd.DataFrame:
    """Aggregate numeric columns by the requested keys."""

    if not value_columns:
        return df[group_columns].drop_duplicates().reset_index(drop=True)

    aggregated = df.groupby(group_columns, observed=True)[value_columns].agg(
        ["mean", "std", "min", "max", "median", "sum", "count"]
    )
    aggregated = _flatten_columns(aggregated).reset_index()
    return aggregated.rename(
        columns={column: f"{prefix}_{column}" for column in aggregated.columns if column not in group_columns}
    )


def _aggregate_value_proportions(
    df: pd.DataFrame,
    group_columns: list[str],
    value_columns: list[str],
    prefix: str,
) -> pd.DataFrame:
    """Create value-ratio features for low-cardinality numeric columns."""

    frames: list[pd.DataFrame] = []
    for column in value_columns:
        if df[column].nunique(dropna=True) > 12:
            continue
        counts = df.groupby(group_columns + [column], observed=True).size().rename("count").reset_index()
        totals = counts.groupby(group_columns, observed=True)["count"].transform("sum")
        counts["ratio"] = counts["count"] / totals
        pivot = counts.pivot_table(
            index=group_columns,
            columns=column,
            values="ratio",
            fill_value=0.0,
            observed=True,
        ).reset_index()
        pivot.columns = [
            *group_columns,
            *[f"{prefix}_{column}_ratio_{str(value).replace('.', '_')}" for value in pivot.columns[len(group_columns) :]],
        ]
        frames.append(pivot)

    if not frames:
        return df[group_columns].drop_duplicates().reset_index(drop=True)

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=group_columns, how="outer")
    return merged


def _aggregate_table(path: Path) -> pd.DataFrame:
    """Build daily and period-level features for one parquet table."""

    name = path.stem.replace("ch2025_", "")
    df = pd.read_parquet(path, engine="fastparquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["lifelog_date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df["period"] = _time_period(df["timestamp"].dt.hour)

    value_columns = [column for column in df.columns if column not in {"subject_id", "timestamp", "lifelog_date", "period"}]
    list_columns = [
        column
        for column in value_columns
        if df[column].dtype == "object"
        and df[column].dropna().map(lambda value: isinstance(value, (list, tuple, np.ndarray))).any()
    ]
    for column in list_columns:
        df = pd.concat([df.drop(columns=[column]), _numeric_row_stats(df[column], column)], axis=1)

    numeric_columns = [
        column
        for column in df.columns
        if column not in {"subject_id", "timestamp", "lifelog_date", "period"}
        and pd.api.types.is_numeric_dtype(df[column])
    ]
    base_keys = ["subject_id", "lifelog_date"]
    daily = df.groupby(base_keys, observed=True).size().rename(f"{name}_row_count").reset_index()
    daily = daily.merge(_aggregate_numeric(df, base_keys, numeric_columns, name), on=base_keys, how="outer")
    daily = daily.merge(_aggregate_value_proportions(df, base_keys, numeric_columns, name), on=base_keys, how="outer")

    period_counts = (
        df.groupby(base_keys + ["period"], observed=True)
        .size()
        .rename("count")
        .reset_index()
        .pivot_table(index=base_keys, columns="period", values="count", fill_value=0.0, observed=True)
        .reset_index()
    )
    period_counts.columns = [
        *base_keys,
        *[f"{name}_{column}_row_count" for column in period_counts.columns[len(base_keys) :]],
    ]
    daily = daily.merge(period_counts, on=base_keys, how="outer")

    if numeric_columns:
        period_numeric = df.groupby(base_keys + ["period"], observed=True)[numeric_columns].agg(["mean", "sum", "count"]).reset_index()
        period_numeric = _flatten_columns(period_numeric)
        value_columns_wide = [column for column in period_numeric.columns if column not in base_keys + ["period"]]
        period_wide = period_numeric.pivot_table(
            index=base_keys,
            columns="period",
            values=value_columns_wide,
            fill_value=np.nan,
            observed=True,
        )
        period_wide = _flatten_columns(period_wide).reset_index()
        period_wide = period_wide.rename(
            columns={column: f"{name}_{column}" for column in period_wide.columns if column not in base_keys}
        )
        daily = daily.merge(period_wide, on=base_keys, how="outer")

    return daily


def build_sensor_features(items_dir: Path) -> pd.DataFrame:
    """Build a single feature table from all available CH2025 parquet files."""

    feature_frames = [_aggregate_table(path) for path in sorted(items_dir.glob("*.parquet"))]
    features = feature_frames[0]
    for frame in feature_frames[1:]:
        features = features.merge(frame, on=["subject_id", "lifelog_date"], how="outer")
    return features


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic date features to train and submission rows."""

    result = df.copy()
    lifelog_date = pd.to_datetime(result["lifelog_date"])
    sleep_date = pd.to_datetime(result["sleep_date"])
    result["lifelog_dayofweek"] = lifelog_date.dt.dayofweek
    result["lifelog_month"] = lifelog_date.dt.month
    result["lifelog_day"] = lifelog_date.dt.day
    result["sleep_lifelog_gap_days"] = (sleep_date - lifelog_date).dt.days
    first_date = lifelog_date.groupby(result["subject_id"]).transform("min")
    result["subject_days_since_first"] = (lifelog_date - first_date).dt.days
    return result


def _subject_tail_streak(values: pd.Series) -> float:
    """Return the signed length of the latest same-label streak."""

    if values.empty:
        return 0.0
    last_value = int(values.iloc[-1])
    streak = 0
    for value in reversed(values.astype(int).tolist()):
        if value != last_value:
            break
        streak += 1
    return float(streak if last_value == 1 else -streak)


def _history_stats(values: pd.Series, dayofweek: pd.Series, current_dow: int, prefix: str) -> dict[str, float]:
    """Summarize prior labels for one row without using the current label."""

    stats: dict[str, float] = {}
    for window in (1, 3, 7, 14):
        tail = values.tail(window)
        stats[f"{prefix}_last_{window}_mean"] = float(tail.mean()) if len(tail) else np.nan
    stats[f"{prefix}_last_value"] = float(values.iloc[-1]) if len(values) else np.nan
    stats[f"{prefix}_streak"] = _subject_tail_streak(values)
    stats[f"{prefix}_change_rate"] = float(values.astype(float).diff().abs().mean()) if len(values) > 1 else np.nan
    same_dow = values.loc[dayofweek == current_dow]
    stats[f"{prefix}_dow_mean"] = float(same_dow.mean()) if len(same_dow) else np.nan
    stats[f"{prefix}_history_count"] = float(len(values))
    return stats


def _add_label_history_features(train_features: pd.DataFrame, sample_features: pd.DataFrame, train_labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add subject-level prior label features to train rows and sample rows."""

    train_result = train_features.copy()
    sample_result = sample_features.copy()
    train_dates = pd.to_datetime(train_result["lifelog_date"])
    sample_dow = pd.to_datetime(sample_result["lifelog_date"]).dt.dayofweek
    train_dow = train_dates.dt.dayofweek

    for target in TARGET_COLUMNS:
        prefix = f"{target.lower()}_history"
        train_columns = {name: [] for name in _history_stats(pd.Series(dtype=int), pd.Series(dtype=int), 0, prefix)}
        for _, subject_rows in train_result.assign(_date=train_dates, _dow=train_dow).groupby("subject_id", sort=False):
            ordered = subject_rows.sort_values("_date")
            prior_values = pd.Series(dtype=int)
            prior_dow = pd.Series(dtype=int)
            for row_index, row in ordered.iterrows():
                stats = _history_stats(prior_values, prior_dow, int(row["_dow"]), prefix)
                for name, value in stats.items():
                    train_columns[name].append((row_index, value))
                prior_values = pd.concat([prior_values, pd.Series([int(train_labels.loc[row_index, target])])], ignore_index=True)
                prior_dow = pd.concat([prior_dow, pd.Series([int(row["_dow"])])], ignore_index=True)
        for name, pairs in train_columns.items():
            series = pd.Series({index: value for index, value in pairs})
            train_result[name] = series.reindex(train_result.index).to_numpy()

        sample_values: list[float] = []
        for row_index, row in sample_result.iterrows():
            subject_mask = train_result["subject_id"] == row["subject_id"]
            subject_train = train_labels.loc[subject_mask, target]
            subject_dow = train_dow.loc[subject_mask]
            stats = _history_stats(subject_train.reset_index(drop=True), subject_dow.reset_index(drop=True), int(sample_dow.loc[row_index]), prefix)
            sample_values.append(stats[f"{prefix}_last_value"])
            for name, value in stats.items():
                if name not in sample_result.columns:
                    sample_result[name] = np.nan
                sample_result.loc[row_index, name] = value

    return train_result, sample_result


def make_model_frame(train: pd.DataFrame, sample: pd.DataFrame, sensor_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach sensor, calendar, and label-history features to rows."""

    train_features = train.merge(sensor_features, on=["subject_id", "lifelog_date"], how="left")
    sample_features = sample.merge(sensor_features, on=["subject_id", "lifelog_date"], how="left")
    combined = pd.concat([train_features, sample_features], ignore_index=True)
    combined = _add_calendar_features(combined)
    train_frame = combined.iloc[: len(train)].copy()
    sample_frame = combined.iloc[len(train) :].copy()
    return _add_label_history_features(train_frame, sample_frame, train[TARGET_COLUMNS])


def _feature_columns(train_frame: pd.DataFrame) -> list[str]:
    """Return numeric feature columns that are not keys or targets."""

    excluded = set(KEY_COLUMNS + TARGET_COLUMNS)
    return [
        column
        for column in train_frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(train_frame[column])
    ]


def prepare_feature_matrices(train_frame: pd.DataFrame, sample_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Impute feature values and add subject one-hot columns."""

    columns = _feature_columns(train_frame)
    train_x = train_frame[columns].copy()
    sample_x = sample_frame[columns].copy()
    medians = train_x.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    train_x = train_x.replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
    sample_x = sample_x.replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)

    subjects = sorted(set(train_frame["subject_id"]).union(sample_frame["subject_id"]))
    train_subjects = pd.DataFrame(
        {f"subject_{subject}": (train_frame["subject_id"] == subject).astype(float).to_numpy() for subject in subjects},
        index=train_frame.index,
    )
    sample_subjects = pd.DataFrame(
        {f"subject_{subject}": (sample_frame["subject_id"] == subject).astype(float).to_numpy() for subject in subjects},
        index=sample_frame.index,
    )
    train_x = pd.concat([train_x, train_subjects], axis=1).copy()
    sample_x = pd.concat([sample_x, sample_subjects], axis=1).copy()
    return train_x.astype(float), sample_x.astype(float)
