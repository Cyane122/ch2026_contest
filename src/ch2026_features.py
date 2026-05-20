# ================================
# src/ch2026_features.py
#
# Feature engineering for the CH2026 metrics prediction task.
#
# Functions
#   - build_sensor_features(items_dir: Path) -> pd.DataFrame : Aggregate parquet sensor tables (includes 43 semantic cross-features).
#   - make_model_frame(train: pd.DataFrame, sample: pd.DataFrame, sensor_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] : Attach features to rows.
#   - prepare_feature_matrices(train_frame: pd.DataFrame, sample_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] : Build numeric model matrices.
# ================================

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMNS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
KEY_COLUMNS = ["subject_id", "sleep_date", "lifelog_date"]
WINDOW_DEFINITIONS = {
    "daily_00_24": tuple(range(24)),
    "day_09_18": tuple(range(9, 18)),
    "evening_18_21": tuple(range(18, 21)),
    "prebed_21_24": tuple(range(21, 24)),
    "sleep_21_09": tuple(list(range(21, 24)) + list(range(0, 9))),
    "late_00_03": tuple(range(0, 3)),
    "deep_03_06": tuple(range(3, 6)),
    "wake_06_09": tuple(range(6, 9)),
}
WINDOW_INTERACTIONS = [
    ("prebed_21_24", "sleep_21_09"),
    ("sleep_21_09", "wake_06_09"),
    ("late_00_03", "deep_03_06"),
    ("deep_03_06", "wake_06_09"),
    ("day_09_18", "evening_18_21"),
    ("evening_18_21", "prebed_21_24"),
]
AMBIENCE_GROUPS = {
    "speech": ("speech", "conversation", "narration", "monologue"),
    "silence": ("silence",),
    "music": ("music", "television"),
    "vehicle": ("vehicle", "car", "truck", "motor vehicle"),
    "outdoor": ("outside", "wind", "rural", "urban", "natural"),
    "sleep_noise": ("breathing", "snoring", "white noise", "mechanical fan"),
}
APP_GROUPS = {
    "social": ("카카오톡", "instagram", "threads", " x", " twitter", "facebook"),
    "media": ("youtube", "tiktok", "웹툰", "music", "netflix", "영상", "뮤직"),
    "communication": ("통화", "전화", "메시지", "message", "mail", "gmail"),
    "game": ("game", "게임", "클래시", "royal", "number match", "whiteout", "merge"),
    "finance_shopping": ("토스", "카드", "pay", "쿠팡", "당근", "쇼핑", "bank", "은행"),
    "routine_health": ("캐시워크", "withings", "시계", "health", "헬스", "만보기"),
    "browser_search": ("naver", "chrome", "삼성 인터넷", "browser", "검색"),
    "camera_gallery": ("카메라", "갤러리", "photo", "camera"),
    "system_home": ("one ui", "시스템 ui", "launcher", "홈"),
}


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
            f"{prefix}_median": arrays.map(lambda item: float(np.median(item)) if len(item) else np.nan),
            f"{prefix}_p10": arrays.map(lambda item: float(np.percentile(item, 10)) if len(item) else np.nan),
            f"{prefix}_p90": arrays.map(lambda item: float(np.percentile(item, 90)) if len(item) else np.nan),
            f"{prefix}_range": arrays.map(lambda item: float(item.max() - item.min()) if len(item) else np.nan),
            f"{prefix}_spike": arrays.map(lambda item: float(np.percentile(item, 90) - np.median(item)) if len(item) else np.nan),
            f"{prefix}_recovery": arrays.map(lambda item: float(np.median(item) - item.min()) if len(item) else np.nan),
        },
        index=values.index,
    )


def _list_like(value: object) -> list[object]:
    """Return a Python list for ndarray/list-like parquet cells."""

    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _semantic_ambience_features(values: pd.Series) -> pd.DataFrame:
    """Convert ambience label-score arrays into grouped acoustic features."""

    rows: list[dict[str, float]] = []
    for value in values:
        scores = {group: 0.0 for group in AMBIENCE_GROUPS}
        max_scores = {f"{group}_max": 0.0 for group in AMBIENCE_GROUPS}
        top_group = "unknown"
        top_score = 0.0
        label_count = 0
        for pair in _list_like(value):
            if len(pair) < 2:
                continue
            label = str(pair[0]).lower()
            try:
                score = float(pair[1])
            except (TypeError, ValueError):
                continue
            label_count += 1
            if score > top_score:
                top_score = score
                top_group = "unknown"
            for group, keywords in AMBIENCE_GROUPS.items():
                if any(keyword in label for keyword in keywords):
                    scores[group] += score
                    max_scores[f"{group}_max"] = max(max_scores[f"{group}_max"], score)
                    if score >= top_score:
                        top_group = group
        row = {f"ambience_{group}_score": value for group, value in scores.items()}
        row.update({f"ambience_{group}": value for group, value in max_scores.items()})
        row["ambience_label_count"] = float(label_count)
        row["ambience_top_score"] = top_score
        for group in AMBIENCE_GROUPS:
            row[f"ambience_top_is_{group}"] = float(top_group == group)
        rows.append(row)
    return pd.DataFrame(rows, index=values.index)


def _app_category(app_name: object) -> str:
    """Map one app name to a coarse behavior category."""

    normalized = str(app_name).strip().lower()
    padded = f" {normalized}"
    for group, keywords in APP_GROUPS.items():
        if any(keyword.lower() in padded for keyword in keywords):
            return group
    return "other"


def _semantic_usage_features(values: pd.Series) -> pd.DataFrame:
    """Convert usage-stat app arrays into behavior category features."""

    rows: list[dict[str, float]] = []
    categories = list(APP_GROUPS) + ["other"]
    for value in values:
        totals = {category: 0.0 for category in categories}
        counts = {f"{category}_app_count": 0.0 for category in categories}
        total_time = 0.0
        top_time = 0.0
        app_count = 0
        for item in _list_like(value):
            if not isinstance(item, dict):
                continue
            category = _app_category(item.get("app_name", ""))
            try:
                usage_time = float(item.get("total_time", 0.0))
            except (TypeError, ValueError):
                usage_time = 0.0
            totals[category] += usage_time
            counts[f"{category}_app_count"] += 1.0
            total_time += usage_time
            top_time = max(top_time, usage_time)
            app_count += 1
        row = {f"usage_{category}_time": value for category, value in totals.items()}
        row.update({f"usage_{name}": value for name, value in counts.items()})
        row["usage_total_time"] = total_time
        row["usage_app_count"] = float(app_count)
        row["usage_top_app_time_ratio"] = top_time / (total_time + 1e-6)
        rows.append(row)
    return pd.DataFrame(rows, index=values.index)


def _semantic_gps_features(df: pd.DataFrame, values: pd.Series) -> pd.DataFrame:
    """Convert GPS point arrays into movement and home-distance features."""

    rows: list[dict[str, float]] = []
    for value in values:
        points = [item for item in _list_like(value) if isinstance(item, dict)]
        row: dict[str, float] = {"gps_point_count": float(len(points))}
        for key in ("latitude", "longitude", "altitude", "speed"):
            series = np.asarray([float(item.get(key, np.nan)) for item in points], dtype=float)
            series = series[np.isfinite(series)]
            row[f"gps_{key}_mean"] = float(series.mean()) if len(series) else np.nan
            row[f"gps_{key}_std"] = float(series.std()) if len(series) else np.nan
            row[f"gps_{key}_min"] = float(series.min()) if len(series) else np.nan
            row[f"gps_{key}_max"] = float(series.max()) if len(series) else np.nan
            row[f"gps_{key}_range"] = float(series.max() - series.min()) if len(series) else np.nan
        rows.append(row)

    features = pd.DataFrame(rows, index=values.index)
    night_mask = pd.to_datetime(df["timestamp"]).dt.hour.isin(range(0, 6))
    home = (
        pd.concat([df[["subject_id"]], features[["gps_latitude_mean", "gps_longitude_mean"]]], axis=1)
        .loc[night_mask]
        .groupby("subject_id", observed=True)
        .median(numeric_only=True)
        .rename(columns={"gps_latitude_mean": "home_latitude", "gps_longitude_mean": "home_longitude"})
    )
    aligned_home = df[["subject_id"]].merge(home, left_on="subject_id", right_index=True, how="left")
    distance = np.sqrt(
        (features["gps_latitude_mean"].to_numpy(dtype=float) - aligned_home["home_latitude"].to_numpy(dtype=float)) ** 2
        + (features["gps_longitude_mean"].to_numpy(dtype=float) - aligned_home["home_longitude"].to_numpy(dtype=float)) ** 2
    )
    features["gps_home_distance"] = distance
    features["gps_mobility_radius"] = np.sqrt(features["gps_latitude_range"].fillna(0.0) ** 2 + features["gps_longitude_range"].fillna(0.0) ** 2)
    return features


def _frequent_ids(df: pd.DataFrame, column: str, id_key: str, min_count: int) -> dict[str, set[str]]:
    """Return subject-level frequently observed WiFi/BLE identifiers."""

    counters: dict[str, dict[str, int]] = {}
    for subject_id, value in zip(df["subject_id"], df[column]):
        subject_counter = counters.setdefault(str(subject_id), {})
        for item in _list_like(value):
            if isinstance(item, dict) and item.get(id_key) is not None:
                identifier = str(item[id_key])
                subject_counter[identifier] = subject_counter.get(identifier, 0) + 1
    return {
        subject_id: {identifier for identifier, count in counter.items() if count >= min_count}
        for subject_id, counter in counters.items()
    }


def _semantic_network_features(df: pd.DataFrame, column: str, id_key: str, prefix: str, min_count: int) -> pd.DataFrame:
    """Convert WiFi/BLE scans into density, signal, and familiarity features."""

    familiar_ids = _frequent_ids(df, column, id_key, min_count)
    rows: list[dict[str, float]] = []
    for subject_id, value in zip(df["subject_id"], df[column]):
        items = [item for item in _list_like(value) if isinstance(item, dict)]
        ids = [str(item[id_key]) for item in items if item.get(id_key) is not None]
        rssi = np.asarray([float(item.get("rssi", np.nan)) for item in items], dtype=float)
        rssi = rssi[np.isfinite(rssi)]
        familiar = familiar_ids.get(str(subject_id), set())
        familiar_count = sum(identifier in familiar for identifier in ids)
        row = {
            f"{prefix}_detected_count": float(len(items)),
            f"{prefix}_unique_id_count": float(len(set(ids))),
            f"{prefix}_familiar_ratio": familiar_count / (len(ids) + 1e-6),
            f"{prefix}_novel_ratio": 1.0 - familiar_count / (len(ids) + 1e-6),
            f"{prefix}_rssi_mean": float(rssi.mean()) if len(rssi) else np.nan,
            f"{prefix}_rssi_max": float(rssi.max()) if len(rssi) else np.nan,
            f"{prefix}_rssi_std": float(rssi.std()) if len(rssi) else np.nan,
            f"{prefix}_strong_rssi_count": float(np.sum(rssi >= -65)) if len(rssi) else 0.0,
        }
        rows.append(row)
    return pd.DataFrame(rows, index=df.index)


def _apply_semantic_parsers(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Expand object-valued sensor columns into semantically meaningful numeric features."""

    result = df.copy()
    parsers = {
        "mAmbience": lambda frame: _semantic_ambience_features(frame["m_ambience"]),
        "mUsageStats": lambda frame: _semantic_usage_features(frame["m_usage_stats"]),
        "mGps": lambda frame: _semantic_gps_features(frame, frame["m_gps"]),
        "mWifi": lambda frame: _semantic_network_features(frame, "m_wifi", "bssid", "wifi", min_count=20),
        "mBle": lambda frame: _semantic_network_features(frame, "m_ble", "address", "ble", min_count=8),
    }
    if table_name in parsers:
        source_column = [column for column in result.columns if column not in {"subject_id", "timestamp"} and result[column].dtype == "object"][0]
        semantic = parsers[table_name](result)
        result = pd.concat([result.drop(columns=[source_column]), semantic], axis=1)
    return result


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


def _aggregate_window_numeric(
    df: pd.DataFrame,
    group_columns: list[str],
    value_columns: list[str],
    prefix: str,
) -> pd.DataFrame:
    """Aggregate numeric columns with a wider statistic set for one time window."""

    base = df[group_columns].drop_duplicates().reset_index(drop=True)
    if not value_columns:
        return base

    aggregated = df.groupby(group_columns, observed=True)[value_columns].agg(
        ["mean", "std", "min", "max", "median", "sum", "count", "nunique"]
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


def _aggregate_time_windows(
    df: pd.DataFrame,
    base_keys: list[str],
    numeric_columns: list[str],
    table_name: str,
) -> pd.DataFrame:
    """Create sleep-centered window aggregations and window-pair interactions."""

    daily = df[base_keys].drop_duplicates().reset_index(drop=True)
    window_suffixes: dict[str, set[str]] = {}

    for window_name, hours in WINDOW_DEFINITIONS.items():
        window_df = df.loc[df["timestamp"].dt.hour.isin(hours)].copy()
        if window_df.empty:
            continue

        prefix = f"{table_name}_{window_name}"
        row_count = window_df.groupby(base_keys, observed=True).size().rename(f"{prefix}_row_count").reset_index()
        window_features = row_count.merge(
            _aggregate_window_numeric(window_df, base_keys, numeric_columns, prefix),
            on=base_keys,
            how="outer",
        )
        window_features = window_features.merge(
            _aggregate_value_proportions(window_df, base_keys, numeric_columns, prefix),
            on=base_keys,
            how="outer",
        )
        daily = daily.merge(window_features, on=base_keys, how="outer")
        window_suffixes[window_name] = {
            column.removeprefix(prefix + "_")
            for column in window_features.columns
            if column not in base_keys and pd.api.types.is_numeric_dtype(window_features[column])
        }

    interaction_frames: list[pd.DataFrame] = []
    for left_window, right_window in WINDOW_INTERACTIONS:
        common_suffixes = sorted(window_suffixes.get(left_window, set()).intersection(window_suffixes.get(right_window, set())))
        if not common_suffixes:
            continue

        left_prefix = f"{table_name}_{left_window}"
        right_prefix = f"{table_name}_{right_window}"
        interaction_columns: dict[str, pd.Series] = {}
        for suffix in common_suffixes:
            left = daily[f"{left_prefix}_{suffix}"]
            right = daily[f"{right_prefix}_{suffix}"]
            pair_prefix = f"{table_name}_{left_window}_to_{right_window}_{suffix}"
            interaction_columns[f"{pair_prefix}_diff"] = right - left
            interaction_columns[f"{pair_prefix}_ratio"] = right / (left.abs() + 1e-6)
            interaction_columns[f"{pair_prefix}_abs_diff"] = (right - left).abs()
        interaction = pd.concat([daily[base_keys], pd.DataFrame(interaction_columns, index=daily.index)], axis=1)
        interaction_frames.append(interaction)

    for frame in interaction_frames:
        daily = daily.merge(frame, on=base_keys, how="outer")

    return daily


def _aggregate_table(path: Path) -> pd.DataFrame:
    """Build daily and period-level features for one parquet table."""

    name = path.stem.replace("ch2025_", "")
    df = pd.read_parquet(path, engine="pyarrow")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = _apply_semantic_parsers(df, name)
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

    window_features = _aggregate_time_windows(df, base_keys, numeric_columns, name)
    daily = daily.merge(window_features, on=base_keys, how="outer")

    return daily


def _optional_column(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric column or a zero series when it is unavailable."""

    if column in df.columns:
        return df[column].astype(float)
    return pd.Series(0.0, index=df.index)


def _add_semantic_cross_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add domain-level sleep, mobility, social, and app behavior features."""

    result = features.copy()
    engineered: dict[str, pd.Series] = {}

    engineered["semantic_sleep_night_screen_minutes"] = (
        _optional_column(result, "mScreenStatus_late_00_03_m_screen_use_sum")
        + _optional_column(result, "mScreenStatus_deep_03_06_m_screen_use_sum")
    )
    engineered["semantic_sleep_prebed_screen_ratio"] = _optional_column(result, "mScreenStatus_prebed_21_24_m_screen_use_mean")
    engineered["semantic_sleep_charging_overnight_ratio"] = _optional_column(result, "mACStatus_sleep_21_09_m_charging_mean")
    engineered["semantic_sleep_charging_break_proxy"] = (
        _optional_column(result, "mACStatus_prebed_21_24_m_charging_mean")
        - _optional_column(result, "mACStatus_wake_06_09_m_charging_mean")
    ).abs()
    engineered["semantic_sleep_light_spike"] = (
        _optional_column(result, "mLight_late_00_03_m_light_max")
        + _optional_column(result, "wLight_late_00_03_w_light_max")
        + _optional_column(result, "mLight_deep_03_06_m_light_max")
        + _optional_column(result, "wLight_deep_03_06_w_light_max")
    )
    engineered["semantic_sleep_late_steps_sum"] = (
        _optional_column(result, "wPedo_late_00_03_step_sum")
        + _optional_column(result, "wPedo_deep_03_06_step_sum")
    )
    engineered["semantic_sleep_wake_hr_rise"] = (
        _optional_column(result, "wHr_wake_06_09_heart_rate_mean_mean")
        - _optional_column(result, "wHr_deep_03_06_heart_rate_mean_mean")
    )
    engineered["semantic_sleep_hr_spike_proxy"] = _optional_column(result, "wHr_sleep_21_09_heart_rate_spike_mean")

    engineered["semantic_mobility_daily_radius"] = _optional_column(result, "mGps_daily_00_24_gps_mobility_radius_mean")
    engineered["semantic_mobility_prebed_home_distance"] = _optional_column(result, "mGps_prebed_21_24_gps_home_distance_mean")
    engineered["semantic_mobility_late_home_distance"] = _optional_column(result, "mGps_late_00_03_gps_home_distance_mean")
    engineered["semantic_mobility_day_speed"] = _optional_column(result, "mGps_day_09_18_gps_speed_mean_mean")
    engineered["semantic_mobility_wifi_familiar_ratio"] = _optional_column(result, "mWifi_daily_00_24_wifi_familiar_ratio_mean")
    engineered["semantic_mobility_ble_density_evening"] = _optional_column(result, "mBle_evening_18_21_ble_detected_count_mean")
    engineered["semantic_mobility_novel_place_proxy"] = (
        _optional_column(result, "mWifi_daily_00_24_wifi_novel_ratio_mean")
        + _optional_column(result, "mBle_daily_00_24_ble_novel_ratio_mean")
    )

    engineered["semantic_social_day_speech"] = _optional_column(result, "mAmbience_day_09_18_ambience_speech_score_mean")
    engineered["semantic_social_evening_music"] = _optional_column(result, "mAmbience_evening_18_21_ambience_music_score_mean")
    engineered["semantic_social_sleep_silence"] = _optional_column(result, "mAmbience_sleep_21_09_ambience_silence_score_mean")
    engineered["semantic_social_sleep_noise"] = _optional_column(result, "mAmbience_sleep_21_09_ambience_sleep_noise_score_mean")
    engineered["semantic_social_vehicle_commute"] = (
        _optional_column(result, "mAmbience_day_09_18_ambience_vehicle_score_mean")
        + _optional_column(result, "mAmbience_evening_18_21_ambience_vehicle_score_mean")
    )
    engineered["semantic_social_isolation_proxy"] = (
        _optional_column(result, "mAmbience_daily_00_24_ambience_silence_score_mean")
        - _optional_column(result, "mAmbience_daily_00_24_ambience_speech_score_mean")
    )

    engineered["semantic_app_late_social_time"] = _optional_column(result, "mUsageStats_late_00_03_usage_social_time_sum")
    engineered["semantic_app_prebed_media_time"] = _optional_column(result, "mUsageStats_prebed_21_24_usage_media_time_sum")
    engineered["semantic_app_day_communication_time"] = _optional_column(result, "mUsageStats_day_09_18_usage_communication_time_sum")
    engineered["semantic_app_game_time_ratio"] = _optional_column(result, "mUsageStats_daily_00_24_usage_game_time_sum") / (
        _optional_column(result, "mUsageStats_daily_00_24_usage_total_time_sum") + 1e-6
    )
    engineered["semantic_app_passive_vs_communication"] = (
        _optional_column(result, "mUsageStats_daily_00_24_usage_media_time_sum")
        + _optional_column(result, "mUsageStats_daily_00_24_usage_social_time_sum")
    ) / (_optional_column(result, "mUsageStats_daily_00_24_usage_communication_time_sum") + 1e-6)
    engineered["semantic_app_top_concentration"] = _optional_column(result, "mUsageStats_daily_00_24_usage_top_app_time_ratio_mean")

    # ── 생리적 리듬 (Physiological Rhythm) ──────────────────────────────
    # 심부 수면 중 하위 10th percentile HR → resting HR proxy
    engineered["semantic_physio_resting_hr"] = _optional_column(result, "wHr_deep_03_06_heart_rate_p10_mean")
    # 활동 시간대 HR / 수면 HR 비율 → 각성-수면 HR 격차
    engineered["semantic_physio_hr_day_sleep_ratio"] = _optional_column(result, "wHr_day_09_18_heart_rate_mean_mean") / (
        _optional_column(result, "wHr_sleep_21_09_heart_rate_mean_mean") + 1e-6
    )
    # 기상 직후 걸음 수 → 아침 활동량
    engineered["semantic_physio_morning_steps"] = _optional_column(result, "wPedo_wake_06_09_step_sum")
    # 낮 걸음 비율 (낮 걸음 / 일일 걸음)
    engineered["semantic_physio_day_step_fraction"] = _optional_column(result, "wPedo_day_09_18_step_sum") / (
        _optional_column(result, "wPedo_daily_00_24_step_sum") + 1e-6
    )
    # 심야 활동 비율 → 수면 방해 지표
    engineered["semantic_physio_late_step_fraction"] = (
        _optional_column(result, "wPedo_late_00_03_step_sum")
        + _optional_column(result, "wPedo_deep_03_06_step_sum")
    ) / (_optional_column(result, "wPedo_daily_00_24_step_sum") + 1e-6)

    # ── 스크린 행동 (Screen Behavior) ────────────────────────────────────
    # 저녁 스크린 사용 강도
    engineered["semantic_screen_evening_sum"] = _optional_column(result, "mScreenStatus_evening_18_21_m_screen_use_sum")
    # 기상 직후 스크린 비율 (일일 평균 대비)
    engineered["semantic_screen_wake_ratio"] = _optional_column(result, "mScreenStatus_wake_06_09_m_screen_use_mean") / (
        _optional_column(result, "mScreenStatus_daily_00_24_m_screen_use_mean") + 1e-6
    )
    # 수면 중 스크린 총합 → 수면 중 각성 지표
    engineered["semantic_screen_sleep_total"] = _optional_column(result, "mScreenStatus_sleep_21_09_m_screen_use_sum")

    # ── 앱 사용 패턴 (App Behavior) ──────────────────────────────────────
    # 건강·루틴 앱 사용 시간 → 자기관리 지표
    engineered["semantic_app_routine_health_time"] = _optional_column(result, "mUsageStats_daily_00_24_usage_routine_health_time_sum")
    # 저녁 소셜 앱 시간
    engineered["semantic_app_evening_social_time"] = _optional_column(result, "mUsageStats_evening_18_21_usage_social_time_sum")
    # 심야 게임 시간 → 수면 방해 지표
    engineered["semantic_app_night_game_time"] = (
        _optional_column(result, "mUsageStats_late_00_03_usage_game_time_sum")
        + _optional_column(result, "mUsageStats_deep_03_06_usage_game_time_sum")
    )
    # 저녁 소셜 집중도 (낮 소셜 대비)
    engineered["semantic_app_social_evening_shift"] = _optional_column(result, "mUsageStats_evening_18_21_usage_social_time_sum") / (
        _optional_column(result, "mUsageStats_day_09_18_usage_social_time_sum") + 1e-6
    )

    # ── Cross-Sensor 조합 ────────────────────────────────────────────────
    # 낮 HR × 스크린 사용량 → 각성·스트레스 proxy
    engineered["semantic_cross_hr_screen_day"] = (
        _optional_column(result, "wHr_day_09_18_heart_rate_mean_mean")
        * _optional_column(result, "mScreenStatus_daily_00_24_m_screen_use_mean")
    )
    # 이동 반경 / 일일 걸음 수 → 교통수단 이용 비율 proxy
    engineered["semantic_cross_mobility_step_ratio"] = _optional_column(result, "mGps_daily_00_24_gps_mobility_radius_mean") / (
        _optional_column(result, "wPedo_daily_00_24_step_sum") + 1e-6
    )
    # 수면 중 빛 max + HR spike → 수면 방해 지표 조합
    engineered["semantic_cross_sleep_disturbance"] = (
        _optional_column(result, "mLight_sleep_21_09_m_light_max")
        + _optional_column(result, "wHr_sleep_21_09_heart_rate_spike_mean")
    )
    # BLE 밀도 × GPS 반경 → 사회적 활동 지표
    engineered["semantic_cross_social_activity"] = (
        _optional_column(result, "mBle_daily_00_24_ble_detected_count_mean")
        * _optional_column(result, "mGps_daily_00_24_gps_mobility_radius_mean")
    )

    return pd.concat([result, pd.DataFrame(engineered, index=result.index)], axis=1)


def build_sensor_features(items_dir: Path) -> pd.DataFrame:
    """Build a single feature table from all available CH2025 parquet files."""

    feature_frames = [_aggregate_table(path) for path in sorted(items_dir.glob("*.parquet"))]
    features = feature_frames[0]
    for frame in feature_frames[1:]:
        features = features.merge(frame, on=["subject_id", "lifelog_date"], how="outer")
    return _add_semantic_cross_features(features)


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
