from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ITEMS = DATA / "ch2025_data_items"
REPORTS = ROOT / "reports"
TABLES = REPORTS / "tables"
SUBMISSIONS = ROOT / "submissions"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
KEYS = ["subject_id", "sleep_date", "lifelog_date"]
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
APP_GROUPS = {
    "social": ("카카오톡", "instagram", "threads", "twitter", "facebook", " x"),
    "media": ("youtube", "tiktok", "웹툰", "music", "netflix", "영상", "뮤직"),
    "communication": ("통화", "전화", "메시지", "message", "mail", "gmail"),
    "game": ("game", "게임", "클래시", "royal", "number match", "whiteout", "merge"),
    "finance_shopping": ("토스", "카드", "pay", "쿠팡", "당근", "쇼핑", "bank", "은행"),
    "routine_health": ("캐시워크", "withings", "시계", "health", "헬스", "만보기"),
    "browser_search": ("naver", "chrome", "삼성 인터넷", "browser", "검색"),
}
AMBIENCE_GROUPS = {
    "speech": ("speech", "conversation", "narration", "monologue"),
    "silence": ("silence",),
    "music": ("music", "television"),
    "vehicle": ("vehicle", "car", "truck", "motor vehicle"),
    "outdoor": ("outside", "wind", "rural", "urban", "natural"),
    "sleep_noise": ("breathing", "snoring", "white noise", "mechanical fan"),
}
DAYPARTS = {
    "night": (0, 6),
    "morning": (6, 12),
    "afternoon": (12, 18),
    "evening": (18, 24),
}
SLEEP_WINDOWS = {
    "shift6_all": tuple(range(24)),
    "shift6_day_09_18": tuple(range(9, 18)),
    "shift6_evening_18_21": tuple(range(18, 21)),
    "shift6_prebed_21_24": tuple(range(21, 24)),
    "shift6_sleep_21_09": (*range(21, 24), *range(0, 9)),
    "shift6_late_00_03": tuple(range(0, 3)),
    "shift6_deep_03_06": tuple(range(3, 6)),
    "shift6_wake_06_09": tuple(range(6, 9)),
}


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)


def clip_proba(p: np.ndarray | pd.Series) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)


def binary_logloss(y_true: pd.Series, p_one: np.ndarray | pd.Series) -> float:
    p_one = clip_proba(p_one)
    return log_loss(y_true, np.column_stack([1 - p_one, p_one]), labels=[0, 1])


def flatten_columns(columns: pd.Index) -> list[str]:
    out = []
    for col in columns:
        if isinstance(col, tuple):
            out.append("_".join(str(part) for part in col if str(part)))
        else:
            out.append(str(col))
    return out


def load_rows() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA / "ch2026_metrics_train.csv", parse_dates=["sleep_date", "lifelog_date"])
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv", parse_dates=["sleep_date", "lifelog_date"])
    all_rows = pd.concat([train[KEYS], sample[KEYS]], ignore_index=True).drop_duplicates()
    all_rows["lifelog_date"] = pd.to_datetime(all_rows["lifelog_date"]).dt.normalize()
    return train, sample, all_rows


def attach_daypart(df: pd.DataFrame) -> pd.DataFrame:
    hour = pd.to_datetime(df["timestamp"]).dt.hour
    labels = np.full(len(df), "all", dtype=object)
    for name, (start, end) in DAYPARTS.items():
        labels[(hour >= start) & (hour < end)] = name
    out = df.copy()
    out["lifelog_date"] = pd.to_datetime(out["timestamp"]).dt.normalize()
    out["daypart"] = labels
    return out


def attach_shifted_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"])
    out["lifelog_date"] = (ts - pd.Timedelta(hours=6)).dt.normalize()
    out["hour"] = ts.dt.hour
    return out


def aggregate_shifted_numeric_table(table: str, cols: list[str], required: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_parquet(ITEMS / f"{table}.parquet")
    df = attach_shifted_date(df)
    df = df.merge(required, on=["subject_id", "lifelog_date"], how="inner")
    frames = []
    for window, hours in SLEEP_WINDOWS.items():
        part_df = df if window == "shift6_all" else df[df["hour"].isin(hours)]
        if part_df.empty:
            continue
        agg = part_df.groupby(["subject_id", "lifelog_date"])[cols].agg(["count", "mean", "std", "min", "max", "sum", "median"])
        agg.columns = [f"{table}_{window}_{col}" for col in flatten_columns(agg.columns)]
        frames.append(agg.reset_index())
    return merge_feature_frames(required, frames)


def aggregate_shifted_hr(required: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_parquet(ITEMS / "ch2025_wHr.parquet")
    df = attach_shifted_date(df)
    df = df.merge(required, on=["subject_id", "lifelog_date"], how="inner")
    df = df.explode("heart_rate")
    df["heart_rate"] = pd.to_numeric(df["heart_rate"], errors="coerce")
    frames = []
    for window, hours in SLEEP_WINDOWS.items():
        part_df = df if window == "shift6_all" else df[df["hour"].isin(hours)]
        if part_df.empty:
            continue
        agg = part_df.groupby(["subject_id", "lifelog_date"])["heart_rate"].agg(["count", "mean", "std", "min", "max", "median"])
        agg.columns = [f"ch2025_wHr_{window}_heart_rate_{col}" for col in agg.columns]
        frames.append(agg.reset_index())
    return merge_feature_frames(required, frames)


def aggregate_shifted_usage(required: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_parquet(ITEMS / "ch2025_mUsageStats.parquet")
    df = attach_shifted_date(df)
    df = df.merge(required, on=["subject_id", "lifelog_date"], how="inner")
    df["usage_app_count"] = df["m_usage_stats"].map(lambda xs: len(xs) if isinstance(xs, list) else 0)
    df["usage_total_time"] = df["m_usage_stats"].map(
        lambda xs: sum(float(x.get("total_time", 0)) for x in xs) if isinstance(xs, list) else 0.0
    )
    frames = []
    for window, hours in SLEEP_WINDOWS.items():
        part_df = df if window == "shift6_all" else df[df["hour"].isin(hours)]
        if part_df.empty:
            continue
        agg = part_df.groupby(["subject_id", "lifelog_date"])[["usage_app_count", "usage_total_time"]].agg(
            ["count", "mean", "std", "max", "sum", "median"]
        )
        agg.columns = [f"ch2025_mUsageStats_{window}_{col}" for col in flatten_columns(agg.columns)]
        frames.append(agg.reset_index())
    return merge_feature_frames(required, frames)


def app_category(app_name: object) -> str:
    normalized = f" {str(app_name).strip().lower()}"
    for group, keywords in APP_GROUPS.items():
        if any(keyword.lower() in normalized for keyword in keywords):
            return group
    return "other"


def usage_category_totals(xs) -> dict[str, float]:
    totals = {group: 0.0 for group in [*APP_GROUPS.keys(), "other"]}
    if not isinstance(xs, list):
        return totals
    for item in xs:
        if not isinstance(item, dict):
            continue
        try:
            usage_time = float(item.get("total_time", 0.0))
        except (TypeError, ValueError):
            usage_time = 0.0
        totals[app_category(item.get("app_name", ""))] += usage_time
    return totals


def add_usage_semantic_columns(df: pd.DataFrame) -> pd.DataFrame:
    totals = df["m_usage_stats"].map(usage_category_totals)
    out = df.copy()
    for group in [*APP_GROUPS.keys(), "other"]:
        out[f"usage_{group}_time"] = totals.map(lambda item: item[group])
    out["usage_passive_time"] = out["usage_social_time"] + out["usage_media_time"] + out["usage_game_time"]
    out["usage_productive_time"] = out["usage_communication_time"] + out["usage_routine_health_time"]
    out["usage_passive_ratio"] = out["usage_passive_time"] / (out["usage_total_time"] + 1e-6)
    out["usage_social_media_ratio"] = (out["usage_social_time"] + out["usage_media_time"]) / (out["usage_total_time"] + 1e-6)
    return out


def aggregate_usage_semantic(required: pd.DataFrame, shifted: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(ITEMS / "ch2025_mUsageStats.parquet")
    if shifted:
        df = attach_shifted_date(df)
        windows = SLEEP_WINDOWS
        prefix = "ch2025_mUsageStats_sem_shift6"
    else:
        df = attach_daypart(df)
        windows = {"all": tuple(range(24)), **{name: tuple(range(start, end)) for name, (start, end) in DAYPARTS.items()}}
        df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
        prefix = "ch2025_mUsageStats_sem"
    df = df.merge(required, on=["subject_id", "lifelog_date"], how="inner")
    df["usage_app_count"] = df["m_usage_stats"].map(lambda xs: len(xs) if isinstance(xs, list) else 0)
    df["usage_total_time"] = df["m_usage_stats"].map(
        lambda xs: sum(float(x.get("total_time", 0)) for x in xs) if isinstance(xs, list) else 0.0
    )
    df = add_usage_semantic_columns(df)
    value_cols = [
        "usage_total_time",
        "usage_app_count",
        "usage_social_time",
        "usage_media_time",
        "usage_communication_time",
        "usage_game_time",
        "usage_finance_shopping_time",
        "usage_routine_health_time",
        "usage_browser_search_time",
        "usage_passive_time",
        "usage_productive_time",
        "usage_passive_ratio",
        "usage_social_media_ratio",
    ]
    frames = []
    for window, hours in windows.items():
        part_df = df if window == "all" or window == "shift6_all" else df[df["hour"].isin(hours)]
        if part_df.empty:
            continue
        agg = part_df.groupby(["subject_id", "lifelog_date"])[value_cols].agg(["count", "mean", "std", "max", "sum", "median"])
        agg.columns = [f"{prefix}_{window}_{col}" for col in flatten_columns(agg.columns)]
        frames.append(agg.reset_index())
    return merge_feature_frames(required, frames)


def aggregate_numeric_table(table: str, cols: list[str], required: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_parquet(ITEMS / f"{table}.parquet")
    df = attach_daypart(df)
    df = df.merge(required, on=["subject_id", "lifelog_date"], how="inner")
    frames = []
    for part in ["all", *DAYPARTS.keys()]:
        part_df = df if part == "all" else df[df["daypart"] == part]
        if part_df.empty:
            continue
        agg = part_df.groupby(["subject_id", "lifelog_date"])[cols].agg(["count", "mean", "std", "min", "max", "sum"])
        agg.columns = [f"{table}_{part}_{col}" for col in flatten_columns(agg.columns)]
        frames.append(agg.reset_index())
    return merge_feature_frames(required, frames)


def aggregate_list_numeric(
    table: str,
    source_col: str,
    value_name: str,
    required: pd.DataFrame,
    extractor,
) -> pd.DataFrame:
    df = pd.read_parquet(ITEMS / f"{table}.parquet")
    df = attach_daypart(df)
    df = df.merge(required, on=["subject_id", "lifelog_date"], how="inner")
    df[value_name] = df[source_col].map(extractor)
    df["list_len"] = df[source_col].map(lambda x: len(x) if isinstance(x, list) else 0)
    frames = []
    for part in ["all", *DAYPARTS.keys()]:
        part_df = df if part == "all" else df[df["daypart"] == part]
        if part_df.empty:
            continue
        agg = part_df.groupby(["subject_id", "lifelog_date"])[[value_name, "list_len"]].agg(
            ["count", "mean", "std", "min", "max", "sum"]
        )
        agg.columns = [f"{table}_{part}_{col}" for col in flatten_columns(agg.columns)]
        frames.append(agg.reset_index())
    return merge_feature_frames(required, frames)


def gps_row_features(xs) -> dict[str, float]:
    points = [item for item in xs if isinstance(item, dict)] if isinstance(xs, list) else []
    out = {"gps_point_count": float(len(points))}
    for key in ["latitude", "longitude", "altitude", "speed"]:
        values = np.asarray([float(item.get(key, np.nan)) for item in points], dtype=float)
        values = values[np.isfinite(values)]
        out[f"gps_{key}_mean"] = float(values.mean()) if len(values) else np.nan
        out[f"gps_{key}_std"] = float(values.std()) if len(values) else np.nan
        out[f"gps_{key}_min"] = float(values.min()) if len(values) else np.nan
        out[f"gps_{key}_max"] = float(values.max()) if len(values) else np.nan
        out[f"gps_{key}_range"] = float(values.max() - values.min()) if len(values) else np.nan
    out["gps_mobility_radius"] = float(np.sqrt((out.get("gps_latitude_range") or 0.0) ** 2 + (out.get("gps_longitude_range") or 0.0) ** 2))
    return out


def aggregate_gps_semantic(required: pd.DataFrame, shifted: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(ITEMS / "ch2025_mGps.parquet")
    if shifted:
        df = attach_shifted_date(df)
        windows = SLEEP_WINDOWS
        prefix = "ch2025_mGps_sem_shift6"
    else:
        df = attach_daypart(df)
        df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
        windows = {"all": tuple(range(24)), **{name: tuple(range(start, end)) for name, (start, end) in DAYPARTS.items()}}
        prefix = "ch2025_mGps_sem"
    semantic = pd.DataFrame([gps_row_features(xs) for xs in df["m_gps"]], index=df.index)
    df = pd.concat([df, semantic], axis=1)
    night_home = (
        df[df["hour"].isin(range(0, 6))]
        .groupby("subject_id")[["gps_latitude_mean", "gps_longitude_mean"]]
        .median()
        .rename(columns={"gps_latitude_mean": "home_latitude", "gps_longitude_mean": "home_longitude"})
    )
    df = df.merge(night_home, on="subject_id", how="left")
    df["gps_home_distance"] = np.sqrt(
        (df["gps_latitude_mean"] - df["home_latitude"]) ** 2 + (df["gps_longitude_mean"] - df["home_longitude"]) ** 2
    )
    df = df.merge(required, on=["subject_id", "lifelog_date"], how="inner")
    value_cols = [*semantic.columns, "gps_home_distance"]
    frames = []
    for window, hours in windows.items():
        part_df = df if window == "all" or window == "shift6_all" else df[df["hour"].isin(hours)]
        if part_df.empty:
            continue
        agg = part_df.groupby(["subject_id", "lifelog_date"])[value_cols].agg(["count", "mean", "std", "max", "sum", "median"])
        agg.columns = [f"{prefix}_{window}_{flat}" for flat in flatten_columns(agg.columns)]
        frames.append(agg.reset_index())
    return merge_feature_frames(required, frames)


def aggregate_hr(required: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_parquet(ITEMS / "ch2025_wHr.parquet")
    df = attach_daypart(df)
    df = df.merge(required, on=["subject_id", "lifelog_date"], how="inner")
    df = df.explode("heart_rate")
    df["heart_rate"] = pd.to_numeric(df["heart_rate"], errors="coerce")
    frames = []
    for part in ["all", *DAYPARTS.keys()]:
        part_df = df if part == "all" else df[df["daypart"] == part]
        if part_df.empty:
            continue
        agg = part_df.groupby(["subject_id", "lifelog_date"])["heart_rate"].agg(["count", "mean", "std", "min", "max"])
        agg.columns = [f"ch2025_wHr_{part}_heart_rate_{col}" for col in agg.columns]
        frames.append(agg.reset_index())
    return merge_feature_frames(required, frames)


def aggregate_usage(required: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_parquet(ITEMS / "ch2025_mUsageStats.parquet")
    df = attach_daypart(df)
    df = df.merge(required, on=["subject_id", "lifelog_date"], how="inner")
    df["usage_app_count"] = df["m_usage_stats"].map(lambda xs: len(xs) if isinstance(xs, list) else 0)
    df["usage_total_time"] = df["m_usage_stats"].map(
        lambda xs: sum(float(x.get("total_time", 0)) for x in xs) if isinstance(xs, list) else 0.0
    )
    frames = []
    for part in ["all", *DAYPARTS.keys()]:
        part_df = df if part == "all" else df[df["daypart"] == part]
        if part_df.empty:
            continue
        agg = part_df.groupby(["subject_id", "lifelog_date"])[["usage_app_count", "usage_total_time"]].agg(
            ["count", "mean", "std", "max", "sum"]
        )
        agg.columns = [f"ch2025_mUsageStats_{part}_{col}" for col in flatten_columns(agg.columns)]
        frames.append(agg.reset_index())
    return merge_feature_frames(required, frames)


def aggregate_ble_wifi(table: str, col: str, required: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_parquet(ITEMS / f"{table}.parquet")
    df = attach_daypart(df)
    df = df.merge(required, on=["subject_id", "lifelog_date"], how="inner")
    df["device_count"] = df[col].map(lambda xs: len(xs) if isinstance(xs, list) else 0)
    df["rssi_mean"] = df[col].map(
        lambda xs: np.mean([d.get("rssi", np.nan) for d in xs]) if isinstance(xs, list) and xs else np.nan
    )
    df["rssi_max"] = df[col].map(
        lambda xs: np.nanmax([d.get("rssi", np.nan) for d in xs]) if isinstance(xs, list) and xs else np.nan
    )
    frames = []
    for part in ["all", *DAYPARTS.keys()]:
        part_df = df if part == "all" else df[df["daypart"] == part]
        if part_df.empty:
            continue
        agg = part_df.groupby(["subject_id", "lifelog_date"])[["device_count", "rssi_mean", "rssi_max"]].agg(
            ["count", "mean", "std", "max", "sum"]
        )
        agg.columns = [f"{table}_{part}_{flat}" for flat in flatten_columns(agg.columns)]
        frames.append(agg.reset_index())
    return merge_feature_frames(required, frames)


def aggregate_network_semantic(table: str, col: str, id_key: str, required: pd.DataFrame, shifted: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(ITEMS / f"{table}.parquet")
    if shifted:
        df = attach_shifted_date(df)
        windows = SLEEP_WINDOWS
        prefix = f"{table}_sem_shift6"
    else:
        df = attach_daypart(df)
        df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
        windows = {"all": tuple(range(24)), **{name: tuple(range(start, end)) for name, (start, end) in DAYPARTS.items()}}
        prefix = f"{table}_sem"

    # Familiar devices are estimated from the full train+sample sensor period per subject.
    counters: dict[str, dict[str, int]] = {}
    for sid, xs in zip(df["subject_id"], df[col]):
        counter = counters.setdefault(str(sid), {})
        if not isinstance(xs, list):
            continue
        for item in xs:
            if isinstance(item, dict) and item.get(id_key) is not None:
                identifier = str(item[id_key])
                counter[identifier] = counter.get(identifier, 0) + 1
    threshold = 20 if table.endswith("Wifi") else 8
    familiar = {sid: {identifier for identifier, count in counter.items() if count >= threshold} for sid, counter in counters.items()}

    def row_features(sid, xs):
        if not isinstance(xs, list):
            xs = []
        ids = [str(item[id_key]) for item in xs if isinstance(item, dict) and item.get(id_key) is not None]
        rssi = [float(item.get("rssi", np.nan)) for item in xs if isinstance(item, dict)]
        rssi = np.asarray(rssi, dtype=float)
        rssi = rssi[np.isfinite(rssi)]
        familiar_count = sum(identifier in familiar.get(str(sid), set()) for identifier in ids)
        return {
            "detected_count": len(xs),
            "unique_count": len(set(ids)),
            "familiar_ratio": familiar_count / (len(ids) + 1e-6),
            "novel_ratio": 1.0 - familiar_count / (len(ids) + 1e-6),
            "rssi_mean": float(rssi.mean()) if len(rssi) else np.nan,
            "rssi_max": float(rssi.max()) if len(rssi) else np.nan,
            "strong_rssi_count": float(np.sum(rssi >= -65)) if len(rssi) else 0.0,
        }

    semantic = pd.DataFrame([row_features(sid, xs) for sid, xs in zip(df["subject_id"], df[col])], index=df.index)
    df = pd.concat([df, semantic], axis=1)
    df = df.merge(required, on=["subject_id", "lifelog_date"], how="inner")
    value_cols = list(semantic.columns)
    frames = []
    for window, hours in windows.items():
        part_df = df if window == "all" or window == "shift6_all" else df[df["hour"].isin(hours)]
        if part_df.empty:
            continue
        agg = part_df.groupby(["subject_id", "lifelog_date"])[value_cols].agg(["count", "mean", "std", "max", "sum", "median"])
        agg.columns = [f"{prefix}_{window}_{flat}" for flat in flatten_columns(agg.columns)]
        frames.append(agg.reset_index())
    return merge_feature_frames(required, frames)


def aggregate_ambience(required: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_parquet(ITEMS / "ch2025_mAmbience.parquet")
    df = attach_daypart(df)
    df = df.merge(required, on=["subject_id", "lifelog_date"], how="inner")
    df["ambience_top_score"] = df["m_ambience"].map(
        lambda xs: float(xs[0][1]) if isinstance(xs, list) and xs and len(xs[0]) > 1 else np.nan
    )
    df["ambience_music_score"] = df["m_ambience"].map(lambda xs: ambience_score(xs, "Music"))
    df["ambience_speech_score"] = df["m_ambience"].map(lambda xs: ambience_score(xs, "Speech"))
    df["ambience_vehicle_score"] = df["m_ambience"].map(lambda xs: ambience_score(xs, "Vehicle"))
    frames = []
    for part in ["all", *DAYPARTS.keys()]:
        part_df = df if part == "all" else df[df["daypart"] == part]
        if part_df.empty:
            continue
        cols = ["ambience_top_score", "ambience_music_score", "ambience_speech_score", "ambience_vehicle_score"]
        agg = part_df.groupby(["subject_id", "lifelog_date"])[cols].agg(["count", "mean", "std", "max", "sum"])
        agg.columns = [f"ch2025_mAmbience_{part}_{flat}" for flat in flatten_columns(agg.columns)]
        frames.append(agg.reset_index())
    return merge_feature_frames(required, frames)


def aggregate_shifted_ambience(required: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_parquet(ITEMS / "ch2025_mAmbience.parquet")
    df = attach_shifted_date(df)
    df = df.merge(required, on=["subject_id", "lifelog_date"], how="inner")
    df["ambience_top_score"] = df["m_ambience"].map(
        lambda xs: float(xs[0][1]) if isinstance(xs, list) and xs and len(xs[0]) > 1 else np.nan
    )
    df["ambience_music_score"] = df["m_ambience"].map(lambda xs: ambience_score(xs, "Music"))
    df["ambience_speech_score"] = df["m_ambience"].map(lambda xs: ambience_score(xs, "Speech"))
    df["ambience_vehicle_score"] = df["m_ambience"].map(lambda xs: ambience_score(xs, "Vehicle"))
    frames = []
    for window, hours in SLEEP_WINDOWS.items():
        part_df = df if window == "shift6_all" else df[df["hour"].isin(hours)]
        if part_df.empty:
            continue
        cols = ["ambience_top_score", "ambience_music_score", "ambience_speech_score", "ambience_vehicle_score"]
        agg = part_df.groupby(["subject_id", "lifelog_date"])[cols].agg(["count", "mean", "std", "max", "sum", "median"])
        agg.columns = [f"ch2025_mAmbience_{window}_{flat}" for flat in flatten_columns(agg.columns)]
        frames.append(agg.reset_index())
    return merge_feature_frames(required, frames)


def ambience_group_scores(xs) -> dict[str, float]:
    scores = {group: 0.0 for group in AMBIENCE_GROUPS}
    if not isinstance(xs, list):
        return scores
    for item in xs:
        if len(item) < 2:
            continue
        label = str(item[0]).lower()
        try:
            score = float(item[1])
        except (TypeError, ValueError):
            continue
        for group, keywords in AMBIENCE_GROUPS.items():
            if any(keyword in label for keyword in keywords):
                scores[group] += score
    return scores


def aggregate_ambience_semantic(required: pd.DataFrame, shifted: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(ITEMS / "ch2025_mAmbience.parquet")
    if shifted:
        df = attach_shifted_date(df)
        windows = SLEEP_WINDOWS
        prefix = "ch2025_mAmbience_sem_shift6"
    else:
        df = attach_daypart(df)
        df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
        windows = {"all": tuple(range(24)), **{name: tuple(range(start, end)) for name, (start, end) in DAYPARTS.items()}}
        prefix = "ch2025_mAmbience_sem"
    df = df.merge(required, on=["subject_id", "lifelog_date"], how="inner")
    group_scores = df["m_ambience"].map(ambience_group_scores)
    for group in AMBIENCE_GROUPS:
        df[f"ambience_{group}_score"] = group_scores.map(lambda item: item[group])
    df["ambience_social_noise"] = df["ambience_speech_score"] + df["ambience_music_score"]
    df["ambience_sleep_disturbance"] = df["ambience_sleep_noise_score"] + df["ambience_vehicle_score"]
    value_cols = [
        *[f"ambience_{group}_score" for group in AMBIENCE_GROUPS],
        "ambience_social_noise",
        "ambience_sleep_disturbance",
    ]
    frames = []
    for window, hours in windows.items():
        part_df = df if window == "all" or window == "shift6_all" else df[df["hour"].isin(hours)]
        if part_df.empty:
            continue
        agg = part_df.groupby(["subject_id", "lifelog_date"])[value_cols].agg(["count", "mean", "std", "max", "sum", "median"])
        agg.columns = [f"{prefix}_{window}_{col}" for col in flatten_columns(agg.columns)]
        frames.append(agg.reset_index())
    return merge_feature_frames(required, frames)


def ambience_score(xs, label: str) -> float:
    if not isinstance(xs, list):
        return np.nan
    for item in xs:
        if len(item) >= 2 and item[0] == label:
            return float(item[1])
    return 0.0


def merge_feature_frames(base: pd.DataFrame, frames: list[pd.DataFrame]) -> pd.DataFrame:
    out = base[["subject_id", "lifelog_date"]].drop_duplicates().copy()
    for frame in frames:
        out = out.merge(frame, on=["subject_id", "lifelog_date"], how="left")
    return out


def build_features(all_rows: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    path = TABLES / "model_features_v3_semantic.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)

    required = all_rows[["subject_id", "lifelog_date"]].drop_duplicates().copy()
    required["lifelog_date"] = pd.to_datetime(required["lifelog_date"]).dt.normalize()

    frames = [
        aggregate_numeric_table("ch2025_mACStatus", ["m_charging"], required),
        aggregate_numeric_table("ch2025_mActivity", ["m_activity"], required),
        aggregate_numeric_table("ch2025_mLight", ["m_light"], required),
        aggregate_numeric_table("ch2025_mScreenStatus", ["m_screen_use"], required),
        aggregate_numeric_table("ch2025_wLight", ["w_light"], required),
        aggregate_numeric_table(
            "ch2025_wPedo",
            ["step", "step_frequency", "running_step", "walking_step", "distance", "speed", "burned_calories"],
            required,
        ),
        aggregate_hr(required),
        aggregate_list_numeric(
            "ch2025_mGps",
            "m_gps",
            "gps_speed_mean",
            required,
            lambda xs: np.mean([d.get("speed", np.nan) for d in xs]) if isinstance(xs, list) and xs else np.nan,
        ),
        aggregate_ble_wifi("ch2025_mBle", "m_ble", required),
        aggregate_ble_wifi("ch2025_mWifi", "m_wifi", required),
        aggregate_usage(required),
        aggregate_ambience(required),
        aggregate_gps_semantic(required, shifted=False),
        aggregate_network_semantic("ch2025_mBle", "m_ble", "address", required, shifted=False),
        aggregate_network_semantic("ch2025_mWifi", "m_wifi", "bssid", required, shifted=False),
        aggregate_usage_semantic(required, shifted=False),
        aggregate_ambience_semantic(required, shifted=False),
        aggregate_shifted_numeric_table("ch2025_mACStatus", ["m_charging"], required),
        aggregate_shifted_numeric_table("ch2025_mActivity", ["m_activity"], required),
        aggregate_shifted_numeric_table("ch2025_mLight", ["m_light"], required),
        aggregate_shifted_numeric_table("ch2025_mScreenStatus", ["m_screen_use"], required),
        aggregate_shifted_numeric_table("ch2025_wLight", ["w_light"], required),
        aggregate_shifted_numeric_table(
            "ch2025_wPedo",
            ["step", "step_frequency", "running_step", "walking_step", "distance", "speed", "burned_calories"],
            required,
        ),
        aggregate_shifted_hr(required),
        aggregate_shifted_usage(required),
        aggregate_shifted_ambience(required),
        aggregate_gps_semantic(required, shifted=True),
        aggregate_network_semantic("ch2025_mBle", "m_ble", "address", required, shifted=True),
        aggregate_network_semantic("ch2025_mWifi", "m_wifi", "bssid", required, shifted=True),
        aggregate_usage_semantic(required, shifted=True),
        aggregate_ambience_semantic(required, shifted=True),
    ]

    features = required.copy()
    for frame in frames:
        extra_cols = [c for c in frame.columns if c not in {"subject_id", "lifelog_date"}]
        features = features.merge(frame[["subject_id", "lifelog_date", *extra_cols]], on=["subject_id", "lifelog_date"], how="left")

    features["dow"] = features["lifelog_date"].dt.dayofweek
    features["day"] = features["lifelog_date"].dt.day
    features["month"] = features["lifelog_date"].dt.month
    features["days_from_start"] = (features["lifelog_date"] - features["lifelog_date"].min()).dt.days
    subject_dummies = pd.get_dummies(features["subject_id"], prefix="subject", dtype=float)
    features = pd.concat([features, subject_dummies], axis=1)
    features.to_parquet(path, index=False)
    return features


def add_features(rows: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["lifelog_date"] = pd.to_datetime(out["lifelog_date"]).dt.normalize()
    return out.merge(features, on=["lifelog_date"], how="left")


def make_design(train: pd.DataFrame, sample: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_x = train.copy()
    sample_x = sample.copy()
    train_x["lifelog_date"] = pd.to_datetime(train_x["lifelog_date"]).dt.normalize()
    sample_x["lifelog_date"] = pd.to_datetime(sample_x["lifelog_date"]).dt.normalize()

    train_x = train_x.merge(features, on=["subject_id", "lifelog_date"], how="left")
    sample_x = sample_x.merge(features, on=["subject_id", "lifelog_date"], how="left")
    return train_x, sample_x


def candidate_models(random_state: int = 42) -> dict[str, object]:
    return {
        "logreg_l2": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=0.25, solver="lbfgs"),
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=500,
                max_depth=4,
                min_samples_leaf=4,
                max_features=0.45,
                random_state=random_state,
                class_weight="balanced",
            ),
        ),
        "hist_gbdt": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=120,
                learning_rate=0.035,
                max_leaf_nodes=7,
                min_samples_leaf=6,
                l2_regularization=1.0,
                random_state=random_state,
            ),
        ),
    }


def best_blend(y_true: pd.Series, model_p: np.ndarray, prior_p: np.ndarray) -> tuple[float, float]:
    best_alpha, best_score = 1.0, binary_logloss(y_true, model_p)
    for alpha in np.linspace(0, 1, 41):
        p = alpha * model_p + (1 - alpha) * prior_p
        score = binary_logloss(y_true, p)
        if score < best_score:
            best_alpha, best_score = float(alpha), float(score)
    return best_alpha, best_score


def interleaved_validation_mask(train: pd.DataFrame, offset: int = 1) -> pd.Series:
    mask = pd.Series(False, index=train.index)
    ordered = train.sort_values(["subject_id", "sleep_date"])
    for _, group in ordered.groupby("subject_id"):
        pos = np.arange(len(group))
        group_mask = (pos >= 5) & (pos % 3 == offset)
        mask.loc[group.index[group_mask]] = True
    return mask


def late_validation_mask(train: pd.DataFrame, days: int = 7) -> pd.Series:
    max_by_subject = train.groupby("subject_id")["sleep_date"].transform("max")
    return train["sleep_date"] > max_by_subject - pd.Timedelta(days=days)


def subject_hole_validation_mask(train: pd.DataFrame, fold_id: int, n_folds: int = 5) -> pd.Series:
    mask = pd.Series(False, index=train.index)
    block_count = max(n_folds * 2, 4)
    for _, group in train.sort_values(["subject_id", "sleep_date"]).groupby("subject_id"):
        chunks = [chunk for chunk in np.array_split(group.index.to_numpy(), block_count) if len(chunk)]
        for hole_id in (fold_id, fold_id + n_folds):
            if hole_id < len(chunks):
                mask.loc[chunks[hole_id]] = True
    return mask


def validation_splits(train: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "interleaved_offset1": interleaved_validation_mask(train, offset=1),
        "interleaved_offset2": interleaved_validation_mask(train, offset=2),
        "subject_hole0": subject_hole_validation_mask(train, fold_id=0, n_folds=5),
        "subject_hole1": subject_hole_validation_mask(train, fold_id=1, n_folds=5),
        "subject_hole2": subject_hole_validation_mask(train, fold_id=2, n_folds=5),
        "late7": late_validation_mask(train, days=7),
    }


def target_history_features(rows: pd.DataFrame, known: pd.DataFrame, target: str) -> pd.DataFrame:
    out = pd.DataFrame(index=rows.index)
    out[f"{target}_known_mean"] = float(known[target].mean()) if len(known) else 0.5
    out[f"{target}_known_count"] = 0.0
    out[f"{target}_prev_label"] = np.nan
    out[f"{target}_prev_gap"] = np.nan
    out[f"{target}_next_label"] = np.nan
    out[f"{target}_next_gap"] = np.nan
    out[f"{target}_neighbor_mean"] = np.nan

    known = known.sort_values(["subject_id", "sleep_date"])
    row_dates = pd.to_datetime(rows["sleep_date"])
    for sid, idx in rows.groupby("subject_id").groups.items():
        subject_known = known[known["subject_id"] == sid]
        if subject_known.empty:
            continue
        dates = pd.to_datetime(subject_known["sleep_date"]).to_numpy(dtype="datetime64[D]")
        values = subject_known[target].to_numpy(dtype=float)
        query_dates = row_dates.loc[idx].to_numpy(dtype="datetime64[D]")
        positions = np.searchsorted(dates, query_dates, side="left")
        subject_mean = float(np.mean(values))
        out.loc[idx, f"{target}_known_mean"] = subject_mean
        out.loc[idx, f"{target}_known_count"] = len(values)
        for row_idx, pos, q_date in zip(idx, positions, query_dates):
            prev_i = pos - 1
            next_i = pos
            prev_value = np.nan
            next_value = np.nan
            if prev_i >= 0:
                prev_value = values[prev_i]
                out.loc[row_idx, f"{target}_prev_label"] = prev_value
                out.loc[row_idx, f"{target}_prev_gap"] = int((q_date - dates[prev_i]).astype("timedelta64[D]").astype(int))
            if next_i < len(values):
                if dates[next_i] == q_date:
                    next_i += 1
                if next_i < len(values):
                    next_value = values[next_i]
                    out.loc[row_idx, f"{target}_next_label"] = next_value
                    out.loc[row_idx, f"{target}_next_gap"] = int((dates[next_i] - q_date).astype("timedelta64[D]").astype(int))
            if not np.isnan(prev_value) and not np.isnan(next_value):
                out.loc[row_idx, f"{target}_neighbor_mean"] = (prev_value + next_value) / 2
            elif not np.isnan(prev_value):
                out.loc[row_idx, f"{target}_neighbor_mean"] = prev_value
            elif not np.isnan(next_value):
                out.loc[row_idx, f"{target}_neighbor_mean"] = next_value

    return out.fillna({f"{target}_neighbor_mean": out[f"{target}_known_mean"]})


def label_knn_prior(rows: pd.DataFrame, known: pd.DataFrame, target: str) -> np.ndarray:
    global_prior = float(known[target].mean()) if len(known) else 0.5
    preds = []
    for _, row in rows.iterrows():
        subject_known = known[known["subject_id"] == row["subject_id"]]
        if subject_known.empty:
            preds.append(global_prior)
            continue
        distance = (subject_known["sleep_date"] - row["sleep_date"]).abs().dt.days.to_numpy(dtype=float)
        values = subject_known[target].to_numpy(dtype=float)
        weights = 1.0 / np.power(distance + 1.0, 1.0)
        pred = float(np.average(values, weights=weights)) if weights.sum() > 0 else float(subject_known[target].mean())
        subject_prior = float(subject_known[target].mean())
        preds.append(0.88 * pred + 0.12 * subject_prior)
    return clip_proba(np.asarray(preds, dtype=float))


def feature_columns(train_x: pd.DataFrame) -> list[str]:
    cols = [
        c
        for c in train_x.columns
        if c not in {*KEYS, *TARGETS}
        and pd.api.types.is_numeric_dtype(train_x[c])
        and train_x[c].notna().any()
    ]
    return sorted(set(cols))


def evaluate_split(
    split_name: str,
    valid_mask: pd.Series,
    train: pd.DataFrame,
    sample: pd.DataFrame,
    train_x: pd.DataFrame,
    sample_x: pd.DataFrame,
    feature_cols: list[str],
    make_submission: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    fit_mask = ~valid_mask

    submission = sample[KEYS].copy()
    rows = []
    for target in TARGETS:
        y_fit = train.loc[fit_mask, target]
        y_valid = train.loc[valid_mask, target]
        global_prior = float(y_fit.mean())
        subject_prior = train.loc[fit_mask].groupby("subject_id")[target].mean()
        valid_prior = train.loc[valid_mask, "subject_id"].map(subject_prior).fillna(global_prior).to_numpy()
        sample_prior = sample["subject_id"].map(train.groupby("subject_id")[target].mean()).fillna(float(train[target].mean())).to_numpy()
        fit_history = target_history_features(train.loc[fit_mask], train.loc[fit_mask], target)
        valid_history = target_history_features(train.loc[valid_mask], train.loc[fit_mask], target)
        all_history = target_history_features(train, train, target)
        sample_history = target_history_features(sample, train, target)
        valid_neighbor_prior = valid_history[f"{target}_neighbor_mean"].to_numpy(dtype=float)
        valid_neighbor_prior = np.where(np.isnan(valid_neighbor_prior), valid_prior, valid_neighbor_prior)
        sample_neighbor_prior = sample_history[f"{target}_neighbor_mean"].to_numpy(dtype=float)
        sample_neighbor_prior = np.where(np.isnan(sample_neighbor_prior), sample_prior, sample_neighbor_prior)
        valid_knn_prior = label_knn_prior(train.loc[valid_mask], train.loc[fit_mask], target)
        sample_knn_prior = label_knn_prior(sample, train, target)
        prior_candidates = {
            "subject_prior": (valid_prior, sample_prior),
            "neighbor_prior": (valid_neighbor_prior, sample_neighbor_prior),
            "subject_neighbor_prior": (
                0.5 * valid_prior + 0.5 * valid_neighbor_prior,
                0.5 * sample_prior + 0.5 * sample_neighbor_prior,
            ),
            "label_knn_prior": (valid_knn_prior, sample_knn_prior),
            "subject_knn_prior": (
                0.5 * valid_prior + 0.5 * valid_knn_prior,
                0.5 * sample_prior + 0.5 * sample_knn_prior,
            ),
            "neighbor_knn_prior": (
                0.5 * valid_neighbor_prior + 0.5 * valid_knn_prior,
                0.5 * sample_neighbor_prior + 0.5 * sample_knn_prior,
            ),
        }

        best = None
        best_sample_pred = sample_prior.copy()
        for prior_name, (valid_p, sample_p) in prior_candidates.items():
            score = binary_logloss(y_valid, valid_p)
            if best is None or score < best["valid_log_loss"]:
                best = {
                    "target": target,
                    "model": prior_name,
                    "alpha": 0.0,
                    "valid_log_loss": score,
                    "valid_model_log_loss": np.nan,
                }
                best_sample_pred = sample_p.copy()
        assert best is not None

        for name, model in candidate_models().items():
            x_fit = pd.concat([train_x.loc[fit_mask, feature_cols].reset_index(drop=True), fit_history.reset_index(drop=True)], axis=1)
            x_valid = pd.concat([train_x.loc[valid_mask, feature_cols].reset_index(drop=True), valid_history.reset_index(drop=True)], axis=1)
            model.fit(x_fit, y_fit)
            valid_p = model.predict_proba(x_valid)[:, 1]
            model_score = binary_logloss(y_valid, valid_p)
            for prior_name, (valid_prior_p, sample_prior_p) in prior_candidates.items():
                alpha, blend_score = best_blend(y_valid, valid_p, valid_prior_p)
                if blend_score < best["valid_log_loss"]:
                    final_model = candidate_models()[name]
                    x_all = pd.concat([train_x[feature_cols].reset_index(drop=True), all_history.reset_index(drop=True)], axis=1)
                    x_sample = pd.concat([sample_x[feature_cols].reset_index(drop=True), sample_history.reset_index(drop=True)], axis=1)
                    final_model.fit(x_all, train[target])
                    sample_model_p = final_model.predict_proba(x_sample)[:, 1]
                    best_sample_pred = alpha * sample_model_p + (1 - alpha) * sample_prior_p
                    best = {
                        "target": target,
                        "model": f"{name}+{prior_name}",
                        "alpha": alpha,
                        "valid_log_loss": blend_score,
                        "valid_model_log_loss": model_score,
                    }

        submission[target] = clip_proba(best_sample_pred)
        best["split"] = split_name
        best["valid_rows"] = int(valid_mask.sum())
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
    return scores, submission if make_submission else None


def train_and_predict(train: pd.DataFrame, sample: pd.DataFrame, train_x: pd.DataFrame, sample_x: pd.DataFrame) -> pd.DataFrame:
    feature_cols = feature_columns(train_x)
    split_scores = []
    split_submissions = {}

    for split_name, valid_mask in validation_splits(train).items():
        scores, submission = evaluate_split(
            split_name,
            valid_mask,
            train,
            sample,
            train_x,
            sample_x,
            feature_cols,
            make_submission=True,
        )
        split_scores.append(scores)
        if submission is not None:
            split_submissions[split_name] = submission

    all_scores = pd.concat(split_scores, ignore_index=True)
    all_scores.to_csv(TABLES / "model_validation_scores_by_split.csv", index=False)
    primary_scores = all_scores[all_scores["split"] == "interleaved_offset1"].copy()
    primary_scores.to_csv(TABLES / "model_validation_scores.csv", index=False)
    if "interleaved_offset1" not in split_submissions:
        raise RuntimeError("Primary submission was not created.")

    primary_submission = split_submissions["interleaved_offset1"]
    primary_submission.to_csv(SUBMISSIONS / "baseline_submission.csv", index=False)
    primary_submission.to_csv(SUBMISSIONS / "primary_interleaved_submission.csv", index=False)
    for split_name, submission in split_submissions.items():
        submission.to_csv(SUBMISSIONS / f"{split_name}_submission.csv", index=False)

    ensemble = primary_submission[KEYS].copy()
    weights = {
        "interleaved_offset1": 0.25,
        "interleaved_offset2": 0.25,
        "subject_hole0": 0.15,
        "subject_hole1": 0.15,
        "subject_hole2": 0.10,
        "late7": 0.10,
    }
    for target in TARGETS:
        values = np.zeros(len(ensemble), dtype=float)
        for split_name, weight in weights.items():
            values += weight * split_submissions[split_name][target].to_numpy()
        ensemble[target] = clip_proba(values)
    ensemble.to_csv(SUBMISSIONS / "robust_ensemble_submission.csv", index=False)

    meta = {
        "feature_count": len(feature_cols),
        "primary_validation_split": "interleaved_offset1",
        "validation_average_log_loss": float(primary_scores.query("target == 'average'")["valid_log_loss"].iloc[0]),
        "split_average_log_loss": {
            row["split"]: float(row["valid_log_loss"])
            for row in all_scores[all_scores["target"] == "average"].to_dict("records")
        },
        "submission_path": str(SUBMISSIONS / "baseline_submission.csv"),
        "primary_submission_path": str(SUBMISSIONS / "primary_interleaved_submission.csv"),
        "robust_ensemble_submission_path": str(SUBMISSIONS / "robust_ensemble_submission.csv"),
    }
    (REPORTS / "model_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return all_scores


def main() -> None:
    ensure_dirs()
    train, sample, all_rows = load_rows()
    features = build_features(all_rows)
    train_x, sample_x = make_design(train, sample, features)
    scores = train_and_predict(train, sample, train_x, sample_x)
    print(scores.to_string(index=False))
    print(f"Wrote {SUBMISSIONS / 'baseline_submission.csv'}")


if __name__ == "__main__":
    main()
