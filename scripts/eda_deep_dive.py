from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ITEMS = DATA / "ch2025_data_items"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"
os.environ.setdefault("MPLCONFIGDIR", str(REPORTS / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
KEYS = ["subject_id", "sleep_date", "lifelog_date"]


def flatten_columns(columns: pd.Index) -> list[str]:
    flat = []
    for col in columns:
        if isinstance(col, tuple):
            flat.append("_".join(str(part) for part in col if str(part)))
        else:
            flat.append(str(col))
    return flat


def to_md(df: pd.DataFrame, index: bool = True) -> str:
    frame = df.copy()
    if index:
        frame = frame.reset_index()
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in frame.to_numpy())
    return "\n".join(lines)


def ensure_dirs() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)


def binary_logloss(y_true: pd.Series, p_one: pd.Series | np.ndarray) -> float:
    p_one = np.clip(np.asarray(p_one, dtype=float), 1e-5, 1 - 1e-5)
    return log_loss(y_true, np.column_stack([1 - p_one, p_one]), labels=[0, 1])


def table_inventory() -> pd.DataFrame:
    rows = []
    for path in sorted(ITEMS.glob("*.parquet")):
        df = pd.read_parquet(path)
        ts = pd.to_datetime(df["timestamp"])
        rows.append(
            {
                "table": path.stem,
                "rows": len(df),
                "subjects": df["subject_id"].nunique(),
                "start": ts.min(),
                "end": ts.max(),
                "columns": ", ".join(df.columns),
                "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            }
        )
    return pd.DataFrame(rows)


def read_metric_description() -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(DATA / "ch2026_metrics_description.pdf")
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text.replace("?셲", "'s").replace("’", "'")
    except Exception as exc:  # pragma: no cover - report convenience
        return f"Failed to read PDF: {exc}"


def load_labels() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA / "ch2026_metrics_train.csv", parse_dates=["sleep_date", "lifelog_date"])
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv", parse_dates=["sleep_date", "lifelog_date"])
    return train, sample


def sensor_daily_coverage(rows: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for path in sorted(ITEMS.glob("*.parquet")):
        table = path.stem
        df = pd.read_parquet(path, columns=["subject_id", "timestamp"])
        df["lifelog_date"] = pd.to_datetime(df["timestamp"]).dt.normalize()
        cov = (
            df.groupby(["subject_id", "lifelog_date"])
            .size()
            .rename(f"{table}_rows")
            .reset_index()
        )
        frames.append(cov)

    out = rows[["subject_id", "lifelog_date"]].drop_duplicates().copy()
    out["lifelog_date"] = pd.to_datetime(out["lifelog_date"]).dt.normalize()
    for cov in frames:
        out = out.merge(cov, on=["subject_id", "lifelog_date"], how="left")
    count_cols = [c for c in out.columns if c.endswith("_rows")]
    out[count_cols] = out[count_cols].fillna(0).astype(int)
    return out


def aggregate_simple_daily_features(rows: pd.DataFrame) -> pd.DataFrame:
    base = rows[["subject_id", "lifelog_date"]].drop_duplicates().copy()
    base["lifelog_date"] = pd.to_datetime(base["lifelog_date"]).dt.normalize()

    feature_frames = []
    simple_specs = {
        "ch2025_mACStatus": ["m_charging"],
        "ch2025_mActivity": ["m_activity"],
        "ch2025_mLight": ["m_light"],
        "ch2025_mScreenStatus": ["m_screen_use"],
        "ch2025_wLight": ["w_light"],
        "ch2025_wPedo": ["step", "step_frequency", "running_step", "walking_step", "distance", "speed", "burned_calories"],
    }

    for table, cols in simple_specs.items():
        path = ITEMS / f"{table}.parquet"
        df = pd.read_parquet(path)
        df["lifelog_date"] = pd.to_datetime(df["timestamp"]).dt.normalize()
        agg = df.groupby(["subject_id", "lifelog_date"])[cols].agg(["count", "mean", "std", "min", "max", "sum"])
        agg.columns = [f"{table}_{col}_{stat}" for col, stat in agg.columns]
        feature_frames.append(agg.reset_index())

    for table, col in [("ch2025_wHr", "heart_rate"), ("ch2025_mGps", "m_gps")]:
        df = pd.read_parquet(ITEMS / f"{table}.parquet")
        df["lifelog_date"] = pd.to_datetime(df["timestamp"]).dt.normalize()
        if col == "heart_rate":
            expanded = df.explode(col)
            expanded[col] = pd.to_numeric(expanded[col], errors="coerce")
            agg = expanded.groupby(["subject_id", "lifelog_date"])[col].agg(["count", "mean", "std", "min", "max"])
        else:
            df["gps_points"] = df[col].map(lambda x: len(x) if isinstance(x, (list, tuple, np.ndarray)) else 0)
            df["gps_speed_mean"] = df[col].map(
                lambda xs: np.mean([d.get("speed", np.nan) for d in xs]) if isinstance(xs, list) and xs else np.nan
            )
            agg = df.groupby(["subject_id", "lifelog_date"])[["gps_points", "gps_speed_mean"]].agg(["count", "mean", "std", "max", "sum"])
        agg.columns = [f"{table}_{col}" for col in flatten_columns(agg.columns)]
        feature_frames.append(agg.reset_index())

    out = base
    for feat in feature_frames:
        out = out.merge(feat, on=["subject_id", "lifelog_date"], how="left")
    return out


def validation_baselines(train: pd.DataFrame) -> pd.DataFrame:
    train = train.sort_values(["subject_id", "sleep_date"]).copy()
    max_by_subject = train.groupby("subject_id")["sleep_date"].transform("max")
    valid = train[train["sleep_date"] > max_by_subject - pd.Timedelta(days=7)].copy()
    fit = train[train["sleep_date"] <= max_by_subject - pd.Timedelta(days=7)].copy()

    rows = []
    for target in TARGETS:
        global_p = fit[target].mean()
        rows.append(
            {
                "target": target,
                "baseline": "global_train_prior",
                "log_loss": binary_logloss(valid[target], np.full(len(valid), global_p)),
                "mean_p1": global_p,
            }
        )

        subject_p = fit.groupby("subject_id")[target].mean()
        pred = valid["subject_id"].map(subject_p).fillna(global_p)
        rows.append(
            {
                "target": target,
                "baseline": "subject_prior",
                "log_loss": binary_logloss(valid[target], pred),
                "mean_p1": pred.mean(),
            }
        )

        history = fit.sort_values(["subject_id", "sleep_date"]).groupby("subject_id")[target].last()
        pred = valid["subject_id"].map(history).fillna(global_p).astype(float) * 0.9 + 0.05
        rows.append(
            {
                "target": target,
                "baseline": "last_label_smoothed",
                "log_loss": binary_logloss(valid[target], pred),
                "mean_p1": pred.mean(),
            }
        )

    result = pd.DataFrame(rows)
    result.loc[len(result)] = {
        "target": "average",
        "baseline": "best_per_target_floor",
        "log_loss": result.groupby("target")["log_loss"].min().mean(),
        "mean_p1": np.nan,
    }
    return result


def save_figures(train: pd.DataFrame, coverage: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    target_rates = train[TARGETS].mean().rename("positive_rate").reset_index().rename(columns={"index": "target"})
    plt.figure(figsize=(8, 4.5))
    ax = sns.barplot(data=target_rates, x="target", y="positive_rate", color="#4C78A8")
    ax.axhline(0.5, color="black", linewidth=1, linestyle="--", alpha=0.4)
    ax.set_ylim(0, 1)
    ax.set_title("Target positive class rate")
    plt.tight_layout()
    plt.savefig(FIGURES / "target_positive_rates.png", dpi=160)
    plt.close()

    corr = train[TARGETS].corr()
    plt.figure(figsize=(6.5, 5.5))
    ax = sns.heatmap(corr, vmin=-1, vmax=1, cmap="vlag", annot=True, fmt=".2f", square=True)
    ax.set_title("Target correlation")
    plt.tight_layout()
    plt.savefig(FIGURES / "target_correlation.png", dpi=160)
    plt.close()

    cov_cols = [c for c in coverage.columns if c.endswith("_rows")]
    cov_rate = (coverage[cov_cols] > 0).mean().sort_values(ascending=False).reset_index()
    cov_rate.columns = ["sensor", "day_coverage_rate"]
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=cov_rate, y="sensor", x="day_coverage_rate", color="#59A14F")
    ax.set_xlim(0, 1)
    ax.set_title("Sensor availability by target day")
    plt.tight_layout()
    plt.savefig(FIGURES / "sensor_day_coverage.png", dpi=160)
    plt.close()


def write_report(
    train: pd.DataFrame,
    sample: pd.DataFrame,
    inventory: pd.DataFrame,
    coverage: pd.DataFrame,
    features: pd.DataFrame,
    baselines: pd.DataFrame,
    metric_text: str,
) -> None:
    label_summary = train[TARGETS].agg(["mean", "sum", "count"]).T
    label_summary["negative"] = label_summary["count"] - label_summary["sum"]
    label_summary = label_summary[["count", "sum", "negative", "mean"]].rename(columns={"sum": "positive", "mean": "positive_rate"})

    train_subject = train.groupby("subject_id").agg(
        rows=("sleep_date", "size"),
        start=("sleep_date", "min"),
        end=("sleep_date", "max"),
    )
    sample_subject = sample.groupby("subject_id").agg(
        rows=("sleep_date", "size"),
        start=("sleep_date", "min"),
        end=("sleep_date", "max"),
    )

    coverage_cols = [c for c in coverage.columns if c.endswith("_rows")]
    coverage_summary = pd.DataFrame(
        {
            "day_coverage_rate": (coverage[coverage_cols] > 0).mean(),
            "median_rows_on_available_day": coverage[coverage_cols].replace(0, np.nan).median(),
            "mean_rows_all_days": coverage[coverage_cols].mean(),
        }
    ).sort_values("day_coverage_rate", ascending=False)

    label_summary.to_csv(TABLES / "target_label_summary.csv")
    train_subject.to_csv(TABLES / "train_subject_windows.csv")
    sample_subject.to_csv(TABLES / "sample_subject_windows.csv")
    inventory.to_csv(TABLES / "sensor_inventory.csv", index=False)
    coverage_summary.to_csv(TABLES / "sensor_coverage_summary.csv")
    baselines.to_csv(TABLES / "time_holdout_baselines.csv", index=False)
    features.describe(include="all").T.to_csv(TABLES / "daily_feature_describe.csv")

    average_global = baselines[baselines["baseline"] == "global_train_prior"]["log_loss"].mean()
    average_subject = baselines[baselines["baseline"] == "subject_prior"]["log_loss"].mean()
    best_floor = baselines.query("target == 'average'")["log_loss"].iloc[0]

    md = f"""# ETRI 2026 심층 EDA

로컬 데이터 `{DATA}`를 기준으로 생성한 리포트입니다.

## 대회 구조

- 과제: 7개 이진 지표 `{', '.join(TARGETS)}` 예측
- 평가 지표: 타깃 전체의 Average Log Loss
- 학습 데이터: {len(train):,}행; 제출 데이터: {len(sample):,}행
- Subject 수: 학습 {train['subject_id'].nunique()}명, 제출 {sample['subject_id'].nunique()}명
- 학습 sleep_date 범위: {train['sleep_date'].min().date()} ~ {train['sleep_date'].max().date()}
- 제출 sleep_date 범위: {sample['sleep_date'].min().date()} ~ {sample['sleep_date'].max().date()}

## 지표 정의

```text
2026 챌린지의 7개 지표

Q1: 기상 직후 subject가 인식한 전반적인 수면의 질
    0: 개인 평균보다 낮음
    1: 개인 평균보다 높음

Q2: 취침 직전 subject의 신체적 피로도
    0: 피로도가 높음
    1: 피로도가 낮음

Q3: 취침 직전 subject가 경험한 스트레스 수준
    0: 스트레스가 높음
    1: 스트레스가 낮음

S1: 총 수면시간(TST)에 대한 수면 가이드라인 충족 여부
    0: 권장 기준 미충족
    1: 권장 기준 충족

S2: 수면 효율(SE)에 대한 수면 가이드라인 충족 여부
    0: 부적절
    1: 권장 기준 충족

S3: 수면 지연시간(SOL)에 대한 수면 가이드라인 충족 여부
    0: 부적절
    1: 권장 기준 충족

S4: 수면 중 각성 시간(WASO)에 대한 수면 가이드라인 충족 여부
    0: 부적절
    1: 권장 기준 충족

Q1, Q2, Q3는 실험 전체 기간의 개인 평균을 기준으로 설문 응답을 이진화한 지표입니다.
Q1은 개인의 자기보고 수면 품질이 해당 개인의 전체 기간 평균보다 높은 날에 1,
낮은 날에 0으로 부여됩니다. Q2와 Q3는 피로도와 스트레스가 개인 평균보다 높으면 0,
낮으면 1로 부여됩니다. 즉 설문 기반 지표에서 1은 개인 관점의 긍정적 상태를 의미합니다.
```

## 첫 발견사항

1. 이 데이터는 subject별 짧은 미래 구간을 예측하는 문제입니다. 각 subject마다 과거 라벨이 있고, submission은 같은 subject들의 미래 날짜를 요구합니다.
2. 타깃은 참가자 개인 평균을 기준으로 이진화되어 있으므로, subject별 prior와 최근 이력 피처가 중요할 가능성이 큽니다.
3. Public/Private은 제출 샘플 기준으로 나뉘지만, 로컬 검증은 미래 예측 구조를 흉내 내도록 subject별 시간순 분할을 써야 합니다.
4. Average Log Loss 0.6 이하는 hard 0/1 예측이 아니라 잘 보정된 확률 예측이 필요합니다. 아래 EDA 베이스라인이 첫 번째로 넘어야 할 기준점입니다.

## 타깃 분포

{to_md(label_summary.round(4))}

## 센서 인벤토리

{to_md(inventory[['table', 'rows', 'subjects', 'start', 'end', 'memory_mb']], index=False)}

## 필요한 lifelog 날짜 기준 센서 커버리지

{to_md(coverage_summary.round(3))}

## 시간순 Holdout 베이스라인

검증 방식: subject별 마지막 7개의 라벨 sleep_date를 검증으로 사용했습니다. 의도적으로 시간순 분할을 사용했기 때문에 random CV보다 리더보드의 미래 예측 구조에 더 가깝습니다.

{to_md(baselines.round(4), index=False)}

- Global prior 평균 Log Loss: {average_global:.4f}
- Subject prior 평균 Log Loss: {average_subject:.4f}
- 단순 per-target 최저 베이스라인: {best_floor:.4f}

## 권장 모델링 방향

1. `subject_id + lifelog_date`를 키로 하는 일별 피처 테이블을 구축합니다.
2. 과거 라벨만 사용해 subject별 expanding/rolling target prior를 추가합니다.
3. 수면/피로/스트레스는 하루 전체 평균보다 저녁/야간 행동의 영향을 더 받을 수 있으므로 시간대별 집계 피처를 추가합니다.
4. 시간순 group validation을 사용해 타깃별 calibrated binary model을 학습합니다.
5. 타깃별 확률 보정을 최적화합니다. 0.6 달성은 원시 분류 정확도만큼이나 calibration과 검증 설계에 달려 있을 가능성이 큽니다.

## 생성된 산출물

- `reports/tables/target_label_summary.csv`
- `reports/tables/sensor_inventory.csv`
- `reports/tables/sensor_coverage_summary.csv`
- `reports/tables/time_holdout_baselines.csv`
- `reports/tables/daily_feature_describe.csv`
- `reports/figures/target_positive_rates.png`
- `reports/figures/target_correlation.png`
- `reports/figures/sensor_day_coverage.png`
"""

    (REPORTS / "eda_summary.md").write_text(md, encoding="utf-8")

    metadata = {
        "targets": TARGETS,
        "train_rows": int(len(train)),
        "sample_rows": int(len(sample)),
        "average_global_prior_log_loss": float(average_global),
        "average_subject_prior_log_loss": float(average_subject),
        "best_simple_floor_log_loss": float(best_floor),
    }
    (REPORTS / "eda_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    train, sample = load_labels()
    all_rows = pd.concat([train[KEYS], sample[KEYS]], ignore_index=True)
    inventory = table_inventory()
    coverage = sensor_daily_coverage(all_rows)
    features = aggregate_simple_daily_features(all_rows)
    baselines = validation_baselines(train)
    metric_text = read_metric_description()

    inventory.to_csv(TABLES / "sensor_inventory.csv", index=False)
    coverage.to_parquet(TABLES / "sensor_daily_coverage.parquet", index=False)
    features.to_parquet(TABLES / "daily_simple_features.parquet", index=False)
    save_figures(train, coverage)
    write_report(train, sample, inventory, coverage, features, baselines, metric_text)
    print(f"Wrote {REPORTS / 'eda_summary.md'}")


if __name__ == "__main__":
    main()
