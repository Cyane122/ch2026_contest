from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
TABLES = REPORTS / "tables"
SUBMISSIONS = ROOT / "submissions"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
KEYS = ["subject_id", "sleep_date", "lifelog_date"]


def clip(p: np.ndarray | pd.Series) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)


def binary_logloss(y_true: pd.Series, p_one: np.ndarray | pd.Series) -> float:
    p = clip(p_one)
    y = y_true.to_numpy(dtype=float)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA / "ch2026_metrics_train.csv", parse_dates=["sleep_date", "lifelog_date"])
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv", parse_dates=["sleep_date", "lifelog_date"])
    return train, sample


def temporal_splits(train: pd.DataFrame) -> dict[str, pd.Series]:
    masks: dict[str, pd.Series] = {}
    ordered = train.sort_values(["subject_id", "sleep_date"])
    for offset in [1, 2]:
        mask = pd.Series(False, index=train.index)
        for _, group in ordered.groupby("subject_id"):
            pos = np.arange(len(group))
            mask.loc[group.index[(pos >= 5) & (pos % 3 == offset)]] = True
        masks[f"interleaved_offset{offset}"] = mask

    max_by_subject = train.groupby("subject_id")["sleep_date"].transform("max")
    masks["late7"] = train["sleep_date"] > max_by_subject - pd.Timedelta(days=7)
    return masks


def weights_from_distance(distance: np.ndarray, method: str, tau: float, power: float) -> np.ndarray:
    distance = np.asarray(distance, dtype=float)
    if method == "exp":
        return np.exp(-distance / tau)
    if method == "gaussian":
        return np.exp(-0.5 * (distance / tau) ** 2)
    if method == "inverse":
        return 1.0 / np.power(distance + 1.0, power)
    raise ValueError(f"Unknown method: {method}")


def predict_label_knn(
    query: pd.DataFrame,
    known: pd.DataFrame,
    target: str,
    method: str = "exp",
    tau: float = 14.0,
    power: float = 1.0,
    max_days: int | None = 35,
    min_weight: float = 1e-8,
    shrink: float = 0.05,
    date_col: str = "sleep_date",
) -> np.ndarray:
    global_prior = float(known[target].mean()) if len(known) else 0.5
    preds = []
    for _, row in query.iterrows():
        subject_known = known[known["subject_id"] == row["subject_id"]]
        if subject_known.empty:
            preds.append(global_prior)
            continue
        distance = (subject_known[date_col] - row[date_col]).abs().dt.days.to_numpy(dtype=float)
        values = subject_known[target].to_numpy(dtype=float)
        if max_days is not None:
            keep = distance <= max_days
            if keep.any():
                distance = distance[keep]
                values = values[keep]
        weights = weights_from_distance(distance, method=method, tau=tau, power=power)
        weights = np.where(weights < min_weight, 0.0, weights)
        if weights.sum() <= 0:
            subject_prior = float(subject_known[target].mean())
            pred = subject_prior
        else:
            pred = float(np.average(values, weights=weights))
        subject_prior = float(subject_known[target].mean())
        pred = (1.0 - shrink) * pred + shrink * subject_prior
        preds.append(pred)
    return clip(np.asarray(preds))


def candidate_grid() -> list[dict[str, float | int | str | None]]:
    candidates = []
    for method in ["exp", "gaussian", "inverse"]:
        if method in {"exp", "gaussian"}:
            for tau in [5, 7, 10, 14, 21, 28]:
                for max_days in [14, 21, 35, 56, None]:
                    for shrink in [0.0, 0.03, 0.07, 0.12]:
                        candidates.append({"method": method, "tau": tau, "power": 1.0, "max_days": max_days, "shrink": shrink})
        else:
            for power in [0.75, 1.0, 1.5, 2.0]:
                for max_days in [14, 21, 35, 56, None]:
                    for shrink in [0.0, 0.03, 0.07, 0.12]:
                        candidates.append({"method": method, "tau": 14.0, "power": power, "max_days": max_days, "shrink": shrink})
    return candidates


def score_candidates(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    selected_rows = []
    splits = temporal_splits(train)
    candidates = candidate_grid()
    for target in TARGETS:
        target_rows = []
        for candidate in candidates:
            split_losses = []
            for split_name, valid_mask in splits.items():
                fit = train.loc[~valid_mask].copy()
                valid = train.loc[valid_mask].copy()
                pred = predict_label_knn(valid, fit, target, **candidate)
                loss = binary_logloss(valid[target], pred)
                rows.append({"target": target, "split": split_name, "logloss": loss, **candidate})
                split_losses.append(loss)
            mean_loss = float(np.mean(split_losses))
            std_loss = float(np.std(split_losses))
            max_loss = float(np.max(split_losses))
            score = mean_loss + 0.25 * std_loss + 0.05 * max_loss
            target_rows.append({"target": target, "mean_logloss": mean_loss, "std_logloss": std_loss, "max_logloss": max_loss, "selection_score": score, **candidate})
        best = pd.DataFrame(target_rows).sort_values(["selection_score", "mean_logloss", "max_logloss"]).iloc[0].to_dict()
        selected_rows.append(best)
    return pd.DataFrame(rows), pd.DataFrame(selected_rows)


def make_submission(train: pd.DataFrame, sample: pd.DataFrame, selected: pd.DataFrame, output_name: str) -> pd.DataFrame:
    out = sample[KEYS].copy()
    for _, row in selected.iterrows():
        target = row["target"]
        params = {
            "method": row["method"],
            "tau": float(row["tau"]),
            "power": float(row["power"]),
            "max_days": None if pd.isna(row["max_days"]) else int(row["max_days"]),
            "shrink": float(row["shrink"]),
        }
        out[target] = predict_label_knn(sample, train, target, **params)
    out.to_csv(SUBMISSIONS / output_name, index=False)
    return out


def weighted_submission(parts: dict[str, float], output_name: str) -> pd.DataFrame:
    frames = {name: pd.read_csv(SUBMISSIONS / name) for name in parts}
    first = next(iter(frames.values()))
    out = first[KEYS].copy()
    for target in TARGETS:
        values = np.zeros(len(out), dtype=float)
        for name, weight in parts.items():
            values += weight * frames[name][target].to_numpy(dtype=float)
        out[target] = clip(values)
    out.to_csv(SUBMISSIONS / output_name, index=False)
    return out


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)

    train, sample = load_data()
    scores, selected = score_candidates(train)
    scores.to_csv(TABLES / "label_knn_candidate_scores.csv", index=False)
    selected.to_csv(TABLES / "label_knn_selected_strategies.csv", index=False)

    make_submission(train, sample, selected, "label_knn_submission.csv")

    if (SUBMISSIONS / "v3_semantic_robust_ensemble_submission.csv").exists():
        weighted_submission(
            {
                "label_knn_submission.csv": 0.50,
                "v3_semantic_robust_ensemble_submission.csv": 0.50,
            },
            "ensemble_labelknn50_v3robust50_submission.csv",
        )
        weighted_submission(
            {
                "label_knn_submission.csv": 0.30,
                "v3_semantic_robust_ensemble_submission.csv": 0.70,
            },
            "ensemble_labelknn30_v3robust70_submission.csv",
        )
    if (SUBMISSIONS / "friend_semantic_anchor_blend_10.csv").exists():
        weighted_submission(
            {
                "label_knn_submission.csv": 0.20,
                "friend_semantic_anchor_blend_10.csv": 0.40,
                "v3_semantic_robust_ensemble_submission.csv": 0.40,
            },
            "ensemble_labelknn20_friend40_v3robust40_submission.csv",
        )
        weighted_submission(
            {
                "label_knn_submission.csv": 0.15,
                "friend_semantic_anchor_blend_10.csv": 0.60,
                "v3_semantic_robust_ensemble_submission.csv": 0.25,
            },
            "ensemble_labelknn15_friend60_v3robust25_submission.csv",
        )
    if (SUBMISSIONS / "v3_semantic_robust_ensemble_submission.csv").exists():
        knn = pd.read_csv(SUBMISSIONS / "label_knn_submission.csv")
        v3 = pd.read_csv(SUBMISSIONS / "v3_semantic_robust_ensemble_submission.csv")
        out = v3[KEYS].copy()
        for target in TARGETS:
            if target in {"S2", "S3"}:
                out[target] = clip(0.30 * knn[target].to_numpy(dtype=float) + 0.70 * v3[target].to_numpy(dtype=float))
            else:
                out[target] = v3[target].to_numpy(dtype=float)
        out.to_csv(SUBMISSIONS / "targetwise_knn_s2s3_30_v3_submission.csv", index=False)

        out = v3[KEYS].copy()
        for target in TARGETS:
            if target in {"S2", "S3"}:
                out[target] = clip(0.50 * knn[target].to_numpy(dtype=float) + 0.50 * v3[target].to_numpy(dtype=float))
            else:
                out[target] = v3[target].to_numpy(dtype=float)
        out.to_csv(SUBMISSIONS / "targetwise_knn_s2s3_50_v3_submission.csv", index=False)

    if (SUBMISSIONS / "friend_semantic_anchor_blend_10.csv").exists() and (SUBMISSIONS / "v3_semantic_robust_ensemble_submission.csv").exists():
        knn = pd.read_csv(SUBMISSIONS / "label_knn_submission.csv")
        friend = pd.read_csv(SUBMISSIONS / "friend_semantic_anchor_blend_10.csv")
        v3 = pd.read_csv(SUBMISSIONS / "v3_semantic_robust_ensemble_submission.csv")
        out = friend[KEYS].copy()
        for target in TARGETS:
            base = 0.60 * friend[target].to_numpy(dtype=float) + 0.40 * v3[target].to_numpy(dtype=float)
            if target in {"S2", "S3"}:
                out[target] = clip(0.25 * knn[target].to_numpy(dtype=float) + 0.75 * base)
            else:
                out[target] = clip(base)
        out.to_csv(SUBMISSIONS / "targetwise_friend60_v3_40_knn_s2s3_submission.csv", index=False)

    metadata = {
        "selected": selected.to_dict("records"),
        "submission": str(SUBMISSIONS / "label_knn_submission.csv"),
    }
    (REPORTS / "label_knn_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(selected[["target", "method", "tau", "power", "max_days", "shrink", "mean_logloss", "selection_score"]].round(5).to_string(index=False))
    print(f"Wrote {SUBMISSIONS / 'label_knn_submission.csv'}")


if __name__ == "__main__":
    main()
