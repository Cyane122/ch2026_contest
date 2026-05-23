from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SUBMISSIONS = ROOT / "submissions"
REPORTS = ROOT / "reports"
TABLES = REPORTS / "tables"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
KEYS = ["subject_id", "sleep_date", "lifelog_date"]


def clip(p: np.ndarray) -> np.ndarray:
    return np.clip(p, 1e-5, 1 - 1e-5)


def read_submission(name: str) -> pd.DataFrame:
    return pd.read_csv(SUBMISSIONS / name)


def weighted_submission(parts: dict[str, float], output_name: str) -> pd.DataFrame:
    frames = {name: read_submission(name) for name in parts}
    first = next(iter(frames.values()))
    out = first[KEYS].copy()
    for target in TARGETS:
        values = np.zeros(len(out), dtype=float)
        for name, weight in parts.items():
            values += weight * frames[name][target].to_numpy(dtype=float)
        out[target] = clip(values)
    out.to_csv(SUBMISSIONS / output_name, index=False)
    return out


def shrink_submission(base_name: str, output_name: str, strength: float, target_means: pd.Series) -> pd.DataFrame:
    base = read_submission(base_name)
    out = base[KEYS].copy()
    for target in TARGETS:
        out[target] = clip((1 - strength) * base[target].to_numpy(dtype=float) + strength * float(target_means[target]))
    out.to_csv(SUBMISSIONS / output_name, index=False)
    return out


def summarize() -> pd.DataFrame:
    rows = []
    for path in sorted(SUBMISSIONS.glob("*.csv")):
        df = pd.read_csv(path)
        assert df.shape == (250, 10), (path.name, df.shape)
        assert df.isna().sum().sum() == 0, path.name
        assert ((df[TARGETS] >= 0).all().all() and (df[TARGETS] <= 1).all().all()), path.name
        row = {"file": path.name}
        for target in TARGETS:
            row[f"{target}_mean"] = df[target].mean()
            row[f"{target}_std"] = df[target].std()
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES / "submission_variants_summary.csv", index=False)
    return summary


def targetwise_cv_weighted_submission(output_name: str, temperature: float = 0.04) -> pd.DataFrame | None:
    score_path = TABLES / "model_validation_scores_by_split.csv"
    if not score_path.exists():
        return None
    scores = pd.read_csv(score_path)
    split_files = {
        "interleaved_offset1": "interleaved_offset1_submission.csv",
        "interleaved_offset2": "interleaved_offset2_submission.csv",
        "subject_hole0": "subject_hole0_submission.csv",
        "subject_hole1": "subject_hole1_submission.csv",
        "subject_hole2": "subject_hole2_submission.csv",
        "late7": "late7_submission.csv",
    }
    available = {split: file for split, file in split_files.items() if (SUBMISSIONS / file).exists()}
    if not available:
        return None
    frames = {split: read_submission(file) for split, file in available.items()}
    first = next(iter(frames.values()))
    out = first[KEYS].copy()
    weight_rows = []
    for target in TARGETS:
        target_scores = scores[(scores["target"] == target) & (scores["split"].isin(available))]
        target_scores = target_scores[["split", "valid_log_loss"]].dropna()
        if target_scores.empty:
            out[target] = np.mean([frame[target].to_numpy(dtype=float) for frame in frames.values()], axis=0)
            continue
        losses = target_scores["valid_log_loss"].to_numpy(dtype=float)
        raw_weights = np.exp(-(losses - losses.min()) / temperature)
        weights = raw_weights / raw_weights.sum()
        values = np.zeros(len(out), dtype=float)
        for split, weight in zip(target_scores["split"], weights):
            values += float(weight) * frames[str(split)][target].to_numpy(dtype=float)
            weight_rows.append({"target": target, "split": str(split), "weight": float(weight), "valid_log_loss": float(target_scores.loc[target_scores["split"] == split, "valid_log_loss"].iloc[0])})
        out[target] = clip(values)
    out.to_csv(SUBMISSIONS / output_name, index=False)
    pd.DataFrame(weight_rows).to_csv(TABLES / f"{output_name.replace('.csv', '')}_weights.csv", index=False)
    return out


def main() -> None:
    SUBMISSIONS.mkdir(exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    target_means = train[TARGETS].mean()

    weighted_submission(
        {
            "interleaved_offset1_submission.csv": 0.40,
            "interleaved_offset2_submission.csv": 0.40,
            "late7_submission.csv": 0.20,
        },
        "ensemble_i40_i40_l20_submission.csv",
    )
    weighted_submission(
        {
            "interleaved_offset1_submission.csv": 0.35,
            "interleaved_offset2_submission.csv": 0.35,
            "late7_submission.csv": 0.30,
        },
        "ensemble_i35_i35_l30_submission.csv",
    )
    weighted_submission(
        {
            "interleaved_offset1_submission.csv": 0.30,
            "interleaved_offset2_submission.csv": 0.30,
            "late7_submission.csv": 0.40,
        },
        "ensemble_i30_i30_l40_submission.csv",
    )

    shrink_submission("robust_ensemble_submission.csv", "robust_shrink05_submission.csv", 0.05, target_means)
    shrink_submission("robust_ensemble_submission.csv", "robust_shrink10_submission.csv", 0.10, target_means)
    shrink_submission("robust_ensemble_submission.csv", "robust_shrink15_submission.csv", 0.15, target_means)
    shrink_submission("ensemble_i35_i35_l30_submission.csv", "ensemble_i35_l30_shrink10_submission.csv", 0.10, target_means)
    targetwise_cv_weighted_submission("v3_targetwise_cv_weighted_submission.csv")

    friend_name = "friend_semantic_anchor_blend_10.csv"
    if (SUBMISSIONS / friend_name).exists():
        weighted_submission(
            {
                friend_name: 0.70,
                "robust_ensemble_submission.csv": 0.30,
            },
            "blend_friend70_robust30_submission.csv",
        )
        weighted_submission(
            {
                friend_name: 0.80,
                "robust_ensemble_submission.csv": 0.20,
            },
            "blend_friend80_robust20_submission.csv",
        )
        weighted_submission(
            {
                friend_name: 0.90,
                "robust_ensemble_submission.csv": 0.10,
            },
            "blend_friend90_robust10_submission.csv",
        )
        weighted_submission(
            {
                friend_name: 0.60,
                "robust_ensemble_submission.csv": 0.40,
            },
            "blend_friend60_robust40_submission.csv",
        )
        weighted_submission(
            {
                friend_name: 0.50,
                "robust_ensemble_submission.csv": 0.50,
            },
            "blend_friend50_robust50_submission.csv",
        )
        weighted_submission(
            {
                friend_name: 0.60,
                "robust_shrink05_submission.csv": 0.40,
            },
            "blend_friend60_robustshrink05_40_submission.csv",
        )
        weighted_submission(
            {
                friend_name: 0.50,
                "robust_shrink05_submission.csv": 0.30,
                "ensemble_i40_i40_l20_submission.csv": 0.20,
            },
            "blend_friend50_robust30_i40l20_20_submission.csv",
        )
        targetwise = read_submission(friend_name)
        robust = read_submission("robust_ensemble_submission.csv")
        i40_l20 = read_submission("ensemble_i40_i40_l20_submission.csv")
        out = targetwise[KEYS].copy()
        friend_heavy = {"Q1": 0.80, "Q2": 0.80, "Q3": 0.85, "S1": 0.70, "S2": 0.65, "S3": 0.65, "S4": 0.75}
        for target, friend_weight in friend_heavy.items():
            out[target] = clip(friend_weight * targetwise[target].to_numpy() + (1 - friend_weight) * robust[target].to_numpy())
        out.to_csv(SUBMISSIONS / "blend_targetwise_friend_heavy_submission.csv", index=False)

        out = targetwise[KEYS].copy()
        for target in TARGETS:
            if target in {"Q1", "Q2", "Q3", "S4"}:
                out[target] = clip(0.80 * targetwise[target].to_numpy() + 0.20 * robust[target].to_numpy())
            else:
                out[target] = clip(0.70 * targetwise[target].to_numpy() + 0.20 * robust[target].to_numpy() + 0.10 * i40_l20[target].to_numpy())
        out.to_csv(SUBMISSIONS / "blend_targetwise_friend_q80_s70_submission.csv", index=False)

        named_robust_sources = {
            "v2": "v2_shift6_robust_ensemble_submission.csv",
            "v3": "v3_semantic_robust_ensemble_submission.csv",
            "v4knn": "v4_labelknnprior_robust_ensemble_submission.csv",
        }
        for tag, source_name in named_robust_sources.items():
            if not (SUBMISSIONS / source_name).exists():
                continue
            weighted_submission(
                {
                    friend_name: 0.90,
                    source_name: 0.10,
                },
                f"blend_friend90_{tag}robust10_submission.csv",
            )
            weighted_submission(
                {
                    friend_name: 0.80,
                    source_name: 0.20,
                },
                f"blend_friend80_{tag}robust20_submission.csv",
            )
            weighted_submission(
                {
                    friend_name: 0.70,
                    source_name: 0.30,
                },
                f"blend_friend70_{tag}robust30_submission.csv",
            )

        if (SUBMISSIONS / "v2_shift6_robust_ensemble_submission.csv").exists() and (SUBMISSIONS / "v3_semantic_robust_ensemble_submission.csv").exists():
            weighted_submission(
                {
                    friend_name: 0.80,
                    "v2_shift6_robust_ensemble_submission.csv": 0.10,
                    "v3_semantic_robust_ensemble_submission.csv": 0.10,
                },
                "blend_friend80_v2robust10_v3robust10_submission.csv",
            )
            weighted_submission(
                {
                    friend_name: 0.70,
                    "v2_shift6_robust_ensemble_submission.csv": 0.15,
                    "v3_semantic_robust_ensemble_submission.csv": 0.15,
                },
                "blend_friend70_v2robust15_v3robust15_submission.csv",
            )
            weighted_submission(
                {
                    "v2_shift6_robust_ensemble_submission.csv": 0.50,
                    "v3_semantic_robust_ensemble_submission.csv": 0.50,
                },
                "ensemble_v2robust50_v3robust50_submission.csv",
            )
            weighted_submission(
                {
                    friend_name: 0.60,
                    "v2_shift6_robust_ensemble_submission.csv": 0.20,
                    "v3_semantic_robust_ensemble_submission.csv": 0.20,
                },
                "blend_friend60_v2robust20_v3robust20_submission.csv",
            )
            weighted_submission(
                {
                    friend_name: 0.50,
                    "v2_shift6_robust_ensemble_submission.csv": 0.25,
                    "v3_semantic_robust_ensemble_submission.csv": 0.25,
                },
                "blend_friend50_v2robust25_v3robust25_submission.csv",
            )
            weighted_submission(
                {
                    friend_name: 0.40,
                    "v2_shift6_robust_ensemble_submission.csv": 0.20,
                    "v3_semantic_robust_ensemble_submission.csv": 0.40,
                },
                "blend_friend40_v2robust20_v3robust40_submission.csv",
            )
            weighted_submission(
                {
                    friend_name: 0.30,
                    "v2_shift6_robust_ensemble_submission.csv": 0.20,
                    "v3_semantic_robust_ensemble_submission.csv": 0.50,
                },
                "blend_friend30_v2robust20_v3robust50_submission.csv",
            )
            weighted_submission(
                {
                    friend_name: 0.20,
                    "v2_shift6_robust_ensemble_submission.csv": 0.20,
                    "v3_semantic_robust_ensemble_submission.csv": 0.60,
                },
                "blend_friend20_v2robust20_v3robust60_submission.csv",
            )
            weighted_submission(
                {
                    "v2_shift6_robust_ensemble_submission.csv": 0.25,
                    "v3_semantic_robust_ensemble_submission.csv": 0.75,
                },
                "ensemble_v2robust25_v3robust75_submission.csv",
            )
        if (SUBMISSIONS / "v3_semantic_robust_ensemble_submission.csv").exists() and (SUBMISSIONS / "v4_labelknnprior_robust_ensemble_submission.csv").exists():
            weighted_submission(
                {
                    "v3_semantic_robust_ensemble_submission.csv": 0.50,
                    "v4_labelknnprior_robust_ensemble_submission.csv": 0.50,
                },
                "ensemble_v3robust50_v4knnrobust50_submission.csv",
            )
            weighted_submission(
                {
                    friend_name: 0.80,
                    "v4_labelknnprior_robust_ensemble_submission.csv": 0.20,
                },
                "blend_friend80_v4knnrobust20_submission.csv",
            )
            weighted_submission(
                {
                    friend_name: 0.60,
                    "v3_semantic_robust_ensemble_submission.csv": 0.20,
                    "v4_labelknnprior_robust_ensemble_submission.csv": 0.20,
                },
                "blend_friend60_v3robust20_v4knnrobust20_submission.csv",
            )
            weighted_submission(
                {
                    friend_name: 0.40,
                    "v3_semantic_robust_ensemble_submission.csv": 0.30,
                    "v4_labelknnprior_robust_ensemble_submission.csv": 0.30,
                },
                "blend_friend40_v3robust30_v4knnrobust30_submission.csv",
            )
            weighted_submission(
                {
                    friend_name: 0.30,
                    "v3_semantic_robust_ensemble_submission.csv": 0.35,
                    "v4_labelknnprior_robust_ensemble_submission.csv": 0.35,
                },
                "blend_friend30_v3robust35_v4knnrobust35_submission.csv",
            )
            weighted_submission(
                {
                    friend_name: 0.35,
                    "v3_semantic_robust_ensemble_submission.csv": 0.325,
                    "v4_labelknnprior_robust_ensemble_submission.csv": 0.325,
                },
                "blend_friend35_v3robust325_v4knnrobust325_submission.csv",
            )
            weighted_submission(
                {
                    friend_name: 0.25,
                    "v3_semantic_robust_ensemble_submission.csv": 0.375,
                    "v4_labelknnprior_robust_ensemble_submission.csv": 0.375,
                },
                "blend_friend25_v3robust375_v4knnrobust375_submission.csv",
            )
            weighted_submission(
                {
                    friend_name: 0.30,
                    "v3_semantic_robust_ensemble_submission.csv": 0.30,
                    "v4_labelknnprior_robust_ensemble_submission.csv": 0.40,
                },
                "blend_friend30_v3robust30_v4knnrobust40_submission.csv",
            )
            weighted_submission(
                {
                    friend_name: 0.35,
                    "v3_semantic_robust_ensemble_submission.csv": 0.25,
                    "v4_labelknnprior_robust_ensemble_submission.csv": 0.40,
                },
                "blend_friend35_v3robust25_v4knnrobust40_submission.csv",
            )
        if (
            (SUBMISSIONS / "v2_shift6_robust_ensemble_submission.csv").exists()
            and (SUBMISSIONS / "v3_semantic_robust_ensemble_submission.csv").exists()
            and (SUBMISSIONS / "v4_labelknnprior_robust_ensemble_submission.csv").exists()
        ):
            weighted_submission(
                {
                    friend_name: 0.50,
                    "v2_shift6_robust_ensemble_submission.csv": 0.15,
                    "v3_semantic_robust_ensemble_submission.csv": 0.20,
                    "v4_labelknnprior_robust_ensemble_submission.csv": 0.15,
                },
                "blend_friend50_v2robust15_v3robust20_v4knnrobust15_submission.csv",
            )
            weighted_submission(
                {
                    friend_name: 0.40,
                    "v2_shift6_robust_ensemble_submission.csv": 0.15,
                    "v3_semantic_robust_ensemble_submission.csv": 0.25,
                    "v4_labelknnprior_robust_ensemble_submission.csv": 0.20,
                },
                "blend_friend40_v2robust15_v3robust25_v4knnrobust20_submission.csv",
            )
        if (SUBMISSIONS / "v3_targetwise_cv_weighted_submission.csv").exists():
            weighted_submission(
                {
                    friend_name: 0.80,
                    "v3_targetwise_cv_weighted_submission.csv": 0.20,
                },
                "blend_friend80_v3targetwise20_submission.csv",
            )
            weighted_submission(
                {
                    friend_name: 0.90,
                    "v3_targetwise_cv_weighted_submission.csv": 0.10,
                },
                "blend_friend90_v3targetwise10_submission.csv",
            )
        if (SUBMISSIONS / "mlp_robust_ensemble_submission.csv").exists():
            weighted_submission(
                {
                    friend_name: 0.80,
                    "v3_semantic_robust_ensemble_submission.csv": 0.15,
                    "mlp_robust_ensemble_submission.csv": 0.05,
                },
                "blend_friend80_v3robust15_mlp05_submission.csv",
            )
            weighted_submission(
                {
                    friend_name: 0.60,
                    "v3_semantic_robust_ensemble_submission.csv": 0.35,
                    "mlp_robust_ensemble_submission.csv": 0.05,
                },
                "blend_friend60_v3robust35_mlp05_submission.csv",
            )
            weighted_submission(
                {
                    "v3_semantic_robust_ensemble_submission.csv": 0.90,
                    "mlp_robust_ensemble_submission.csv": 0.10,
                },
                "ensemble_v3robust90_mlp10_submission.csv",
            )

    summary = summarize()
    print(summary[["file", *[f"{t}_mean" for t in TARGETS]]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
