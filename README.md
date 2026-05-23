# ETRI 2026 대회 작업 공간

이 프로젝트는 DACON ETRI 휴먼이해 인공지능 논문경진대회를 위한 재현 가능한 EDA에서 시작합니다.

## 목표

- 대회 평가 지표: `Q1`, `Q2`, `Q3`, `S1`, `S2`, `S3`, `S4` 전체의 Average Log Loss
- 작업 목표: 로컬 검증 Log Loss `0.6` 이하
- 단순 시간순 prior 기준 현재 베이스라인 하한: `0.6621`

## 데이터

로컬 데이터는 `data/` 아래에 있다고 가정합니다.

- `ch2026_metrics_train.csv`: subject별 수면/라이프로그 날짜와 라벨
- `ch2026_submission_sample.csv`: 예측해야 할 미래 날짜
- `ch2025_data_items/*.parquet`: 스마트폰 및 웨어러블 라이프로그 스트림
- `ch2026_metrics_description.pdf`: 타깃 정의

## EDA 실행

이 작업 공간에서 사용한 Codex 번들 Python 런타임으로 실행합니다.

```powershell
& 'C:\Users\KNova\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -X utf8 scripts\eda_deep_dive.py
```

주요 산출물:

- `reports/eda_summary.md`
- `reports/tables/target_label_summary.csv`
- `reports/tables/sensor_inventory.csv`
- `reports/tables/sensor_coverage_summary.csv`
- `reports/tables/time_holdout_baselines.csv`
- `reports/tables/daily_simple_features.parquet`
- `reports/figures/target_positive_rates.png`
- `reports/figures/target_correlation.png`
- `reports/figures/sensor_day_coverage.png`

## 초기 발견사항

- 이 문제는 동일한 10명의 subject에 대한 단기 미래 예측 문제입니다.
- 라벨은 개인별 평균을 기준으로 이진화된 결과이므로, subject별 prior와 rolling history가 강한 신호가 될 가능성이 큽니다.
- 타깃 lifelog 날짜 기준 센서 커버리지는 전반적으로 높습니다. 심박, BLE, 보행계, GPS, 웨어러블 조도는 상대적으로 낮지만 여전히 활용 가치가 있습니다.
- random validation보다는 subject별 마지막 7일을 검증으로 두는 시간순 분할이 첫 검증 기준으로 더 적절합니다.

## 다음 단계

일별/시간대별 집계 피처, rolling target history, 타깃별 calibrated classifier, 제출 파일 생성기를 중심으로 모델링 파이프라인을 구축합니다.

## 첫 모델링 결과

- 모델 스크립트: `scripts/train_baseline_model.py`
- 검증 리포트: `reports/model_summary.md`
- 제출 파일: `submissions/baseline_submission.csv`
- 안정성 후보 제출 파일: `submissions/robust_ensemble_submission.csv`
- V2 shift-6 개선 리포트: `reports/v2_shift6_improvement_summary.md`
- V3 semantic/subject-hole 개선 리포트: `reports/v3_semantic_subjecthole_summary.md`
- MLP 딥러닝 실험 리포트: `reports/mlp_deep_model_summary.md`
- Label KNN 실험 리포트: `reports/label_knn_summary.md`
- Interleaved holdout 기준 평균 Log Loss: `0.5750`
- 보조 interleaved holdout 기준 평균 Log Loss: `0.5868`
- 마지막 7일 미래형 holdout 기준 평균 Log Loss: `0.6198`
- Public LB: `baseline_submission.csv` 0.6339, `robust_ensemble_submission.csv` 0.6240

이 점수는 submission 날짜가 train 기간 사이에 끼어 있는 구조를 반영한 검증 결과입니다. 시간순 마지막 7일 검증보다 실제 제출 구조에 더 가깝지만, 다음 단계에서는 여러 validation split으로 안정성을 함께 확인해야 합니다.

Public LB에서는 `robust_ensemble_submission.csv`가 더 좋았습니다. 친구 후보까지 포함한 다음 후보는 `scripts/make_submission_variants.py`로 생성한 `blend_friend80_v2robust10_v3robust10_submission.csv`, `blend_friend90_v3robust10_submission.csv`, `blend_friend90_v3targetwise10_submission.csv`를 우선 확인합니다.

## Public Round 2

- `blend_friend80_v2robust10_v3robust10_submission.csv`: `0.6090`
- `blend_friend90_v3robust10_submission.csv`: `0.6126`
- `v3_semantic_robust_ensemble_submission.csv`: `0.6091`
- 전략 리포트: `reports/leaderboard_round2_strategy.md`

다음 후보는 friend 비중을 낮추고 v3 중심으로 이동한 `blend_friend60_v2robust20_v3robust20_submission.csv`, `blend_friend40_v2robust20_v3robust40_submission.csv`, `ensemble_v2robust25_v3robust75_submission.csv`를 우선 확인합니다.
