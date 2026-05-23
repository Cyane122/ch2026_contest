# ETRI 2026 심층 EDA

로컬 데이터 `C:\Users\KNova\Documents\ETRI\data`를 기준으로 생성한 리포트입니다.

## 대회 구조

- 과제: 7개 이진 지표 `Q1, Q2, Q3, S1, S2, S3, S4` 예측
- 평가 지표: 타깃 전체의 Average Log Loss
- 학습 데이터: 450행; 제출 데이터: 250행
- Subject 수: 학습 10명, 제출 10명
- 학습 sleep_date 범위: 2024-06-04 ~ 2024-11-15
- 제출 sleep_date 범위: 2024-07-07 ~ 2024-11-20

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

| index | count | positive | negative | positive_rate |
| --- | --- | --- | --- | --- |
| Q1 | 450.0 | 223.0 | 227.0 | 0.4956 |
| Q2 | 450.0 | 253.0 | 197.0 | 0.5622 |
| Q3 | 450.0 | 270.0 | 180.0 | 0.6 |
| S1 | 450.0 | 307.0 | 143.0 | 0.6822 |
| S2 | 450.0 | 293.0 | 157.0 | 0.6511 |
| S3 | 450.0 | 298.0 | 152.0 | 0.6622 |
| S4 | 450.0 | 252.0 | 198.0 | 0.56 |

## 센서 인벤토리

| table | rows | subjects | start | end | memory_mb |
| --- | --- | --- | --- | --- | --- |
| ch2025_mACStatus | 939896 | 10 | 2024-06-03 12:57:00 | 2024-11-19 23:59:00 | 25.1 |
| ch2025_mActivity | 961062 | 10 | 2024-06-03 12:57:00 | 2024-11-19 23:59:00 | 25.66 |
| ch2025_mAmbience | 476577 | 10 | 2024-06-03 13:00:10 | 2024-11-19 23:58:10 | 63.63 |
| ch2025_mBle | 21830 | 10 | 2024-06-03 13:07:00 | 2024-11-19 21:27:00 | 2.91 |
| ch2025_mGps | 800611 | 10 | 2024-06-03 12:57:00 | 2024-11-19 23:59:00 | 106.89 |
| ch2025_mLight | 96258 | 10 | 2024-06-03 12:57:00 | 2024-11-19 23:57:00 | 2.57 |
| ch2025_mScreenStatus | 939653 | 10 | 2024-06-03 12:57:00 | 2024-11-19 23:59:00 | 25.09 |
| ch2025_mUsageStats | 45197 | 10 | 2024-06-09 11:00:00 | 2024-11-19 22:10:00 | 6.03 |
| ch2025_mWifi | 76336 | 10 | 2024-06-03 12:57:00 | 2024-11-19 23:57:00 | 10.19 |
| ch2025_wHr | 382918 | 10 | 2024-06-03 14:28:00 | 2024-11-19 15:37:00 | 51.13 |
| ch2025_wLight | 633741 | 10 | 2024-06-03 14:28:00 | 2024-11-19 23:59:00 | 16.92 |
| ch2025_wPedo | 748100 | 10 | 2024-06-03 14:15:00 | 2024-11-19 23:59:00 | 54.22 |

## 필요한 lifelog 날짜 기준 센서 커버리지

| index | day_coverage_rate | median_rows_on_available_day | mean_rows_all_days |
| --- | --- | --- | --- |
| ch2025_mACStatus_rows | 1.0 | 1398.5 | 1342.709 |
| ch2025_mActivity_rows | 1.0 | 1440.0 | 1372.946 |
| ch2025_mAmbience_rows | 1.0 | 720.0 | 680.824 |
| ch2025_mLight_rows | 1.0 | 144.0 | 137.511 |
| ch2025_mScreenStatus_rows | 1.0 | 1400.0 | 1342.361 |
| ch2025_mUsageStats_rows | 0.986 | 67.0 | 64.567 |
| ch2025_mWifi_rows | 0.979 | 119.0 | 109.051 |
| ch2025_wLight_rows | 0.949 | 1027.5 | 905.344 |
| ch2025_mGps_rows | 0.943 | 1387.0 | 1143.73 |
| ch2025_wPedo_rows | 0.933 | 1280.0 | 1068.714 |
| ch2025_mBle_rows | 0.93 | 32.0 | 31.186 |
| ch2025_wHr_rows | 0.909 | 611.0 | 547.026 |

## 시간순 Holdout 베이스라인

검증 방식: subject별 마지막 7개의 라벨 sleep_date를 검증으로 사용했습니다. 의도적으로 시간순 분할을 사용했기 때문에 random CV보다 리더보드의 미래 예측 구조에 더 가깝습니다.

| target | baseline | log_loss | mean_p1 |
| --- | --- | --- | --- |
| Q1 | global_train_prior | 0.7026 | 0.4814 |
| Q1 | subject_prior | 0.708 | 0.5184 |
| Q1 | last_label_smoothed | 1.1163 | 0.7777 |
| Q2 | global_train_prior | 0.6627 | 0.5484 |
| Q2 | subject_prior | 0.7145 | 0.5525 |
| Q2 | last_label_smoothed | 1.179 | 0.816 |
| Q3 | global_train_prior | 0.6507 | 0.5931 |
| Q3 | subject_prior | 0.7207 | 0.5851 |
| Q3 | last_label_smoothed | 1.179 | 0.6053 |
| S1 | global_train_prior | 0.6262 | 0.6824 |
| S1 | subject_prior | 0.5861 | 0.6904 |
| S1 | last_label_smoothed | 1.0537 | 0.816 |
| S2 | global_train_prior | 0.7289 | 0.665 |
| S2 | subject_prior | 0.7125 | 0.6764 |
| S2 | last_label_smoothed | 1.179 | 0.6436 |
| S3 | global_train_prior | 0.7029 | 0.6725 |
| S3 | subject_prior | 0.6822 | 0.6938 |
| S3 | last_label_smoothed | 1.6801 | 0.6819 |
| S4 | global_train_prior | 0.6931 | 0.5633 |
| S4 | subject_prior | 0.6377 | 0.5301 |
| S4 | last_label_smoothed | 1.3669 | 0.6628 |
| average | best_per_target_floor | 0.6621 | nan |

- Global prior 평균 Log Loss: 0.6810
- Subject prior 평균 Log Loss: 0.6802
- 단순 per-target 최저 베이스라인: 0.6621

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
