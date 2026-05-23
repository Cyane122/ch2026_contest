# V2 Shift-6 개선 요약

## 변경 내용

우리 모델에 친구 코드의 아이디어를 그대로 복사하지 않고, 수면 문제에 맞는 구조를 직접 추가했습니다.

- 기존 calendar date 피처 유지
- `timestamp - 6시간` 기준 shifted lifelog date 추가
- 수면 중심 시간창 추가
  - `shift6_prebed_21_24`
  - `shift6_sleep_21_09`
  - `shift6_late_00_03`
  - `shift6_deep_03_06`
  - `shift6_wake_06_09`
  - `shift6_day_09_18`
  - `shift6_evening_18_21`
- 적용 센서
  - `mACStatus`
  - `mActivity`
  - `mLight`
  - `mScreenStatus`
  - `wLight`
  - `wPedo`
  - `wHr`
  - `mUsageStats`
  - `mAmbience`

## 내부 검증 변화

| split | v1 logloss | v2 shift6 logloss | 변화 |
| --- | ---: | ---: | ---: |
| interleaved_offset1 | 0.5800 | 0.5750 | -0.0050 |
| interleaved_offset2 | 0.5914 | 0.5868 | -0.0045 |
| late7 | 0.6305 | 0.6198 | -0.0107 |

late7 개선폭이 가장 큽니다. Public LB에서 기존 robust가 baseline보다 좋았던 점을 고려하면, 더 보수적인 날짜 구조에도 강해진 것은 좋은 신호입니다.

## V2 산출물

| file | 설명 |
| --- | --- |
| `v2_shift6_baseline_submission.csv` | v2 primary interleaved 후보 |
| `v2_shift6_robust_ensemble_submission.csv` | v2 split ensemble 후보 |
| `blend_friend80_robust20_submission.csv` | 친구 후보 80% + v2 robust 20% |
| `blend_friend90_robust10_submission.csv` | 친구 후보 90% + v2 robust 10% |
| `blend_friend70_robust30_submission.csv` | 친구 후보 70% + v2 robust 30% |
| `blend_targetwise_friend_heavy_submission.csv` | target별 친구 중심 가중 blend |

## 친구 후보와 V2 Robust 비교

| target | corr_friend_v2robust | mae_friend_v2robust | friend_mean - v2robust_mean |
| --- | ---: | ---: | ---: |
| Q1 | 0.4895 | 0.1559 | 0.0138 |
| Q2 | 0.4727 | 0.1228 | -0.0051 |
| Q3 | 0.3983 | 0.1321 | 0.0202 |
| S1 | 0.8020 | 0.0928 | 0.0619 |
| S2 | 0.8954 | 0.0958 | -0.0603 |
| S3 | 0.8705 | 0.1073 | -0.0451 |
| S4 | 0.6444 | 0.1033 | -0.0037 |

친구 후보와 V2 robust의 전체 상관은 `0.6807`입니다. Q 계열은 여전히 상관이 낮아 blend 여지가 있고, S 계열은 calibration 차이가 주된 차이로 보입니다.

## 다음 제출 우선순위

오늘은 제출 횟수가 끝났으므로 내일 아래 순서로 확인합니다.

1. `blend_friend80_robust20_submission.csv`
2. `blend_friend90_robust10_submission.csv`
3. `v2_shift6_robust_ensemble_submission.csv`
4. `blend_targetwise_friend_heavy_submission.csv`
5. `blend_friend70_robust30_submission.csv`

친구 단독이 이미 `0.6171`이므로, 첫 시도는 친구 후보에서 크게 멀어지지 않는 `80/20` 또는 `90/10`이 안전합니다. 다만 V2 robust의 late7 내부 점수가 개선되었으므로, 친구 후보와 섞었을 때 기존 `0.6171`보다 좋아질 가능성이 있습니다.

## 다음 구현 후보

1. subject-hole 5-fold validation을 우리 코드에 추가
2. target별 shrink alpha를 split 평균 기준으로 선택
3. UsageStats 앱 카테고리 피처 추가
4. Ambience semantic group 피처 확장
5. S-to-Q chained Q 모델 추가
