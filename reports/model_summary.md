# 모델링 진행 결과

## 목적

EDA 이후 재현 가능한 모델링 파이프라인을 만들고, 목표인 Average Log Loss `0.6` 이하 가능성과 점수 안정성을 함께 확인했습니다.

## 현재 핵심 결과

- 사용 피처 수: v2 기준 shift-6 window 피처 포함
- Primary 검증 방식: `interleaved_offset1`
- Primary 평균 검증 Log Loss: `0.5750`
- 보조 interleaved 검증 Log Loss: `0.5868`
- 마지막 7일 미래형 검증 Log Loss: `0.6198`
- Public LB 점수: `baseline_submission.csv` 0.6339, `robust_ensemble_submission.csv` 0.6240
- 기본 제출 파일: `submissions/baseline_submission.csv`
- 안정성 후보 제출 파일: `submissions/robust_ensemble_submission.csv`

submission 날짜가 train 기간 이후에만 존재하지 않고, 많은 subject에서 train 날짜 사이에 끼어 있습니다. 그래서 현재 primary 검증은 train 날짜 사이의 빈 날짜를 예측하는 구조를 반영한 interleaved holdout입니다.

## Split별 평균 점수

| split | valid_rows | valid_log_loss | valid_model_log_loss |
| --- | ---: | ---: | ---: |
| interleaved_offset1 | 131 | 0.5750 | 0.8358 |
| interleaved_offset2 | 137 | 0.5868 | 0.7065 |
| late7 | 47 | 0.6198 | 0.6879 |

## 타깃별 점수

| target | interleaved_offset1 | interleaved_offset2 | late7 |
| --- | ---: | ---: | ---: |
| Q1 | 0.6234 | 0.6215 | 0.6142 |
| Q2 | 0.6209 | 0.5350 | 0.5970 |
| Q3 | 0.5784 | 0.6453 | 0.6268 |
| S1 | 0.5296 | 0.6189 | 0.5703 |
| S2 | 0.5304 | 0.5658 | 0.6367 |
| S3 | 0.5186 | 0.5057 | 0.6650 |
| S4 | 0.6233 | 0.6158 | 0.6285 |

## 제출 후보

| file | 설명 |
| --- | --- |
| `baseline_submission.csv` | primary interleaved split 기준 최종 후보 |
| `primary_interleaved_submission.csv` | `baseline_submission.csv`와 동일한 primary 후보 |
| `interleaved_offset1_submission.csv` | interleaved offset 1에서 선택된 모델 조합 |
| `interleaved_offset2_submission.csv` | interleaved offset 2에서 선택된 모델 조합 |
| `late7_submission.csv` | 마지막 7일 미래형 검증에서 선택된 모델 조합 |
| `robust_ensemble_submission.csv` | interleaved 후보 2개와 late7 후보를 `0.45/0.45/0.10`으로 섞은 안정성 후보 |

현재 첫 제출은 `baseline_submission.csv`를 우선 추천합니다. 점수 변동을 줄이고 싶다면 `robust_ensemble_submission.csv`도 비교 제출 후보로 볼 수 있습니다.

Public LB에서는 `robust_ensemble_submission.csv`가 `baseline_submission.csv`보다 좋았습니다. 따라서 이후 제출은 primary 단일 후보보다 split ensemble과 확률 shrinkage 주변을 탐색하는 것이 더 합리적입니다.

## Public 점수 기록

| file | public_log_loss | note |
| --- | ---: | --- |
| `baseline_submission.csv` | 0.6339 | primary interleaved 후보 |
| `robust_ensemble_submission.csv` | 0.6240 | 현재 관측 최고 점수 |

## 추가 생성 후보

| file | 의도 |
| --- | --- |
| `ensemble_i40_i40_l20_submission.csv` | late7 비중을 20%로 올린 split ensemble |
| `ensemble_i35_i35_l30_submission.csv` | late7 비중을 30%로 올린 split ensemble |
| `ensemble_i30_i30_l40_submission.csv` | late7 비중을 40%로 올린 더 보수적인 split ensemble |
| `robust_shrink05_submission.csv` | robust를 train 타깃 평균 쪽으로 5% shrink |
| `robust_shrink10_submission.csv` | robust를 train 타깃 평균 쪽으로 10% shrink |
| `robust_shrink15_submission.csv` | robust를 train 타깃 평균 쪽으로 15% shrink |
| `ensemble_i35_l30_shrink10_submission.csv` | late7 30% ensemble에 10% shrink 적용 |

다음 제출 우선순위는 `robust_shrink05_submission.csv`, `ensemble_i40_i40_l20_submission.csv`, `robust_shrink10_submission.csv` 순서가 무난합니다. `robust_ensemble`이 이미 좋은 편이므로 너무 큰 late7 이동은 후순위로 두는 편이 안전합니다.

## 해석

- neighbor prior, 즉 예측 날짜의 앞/뒤 train 라벨을 기반으로 만든 prior를 추가하면서 primary 점수가 `0.5935`에서 `0.5800`으로 개선되었습니다.
- 두 interleaved split이 모두 0.6 아래라, train 날짜 사이의 빈 날짜를 예측하는 구조에서는 목표선 통과 가능성이 있습니다.
- Public LB에서 robust ensemble이 더 좋았으므로, 실제 public split은 primary interleaved 하나보다 여러 split을 섞은 보수적 확률을 선호하는 쪽으로 보입니다.
- late7 split은 0.6305로 더 어렵습니다. Public/Private이 마지막 날짜에 치우쳐 있으면 현재 모델은 보수적으로 봐야 합니다.
- `Q1`, `S4`는 모든 split에서 0.61 이상이라 다음 개선의 우선순위입니다.
- `S2`, `S3`는 split에 따라 흔들림이 커서, subject prior에 기대는 구간과 센서 모델이 필요한 구간을 나눠 볼 필요가 있습니다.

## 다음 개선 방향

1. Q 계열에는 취침 전 6-12시간의 activity, screen, usage, ambience 피처를 더 세밀하게 추가합니다.
2. S 계열에는 수면 직전/직후 wearable HR, pedometer, light의 변화량과 결측 패턴 피처를 강화합니다.
3. 타깃별로 probability shrinkage와 clipping 범위를 검증 split 평균 기준으로 튜닝합니다.
4. 리더보드 제출 후 Public 점수를 기준으로 primary와 robust 후보 중 어느 쪽이 split 구조에 가까운지 판단합니다.
