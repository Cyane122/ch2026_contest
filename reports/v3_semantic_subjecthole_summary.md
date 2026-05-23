# V3 Semantic + Subject-Hole 개선 요약

## 변경 내용

V2 shift-6 피처에 더해 semantic feature와 subject-hole validation을 추가했습니다.

추가 피처:

- `mUsageStats` 앱 카테고리 피처
  - social, media, communication, game, finance/shopping, routine/health, browser/search
  - passive/productive usage ratio
- `mAmbience` semantic group 피처
  - speech, silence, music, vehicle, outdoor, sleep_noise
  - social_noise, sleep_disturbance proxy
- `mGps` semantic 피처
  - point count, speed/altitude/lat/lon 통계
  - mobility radius
  - night home 기준 home distance
- `mWifi`, `mBle` semantic 피처
  - detected/unique count
  - familiar/novel ratio
  - RSSI 통계와 strong signal count

추가 검증 split:

- `subject_hole0`
- `subject_hole1`
- `subject_hole2`

## V3 내부 검증 결과

| split | valid_rows | valid_log_loss |
| --- | ---: | ---: |
| interleaved_offset1 | 131 | 0.5745 |
| interleaved_offset2 | 137 | 0.5861 |
| subject_hole0 | 95 | 0.6232 |
| subject_hole1 | 92 | 0.5718 |
| subject_hole2 | 91 | 0.5994 |
| late7 | 47 | 0.6136 |

## 해석

- V3는 `late7`에서 V2보다 더 좋아졌습니다: `0.6198 -> 0.6136`.
- `interleaved_offset1/2`도 V2와 비슷하거나 소폭 개선되었습니다.
- subject-hole split은 편차가 큽니다. 특히 `subject_hole0`은 어렵고 `subject_hole1`은 매우 좋습니다. 따라서 subject-hole submission을 강하게 섞는 것은 위험할 수 있습니다.
- 현재 V3 robust ensemble은 subject-hole 후보까지 섞기 때문에 Public에서 안정적일지는 아직 확인이 필요합니다.

## 생성된 주요 후보

| file | 설명 |
| --- | --- |
| `v3_semantic_baseline_submission.csv` | V3 primary interleaved 후보 |
| `v3_semantic_robust_ensemble_submission.csv` | V3 split ensemble 후보 |
| `blend_friend80_v3robust20_submission.csv` | 친구 80% + V3 robust 20% |
| `blend_friend90_v3robust10_submission.csv` | 친구 90% + V3 robust 10% |
| `blend_friend80_v2robust10_v3robust10_submission.csv` | 친구 80% + V2/V3 robust 각각 10% |
| `ensemble_v2robust50_v3robust50_submission.csv` | 우리 V2/V3 robust 50/50 |
| `v3_targetwise_cv_weighted_submission.csv` | target별 validation loss 기반 split 가중 ensemble |
| `blend_friend80_v3targetwise20_submission.csv` | 친구 80% + V3 targetwise 20% |
| `blend_friend90_v3targetwise10_submission.csv` | 친구 90% + V3 targetwise 10% |

## 다음 제출 우선순위

현재 가장 좋은 공개 점수는 친구 후보 `0.6171`입니다. V3는 내부 검증상 좋아졌지만 Public 검증 전이므로 친구 후보에서 크게 멀어지지 않는 blend를 먼저 확인합니다.

1. `blend_friend80_v2robust10_v3robust10_submission.csv`
2. `blend_friend90_v3robust10_submission.csv`
3. `blend_friend90_v3targetwise10_submission.csv`
4. `blend_friend80_v3robust20_submission.csv`
5. `v3_semantic_robust_ensemble_submission.csv`
6. `ensemble_v2robust50_v3robust50_submission.csv`

## 남은 개선 후보

1. target별로 V2/V3/friend 가중치를 더 정교하게 적용
2. S1-S4 OOF 예측을 Q1-Q3 feature로 넣는 chained Q 모델 추가
3. feature importance 기반으로 semantic feature pruning
