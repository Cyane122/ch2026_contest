# Label KNN 실험 요약

## 아이디어

센서 데이터를 쓰지 않고, 같은 subject의 가까운 날짜 label만 사용합니다.

예측 날짜와 train 날짜 사이의 거리를 계산하고 가까운 날짜에 더 큰 가중치를 줍니다. 직관적으로는 “비슷한 시기에는 같은 사람의 컨디션이 비슷하다”는 가정입니다.

## 구현

스크립트:

- `scripts/make_label_knn_submission.py`

가중 방식 후보:

- exponential decay
- gaussian decay
- inverse distance

튜닝한 값:

- decay scale `tau`
- 최대 참조 거리 `max_days`
- subject prior shrink
- target별 최적 조합

## 선택된 target별 전략

| target | method | tau | power | max_days | shrink | mean_logloss |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Q1 | inverse | 14 | 1.00 | all | 0.12 | 0.6721 |
| Q2 | inverse | 14 | 1.00 | all | 0.12 | 0.6438 |
| Q3 | inverse | 14 | 0.75 | 56 | 0.12 | 0.6605 |
| S1 | gaussian | 28 | 1.00 | all | 0.12 | 0.6611 |
| S2 | inverse | 14 | 0.75 | all | 0.12 | 0.5956 |
| S3 | exp | 28 | 1.00 | all | 0.12 | 0.5868 |
| S4 | gaussian | 28 | 1.00 | 56 | 0.12 | 0.6409 |

## 해석

- Label KNN 단독은 Q 계열과 S1에서 약합니다.
- S2/S3는 꽤 쓸 만합니다.
- 따라서 단독 제출 후보가 아니라 `S2/S3 보정용` 또는 약한 ensemble 성분으로 쓰는 것이 적절합니다.

## 생성된 후보

| file | 설명 |
| --- | --- |
| `label_knn_submission.csv` | label-only KNN 단독 |
| `ensemble_labelknn50_v3robust50_submission.csv` | KNN 50% + V3 robust 50% |
| `ensemble_labelknn30_v3robust70_submission.csv` | KNN 30% + V3 robust 70% |
| `ensemble_labelknn20_friend40_v3robust40_submission.csv` | KNN 20% + friend 40% + V3 40% |
| `ensemble_labelknn15_friend60_v3robust25_submission.csv` | KNN 15% + friend 60% + V3 25% |
| `targetwise_knn_s2s3_30_v3_submission.csv` | S2/S3만 KNN 30%, 나머지 V3 |
| `targetwise_knn_s2s3_50_v3_submission.csv` | S2/S3만 KNN 50%, 나머지 V3 |
| `targetwise_friend60_v3_40_knn_s2s3_submission.csv` | friend/V3 blend에 S2/S3만 KNN 보정 |

## 제출 우선순위

기존 최고권 후보를 먼저 제출한 뒤, 제출 횟수가 남으면 아래를 확인합니다.

1. `targetwise_knn_s2s3_30_v3_submission.csv`
2. `targetwise_friend60_v3_40_knn_s2s3_submission.csv`
3. `ensemble_labelknn20_friend40_v3robust40_submission.csv`

KNN 단독은 우선순위가 낮습니다.
