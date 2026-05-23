# MLP 딥러닝 모델 실험 요약

## 목적

PyTorch/TensorFlow 없이 현재 환경에서 바로 재현 가능한 tabular neural network를 추가했습니다. 데이터가 450행으로 작기 때문에 큰 딥러닝 모델 대신 강하게 정규화한 `sklearn.neural_network.MLPClassifier`를 사용했습니다.

## 모델 구조

- feature source: V3 semantic feature table
- preprocessing:
  - median imputation
  - variance threshold
  - `SelectKBest`
  - standard scaling
- model candidates:
  - `mlp_tiny`: hidden `(32,)`
  - `mlp_small`: hidden `(64, 16)`
  - `mlp_wide_regularized`: hidden `(96, 24)`
- regularization:
  - high `alpha`
  - early stopping
  - prior/neighbor-prior blend

## 내부 검증 결과

| split | MLP average logloss |
| --- | ---: |
| interleaved_offset1 | 0.5810 |
| interleaved_offset2 | 0.5957 |
| subject_hole0 | 0.6512 |
| subject_hole1 | 0.5569 |
| subject_hole2 | 0.6078 |
| late7 | 0.6455 |

V3 tree 계열보다 전반적으로 약합니다. 따라서 단독 제출 후보보다는 diversity blend 용도로만 쓰는 것이 맞습니다.

## Diversity 비교

| target | corr_mlp_v3 | corr_mlp_friend |
| --- | ---: | ---: |
| Q1 | 0.9389 | 0.6113 |
| Q2 | 0.8592 | 0.4868 |
| Q3 | 0.8820 | 0.5047 |
| S1 | 0.9222 | 0.7993 |
| S2 | 0.9777 | 0.8978 |
| S3 | 0.9885 | 0.8780 |
| S4 | 0.9208 | 0.7301 |

MLP와 V3의 전체 상관은 `0.9335`로 높습니다. 다양성이 크지 않으므로 MLP 가중치는 5~10% 이하가 적절합니다.

## 생성된 후보

| file | 설명 |
| --- | --- |
| `mlp_robust_ensemble_submission.csv` | MLP split ensemble |
| `ensemble_v3robust90_mlp10_submission.csv` | V3 90% + MLP 10% |
| `blend_friend80_v3robust15_mlp05_submission.csv` | friend 80% + V3 15% + MLP 5% |
| `blend_friend60_v3robust35_mlp05_submission.csv` | friend 60% + V3 35% + MLP 5% |

## 결론

딥러닝 단독 모델은 현재 데이터 크기에서는 tree/prior 계열을 이기기 어렵습니다. 다만 Public LB에서 이미 0.609 근처까지 내려온 상태라, MLP 5% 정도를 섞은 후보는 다양성 확인용으로 한 번쯤 제출해볼 수 있습니다.

추천 우선순위는 낮습니다. 먼저 `v3 중심 + friend/v2 보정` 후보를 확인한 뒤, 제출 횟수가 남으면 MLP blend를 확인합니다.
