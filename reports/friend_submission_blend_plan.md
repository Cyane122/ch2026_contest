# 친구 Submission 기반 Blend 계획

## 입력 파일

- 원본: `C:/Users/KNova/Documents/카카오톡 받은 파일/ch2026_submission_semantic_anchor_blend_10.csv`
- workspace 복사본: `submissions/friend_semantic_anchor_blend_10.csv`
- 친구 후보 Public LB: `0.6171`
- 우리 기존 최고 Public LB: `robust_ensemble_submission.csv = 0.6240`

## 분포 비교

| target | corr_friend_robust | mae_friend_robust | friend_mean - robust_mean |
| --- | ---: | ---: | ---: |
| Q1 | 0.4724 | 0.1681 | 0.0104 |
| Q2 | 0.4085 | 0.1286 | 0.0058 |
| Q3 | 0.3208 | 0.1642 | 0.0339 |
| S1 | 0.8123 | 0.0888 | 0.0652 |
| S2 | 0.9044 | 0.1007 | -0.0444 |
| S3 | 0.8733 | 0.1064 | -0.0457 |
| S4 | 0.6129 | 0.1151 | -0.0342 |

Q 계열은 상관이 낮아 blend 효과를 기대할 수 있습니다. S 계열은 상관이 높지만 평균 레벨 차이가 커서, 친구 후보의 calibration이 Public에 더 잘 맞았을 가능성이 있습니다.

## 생성한 후보

| file | 의도 |
| --- | --- |
| `blend_friend90_robust10_submission.csv` | 친구 후보를 거의 유지하면서 우리 robust 정보만 약하게 추가 |
| `blend_friend80_robust20_submission.csv` | 가장 우선적인 blend 후보 |
| `blend_friend70_robust30_submission.csv` | 우리 robust를 조금 더 반영 |
| `blend_friend60_robust40_submission.csv` | 공격적인 혼합 후보 |
| `blend_friend50_robust50_submission.csv` | 다양성 확인용 50/50 후보 |
| `blend_targetwise_friend_heavy_submission.csv` | target별로 친구 가중치 65~85% 적용 |
| `blend_targetwise_friend_q80_s70_submission.csv` | Q는 친구 80%, S는 친구 70% 중심 |

## 다음 제출 우선순위

현재 blend 후보들은 V2 shift-6 robust 후보를 기준으로 다시 생성되었습니다.

1. `blend_friend80_robust20_submission.csv`
2. `blend_friend90_robust10_submission.csv`
3. `v2_shift6_robust_ensemble_submission.csv`
4. `blend_targetwise_friend_heavy_submission.csv`
5. `blend_friend70_robust30_submission.csv`

이유: 친구 단독이 이미 `0.6171`로 가장 좋으므로, 첫 시도는 친구 후보에서 멀리 움직이지 않는 것이 안전합니다. 동시에 Q 계열 상관이 낮아 `10~20%` 정도의 우리 robust 정보가 Public을 더 낮출 가능성이 있습니다.

## 이후 개선 방향

친구 코드의 강점인 `6시간 shift`, `subject-hole CV`, `semantic feature`, `target별 shrink/strategy selection`을 우리 파이프라인에 흡수합니다. 단기적으로는 blend 제출 후보를 먼저 확인하고, 중기적으로는 feature/model 자체를 통합합니다.
