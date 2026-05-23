# Public LB Round 2 전략

## 관측된 Public 점수

| file | public_log_loss | 해석 |
| --- | ---: | --- |
| `friend_semantic_anchor_blend_10.csv` | 0.6171 | 친구 semantic anchor 후보 |
| `blend_friend80_v2robust10_v3robust10_submission.csv` | 0.6090 | 현재 최고 |
| `blend_friend90_v3robust10_submission.csv` | 0.6126 | 친구 비중 90%는 과함 |
| `v3_semantic_robust_ensemble_submission.csv` | 0.6091 | 우리 V3 단독도 매우 강함 |

## 해석

- `friend80 + v2/v3`와 `v3 robust`가 거의 동률입니다.
- 친구 90%가 나빠졌으므로, friend 비중을 더 올리는 방향은 좋지 않습니다.
- 친구 단독 0.6171에서 v2/v3 정보를 섞어 0.6090까지 내려왔으므로, 우리 모델이 확실히 추가 정보를 갖고 있습니다.
- v3 단독 0.6091이므로, 다음 탐색은 friend-heavy가 아니라 `v3 중심 + 약한 friend/v2 보정`입니다.

## 새로 생성한 후보

| file | 의도 |
| --- | --- |
| `blend_friend60_v2robust20_v3robust20_submission.csv` | friend 비중을 60%로 낮춘 후보 |
| `blend_friend50_v2robust25_v3robust25_submission.csv` | friend/v2/v3 균형 후보 |
| `blend_friend40_v2robust20_v3robust40_submission.csv` | v3 중심 후보 |
| `blend_friend30_v2robust20_v3robust50_submission.csv` | v3 강한 후보 |
| `blend_friend20_v2robust20_v3robust60_submission.csv` | v3 매우 강한 후보 |
| `ensemble_v2robust25_v3robust75_submission.csv` | friend 없이 v3 중심으로 v2만 보정 |

## 다음 제출 우선순위

1. `blend_friend60_v2robust20_v3robust20_submission.csv`
2. `blend_friend40_v2robust20_v3robust40_submission.csv`
3. `ensemble_v2robust25_v3robust75_submission.csv`
4. `blend_friend50_v2robust25_v3robust25_submission.csv`
5. `blend_friend30_v2robust20_v3robust50_submission.csv`

이 순서는 `0.6090` 근처에서 너무 멀리 벗어나지 않으면서, friend 비중을 낮추는 방향을 점검하도록 설계했습니다.

## 현재 결론

최고점 근처의 구조는 `friend + v2 + v3`입니다. 하지만 friend를 90%로 올리면 나빠지고, v3 단독도 0.6091로 강하므로, 앞으로는 v3를 중심 모델로 두고 friend는 보정 성분으로 쓰는 것이 맞습니다.
