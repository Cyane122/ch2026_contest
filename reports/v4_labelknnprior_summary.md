# V4 날짜 가중 Label KNN Prior 개선 요약

## 핵심 아이디어

친구 코드의 핵심 가정인 "같은 사람의 가까운 날짜 레이블이 가장 강한 힌트"를 우리 트리 기반 모델의 prior 후보로 통합했다.

- 같은 subject의 과거/주변 train 레이블만 사용한다.
- 날짜 거리는 `1 / (days + 1)`로 가중한다.
- subject 평균을 12% 섞어 과도한 단일 근접값 의존을 줄인다.
- 이 KNN prior를 `subject_prior`, `neighbor_prior`, `subject_neighbor_prior`와 같은 후보군에 넣고 validation split별로 자동 선택하게 했다.

## 구현 파일

- `scripts/train_baseline_model.py`
  - `label_knn_prior()` 추가
  - `label_knn_prior`, `subject_knn_prior`, `neighbor_knn_prior`를 prior 후보로 추가
- `scripts/make_submission_variants.py`
  - v4 robust와 friend/v2/v3 조합 blend 후보 추가

## 내부 Validation 관찰

KNN prior는 모든 target에서 일괄적으로 강하지는 않았지만, 여러 split에서 실제 best 조합으로 선택됐다.

- Q2: `label_knn_prior`, `neighbor_knn_prior`가 여러 split에서 선택
- S2: interleaved/subject-hole/late7에서 KNN 계열 prior가 자주 선택
- S3: subject-hole에서 `label_knn_prior` 선택
- S4: 일부 split에서 `label_knn_prior`, `subject_knn_prior` 선택

평균 validation log loss:

- `interleaved_offset1`: 0.5731
- `interleaved_offset2`: 0.5832
- `subject_hole0`: 0.6220
- `subject_hole1`: 0.5669
- `subject_hole2`: 0.5952
- `late7`: 0.6108

## 생성된 V4 제출 후보

- `submissions/v4_labelknnprior_baseline_submission.csv`
- `submissions/v4_labelknnprior_robust_ensemble_submission.csv`
- `submissions/ensemble_v3robust50_v4knnrobust50_submission.csv`
- `submissions/blend_friend80_v4knnrobust20_submission.csv`
- `submissions/blend_friend60_v3robust20_v4knnrobust20_submission.csv`
- `submissions/blend_friend40_v3robust30_v4knnrobust30_submission.csv`
- `submissions/blend_friend30_v3robust35_v4knnrobust35_submission.csv`
- `submissions/blend_friend50_v2robust15_v3robust20_v4knnrobust15_submission.csv`
- `submissions/blend_friend40_v2robust15_v3robust25_v4knnrobust20_submission.csv`

## 다음 제출 우선순위

현재 public LB 기준으로 이미 `friend` 비중 80%가 0.6090, `v3` 단독이 0.6091로 거의 동률이었다. 그래서 다음 실험은 friend 비중을 더 줄이고 v3/v4 쪽 다양성을 올리는 방향이 합리적이다.

1. `blend_friend40_v3robust30_v4knnrobust30_submission.csv`
2. `blend_friend30_v3robust35_v4knnrobust35_submission.csv`
3. `ensemble_v3robust50_v4knnrobust50_submission.csv`
4. `blend_friend40_v2robust15_v3robust25_v4knnrobust20_submission.csv`
5. `blend_friend60_v3robust20_v4knnrobust20_submission.csv`

## 해석

Label KNN 단독 모델은 내부 검증에서 Q 계열과 S1에는 약했지만, S2/S3 및 일부 S4에 유효했다. 따라서 전체를 KNN으로 대체하기보다는, 지금처럼 model prior 후보나 target-wise blend로 제한적으로 섞는 편이 안전하다.

public LB가 train 기간 내 누락 날짜 구조를 강하게 반영한다면 1~2번 후보가 0.609 근처에서 추가 개선될 가능성이 있다. 반대로 private가 더 시간 외삽에 가까우면 3번처럼 friend 없이 v3/v4만 섞은 후보가 더 안정적일 수 있다.

## 2026-05-23 Public LB 결과와 다음 후보

제출 결과:

- `blend_friend40_v3robust30_v4knnrobust30_submission.csv`: 0.5992
- `blend_friend30_v3robust35_v4knnrobust35_submission.csv`: 0.5991
- `ensemble_v3robust50_v4knnrobust50_submission.csv`: 0.6049

해석:

- friend가 없는 v3/v4 앙상블보다 friend 30~40% blend가 확실히 좋다.
- friend 30%와 40% 차이가 0.0001뿐이라 최적점은 30~40% 사이 또는 30%보다 약간 낮은 곳일 수 있다.
- 다음 제출은 넓은 탐색보다 25/35% 및 v4 비중 미세 조정이 우선이다.

다음 3회 제출 우선순위:

1. `submissions/blend_friend35_v3robust325_v4knnrobust325_submission.csv`
2. `submissions/blend_friend25_v3robust375_v4knnrobust375_submission.csv`
3. `submissions/blend_friend30_v3robust30_v4knnrobust40_submission.csv`
