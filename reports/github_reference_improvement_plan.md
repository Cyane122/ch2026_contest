# GitHub 참고 코드 기반 개선 방향

참고 repo: `Cyane122/ch2026_contest`

친구 코드의 Public LB가 약 `0.6171`, 우리 현재 최고 관측값이 `robust_ensemble_submission.csv = 0.6240`이므로, 차이는 약 `0.0069`입니다. 현재 격차는 충분히 따라갈 수 있는 범위로 보입니다.

## 참고 코드의 핵심 차이

### 1. 6시간 shift 날짜 정렬

친구 코드의 `src/ch2026_chained.py`는 timestamp에서 6시간을 빼서 sensor date를 만듭니다.

```python
return (pd.to_datetime(timestamp) - pd.Timedelta(hours=6)).dt.date.astype(str)
```

즉 하루를 `00:00~24:00`이 아니라 `06:00~다음날 06:00`에 가깝게 봅니다. 수면/취침 전 행동 문제에서는 이 정렬이 꽤 중요합니다.

우리 현재 모델은 lifelog calendar date 기준 daypart 집계가 중심입니다. 다음 개선 1순위는 같은 센서를 `shifted_lifelog_date = timestamp - 6h` 기준으로 다시 집계하고, 기존 feature와 함께 비교하는 것입니다.

### 2. Subject-hole CV

친구 코드는 각 subject timeline을 여러 block으로 나눈 뒤, fold마다 중간중간 hole을 검증으로 둡니다.

이 방식은 submission이 train 이후만 있는 게 아니라 train 날짜 사이에 끼어 있는 구조와 잘 맞습니다. 우리도 interleaved split을 넣었지만, 아직 offset 2개와 late7만 있습니다.

개선 방향:

- subject-hole 5 folds 추가
- split별 평균만 보지 말고 `mean + 0.25*std + 0.05*max` 같은 안정성 점수로 후보 선택
- Public에서 robust가 baseline보다 좋았으므로, 단일 split 최적화보다 multi-fold 안정성 선택을 우선

### 3. 수면 중심 시간창

친구 코드의 `src/ch2026_features.py`는 시간창을 훨씬 세밀하게 나눕니다.

- `prebed_21_24`
- `sleep_21_09`
- `late_00_03`
- `deep_03_06`
- `wake_06_09`
- `day_09_18`
- `evening_18_21`

우리 모델은 `night/morning/afternoon/evening` 중심이라 수면 도메인 정보가 덜 직접적입니다.

개선 방향:

- 기존 619개 피처에 sleep-centered window features 추가
- 특히 `screen`, `light`, `HR`, `pedometer`, `usage`는 `prebed/sleep/late/deep/wake` 집계를 별도로 만들기
- window 간 차이/비율도 추가: `prebed -> sleep`, `late -> deep`, `deep -> wake`

### 4. Semantic feature engineering

친구 코드는 object column을 단순 길이/평균으로만 보지 않고 의미 카테고리로 풉니다.

- Ambience: speech, silence, music, vehicle, outdoor, sleep_noise
- UsageStats: social, media, communication, game, finance/shopping, routine/health 등
- GPS: home distance, mobility radius
- WiFi/BLE: familiar ratio, novel ratio, density, RSSI

우리 모델도 일부 `music/speech/vehicle`, usage total, BLE/WiFi density를 쓰지만 범위가 훨씬 좁습니다.

개선 방향:

- `mUsageStats`의 app category feature를 우선 추가
- `mAmbience` 그룹 feature 추가
- GPS home distance와 mobility radius 추가
- WiFi/BLE familiar/novel ratio 추가

### 5. Label history feature 확장

친구 코드는 target별 label history를 feature로 넣습니다.

- last 1/3/7/14 mean
- last value
- streak
- change rate
- day-of-week mean
- history count

우리는 neighbor prior와 subject prior를 강하게 쓰고 있지만, 모델 입력 feature로 target history를 충분히 확장하지는 않았습니다.

개선 방향:

- 현재 `neighbor_prior`는 유지
- target별 `last_1/3/7/14`, `same_dow_mean`, `streak`, `change_rate`를 feature로 추가
- 단, validation에서 현재 row 라벨 누수 없이 fold별 fit history만 사용하도록 구현

### 6. 모델 registry와 shrinkage 선택

친구 코드의 `ch2026_logloss.py`는 여러 strategy를 fold별로 점수화하고, target별로 `shrink_alpha`까지 선택합니다.

선택 예시:

| target | strategy | shrink_alpha | mean_logloss |
| --- | --- | ---: | ---: |
| Q1 | model_extra_trees | 1.0 | 0.6568 |
| Q2 | meta_rule_extra_trees | 1.0 | 0.6267 |
| Q3 | blend_catboost_rule_0.75 | 0.75 | 0.6158 |
| S1 | blend_catboost_rule_0.75 | 1.0 | 0.5470 |
| S2 | meta_rule_extra_trees | 1.0 | 0.6105 |
| S3 | blend_extra_trees_rule_0.75 | 0.75 | 0.6263 |
| S4 | blend_extra_trees_rule_0.75 | 1.0 | 0.6353 |

우리도 shrink submission 후보를 만들었지만, target별 shrink를 CV로 고르지는 않았습니다.

개선 방향:

- submission 전체 shrink보다 target별 shrink alpha 적용
- 후보 alpha: `0.70, 0.80, 0.90, 1.00` plus prior-blend
- selection score는 Public 결과를 반영해 robust/late 성향에 penalty를 낮게 주기

### 7. S-to-Q chained model

친구 코드는 S1-S4를 먼저 예측한 뒤, 그 확률을 Q1-Q3 feature에 넣는 chained 모델을 둡니다.

로컬 chained score에서 Q 계열은 아주 좋지는 않지만, Q1/Q2/Q3는 수면 지표와 상관이 있으므로 우리 모델에도 `predicted_S*_prob`를 Q model input으로 넣어볼 가치가 있습니다.

개선 방향:

- S target 모델을 먼저 학습
- fold OOF S probability를 train Q feature에 추가
- full train S probability를 sample Q feature에 추가
- Q1/Q2/Q3만 chained version과 기존 version을 ensemble

## 우선 적용 순서

1. **6시간 shift date 집계 추가**
   - 기대 효과가 가장 크고, 코드 변경 대비 리스크가 낮습니다.

2. **subject-hole 5-fold validation 추가**
   - Public에서 robust가 더 좋았으므로, selection 방식을 robust하게 바꿔야 합니다.

3. **target history features 확장**
   - 이미 neighbor prior가 효과가 있었으므로, 같은 계열의 개선 여지가 큽니다.

4. **target별 shrink alpha 선택**
   - 오늘 제출 결과상 과신을 줄이는 방향이 맞습니다.

5. **semantic app/ambience/GPS/WiFi/BLE feature 추가**
   - 구현량은 크지만 Q 계열 개선 가능성이 큽니다.

6. **S-to-Q chaining**
   - Q 계열 개선 후보로 실험하되, 단독 채택보다 ensemble 후보로 두는 편이 안전합니다.

## 다음 제출일 전략

다음 제출 가능일에는 한 번에 큰 변화를 넣기보다 아래 순서로 확인하는 것이 좋습니다.

1. 기존 `robust_ensemble_submission.csv` 재기준 확인
2. `robust_shrink05_submission.csv` 또는 targetwise shrink 후보
3. 6시간 shift + subject-hole CV 기반 robust 후보
4. semantic feature 추가 후보
5. 친구 submission과 우리 robust submission의 단순 blend 후보

친구 submission CSV를 받을 수 있으면, `0.6171` 후보와 우리 `0.6240` 후보를 50/50, 60/40, 70/30으로 섞는 것이 가장 빠른 개선 실험입니다. 두 모델 계열이 다르면 Public에서 0.6171보다 더 좋아질 가능성도 있습니다.
