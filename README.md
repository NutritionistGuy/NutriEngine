# 병무청 신검 × 식단 의사결정 시스템

병무청 신체검사 BMI 분포(B.2)와 군 급식 영양정보를 융합해,
식단 변화에 따른 건강 위험도를 BMI군별로 예측하고 최적 식단을 제안하는 프로토타입.

---

## 핵심 아이디어

- **신검(B.2) = 수요**: 입영 청년의 실측 BMI 분포
- **식단(D.2) = 공급**: 육군훈련소 실제 급식 메뉴
- 같은 식단도 BMI군마다 위험이 다르다 → 신검 분포가 있어야 가능한 분석

---

## 기능

**축 A — 식단 영양 판단**
- 1일 영양소 충족도 / 부족·과잉 감지
- 메뉴 영양밀도 랭킹
- 나트륨·포화지방 과다 메뉴 주의 표시
- 대체 메뉴 추천

**축 B — 건강위험 예측**
- BMI군별 에너지 균형 및 30일 체중 변화 시뮬
- 고혈압·대사·저체중 위험 점수 산출
- What-if: 나트륨·열량·포화지방 슬라이더로 식단 조정 효과 비교

---

## 사용 데이터

| 코드 | 데이터 | 출처 | 키 |
|---|---|---|---|
| B.2 | 병무청 병역판정 신체검사 | data.go.kr REST API | `MMA_KEY` |
| C.1 | 식약처 식품영양성분DB | data.go.kr REST API | `FOOD_KEY` |
| C.2 | 해수부 원재료 영양성분 | data.go.kr REST API | `RAW_KEY` |
| D.2 | 육군훈련소 급식 식단 | 로컬 JSON 파일 | 불필요 |

API 키가 없는 항목은 자동으로 더미 데이터로 동작합니다.

---

## 설치 및 실행

```bash
pip install -r requirements.txt
```

`.env` 파일 생성 (키 없는 항목은 생략 가능):

```
MMA_KEY=병무청_발급키
FOOD_KEY=식약처_발급키
RAW_KEY=해수부_발급키
```

식단 파일 배치:

```
data/army_diet.json   ← 국방부 포털에서 직접 다운로드
```

앱 실행:

```bash
streamlit run app.py
```

---

## 파일 구조

```
app.py               Streamlit UI
api_client.py        API 호출 / 로컬 식단 로드 / SQLite 캐시
nutrition_engine.py  축A — 영양 분석
risk_engine.py       축B — 위험 예측 · 시뮬레이션
bmi_model.py         GMM 기반 BMI 분포 피팅
colab_bmi_gmm.py     GMM 학습 스크립트 (Colab 전용)
data/army_diet.json  급식 식단 데이터
data/gmm_params.json GMM 학습 파라미터 (bmi_model 사용)
```

---

## 주의사항

- B.2 공개 BMI는 18.5~35 구간만 포함 → 저체중·고도비만 비율 과소계상
- 신검은 입대 전 건강 상태이므로 "식단 → 신검 건강" 인과관계로 해석하면 안 됨
- 개인 병명 진단이 아닌 **집단 위험 분포의 식단 의존성** 시뮬 결과
