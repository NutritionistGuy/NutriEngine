"""
risk_engine.py
==============
축 B 핵심: 병무청 신검 BMI 분포 × 식단 영양 → 위험 예측 · what-if 시뮬.

⚠ B.2 데이터 한계 (설계 전제)
  1. 개인 병명 없음 — BMI·체격·시력만 존재. 개인 진단 불가.
  2. 공개 BMI는 18.5~35 구간만 포함 → 절단 분포. 결과에 반드시 명시.
  3. 신검은 입대 *전* 19세 기준.
     B.2 = 수요(어떤 몸이 입대하나), D.2 = 공급(무슨 밥을 먹나).
     분석 의미: 정합성 평가 + 식단 지속 시 군별 위험 시뮬.
     "식단→신검건강" 인과관계로 해석하면 틀림.

주요 외부 인터페이스 (api_client 포함 앱이 호출)
  classify_bmi(bmi)                     → BMI군 문자열
  bmi_group_ratio(bmi_list)             → 군별 비율 dict
  compute_bmr(weight_kg, height_cm, age)→ 기초대사량 kcal
  estimate_tdee(bmr, activity)          → 총에너지소비량 kcal
  energy_balance(diet_kcal, tdee)       → 일일 과부족 kcal
  weight_change_30d(delta_kcal)         → 30일 체중 변화 kg
  risk_score(nutrition, bmi, tdee)      → 위험지수 dict (0~1)
  apply_whatif(nutrition, ...)          → 조정 후 영양성분 dict
  run_simulation(dist, diet_nutrition, days, what_if) → 전체 시뮬 결과
  sum_diet_nutrition(menus, get_nutrition_fn)         → 하루 합산 영양성분
  get_bmi_ratio_from_dist(dist)         → 군별 비율 추출/근사
"""

from __future__ import annotations
from typing import Callable, Optional

# ────────────────────────────────────────────────────────────────────────────
# 1. BMI 군 분류 (한국 비만학회 기준)
# ────────────────────────────────────────────────────────────────────────────

BMI_GROUPS = {
    "저체중":   (0.0,  18.5),
    "정상":     (18.5, 23.0),
    "과체중":   (23.0, 25.0),
    "비만":     (25.0, 30.0),
    "고도비만": (30.0, float("inf")),
}
BMI_GROUP_ORDER = ["저체중", "정상", "과체중", "비만", "고도비만"]

# B.2 공개 구간 18.5~35 → 저체중·고도비만 실질 절단 → 아래 대표값도 절단 영향 받음
_GROUP_REPR_BMI = {
    "저체중": 17.5, "정상": 21.0, "과체중": 24.0,
    "비만": 27.0,  "고도비만": 32.0,
}


def classify_bmi(bmi: float) -> str:
    """BMI 수치 → 군 이름."""
    for group, (lo, hi) in BMI_GROUPS.items():
        if lo <= bmi < hi:
            return group
    return "고도비만"


def bmi_group_ratio(bmi_list: list[float]) -> dict:
    """BMI 목록 → 각 군 비율 dict. B.2 절단 경고 포함."""
    total = len(bmi_list)
    if not total:
        return {g: 0.0 for g in BMI_GROUP_ORDER}
    counts = {g: 0 for g in BMI_GROUP_ORDER}
    for b in bmi_list:
        counts[classify_bmi(b)] += 1
    ratios: dict = {g: round(v / total, 4) for g, v in counts.items()}
    ratios["_note"] = "B.2 공개 BMI 18.5~35 구간만 포함 → 저체중·고도비만 과소계상"
    return ratios


# ────────────────────────────────────────────────────────────────────────────
# 2. 에너지 대사 계산
# ────────────────────────────────────────────────────────────────────────────

ACTIVITY_LEVELS = {
    "sedentary":  1.200,
    "light":      1.375,
    "moderate":   1.550,
    "active":     1.725,   # 훈련소 기본값 (군사훈련 수준)
    "very_active": 1.900,
}
DEFAULT_ACTIVITY   = "active"
DEFAULT_HEIGHT_CM  = 173.0   # 20대 초반 한국 남성 평균
DEFAULT_AGE        = 19      # 병역판정검사 연령
KCAL_PER_KG_FAT    = 7700    # 체지방 1kg ≈ 7,700 kcal


def compute_bmr(weight_kg: float, height_cm: float = DEFAULT_HEIGHT_CM,
                age: int = DEFAULT_AGE) -> float:
    """Mifflin-St Jeor 공식 (남성). kcal/일."""
    return round(10 * weight_kg + 6.25 * height_cm - 5 * age + 5, 1)


def estimate_tdee(bmr: float, activity: str = DEFAULT_ACTIVITY) -> float:
    """TDEE = BMR × 활동계수."""
    pal = ACTIVITY_LEVELS.get(activity, ACTIVITY_LEVELS[DEFAULT_ACTIVITY])
    return round(bmr * pal, 1)


def energy_balance(diet_kcal: float, tdee: float) -> float:
    """일일 에너지 과부족 kcal (양수=과잉, 음수=부족)."""
    return round(diet_kcal - tdee, 1)


def weight_change_30d(delta_kcal_per_day: float) -> float:
    """30일 누적 체중 변화 kg. 7,700 kcal ≈ 체지방 1 kg."""
    return round(delta_kcal_per_day * 30 / KCAL_PER_KG_FAT, 3)


def bmi_from_weight(weight_kg: float, height_cm: float = DEFAULT_HEIGHT_CM) -> float:
    h_m = height_cm / 100
    return round(weight_kg / (h_m ** 2), 2)


# ────────────────────────────────────────────────────────────────────────────
# 3. 영양소 기반 위험 룰 (한국 영양소 섭취기준 2020 · WHO 기준)
# ────────────────────────────────────────────────────────────────────────────

DAILY_LIMITS = {
    "na":     2300,   # 나트륨 mg  (WHO·한국 공통 상한 목표)
    "satfat":   20,   # 포화지방산 g (총열량 7~10% 수준)
    "sugar":    50,   # 자유당 g 상한
    "chol":    300,   # 콜레스테롤 mg
}


def risk_score(nutrition: dict, bmi: float, tdee: float) -> dict:
    """
    단일 BMI 기준점 × 영양성분 → 위험 지수 dict.

    점수 0~1 (0=위험없음, 1=최고위험). 집단 위험 지표, 개인 진단 아님.

    반환 키:
      hypertension   — 나트륨 과잉 × 비만 가중 위험
      metabolic      — 열량 과잉 × 과체중/비만 가중 위험
      underweight    — 열량 부족 × 저체중 가중 위험
      nutrient_excess— 포화지방·당류 과잉 지수
      overall        — 가중 평균 종합 점수
    """
    na     = nutrition.get("na", 0.0)
    satfat = nutrition.get("satfat", 0.0)
    sugar  = nutrition.get("sugar", 0.0)
    kcal   = nutrition.get("kcal", 0.0)

    # 고혈압 위험: 나트륨 초과율 × BMI 25 이상 가중
    na_excess        = max(0.0, (na - DAILY_LIMITS["na"]) / DAILY_LIMITS["na"])
    bmi_ht_weight    = 1.0 + max(0.0, (bmi - 25.0) / 10.0)   # BMI 25→1.0, 35→2.0
    hypertension     = min(1.0, na_excess * bmi_ht_weight)

    # 대사 위험: 열량 과잉 × BMI 23 이상 가중
    kcal_over        = max(0.0, (kcal - tdee) / tdee) if tdee > 0 else 0.0
    bmi_meta_weight  = 1.0 + max(0.0, (bmi - 23.0) / 10.0)
    metabolic        = min(1.0, kcal_over * bmi_meta_weight * 2)

    # 저체중·저열량 위험
    kcal_deficit     = max(0.0, (tdee - kcal) / tdee) if tdee > 0 else 0.0
    uw_weight        = 1.5 if classify_bmi(bmi) == "저체중" else 0.5
    underweight      = min(1.0, kcal_deficit * uw_weight)

    # 영양소 과잉 지수 (포화지방·당류)
    satfat_ex        = max(0.0, (satfat - DAILY_LIMITS["satfat"]) / DAILY_LIMITS["satfat"])
    sugar_ex         = max(0.0, (sugar  - DAILY_LIMITS["sugar"])  / DAILY_LIMITS["sugar"])
    nutrient_excess  = min(1.0, (satfat_ex + sugar_ex) / 2)

    overall = round(
        hypertension   * 0.35 +
        metabolic      * 0.35 +
        underweight    * 0.15 +
        nutrient_excess* 0.15, 3
    )
    return {
        "hypertension":    round(hypertension, 3),
        "metabolic":       round(metabolic, 3),
        "underweight":     round(underweight, 3),
        "nutrient_excess": round(nutrient_excess, 3),
        "overall":         overall,
    }


# ────────────────────────────────────────────────────────────────────────────
# 4. What-if 식단 조정
# ────────────────────────────────────────────────────────────────────────────

def apply_whatif(
    nutrition: dict,
    sodium_reduction_pct: float = 0.0,
    calorie_reduction_pct: float = 0.0,
    satfat_reduction_pct: float  = 0.0,
) -> dict:
    """
    식단 조정 슬라이더 적용. 각 pct = 0~100 사이 퍼센트 감소율.
    원본을 변형하지 않고 새 dict 반환.
    """
    n = dict(nutrition)
    n["na"]     = round(n.get("na",     0.0) * (1 - sodium_reduction_pct  / 100), 1)
    n["kcal"]   = round(n.get("kcal",   0.0) * (1 - calorie_reduction_pct / 100), 1)
    n["satfat"] = round(n.get("satfat", 0.0) * (1 - satfat_reduction_pct  / 100), 1)
    # 열량 감소 → 탄수화물·지방 비례 감소 (단백질 보존)
    if calorie_reduction_pct > 0:
        r = 1 - calorie_reduction_pct / 100
        n["carb"] = round(n.get("carb", 0.0) * r, 1)
        n["fat"]  = round(n.get("fat",  0.0) * r, 1)
    return n


# ────────────────────────────────────────────────────────────────────────────
# 5. B.2 분포 비율 추출/근사
# ────────────────────────────────────────────────────────────────────────────

def get_bmi_ratio_from_dist(dist: dict) -> dict[str, float]:
    """
    get_mma_bmi_distribution() 반환값 → BMI 군별 비율 dict.
    원시 rows가 없으면 mean·obese_ratio로 정규분포 근사 (설명 자료 수준).
    """
    # dist에 직접 비율이 있으면 우선 사용
    direct = {g: dist.get(f"ratio_{g}", 0.0) for g in BMI_GROUP_ORDER}
    if any(v > 0 for v in direct.values()):
        return direct

    # mean·obese_ratio 기반 간이 근사
    mean   = dist.get("mean", 23.0)
    obese  = dist.get("obese_ratio", 0.15)
    over   = max(0.0, min(0.30, 0.20 + (mean - 23.0) * 0.04))
    under  = max(0.0, min(0.05, 0.03 - (mean - 21.0) * 0.01))
    normal = max(0.0, 1.0 - under - over - obese)
    return {
        "저체중":   round(under,  4),
        "정상":     round(normal, 4),
        "과체중":   round(over,   4),
        "비만":     round(obese,  4),
        "고도비만": 0.0,
    }


# ────────────────────────────────────────────────────────────────────────────
# 6. 전체 시뮬레이션 (군별 분포 × N일 후 분포 이동 예측)
# ────────────────────────────────────────────────────────────────────────────

def run_simulation(
    dist: dict,
    diet_nutrition: dict,
    days: int = 30,
    what_if: Optional[dict] = None,
    activity: str = DEFAULT_ACTIVITY,
) -> dict:
    """
    B.2 BMI 분포 × 1일 식단 영양 → N일 후 위험 예측.

    Parameters
    ----------
    dist          : get_mma_bmi_distribution() 반환값
    diet_nutrition: 1일 합산 영양성분 dict (sum_diet_nutrition 결과 권장)
    days          : 시뮬 기간 (기본 30일)
    what_if       : apply_whatif 슬라이더 키워드 인자 dict
                    예) {"sodium_reduction_pct": 20, "calorie_reduction_pct": 10}
    activity      : 활동 수준 (기본 "active" = 군사훈련)

    Returns
    -------
    dict:
      group_results      : 각 BMI군별 에너지균형·체중변화·위험지수
      distribution_shift : 원래 군 비율 vs 시뮬 후 군 비율
      summary            : 핵심 지표 요약
      _warnings          : B.2 데이터 한계 경고 목록
    """
    nutrition = apply_whatif(diet_nutrition, **(what_if or {}))
    mean_height = dist.get("mean_height_cm", DEFAULT_HEIGHT_CM)
    ratio_by_group = get_bmi_ratio_from_dist(dist)

    group_results: dict = {}
    for group in BMI_GROUP_ORDER:
        rep_bmi   = _GROUP_REPR_BMI[group]
        weight_kg = rep_bmi * (mean_height / 100) ** 2

        bmr   = compute_bmr(weight_kg, mean_height)
        tdee  = estimate_tdee(bmr, activity)
        delta = energy_balance(nutrition.get("kcal", 0.0), tdee)

        # N일 누적 체중 변화 (7,700kcal ≈ 1kg)
        dw        = round(delta * days / KCAL_PER_KG_FAT, 3)
        new_bmi   = bmi_from_weight(weight_kg + dw, mean_height)
        new_group = classify_bmi(new_bmi)
        scores    = risk_score(nutrition, rep_bmi, tdee)

        group_results[group] = {
            "rep_bmi":          rep_bmi,
            "weight_kg":        round(weight_kg, 1),
            "tdee":             round(tdee, 1),
            "diet_kcal":        nutrition.get("kcal", 0.0),
            "energy_delta_kcal": delta,
            "weight_change_kg": dw,
            "new_bmi":          new_bmi,
            "new_group":        new_group,
            "group_changed":    new_group != group,
            "risk":             scores,
        }

    # 군 비율 이동: 각 군에 속한 사람이 새 BMI로 이동하면 어느 군으로 가나
    new_ratio: dict[str, float] = {g: 0.0 for g in BMI_GROUP_ORDER}
    for orig_group in BMI_GROUP_ORDER:
        orig_r = ratio_by_group.get(orig_group, 0.0)
        if orig_r > 0:
            dest = group_results[orig_group]["new_group"]
            new_ratio[dest] = round(new_ratio[dest] + orig_r, 4)

    obese_orig = sum(ratio_by_group.get(g, 0.0) for g in ["비만", "고도비만"])
    obese_new  = sum(new_ratio.get(g, 0.0)      for g in ["비만", "고도비만"])

    return {
        "group_results": group_results,
        "distribution_shift": {
            "original":   ratio_by_group,
            "after_days": new_ratio,
            "days":       days,
        },
        "summary": {
            "days":                    days,
            "diet_kcal_per_day":       nutrition.get("kcal", 0.0),
            "diet_na_mg":              nutrition.get("na",   0.0),
            "diet_satfat_g":           nutrition.get("satfat", 0.0),
            "obese_ratio_change_ppt":  round((obese_new - obese_orig) * 100, 2),
            "what_if_applied":         what_if or {},
        },
        "_warnings": [
            "B.2 공개 BMI 18.5~35 구간만 포함 → 저체중·고도비만 비율 과소계상.",
            "신검은 입대 전 건강상태(수요). 식단→신검건강 인과관계 아님.",
            "개인 병명 진단 불가. 집단 위험 분포의 식단 의존성 시뮬 결과임.",
            "BMR: Mifflin-St Jeor 남성 공식. 활동계수: 훈련소 active(1.725) 가정.",
        ],
    }


# ────────────────────────────────────────────────────────────────────────────
# 7. api_client 연동 헬퍼
# ────────────────────────────────────────────────────────────────────────────

def sum_diet_nutrition(
    menus: list[dict],
    get_nutrition_fn: Callable[[str], Optional[dict]],
) -> dict:
    """
    메뉴 목록 × 영양조회함수 → 1일 영양성분 합산.
    get_nutrition_fn = api_client.get_nutrition
    영양정보 없는 메뉴는 건너뜀 (커버리지 메모 포함).
    """
    totals: dict[str, float] = {}
    found = 0
    for m in menus:
        n = get_nutrition_fn(m["name"])
        if not n:
            # API 매칭 실패 시 JSON cal만 kcal에 반영
            if m.get("cal"):
                totals["kcal"] = round(totals.get("kcal", 0.0) + m["cal"], 2)
            continue
        found += 1
        merged = dict(n)
        # API kcal이 0이면 army_diet.json의 cal 필드로 폴백
        if not merged.get("kcal") and m.get("cal"):
            merged["kcal"] = m["cal"]
        for k, v in merged.items():
            if k.startswith("_"):
                continue
            totals[k] = round(totals.get(k, 0.0) + (v or 0.0), 2)
    totals["_coverage"] = f"{found}/{len(menus)} 메뉴 영양정보 매칭됨"
    return totals


# ────────────────────────────────────────────────────────────────────────────
# 8. 자체 점검 (python risk_engine.py)
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== risk_engine 자체 점검 ===\n")

    # 더미 1일 합산 영양 (훈련소 식단 수준)
    DUMMY_NUTRITION = {
        "kcal": 2350, "protein": 85, "fat": 70, "carb": 330,
        "sugar": 55,  "fiber": 15,   "na": 3800, "satfat": 22,
        "chol": 280,  "ca": 600,     "fe": 12,   "k": 2800,
        "vita": 700,  "vitc": 80,    "vitd": 4,
    }
    # 더미 B.2 분포 통계 (실제는 get_mma_bmi_distribution 반환값)
    DUMMY_DIST = {
        "count": 500, "mean": 23.2, "p50": 22.8,
        "p90": 27.1,  "obese_ratio": 0.18,
        "note": "더미 데이터 — 실제 API 키 설정 후 갱신 필요",
    }

    # 1. BMI 분류
    print("[1] BMI 분류 (한국 비만학회 기준)")
    for b in [16.5, 20.0, 24.0, 27.0, 32.0]:
        print(f"   BMI {b:5.1f} → {classify_bmi(b)}")

    # 2. 에너지 균형 (정상군 BMI 21 기준)
    print("\n[2] 에너지 균형 (BMI=21, 신장 173cm)")
    w_kg = 21.0 * (DEFAULT_HEIGHT_CM / 100) ** 2
    bmr_ = compute_bmr(w_kg)
    tdee_ = estimate_tdee(bmr_)
    delta_ = energy_balance(DUMMY_NUTRITION["kcal"], tdee_)
    dw30 = weight_change_30d(delta_)
    print(f"   체중={w_kg:.1f}kg  BMR={bmr_:.0f}kcal  TDEE={tdee_:.0f}kcal")
    print(f"   식단={DUMMY_NUTRITION['kcal']}kcal  과부족={delta_:+.0f}  30일 Δ체중={dw30:+.3f}kg")

    # 3. 위험 점수 (비만군 BMI 27)
    print("\n[3] 위험 점수 (BMI=27, 고나트륨 식단)")
    tdee_ob = estimate_tdee(compute_bmr(27.0 * (DEFAULT_HEIGHT_CM / 100) ** 2))
    scores_ = risk_score(DUMMY_NUTRITION, 27.0, tdee_ob)
    for k, v in scores_.items():
        bar = "#" * int(v * 10)
        print(f"   {k:<18} {v:.3f}  {bar}")

    # 4. What-if (나트륨 30%↓ · 열량 10%↓)
    print("\n[4] What-if: 나트륨 30%↓ · 열량 10%↓")
    n2 = apply_whatif(DUMMY_NUTRITION, sodium_reduction_pct=30, calorie_reduction_pct=10)
    s2 = risk_score(n2, 27.0, tdee_ob)
    print(f"   나트륨   {DUMMY_NUTRITION['na']:.0f}mg → {n2['na']:.0f}mg")
    print(f"   열량     {DUMMY_NUTRITION['kcal']:.0f}kcal → {n2['kcal']:.0f}kcal")
    print(f"   고혈압   {scores_['hypertension']:.3f} → {s2['hypertension']:.3f}")
    print(f"   대사위험 {scores_['metabolic']:.3f} → {s2['metabolic']:.3f}")
    print(f"   종합     {scores_['overall']:.3f} → {s2['overall']:.3f}")

    # 5. 30일 시뮬
    print("\n[5] 30일 시뮬 (더미 B.2 분포 × 고나트륨 식단)")
    result = run_simulation(DUMMY_DIST, DUMMY_NUTRITION, days=30)
    print(f"   비만군 변화: {result['summary']['obese_ratio_change_ppt']:+.2f}%p")
    for g in BMI_GROUP_ORDER:
        gr = result["group_results"][g]
        chg = f"→ {gr['new_group']}" if gr["group_changed"] else "유지"
        print(f"   {g:<6}  TDEE={gr['tdee']:5.0f}  "
              f"Δ체중={gr['weight_change_kg']:+.2f}kg  군={chg:10s}  "
              f"위험종합={gr['risk']['overall']:.3f}")

    print("\n[6] What-if 30일 시뮬 (나트륨30%↓ · 열량10%↓ 적용)")
    result2 = run_simulation(DUMMY_DIST, DUMMY_NUTRITION, days=30,
                             what_if={"sodium_reduction_pct": 30,
                                      "calorie_reduction_pct": 10})
    print(f"   비만군 변화: {result2['summary']['obese_ratio_change_ppt']:+.2f}%p")
    for g in ["비만", "고도비만"]:
        orig = result["distribution_shift"]["original"].get(g, 0)
        new1 = result["distribution_shift"]["after_days"].get(g, 0)
        new2 = result2["distribution_shift"]["after_days"].get(g, 0)
        print(f"   {g}: 원본={orig:.1%}  현식단후={new1:.1%}  개선식단후={new2:.1%}")

    print("\n경고사항:")
    for w in result["_warnings"]:
        print(f"  [!] {w}")

    print("\n완료.")
