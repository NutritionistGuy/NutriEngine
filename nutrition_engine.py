"""
nutrition_engine.py
===================
축 A 핵심: 식단 자체 영양 판단 (식단만 보고 — B.2 신검 불필요).

기능
  1. 영양소 충족도 & 부족/과잉 영양소 감지
  2. 영양 효율 좋은 메뉴 랭킹 (영양밀도 점수)
  3. 주의 음식 감지 (나트륨·포화지방 과잉 / 식중독 리스크 휴리스틱)
  5. 대체식 추천 (같은 열량대, 더 나은 영양소 프로파일)

데이터: D.2(식단) × C.1/C.2(영양 API) — api_client 연동 전제.
기능4(위험 예측·what-if)는 축B이므로 risk_engine.py가 담당.

주요 외부 인터페이스 (app.py 또는 Streamlit이 호출)
  nutrient_adequacy(daily_nutrition)          → 영양소별 충족률 dict
  deficient_nutrients(daily_nutrition, thr)   → 부족 영양소 목록
  excess_nutrients(daily_nutrition)           → 과잉 영양소 목록
  menu_nutrient_score(nutrition)              → 영양밀도 점수 float
  rank_menus_by_efficiency(menus, fn)         → 효율 순 메뉴 목록
  flag_caution_menus(menus, fn)               → 주의 메뉴 목록
  suggest_alternatives(target, pool, fn, n)   → 대체식 추천 목록
  sum_diet_nutrition(menus, fn)               → 1일 합산 영양성분 dict
  daily_report(menus, fn)                     → 축A 종합 리포트 dict
"""

from __future__ import annotations
from typing import Callable, Optional

# ────────────────────────────────────────────────────────────────────────────
# 1. 한국인 영양소 섭취기준 (2020, 19~29세 남성 — 입영 청년 대상)
# ────────────────────────────────────────────────────────────────────────────

# 권장/충분섭취량: 이 이상 섭취를 목표로 함
DAILY_RECOMMENDED = {
    "kcal":    2600,   # kcal (보통 활동. 훈련소 active 기준은 risk_engine TDEE 참고)
    "protein":   65,   # g
    "fat":       58,   # g (총에너지 20% 기준)
    "carb":     390,   # g (총에너지 60% 기준)
    "fiber":     25,   # g
    "ca":       800,   # mg
    "fe":        10,   # mg
    "k":       3500,   # mg
    "vita":     800,   # μg RAE
    "vitc":     100,   # mg
    "vitd":      10,   # μg
}

# 목표/상한섭취량: 이 이하 유지를 목표로 함 (risk_engine.DAILY_LIMITS와 동일 기준)
DAILY_UPPER = {
    "na":      2300,   # mg
    "sugar":     50,   # g (자유당)
    "satfat":    20,   # g (총에너지 7% 미만)
    "chol":     300,   # mg
}

# 영양밀도 점수 가중치 (긍정 영양소 — 100kcal당 충족 기여도에 곱함)
_POS_WEIGHTS = {
    "protein": 2.0, "fiber": 1.5, "ca": 1.0, "fe": 1.0,
    "k": 0.8, "vita": 0.7, "vitc": 0.7, "vitd": 0.8,
}

# 식중독 주의 키워드 (이름 기반 휴리스틱)
_FOOD_SAFETY_KEYWORDS = ["회", "육회", "생굴", "생새우", "날것", "레어", "타르타르"]

# 단품 나트륨·포화지방 주의 기준 (1일 상한의 35%)
_NA_CAUTION_PER_ITEM     = DAILY_UPPER["na"]     * 0.35   # ≈ 805 mg
_SATFAT_CAUTION_PER_ITEM = DAILY_UPPER["satfat"] * 0.35   # ≈ 7 g


# ────────────────────────────────────────────────────────────────────────────
# 2. 기능1: 영양소 충족도 분석
# ────────────────────────────────────────────────────────────────────────────

def nutrient_adequacy(daily_nutrition: dict) -> dict:
    """
    1일 합산 영양성분 → 영양소별 충족률 dict.

    권장 영양소: ratio = 실제 / 권장 (1.0 = 100% 충족, 1.0 미만 = 부족)
    상한 영양소: ratio = 실제 / 상한  (1.0 초과 = 초과 섭취)
    _type 키로 각 영양소가 '권장'인지 '상한'인지 구분 가능.
    """
    result: dict = {}
    for key, ref in DAILY_RECOMMENDED.items():
        result[key] = round(daily_nutrition.get(key, 0.0) / ref, 3) if ref else 0.0
    for key, ref in DAILY_UPPER.items():
        result[key] = round(daily_nutrition.get(key, 0.0) / ref, 3) if ref else 0.0
    result["_type"] = {k: "권장" for k in DAILY_RECOMMENDED}
    result["_type"].update({k: "상한" for k in DAILY_UPPER})
    return result


def deficient_nutrients(daily_nutrition: dict, threshold: float = 0.75) -> list[str]:
    """충족률 threshold 미만인 권장 영양소 목록 반환 (기본 75% 미만)."""
    adequacy = nutrient_adequacy(daily_nutrition)
    return [k for k in DAILY_RECOMMENDED if adequacy.get(k, 0.0) < threshold]


def excess_nutrients(daily_nutrition: dict) -> list[str]:
    """1일 상한을 초과한 영양소 목록 반환 (ratio > 1.0)."""
    adequacy = nutrient_adequacy(daily_nutrition)
    return [k for k in DAILY_UPPER if adequacy.get(k, 0.0) > 1.0]


# ────────────────────────────────────────────────────────────────────────────
# 3. 기능2: 영양밀도 점수 & 메뉴 효율 랭킹
# ────────────────────────────────────────────────────────────────────────────

def menu_nutrient_score(nutrition: dict) -> float:
    """
    단품 영양밀도 점수 (100kcal당 기준, 높을수록 효율적).

    긍정 영양소(단백질·식이섬유·미량영양소) 충족 기여도에서
    부정 영양소(나트륨·포화지방·당류) 일일 상한 30% 초과분 패널티를 차감.
    """
    kcal = max(nutrition.get("kcal", 0.0), 1.0)

    pos = sum(
        nutrition.get(k, 0.0) / DAILY_RECOMMENDED[k] * w
        for k, w in _POS_WEIGHTS.items()
        if DAILY_RECOMMENDED.get(k, 0)
    )
    penalty = (
        max(0.0, nutrition.get("na",     0.0) / DAILY_UPPER["na"]     - 0.3) * 1.5 +
        max(0.0, nutrition.get("satfat", 0.0) / DAILY_UPPER["satfat"] - 0.3) * 1.0 +
        max(0.0, nutrition.get("sugar",  0.0) / DAILY_UPPER["sugar"]  - 0.3) * 0.8
    )
    return round((pos - penalty) * 100 / kcal, 4)


def rank_menus_by_efficiency(
    menus: list[dict],
    get_nutrition_fn: Callable[[str], Optional[dict]],
) -> list[dict]:
    """
    메뉴 목록 → 영양밀도 점수 내림차순 정렬.
    영양정보 없는 메뉴는 efficiency_score=None으로 포함 (맨 뒤).
    """
    scored = []
    for m in menus:
        n = get_nutrition_fn(m["name"])
        score = menu_nutrient_score(n) if n else None
        scored.append({**m, "nutrition": n, "efficiency_score": score})

    scored.sort(key=lambda x: (x["efficiency_score"] is None, -(x["efficiency_score"] or 0)))
    return scored


# ────────────────────────────────────────────────────────────────────────────
# 4. 기능3: 주의 음식 감지
# ────────────────────────────────────────────────────────────────────────────

def flag_caution_menus(
    menus: list[dict],
    get_nutrition_fn: Callable[[str], Optional[dict]],
) -> list[dict]:
    """
    주의 사유가 하나 이상인 메뉴만 반환.
    각 항목: {메뉴정보, nutrition, cautions: [사유 문자열 목록]}

    감지 기준:
    - 나트륨 단품 805mg 초과 (1일 상한 35%)
    - 포화지방 단품 7g 초과
    - 콜레스테롤 단품 150mg 초과 (1일 상한 50%)
    - 식중독 주의 키워드 포함 (이름 기반 휴리스틱)
    """
    flagged = []
    for m in menus:
        cautions: list[str] = []
        n = get_nutrition_fn(m["name"])

        if n:
            if n.get("na", 0.0) > _NA_CAUTION_PER_ITEM:
                cautions.append(
                    f"나트륨 과다 ({n['na']:.0f}mg ＞ 단품기준 {_NA_CAUTION_PER_ITEM:.0f}mg)")
            if n.get("satfat", 0.0) > _SATFAT_CAUTION_PER_ITEM:
                cautions.append(f"포화지방 과다 ({n['satfat']:.1f}g)")
            if n.get("chol", 0.0) > DAILY_UPPER["chol"] * 0.5:
                cautions.append(f"콜레스테롤 주의 ({n['chol']:.0f}mg)")

        for kw in _FOOD_SAFETY_KEYWORDS:
            if kw in m["name"]:
                cautions.append(f"식중독 주의 키워드 포함 ({kw})")
                break

        if cautions:
            flagged.append({**m, "nutrition": n, "cautions": cautions})

    return flagged


# ────────────────────────────────────────────────────────────────────────────
# 5. 기능5: 대체식 추천
# ────────────────────────────────────────────────────────────────────────────

def suggest_alternatives(
    target: dict,
    pool: list[dict],
    get_nutrition_fn: Callable[[str], Optional[dict]],
    n: int = 3,
    kcal_tolerance: float = 0.30,
) -> list[dict]:
    """
    target 메뉴와 유사 열량대(±kcal_tolerance) pool에서
    영양밀도 점수가 높은 대체 메뉴 n개 반환.

    target과 이름이 같은 항목은 제외.
    영양정보가 없는 후보는 랭킹에서 제외.
    """
    t_n = get_nutrition_fn(target["name"])
    t_kcal = t_n.get("kcal", 0.0) if t_n else target.get("cal", 0.0)
    lo, hi = t_kcal * (1 - kcal_tolerance), t_kcal * (1 + kcal_tolerance)

    candidates = []
    for m in pool:
        if m["name"] == target["name"]:
            continue
        mn = get_nutrition_fn(m["name"])
        if not mn:
            continue
        if not (lo <= mn.get("kcal", 0.0) <= hi):
            continue
        candidates.append({**m, "nutrition": mn, "efficiency_score": menu_nutrient_score(mn)})

    candidates.sort(key=lambda x: -(x["efficiency_score"] or 0))
    return candidates[:n]


# ────────────────────────────────────────────────────────────────────────────
# 6. 1일 합산 헬퍼 (risk_engine.sum_diet_nutrition과 동일 인터페이스)
# ────────────────────────────────────────────────────────────────────────────

def sum_diet_nutrition(
    menus: list[dict],
    get_nutrition_fn: Callable[[str], Optional[dict]],
) -> dict:
    """메뉴 목록 × 영양조회함수 → 1일 영양성분 합산 dict."""
    totals: dict = {}
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
# 7. 축A 종합 일일 리포트
# ────────────────────────────────────────────────────────────────────────────

def daily_report(
    menus: list[dict],
    get_nutrition_fn: Callable[[str], Optional[dict]],
) -> dict:
    """
    메뉴 목록 → 축A 전체 분석 리포트 dict.

    반환 키:
      daily_nutrition — 합산 영양성분
      adequacy        — 영양소별 충족률 (+ _type 메타)
      deficient       — 부족 영양소 목록 (충족률 < 75%)
      excess          — 초과 영양소 목록
      ranked_menus    — 효율 순 메뉴 목록
      caution_menus   — 주의 메뉴 목록
      _coverage       — 영양정보 매칭 커버리지 문자열
    """
    daily   = sum_diet_nutrition(menus, get_nutrition_fn)
    ranked  = rank_menus_by_efficiency(menus, get_nutrition_fn)
    caution = flag_caution_menus(menus, get_nutrition_fn)

    return {
        "daily_nutrition": daily,
        "adequacy":        nutrient_adequacy(daily),
        "deficient":       deficient_nutrients(daily),
        "excess":          excess_nutrients(daily),
        "ranked_menus":    ranked,
        "caution_menus":   caution,
        "_coverage":       daily.get("_coverage", "0/0"),
    }


# ────────────────────────────────────────────────────────────────────────────
# 8. 자체 점검 (python nutrition_engine.py)
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== nutrition_engine 자체 점검 ===\n")

    DUMMY_MENUS = [
        {"date": "2025-01-31", "meal": "조식", "name": "불고기버거",  "cal": 550},
        {"date": "2025-01-31", "meal": "조식", "name": "계란후라이",  "cal": 120},
        {"date": "2025-01-31", "meal": "중식", "name": "쌀밥",       "cal": 300},
        {"date": "2025-01-31", "meal": "중식", "name": "제육볶음",   "cal": 380},
        {"date": "2025-01-31", "meal": "중식", "name": "배추김치",   "cal": 30},
        {"date": "2025-01-31", "meal": "석식", "name": "고등어구이", "cal": 260},
        {"date": "2025-01-31", "meal": "석식", "name": "된장찌개",   "cal": 90},
        {"date": "2025-01-31", "meal": "석식", "name": "쌀밥",       "cal": 300},
    ]

    _DB = {
        "불고기버거":  {"kcal": 550, "protein": 22, "fat": 28, "carb": 50, "sugar": 8,
                        "fiber": 2,   "na": 980,  "satfat": 10,  "chol": 75,
                        "ca": 80,  "fe": 2.5, "k": 250, "vita": 30,  "vitc": 2,  "vitd": 0.5},
        "계란후라이":  {"kcal": 120, "protein": 8,  "fat": 9,  "carb": 1,  "sugar": 0,
                        "fiber": 0,   "na": 180,  "satfat": 2.5, "chol": 210,
                        "ca": 30,  "fe": 1.2, "k": 90,  "vita": 80,  "vitc": 0,  "vitd": 1.2},
        "쌀밥":        {"kcal": 300, "protein": 5,  "fat": 0.5,"carb": 66, "sugar": 0,
                        "fiber": 0.5, "na": 2,    "satfat": 0.1, "chol": 0,
                        "ca": 5,   "fe": 0.3, "k": 50,  "vita": 0,   "vitc": 0,  "vitd": 0},
        "제육볶음":    {"kcal": 380, "protein": 28, "fat": 22, "carb": 12, "sugar": 5,
                        "fiber": 1,   "na": 900,  "satfat": 7.5, "chol": 90,
                        "ca": 40,  "fe": 2.8, "k": 380, "vita": 20,  "vitc": 5,  "vitd": 0.2},
        "배추김치":    {"kcal": 30,  "protein": 2,  "fat": 0.5,"carb": 5,  "sugar": 2,
                        "fiber": 2,   "na": 700,  "satfat": 0.1, "chol": 0,
                        "ca": 50,  "fe": 0.8, "k": 220, "vita": 10,  "vitc": 15, "vitd": 0},
        "고등어구이":  {"kcal": 260, "protein": 30, "fat": 14, "carb": 0,  "sugar": 0,
                        "fiber": 0,   "na": 420,  "satfat": 3.2, "chol": 80,
                        "ca": 25,  "fe": 1.5, "k": 450, "vita": 25,  "vitc": 0,  "vitd": 8.0},
        "된장찌개":    {"kcal": 90,  "protein": 6,  "fat": 3,  "carb": 8,  "sugar": 1,
                        "fiber": 1.5, "na": 1100, "satfat": 0.5, "chol": 5,
                        "ca": 60,  "fe": 1.2, "k": 300, "vita": 5,   "vitc": 2,  "vitd": 0},
    }

    def _get_nutrition(name):
        return _DB.get(name)

    # 1. 충족도
    print("[1] 영양소 충족도")
    daily = sum_diet_nutrition(DUMMY_MENUS, _get_nutrition)
    adequacy = nutrient_adequacy(daily)
    for k, v in adequacy.items():
        if k.startswith("_"):
            continue
        typ = adequacy["_type"].get(k, "")
        bar = "#" * min(20, int(v * 10))
        warn = " !" if (typ == "권장" and v < 0.75) or (typ == "상한" and v > 1.0) else ""
        print(f"   {k:<8} {typ:<2}  {v:.3f}  {bar}{warn}")

    print(f"\n[2] 부족: {deficient_nutrients(daily)}")
    print(f"    과잉: {excess_nutrients(daily)}")

    print("\n[3] 영양밀도 효율 랭킹 (상위 5)")
    for i, m in enumerate(rank_menus_by_efficiency(DUMMY_MENUS, _get_nutrition)[:5], 1):
        sc = m["efficiency_score"]
        print(f"   {i}. {m['name']:<10} score={sc:.4f}" if sc is not None else f"   {i}. {m['name']} 정보없음")

    print("\n[4] 주의 메뉴")
    cautions = flag_caution_menus(DUMMY_MENUS, _get_nutrition)
    if cautions:
        for m in cautions:
            print(f"   ! {m['name']}")
            for c in m["cautions"]:
                print(f"      - {c}")
    else:
        print("   없음")

    print("\n[5] '불고기버거' 대체식 추천")
    alts = suggest_alternatives(DUMMY_MENUS[0], DUMMY_MENUS[1:], _get_nutrition)
    if alts:
        for a in alts:
            print(f"   -> {a['name']:<10} score={a['efficiency_score']:.4f}")
    else:
        print("   후보 없음")

    print("\n[6] daily_report 반환 키")
    report = daily_report(DUMMY_MENUS, _get_nutrition)
    for k in report:
        print(f"   {k}")

    print("\n완료.")
