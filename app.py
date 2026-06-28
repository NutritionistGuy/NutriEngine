"""
app.py
======
Streamlit 통합 데모 — api_client × nutrition_engine × risk_engine.

실행: streamlit run app.py

환경변수 (없으면 해당 API 자동 더미):
  MMA_KEY   병무청 병역판정 신체검사 (B.2)
  FOOD_KEY  식약처 식품영양성분DB    (C.1)
  RAW_KEY   해수부 원재료 영양성분   (C.2)

식단 파일 (로컬): data/army_diet.json  (D.2)
"""

import datetime as dt

import pandas as pd
import streamlit as st

from api_client import get_mma_bmi_distribution, get_menu, get_nutrition, is_dummy
from nutrition_engine import (
    DAILY_RECOMMENDED,
    DAILY_UPPER,
    daily_report,
    flag_caution_menus,
    rank_menus_by_efficiency,
    suggest_alternatives,
)
from risk_engine import (
    ACTIVITY_LEVELS,
    BMI_GROUP_ORDER,
    get_bmi_ratio_from_dist,
    run_simulation,
    sum_diet_nutrition,
)

st.set_page_config(
    page_title="병무청 신검 × 식단 의사결정 시스템",
    layout="wide",
)

# ────────────────────────────────────────────────────────────────────────────
# Streamlit 캐시 래퍼
# (api_client 내부 SQLite 캐시 위에 Streamlit 세션 캐시를 추가해 rerun 비용 최소화)
# ────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3_600, show_spinner=False)
def _menu(date_str: str):
    return get_menu(date_str)

@st.cache_data(ttl=86_400, show_spinner=False)
def _dist(year: int):
    return get_mma_bmi_distribution(year)

@st.cache_data(ttl=86_400, show_spinner=False)
def _nutrition(name: str):
    return get_nutrition(name)


# ────────────────────────────────────────────────────────────────────────────
# 사이드바
# ────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("설정")

    # API 연결 상태
    st.subheader("API 상태")
    for api_id, label in [("mma", "병무청 B.2"), ("food", "식약처 C.1"), ("raw", "해수부 C.2")]:
        icon = "🟡" if is_dummy(api_id) else "🟢"
        st.caption(f"{icon} {label}: {'더미' if is_dummy(api_id) else '실제 API'}")

    st.divider()

    sel_date = st.date_input("식단 날짜", value=dt.date.today())
    sel_year = st.selectbox("병무청 기준 연도", list(range(2024, 2019, -1)), index=0)

    st.divider()

    st.subheader("시뮬레이션 설정")
    _activity_labels = {
        "sedentary": "좌식", "light": "가벼운", "moderate": "보통",
        "active": "훈련소 (기본)", "very_active": "고강도",
    }
    sel_activity = st.selectbox(
        "활동 수준",
        list(ACTIVITY_LEVELS.keys()),
        index=list(ACTIVITY_LEVELS.keys()).index("active"),
        format_func=lambda x: _activity_labels[x],
    )
    sel_days = st.slider("시뮬 기간 (일)", 7, 90, 30)

    st.divider()

    st.subheader("What-if 식단 조정")
    wi_na   = st.slider("나트륨 감소 %",    0, 50, 0, step=5)
    wi_kcal = st.slider("열량 감소 %",      0, 30, 0, step=5)
    wi_fat  = st.slider("포화지방 감소 %",  0, 50, 0, step=5)
    what_if = {k: v for k, v in [
        ("sodium_reduction_pct",  wi_na),
        ("calorie_reduction_pct", wi_kcal),
        ("satfat_reduction_pct",  wi_fat),
    ] if v > 0}


# ────────────────────────────────────────────────────────────────────────────
# 데이터 로드 (실제 API / 로컬 JSON / 더미 폴백)
# ────────────────────────────────────────────────────────────────────────────

date_str = sel_date.isoformat()

with st.spinner("데이터 로딩 중..."):
    menus  = _menu(date_str)
    dist   = _dist(sel_year)
    report = daily_report(menus, _nutrition) if menus else {}
    daily  = report.get("daily_nutrition", {})

st.title("병무청 신검 × 식단 의사결정 시스템")
st.caption(
    f"식단: {date_str}  |  병무청: {sel_year}년  |  시뮬: {sel_days}일  |  "
    f"활동: {_activity_labels[sel_activity]}"
)

if not menus:
    st.warning("해당 날짜 식단 정보가 없습니다. `data/army_diet.json`을 확인하거나 다른 날짜를 선택하세요.")
    st.stop()


tab_a, tab_b = st.tabs(["축A — 식단 영양 판단", "축B — 건강위험 예측 (B.2 신검 기반)"])


# ════════════════════════════════════════════════════════════════════════════
# 축A 탭 — 식단 자체 영양 판단
# ════════════════════════════════════════════════════════════════════════════

with tab_a:

    # 1. 오늘 식단 목록
    st.subheader("오늘 식단")
    df_menu = pd.DataFrame(menus)[["meal", "name", "cal"]]
    df_menu.columns = ["구분", "메뉴명", "열량(kcal)"]
    st.dataframe(df_menu, use_container_width=True, hide_index=True)
    st.caption(f"영양정보 커버리지: {report.get('_coverage', '-')}")

    st.divider()

    # 2. 부족 / 과잉 요약
    deficient = report.get("deficient", [])
    excess    = report.get("excess", [])

    col_d, col_e = st.columns(2)
    with col_d:
        if deficient:
            st.error(f"**부족 영양소 ({len(deficient)}개):** " + ", ".join(deficient))
        else:
            st.success("부족 영양소 없음")
    with col_e:
        if excess:
            st.warning(f"**과잉 영양소 ({len(excess)}개):** " + ", ".join(excess))
        else:
            st.success("과잉 영양소 없음")

    st.divider()

    # 3. 영양소 충족도 차트
    st.subheader("영양소 충족도")
    adequacy = report.get("adequacy", {})
    _type    = adequacy.get("_type", {})

    chart_rows = []
    for k in list(DAILY_RECOMMENDED) + list(DAILY_UPPER):
        v   = adequacy.get(k, 0.0)
        typ = _type.get(k, "")
        ref = DAILY_RECOMMENDED.get(k) or DAILY_UPPER.get(k, 1)
        chart_rows.append({
            "영양소": f"{k} ({typ})",
            "충족률": round(v, 3),
        })

    df_adeq = pd.DataFrame(chart_rows).set_index("영양소")
    st.bar_chart(df_adeq["충족률"], use_container_width=True, height=320)
    st.caption("권장 영양소: 1.0 이상 = 충족.  상한 영양소: 1.0 초과 = 초과 섭취.")

    st.divider()

    # 4. 주의 메뉴
    st.subheader("주의 메뉴")
    caution_menus = report.get("caution_menus", [])
    if caution_menus:
        for m in caution_menus:
            with st.expander(f"주의  {m['name']}  ({m['meal']})"):
                for c in m["cautions"]:
                    st.write(f"- {c}")
    else:
        st.success("주의 메뉴 없음")

    st.divider()

    # 5. 영양밀도 랭킹
    st.subheader("메뉴 영양밀도 랭킹 (100kcal 기준)")
    ranked = [m for m in report.get("ranked_menus", []) if m.get("efficiency_score") is not None]
    if ranked:
        df_ranked = pd.DataFrame([
            {
                "순위": i + 1,
                "구분": m["meal"],
                "메뉴명": m["name"],
                "영양밀도점수": m["efficiency_score"],
                "열량(kcal)": m["nutrition"].get("kcal", 0) if m.get("nutrition") else "-",
                "단백질(g)":  m["nutrition"].get("protein", 0) if m.get("nutrition") else "-",
                "나트륨(mg)": m["nutrition"].get("na", 0) if m.get("nutrition") else "-",
            }
            for i, m in enumerate(ranked)
        ])
        st.dataframe(df_ranked, use_container_width=True, hide_index=True)
    else:
        st.info("영양정보가 있는 메뉴가 없습니다.")

    st.divider()

    # 6. 대체식 추천 (주의 메뉴가 있을 때)
    if caution_menus:
        st.subheader("대체식 추천")
        target = caution_menus[0]
        pool   = [m for m in menus if m["name"] != target["name"]]
        alts   = suggest_alternatives(target, pool, _nutrition)
        if alts:
            st.write(f"**'{target['name']}'** 대신 영양밀도가 높은 메뉴:")
            for a in alts:
                st.write(f"- {a['name']}  (점수 {a['efficiency_score']:.3f})")
        else:
            st.info("같은 열량대 대체 메뉴가 없습니다.")


# ════════════════════════════════════════════════════════════════════════════
# 축B 탭 — 건강위험 예측 (B.2 신검 × 식단)
# ════════════════════════════════════════════════════════════════════════════

with tab_b:

    st.caption(
        "B.2 공개 BMI 18.5~35 구간만 포함 — 저체중·고도비만 비율 과소계상.  "
        "신검은 입대 전 건강 상태(수요)이며 식단→신검 인과관계 아님."
    )

    # 1. 병무청 BMI 분포 현황
    st.subheader(f"병무청 {sel_year}년 신검 BMI 분포")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("샘플 수",    f"{dist.get('count', 0):,}명")
    c2.metric("평균 BMI",   f"{dist.get('mean', 0):.2f}")
    c3.metric("중앙값 BMI", f"{dist.get('p50', 0):.2f}")
    c4.metric("비만 비율",  f"{dist.get('obese_ratio', 0):.1%}")

    ratio_by_group = get_bmi_ratio_from_dist(dist)
    df_ratio = pd.DataFrame({
        "비율 (%)": {g: round(ratio_by_group.get(g, 0) * 100, 2) for g in BMI_GROUP_ORDER}
    })
    st.bar_chart(df_ratio, use_container_width=True, height=240)

    st.divider()

    # 2. 시뮬레이션 실행
    if not daily or daily.get("kcal", 0) == 0:
        st.warning("식단 영양정보가 없어 시뮬레이션을 실행할 수 없습니다.")
        st.stop()

    sim_base = run_simulation(dist, daily, days=sel_days, activity=sel_activity)
    sim_wi   = (run_simulation(dist, daily, days=sel_days,
                               what_if=what_if, activity=sel_activity)
                if what_if else None)

    # 3. 군별 위험 점수 테이블
    st.subheader(f"BMI군별 위험 점수 ({sel_days}일 후)")
    rows_risk = []
    for g in BMI_GROUP_ORDER:
        gr = sim_base["group_results"][g]
        r  = gr["risk"]
        rows_risk.append({
            "BMI군":            g,
            "대표 BMI":         gr["rep_bmi"],
            "TDEE (kcal)":     int(gr["tdee"]),
            "에너지 과부족":    f"{gr['energy_delta_kcal']:+.0f}",
            f"{sel_days}일 Δ체중 (kg)": f"{gr['weight_change_kg']:+.3f}",
            "군 이동":          f"→ {gr['new_group']}" if gr["group_changed"] else "-",
            "고혈압":           r["hypertension"],
            "대사위험":         r["metabolic"],
            "저체중":           r["underweight"],
            "종합":             r["overall"],
        })
    df_risk = pd.DataFrame(rows_risk)
    st.dataframe(df_risk, use_container_width=True, hide_index=True)

    # 종합 위험 점수 막대차트
    df_overall = pd.DataFrame({
        "종합 위험 점수": {g: sim_base["group_results"][g]["risk"]["overall"] for g in BMI_GROUP_ORDER}
    })
    st.bar_chart(df_overall, use_container_width=True, height=220)

    st.divider()

    # 4. 분포 이동 (현재 식단 지속 시)
    st.subheader(f"BMI군 분포 이동 ({sel_days}일 후, 현재 식단 지속)")
    shift = sim_base["distribution_shift"]
    df_shift = pd.DataFrame({
        "현재 분포 (%)":       {g: round(shift["original"].get(g, 0) * 100, 1)  for g in BMI_GROUP_ORDER},
        f"{sel_days}일 후 (%)": {g: round(shift["after_days"].get(g, 0) * 100, 1) for g in BMI_GROUP_ORDER},
    })
    st.bar_chart(df_shift, use_container_width=True, height=260)

    obese_chg = sim_base["summary"]["obese_ratio_change_ppt"]
    if obese_chg > 0:
        st.error(f"비만군 비율 변화: **+{obese_chg:.2f}%p** — 현 식단 지속 시 악화")
    elif obese_chg < 0:
        st.success(f"비만군 비율 변화: **{obese_chg:.2f}%p** — 개선")
    else:
        st.info("비만군 비율 변화 없음")

    # 5. What-if 비교 (슬라이더 조정 시)
    if what_if and sim_wi:
        st.divider()
        st.subheader("What-if 비교 — 식단 조정 효과")

        shift_wi = sim_wi["distribution_shift"]
        df_wi = pd.DataFrame({
            "현재 식단 후 (%)": {g: round(shift["after_days"].get(g, 0) * 100, 1)    for g in BMI_GROUP_ORDER},
            "조정 식단 후 (%)": {g: round(shift_wi["after_days"].get(g, 0) * 100, 1) for g in BMI_GROUP_ORDER},
        })
        st.bar_chart(df_wi, use_container_width=True, height=260)

        obese_wi = sim_wi["summary"]["obese_ratio_change_ppt"]
        ca, cb, cc = st.columns(3)
        ca.metric("현재 식단 비만군 변화",  f"{obese_chg:+.2f}%p")
        cb.metric("조정 식단 비만군 변화",  f"{obese_wi:+.2f}%p",
                  delta=f"{obese_wi - obese_chg:+.2f}%p")
        adj_parts = []
        if wi_na:   adj_parts.append(f"나트륨 -{wi_na}%")
        if wi_kcal: adj_parts.append(f"열량 -{wi_kcal}%")
        if wi_fat:  adj_parts.append(f"포화지방 -{wi_fat}%")
        cc.metric("적용된 조정", "  /  ".join(adj_parts))

        # 군별 위험 점수 비교 (현재 vs 조정)
        st.write("**군별 종합 위험 점수 비교**")
        df_risk_cmp = pd.DataFrame({
            "현재 식단": {g: sim_base["group_results"][g]["risk"]["overall"] for g in BMI_GROUP_ORDER},
            "조정 식단": {g: sim_wi["group_results"][g]["risk"]["overall"]   for g in BMI_GROUP_ORDER},
        })
        st.bar_chart(df_risk_cmp, use_container_width=True, height=240)

    st.divider()

    # 6. 데이터 한계 경고
    with st.expander("데이터 한계 및 해석 주의사항"):
        for w in sim_base["_warnings"]:
            st.caption(f"  {w}")
