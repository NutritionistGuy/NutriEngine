"""
bmi_model.py
============
B.2 신검 BMI 데이터 → GMM(가우시안 혼합 모델) 피팅 모듈.

역할
  - 실측 BMI 분포를 GMM으로 학습 → 각 건강 군별 확률(비율)을 통계적으로 추정
  - BIC 기준으로 최적 컴포넌트 수 자동 선택
  - 피팅 결과를 JSON으로 저장/로드 → Colab 학습 후 앱에 반영하는 브릿지
  - risk_engine.run_simulation()이 그대로 사용할 수 있도록 dist에 ratio 키 주입

왜 GMM인가
  - 단순 빈도 집계(히스토그램)는 샘플 노이즈에 취약하고 절단 구간 처리 불가
  - GMM은 관측된 BMI 분포를 연속 확률밀도로 모델링 → CDF 적분으로 군별 비율 계산
  - BIC 자동 선택 = '적절한 복잡도를 AI가 판단' → 공모전 AI 기술 활용 포인트

주요 인터페이스
  fit_gmm(bmi_list, max_k)          → params dict
  gmm_group_ratios(params)           → {군이름: 비율} dict
  enrich_dist_with_gmm(dist, params) → dist에 ratio_XX 키 추가 (risk_engine 호환)
  save_params(params, path)
  load_params(path)                  → params dict | None
  cluster_summary(params)            → 사람이 읽을 수 있는 요약 문자열
"""

from __future__ import annotations
import json
import os
from typing import Optional

# BMI 건강 군 (risk_engine.BMI_GROUPS와 동일 기준 유지)
_BMI_GROUPS = {
    "저체중":   (0.0,  18.5),
    "정상":     (18.5, 23.0),
    "과체중":   (23.0, 25.0),
    "비만":     (25.0, 30.0),
    "고도비만": (30.0, float("inf")),
}
_GROUP_ORDER = ["저체중", "정상", "과체중", "비만", "고도비만"]

DEFAULT_PARAMS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "gmm_params.json"
)


# ────────────────────────────────────────────────────────────────────────────
# 1. GMM 피팅
# ────────────────────────────────────────────────────────────────────────────

def fit_gmm(bmi_list: list[float], max_k: int = 5) -> dict:
    """
    BMI 목록 → BIC 최소 기준 최적 GMM 피팅.

    Parameters
    ----------
    bmi_list : 실측 BMI 값 목록 (api_client로 받은 rows에서 추출)
    max_k    : 탐색할 최대 컴포넌트 수 (기본 5)

    Returns
    -------
    dict:
      n_components  — 최적 컴포넌트 수
      means         — 각 컴포넌트 평균 BMI 목록
      stds          — 각 컴포넌트 표준편차 목록
      weights       — 각 컴포넌트 혼합 가중치 목록
      bic_scores    — 탐색한 k별 BIC 점수
      n_samples     — 학습에 사용된 샘플 수
    """
    try:
        import numpy as np
        from sklearn.mixture import GaussianMixture
    except ImportError as e:
        raise ImportError(f"pip install scikit-learn numpy 필요: {e}") from e

    if len(bmi_list) < 10:
        raise ValueError(f"BMI 샘플 수가 너무 적습니다: {len(bmi_list)}개 (최소 10개)")

    X = np.array(bmi_list, dtype=float).reshape(-1, 1)

    bic_scores: dict[int, float] = {}
    best_k, best_bic, best_gmm = 1, float("inf"), None

    for k in range(1, min(max_k, len(bmi_list)) + 1):
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                              random_state=42, n_init=5)
        gmm.fit(X)
        bic = gmm.bic(X)
        bic_scores[k] = round(bic, 2)
        if bic < best_bic:
            best_bic, best_k, best_gmm = bic, k, gmm

    # 컴포넌트를 평균 오름차순으로 정렬
    order = best_gmm.means_.flatten().argsort()
    means   = best_gmm.means_.flatten()[order].tolist()
    # covariance_type='full' → covariances_ shape: (k, 1, 1)
    stds    = [float(best_gmm.covariances_[i][0][0] ** 0.5) for i in order]
    weights = best_gmm.weights_[order].tolist()

    return {
        "n_components": best_k,
        "means":        [round(m, 4) for m in means],
        "stds":         [round(s, 4) for s in stds],
        "weights":      [round(w, 4) for w in weights],
        "bic_scores":   bic_scores,
        "n_samples":    len(bmi_list),
        "bic_optimal":  round(best_bic, 2),
    }


# ────────────────────────────────────────────────────────────────────────────
# 2. GMM → 군별 비율 계산
# ────────────────────────────────────────────────────────────────────────────

def gmm_group_ratios(params: dict, bmi_groups: Optional[dict] = None) -> dict[str, float]:
    """
    GMM 파라미터 → 각 BMI 건강군에 속할 확률 (군별 비율).

    각 Gaussian 컴포넌트의 CDF를 군 경계에서 적분해 군별 확률을 구하고,
    가중 합산(∑ weight_i × P(lo ≤ X_i ≤ hi))으로 최종 비율을 계산.
    """
    try:
        from scipy.stats import norm
    except ImportError as e:
        raise ImportError(f"pip install scipy 필요: {e}") from e

    if bmi_groups is None:
        bmi_groups = _BMI_GROUPS

    means   = params["means"]
    stds    = params["stds"]
    weights = params["weights"]

    ratios: dict[str, float] = {}
    for group, (lo, hi) in bmi_groups.items():
        prob = 0.0
        for mean, std, w in zip(means, stds, weights):
            hi_p = 1.0 if hi == float("inf") else norm.cdf(hi, mean, std)
            lo_p = norm.cdf(lo, mean, std)
            prob += w * (hi_p - lo_p)
        ratios[group] = prob

    # 부동소수점 오차 보정 후 정규화
    total = sum(ratios.values())
    if total > 0:
        ratios = {k: round(v / total, 4) for k, v in ratios.items()}

    return {g: ratios.get(g, 0.0) for g in _GROUP_ORDER}


# ────────────────────────────────────────────────────────────────────────────
# 3. risk_engine 호환 브릿지
# ────────────────────────────────────────────────────────────────────────────

def enrich_dist_with_gmm(dist: dict, params: dict) -> dict:
    """
    get_mma_bmi_distribution() 반환 dist에 GMM 기반 ratio_XX 키를 주입.

    risk_engine.get_bmi_ratio_from_dist()는 dist에 ratio_저체중 등 키가 있으면
    바로 사용하므로, 이 함수 하나로 risk_engine 수정 없이 GMM을 연동할 수 있음.
    """
    ratios = gmm_group_ratios(params)
    enriched = dict(dist)
    for group, ratio in ratios.items():
        enriched[f"ratio_{group}"] = ratio
    enriched["_gmm_n_components"] = params["n_components"]
    enriched["_gmm_n_samples"]    = params["n_samples"]
    return enriched


# ────────────────────────────────────────────────────────────────────────────
# 4. 파라미터 저장 / 로드
# ────────────────────────────────────────────────────────────────────────────

def save_params(params: dict, path: str = DEFAULT_PARAMS_PATH) -> None:
    """GMM 파라미터를 JSON으로 저장."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    print(f"[bmi_model] 파라미터 저장: {path}")


def load_params(path: str = DEFAULT_PARAMS_PATH) -> Optional[dict]:
    """저장된 GMM 파라미터 로드. 파일 없으면 None 반환."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[bmi_model] 파라미터 로드 실패: {e}")
        return None


# ────────────────────────────────────────────────────────────────────────────
# 5. 요약 출력
# ────────────────────────────────────────────────────────────────────────────

def cluster_summary(params: dict) -> str:
    """GMM 파라미터 → 사람이 읽을 수 있는 요약 문자열."""
    lines = [
        f"GMM 컴포넌트 수: {params['n_components']} (BIC 최적)",
        f"학습 샘플 수: {params['n_samples']}명",
        f"최적 BIC: {params.get('bic_optimal', '-')}",
        "",
        "컴포넌트별 특성:",
    ]
    for i, (mean, std, w) in enumerate(
        zip(params["means"], params["stds"], params["weights"])
    ):
        lines.append(f"  {i+1}. 평균 BMI={mean:.2f}  표준편차={std:.2f}  비중={w:.1%}")

    lines += ["", "건강군별 추정 비율 (GMM CDF):"]
    try:
        ratios = gmm_group_ratios(params)
        for g, r in ratios.items():
            bar = "#" * int(r * 40)
            lines.append(f"  {g:<6}  {r:.1%}  {bar}")
    except ImportError:
        lines.append("  (scipy 없음 — 비율 계산 불가)")

    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────
# 6. 자체 점검 (python bmi_model.py)
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== bmi_model 자체 점검 ===\n")

    # 더미 BMI 데이터 (api_client 더미 모드와 동일한 분포)
    import random
    random.seed(42)
    dummy_bmis = [
        round(random.gauss(23.0, 3.0), 1)
        for _ in range(500)
        if 18.5 <= (b := round(random.gauss(23.0, 3.0), 1)) <= 35
        for _ in [None]  # walrus trick
    ]
    # 위 방식이 파이썬 버전에 따라 문제될 수 있으니 단순하게
    dummy_bmis = []
    random.seed(42)
    while len(dummy_bmis) < 500:
        b = round(random.gauss(23.0, 3.0), 1)
        if 18.5 <= b <= 35:
            dummy_bmis.append(b)

    print(f"[1] 샘플 {len(dummy_bmis)}개로 GMM 피팅")
    params = fit_gmm(dummy_bmis, max_k=5)
    print(f"    최적 컴포넌트 수: {params['n_components']}")
    print(f"    BIC 점수: {params['bic_scores']}")

    print("\n[2] 요약")
    print(cluster_summary(params))

    print("\n[3] dist 주입 테스트")
    fake_dist = {"count": 500, "mean": 23.0, "obese_ratio": 0.15}
    enriched  = enrich_dist_with_gmm(fake_dist, params)
    for k, v in enriched.items():
        if k.startswith("ratio_") or k.startswith("_gmm"):
            print(f"    {k}: {v}")

    print("\n[4] 파라미터 저장/로드")
    save_params(params)
    loaded = load_params()
    print(f"    로드 성공: {loaded is not None}")

    print("\n완료.")