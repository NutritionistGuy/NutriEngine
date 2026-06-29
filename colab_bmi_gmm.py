# ============================================================
# 병무청 신검 BMI 분포 GMM 클러스터링 — Colab 실행 스크립트
# ============================================================
# 사용법:
#   1. Google Colab에서 새 노트북 생성
#   2. 아래 섹션을 각각 셀에 붙여넣고 순서대로 실행
#   3. 마지막 셀에서 gmm_params.json 다운로드
#   4. 프로젝트의 data/gmm_params.json 에 저장
#
# 환경변수: MMA_KEY (없으면 더미 데이터로 실행됨)
# ============================================================

# %% [markdown]
# ## 0. 환경 설치

# %%
# !pip install scikit-learn scipy matplotlib seaborn requests -q

import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

matplotlib.rcParams["font.family"] = "NanumGothic"   # Colab 한글 폰트
plt.rcParams["axes.unicode_minus"] = False

print("환경 준비 완료")


# %% [markdown]
# ## 1. 병무청 B.2 BMI 데이터 수집
# 실제 API 키가 있으면 실제 데이터를, 없으면 더미 데이터를 사용합니다.

# %%
MMA_KEY = os.environ.get("MMA_KEY", "")   # 본인 키 입력 or 환경변수 설정
TARGET_YEAR = 2026

def fetch_bmi_from_api(api_key: str, year: int) -> list[float]:
    """병무청 API에서 BMI 데이터 수집."""
    import requests

    # api_client.py 확인 엔드포인트: jBGSSCJeongBo2/getlist
    # serviceKey 인코딩 우회: params 대신 URL 직접 조립
    base = "https://apis.data.go.kr/1300000/jBGSSCJeongBo2/getlist"
    url = f"{base}?serviceKey={api_key}&numOfRows=1000&pageNo=1&type=json"
    try:
        r = requests.get(url, timeout=15)
        print(f"  HTTP {r.status_code}  응답 길이: {len(r.text)} 바이트")
        print(f"  응답 앞부분: {r.text[:300]}")
        r.raise_for_status()

        # 이 API는 type=json 파라미터를 무시하고 XML을 반환함 → XML 파싱
        import xml.etree.ElementTree as ET
        root = ET.fromstring(r.text)
        rows = [{c.tag: c.text for c in it} for it in root.iter("item")]
        print(f"  파싱된 행 수: {len(rows)}")

        if year:
            rows = [row for row in rows if str(row.get("geomsaDt", "")).startswith(str(year))]
        print(f"  {year}년 필터 후: {len(rows)}행")

        bmis = []
        for row in rows:
            try:
                b = float(str(row.get("bmi", "")).replace(",", ""))
                if 10 < b < 50:
                    bmis.append(b)
            except (ValueError, TypeError):
                pass
        return bmis
    except Exception as e:
        print(f"API 호출 실패: {e}")
        return []

def dummy_bmi_data(n: int = 500, seed: int = 42) -> list[float]:
    """API 키 없을 때 사용할 더미 BMI 데이터."""
    random.seed(seed)
    result = []
    while len(result) < n:
        b = round(random.gauss(23.0, 3.0), 1)
        if 18.5 <= b <= 35:     # B.2 공개 구간
            result.append(b)
    return result

if MMA_KEY:
    bmi_list = fetch_bmi_from_api(MMA_KEY, TARGET_YEAR)
    print(f"실제 API: {len(bmi_list)}개 수집")
    if len(bmi_list) == 0:
        print("⚠ API에서 데이터를 받지 못했습니다. 위 응답 내용을 확인하세요.")
        print("  → 더미 데이터로 대체합니다 (GMM 구조 확인용)")
        bmi_list = dummy_bmi_data(500)
        print(f"  더미 데이터: {len(bmi_list)}개")
else:
    bmi_list = dummy_bmi_data(500)
    print(f"더미 데이터: {len(bmi_list)}개 (MMA_KEY 없음)")

if not bmi_list:
    raise RuntimeError("BMI 데이터가 없습니다. 위 오류 메시지를 확인하세요.")

print(f"BMI 범위: {min(bmi_list):.1f} ~ {max(bmi_list):.1f}")
print(f"평균: {np.mean(bmi_list):.2f}  표준편차: {np.std(bmi_list):.2f}")


# %% [markdown]
# ## 2. 탐색적 데이터 분석 (EDA)

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 히스토그램 + KDE
ax = axes[0]
ax.hist(bmi_list, bins=30, density=True, alpha=0.5, color="steelblue", label="실측 분포")
x_range = np.linspace(min(bmi_list) - 2, max(bmi_list) + 2, 300)
kde_vals = sum(norm.pdf(x_range, b, 0.5) for b in bmi_list) / len(bmi_list)
ax.plot(x_range, kde_vals, "k-", lw=1.5, label="KDE")
# BMI 군 경계선
for bmi_cut, label in [(18.5, "저체중"), (23, "정상"), (25, "과체중"), (30, "비만")]:
    ax.axvline(bmi_cut, color="red", ls="--", lw=0.8, alpha=0.6)
    ax.text(bmi_cut + 0.1, ax.get_ylim()[1] * 0.9, label, fontsize=8, color="red")
ax.set_title(f"BMI 분포 (n={len(bmi_list)})")
ax.set_xlabel("BMI")
ax.set_ylabel("밀도")
ax.legend()

# 군별 빈도
groups = {"저체중": (0, 18.5), "정상": (18.5, 23), "과체중": (23, 25),
          "비만": (25, 30), "고도비만": (30, 99)}
counts = {g: sum(lo <= b < hi for b in bmi_list) for g, (lo, hi) in groups.items()}
ax2 = axes[1]
ax2.bar(counts.keys(), [v / len(bmi_list) * 100 for v in counts.values()],
        color=["skyblue", "green", "gold", "orange", "red"], alpha=0.8)
ax2.set_title("BMI 군별 단순 빈도 비율 (%)")
ax2.set_ylabel("%")

plt.tight_layout()
plt.savefig("eda_bmi.png", dpi=150, bbox_inches="tight")
plt.show()
print("EDA 완료")


# %% [markdown]
# ## 3. BIC로 최적 컴포넌트 수 선택

# %%
X = np.array(bmi_list).reshape(-1, 1)
max_k = 6

bic_scores = {}
aic_scores = {}

for k in range(1, max_k + 1):
    gmm = GaussianMixture(n_components=k, covariance_type="full",
                          random_state=42, n_init=10)
    gmm.fit(X)
    bic_scores[k] = round(gmm.bic(X), 2)
    aic_scores[k] = round(gmm.aic(X), 2)
    print(f"  k={k}  BIC={bic_scores[k]:.1f}  AIC={aic_scores[k]:.1f}")

optimal_k = min(bic_scores, key=bic_scores.get)
print(f"\n최적 컴포넌트 수 (BIC 최소): k={optimal_k}")

plt.figure(figsize=(8, 4))
plt.plot(list(bic_scores.keys()), list(bic_scores.values()), "bo-", label="BIC")
plt.plot(list(aic_scores.keys()), list(aic_scores.values()), "rs-", label="AIC")
plt.axvline(optimal_k, color="green", ls="--", label=f"최적 k={optimal_k}")
plt.xlabel("컴포넌트 수 k")
plt.ylabel("정보기준 점수 (낮을수록 좋음)")
plt.title("BIC / AIC로 최적 k 선택")
plt.legend()
plt.tight_layout()
plt.savefig("bic_selection.png", dpi=150, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## 4. 최적 k로 GMM 피팅

# %%
final_gmm = GaussianMixture(n_components=optimal_k, covariance_type="full",
                             random_state=42, n_init=20)
final_gmm.fit(X)

# 컴포넌트를 평균 오름차순 정렬
order   = final_gmm.means_.flatten().argsort()
means   = final_gmm.means_.flatten()[order]
stds    = np.array([final_gmm.covariances_[i][0][0] ** 0.5 for i in order])
weights = final_gmm.weights_[order]

print("=== 최종 GMM 파라미터 ===")
for i, (m, s, w) in enumerate(zip(means, stds, weights)):
    print(f"  컴포넌트 {i+1}: 평균 BMI={m:.2f}  표준편차={s:.2f}  비중={w:.1%}")

# 피팅 결과 시각화
x_plot = np.linspace(14, 40, 400)
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(bmi_list, bins=30, density=True, alpha=0.4, color="steelblue", label="실측")

colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6"]
total_pdf = np.zeros_like(x_plot)
for i, (m, s, w) in enumerate(zip(means, stds, weights)):
    component_pdf = w * norm.pdf(x_plot, m, s)
    total_pdf += component_pdf
    ax.plot(x_plot, component_pdf, "--", color=colors[i % len(colors)],
            lw=1.5, alpha=0.8, label=f"컴포넌트 {i+1} (μ={m:.1f}, w={w:.1%})")

ax.plot(x_plot, total_pdf, "k-", lw=2.5, label="GMM 전체 밀도")

for bmi_cut in [18.5, 23, 25, 30]:
    ax.axvline(bmi_cut, color="gray", ls=":", lw=1)

ax.set_title(f"GMM 피팅 결과 (k={optimal_k}, n={len(bmi_list)})")
ax.set_xlabel("BMI")
ax.set_ylabel("밀도")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig("gmm_fit.png", dpi=150, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## 5. GMM → BMI 건강군별 확률 계산

# %%
bmi_group_ranges = {
    "저체중":   (0.0,  18.5),
    "정상":     (18.5, 23.0),
    "과체중":   (23.0, 25.0),
    "비만":     (25.0, 30.0),
    "고도비만": (30.0, float("inf")),
}

gmm_ratios = {}
for group, (lo, hi) in bmi_group_ranges.items():
    prob = 0.0
    for m, s, w in zip(means, stds, weights):
        hi_p = 1.0 if hi == float("inf") else norm.cdf(hi, m, s)
        lo_p = norm.cdf(lo, m, s)
        prob += w * (hi_p - lo_p)
    gmm_ratios[group] = prob

# 정규화
total = sum(gmm_ratios.values())
gmm_ratios = {k: round(v / total, 4) for k, v in gmm_ratios.items()}

# 단순 빈도 비율과 비교
freq_ratios = {g: sum(lo <= b < hi for b in bmi_list) / len(bmi_list)
               for g, (lo, hi) in bmi_group_ranges.items()}

print("=== 군별 비율 비교 ===")
print(f"{'군':<8}  {'단순 빈도':>8}  {'GMM 추정':>8}  {'차이':>8}")
for g in bmi_group_ranges:
    diff = gmm_ratios[g] - freq_ratios[g]
    print(f"{g:<8}  {freq_ratios[g]:>7.1%}  {gmm_ratios[g]:>7.1%}  {diff:>+7.1%}")

# 비교 차트
x = np.arange(len(gmm_ratios))
w_bar = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - w_bar/2, [freq_ratios[g] * 100 for g in bmi_group_ranges],
       w_bar, label="단순 빈도 집계", color="steelblue", alpha=0.8)
ax.bar(x + w_bar/2, [gmm_ratios[g] * 100 for g in bmi_group_ranges],
       w_bar, label="GMM 추정 (AI)", color="tomato", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(list(bmi_group_ranges.keys()))
ax.set_ylabel("%")
ax.set_title("단순 빈도 집계 vs GMM 추정 군별 비율")
ax.legend()
plt.tight_layout()
plt.savefig("gmm_vs_freq.png", dpi=150, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## 6. 파라미터 저장 (앱에 반영)

# %%
gmm_params = {
    "n_components": int(optimal_k),
    "means":        [round(float(m), 4) for m in means],
    "stds":         [round(float(s), 4) for s in stds],
    "weights":      [round(float(w), 4) for w in weights],
    "bic_scores":   {int(k): float(v) for k, v in bic_scores.items()},
    "bic_optimal":  round(float(bic_scores[optimal_k]), 2),
    "n_samples":    len(bmi_list),
    "gmm_ratios":   gmm_ratios,
    "source_year":  TARGET_YEAR,
    "data_type":    "real_api" if MMA_KEY else "dummy",
}

with open("gmm_params.json", "w", encoding="utf-8") as f:
    json.dump(gmm_params, f, ensure_ascii=False, indent=2)

print("gmm_params.json 저장 완료")
print(json.dumps(gmm_params, ensure_ascii=False, indent=2))

# Colab에서 다운로드
try:
    from google.colab import files
    files.download("gmm_params.json")
    print("다운로드 창이 열립니다 → 프로젝트의 data/ 폴더에 저장하세요")
except ImportError:
    print("로컬 실행: gmm_params.json을 data/ 폴더에 복사하세요")
