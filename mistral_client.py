"""
mistral_client.py
==================
Mistral AI 연동 — 대체식 추천 자연어 답변 생성.

환경변수:
  MISTRAL_KEY  Mistral API 키 (없으면 AI 기능 비활성화, 규칙 기반 폴백)
"""

from __future__ import annotations
import os
from typing import Optional

_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_ENV_PATH):
    with open(_ENV_PATH, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if "=" in _line and not _line.startswith("#"):
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

_MISTRAL_KEY = os.environ.get("MISTRAL_KEY", "").strip()

_NUTRIENT_KR = {
    "kcal": "열량", "protein": "단백질", "fat": "지방", "carb": "탄수화물",
    "fiber": "식이섬유", "ca": "칼슘", "fe": "철분", "k": "칼륨",
    "vita": "비타민A", "vitc": "비타민C", "vitd": "비타민D",
    "na": "나트륨", "sugar": "당류", "satfat": "포화지방", "chol": "콜레스테롤",
}


def is_mistral_available() -> bool:
    """MISTRAL_KEY가 설정되어 있고 mistralai 패키지가 설치되어 있으면 True."""
    if not _MISTRAL_KEY:
        return False
    try:
        from mistralai.client.sdk import Mistral  # noqa: F401
        return True
    except ImportError:
        return False


def get_alternative_recommendation(
    target_name: str,
    cautions: list[str],
    alternatives: list[dict],
    deficient: list[str],
    excess: list[str],
    daily_nutrition: dict,
    model: str = "mistral-small-latest",
) -> Optional[str]:
    """
    Mistral AI를 사용해 대체식 추천 자연어 답변 생성.

    Parameters
    ----------
    target_name   : 주의 메뉴 이름
    cautions      : 주의 사유 목록
    alternatives  : suggest_alternatives() 반환값 (영양 정보 포함)
    deficient     : 오늘 식단 부족 영양소 키 목록
    excess        : 오늘 식단 과잉 영양소 키 목록
    daily_nutrition: 오늘 1일 합산 영양성분
    model         : 사용할 Mistral 모델 ID

    Returns
    -------
    str  — AI 생성 추천 텍스트, 실패 시 None
    """
    if not is_mistral_available():
        return None

    try:
        from mistralai.client.sdk import Mistral
    except ImportError:
        return None

    def _fmt(keys: list[str]) -> str:
        return ", ".join(_NUTRIENT_KR.get(k, k) for k in keys) if keys else "없음"

    alts_lines = ""
    for i, a in enumerate(alternatives, 1):
        n = a.get("nutrition") or {}
        score = a.get("efficiency_score")
        score_str = f"{score:.3f}" if score is not None else "N/A"
        alts_lines += (
            f"  {i}. {a['name']} (영양밀도 점수: {score_str})\n"
            f"     열량 {n.get('kcal', 0):.0f}kcal · "
            f"단백질 {n.get('protein', 0):.1f}g · "
            f"나트륨 {n.get('na', 0):.0f}mg · "
            f"포화지방 {n.get('satfat', 0):.1f}g\n"
        )

    prompt = f"""당신은 대한민국 육군 훈련소 식단을 분석하는 영양 전문가입니다.
훈련병의 건강과 체력 유지를 위해 아래 정보를 바탕으로 대체식 추천 이유를 친절하고 전문적으로 설명해주세요.

【주의 메뉴】: {target_name}
【주의 사유】: {', '.join(cautions) if cautions else '없음'}

【오늘 식단 현황】
  - 부족 영양소: {_fmt(deficient)}
  - 과잉 영양소: {_fmt(excess)}

【추천 대체 메뉴 (영양밀도 높은 순)】:
{alts_lines.strip() if alts_lines else '  같은 열량대 대체 메뉴 없음'}

아래 세 가지를 포함해 한국어로 4~6문장으로 간결하게 답변해주세요:
1. '{target_name}'의 영양학적 문제점 (주의 사유 근거)
2. 추천 대체 메뉴가 더 나은 이유 (구체적 영양 수치 포함)
3. 훈련소 생활에서 실질적으로 도움이 되는 짧은 조언 1가지"""

    try:
        client = Mistral(api_key=_MISTRAL_KEY)
        response = client.chat.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Mistral AI 오류: {e}]"
