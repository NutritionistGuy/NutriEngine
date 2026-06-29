"""
api_client.py  (v2 — 키 분리 / 식단 로컬화 반영)
================================================
병무청 신검 · 식약처 영양DB · 해수부 원재료 = 3개 REST API
국방부 식단(D.2) = 로컬 JSON 파일 (API 아님, 직접 다운로드)

핵심 변경 (v1 -> v2)
-------------------
- API 키를 3개로 분리: MMA_KEY / FOOD_KEY / RAW_KEY (각각 다른 키)
- 키가 일부만 있어도 그 API만 실제, 나머지는 자동 더미 -> 개발 안 멈춤
- 식단은 call_api가 아니라 로컬 data/army_diet.json 에서 읽음

환경변수 (없으면 해당 API만 더미)
  export MMA_KEY="병무청_발급키"
  export FOOD_KEY="식약처_발급키"
  export RAW_KEY="해수부_발급키"
식단 파일: diet_health/data/army_diet.json (없으면 더미 식단)
"""

from __future__ import annotations
import os, re, json, time, sqlite3, difflib, datetime as dt
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET
import requests

_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_ENV_PATH):
    with open(_ENV_PATH, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if "=" in _line and not _line.startswith("#"):
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# 0. 설정 -------------------------------------------------------------------
SERVICE_KEYS = {
    "mma":  os.environ.get("MMA_KEY", "").strip(),
    "food": os.environ.get("FOOD_KEY", "").strip(),
    "raw":  os.environ.get("RAW_KEY", "").strip(),
}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MENU_JSON_PATH = os.path.join(BASE_DIR, "data", "army_diet.json")
CACHE_DB = os.path.join(BASE_DIR, "cache", "diet_health.sqlite")
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
os.makedirs(os.path.dirname(CACHE_DB), exist_ok=True)

TIMEOUT, MAX_RETRY, RETRY_BACKOFF = 8, 3, 1.5
ENDPOINTS = {
    # 병무청 병역판정검사 신체정보 (data.go.kr 데이터셋 3064321, GET /getlist)
    "mma":  "https://apis.data.go.kr/1300000/jBGSSCJeongBo2/getlist",
    "food": "http://apis.data.go.kr/1471000/FoodNtrCpntDbInfo02/getFoodNtrCpntDbInq02",
    "raw":  "http://apis.data.go.kr/1192000/select/getNutrient",
}

def is_dummy(name: str) -> bool:
    return SERVICE_KEYS.get(name, "") == ""

# 1. 캐시 -------------------------------------------------------------------
def _cache_conn():
    conn = sqlite3.connect(CACHE_DB)
    conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT, ts REAL)")
    return conn

def cache_get(key, max_age_sec=None):
    with _cache_conn() as conn:
        row = conn.execute("SELECT v, ts FROM kv WHERE k=?", (key,)).fetchone()
    if not row: return None
    v, ts = row
    if max_age_sec is not None and (time.time() - ts) > max_age_sec: return None
    return json.loads(v)

def cache_set(key, value):
    with _cache_conn() as conn:
        conn.execute("INSERT OR REPLACE INTO kv (k, v, ts) VALUES (?,?,?)",
                     (key, json.dumps(value, ensure_ascii=False), time.time()))

# 2. 저수준 호출기 ----------------------------------------------------------
def _request(name, params):
    url = ENDPOINTS[name]
    params = {**params, "serviceKey": SERVICE_KEYS[name]}
    last_err = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRY: time.sleep(RETRY_BACKOFF ** attempt)
    print(f"[api_client] 요청 실패({name}): {last_err}")
    return None

def _parse(text):
    if not text: return []
    text = text.strip()
    if text[:1] in "{[":
        try:
            data = json.loads(text)
            # 형식 C (GW): {"currentCount": N, "data": [...]}  (api.odcloud.kr)
            if "data" in data and isinstance(data["data"], list):
                return data["data"]
            # 형식 A (표준): {"response": {"body": {"items": {"item": [...]}}}}
            items = data.get("response", {}).get("body", {}).get("items", {})
            # 형식 B (식약처): {"header": ..., "body": {"items": [...]}}
            if not items:
                items = data.get("body", {}).get("items", {})
            if isinstance(items, list): return items
            if isinstance(items, dict):
                item = items.get("item", [])
                if isinstance(item, dict): item = [item]
                return item or []
            return []
        except Exception: pass
    try:
        root = ET.fromstring(text)
        return [{c.tag: c.text for c in it} for it in root.iter("item")]
    except Exception:
        return []

def call_api(name, params):
    if is_dummy(name): return _dummy(name, params)
    base = {"pageNo": 1, "numOfRows": 100, "type": "json"}
    return _parse(_request(name, {**base, **params}))

# 3. 고수준 진입점 ----------------------------------------------------------
def get_menu(date):
    """[D.2] 로컬 JSON에서 식단 읽어 메뉴 1개당 1건으로 펼침. (API 아님)"""
    ck = f"menu:{date}"
    cached = cache_get(ck)
    if cached is not None: return cached
    rows = _load_menu_json()
    target = date.replace("-", "")
    # 날짜 형식 "2025-01-31(금)" → "-" 제거 후 앞 8자리만 비교
    rows = [r for r in rows
            if isinstance(r, dict) and
            re.sub(r"[^0-9]", "", str(r.get("dates", "")))[:8] == target]
    if not rows: rows = _dummy("menu", {"dates": target})
    menus = _normalize_menu_rows(rows, date)
    cache_set(ck, menus)
    return menus

def _load_menu_json():
    if not os.path.exists(MENU_JSON_PATH): return []
    try:
        with open(MENU_JSON_PATH, encoding="utf-8") as f:
            data = json.load(f)
        # 키가 "DATA"(대문자) 또는 "data" 모두 지원
        if isinstance(data, dict):
            return data.get("DATA", data.get("data", []))
        return data
    except Exception as e:
        print(f"[api_client] 식단 JSON 로드 실패: {e}")
        return []

def get_nutrition(food_name):
    """[C.1/C.2] 식품명 -> 영양성분. 캐시->식약처->해수부 폴백."""
    ck = f"nutri:{food_name}"
    cached = cache_get(ck)
    if cached is not None: return cached
    rows = call_api("food", {"FOOD_NM_KR": food_name, "numOfRows": 30})
    if not rows: rows = call_api("raw", {"foodNm": food_name, "numOfRows": 30})
    row = _best_nutrition_row(food_name, rows) if rows else None
    result = _normalize_nutrition_row(row) if row else None
    cache_set(ck, result)
    return result

def get_mma_bmi_distribution(year, jbceong=None):
    """[B.2] BMI 분포 통계. year로 geomsaDt 클라이언트 필터링."""
    ck = f"mma:{year}:{jbceong or 'ALL'}"
    cached = cache_get(ck)
    if cached is not None: return cached
    # API 파라미터: numOfRows·pageNo·serviceKey만 지원 (연도 서버 필터 없음)
    rows = call_api("mma", {"numOfRows": 1000})
    # 클라이언트 필터: 수검년도·지방청
    if year:
        rows = [r for r in rows if str(r.get("geomsaDt", "")).startswith(str(year))]
    if jbceong:
        rows = [r for r in rows if r.get("jbceong") == jbceong]
    stats = _summarize_bmi(rows)
    cache_set(ck, stats)
    return stats

# 4. 정규화 -----------------------------------------------------------------
def _f(v):
    try:
        s = re.sub(r"[^\d.\-]", "", str(v).replace(",", ""))  # "560.16kcal" → "560.16"
        return float(s) if s else 0.0
    except (TypeError, ValueError): return 0.0

NUTRI_FIELD_MAP = {
    "kcal": ["AMT_NUM1","enerc","에너지(kcal)"], "protein": ["AMT_NUM3","prot","단백질(g)"],
    "fat": ["AMT_NUM4","fatce","지방(g)"], "carb": ["AMT_NUM6","chocdf","탄수화물(g)"],
    "sugar": ["AMT_NUM7","sugar","당류(g)"], "fiber": ["AMT_NUM8","fibtg","식이섬유(g)"],
    "ca": ["AMT_NUM9","ca","칼슘(mg)"], "fe": ["AMT_NUM10","fe","철(mg)"],
    "k": ["AMT_NUM12","k","칼륨(mg)"], "na": ["AMT_NUM13","nat","나트륨(mg)"],
    "vita": ["AMT_NUM14","vitaRae","비타민 A(μg RAE)"], "vitc": ["AMT_NUM21","vitc","비타민 C(mg)"],
    "vitd": ["AMT_NUM22","vitd","비타민 D(μg)"], "chol": ["AMT_NUM23","chole","콜레스테롤(mg)"],
    "satfat": ["AMT_NUM24","fasat","포화지방산(g)"],
}

def _best_nutrition_row(food_name: str, rows: list) -> dict:
    """여러 행 중 food_name에 가장 가까운 이름의 행을 반환."""
    if not rows: return None
    candidates = {r.get("FOOD_NM_KR") or r.get("foodNm") or r.get("식품명", ""): r
                  for r in rows}
    candidates = {k: v for k, v in candidates.items() if k}
    if not candidates: return rows[0]
    hit = best_match(food_name, list(candidates.keys()))
    return candidates[hit] if hit else rows[0]

def _normalize_nutrition_row(row):
    out = {}
    for std_key, cands in NUTRI_FIELD_MAP.items():
        val = 0.0
        for c in cands:
            if c in row and row[c] not in (None, ""):
                val = _f(row[c]); break
        out[std_key] = val
    out["_name"] = row.get("FOOD_NM_KR") or row.get("foodNm") or row.get("식품명") or ""
    return out

def _normalize_menu_rows(rows, date):
    out = []
    meals = [("조식","brst","brst_cal"),("중식","lunc","lunc_cal"),
             ("석식","dinr","dinr_cal"),("중특식","adspcfd","adspcfd_cal")]
    for row in rows:
        for meal_kr, fname, cname in meals:
            text = row.get(fname) or ""
            if not text: continue
            cal_total = _f(row.get(cname))
            menus = _split_menu_text(text)
            per = round(cal_total / len(menus), 1) if menus else 0.0
            for m in menus:
                out.append({"date": date, "meal": meal_kr, "name": m, "cal": per})
    return out

def _split_menu_text(text):
    text = re.sub(r"\([0-9]+\)", "", text)
    text = re.sub(r"[0-9]+\.", "", text)
    return [p.strip() for p in re.split(r"[,/·\n]", text) if p.strip()]

def _summarize_bmi(rows):
    bmis = [_f(r.get("bmi")) for r in rows if _f(r.get("bmi")) > 0]
    if not bmis: return {"count":0,"mean":0,"p50":0,"p90":0,"obese_ratio":0}
    bmis.sort(); n = len(bmis)
    pct = lambda p: bmis[min(n-1, int(n*p))]
    return {"count": n, "mean": round(sum(bmis)/n, 2),
            "p50": round(pct(0.5), 2), "p90": round(pct(0.9), 2),
            "obese_ratio": round(sum(1 for b in bmis if b >= 25)/n, 3),
            "note": "공개 BMI는 18.5~35 구간만 포함 -> 실제 분포보다 절단됨"}

# 5. 메뉴명 매칭 ------------------------------------------------------------
def normalize_name(name):
    name = re.sub(r"\(.*?\)", "", name)
    return re.sub(r"[^가-힣A-Za-z0-9]", "", name).strip()

def best_match(menu_name, candidates):
    target = normalize_name(menu_name)
    if not target or not candidates: return None
    norm = {normalize_name(c): c for c in candidates}
    try:
        from rapidfuzz import process, fuzz
        hit = process.extractOne(target, list(norm.keys()), scorer=fuzz.WRatio)
        if hit and hit[1] >= 60: return norm[hit[0]]
    except ImportError:
        m = difflib.get_close_matches(target, list(norm.keys()), n=1, cutoff=0.6)
        if m: return norm[m[0]]
    return None

# 6. 더미 -------------------------------------------------------------------
def _dummy(name, params):
    if name == "menu":
        return [{"dates": params.get("dates",""),
            "brst":"불고기버거(01)(02), 계란후라이, 시리얼, 백색우유","brst_cal":"650",
            "lunc":"쌀밥, 미역국, 제육볶음, 콩나물무침, 배추김치","lunc_cal":"820",
            "dinr":"쌀밥, 된장찌개, 고등어구이, 시금치나물, 깍두기","dinr_cal":"780",
            "adspcfd":"","adspcfd_cal":"0"}]
    if name in ("food","raw"):
        fn = params.get("FOOD_NM_KR") or params.get("foodNm") or "샘플식품"
        s = sum(ord(c) for c in fn)
        return [{"FOOD_NM_KR":fn,"AMT_NUM1":150+s%200,"AMT_NUM3":6+s%20,"AMT_NUM4":4+s%15,
            "AMT_NUM6":18+s%30,"AMT_NUM7":s%12,"AMT_NUM8":1+s%5,"AMT_NUM9":20+s%120,
            "AMT_NUM10":s%6,"AMT_NUM12":100+s%300,"AMT_NUM13":200+s%700,"AMT_NUM14":s%300,
            "AMT_NUM21":s%40,"AMT_NUM22":s%5,"AMT_NUM23":s%80,"AMT_NUM24":s%6}]
    if name == "mma":
        import random
        random.seed(int(params.get("geomsaDt", 2024)))
        rows = []
        for _ in range(500):
            bmi = round(random.gauss(23.0, 3.0), 1)
            if 18.5 <= bmi <= 35:
                rows.append({"bmi":bmi,"height":random.randint(160,185),
                             "weight":random.randint(55,95),"geomsaDt":params.get("geomsaDt")})
        return rows
    return []

# 7. 자체 점검 --------------------------------------------------------------
if __name__ == "__main__":
    print("=== api_client v2 자체 점검 ===")
    for k in ("mma","food","raw"):
        print(f"  {k}: {'더미' if is_dummy(k) else '실제 API'}")
    print(f"  menu: 로컬파일 {'있음' if os.path.exists(MENU_JSON_PATH) else '없음(더미)'}\n")
    today = dt.date.today().isoformat()
    menus = get_menu(today)
    print(f"[1] 식단 {len(menus)}건 (예시 3):")
    for m in menus[:3]: print("   ", m)
    print("\n[2] 메뉴별 영양:")
    for m in menus[:3]:
        n = get_nutrition(m["name"])
        print(f"    {m['name']:<12} kcal={n['kcal'] if n else '?'} 나트륨={n['na'] if n else '?'}mg")
    cur_year = dt.date.today().year
    print(f"\n[3] 병무청 BMI 분포({cur_year}):"); print("   ", get_mma_bmi_distribution(cur_year))
    print("\n[4] 매칭:", best_match("불고기버거", ["불고기 버거","치즈버거","계란말이"]))
    print("\n완료.")
