"""
inject_data.py
==============
data/*.csv 를 읽어 dashboard.html 의 @@INJECT_START@@ ~ @@INJECT_END@@ 블록에
실제 데이터를 주입합니다.

사용법:
    python execution/inject_data.py
    (또는 프로젝트 루트에서)  python -m execution.inject_data

주의: pandas 없이 내장 csv/json 모듈만 사용 (OpenBLAS 메모리 오류 방지)
"""

import csv
import json
import os
import re
from datetime import datetime, timedelta

# ── 경로 설정 ─────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR       = os.path.join(BASE_DIR, "data")
DASHBOARD_HTML = os.path.join(BASE_DIR, "dashboard.html")
MODEL_PERF_JSON = os.path.join(BASE_DIR, "model_performance.json")
LGBM_DATED_JSON = os.path.join(BASE_DIR, "lgbm_forecast_dated.json")
GLOBAL_CLEAN_DATA = os.path.join(BASE_DIR, "농넷_과거파일", "final_clean_dataset.csv")

# ── 작물 설정 ──────────────────────────────────────────────────────────────
CROPS = {
    'strawberry': {'id': 0, 'csv': 'strawberry_prices.csv', 'weight': 1,  'box_unit': '1kg상자'},
    'cucumber':   {'id': 1, 'csv': 'cucumber_prices.csv',   'weight': 10, 'box_unit': '10kg상자'},
    'tomato':     {'id': 2, 'csv': 'tomato_prices.csv',     'weight': 5,  'box_unit': '5kg상자'},
    'paprika':    {'id': 3, 'csv': 'paprika_prices.csv',    'weight': 5,  'box_unit': '5kg상자'},
}

HISTORY_DAYS = 730   # 히스토리컬 데이터 기간 (2년치)
FORECAST_DAYS = 180  # 예측 기간
MAE_BASE = 150      # 신뢰 구간 기본 오차 (원/kg, fallback용)
STEP_INC = 20       # 스텝별 오차 증가량 (원/kg, fallback용)


# ── 유틸 ───────────────────────────────────────────────────────────────────
def read_csv(path):
    """CSV 파일을 읽어 딕셔너리 리스트로 반환"""
    rows = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def indent_json(obj, base_indent=4):
    """JSON 직렬화 후 모든 줄에 base_indent 공백 추가 (JS 인라인 포매팅용)"""
    s = json.dumps(obj, ensure_ascii=False, indent=2)
    pad = ' ' * base_indent
    return ('\n' + pad).join(s.split('\n'))


# ── 메인 ───────────────────────────────────────────────────────────────────
def generate():
    print("=" * 60)
    print("inject_data.py - Dashboard Data Injection")
    print("=" * 60)

    # ── lgbm_forecast_dated.json 로드 (실제 AI 예측) ──────────────────────
    lgbm_dated = None
    if os.path.exists(LGBM_DATED_JSON):
        with open(LGBM_DATED_JSON, encoding='utf-8') as f:
            lgbm_dated = json.load(f)
        print(f"[OK] lgbm_forecast_dated.json loaded (generated: {lgbm_dated.get('generated_at', 'unknown')})")
    else:
        print(f"[WARN] lgbm_forecast_dated.json not found — using trend-based fallback forecast")
        print(f"       Run: python execution/train_lgbm_dated.py")

    historical   = {}
    forecast_data = {}
    test_results  = {}   # backtesting: 날짜별 실제 vs AI 예측
    all_last_dates = []

    # ── 히스토리컬 데이터 로드 (통합 파일 사용) ───────────────────────────
    all_rows = []
    if os.path.exists(GLOBAL_CLEAN_DATA):
        all_rows = read_csv(GLOBAL_CLEAN_DATA)
        print(f"[OK] {len(all_rows)} records loaded from global dataset.")
    else:
        print(f"[ERROR] Global dataset not found: {GLOBAL_CLEAN_DATA}")
        return

    for name, cfg in CROPS.items():
        # 1. 통합 데이터셋에서 과거 데이터 추출 및 날짜별 평균 계산
        crop_id = str(cfg['id'])
        crop_rows = [r for r in all_rows if str(r.get('item')) == crop_id]
        
        daily_map = {} # date -> {price_sum, count, volume_sum}
        for r in crop_rows:
            d = r.get('date', r.get('DATE'))
            if not d: continue
            try:
                p_val = float(r.get('price_per_kg', r.get('avg_price', 0)))
                v_val = float(r.get('volume', 0))
                if d not in daily_map: daily_map[d] = {'p_sum': 0, 'count': 0, 'v_sum': 0}
                daily_map[d]['p_sum'] += p_val * v_val  # volume-weighted sum
                daily_map[d]['v_sum'] += v_val
            except (ValueError, TypeError): continue
        
        # 2. 개별 CSV(최신 데이터) 로드 및 병합
        csv_path = os.path.join(DATA_DIR, cfg['csv'])
        if os.path.exists(csv_path):
            latest_rows = read_csv(csv_path)
            for r in latest_rows:
                d = r.get('DATE', r.get('date'))
                if not d: continue
                try:
                    p_val = float(r.get('price_per_kg', r.get('avg_price', 0)))
                    v_val = float(r.get('volume', 0))
                    # 최신 파일 데이터로 덮어쓰거나(보통 이쪽이 더 정확) 합침. 여기서는 덮어씀.
                    daily_map[d] = {'p_sum': p_val, 'count': 1, 'v_sum': v_val}
                except: continue

        # 3. 정렬 및 최근 HISTORY_DAYS일 추출
        sorted_dates = sorted(daily_map.keys())
        recent_dates = sorted_dates[-HISTORY_DAYS:]
        
        hist_list = []
        w = cfg['weight']
        box_unit = cfg['box_unit']
        
        for d in recent_dates:
            data = daily_map[d]
            avg_p = data['p_sum'] / max(data['v_sum'], 1)  # volume-weighted average
            hist_list.append({
                "date":      d,
                "price":     int(round(avg_p)),
                "avg_price": int(round(avg_p * w)),
                "unit":      "원/kg",
                "box_unit":  box_unit,
                "volume":    int(data['v_sum']),
            })
        
        historical[name] = hist_list
        if not hist_list:
            print(f"[SKIP] No data generated for {name}")
            continue

        # ── 마지막 날짜 & 가격 (예측용) ───────────────────────────────
        last_date_str = hist_list[-1]['date']
        all_last_dates.append(last_date_str)
        last_price = hist_list[-1]['price']
        last_dt = datetime.strptime(last_date_str, '%Y-%m-%d')

        # ── 추세 계산 (최근 3일 데이터) ───────────────────────────────
        prices3 = [h['price'] for h in hist_list[-3:]]

        if len(prices3) >= 2:
            raw_trend = (prices3[-1] - prices3[0]) / max(len(prices3) - 1, 1)
        else:
            raw_trend = 0

        # 일일 변동 폭을 현재가의 ±1.5%로 제한 (과도한 발산 방지)
        max_daily = last_price * 0.015
        daily_trend = max(-max_daily, min(max_daily, raw_trend * 0.4))

        # ── 180일 예측: lgbm_forecast_dated.json 우선, fallback은 추세 ──────
        if lgbm_dated and name in lgbm_dated.get('forecasts', {}):
            # 실제 AI 모델 예측 사용
            fc_list = lgbm_dated['forecasts'][name]
            # 반입량 데이터가 누락된 경우 보완 (최근 7일 평균 사용)
            vols7 = [h.get('volume', 0) for h in hist_list[-7:]]
            avg_vol = int(sum(vols7) / len(vols7)) if vols7 else 0
            for f in fc_list:
                if 'volume' not in f:
                    f['volume'] = avg_vol
            print(f"         → AI forecast: {len(fc_list)}일 (from lgbm_forecast_dated.json)")
        else:
            # Fallback: 추세 기반 결정론적 예측
            fc_list = []
            curr_price = last_price
            
            # 최근 7일 평균 반입량을 예측 반입량으로 사용
            vols7 = [h.get('volume', 0) for h in hist_list[-7:]]
            avg_vol = int(sum(vols7) / len(vols7)) if vols7 else 0
            
            for i in range(1, FORECAST_DAYS + 1):
                curr_dt   = last_dt + timedelta(days=i)
                damping = max(0.3, 1.0 - i * 0.05)
                pred = int(curr_price + daily_trend * damping)
                pred = max(pred, 100)
                hi   = int(pred + MAE_BASE + i * STEP_INC)
                lo   = max(int(pred - MAE_BASE - i * STEP_INC), int(pred * 0.5))
                fc_list.append({
                    "date":      curr_dt.strftime('%Y-%m-%d'),
                    "price":     pred,
                    "hi":        hi,
                    "lo":        lo,
                    "unit":      "원/kg",
                    "avg_price": int(pred * w),
                    "box_unit":  box_unit,
                    "volume":    avg_vol, # 반입량 추가
                })
                curr_price = pred
            print(f"         → Trend-based forecast (fallback): {len(fc_list)}일")

        forecast_data[name] = {"forecast": fc_list}

        # ── Backtesting 결과: 날짜별 실제 vs AI 예측 ──────────────────────
        if lgbm_dated and name in lgbm_dated.get('test_results', {}):
            test_results[name] = lgbm_dated['test_results'][name]
            print(f"         → Test results: {len(test_results[name])}일 (backtesting)")

        print(f"[OK] {name:12s} | history {len(hist_list):3d}d | last: {last_date_str} "
              f"| base: {last_price:,} won/kg | trend: {daily_trend:+.1f} won/d")

    if not all_last_dates:
        print("Error: No CSV files could be read. Check the data/ folder.")
        return

    # TODAY = 가장 최근 데이터 날짜
    today_str = max(all_last_dates)
    print(f"\n[TODAY] {today_str}")

    # ── 모델 성능 로드 ────────────────────────────────────────────────────
    model_perf = None
    if os.path.exists(MODEL_PERF_JSON):
        with open(MODEL_PERF_JSON, encoding='utf-8') as f:
            model_perf = json.load(f)
        print(f"[OK] model_performance.json loaded")
    else:
        print(f"[WARN] model_performance.json not found — injecting without model perf")

    # ── 주입 블록 생성 ────────────────────────────────────────────────────
    pad = '    '  # JS 4-space indent
    inject_lines = [
        f"{pad}// @@INJECT_START@@",
        f"{pad}// 자동 생성 — inject_data.py ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
        f"{pad}const INJECTED_TODAY        = {json.dumps(today_str)};",
        f"{pad}const INJECTED_HISTORICAL   = {pad}{indent_json(historical, base_indent=len(pad))};",
        f"{pad}const INJECTED_FORECAST     = {pad}{indent_json(forecast_data, base_indent=len(pad))};",
        f"{pad}const INJECTED_MODEL_PERF   = {pad}{indent_json(model_perf, base_indent=len(pad))};",
        f"{pad}// test_results: 날짜별 실제vs예측 (AI backtesting, null이면 미사용)",
        f"{pad}const INJECTED_TEST_RESULTS = {pad}{indent_json(test_results if test_results else None, base_indent=len(pad))};",
        f"{pad}// @@INJECT_END@@",
    ]
    inject_block = '\n'.join(inject_lines)

    # ── dashboard.html 업데이트 ───────────────────────────────────────────
    if not os.path.exists(DASHBOARD_HTML):
        print(f"Error: {DASHBOARD_HTML} not found.")
        return

    with open(DASHBOARD_HTML, encoding='utf-8') as f:
        content = f.read()

    pattern = r'[ \t]*// @@INJECT_START@@.*?// @@INJECT_END@@'
    new_content, n_subs = re.subn(pattern, inject_block, content, flags=re.DOTALL)

    if n_subs == 0:
        print("Error: @@INJECT_START@@ / @@INJECT_END@@ markers not found in dashboard.html.")
        return

    with open(DASHBOARD_HTML, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"\n[SUCCESS] dashboard.html updated!")
    print(f"  TODAY   = {today_str}")
    print(f"  crops   = {list(historical.keys())}")
    print(f"  blocks  = {n_subs} replaced")
    print(f"\nRefresh dashboard.html in browser (F5).")


if __name__ == "__main__":
    generate()
