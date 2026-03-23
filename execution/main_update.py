import csv
import json
import os
import sys
import subprocess
import argparse
from datetime import datetime, timedelta

# ── 경로 설정
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXEC_DIR        = os.path.join(BASE_DIR, "execution")
DATA_DIR        = os.path.join(BASE_DIR, "data")
OUTPUT_JSON     = os.path.join(DATA_DIR, "prices.json")
MODEL_PERF_JSON = os.path.join(BASE_DIR, "model_performance.json")
LGBM_DATED_JSON = os.path.join(BASE_DIR, "lgbm_forecast_dated.json")
GLOBAL_CLEAN_DATA = os.path.join(BASE_DIR, "농넷_과거파일", "final_clean_dataset.csv")

CROPS = {
    'strawberry': {'id': 0, 'csv': 'strawberry_prices.csv', 'weight': 1,  'box_unit': '1kg상자'},
    'cucumber':   {'id': 1, 'csv': 'cucumber_prices.csv',   'weight': 10, 'box_unit': '10kg상자'},
    'tomato':     {'id': 2, 'csv': 'tomato_prices.csv',     'weight': 5,  'box_unit': '5kg상자'},
    'paprika':    {'id': 3, 'csv': 'paprika_prices.csv',    'weight': 5,  'box_unit': '5kg상자'},
}

HISTORY_DAYS  = 730
FORECAST_DAYS = 90

def read_csv(path):
    rows = []
    if not os.path.exists(path): return []
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader: rows.append(row)
    return rows

def run_script(script_name, desc):
    print(f"\n[STEP] {desc}...")
    script_path = os.path.join(EXEC_DIR, script_name)
    result = subprocess.run([sys.executable, script_path], cwd=BASE_DIR)
    return result.returncode == 0

def generate_prices_json():
    print("\n[STEP] Finalizing prices.json with latest data...")
    
    lgbm_dated = None
    if os.path.exists(LGBM_DATED_JSON):
        with open(LGBM_DATED_JSON, encoding='utf-8') as f:
            lgbm_dated = json.load(f)

    # global 과거 데이터 로드 (시드용)
    all_global = read_csv(GLOBAL_CLEAN_DATA)
    
    historical = {}
    forecast_data = {}
    test_results = {}
    all_last_dates = []

    for name, cfg in CROPS.items():
        daily_map = {}
        # 1. 시드 데이터
        crop_id = str(cfg['id'])
        for r in all_global:
            if str(r.get('item')) == crop_id:
                d = r.get('date', r.get('DATE'))
                if d: daily_map[d] = {'price': float(r.get('price_per_kg', 0)), 'volume': float(r.get('volume', 0))}
        
        # 2. 크롤링 최신 데이터
        csv_path = os.path.join(DATA_DIR, cfg['csv'])
        for r in read_csv(csv_path):
            d = r.get('DATE', r.get('date'))
            if d: daily_map[d] = {'price': float(r.get('price_per_kg', 0)), 'volume': float(r.get('volume', 0))}
            
        # 3. 정렬 및 리스트화
        sorted_dates = sorted(daily_map.keys())
        hist_list = []
        for d in sorted_dates[-HISTORY_DAYS:]:
            p = int(round(daily_map[d]['price']))
            hist_list.append({
                "date": d,
                "price": p,
                "avg_price": int(p * cfg['weight']),
                "unit": "원/kg",
                "box_unit": cfg['box_unit'],
                "volume": int(daily_map[d]['volume'])
            })
        
        historical[name] = hist_list
        last_date_str = hist_list[-1]['date']
        all_last_dates.append(last_date_str)
        
        # 4. AI 예측 통합
        fc_list = []
        if lgbm_dated and name in lgbm_dated.get('forecasts', {}):
            ai_fc = lgbm_dated['forecasts'][name]
            # 기준일 이후의 예측만 필터링
            fc_list = [f for f in ai_fc if f['date'] > last_date_str]
            
        # 예측 데이터가 없거나 기준일과 맞지 않는 경우를 위한 Fallback
        if not fc_list:
            last_p = hist_list[-1]['price']
            last_dt = datetime.strptime(last_date_str, '%Y-%m-%d')
            for i in range(1, FORECAST_DAYS + 1):
                dt = (last_dt + timedelta(days=i)).strftime('%Y-%m-%d')
                fc_list.append({
                    "date": dt, "price": last_p, "hi": int(last_p*1.15), "lo": int(last_p*0.85),
                    "unit": "원/kg", "box_unit": cfg['box_unit'], "avg_price": int(last_p * cfg['weight']), "volume": 0
                })
        
        forecast_data[name] = {"forecast": fc_list}
        if lgbm_dated and name in lgbm_dated.get('test_results', {}):
            test_results[name] = lgbm_dated['test_results'][name]

    # 최종 메타데이터
    today_now = datetime.now().strftime('%Y-%m-%d %H:%M')
    data_today = max(all_last_dates) if all_last_dates else datetime.now().strftime('%Y-%m-%d')
    
    model_perf = None
    if os.path.exists(MODEL_PERF_JSON):
        with open(MODEL_PERF_JSON, encoding='utf-8') as f:
            model_perf = json.load(f)
        if model_perf: model_perf['last_updated'] = today_now

    output = {
        "last_updated": today_now,
        "today": data_today,
        "historical": historical,
        "forecast": forecast_data,
        "model_performance": model_perf,
        "test_results": test_results
    }
    
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"[SUCCESS] prices.json ready. Today is {data_today}")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-only', action='store_true')
    parser.add_argument('--skip-train', action='store_true')
    args = parser.parse_args()

    if not args.json_only:
        run_script('nongnet_crawler.py', '크롤링 업데이트')
        if not args.skip_train:
            run_script('train_lgbm_dated.py', '모델 학습')
            
    if generate_prices_json():
        run_script('inject_data.py', '대시보드 반영')
        print("\n[V] 업데이트가 완료되었습니다.")

if __name__ == "__main__":
    main()
