"""
main_update.py
==============
전체 파이프라인 오케스트레이터:
  1. nongnet_crawler.py  → data/*.csv 업데이트
  2. update_dataset.py   → final_clean_dataset.csv 병합 (선택)
  3. train_lgbm_dated.py → lgbm_forecast_dated.json 업데이트 (선택)
  4. data/prices.json    → 대시보드용 JSON 생성

사용법:
  python execution/main_update.py                # 전체 파이프라인
  python execution/main_update.py --json-only    # JSON 생성만 (빠름)
  python execution/main_update.py --skip-train   # 크롤링+JSON (모델 학습 건너뜀)

GitHub Actions 환경에서는 --skip-train 권장
(final_clean_dataset.csv가 없으면 모델 학습 불가)
"""

import csv
import json
import os
import sys
import subprocess
import argparse
from datetime import datetime, timedelta

try:
    from execution import kakao_utils
except ImportError:
    import kakao_utils

# ── 경로 설정 (크로스플랫폼) ──────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXEC_DIR        = os.path.join(BASE_DIR, "execution")
DATA_DIR        = os.path.join(BASE_DIR, "data")
OUTPUT_JSON     = os.path.join(DATA_DIR, "prices.json")
MODEL_PERF_JSON = os.path.join(BASE_DIR, "model_performance.json")
LGBM_DATED_JSON = os.path.join(BASE_DIR, "lgbm_forecast_dated.json")
GLOBAL_CLEAN_DATA = os.path.join(BASE_DIR, "농넷_과거파일", "final_clean_dataset.csv")

# ── 작물 설정 ──────────────────────────────────────────────────────────────────
CROPS = {
    'strawberry': {'id': 0, 'csv': 'strawberry_prices.csv', 'weight': 1,  'box_unit': '1kg상자'},
    'cucumber':   {'id': 1, 'csv': 'cucumber_prices.csv',   'weight': 10, 'box_unit': '10kg상자'},
    'tomato':     {'id': 2, 'csv': 'tomato_prices.csv',     'weight': 5,  'box_unit': '5kg상자'},
    'paprika':    {'id': 3, 'csv': 'paprika_prices.csv',    'weight': 5,  'box_unit': '5kg상자'},
}

HISTORY_DAYS  = 730   # 히스토리컬 데이터 기간 (2년치)
FORECAST_DAYS = 180   # 예측 기간
MAE_BASE      = 150   # 신뢰 구간 기본 오차 (원/kg, fallback용)
STEP_INC      = 20    # 스텝별 오차 증가량 (원/kg, fallback용)


# ── 유틸 ───────────────────────────────────────────────────────────────────────
def read_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def run_script(script_name, desc):
    """서브프로세스로 실행 스크립트 호출"""
    print(f"\n{'─'*60}")
    print(f"[STEP] {desc}")
    print(f"{'─'*60}")
    script_path = os.path.join(EXEC_DIR, script_name)
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=BASE_DIR,
    )
    if result.returncode != 0:
        print(f"[WARN] {script_name} 실패 (code={result.returncode}) — 계속 진행합니다.")
    return result.returncode == 0


# ── prices.json 생성 ──────────────────────────────────────────────────────────
def generate_prices_json():
    print(f"\n{'─'*60}")
    print("[STEP] data/prices.json 생성")
    print(f"{'─'*60}")

    # lgbm_forecast_dated.json 로드
    lgbm_dated = None
    if os.path.exists(LGBM_DATED_JSON):
        with open(LGBM_DATED_JSON, encoding='utf-8') as f:
            lgbm_dated = json.load(f)
        print(f"[OK] lgbm_forecast_dated.json (생성: {lgbm_dated.get('generated_at', 'unknown')})")
    else:
        print("[WARN] lgbm_forecast_dated.json 없음 — 추세 기반 fallback 예측")

    # global 과거 데이터 로드 (있으면)
    all_rows = []
    if os.path.exists(GLOBAL_CLEAN_DATA):
        all_rows = read_csv(GLOBAL_CLEAN_DATA)
        print(f"[OK] {len(all_rows)} 레코드 (global dataset)")
    else:
        print("[INFO] global dataset 없음 — data/*.csv만 사용")

    historical   = {}
    forecast_data = {}
    test_results  = {}
    all_last_dates = []

    for name, cfg in CROPS.items():
        # 1. global dataset에서 과거 데이터 추출
        crop_id = str(cfg['id'])
        crop_rows = [r for r in all_rows if str(r.get('item')) == crop_id]

        daily_map = {}
        for r in crop_rows:
            d = r.get('date', r.get('DATE'))
            if not d:
                continue
            try:
                p_val = float(r.get('price_per_kg', r.get('avg_price', 0)))
                v_val = float(r.get('volume', 0))
                if d not in daily_map:
                    daily_map[d] = {'p_sum': 0, 'count': 0, 'v_sum': 0}
                daily_map[d]['p_sum'] += p_val
                daily_map[d]['count'] += 1
                daily_map[d]['v_sum'] += v_val
            except (ValueError, TypeError):
                continue

        # 2. data/*.csv (최신 크롤링 데이터) 병합 — 최신 파일 우선
        csv_path = os.path.join(DATA_DIR, cfg['csv'])
        if os.path.exists(csv_path):
            for r in read_csv(csv_path):
                d = r.get('DATE', r.get('date'))
                if not d:
                    continue
                try:
                    p_val = float(r.get('price_per_kg', r.get('avg_price', 0)))
                    v_val = float(r.get('volume', 0))
                    daily_map[d] = {'p_sum': p_val, 'count': 1, 'v_sum': v_val}
                except Exception:
                    continue

        # 3. 정렬 후 최근 HISTORY_DAYS일
        sorted_dates = sorted(daily_map.keys())
        hist_list = []
        w        = cfg['weight']
        box_unit = cfg['box_unit']

        for d in sorted_dates[-HISTORY_DAYS:]:
            rec = daily_map[d]
            avg_p = rec['p_sum'] / rec['count']
            hist_list.append({
                "date":      d,
                "price":     int(round(avg_p)),
                "avg_price": int(round(avg_p * w)),
                "unit":      "원/kg",
                "box_unit":  box_unit,
                "volume":    int(rec['v_sum']),
            })

        historical[name] = hist_list
        if not hist_list:
            print(f"[SKIP] {name} 데이터 없음")
            continue

        last_date_str = hist_list[-1]['date']
        all_last_dates.append(last_date_str)
        last_price    = hist_list[-1]['price']
        last_dt       = datetime.strptime(last_date_str, '%Y-%m-%d')

        # 추세 계산 (최근 3일)
        prices3    = [h['price'] for h in hist_list[-3:]]
        raw_trend  = (prices3[-1] - prices3[0]) / max(len(prices3) - 1, 1) if len(prices3) >= 2 else 0
        max_daily  = last_price * 0.015
        daily_trend = max(-max_daily, min(max_daily, raw_trend * 0.4))

        # 예측 데이터: AI 우선, fallback 추세 기반
        vols7   = [h.get('volume', 0) for h in hist_list[-7:]]
        avg_vol = int(sum(vols7) / len(vols7)) if vols7 else 0

        if lgbm_dated and name in lgbm_dated.get('forecasts', {}):
            fc_list = lgbm_dated['forecasts'][name]
            for f in fc_list:
                if 'volume' not in f:
                    f['volume'] = avg_vol
            print(f"[OK] {name:12s} | AI 예측 {len(fc_list)}일 | last: {last_date_str}")
        else:
            fc_list    = []
            curr_price = last_price
            for i in range(1, FORECAST_DAYS + 1):
                curr_dt = last_dt + timedelta(days=i)
                damping = max(0.3, 1.0 - i * 0.05)
                pred    = max(int(curr_price + daily_trend * damping), 100)
                hi      = int(pred + MAE_BASE + i * STEP_INC)
                lo      = max(int(pred - MAE_BASE - i * STEP_INC), int(pred * 0.5))
                fc_list.append({
                    "date":      curr_dt.strftime('%Y-%m-%d'),
                    "price":     pred,
                    "hi":        hi,
                    "lo":        lo,
                    "unit":      "원/kg",
                    "avg_price": int(pred * w),
                    "box_unit":  box_unit,
                    "volume":    avg_vol,
                })
                curr_price = pred
            print(f"[OK] {name:12s} | 추세 예측 {len(fc_list)}일 (fallback) | last: {last_date_str}")

        forecast_data[name] = {"forecast": fc_list}

        if lgbm_dated and name in lgbm_dated.get('test_results', {}):
            test_results[name] = lgbm_dated['test_results'][name]

    if not all_last_dates:
        print("[ERROR] 데이터를 읽을 수 없습니다. data/ 폴더를 확인하세요.")
        return False

    today_str = max(all_last_dates)

    # 모델 성능 로드
    model_perf = None
    if os.path.exists(MODEL_PERF_JSON):
        with open(MODEL_PERF_JSON, encoding='utf-8') as f:
            model_perf = json.load(f)

    # prices.json 저장
    output = {
        "last_updated":    datetime.now().strftime('%Y-%m-%d %H:%M'),
        "today":           today_str,
        "historical":      historical,
        "forecast":        forecast_data,
        "model_performance": model_perf,
        "test_results":    test_results if test_results else None,
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    size_kb = os.path.getsize(OUTPUT_JSON) / 1024
    print(f"\n[SUCCESS] {OUTPUT_JSON}")
    print(f"  today   = {today_str}")
    print(f"  crops   = {list(historical.keys())}")
    print(f"  size    = {size_kb:.0f} KB")

    # ── 가격 급등락 알림 (카카오톡) ───────────────────────────────────────────
    try:
        notify_list = []
        for name, hists in historical.items():
            if len(hists) < 2: continue
            curr = hists[-1]['price']
            prev = hists[-2]['price']
            change_pct = (curr - prev) / prev * 100
            
            if abs(change_pct) >= 10: # 10% 이상 변동 시
                direction = "상승" if change_pct > 0 else "하락"
                notify_list.append(f"· {name}: {curr:,}원 ({direction} {abs(change_pct):.1f}%)")
        
        if notify_list:
            msg = f"[GREF AI 가격 알림]\n오늘 주요 품목의 단가가 크게 변동되었습니다.\n\n" + "\n".join(notify_list)
            print("\n[INFO] 카카오톡 알림 전송 시도...")
            kakao_utils.send_kakao_memo(msg)
    except Exception as e:
        print(f"[WARN] 카카오 알림 실패: {e}")

    return True


# ── 메인 ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='도매단가 대시보드 데이터 업데이트')
    parser.add_argument('--json-only',   action='store_true', help='prices.json 생성만 (크롤링/학습 건너뜀)')
    parser.add_argument('--skip-train',  action='store_true', help='모델 학습 건너뜀 (크롤링+JSON만)')
    parser.add_argument('--skip-crawl',  action='store_true', help='크롤링 건너뜀 (학습+JSON만)')
    args = parser.parse_args()

    print("=" * 60)
    print("main_update.py - 대시보드 데이터 전체 업데이트")
    print(f"실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if not args.json_only:
        if not args.skip_crawl:
            run_script('nongnet_crawler.py', '농넷 도매단가 크롤링')

        if not args.skip_train:
            run_script('train_lgbm_dated.py', 'LightGBM 모델 학습/예측')

    ok = generate_prices_json()

    if ok:
        print("\n✓ 전체 파이프라인 완료.")
    else:
        print("\n✗ 오류 발생. 위 로그를 확인하세요.")
        sys.exit(1)


if __name__ == "__main__":
    main()
