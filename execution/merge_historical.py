"""
merge_historical.py
===================
농넷_과거파일의 CSV 데이터를 data/prices.json에 병합합니다.

소스 우선순위 (낮은 번호가 먼저 로드, 높은 번호가 덮어씀):
  1. 농넷_과거파일/final_clean_dataset.csv  — 2023~2025 전체 이력
  2. data/*.csv                            — 크롤러 누적 데이터 (최신 우선)
  3. 기존 prices.json historical           — 이미 처리된 값

사용법:
  python execution/merge_historical.py
  python execution/merge_historical.py --dry-run   # 저장 없이 결과만 출력
"""

import csv
import json
import os
import sys
import argparse
from datetime import datetime

# ── 경로 ──────────────────────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR          = os.path.join(BASE_DIR, "data")
PRICES_JSON       = os.path.join(DATA_DIR, "prices.json")
GLOBAL_CSV        = os.path.join(BASE_DIR, "농넷_과거파일", "final_clean_dataset.csv")

# ── 작물 설정 ──────────────────────────────────────────────────────────────────
CROP_ID_MAP = {
    '0': 'strawberry',
    '1': 'cucumber',
    '2': 'tomato',
    '3': 'paprika',
}
CROP_CFG = {
    'strawberry': {'weight': 1,  'box_unit': '1kg상자',  'csv': 'strawberry_prices.csv'},
    'cucumber':   {'weight': 10, 'box_unit': '10kg상자', 'csv': 'cucumber_prices.csv'},
    'tomato':     {'weight': 5,  'box_unit': '5kg상자',  'csv': 'tomato_prices.csv'},
    'paprika':    {'weight': 5,  'box_unit': '5kg상자',  'csv': 'paprika_prices.csv'},
}


def read_csv_safe(path):
    rows = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def parse_date(s):
    """다양한 날짜 형식 → YYYY-MM-DD. 실패 시 None"""
    s = str(s).strip()
    for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y'):
        try:
            return datetime.strptime(s, fmt).strftime('%Y-%m-%d')
        except ValueError:
            pass
    return None


# ── 소스별 데이터 로더 ─────────────────────────────────────────────────────────

def load_from_global_csv():
    """
    final_clean_dataset.csv 로드
    컬럼: date, item, price_per_kg, volume
    """
    if not os.path.exists(GLOBAL_CSV):
        print(f"[SKIP] {GLOBAL_CSV} 없음")
        return {}

    print(f"[1/3] final_clean_dataset.csv 로드 중...")
    daily = {name: {} for name in CROP_CFG}
    skipped = 0

    with open(GLOBAL_CSV, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_raw = str(row.get('item', '')).strip()
            crop = CROP_ID_MAP.get(item_raw)
            if crop is None:
                continue

            date_str = parse_date(row.get('date', row.get('DATE', '')))
            if not date_str:
                skipped += 1
                continue

            try:
                p = float(row.get('price_per_kg') or row.get('price') or 0)
                v = float(row.get('volume') or 0)
                if p <= 0:
                    continue
            except (ValueError, TypeError):
                skipped += 1
                continue

            if date_str not in daily[crop]:
                daily[crop][date_str] = {'p_sum': 0.0, 'count': 0, 'v_sum': 0.0}
            daily[crop][date_str]['p_sum']  += p
            daily[crop][date_str]['count']  += 1
            daily[crop][date_str]['v_sum']  += v

    total = sum(len(v) for v in daily.values())
    print(f"      → {total}일치 로드 ({skipped}행 스킵)")
    for name, dm in daily.items():
        if dm:
            dates = sorted(dm)
            print(f"         {name:12s}: {dates[0]} ~ {dates[-1]} ({len(dates)}일)")
    return daily


def load_from_individual_csvs(base_daily):
    """
    data/*.csv 로드하여 base_daily에 덮어쓰기 (최신 데이터 우선)
    컬럼: DATE, price_per_kg, avg_price, volume
    """
    print(f"[2/3] data/*.csv 로드 중...")
    updated = 0

    for crop, cfg in CROP_CFG.items():
        csv_path = os.path.join(DATA_DIR, cfg['csv'])
        if not os.path.exists(csv_path):
            continue

        for row in read_csv_safe(csv_path):
            date_str = parse_date(row.get('DATE') or row.get('date', ''))
            if not date_str:
                continue
            try:
                p = float(row.get('price_per_kg') or row.get('avg_price') or 0)
                v = float(row.get('volume') or 0)
                if p <= 0:
                    continue
            except (ValueError, TypeError):
                continue

            base_daily[crop][date_str] = {'p_sum': p, 'count': 1, 'v_sum': v}
            updated += 1

    print(f"      → {updated}건 병합 (개별 CSV 우선 적용)")
    return base_daily


def load_from_prices_json(base_daily):
    """
    기존 prices.json의 historical을 base_daily에 반영
    (CSV에 없는 날짜만 추가 — 비교적 낮은 우선순위)
    """
    if not os.path.exists(PRICES_JSON):
        print("[SKIP] prices.json 없음 — 새로 생성합니다")
        return base_daily, None

    print(f"[3/3] 기존 prices.json 로드 중...")
    with open(PRICES_JSON, encoding='utf-8') as f:
        existing = json.load(f)

    added = 0
    for crop, rows in existing.get('historical', {}).items():
        if crop not in base_daily:
            continue
        cfg = CROP_CFG[crop]
        w   = cfg['weight']
        for r in rows:
            d = r.get('date')
            if not d or d in base_daily[crop]:
                continue  # 이미 CSV에서 로드된 날짜는 건너뜀
            p = r.get('price', 0)
            if p <= 0:
                continue
            base_daily[crop][d] = {'p_sum': p, 'count': 1, 'v_sum': r.get('volume', 0)}
            added += 1

    print(f"      → {added}일 추가 (prices.json에만 있던 날짜)")
    return base_daily, existing


# ── 병합 & 저장 ───────────────────────────────────────────────────────────────

def build_historical(daily):
    """daily_map → historical 리스트 (오름차순 정렬)"""
    result = {}
    for crop, dm in daily.items():
        cfg = CROP_CFG[crop]
        w, box_unit = cfg['weight'], cfg['box_unit']
        hist = []
        for date_str in sorted(dm.keys()):
            rec   = dm[date_str]
            avg_p = rec['p_sum'] / rec['count']
            hist.append({
                "date":      date_str,
                "price":     int(round(avg_p)),
                "avg_price": int(round(avg_p * w)),
                "unit":      "원/kg",
                "box_unit":  box_unit,
                "volume":    int(rec['v_sum']),
            })
        result[crop] = hist
    return result


def print_summary(historical):
    print("\n" + "=" * 60)
    print("병합 결과 요약")
    print("=" * 60)
    total = 0
    for crop, rows in historical.items():
        if not rows:
            print(f"  {crop:12s}: 데이터 없음")
            continue
        dates  = [r['date'] for r in rows]
        prices = [r['price'] for r in rows]
        print(f"  {crop:12s}: {min(dates)} ~ {max(dates)} "
              f"({len(rows):,}일 | 평균 {int(sum(prices)/len(prices)):,}원/kg)")
        total += len(rows)
    print(f"\n  합계: {total:,}개 데이터 포인트")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='저장 없이 결과만 출력')
    args = parser.parse_args()

    print("=" * 60)
    print("merge_historical.py — 과거 데이터 병합")
    print("=" * 60)

    # 1) global CSV 로드
    daily = load_from_global_csv()
    if not daily:
        daily = {name: {} for name in CROP_CFG}

    # 2) 개별 CSV 병합 (덮어쓰기)
    daily = load_from_individual_csvs(daily)

    # 3) 기존 prices.json 병합 (보완)
    daily, existing = load_from_prices_json(daily)

    # 4) historical 빌드
    historical = build_historical(daily)
    print_summary(historical)

    if args.dry_run:
        print("\n[DRY-RUN] 저장하지 않았습니다.")
        return

    # 5) prices.json 재구성 (forecast/model_perf/test_results 유지)
    output = existing or {}
    output['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    output['historical']   = historical

    # today = 가장 최근 실제 데이터 날짜
    all_last = [rows[-1]['date'] for rows in historical.values() if rows]
    if all_last:
        output['today'] = max(all_last)

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(PRICES_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    size_kb = os.path.getsize(PRICES_JSON) / 1024
    print(f"\n[SAVED] {PRICES_JSON} ({size_kb:.0f} KB)")
    print("완료. dashboard.html을 새로고침하세요.")


if __name__ == "__main__":
    main()
