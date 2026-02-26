import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
HIST_FILE = os.path.join(BASE_DIR, "농넷_과거파일", "final_clean_dataset.csv")

# final_clean_dataset.csv 의 item 컬럼 값과 매핑
# merge_historical.py 의 CROP_ID_MAP 과 반드시 일치해야 함:
#   '0' → strawberry, '1' → cucumber, '2' → tomato, '3' → paprika
CROPS = {
    'strawberry': 0,   # item=0 → 딸기
    'cucumber':   1,   # item=1 → 오이
    'tomato':     2,   # item=2 → 토마토
    'paprika':    3,   # item=3 → 파프리카
}

# 각 작물의 올바른 박스 단위 (merge_historical.py 의 CROP_CFG 와 일치)
BOX_UNITS = {
    'strawberry': '1kg상자',
    'cucumber':   '10kg상자',
    'tomato':     '5kg상자',
    'paprika':    '5kg상자',
}

def seed():
    if not os.path.exists(HIST_FILE):
        print(f"[Error] {HIST_FILE} 을 찾을 수 없습니다.")
        return

    print(f"[Info] {HIST_FILE} 에서 과거 데이터를 추출하여 data/*.csv 에 병합합니다...")
    df_hist = pd.read_csv(HIST_FILE)
    
    # date 컬럼을 표준화
    if 'date' in df_hist.columns:
        df_hist['date'] = pd.to_datetime(df_hist['date']).dt.strftime('%Y-%m-%d')
    elif 'DATE' in df_hist.columns:
        df_hist['date'] = pd.to_datetime(df_hist['DATE']).dt.strftime('%Y-%m-%d')

    for name, item_id in CROPS.items():
        csv_path = os.path.join(DATA_DIR, f"{name}_prices.csv")
        
        # 1. 해당 품종의 과거 데이터만 추출
        df_item_hist = df_hist[df_hist['item'] == item_id].copy()
        if df_item_hist.empty:
            print(f"[Warn] {name} (id:{item_id}) 에 대한 과거 데이터가 없습니다.")
            continue

        # 날짜별로 price_per_kg 평균, volume 합산 집계
        df_item_hist = df_item_hist.rename(columns={'date': 'DATE'})
        df_agg = df_item_hist.groupby('DATE').agg(
            price_per_kg=('price_per_kg', 'mean'),
            volume=('volume', 'sum'),
        ).reset_index()

        # avg_price = price_per_kg (단위 1kg 기준으로 저장, merge_historical 에서 박스 환산)
        df_agg['avg_price'] = df_agg['price_per_kg'].round(2)
        df_agg['price_per_kg'] = df_agg['price_per_kg'].round(2)
        df_agg['volume'] = df_agg['volume'].round(1)
        # 올바른 박스 단위 설정
        df_agg['unit'] = BOX_UNITS[name]

        # 2. 현재 data/*.csv 로드 후 병합
        # - CUTOFF_DATE 이전: final_clean_dataset.csv 데이터로 완전 대체
        # - CUTOFF_DATE 이후: 기존 크롤러 데이터 우선 보존
        CUTOFF_DATE = '2026-01-01'
        if os.path.exists(csv_path):
            df_curr = pd.read_csv(csv_path)
            df_curr['DATE'] = pd.to_datetime(df_curr['DATE']).dt.strftime('%Y-%m-%d')
            # 최신 크롤러 데이터(CUTOFF 이후)만 보존
            df_curr_recent = df_curr[df_curr['DATE'] >= CUTOFF_DATE].copy()
            # 과거 데이터는 seed 데이터 우선, 최신 크롤러 데이터를 그 뒤에 추가
            df_combined = pd.concat([df_agg, df_curr_recent], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=['DATE'], keep='last')
        else:
            df_combined = df_agg

        # 3. 날짜순 정렬 후 저장
        df_combined = df_combined.sort_values('DATE').reset_index(drop=True)

        # 필요 컬럼만 유지
        keep_cols = [c for c in ['DATE', 'unit', 'avg_price', 'price_per_kg', 'volume', 'max_price', 'min_price']
                     if c in df_combined.columns]

        df_combined[keep_cols].to_csv(csv_path, index=False, encoding='utf-8-sig')
        hist_count = len(df_agg)
        recent_count = len(df_combined) - hist_count
        print(f"[OK] {name} (item={item_id}): 과거 {hist_count}건 + 최신 {recent_count}건 = 총 {len(df_combined)}건 저장 → {csv_path}")

if __name__ == "__main__":
    seed()
