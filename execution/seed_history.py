import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
HIST_FILE = os.path.join(BASE_DIR, "농넷_과거파일", "final_clean_dataset.csv")

CROPS = {
    'strawberry': 2,
    'cucumber':   1,
    'tomato':     0,
    'paprika':    3,
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
            
        # 과거 데이터의 컬럼 이름을 data/*.csv 형식에 맞게 조정 (필수 컬럼 위주)
        # DATE, avg_price, max_price, min_price, volume, unit, price_per_kg
        df_item_hist = df_item_hist.rename(columns={
            'date': 'DATE',
            'price_per_kg': 'price_per_kg', # 이미 동일할 가능성 높음
            'volume': 'volume'
        })
        
        # avg_price가 없고 price_per_kg만 있는 경우 채워줌 (단위 1kg 기준)
        if 'avg_price' not in df_item_hist.columns:
            df_item_hist['avg_price'] = df_item_hist['price_per_kg']
        
        # 2. 현재 data/*.csv 로드
        if os.path.exists(csv_path):
            df_curr = pd.read_csv(csv_path)
            df_curr['DATE'] = pd.to_datetime(df_curr['DATE']).dt.strftime('%Y-%m-%d')
            
            # 병합 및 중복 제거
            df_combined = pd.concat([df_item_hist, df_curr], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=['DATE'], keep='last')
        else:
            df_combined = df_item_hist
            
        # 3. 날짜순 정렬 후 저장
        df_combined = df_combined.sort_values('DATE')
        
        # 필요한 주요 컬럼만 유지 (너무 많은 컬럼 방지)
        cols = ['DATE', 'avg_price', 'volume', 'price_per_kg']
        # 기존 파일에 있던 다른 컬럼들도 최대한 유지
        all_cols = [c for c in df_combined.columns if c in ['DATE', 'avg_price', 'max_price', 'min_price', 'volume', 'unit', 'price_per_kg']]
        
        df_combined[all_cols].to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"[OK] {name}: {len(df_combined)} 건의 데이터가 저장되었습니다.")

if __name__ == "__main__":
    seed()
