"""
parse_nongnet_xlsx.py
=====================
농넷 다운로드 xlsx 파일(품목별 x 기간별)을 읽어
전체 시장 가중평균 kg 단가로 집계 -> daily CSV 생성.

사용: python execution/parse_nongnet_xlsx.py
"""

import pandas as pd
import numpy as np
import re
import os
import sys
import io
import warnings

warnings.filterwarnings('ignore')

# Windows 콘솔 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ─── 경로 ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOAD_DIR = os.path.join(os.path.expanduser("~"), "Downloads")
OUT_CSV = os.path.join(BASE_DIR, "data", "nongnet_10yr_daily.csv")

# ─── 파일 매핑 ─────────────────────────────────────────────────────────────────
FILES = {
    'tomato': [
        os.path.join(DOWNLOAD_DIR, "16_210325_토마토.xlsx"),
        os.path.join(DOWNLOAD_DIR, "21_260326_토마토.xlsx"),
    ],
    'strawberry': [
        os.path.join(DOWNLOAD_DIR, "16_210325_딸기.xlsx"),
        os.path.join(DOWNLOAD_DIR, "21_260326_딸기.xlsx"),
    ],
    'paprika': [
        os.path.join(DOWNLOAD_DIR, "16_210325_파프리카.xlsx"),
        os.path.join(DOWNLOAD_DIR, "21_260326_파프리카.xlsx"),
    ],
    'cucumber': [
        os.path.join(DOWNLOAD_DIR, "16_210325_오이.xlsx"),
        os.path.join(DOWNLOAD_DIR, "21_260326_오이.xlsx"),
    ],
}

# ─── 품종 필터 ─────────────────────────────────────────────────────────────────
VARIETY_FILTER = {
    'tomato': '완숙토마토',
    'strawberry': '설향',
    'paprika': '미니파프리카',
    'cucumber': '백다다기',
}

# ─── 등급 필터 (특, 상, 보통만) ────────────────────────────────────────────────
VALID_GRADES = {'특', '상', '보통'}


def extract_kg(unit_str) -> float:
    """거래단위에서 kg 수 추출. 예: '5kg상자' -> 5.0, '10kg파렛트' -> 10.0"""
    if not isinstance(unit_str, str):
        return np.nan
    m = re.search(r'([\d.]+)\s*kg', unit_str, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return np.nan


def load_and_filter(filepath: str, crop: str) -> pd.DataFrame:
    """
    xlsx 파일 로드 -> 품종/등급 필터 -> 반환
    """
    variety = VARIETY_FILTER[crop]
    print(f"  Loading: {os.path.basename(filepath)} ...")

    df = pd.read_excel(filepath, engine='openpyxl')

    # 컬럼 수에 따라 이름 매핑 (인덱스 기반)
    ncols = len(df.columns)
    col_map = {
        df.columns[0]: 'DATE',
        df.columns[1]: '거래단위',
        df.columns[2]: '평균가격',
        df.columns[3]: '총거래물량',
        df.columns[4]: '총거래금액',
        df.columns[5]: '도매시장',
    }
    if ncols >= 12:
        # [6]=도매법인, [7]=품목, [8]=품종, [9]=산지-광역시도, [10]=산지-시군구, [11]=등급
        col_map[df.columns[6]] = '도매법인'
        col_map[df.columns[7]] = '품목'
        col_map[df.columns[8]] = '품종'
        col_map[df.columns[9]] = '산지_광역시도'
        col_map[df.columns[10]] = '산지_시군구'
        col_map[df.columns[11]] = '등급'
    else:
        # 11 cols: [6]=품목, [7]=품종, [8]=산지-광역시도, [9]=산지-시군구, [10]=등급
        col_map[df.columns[6]] = '품목'
        col_map[df.columns[7]] = '품종'
        col_map[df.columns[8]] = '산지_광역시도'
        col_map[df.columns[9]] = '산지_시군구'
        col_map[df.columns[10]] = '등급'

    df = df.rename(columns=col_map)

    # 날짜 변환
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df = df.dropna(subset=['DATE'])

    # 품종 필터
    df_filtered = df[df['품종'] == variety].copy()
    if len(df_filtered) == 0:
        print(f"    WARNING: '{variety}' not found! Available: {df['품종'].unique()[:10]}")
        return pd.DataFrame()

    # 등급 필터
    df_filtered = df_filtered[df_filtered['등급'].isin(VALID_GRADES)]
    if len(df_filtered) == 0:
        print(f"    WARNING: no valid grades!")
        return pd.DataFrame()

    # 숫자 변환
    df_filtered['평균가격'] = pd.to_numeric(df_filtered['평균가격'], errors='coerce')
    df_filtered['총거래물량'] = pd.to_numeric(df_filtered['총거래물량'], errors='coerce')
    df_filtered['총거래금액'] = pd.to_numeric(df_filtered['총거래금액'], errors='coerce')
    df_filtered = df_filtered.dropna(subset=['평균가격', '총거래물량', '총거래금액'])
    df_filtered = df_filtered[df_filtered['총거래물량'] > 0]

    # kg 추출
    df_filtered['kg_per_unit'] = df_filtered['거래단위'].apply(extract_kg)
    df_filtered = df_filtered.dropna(subset=['kg_per_unit'])
    df_filtered = df_filtered[df_filtered['kg_per_unit'] > 0]

    print(f"    OK: {len(df_filtered):,} rows ({df_filtered['DATE'].min().date()} ~ {df_filtered['DATE'].max().date()})")
    return df_filtered


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    전체 시장 일별 가중평균:
      총거래물량은 이미 kg 단위이므로:
      price_per_kg = sum(총거래금액) / sum(총거래물량)
      volume_kg = sum(총거래물량)
    """
    daily = df.groupby('DATE').agg(
        total_amount=('총거래금액', 'sum'),
        total_volume=('총거래물량', 'sum'),
    ).reset_index()

    daily['price_per_kg'] = daily['total_amount'] / daily['total_volume']
    daily['volume_kg'] = daily['total_volume']

    # Remove invalid rows
    daily = daily[(daily['price_per_kg'] > 0) & (daily['volume_kg'] > 0)]
    daily = daily.dropna(subset=['price_per_kg', 'volume_kg'])

    # 이상치 제거: IQR × 5 범위 밖 제거
    q1 = daily['price_per_kg'].quantile(0.01)
    q99 = daily['price_per_kg'].quantile(0.99)
    iqr = q99 - q1
    lower = max(q1 - iqr * 2, 0)
    upper = q99 + iqr * 2
    before = len(daily)
    daily = daily[(daily['price_per_kg'] >= lower) & (daily['price_per_kg'] <= upper)]
    removed = before - len(daily)
    if removed > 0:
        print(f"    Outlier removed: {removed} rows (range: {lower:.0f}~{upper:.0f} won/kg)")

    daily = daily.rename(columns={'DATE': 'date'})
    daily = daily[['date', 'price_per_kg', 'volume_kg']].sort_values('date').reset_index(drop=True)

    return daily


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    all_results = []

    for crop, filepaths in FILES.items():
        variety = VARIETY_FILTER[crop]
        print(f"\n{'='*60}")
        print(f"[{crop.upper()}] variety filter: {variety}")
        print(f"{'='*60}")

        dfs = []
        for fp in filepaths:
            if not os.path.exists(fp):
                print(f"  WARNING: file not found: {fp}")
                continue
            df_part = load_and_filter(fp, crop)
            if len(df_part) > 0:
                dfs.append(df_part)

        if not dfs:
            print(f"  SKIP: {crop} - no data")
            continue

        df_all = pd.concat(dfs, ignore_index=True)

        # 중복 제거
        before = len(df_all)
        df_all = df_all.drop_duplicates()
        if before != len(df_all):
            print(f"  Dedup: {before:,} -> {len(df_all):,}")

        # 일별 집계
        daily = aggregate_daily(df_all)
        daily['crop'] = crop

        all_results.append(daily)

        # Summary stats
        print(f"\n  Summary:")
        print(f"    Date range: {daily['date'].min().date()} ~ {daily['date'].max().date()}")
        print(f"    Total days: {len(daily)}")
        print(f"    Avg price:  {daily['price_per_kg'].mean():,.0f} won/kg")
        print(f"    Min price:  {daily['price_per_kg'].min():,.0f} won/kg")
        print(f"    Max price:  {daily['price_per_kg'].max():,.0f} won/kg")

    if not all_results:
        print("\nERROR: No data to output!")
        return

    # Combine and sort by crop, then date
    df_final = pd.concat(all_results, ignore_index=True)
    df_final = df_final.sort_values(['crop', 'date']).reset_index(drop=True)

    # Format date as YYYY-MM-DD string
    df_final['date'] = df_final['date'].dt.strftime('%Y-%m-%d')

    # Round numeric columns
    df_final['price_per_kg'] = df_final['price_per_kg'].round(2)
    df_final['volume_kg'] = df_final['volume_kg'].round(2)

    # Output columns in specified order
    df_final = df_final[['date', 'crop', 'price_per_kg', 'volume_kg']]

    # Remove zero/null volume rows
    df_final = df_final[(df_final['volume_kg'] > 0) & (df_final['volume_kg'].notna())]

    df_final.to_csv(OUT_CSV, index=False)
    print(f"\n{'='*60}")
    print(f"OUTPUT: {OUT_CSV}")
    print(f"Total rows: {len(df_final):,}")
    print(f"{'='*60}")

    # Per-crop summary
    for crop_name in sorted(df_final['crop'].unique()):
        sub = df_final[df_final['crop'] == crop_name]
        print(f"  {crop_name:12s}: {sub['date'].min()} ~ {sub['date'].max()} | "
              f"{len(sub):,} days | avg {sub['price_per_kg'].mean():,.0f} won/kg | "
              f"min {sub['price_per_kg'].min():,.0f} | max {sub['price_per_kg'].max():,.0f}")


if __name__ == '__main__':
    main()
