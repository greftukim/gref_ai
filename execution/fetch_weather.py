"""
fetch_weather.py - 기상청 ASOS 일자료 수집
사용: python execution/fetch_weather.py
환경변수: WEATHER_API_KEY
"""
import os, sys, io, time, json
import pandas as pd
import requests
from datetime import datetime, timedelta

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_CSV  = os.path.join(DATA_DIR, "weather_daily.csv")

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except Exception:
    pass

API_KEY = os.environ.get("WEATHER_API_KEY", "")
API_URL = "http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"

CROP_STATIONS = {
    'tomato':     [236, 192, 146],
    'strawberry': [236, 192, 156],
    'paprika':    [192, 156, 101],
    'cucumber':   [119, 236, 137],
}
STATION_NAMES = {
    101: 'Chuncheon', 119: 'Suwon', 137: 'Sangju', 146: 'Jeonju',
    156: 'Gwangju', 192: 'Jinju', 236: 'Buyeo',
}
WEATHER_FIELDS = [
    'avgTa','minTa','maxTa','sumRn',
    'avgRhm','sumSsHr','avgWs','minTg','avgTs','sumGsr',
]

def fetch_station_data(stn_id, start_date, end_date, max_retries=4):
    all_rows = []
    page = 1
    per_page = 999
    while True:
        params = {
            'serviceKey': API_KEY, 'numOfRows': per_page, 'pageNo': page,
            'dataType': 'JSON', 'dataCd': 'ASOS', 'dateCd': 'DAY',
            'startDt': start_date, 'endDt': end_date, 'stnIds': stn_id,
        }
        for attempt in range(max_retries):
            try:
                resp = requests.get(API_URL, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                wait = 2 ** (attempt + 1)
                print(f"    Retry {attempt+1}/{max_retries} (wait {wait}s): {e}")
                time.sleep(wait)
        else:
            print(f"    FAILED: stn={stn_id}, page={page}")
            break
        header = data.get('response', {}).get('header', {})
        if header.get('resultCode') != '00':
            print(f"    API error: {header.get('resultMsg')}")
            break
        body = data.get('response', {}).get('body', {})
        items = body.get('items', {}).get('item', [])
        if not items:
            break
        all_rows.extend(items)
        total = int(body.get('totalCount', 0))
        if page * per_page >= total:
            break
        page += 1
        time.sleep(0.3)
    return all_rows

def fetch_all_weather(start_year=2016, end_year=2026):
    if not API_KEY:
        print("ERROR: Set WEATHER_API_KEY environment variable")
        return None
    all_stations = sorted(set(s for ss in CROP_STATIONS.values() for s in ss))
    print(f"Stations: {[f'{s}({STATION_NAMES[s]})' for s in all_stations]}")
    all_data = []
    today = datetime.now()
    for stn_id in all_stations:
        stn_name = STATION_NAMES.get(stn_id, str(stn_id))
        print(f"\n[{stn_id} {stn_name}]")
        for year in range(start_year, end_year + 1):
            start_dt = f"{year}0101"
            end_dt = (today - timedelta(days=1)).strftime('%Y%m%d') if year == end_year else f"{year}1231"
            if int(start_dt) > int(today.strftime('%Y%m%d')):
                continue
            print(f"  {year}: {start_dt}~{end_dt} ... ", end="", flush=True)
            rows = fetch_station_data(stn_id, start_dt, end_dt)
            print(f"{len(rows)} rows")
            for row in rows:
                record = {'date': row.get('tm',''), 'stn_id': int(row.get('stnId',stn_id)),
                          'stn_name': row.get('stnNm',stn_name)}
                for field in WEATHER_FIELDS:
                    val = row.get(field, '')
                    try: record[field] = float(val) if val != '' else None
                    except: record[field] = None
                all_data.append(record)
        time.sleep(0.5)
    if not all_data:
        print("No data collected")
        return None
    df = pd.DataFrame(all_data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['stn_id','date']).reset_index(drop=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {OUT_CSV} ({len(df):,} rows)")
    return df

def build_crop_weather(weather_df=None):
    if weather_df is None:
        if not os.path.exists(OUT_CSV):
            print(f"ERROR: {OUT_CSV} not found")
            return None
        weather_df = pd.read_csv(OUT_CSV, parse_dates=['date'])
    results = []
    for crop, stations in CROP_STATIONS.items():
        crop_wx = weather_df[weather_df['stn_id'].isin(stations)]
        daily_avg = crop_wx.groupby('date')[WEATHER_FIELDS].mean().reset_index()
        daily_avg['crop'] = crop
        results.append(daily_avg)
    df_out = pd.concat(results, ignore_index=True)
    df_out = df_out.sort_values(['crop','date']).reset_index(drop=True)
    out_path = os.path.join(DATA_DIR, "weather_by_crop.csv")
    df_out.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\nCrop weather saved: {out_path} ({len(df_out):,} rows)")
    return df_out

if __name__ == '__main__':
    print("=" * 60)
    print("Weather Data Collection (KMA ASOS)")
    print("=" * 60)
    wx_df = fetch_all_weather(start_year=2016, end_year=2026)
    if wx_df is not None:
        build_crop_weather(wx_df)
    print("\nDone!")
