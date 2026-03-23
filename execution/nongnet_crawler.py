import os
import time
import glob
import json
import urllib.parse
import pandas as pd
from datetime import date, datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# 1. 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOAD_DIR = os.environ.get('NONGNET_DOWNLOAD_DIR', os.path.join(os.path.expanduser('~'), 'Downloads'))
DATA_DIR = os.path.join(BASE_DIR, "data")
HISTORICAL_JSON = os.path.join(BASE_DIR, "historical_data.json")

os.makedirs(DATA_DIR, exist_ok=True)

# 작물 설정
CROPS_CONFIG = [
    {'name': '토마토', 'variety': '완숙',   'id': 'tomato',     'weight': 5,  'unit': '5kg상자'},
    {'name': '오이',   'variety': '백다다기','id': 'cucumber',   'weight': 10, 'unit': '10kg상자'},
    {'name': '딸기',   'variety': '설향',   'id': 'strawberry', 'weight': 1,  'unit': '1kg상자'},
    {'name': '파프리카','variety': '미니',   'id': 'paprika',    'weight': 5,  'unit': '5kg상자'},
]

def generate_urls(crop_name):
    """다양한 필터 명칭을 시도하기 위한 URL 리스트 생성"""
    base = "https://www.nongnet.or.kr/qlik/sso/single/"
    params = urllib.parse.urlencode({
        "appid": "551d7860-2a5d-49e5-915e-56517f3da2a3", 
        "sheet": "d89143e2-368a-4d41-9851-d4f58ce060dc", 
        "opt": "ctxmenu,currsel"
    })
    
    # 1. 시리얼 날짜 (range 60)
    serial_dates = [str((date.today() - timedelta(days=i) - date(1899, 12, 30)).days) for i in range(60)]
    # 2. YYYYMMDD 날짜
    standard_dates = [(date.today() - timedelta(days=i)).strftime('%Y%m%d') for i in range(60)]
    
    urls = []
    # 후보 1: 경락일자_선택 (기존)
    p_item = f"select=$::%ED%92%88%EB%AA%A9%EB%AA%85_%EC%84%A0%ED%83%9D,{urllib.parse.quote(crop_name)}"
    p_date1 = f"select=$::%EA%B2%BD%EB%9D%BD%EC%9D%BC%EC%9E%90_%EC%84%A0%ED%83%9D,{','.join(serial_dates)}"
    urls.append(f"{base}?{params}&{p_item}&{p_date1}")
    
    # 후보 2: 경락일자 (일반 버전)
    p_date2 = f"select=$::%EA%B2%BD%EB%9D%BD%EC%9D%BC%EC%9E%90,{','.join(standard_dates)}"
    urls.append(f"{base}?{params}&{p_item}&{p_date2}")

    # 후보 3: 필터 없이 (최근 데이터 기본 노출 기대)
    urls.append(f"{base}?{params}&{p_item}")
    
    return urls

def find_header_and_read(file_path):
    """엑셀 파일에서 실제 데이터 헤더를 찾아 읽기"""
    # 원본 파일 읽기
    df_raw = pd.read_excel(file_path, header=None)
    
    # '경락일자' 또는 'Date'가 포함된 행 찾기
    header_row = 0
    for i in range(min(15, len(df_raw))):
        row_str = " ".join(df_raw.iloc[i].astype(str))
        if '경락일자' in row_str or 'DATE' in row_str.upper():
            header_row = i
            break
            
    df = pd.read_excel(file_path, skiprows=header_row)
    # 컬럼명 정규화 (공백 제거)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def process_and_update(file_path, config):
    try:
        print(f"[{config['id']}] Processing downloaded file...")
        df = find_header_and_read(file_path)
        
        # 컬럼 맵핑 (유연한 검색)
        col_date = next((c for c in df.columns if '일자' in c or 'DATE' in c.upper()), df.columns[0])
        col_vol  = next((c for c in df.columns if '물량' in c or 'VOLUME' in c.upper()), df.columns[3] if len(df.columns)>3 else None)
        col_amt  = next((c for c in df.columns if '금액' in c or 'AMOUNT' in c.upper() or 'PRICE' in c.upper()), df.columns[4] if len(df.columns)>4 else None)
        
        if not col_vol or not col_amt:
            raise ValueError("Required columns (Volume, Amount) not found in Excel.")

        # 날짜 변환
        df[col_date] = pd.to_datetime(df[col_date], errors='coerce')
        df = df.dropna(subset=[col_date])
        
        # 수치 변환
        for c in [col_vol, col_amt]:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            
        # 단가 계산 (kg당)
        df['price_per_kg'] = 0.0
        mask = df[col_vol] > 0
        df.loc[mask, 'price_per_kg'] = df.loc[mask, col_amt] / df.loc[mask, col_vol]

        # 일자별 집계
        daily = df.groupby(col_date).agg({
            col_vol: 'sum',
            col_amt: 'sum',
            'price_per_kg': ['max', 'min', 'mean']
        }).reset_index()
        
        # 컬럼 이름 정리
        daily.columns = ['DATE', 'volume', 'amount', 'max_p', 'min_p', 'avg_p']
        daily = daily[daily['volume'] > 0].copy()
        
        # 단위 보정 (box weight)
        w = config['weight']
        daily['avg_price'] = (daily['avg_p'] * w).round(0).astype(int)
        daily['max_price'] = (daily['max_p'] * w).round(0).astype(int)
        daily['min_price'] = (daily['min_p'] * w).round(0).astype(int)
        daily['price_per_kg'] = daily['avg_p'].round(0).astype(int)
        daily['volume'] = daily['volume'].round(0).astype(int)
        daily['unit'] = config['unit']
        
        daily['DATE'] = daily['DATE'].dt.strftime('%Y-%m-%d')
        daily = daily[['DATE', 'avg_price', 'max_price', 'min_price', 'volume', 'unit', 'price_per_kg']]

        # CSV 업데이트
        csv_path = os.path.join(DATA_DIR, f"{config['id']}_prices.csv")
        if os.path.exists(csv_path):
            existing = pd.read_csv(csv_path)
            combined = pd.concat([existing, daily]).drop_duplicates(subset=['DATE'], keep='last')
            combined = combined.sort_values('DATE')
        else:
            combined = daily.sort_values('DATE')
            
        combined.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"[{config['id']}] Updated CSV. Latest date: {combined['DATE'].max()}")
        return True
    except Exception as e:
        print(f"Error processing {config['id']}: {e}")
        return False

def download_excel(driver, crop_name):
    urls = generate_urls(crop_name)
    best_file = None
    
    for idx, url in enumerate(urls):
        print(f"Attempting URL variant {idx+1} for {crop_name}...")
        driver.get(url)
        time.sleep(15) # Wait for Qlik to load
        
        try:
            # 팝업 닫기
            popups = driver.find_elements(By.XPATH, "//*[text()='닫기']")
            for p in popups:
                if p.is_displayed(): driver.execute_script("arguments[0].click();", p)
                
            # exportBtn 대기
            wait = WebDriverWait(driver, 20)
            btn = wait.until(EC.element_to_be_clickable((By.ID, "exportBtn")))
            
            before = set(glob.glob(os.path.join(DOWNLOAD_DIR, "*.xlsx")))
            driver.execute_script("arguments[0].click();", btn)
            print("Download button clicked. Waiting...")
            
            for i in range(30):
                time.sleep(1)
                after = set(glob.glob(os.path.join(DOWNLOAD_DIR, "*.xlsx")))
                news = list(after - before)
                if news:
                    latest = max(news, key=os.path.getctime)
                    if not latest.endswith('.crdownload'):
                        print(f"Downloaded: {os.path.basename(latest)}")
                        return latest
        except Exception as e:
            print(f"Variant {idx+1} failed: {e}")
            continue
            
    return None

def main():
    options = webdriver.ChromeOptions()
    options.add_experimental_option("prefs", {"download.default_directory": DOWNLOAD_DIR})
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    any_success = False
    try:
        for crop in CROPS_CONFIG:
            file_path = download_excel(driver, crop['name'])
            if file_path:
                if process_and_update(file_path, crop):
                    any_success = True
                if os.path.exists(file_path): os.remove(file_path)
            time.sleep(2)
            
        if any_success:
            print("Finished crawling. You should run main_update.py now.")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
