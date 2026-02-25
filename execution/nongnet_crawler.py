import os
import time
import glob
import json
import urllib.parse
import pandas as pd
from datetime import date, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# 1. 경로 설정 (크로스플랫폼 — GitHub Actions 호환)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# GitHub Actions 환경에서는 NONGNET_DOWNLOAD_DIR 환경변수로 덮어씀
DOWNLOAD_DIR = os.environ.get(
    'NONGNET_DOWNLOAD_DIR',
    os.path.join(os.path.expanduser('~'), 'Downloads')
)
DATA_DIR = os.path.join(BASE_DIR, "data")
HISTORICAL_JSON = os.path.join(BASE_DIR, "historical_data.json")

for d in [DATA_DIR]:
    if not os.path.exists(d): os.makedirs(d)

# 작물 설정 (농넷 키워드 및 상세 필터)
# standard_weight_kg: 농넷 UI 전국도매단가 표시 기준 거래단위 (kg)
CROPS_CONFIG = [
    {'name': '토마토', 'variety': '완숙',   'id': 'tomato',     'standard_weight_kg': 5,  'unit_label': '5kg상자'},
    {'name': '오이',   'variety': '백다다기','id': 'cucumber',   'standard_weight_kg': 10, 'unit_label': '10kg상자'},
    {'name': '딸기',   'variety': '설향',   'id': 'strawberry', 'standard_weight_kg': 1,  'unit_label': '1kg상자'},
    {'name': '파프리카','variety': '미니',   'id': 'paprika',    'standard_weight_kg': 5,  'unit_label': '5kg상자'},
]

def generate_url(crop_name):
    base = "https://www.nongnet.or.kr/qlik/sso/single/"
    params = urllib.parse.urlencode({
        "appid": "551d7860-2a5d-49e5-915e-56517f3da2a3", 
        "sheet": "d89143e2-368a-4d41-9851-d4f58ce060dc", 
        "opt": "ctxmenu,currsel"
    })
    p_item = f"select=$::%ED%92%88%EB%AA%A9%EB%AA%85_%EC%84%A0%ED%83%9D,{urllib.parse.quote(crop_name)}"
    # 일회성으로 최근 60일치 데이터를 가져오도록 변경 (range(31) -> range(60))
    dates = [str((date.today() - timedelta(days=i) - date(1899, 12, 30)).days) for i in range(60)]
    p_date = f"select=$::%EA%B2%BD%EB%9D%BD%EC%9D%BC%EC%9E%90_%EC%84%A0%ED%83%9D,{','.join(dates)}"
    return f"{base}?{params}&{p_item}&{p_date}"

def process_and_update(file_path, config):
    try:
        print(f"[{config['id']}] Processing data...", flush=True)
        df = pd.read_excel(file_path)
        
        # 0. 컬럼명을 인덱스 기반으로 강제 변경
        df.columns = [f"col_{i}" for i in range(len(df.columns))]
        
        # DATE 컬럼 (0번), 물량/금액 (3, 4번), 품목/품종 (7, 8번)
        col_date, col_vol, col_amount = 'col_0', 'col_3', 'col_4'
        col_item, col_variety = 'col_7', 'col_8'
        
        df[col_date] = pd.to_datetime(df[col_date])
        
        # 작물/품종 필터링
        df = df[df[col_item].astype(str).str.contains(config['name'])].copy()
        if config['variety']:
            df = df[df[col_variety].astype(str).str.contains(config['variety'])].copy()
            
        # 수치형 변환
        for c in [col_vol, col_amount]:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            
        # 단가 계산 (kg당)
        df['kg_price'] = 0.0
        mask = df[col_vol] > 0
        df.loc[mask, 'kg_price'] = df.loc[mask, col_amount] / df.loc[mask, col_vol]

        # 일자별 집계
        daily = df.groupby(col_date).agg({
            col_vol: 'sum',
            col_amount: 'sum',
            'kg_price': ['max', 'min']
        }).reset_index()

        # 컬럼 평탄화 및 이름 통일
        daily.columns = ['DATE', 'volume', 'amount', 'max_price', 'min_price']

        # 데이터가 비어있는 행 제거
        daily = daily[daily['volume'] > 0].copy()

        # 평균 단가 계산 (kg당)
        daily['price_per_kg'] = daily['amount'] / daily['volume']

        # 농넷 UI 표시 기준 단위(박스)로 변환
        w = config.get('standard_weight_kg', 1)
        daily['avg_price'] = daily['price_per_kg'] * w
        daily['max_price'] = daily['max_price'] * w
        daily['min_price'] = daily['min_price'] * w

        # 개별 컬럼별로 안전하게 변환
        for col in ['avg_price', 'max_price', 'min_price', 'price_per_kg', 'volume']:
            temp_col = pd.to_numeric(daily[col], errors='coerce').fillna(0)
            daily[col] = temp_col.round(0).astype('int64')

        daily['unit'] = config.get('unit_label', 'kg')
        daily['DATE'] = daily['DATE'].dt.strftime('%Y-%m-%d')
        daily = daily[['DATE', 'avg_price', 'max_price', 'min_price', 'volume', 'unit', 'price_per_kg']].sort_values('DATE')

        # CSV DB 업데이트 (누적)
        csv_path = os.path.join(DATA_DIR, f"{config['id']}_prices.csv")
        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            # 중복 제거 (날짜 기준)
            combined = pd.concat([existing_df, daily]).drop_duplicates(subset=['DATE'], keep='last')
            combined = combined.sort_values('DATE')
        else:
            combined = daily
            
        combined.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"CSV update complete: {csv_path} (Total {len(combined)} records)")
        
        return True
    except Exception as e:
        print(f"ERROR: Data processing error: {e}", flush=True)
        return False

def update_dashboard_json():
    """모든 작물의 CSV를 읽어 대시보드용 JSON 생성"""
    print("Updating Dashboard JSON data...", flush=True)
    all_data = {}
    for config in CROPS_CONFIG:
        csv_path = os.path.join(DATA_DIR, f"{config['id']}_prices.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # JSON 형식으로 변환 (date, price, volume)
            items = []
            for _, row in df.iterrows():
                items.append({
                    "date": row['DATE'],
                    "price": int(row['avg_price']),
                    "volume": int(row['volume'])
                })
            all_data[config['id']] = items
        else:
            all_data[config['id']] = []
            
    with open(HISTORICAL_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"JSON update complete: {HISTORICAL_JSON}")

def download_nongnet_excel(driver, crop_name):
    # (기존 download_nongnet_excel 로직 유지...)
    url = generate_url(crop_name)
    driver.get(url)
    print(f"\nAccessing and loading data for crop...", flush=True)
    time.sleep(20) # Increased initial wait
    
    print(f"Current Page Title: {driver.title}", flush=True)
    
    # 팝업 닫기 (존재할 경우)
    try:
        popups = driver.find_elements(By.XPATH, "//*[text()='닫기']")
        if popups:
            print(f"Found {len(popups)} popups. Closing...", flush=True)
            for btn in popups:
                if btn.is_displayed():
                    driver.execute_script("arguments[0].click();", btn)
    except Exception as e:
        print(f"Popup close error: {e}", flush=True)
        
    time.sleep(15) 
    
    # Debug: Print all frames
    frames = driver.find_elements(By.TAG_NAME, "iframe")
    print(f"Found {len(frames)} iframes.", flush=True)
    for i, frame in enumerate(frames):
        print(f"Frame {i}: id={frame.get_attribute('id')}, name={frame.get_attribute('name')}", flush=True)

    # 엑셀 다운로드 버튼 클릭
    try:
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        
        # 1. Wait for exportBtn specifically
        print("Waiting for exportBtn...", flush=True)
        try:
            best_btn = WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.ID, "exportBtn"))
            )
            print("Jackpot! exportBtn found by ID through WebDriverWait.", flush=True)
        except:
            print("exportBtn ID not found within 30s. Searching candidates...", flush=True)
            best_btn = None

        if not best_btn:
            candidates = []
            keywords = ['엑셀', 'Excel', '데이터', '저장', 'Export', 'Download']
            possible_els = driver.find_elements(By.XPATH, "//button | //a | //div[@role='button' or @id='exportBtn' or contains(@title, 'Excel')] | //span[contains(text(), '데이터')]")
            
            for el in possible_els:
                try:
                    title = (el.get_attribute('title') or "").lower()
                    text = (el.text or "").lower()
                    aria = (el.get_attribute('aria-label') or "").lower()
                    combined = title + text + aria
                    if any(k.lower() in combined for k in keywords):
                        candidates.append(el)
                except: continue

            if candidates:
                best_btn = candidates[0]
                for btn in candidates:
                    if 'excel' in (btn.get_attribute('title') or "").lower() or 'excel' in (btn.text or "").lower():
                        best_btn = btn
                        break

        if best_btn:
            print(f"Clicking best candidate: Tag={best_btn.tag_name}, ID={best_btn.get_attribute('id')}, Title={best_btn.get_attribute('title')}", flush=True)
            driver.execute_script("arguments[0].scrollIntoView(true);", best_btn)
            time.sleep(1)
            driver.execute_script("arguments[0].click();", best_btn)
            print("Click executed.", flush=True)
        else:
            print("No download button found after comprehensive search.", flush=True)
            return None
    except Exception as e:
        print(f"Error during button click: {e}", flush=True)
        return None

    before = set(glob.glob(os.path.join(DOWNLOAD_DIR, "*.xlsx")))
    for i in range(60):
        time.sleep(1)
        after = set(glob.glob(os.path.join(DOWNLOAD_DIR, "*.xlsx")))
        new_files = list(after - before)
        if new_files:
            latest_file = max(new_files, key=os.path.getctime)
            if not latest_file.endswith('.crdownload'):
                return latest_file
    return None

if __name__ == "__main__":
    options = webdriver.ChromeOptions()
    options.add_experimental_option("prefs", {"download.default_directory": DOWNLOAD_DIR})
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    print("Browser driver initialization...", flush=True)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        any_updated = False
        for crop in CROPS_CONFIG:
            file_path = download_nongnet_excel(driver, crop['name'])
            
            if file_path:
                print(f"File download success: {file_path}", flush=True)
                if process_and_update(file_path, crop):
                    any_updated = True
                    # 임시 엑셀 파일 삭제
                    os.remove(file_path)
                    print(f"Temporary file deleted: {file_path}", flush=True)
                else:
                    print(f"FAILED: Data update failed for {crop['id']}", flush=True)
            else:
                print(f"FAILED: Download failed for {crop['name']}", flush=True)
            
            time.sleep(3)

        if any_updated:
            update_dashboard_json()
            
    finally:
        driver.quit()
        print("\nAll automation tasks completed.", flush=True)
