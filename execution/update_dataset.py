import pandas as pd
import os
from datetime import datetime

BASE_DIR = r"C:\Users\김태우\.antigravity\260222test"
DATA_DIR = os.path.join(BASE_DIR, "data")
OLD_DATA_FILE = os.path.join(BASE_DIR, "농넷_과거파일", "final_clean_dataset.csv")
NEW_CLEAN_FILE = os.path.join(BASE_DIR, "농넷_과거파일", "final_clean_dataset_updated.csv")

# Crop mapping for final_clean_dataset.csv
# Based on grep results and common logic: 0: tomato, 1: cucumber, 2: strawberry, 3: paprika
CROP_MAP = {
    'tomato': 0,
    'cucumber': 1,
    'strawberry': 2,
    'paprika': 3
}

def update():
    print("Updating dataset with newly crawled data...")
    if not os.path.exists(OLD_DATA_FILE):
        print(f"Error: {OLD_DATA_FILE} not found.")
        return

    # 1. Load old dataset
    df_old = pd.read_csv(OLD_DATA_FILE)
    df_old['date'] = pd.to_datetime(df_old['date'])
    
    # 2. Load and process new files
    new_records = []
    crops = ['tomato', 'cucumber', 'strawberry', 'paprika']
    
    for crop in crops:
        csv_path = os.path.join(DATA_DIR, f"{crop}_prices.csv")
        if not os.path.exists(csv_path):
            print(f"Skipping {crop}: CSV not found.")
            continue
            
        df_new = pd.read_csv(csv_path)
        df_new['DATE'] = pd.to_datetime(df_new['DATE'])
        
        # Filter for dates newer than max date in old dataset for this crop
        last_date = df_old[df_old['item'] == CROP_MAP[crop]]['date'].max()
        df_new = df_new[df_new['DATE'] > last_date]
        
        print(f"Adding {len(df_new)} new records for {crop}.")
        
        for _, row in df_new.iterrows():
            # Get last record to fill missing columns (weather, market, etc.)
            last_record = df_old[df_old['item'] == CROP_MAP[crop]].iloc[-1].to_dict()
            
            # Update specific fields
            last_record['date'] = row['DATE']
            # price_per_kg 컬럼이 있으면 직접 사용, 없으면 avg_price(구버전 호환)
            last_record['price_per_kg'] = row.get('price_per_kg', row['avg_price'])
            last_record['volume'] = row['volume']
            # Other columns (weather, etc.) will stay as the last known values
            
            new_records.append(last_record)
            
    if new_records:
        df_update = pd.DataFrame(new_records)
        df_final = pd.concat([df_old, df_update], ignore_index=True)
        df_final = df_final.sort_values(['item', 'date'])
        
        # Save as updated
        df_final.to_csv(OLD_DATA_FILE, index=False) # Overwrite for the training scripts
        print(f"Dataset updated. Total records: {len(df_final)}")
    else:
        print("No new records to add.")

if __name__ == "__main__":
    update()
