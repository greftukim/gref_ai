import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 농넷 UI 전국도매단가 표시 기준 (박스 단위)
STANDARD_WEIGHTS = {0: 5, 1: 10, 2: 1, 3: 5}   # tomato, cucumber, strawberry, paprika
UNIT_LABELS      = {0: '5kg상자', 1: '10kg상자', 2: '1kg상자', 3: '5kg상자'}

# Create dummy data matching the PRD spec
dates = [datetime(2023,1,1) + timedelta(days=i) for i in range(200)]
items = [0, 1, 2, 3] # tomato, cucumber, strawberry, paprika
kinds = [0]
locations = [0]

data = []
for d in dates:
    for item in items:
        price_per_kg = np.random.randint(2000, 5000)
        w = STANDARD_WEIGHTS[item]
        data.append({
            "date": d,
            "item": item,
            "kind": 0,
            "location": 0,
            "도매시장": 10,
            "도매법인": 100,
            "volume": np.random.randint(500, 2000),
            "price_per_kg": price_per_kg,
            "avg_price": price_per_kg * w,          # 박스 단위 가격 (농넷 UI 기준)
            "unit": UNIT_LABELS[item],
            "temp_avg": 15 + np.random.randn(),
            "temp_max": 20 + np.random.randn(),
            "rain": 0 if np.random.rand() > 0.1 else 10,
            "solar": 15 + np.random.randn(),
            "hdd": 5
        })

df = pd.DataFrame(data)
df.to_csv("final_clean_dataset.csv", index=False)
print("Dummy data created for testing.")
