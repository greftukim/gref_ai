import pandas as pd
import json
import os
import numpy as np

# 농넷 UI 전국도매단가 표시 기준 (박스 단위)
STANDARD_WEIGHTS = {
    'tomato':     5,   # 5kg상자
    'cucumber':   10,  # 10kg상자
    'strawberry': 1,   # 1kg상자
    'paprika':    5,   # 5kg상자
}
UNIT_LABELS = {
    'tomato':     '5kg상자',
    'cucumber':   '10kg상자',
    'strawberry': '1kg상자',
    'paprika':    '5kg상자',
}

def generate_dashboard_json():
    # 1. 파일 경로 설정
    # 학습 결과 CSV와 원본 데이터 CSV가 필요합니다.
    result_csv = "tft_result_final_v5.csv"
    source_csv = "final_clean_dataset.csv"

    if not os.path.exists(result_csv):
        print(f"Error: {result_csv}를 찾을 수 없습니다. 학습이 완료된 후 실행해주세요.")
        return

    # 2. 데이터 로드
    print("Loading datasets...")
    df_res = pd.read_csv(result_csv)

    # 아이템 매핑 (학습 시 사용된 인덱스 기준)
    # 실제 item id (0: tomato, 1: cucumber, 2: strawberry, 3: paprika)
    item_map = {
        0: "tomato",
        1: "cucumber",
        2: "strawberry",
        3: "paprika"
    }

    dashboard_export = {}

    # 3. 데이터 변환 로직
    # tft_result_final_v5.csv 구조: sample_idx, forecast_step, actual_price, predicted_price, pred_lower_10, pred_upper_90
    # 주의: sample_idx가 어떤 item인지 매핑하는 로직이 필요합니다.
    # 여기서는 샘플 데이터의 순서가 item 순서(0, 1, 2, 3)대로 되어있다고 가정하거나, 
    # 실제 target column의 값을 보고 유추하도록 구성합니다.
    
    samples = df_res['sample_idx'].unique()
    
    for s_idx in samples:
        # 각 샘플(작물)별 데이터 필터링
        sample_df = df_res[df_res['sample_idx'] == s_idx].sort_values('forecast_step')
        
        # 현재는 sample_idx를 4로 나눈 나머지가 item id라고 가정 (일치하지 않을 경우 매핑 로직 수정 필요)
        item_id = int(s_idx % 4) 
        db_key = item_map.get(item_id, f"unknown_{item_id}")
        
        if db_key not in dashboard_export:
            dashboard_export[db_key] = {"history": [], "forecast": []}
            
        # 예측 데이터 (30일) 구성 — 가격은 원/kg 기준 (대시보드 표시 통일)
        w = STANDARD_WEIGHTS.get(db_key, 1)
        for _, row in sample_df.iterrows():
            predicted_kg = float(row['predicted_price'])
            lo_kg        = float(row['pred_lower_10'])
            hi_kg        = float(row['pred_upper_90'])
            actual_kg    = float(row['actual_price']) if not np.isnan(row['actual_price']) else None
            dashboard_export[db_key]["forecast"].append({
                "step":      int(row['forecast_step']),
                "price":     round(predicted_kg, 0),            # 원/kg (대시보드 표시 기준)
                "lo":        round(lo_kg, 0),
                "hi":        round(hi_kg, 0),
                "actual":    round(actual_kg, 0) if actual_kg is not None else None,
                "avg_price": round(predicted_kg * w, 0),        # 원/박스 (농넷 참조용)
                "unit":      '원/kg',
                "box_unit":  UNIT_LABELS.get(db_key, 'kg'),
            })

    # 4. JSON 저장
    output_path = "dashboard_data.json"
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(dashboard_export, f, ensure_ascii=False, indent=2)

    print(f"Success! {output_path} 생성 완료. 이제 대시보드에서 실제 예측 데이터를 확인하실 수 있습니다.")

if __name__ == "__main__":
    generate_dashboard_json()
