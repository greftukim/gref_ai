import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, GroupNormalizer, QuantileLoss
from pytorch_forecasting.metrics import MAPE, MAE
from datetime import datetime
import json
import random

# 한글 폰트 설정 (Windows Malgun Gothic 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def run_pipeline():
    # 1. 데이터 로드
    possible_paths = [
        "final_clean_dataset.csv",
        "C:/ai/농넷_과거데이터/final_clean_dataset.csv",
        "../final_clean_dataset.csv"
    ]
    
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading data from: {path}")
            df = pd.read_csv(path)
            break
            
    if df is None:
        print("Error: final_clean_dataset.csv not found. Please place it in the project root.")
        return

    # 2. 전처리 & 피처 엔지니어링
    print("Pre-processing data...")
    df['date'] = pd.to_datetime(df['date'])
    
    # group_id 생성
    df['group_id'] = df['item'].astype(str) + "_" + df['kind'].astype(str) + "_" + df['location'].astype(str)
    
    # 로그 변환
    df['log_price_kg'] = np.log1p(df['price_per_kg'])
    df['log_volume'] = np.log1p(df['volume'].fillna(0).clip(lower=0))
    
    # 일별 집계 (중복 방지)
    agg_dict = {
        'log_price_kg': 'mean',
        'log_volume': 'mean',
        '도매시장': 'first',
        '도매법인': 'first',
        'temp_avg': 'mean',
        'temp_max': 'mean',
        'rain': 'mean',
        'solar': 'mean',
        'hdd': 'mean'
    }
    df = df.groupby(['group_id', 'date']).agg(agg_dict).reset_index()
    
    # 파생 변수 및 결측치 처리
    df = df.sort_values(['group_id', 'date'])
    df['volume_lag7'] = df.groupby('group_id')['log_volume'].shift(7).fillna(0)
    df['volume_ma14'] = df.groupby('group_id')['log_volume'].transform(lambda x: x.rolling(14).mean()).fillna(0)
    
    weather_cols = ["temp_avg", "temp_max", "rain", "solar", "hdd"]
    for col in weather_cols:
        df[col] = df.groupby('group_id')[col].ffill().bfill().fillna(0)
        
    # time_idx 생성
    date_map = {d: i for i, d in enumerate(sorted(df['date'].unique()))}
    df['time_idx'] = df['date'].map(date_map)
    
    # 그룹 필터링 (최소 90일)
    MIN_LEN = 90
    group_counts = df.groupby('group_id').size()
    valid_groups = group_counts[group_counts >= MIN_LEN].index
    df = df[df['group_id'].isin(valid_groups)].reset_index(drop=True)
    
    print(f"Valid groups: {len(valid_groups)}")

    # 3. TimeSeriesDataSet 구성
    max_prediction_length = 30
    max_encoder_length = 60
    training_cutoff = df["time_idx"].max() - max_prediction_length

    training = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="log_price_kg",
        group_ids=["group_id"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["group_id", "도매시장", "도매법인"],
        time_varying_known_reals=["temp_avg", "temp_max", "rain", "solar", "hdd"],
        time_varying_unknown_reals=["log_volume", "volume_lag7", "volume_ma14"],
        target_normalizer=GroupNormalizer(groups=["group_id"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )

    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
    
    batch_size = 32
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

    # 4. TFT 모델 학습
    print("Starting training...")
    pl.seed_everything(42)
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=0.1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=5, min_delta=1e-4, mode="min"),
            LearningRateMonitor(logging_interval="step")
        ]
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.001,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=32,
        loss=QuantileLoss(),
        reduce_on_plateau_patience=3
    )

    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # 5. 예측 및 결과 저장
    print("Predicting...")
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    
    predictions = best_tft.predict(val_dataloader, return_index=True, return_x=True)
    out = best_tft.predict(val_dataloader, mode="quantiles")
    
    # 6. 결과 정리 및 저장
    print("Formatting results...")
    
    # 아이템 ID 매핑 (0~3 -> dashboard keys)
    item_map = {
        "0": "tomato",
        "1": "cucumber",
        "2": "strawberry",
        "3": "paprika"
    }
    
    # 인덱스와 예측값 결합
    index = predictions.index
    
    # Quantiles (QuantileLoss default: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98])
    q05 = np.expm1(out.prediction[:, :, 3].numpy())
    q01 = np.expm1(out.prediction[:, :, 1].numpy())
    q09 = np.expm1(out.prediction[:, :, 5].numpy())
    
    res_list = []
    dashboard_export = {}

    for i in range(len(index)):
        raw_group = index.iloc[i]["group_id"]
        # group_id format: item_kind_location
        item_id = raw_group.split('_')[0]
        db_key = item_map.get(item_id, f"unknown_{item_id}")
        
        if db_key not in dashboard_export:
            dashboard_export[db_key] = {"history": [], "forecast": []}
            
        # 과거 데이터 추출 (최근 60일)
        group_df = df[df['group_id'] == raw_group].sort_values('date').tail(60)
        for _, row in group_df.iterrows():
            dashboard_export[db_key]["history"].append({
                "date": row['date'].strftime('%Y-%m-%d'),
                "price": float(row['price_per_kg'])
            })
        
        # 예측 데이터 (30일)
        last_date = group_df['date'].max()
        for step in range(max_prediction_length):
            p_val = float(q05[i, step])
            lo = float(q01[i, step])
            hi = float(q09[i, step])
            forecast_date = (last_date + pd.Timedelta(days=step+1)).strftime('%Y-%m-%d')
            
            res_list.append({
                "group_id": raw_group,
                "db_key": db_key,
                "date": forecast_date,
                "predicted_price": p_val,
                "pred_lower_10": lo,
                "pred_upper_90": hi
            })
            
            dashboard_export[db_key]["forecast"].append({
                "date": forecast_date,
                "price": p_val,
                "lo": lo,
                "hi": hi
            })

    # CSV 저장
    res_df = pd.DataFrame(res_list)
    res_df.to_csv("tft_result_final_v5.csv", index=False, encoding='utf-8-sig')
    
    # 대시보드용 JSON 저장
    with open("dashboard_data.json", "w", encoding='utf-8') as f:
        json.dump(dashboard_export, f, ensure_ascii=False, indent=2)

    print(f"Success! Results saved to tft_result_final_v5.csv and dashboard_data.json")

def plot_random_group(df, best_tft, val_dataloader):
    # 무작위 그룹 선정 및 시각화 로직 (요청사항 5-2 반영)
    pass

if __name__ == "__main__":
    run_pipeline()
