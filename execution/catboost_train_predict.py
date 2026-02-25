"""
CatBoost 농산물 경락 단가 예측 모델
- final_clean_dataset.csv를 사용하여 학습
- TFT/LightGBM과 동일한 전처리 규칙 적용
- CatBoost 고유의 범주형 변수 직접 처리 활용
- 출력: catboost_result.csv (Actual, Predicted)
"""

import os
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """데이터 로드"""
    possible_paths = [
        r"C:\Users\김태우\.antigravity\260222test\농넷_과거파일\final_clean_dataset.csv",
        "../농넷_과거파일/final_clean_dataset.csv",
        "final_clean_dataset.csv",
        "C:/ai/농넷_과거데이터/final_clean_dataset.csv",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading data from: {path}")
            return pd.read_csv(path)
    raise FileNotFoundError("final_clean_dataset.csv를 찾을 수 없습니다.")


def preprocess(df):
    """전처리 & 피처 엔지니어링 (LightGBM과 동일 파이프라인)"""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # 1. group_id 생성
    df['group_id'] = (
        df['item'].astype(str) + "_" +
        df['kind'].astype(str) + "_" +
        df['location'].astype(str)
    )

    # 2. 일별 집계
    agg_dict = {
        'price_per_kg': 'mean',
        'volume': 'mean',
        '도매시장': 'first',
        '도매법인': 'first',
        'item': 'first',
        'kind': 'first',
        'location': 'first',
        'temp_avg': 'mean',
        'temp_max': 'mean',
        'rain': 'mean',
        'solar': 'mean',
        'hdd': 'mean',
    }
    df = df.groupby(['group_id', 'date']).agg(agg_dict).reset_index()
    df = df.sort_values(['group_id', 'date']).reset_index(drop=True)

    # 3. 결측치 처리
    df['volume'] = df['volume'].fillna(0).clip(lower=0)
    weather_cols = ['temp_avg', 'temp_max', 'rain', 'solar', 'hdd']
    for col in weather_cols:
        if col in df.columns:
            df[col] = df.groupby('group_id')[col].ffill().bfill().fillna(0)

    # 4. Lag 피처
    for lag in [1, 3, 7, 14, 30]:
        df[f'price_lag{lag}'] = df.groupby('group_id')['price_per_kg'].shift(lag)

    # 5. Rolling 통계 (Leakage 방지를 위해 1일 shift)
    for window in [7, 14, 30]:
        df[f'price_ma{window}'] = df.groupby('group_id')['price_per_kg'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'price_std{window}'] = df.groupby('group_id')['price_per_kg'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std()
        )

    # 6. Volume lag/rolling
    df['volume_lag7'] = df.groupby('group_id')['volume'].shift(7)
    df['volume_ma14'] = df.groupby('group_id')['volume'].transform(
        lambda x: x.rolling(14, min_periods=1).mean()
    )

    # 7. 날짜 피처
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['day_of_month'] = df['date'].dt.day

    # 8. 가격 변화율
    df['price_change_1d'] = df.groupby('group_id')['price_per_kg'].pct_change(1)
    df['price_change_7d'] = df.groupby('group_id')['price_per_kg'].pct_change(7)

    # 9. 범주형 변수를 문자열로 (CatBoost는 문자열 범주형 직접 처리)
    df['도매시장'] = df['도매시장'].astype(str)
    df['도매법인'] = df['도매법인'].astype(str)
    df['group_id'] = df['group_id'].astype(str)
    df['item'] = df['item'].astype(str)
    df['kind'] = df['kind'].astype(str)
    df['location'] = df['location'].astype(str)

    # 10. 그룹 필터링 (최소 90일)
    MIN_LEN = 90
    group_counts = df.groupby('group_id').size()
    valid_groups = group_counts[group_counts >= MIN_LEN].index
    df = df[df['group_id'].isin(valid_groups)].reset_index(drop=True)
    print(f"Valid groups: {len(valid_groups)}")

    # 11. 결측치 채우기
    df = df.fillna(0)

    return df


def get_feature_columns():
    """학습에 사용할 피처 컬럼 목록"""
    features = [
        # 범주형 (CatBoost가 직접 처리)
        'group_id', '도매시장', '도매법인',
        'item', 'kind', 'location',
        # Lag 피처
        'price_lag1', 'price_lag3', 'price_lag7', 'price_lag14', 'price_lag30',
        # Rolling 통계
        'price_ma7', 'price_ma14', 'price_ma30',
        'price_std7', 'price_std14', 'price_std30',
        # Volume 관련
        'volume', 'volume_lag7', 'volume_ma14',
        # 날짜 피처
        'month', 'day_of_week', 'week_of_year', 'day_of_month',
        # 기상 피처
        'temp_avg', 'temp_max', 'rain', 'solar', 'hdd',
        # 변화율
        'price_change_1d', 'price_change_7d',
    ]
    return features


def get_cat_feature_names():
    """범주형 피처 이름 목록 (CatBoost용)"""
    return ['group_id', '도매시장', '도매법인', 'item', 'kind', 'location']


def train_and_predict(df):
    """CatBoost 학습 및 예측"""
    features = get_feature_columns()
    cat_features = get_cat_feature_names()
    target = 'price_per_kg'

    # 시간 기반 분할 (마지막 30일 검증)
    dates = sorted(df['date'].unique())
    cutoff_date = dates[-30]
    print(f"Train/Val split: cutoff = {cutoff_date}")

    train_mask = df['date'] < cutoff_date
    val_mask = df['date'] >= cutoff_date

    X_train = df.loc[train_mask, features]
    y_train = df.loc[train_mask, target]
    X_val = df.loc[val_mask, features]
    y_val = df.loc[val_mask, target]

    print(f"Train: {len(X_train):,} rows | Val: {len(X_val):,} rows")

    # CatBoost Pool (범주형 자동 처리)
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)

    # 모델 설정
    model = CatBoostRegressor(
        iterations=1000,
        depth=6,
        learning_rate=0.05,
        loss_function='MAE',
        eval_metric='MAE',
        random_seed=42,
        early_stopping_rounds=50,
        verbose=100,
        l2_leaf_reg=3.0,
        border_count=128,
        thread_count=-1,
    )

    # 학습
    print("CatBoost Training Start...")
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    print(f"학습 완료! Best iteration: {model.best_iteration_}")

    # 예측
    y_pred = model.predict(X_val)

    return y_val.values, y_pred, model


def evaluate(actual, predicted):
    """MAPE, MAE 계산"""
    actual = np.array(actual)
    predicted = np.array(predicted)

    # MAE
    mae = np.mean(np.abs(actual - predicted))

    # MAPE (실제값 > 0인 샘플만)
    mask = actual > 0
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

    return mape, mae


def run_pipeline():
    """전체 파이프라인 실행"""
    print("=" * 60)
    print("CatBoost Agricultural Price Prediction Pipeline")
    print("=" * 60)

    # 1. 데이터 로드
    print("Loading data...")
    df = load_data()
    print(f"   Original data: {len(df):,} rows")

    # 2. 전처리
    print("Preprocessing...")
    df = preprocess(df)

    # 3. 학습 & 예측
    actual, predicted, model = train_and_predict(df)

    # 4. 평가
    mape, mae = evaluate(actual, predicted)

    # 5. 결과 저장
    result_df = pd.DataFrame({
        'Actual': actual,
        'Predicted': predicted
    })

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, '..', '농넷_과거파일', 'catboost_result.csv')
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    # 피처 중요도 출력
    importance = model.get_feature_importance()
    feature_names = get_feature_columns()
    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    print("\n" + "=" * 60)
    print("[CatBoost Final Results]")
    print(f"   1. 평균 오차율 (MAPE): {mape:.2f}%")
    print(f"   2. 평균 오차 금액 (MAE): {mae:,.0f}원")
    print("=" * 60)
    print("\nTop 10 Feature Importance:")
    for _, row in fi_df.head(10).iterrows():
        print(f"   {row['feature']:25s} -> {row['importance']:.2f}%")
    print(f"\nResult saved: {output_path}")

    return mape, mae


if __name__ == "__main__":
    run_pipeline()
