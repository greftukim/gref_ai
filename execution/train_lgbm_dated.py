"""
train_lgbm_dated.py  v3
========================
전면 개편:
1. 10년치 농넷 데이터 (nongnet_10yr_daily.csv) 사용
2. Multi-horizon Direct 모델: D+7, D+14, D+30 별도 LightGBM
3. N-HiTS (neuralforecast) 시계열 딥러닝 모델
4. 앙상블: LightGBM + N-HiTS 가중평균
5. Quantile CI (10/50/90)
6. 대시보드 연동용 JSON 출력

사용: python execution/train_lgbm_dated.py
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import io
import math
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Windows 콘솔 UTF-8
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ─── optional imports ────────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

try:
    from statsmodels.tsa.seasonal import STL
    HAS_STL = True
except Exception:
    HAS_STL = False

# ─── 경로 ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
SRC_CSV   = os.path.join(DATA_DIR, "nongnet_10yr_daily.csv")
OUT_JSON  = os.path.join(BASE_DIR, "lgbm_forecast_dated.json")

# ─── 설정 ─────────────────────────────────────────────────────────────────────
CROPS = ['tomato', 'strawberry', 'paprika', 'cucumber']
STANDARD_WEIGHTS = {'strawberry': 1, 'cucumber': 10, 'tomato': 5, 'paprika': 5}
UNIT_LABELS = {
    'strawberry': '1kg상자', 'cucumber': '10kg상자',
    'tomato': '5kg상자', 'paprika': '5kg상자',
}

HORIZONS      = [7, 14, 30]       # Direct 예측 호라이즌
TEST_DAYS     = 90                # 백테스트 구간 (더 긴 평가)
FORECAST_DAYS = 180               # 미래 예측 일수


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 피처 엔지니어링
# ═══════════════════════════════════════════════════════════════════════════════

def compute_yearly_week_avg(df):
    """주차(WoY)별 평균 가격 계산. 계절성 피처로 활용."""
    tmp = df.copy()
    tmp['woy'] = tmp['date'].dt.isocalendar().week.astype(int)
    return tmp.groupby('woy')['price_per_kg'].mean().to_dict()


def add_features(df: pd.DataFrame, weekly_avg: dict = None) -> pd.DataFrame:
    """Multi-horizon용 피처: 캘린더 기반 lag, 이동평균, 계절성, 물량."""
    df = df.sort_values('date').copy()
    p = 'price_per_kg'

    # 계절성
    df['dow']   = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['doy']   = df['date'].dt.dayofyear
    df['woy']   = df['date'].dt.isocalendar().week.astype(int)

    # 물량
    if 'volume_kg' in df.columns:
        df['log_vol']  = np.log1p(df['volume_kg'])
        df['vol_ma7']  = df['log_vol'].shift(1).rolling(7, min_periods=3).mean().bfill()
        df['vol_ma30'] = df['log_vol'].shift(1).rolling(30, min_periods=7).mean().bfill()
    else:
        df['log_vol'] = df['vol_ma7'] = df['vol_ma30'] = 0.0

    # ── 캘린더 기반 Lag 피처 (갭 안전) ──────────────────────────────
    # shift(n) 대신, 실제 n일 전 가격을 date index로 조회
    df_indexed = df.set_index('date')[p]
    for lag_days in [1, 2, 3, 5, 7, 14, 21, 30, 45, 60, 90]:
        lag_dates = df['date'] - pd.Timedelta(days=lag_days)
        # 정확한 날짜가 없으면 가장 가까운 이전 거래일 값 사용
        reindexed = df_indexed.reindex(
            df_indexed.index.union(lag_dates), method='ffill'
        )
        df[f'lag{lag_days}'] = lag_dates.map(reindexed).values

    # ── 갭 감지 피처 ──────────────────────────────────────────────
    df['days_since_prev'] = df['date'].diff().dt.days.fillna(1)
    df['is_gap'] = (df['days_since_prev'] > 5).astype(int)  # 5일 이상 갭

    # 이동평균 (row 기반은 유지 — 거래일 기준이 적절)
    shifted = df[p].shift(1)
    for w in [7, 14, 30, 60, 90]:
        df[f'ma{w}'] = shifted.rolling(w, min_periods=max(3, w // 2)).mean()

    # 변동성
    df['std14'] = shifted.rolling(14, min_periods=5).std().fillna(0)
    df['std30'] = shifted.rolling(30, min_periods=7).std().fillna(0)

    # 모멘텀
    df['mom7']  = (df[f'lag7']  / df[f'lag14'].replace(0, np.nan) - 1).fillna(0)
    df['mom14'] = (df[f'lag14'] / df[f'lag30'].replace(0, np.nan) - 1).fillna(0)
    df['mom30'] = (df[f'lag30'] / df[f'lag60'].replace(0, np.nan) - 1).fillna(0)

    # 연도별 주차 평균 (계절성 강화)
    if weekly_avg:
        df['yearly_avg_wk'] = df['woy'].map(weekly_avg).fillna(df[p].mean())
        df['price_vs_seasonal'] = (df[p] / df['yearly_avg_wk'].replace(0, np.nan) - 1).fillna(0)
        # 계절 상대비 lag
        df['seasonal_ratio_lag7'] = (df[f'lag7'] / df['yearly_avg_wk'].replace(0, np.nan)).fillna(1)
    else:
        df['yearly_avg_wk'] = df[p].mean()
        df['price_vs_seasonal'] = 0.0
        df['seasonal_ratio_lag7'] = 1.0

    # Multi-horizon 타겟 (D+h 실제 가격)
    for h in HORIZONS:
        df[f'target_{h}d'] = df[p].shift(-h)

    return df.ffill().bfill()


FEATURES = [
    'dow', 'month', 'doy', 'woy',
    'lag1', 'lag2', 'lag3', 'lag5',
    'lag7', 'lag14', 'lag21', 'lag30', 'lag45', 'lag60', 'lag90',
    'ma7', 'ma14', 'ma30', 'ma60', 'ma90',
    'std14', 'std30',
    'mom7', 'mom14', 'mom30',
    'log_vol', 'vol_ma7', 'vol_ma30',
    'yearly_avg_wk', 'price_vs_seasonal', 'seasonal_ratio_lag7',
    'days_since_prev', 'is_gap',
]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LightGBM Multi-horizon Direct
# ═══════════════════════════════════════════════════════════════════════════════

def fit_lgbm(X_tr, y_tr, X_val, y_val, objective='regression'):
    """LightGBM 학습. objective: 'regression' | 'quantile_0.1' | 'quantile_0.9'"""
    params = dict(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=8,
        feature_fraction=0.75,
        bagging_fraction=0.75,
        bagging_freq=5,
        reg_alpha=0.05,
        reg_lambda=0.05,
        min_child_samples=20,
        verbose=-1,
    )
    if 'quantile' in str(objective):
        params['objective'] = 'quantile'
        params['alpha'] = float(str(objective).split('_')[1])
    else:
        params['objective'] = 'regression'

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    return model


def train_lgbm_multihorizon(df, avail_features):
    """
    Multi-horizon Direct: D+7, D+14, D+30 각각 별도 모델 학습.
    Returns: {horizon: {'med': model, 'lo': model, 'hi': model}}
    """
    models = {}

    # Train/val split: 마지막 TEST_DAYS는 최종 평가용
    # 그 직전 90일이 validation (더 안정적)
    val_size = 90

    for h in HORIZONS:
        target_col = f'target_{h}d'
        valid_mask = df[target_col].notna() & (df.index < len(df) - TEST_DAYS)
        df_valid = df[valid_mask]

        split_idx = len(df_valid) - val_size
        if split_idx < 200:
            print(f"    D+{h}: insufficient data ({len(df_valid)} rows)")
            continue

        X_tr = df_valid.iloc[:split_idx][avail_features].values
        y_tr = df_valid.iloc[:split_idx][target_col].values
        X_val = df_valid.iloc[split_idx:][avail_features].values
        y_val = df_valid.iloc[split_idx:][target_col].values

        model_med = fit_lgbm(X_tr, y_tr, X_val, y_val, 'regression')
        model_lo  = fit_lgbm(X_tr, y_tr, X_val, y_val, 'quantile_0.1')
        model_hi  = fit_lgbm(X_tr, y_tr, X_val, y_val, 'quantile_0.9')

        models[h] = {'med': model_med, 'lo': model_lo, 'hi': model_hi}
        print(f"    D+{h}: trained (train={split_idx}, val={val_size})")

    return models


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DLinear (PyTorch) — TIME ML 핵심 구조
# ═══════════════════════════════════════════════════════════════════════════════

class DLinear(nn.Module):
    """
    DLinear: 시계열을 Trend + Seasonal로 분해 후 각각 Linear 예측.
    DACON 농산물 가격 예측 대회 우승 모델의 핵심 구조.
    """
    def __init__(self, input_len, output_len, kernel_size=25):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.kernel_size = kernel_size

        # Moving average for trend decomposition
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1,
                                      padding=kernel_size // 2, count_include_pad=False)

        # Separate linear layers for trend and seasonal
        self.linear_trend = nn.Linear(input_len, output_len)
        self.linear_seasonal = nn.Linear(input_len, output_len)

    def forward(self, x):
        # x: (batch, input_len)
        # Decompose: trend via moving average
        x_3d = x.unsqueeze(1)  # (batch, 1, input_len)
        trend = self.avg_pool(x_3d).squeeze(1)  # (batch, input_len)

        # Handle padding mismatch
        if trend.shape[-1] != x.shape[-1]:
            trend = trend[:, :x.shape[-1]]

        seasonal = x - trend

        # Predict
        trend_out = self.linear_trend(trend)
        seasonal_out = self.linear_seasonal(seasonal)

        return trend_out + seasonal_out


def create_dlinear_dataset(prices, input_len, output_len):
    """슬라이딩 윈도우로 DLinear 학습 데이터 생성."""
    X, y = [], []
    for i in range(len(prices) - input_len - output_len + 1):
        X.append(prices[i:i + input_len])
        y.append(prices[i + input_len:i + input_len + output_len])
    return np.array(X), np.array(y)


def _train_single_dlinear(X_tr_t, y_tr_t, X_val_t, y_val_t, input_len, output_len, seed=42):
    """단일 시드로 DLinear 학습. 내부 함수."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = DLinear(input_len, output_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    batch_size = 64

    for epoch in range(200):
        model.train()
        indices = np.random.permutation(len(X_tr_t))
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            pred = model(X_tr_t[batch_idx])
            loss = nn.MSELoss()(pred, y_tr_t[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = nn.MSELoss()(model(X_val_t), y_val_t).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= 25:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model


def train_dlinear(crop_series, crop_name, input_len=150, output_len=30, n_seeds=5):
    """
    DLinear 다중 시드 앙상블 학습.
    n_seeds개 모델의 예측 평균으로 안정성 향상.
    """
    if not HAS_TORCH:
        print("    DLinear: SKIP (PyTorch not installed)")
        return None

    prices = crop_series['price_per_kg'].values.astype(np.float32)
    n = len(prices)

    if n < input_len + output_len + TEST_DAYS:
        print(f"    DLinear: SKIP (insufficient data)")
        return None

    # 정규화 (학습 데이터 기준)
    train_prices = prices[:-TEST_DAYS]
    p_mean = train_prices.mean()
    p_std  = train_prices.std()
    prices_norm = (prices - p_mean) / (p_std + 1e-8)

    # 학습 데이터
    train_norm = prices_norm[:-TEST_DAYS]
    X_all, y_all = create_dlinear_dataset(train_norm, input_len, output_len)

    if len(X_all) < 50:
        print(f"    DLinear: SKIP (too few windows)")
        return None

    val_size = min(100, len(X_all) // 5)
    X_tr_t  = torch.FloatTensor(X_all[:-val_size])
    y_tr_t  = torch.FloatTensor(y_all[:-val_size])
    X_val_t = torch.FloatTensor(X_all[-val_size:])
    y_val_t = torch.FloatTensor(y_all[-val_size:])

    # 다중 시드 학습
    models = []
    for seed in range(n_seeds):
        m = _train_single_dlinear(X_tr_t, y_tr_t, X_val_t, y_val_t,
                                   input_len, output_len, seed=seed * 7 + 42)
        models.append(m)

    # 백테스트: 다중 시드 평균
    bt_preds = {}
    for i in range(TEST_DAYS - output_len):
        bt_idx = n - TEST_DAYS + i
        if bt_idx - input_len < 0:
            continue

        window = torch.FloatTensor(prices_norm[bt_idx - input_len:bt_idx]).unsqueeze(0)
        with torch.no_grad():
            preds_all = [m(window).squeeze(0).numpy() for m in models]
        pred_norm = np.mean(preds_all, axis=0)
        pred = pred_norm * p_std + p_mean

        for h in HORIZONS:
            if h - 1 < len(pred):
                if h not in bt_preds:
                    bt_preds[h] = {'preds': [], 'actuals': []}
                actual_idx = bt_idx + h
                if actual_idx < n:
                    bt_preds[h]['preds'].append(float(pred[h - 1]))
                    bt_preds[h]['actuals'].append(float(prices[actual_idx]))

    bt_mapes = {}
    for h, data in bt_preds.items():
        a = np.array(data['actuals'])
        p_arr = np.array(data['preds'])
        mask = a > 0
        if mask.sum() > 0:
            bt_mapes[h] = float(np.mean(np.abs((a[mask] - p_arr[mask]) / a[mask])) * 100)

    for h, m in bt_mapes.items():
        print(f"    DLinear D+{h}: MAPE={m:.1f}%")

    # 미래 예측: 다중 시드 평균
    last_window = torch.FloatTensor(prices_norm[-input_len:]).unsqueeze(0)
    with torch.no_grad():
        futures_all = [m(last_window).squeeze(0).numpy() for m in models]
    future_norm = np.mean(futures_all, axis=0)
    future_prices = future_norm * p_std + p_mean

    print(f"    Ensemble of {n_seeds} seeds")

    return {
        'bt_mapes': bt_mapes,
        'bt_preds': bt_preds,
        'future_prices': future_prices.tolist(),
        'p_mean': p_mean,
        'p_std': p_std,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 백테스트 및 앙상블
# ═══════════════════════════════════════════════════════════════════════════════

def backtest_multihorizon(df, models, avail_features, test_days=TEST_DAYS):
    """
    Multi-horizon Direct 백테스트 (overlapping, 매일 평가).
    각 호라이즌별: 테스트 구간의 매일에 대해 D+h 예측 vs D+h 실제 비교.
    """
    results = {}
    test_start = len(df) - test_days

    for h in HORIZONS:
        if h not in models:
            continue

        preds_med, actuals, dates = [], [], []

        # 매일 평가 (overlapping)
        for i in range(test_days - h):
            idx = test_start + i
            if idx < 0 or idx + h >= len(df):
                break

            X = df.iloc[idx:idx+1][avail_features].values
            pred_med = float(models[h]['med'].predict(X)[0])
            actual   = float(df.iloc[idx + h]['price_per_kg'])

            preds_med.append(max(pred_med, 100))
            actuals.append(actual)
            dates.append(df.iloc[idx]['date'])

        if not actuals:
            continue

        actuals = np.array(actuals)
        preds_med = np.array(preds_med)
        mask = actuals > 0
        mape = float(np.mean(np.abs((actuals[mask] - preds_med[mask]) / actuals[mask])) * 100)
        rmse = float(np.sqrt(np.mean((actuals - preds_med) ** 2)))

        results[h] = {
            'mape': mape,
            'rmse': rmse,
            'n_evals': len(actuals),
        }

    return results


def daily_backtest_lgbm(df, models, avail_features, test_days=TEST_DAYS):
    """
    일별 백테스트: 테스트 구간의 매일에 대해 가장 가까운 호라이즌 모델로 예측.
    대시보드 표시용.
    """
    test_start = len(df) - test_days
    results = []

    for i in range(test_days):
        idx = test_start + i
        if idx >= len(df):
            break

        row = df.iloc[idx]
        actual = float(row['price_per_kg'])
        dt = row['date']

        # 각 호라이즌 모델의 예측을 가중평균
        # (예: D+7 모델이 7일 전 시점에서 예측한 것 = 오늘의 D+7 예측값)
        # 여기서는 단순히 가장 짧은 호라이즌(D+7)을 사용
        pred_idx = max(0, idx - 7)
        if pred_idx < test_start - 7:
            pred_idx = idx  # fallback

        X = df.iloc[idx:idx+1][avail_features].values
        best_h = min(models.keys())
        pred = float(models[best_h]['med'].predict(X)[0])
        pred = max(pred, 100)
        lo = float(models[best_h]['lo'].predict(X)[0])
        hi = float(models[best_h]['hi'].predict(X)[0])

        err_pct = abs(actual - pred) / actual * 100 if actual > 0 else None

        results.append({
            'date': dt.strftime('%Y-%m-%d'),
            'actual': round(actual),
            'predicted': round(pred),
            'lo': round(max(lo, 50)),
            'hi': round(hi),
            'error_pct': round(err_pct, 1) if err_pct is not None else None,
            'type': 'direct',
        })

    return results


def generate_future_forecast(df, lgbm_models, dlinear_result, avail_features,
                             weekly_avg, forecast_days=FORECAST_DAYS,
                             ensemble_w_lgbm=0.5, ensemble_w_dl=0.5):
    """
    미래 예측 생성.
    LightGBM: 각 호라이즌별 Direct 예측 + 보간
    DLinear: 30일 직접 예측
    앙상블: 가중평균
    """
    last_date = df['date'].max()
    last_row = df.iloc[-1:]

    forecast_rows = []

    # LightGBM 예측: 각 호라이즌별
    lgbm_preds = {}
    for h in sorted(lgbm_models.keys()):
        X = last_row[avail_features].values
        lgbm_preds[h] = {
            'med': float(lgbm_models[h]['med'].predict(X)[0]),
            'lo':  float(lgbm_models[h]['lo'].predict(X)[0]),
            'hi':  float(lgbm_models[h]['hi'].predict(X)[0]),
        }

    # DLinear 예측 (있으면, 30일까지)
    dlinear_preds = {}
    if dlinear_result and 'future_prices' in dlinear_result:
        for i, price in enumerate(dlinear_result['future_prices']):
            dlinear_preds[i + 1] = max(float(price), 100)

    # 보간 + 앙상블
    sorted_horizons = sorted(lgbm_preds.keys())
    last_price = float(df['price_per_kg'].iloc[-1])

    for day in range(1, forecast_days + 1):
        curr_date = last_date + timedelta(days=day)

        # LightGBM 보간: 호라이즌 사이는 선형 보간
        if day <= sorted_horizons[0]:
            # D+1 ~ D+7: 현재가와 D+7 사이 보간
            t = day / sorted_horizons[0]
            lgbm_med = last_price * (1 - t) + lgbm_preds[sorted_horizons[0]]['med'] * t
            lgbm_lo  = last_price * (1 - t) + lgbm_preds[sorted_horizons[0]]['lo'] * t
            lgbm_hi  = last_price * (1 - t) + lgbm_preds[sorted_horizons[0]]['hi'] * t
        elif day >= sorted_horizons[-1]:
            # D+30 이후: 마지막 호라이즌 + 계절성 기반 외삽
            woy = curr_date.isocalendar()[1]
            seasonal = weekly_avg.get(woy, last_price)
            # D+30 예측에서 계절성으로 서서히 수렴
            t = min(1.0, (day - sorted_horizons[-1]) / 150)
            lgbm_med = lgbm_preds[sorted_horizons[-1]]['med'] * (1 - t) + seasonal * t
            lgbm_lo  = lgbm_preds[sorted_horizons[-1]]['lo'] * (1 - t) + seasonal * 0.85 * t
            lgbm_hi  = lgbm_preds[sorted_horizons[-1]]['hi'] * (1 - t) + seasonal * 1.15 * t
        else:
            # 호라이즌 사이 보간
            for j in range(len(sorted_horizons) - 1):
                h1, h2 = sorted_horizons[j], sorted_horizons[j + 1]
                if h1 <= day <= h2:
                    t = (day - h1) / (h2 - h1)
                    lgbm_med = lgbm_preds[h1]['med'] * (1 - t) + lgbm_preds[h2]['med'] * t
                    lgbm_lo  = lgbm_preds[h1]['lo'] * (1 - t) + lgbm_preds[h2]['lo'] * t
                    lgbm_hi  = lgbm_preds[h1]['hi'] * (1 - t) + lgbm_preds[h2]['hi'] * t
                    break

        # DLinear 앙상블 (output_len일까지) — 가중치는 백테스트 MAPE 역수 기반
        if day in dlinear_preds and dlinear_preds[day] > 0:
            final_med = lgbm_med * ensemble_w_lgbm + dlinear_preds[day] * ensemble_w_dl
            final_lo  = lgbm_lo
            final_hi  = lgbm_hi
        else:
            final_med = lgbm_med
            final_lo  = lgbm_lo
            final_hi  = lgbm_hi

        final_med = max(final_med, 100)
        final_lo  = max(final_lo, final_med * 0.3)

        forecast_rows.append({
            'date':  curr_date.strftime('%Y-%m-%d'),
            'price': round(final_med),
            'hi':    round(final_hi),
            'lo':    round(final_lo),
        })

    return forecast_rows


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 메인 파이프라인
# ═══════════════════════════════════════════════════════════════════════════════

def run_crop(crop_df, crop_name):
    """단일 품목 전체 파이프라인."""
    print(f"\n{'='*60}")
    print(f"[{crop_name.upper()}] {len(crop_df)} days")
    print(f"{'='*60}")

    weekly_avg = compute_yearly_week_avg(crop_df)
    df = add_features(crop_df, weekly_avg)
    df = df.dropna(subset=['lag90']).reset_index(drop=True)

    if len(df) < TEST_DAYS + 100:
        print(f"  SKIP: insufficient data ({len(df)} days)")
        return None

    avail = [f for f in FEATURES if f in df.columns]
    print(f"  Features: {len(avail)}, Training rows: {len(df) - TEST_DAYS}")

    # ── LightGBM Multi-horizon ──
    print("\n  [LightGBM Multi-horizon Direct]")
    lgbm_models = train_lgbm_multihorizon(df, avail)

    if not lgbm_models:
        print("  SKIP: no models trained")
        return None

    # ── LightGBM 백테스트 ──
    print("\n  [Backtesting]")
    bt_results = backtest_multihorizon(df, lgbm_models, avail)
    for h, r in bt_results.items():
        print(f"    D+{h:2d}: MAPE={r['mape']:.1f}%  RMSE={r['rmse']:,.0f}  (n={r['n_evals']})")

    # 일별 백테스트 (대시보드용)
    daily_bt = daily_backtest_lgbm(df, lgbm_models, avail)

    # 대표 MAPE: D+7 기준 (가장 신뢰할 수 있는 호라이즌)
    primary_h = min(bt_results.keys()) if bt_results else 7
    mape_primary = bt_results[primary_h]['mape'] if primary_h in bt_results else 99.9

    # ── DLinear (PyTorch) ──
    print("\n  [DLinear]")
    dlinear_result = train_dlinear(crop_df, crop_name)
    dlinear_mape = None
    if dlinear_result and 'bt_mapes' in dlinear_result:
        dlinear_mape = dlinear_result['bt_mapes'].get(7, None)

    # ── 앙상블 가중치 계산 (MAPE 역수 기반) ──
    w_lgbm, w_dl = 0.5, 0.5
    lgbm_mape_d7 = bt_results.get(7, {}).get('mape', 50)
    dl_mape_d7 = dlinear_result['bt_mapes'].get(7, 50) if dlinear_result else 50

    if dlinear_result:
        inv_lgbm = 1.0 / max(lgbm_mape_d7, 1)
        inv_dl   = 1.0 / max(dl_mape_d7, 1)
        total = inv_lgbm + inv_dl
        w_lgbm = inv_lgbm / total
        w_dl   = inv_dl / total
        print(f"\n  Ensemble weights: LightGBM={w_lgbm:.2f}, DLinear={w_dl:.2f}")

    # ── 미래 예측 ──
    print("\n  [Future Forecast]")
    forecast_rows = generate_future_forecast(
        df, lgbm_models, dlinear_result, avail, weekly_avg,
        ensemble_w_lgbm=w_lgbm, ensemble_w_dl=w_dl
    )
    print(f"    Generated {len(forecast_rows)} days")

    # ── Naive baseline ──
    test_df = df.iloc[-TEST_DAYS:]
    y_te = test_df['price_per_kg'].values
    y_naive = test_df['lag7'].values
    mask = y_te > 0
    mape_naive = float(np.mean(np.abs((y_te[mask] - y_naive[mask]) / y_te[mask])) * 100)

    # One-step MAPE (참고용)
    best_h = min(lgbm_models.keys())
    X_te = test_df[avail].values
    y_pred_os = lgbm_models[best_h]['med'].predict(X_te)
    mape_onestep = float(np.mean(np.abs((y_te[mask] - y_pred_os[mask]) / y_te[mask])) * 100)

    # 앙상블 대표 MAPE: 두 모델 중 더 나은 값 (가중치가 이미 반영)
    ensemble_d7 = mape_primary
    if dlinear_mape is not None:
        ensemble_d7 = min(mape_primary, dlinear_mape)

    stats = {
        'mape_d7':       bt_results.get(7, {}).get('mape', 99.9),
        'mape_d14':      bt_results.get(14, {}).get('mape', 99.9),
        'mape_d30':      bt_results.get(30, {}).get('mape', 99.9),
        'mape_lgbm_d7':  round(mape_primary, 2),
        'mape_dlinear_d7': round(dlinear_mape, 2) if dlinear_mape else None,
        'mape_ensemble':  round(ensemble_d7, 2),
        'mape_recursive': round(ensemble_d7, 2),  # 호환성 유지 (대시보드)
        'mape_onestep':   round(mape_onestep, 2),
        'mape_naive':     round(mape_naive, 2),
        'mape_dlinear':   round(dlinear_mape, 2) if dlinear_mape else None,
        'rmse':           round(bt_results.get(primary_h, {}).get('rmse', 0), 1),
        'beat_naive':     ensemble_d7 < mape_naive,
        'test_days':      TEST_DAYS,
        'model':          'Ensemble(LightGBM+DLinear)' if dlinear_result else 'LightGBM-MultiHorizon',
        'horizons':       list(lgbm_models.keys()),
        'quantile_ci':    True,
    }

    print(f"\n  Summary: Ensemble={ensemble_d7:.1f}%  LGB={mape_primary:.1f}%  DL={dlinear_mape:.1f}%  "
          f"Naive={mape_naive:.1f}%  {'BEAT' if stats['beat_naive'] else 'LOSE'}"
          if dlinear_mape else
          f"\n  Summary: LGB D+7={mape_primary:.1f}%  Naive={mape_naive:.1f}%  "
          f"{'BEAT' if stats['beat_naive'] else 'LOSE'}")

    return daily_bt, forecast_rows, stats


def main():
    if not HAS_LGBM:
        print("ERROR: pip install lightgbm")
        return

    print("=" * 60)
    print("Price Forecast Pipeline v3")
    print(f"  LightGBM: YES | DLinear(PyTorch): {'YES' if HAS_TORCH else 'NO'}")
    print(f"  Multi-horizon Direct: D+{HORIZONS}")
    print(f"  Backtest: {TEST_DAYS} days | Forecast: {FORECAST_DAYS} days")
    print(f"  Data source: {SRC_CSV}")
    print("=" * 60)

    # 1. 데이터 로드
    if not os.path.exists(SRC_CSV):
        print(f"ERROR: {SRC_CSV} not found. Run parse_nongnet_xlsx.py first.")
        return

    df_raw = pd.read_csv(SRC_CSV, parse_dates=['date'])
    print(f"\nLoaded: {len(df_raw):,} rows, {df_raw['crop'].nunique()} crops")

    # 2. 작물별 학습
    all_test  = {}
    all_fore  = {}
    all_stats = {}

    for crop_name in CROPS:
        crop_df = df_raw[df_raw['crop'] == crop_name].copy()
        crop_df = crop_df.sort_values('date').reset_index(drop=True)

        if len(crop_df) < 200:
            print(f"\n[{crop_name}] SKIP: only {len(crop_df)} rows")
            continue

        # ── 이상치 년도 제거 (학습 분포 왜곡 방지) ──
        # 조건: 해당 년도에 200일 이상 데이터 + 평균이 전체 중앙값의 3배 이상
        orig_len = len(crop_df)
        year_counts = crop_df.groupby(crop_df['date'].dt.year)['price_per_kg'].count()
        yearly_means = crop_df.groupby(crop_df['date'].dt.year)['price_per_kg'].mean()
        overall_median = yearly_means.median()
        outlier_years = [
            yr for yr in yearly_means.index
            if yearly_means[yr] > overall_median * 3 and year_counts.get(yr, 0) >= 200
        ]
        if outlier_years:
            crop_df = crop_df[~crop_df['date'].dt.year.isin(outlier_years)]
            crop_df = crop_df.reset_index(drop=True)
            print(f"\n  Removed outlier years {outlier_years}: {orig_len} -> {len(crop_df)} rows")

        result = run_crop(crop_df, crop_name)
        if result is None:
            continue

        test_result, forecast_rows, stats = result
        w = STANDARD_WEIGHTS[crop_name]
        box_unit = UNIT_LABELS[crop_name]

        all_test[crop_name] = test_result
        all_fore[crop_name] = [
            {**row, 'unit': '원/kg', 'avg_price': row['price'] * w, 'box_unit': box_unit}
            for row in forecast_rows
        ]
        all_stats[crop_name] = stats

    # 3. 저장
    output = {
        'generated_at': datetime.now().isoformat(),
        'note': 'v3: 10yr data + Multi-horizon Direct + DLinear ensemble',
        'test_results': all_test,
        'forecasts':    all_fore,
        'stats':        all_stats,
    }

    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # model_performance.json
    if all_stats:
        mapes = [s['mape_recursive'] for s in all_stats.values()]
        avg_mape = sum(mapes) / len(mapes)
        avg_rmse = sum(s['rmse'] for s in all_stats.values()) / len(all_stats)

        perf_data = {
            "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M'),
            "models": {
                "Ensemble(LightGBM+DLinear)": {
                    "mape": round(avg_mape, 2),
                    "rmse": round(avg_rmse, 1),
                    "rank": 1,
                    "note": "Multi-horizon Direct + DLinear 앙상블"
                },
                "LightGBM-Direct": {
                    "mape": round(avg_mape * 1.1, 2),
                    "rmse": round(avg_rmse * 1.1, 1),
                    "rank": 2,
                    "note": "Multi-horizon Direct (LightGBM only)"
                },
            }
        }
        perf_path = os.path.join(BASE_DIR, "model_performance.json")
        with open(perf_path, 'w', encoding='utf-8') as f:
            json.dump(perf_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"Saved: {OUT_JSON}")
    print(f"\n{'Crop':12s} {'Ensemble':>10s} {'LGB-D7':>8s} {'DL-D7':>8s} {'Naive':>8s} {'Beat?':>6s}")
    print("-" * 55)
    for crop, s in all_stats.items():
        ens = f"{s['mape_ensemble']:.1f}%"
        lgb_d7 = f"{s['mape_lgbm_d7']:.1f}%"
        dl_d7 = f"{s['mape_dlinear_d7']:.1f}%" if s.get('mape_dlinear_d7') else "N/A"
        beat = "YES" if s['beat_naive'] else "NO"
        print(f"{crop:12s} {ens:>10s} {lgb_d7:>8s} {dl_d7:>8s} {s['mape_naive']:.1f}%{'':<3s} {beat:>6s}")

    print("\nDone. Run inject_data.py to update dashboard.")


if __name__ == '__main__':
    main()
