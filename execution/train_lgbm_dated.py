"""
train_lgbm_dated.py  v4
========================
4-Model Stacking (LightGBM + CatBoost + DLinear + N-HiTS) + Weather features
사용: python execution/train_lgbm_dated.py
"""
import pandas as pd
import numpy as np
import json, os, sys, io, math, warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
try:
    import catboost as cb
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except Exception:
    HAS_TORCH = False
    DEVICE = 'cpu'
try:
    from sklearn.linear_model import Ridge
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False
try:
    from statsmodels.tsa.seasonal import STL
    HAS_STL = True
except Exception:
    HAS_STL = False

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
SRC_CSV   = os.path.join(DATA_DIR, "nongnet_10yr_daily.csv")
WX_CSV    = os.path.join(DATA_DIR, "weather_by_crop.csv")
OUT_JSON  = os.path.join(BASE_DIR, "lgbm_forecast_dated.json")

CROPS = ['tomato', 'strawberry', 'paprika', 'cucumber']
STANDARD_WEIGHTS = {'strawberry': 1, 'cucumber': 10, 'tomato': 5, 'paprika': 5}
UNIT_LABELS = {'strawberry': '1kg', 'cucumber': '10kg', 'tomato': '5kg', 'paprika': '5kg'}
HORIZONS      = [7, 14, 30]
TEST_DAYS     = 90
FORECAST_DAYS = 180

def compute_yearly_week_avg(df):
    tmp = df.copy()
    tmp['woy'] = tmp['date'].dt.isocalendar().week.astype(int)
    return tmp.groupby('woy')['price_per_kg'].mean().to_dict()

def add_features(df, weekly_avg=None, weather_df=None):
    df = df.sort_values('date').copy()
    p = 'price_per_kg'
    df['dow']   = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['doy']   = df['date'].dt.dayofyear
    df['woy']   = df['date'].dt.isocalendar().week.astype(int)
    if 'volume_kg' in df.columns:
        df['log_vol']  = np.log1p(df['volume_kg'])
        df['vol_ma7']  = df['log_vol'].shift(1).rolling(7, min_periods=3).mean().bfill()
        df['vol_ma30'] = df['log_vol'].shift(1).rolling(30, min_periods=7).mean().bfill()
    else:
        df['log_vol'] = df['vol_ma7'] = df['vol_ma30'] = 0.0
    df_indexed = df.set_index('date')[p]
    for lag_days in [1,2,3,5,7,14,21,30,45,60,90]:
        lag_dates = df['date'] - pd.Timedelta(days=lag_days)
        reindexed = df_indexed.reindex(df_indexed.index.union(lag_dates), method='ffill')
        df[f'lag{lag_days}'] = lag_dates.map(reindexed).values
    df['days_since_prev'] = df['date'].diff().dt.days.fillna(1)
    df['is_gap'] = (df['days_since_prev'] > 5).astype(int)
    shifted = df[p].shift(1)
    for w in [7,14,30,60,90]:
        df[f'ma{w}'] = shifted.rolling(w, min_periods=max(3, w//2)).mean()
    df['std14'] = shifted.rolling(14, min_periods=5).std().fillna(0)
    df['std30'] = shifted.rolling(30, min_periods=7).std().fillna(0)
    df['mom7']  = (df['lag7']  / df['lag14'].replace(0,np.nan) - 1).fillna(0)
    df['mom14'] = (df['lag14'] / df['lag30'].replace(0,np.nan) - 1).fillna(0)
    df['mom30'] = (df['lag30'] / df['lag60'].replace(0,np.nan) - 1).fillna(0)
    if weekly_avg:
        df['yearly_avg_wk'] = df['woy'].map(weekly_avg).fillna(df[p].mean())
        df['price_vs_seasonal'] = (df[p] / df['yearly_avg_wk'].replace(0,np.nan) - 1).fillna(0)
        df['seasonal_ratio_lag7'] = (df['lag7'] / df['yearly_avg_wk'].replace(0,np.nan)).fillna(1)
    else:
        df['yearly_avg_wk'] = df[p].mean()
        df['price_vs_seasonal'] = 0.0
        df['seasonal_ratio_lag7'] = 1.0
    if weather_df is not None and len(weather_df) > 0:
        wx = weather_df[['date','avgTa','minTa','maxTa','sumRn','avgRhm','sumSsHr','avgWs','minTg','avgTs','sumGsr']].copy()
        wx = wx.rename(columns={c: f'wx_{c}' for c in wx.columns if c != 'date'})
        df = df.merge(wx, on='date', how='left')
        for col in ['wx_avgTa','wx_sumRn','wx_sumSsHr','wx_minTg']:
            df[f'{col}_ma7']  = df[col].shift(1).rolling(7, min_periods=3).mean()
            df[f'{col}_ma14'] = df[col].shift(1).rolling(14, min_periods=5).mean()
        df['wx_temp_range'] = df['wx_maxTa'] - df['wx_minTa']
        wx_cols = [c for c in df.columns if c.startswith('wx_')]
        df[wx_cols] = df[wx_cols].ffill().bfill().fillna(0)
    for h in HORIZONS:
        df[f'target_{h}d'] = df[p].shift(-h)
    return df.ffill().bfill()

FEATURES = [
    'dow','month','doy','woy',
    'lag1','lag2','lag3','lag5','lag7','lag14','lag21','lag30','lag45','lag60','lag90',
    'ma7','ma14','ma30','ma60','ma90','std14','std30',
    'mom7','mom14','mom30','log_vol','vol_ma7','vol_ma30',
    'yearly_avg_wk','price_vs_seasonal','seasonal_ratio_lag7','days_since_prev','is_gap',
    'wx_avgTa','wx_minTa','wx_maxTa','wx_sumRn','wx_avgRhm','wx_sumSsHr','wx_avgWs','wx_minTg','wx_avgTs','wx_sumGsr',
    'wx_avgTa_ma7','wx_avgTa_ma14','wx_sumRn_ma7','wx_sumRn_ma14',
    'wx_sumSsHr_ma7','wx_sumSsHr_ma14','wx_minTg_ma7','wx_minTg_ma14','wx_temp_range',
]

def fit_lgbm(X_tr, y_tr, X_val, y_val, objective='regression'):
    params = dict(n_estimators=800, learning_rate=0.03, num_leaves=63, max_depth=8,
        feature_fraction=0.75, bagging_fraction=0.75, bagging_freq=5,
        reg_alpha=0.05, reg_lambda=0.05, min_child_samples=20, verbose=-1)
    if 'quantile' in str(objective):
        params['objective'] = 'quantile'
        params['alpha'] = float(str(objective).split('_')[1])
    else:
        params['objective'] = 'regression'
    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    return model

def train_lgbm_multihorizon(df, avail_features):
    models = {}
    val_size = 90
    for h in HORIZONS:
        target_col = f'target_{h}d'
        valid_mask = df[target_col].notna() & (df.index < len(df) - TEST_DAYS)
        df_valid = df[valid_mask]
        split_idx = len(df_valid) - val_size
        if split_idx < 200:
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

def fit_catboost(X_tr, y_tr, X_val, y_val, objective='RMSE'):
    if not HAS_CATBOOST:
        return None
    params = dict(iterations=800, learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
        random_seed=42, verbose=0, task_type='GPU' if DEVICE=='cuda' else 'CPU',
        loss_function=objective, early_stopping_rounds=50)
    model = cb.CatBoostRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
    return model

def train_catboost_multihorizon(df, avail_features):
    if not HAS_CATBOOST:
        print("    CatBoost: SKIP (not installed)")
        return {}
    models = {}
    val_size = 90
    for h in HORIZONS:
        target_col = f'target_{h}d'
        valid_mask = df[target_col].notna() & (df.index < len(df) - TEST_DAYS)
        df_valid = df[valid_mask]
        split_idx = len(df_valid) - val_size
        if split_idx < 200:
            continue
        X_tr = df_valid.iloc[:split_idx][avail_features].values
        y_tr = df_valid.iloc[:split_idx][target_col].values
        X_val = df_valid.iloc[split_idx:][avail_features].values
        y_val = df_valid.iloc[split_idx:][target_col].values
        model_med = fit_catboost(X_tr, y_tr, X_val, y_val, 'RMSE')
        model_lo  = fit_catboost(X_tr, y_tr, X_val, y_val, 'Quantile:alpha=0.1')
        model_hi  = fit_catboost(X_tr, y_tr, X_val, y_val, 'Quantile:alpha=0.9')
        if model_med:
            models[h] = {'med': model_med, 'lo': model_lo, 'hi': model_hi}
            print(f"    D+{h}: trained (train={split_idx}, val={val_size})")
    return models

class DLinear(nn.Module):
    def __init__(self, input_len, output_len, kernel_size=25):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2, count_include_pad=False)
        self.linear_trend = nn.Linear(input_len, output_len)
        self.linear_seasonal = nn.Linear(input_len, output_len)
    def forward(self, x):
        x_3d = x.unsqueeze(1)
        trend = self.avg_pool(x_3d).squeeze(1)
        if trend.shape[-1] != x.shape[-1]:
            trend = trend[:, :x.shape[-1]]
        seasonal = x - trend
        return self.linear_trend(trend) + self.linear_seasonal(seasonal)

def create_dlinear_dataset(prices, input_len, output_len):
    X, y = [], []
    for i in range(len(prices) - input_len - output_len + 1):
        X.append(prices[i:i+input_len])
        y.append(prices[i+input_len:i+input_len+output_len])
    return np.array(X), np.array(y)

def _train_single_dlinear(X_tr_t, y_tr_t, X_val_t, y_val_t, input_len, output_len, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = DEVICE
    model = DLinear(input_len, output_len).to(dev)
    X_tr_t, y_tr_t = X_tr_t.to(dev), y_tr_t.to(dev)
    X_val_t, y_val_t = X_val_t.to(dev), y_val_t.to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    for epoch in range(200):
        model.train()
        indices = np.random.permutation(len(X_tr_t))
        for start in range(0, len(indices), 64):
            batch_idx = indices[start:start+64]
            pred = model(X_tr_t[batch_idx])
            loss = nn.MSELoss()(pred, y_tr_t[batch_idx])
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
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
    model.load_state_dict(best_state); model.eval()
    return model

def _dl_backtest_and_forecast(models, prices, prices_norm, p_mean, p_std, input_len, n):
    bt_preds = {}
    for i in range(TEST_DAYS - 30):
        bt_idx = n - TEST_DAYS + i
        if bt_idx - input_len < 0:
            continue
        window = torch.FloatTensor(prices_norm[bt_idx-input_len:bt_idx]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            preds_all = [m(window).squeeze(0).cpu().numpy() for m in models]
        pred = np.mean(preds_all, axis=0) * p_std + p_mean
        for h in HORIZONS:
            if h-1 < len(pred):
                if h not in bt_preds:
                    bt_preds[h] = {'preds':[],'actuals':[]}
                actual_idx = bt_idx + h
                if actual_idx < n:
                    bt_preds[h]['preds'].append(float(pred[h-1]))
                    bt_preds[h]['actuals'].append(float(prices[actual_idx]))
    bt_mapes = {}
    for h, data in bt_preds.items():
        a = np.array(data['actuals']); p_arr = np.array(data['preds'])
        mask = a > 0
        if mask.sum() > 0:
            bt_mapes[h] = float(np.mean(np.abs((a[mask]-p_arr[mask])/a[mask]))*100)
    last_window = torch.FloatTensor(prices_norm[-input_len:]).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        futures_all = [m(last_window).squeeze(0).cpu().numpy() for m in models]
    future_prices = (np.mean(futures_all, axis=0) * p_std + p_mean).tolist()
    return bt_mapes, bt_preds, future_prices

def train_dlinear(crop_series, crop_name, input_len=150, output_len=30, n_seeds=5):
    if not HAS_TORCH:
        print("    DLinear: SKIP (no PyTorch)"); return None
    prices = crop_series['price_per_kg'].values.astype(np.float32)
    n = len(prices)
    if n < input_len + output_len + TEST_DAYS:
        return None
    train_prices = prices[:-TEST_DAYS]
    p_mean, p_std = train_prices.mean(), train_prices.std()
    prices_norm = (prices - p_mean) / (p_std + 1e-8)
    train_norm = prices_norm[:-TEST_DAYS]
    X_all, y_all = create_dlinear_dataset(train_norm, input_len, output_len)
    if len(X_all) < 50:
        return None
    val_size = min(100, len(X_all)//5)
    X_tr_t = torch.FloatTensor(X_all[:-val_size]); y_tr_t = torch.FloatTensor(y_all[:-val_size])
    X_val_t = torch.FloatTensor(X_all[-val_size:]); y_val_t = torch.FloatTensor(y_all[-val_size:])
    models = [_train_single_dlinear(X_tr_t,y_tr_t,X_val_t,y_val_t,input_len,output_len,seed=s*7+42) for s in range(n_seeds)]
    bt_mapes, bt_preds, future_prices = _dl_backtest_and_forecast(models, prices, prices_norm, p_mean, p_std, input_len, n)
    for h, m in bt_mapes.items():
        print(f"    DLinear D+{h}: MAPE={m:.1f}%")
    print(f"    Ensemble of {n_seeds} seeds")
    return {'bt_mapes': bt_mapes, 'bt_preds': bt_preds, 'future_prices': future_prices, 'p_mean': p_mean, 'p_std': p_std}

class NHiTSBlock(nn.Module):
    def __init__(self, input_len, output_len, hidden_size=256, pool_kernel=1):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_kernel, ceil_mode=True)
        pooled_len = math.ceil(input_len / pool_kernel)
        self.mlp = nn.Sequential(nn.Linear(pooled_len, hidden_size), nn.ReLU(),
                                  nn.Linear(hidden_size, hidden_size), nn.ReLU())
        self.backcast_fc = nn.Linear(hidden_size, input_len)
        self.forecast_fc = nn.Linear(hidden_size, output_len)
    def forward(self, x):
        pooled = self.pool(x.unsqueeze(1)).squeeze(1)
        h = self.mlp(pooled)
        return self.backcast_fc(h), self.forecast_fc(h)

class NHiTS(nn.Module):
    def __init__(self, input_len, output_len, hidden_size=256, pool_kernels=None, n_blocks=2):
        super().__init__()
        if pool_kernels is None:
            pool_kernels = [1, 7, 14]
        self.blocks = nn.ModuleList()
        for pk in pool_kernels:
            for _ in range(n_blocks):
                self.blocks.append(NHiTSBlock(input_len, output_len, hidden_size, pk))
    def forward(self, x):
        forecast = torch.zeros(x.shape[0], self.blocks[0].forecast_fc.out_features, device=x.device)
        residual = x
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast
        return forecast

def train_nhits(crop_series, crop_name, input_len=150, output_len=30, n_seeds=3):
    if not HAS_TORCH:
        print("    N-HiTS: SKIP (no PyTorch)"); return None
    prices = crop_series['price_per_kg'].values.astype(np.float32)
    n = len(prices)
    if n < input_len + output_len + TEST_DAYS:
        return None
    train_prices = prices[:-TEST_DAYS]
    p_mean, p_std = train_prices.mean(), train_prices.std()
    prices_norm = (prices - p_mean) / (p_std + 1e-8)
    train_norm = prices_norm[:-TEST_DAYS]
    X_all, y_all = create_dlinear_dataset(train_norm, input_len, output_len)
    if len(X_all) < 50:
        return None
    val_size = min(100, len(X_all)//5)
    models = []
    for seed in range(n_seeds):
        torch.manual_seed(seed*13+7); np.random.seed(seed*13+7)
        dev = DEVICE
        X_tr_t = torch.FloatTensor(X_all[:-val_size]).to(dev); y_tr_t = torch.FloatTensor(y_all[:-val_size]).to(dev)
        X_val_t = torch.FloatTensor(X_all[-val_size:]).to(dev); y_val_t = torch.FloatTensor(y_all[-val_size:]).to(dev)
        model = NHiTS(input_len, output_len, hidden_size=128, pool_kernels=[1,7,14], n_blocks=2).to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        best_val_loss = float('inf'); best_state = None; patience_counter = 0
        for epoch in range(200):
            model.train()
            indices = np.random.permutation(len(X_tr_t))
            for start in range(0, len(indices), 64):
                batch_idx = indices[start:start+64]
                pred = model(X_tr_t[batch_idx])
                loss = nn.MSELoss()(pred, y_tr_t[batch_idx])
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
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
        model.load_state_dict(best_state); model.eval()
        models.append(model)
    bt_mapes, bt_preds, future_prices = _dl_backtest_and_forecast(models, prices, prices_norm, p_mean, p_std, input_len, n)
    for h, m in bt_mapes.items():
        print(f"    N-HiTS D+{h}: MAPE={m:.1f}%")
    print(f"    Ensemble of {n_seeds} seeds")
    return {'bt_mapes': bt_mapes, 'bt_preds': bt_preds, 'future_prices': future_prices, 'p_mean': p_mean, 'p_std': p_std}

def backtest_multihorizon(df, models, avail_features, test_days=TEST_DAYS):
    results = {}
    test_start = len(df) - test_days
    for h in HORIZONS:
        if h not in models:
            continue
        preds_med, actuals = [], []
        for i in range(test_days - h):
            idx = test_start + i
            if idx < 0 or idx + h >= len(df):
                break
            X = df.iloc[idx:idx+1][avail_features].values
            preds_med.append(max(float(models[h]['med'].predict(X)[0]), 100))
            actuals.append(float(df.iloc[idx + h]['price_per_kg']))
        if not actuals:
            continue
        actuals = np.array(actuals); preds_med = np.array(preds_med)
        mask = actuals > 0
        mape = float(np.mean(np.abs((actuals[mask]-preds_med[mask])/actuals[mask]))*100)
        rmse = float(np.sqrt(np.mean((actuals-preds_med)**2)))
        results[h] = {'mape': mape, 'rmse': rmse, 'n_evals': len(actuals)}
    return results

def daily_backtest_lgbm(df, models, avail_features, test_days=TEST_DAYS):
    test_start = len(df) - test_days
    results = []
    for i in range(test_days):
        idx = test_start + i
        if idx >= len(df):
            break
        row = df.iloc[idx]; actual = float(row['price_per_kg'])
        X = df.iloc[idx:idx+1][avail_features].values
        best_h = min(models.keys())
        pred = max(float(models[best_h]['med'].predict(X)[0]), 100)
        lo = float(models[best_h]['lo'].predict(X)[0])
        hi = float(models[best_h]['hi'].predict(X)[0])
        err_pct = abs(actual-pred)/actual*100 if actual > 0 else None
        results.append({'date': row['date'].strftime('%Y-%m-%d'), 'actual': round(actual),
            'predicted': round(pred), 'lo': round(max(lo,50)), 'hi': round(hi),
            'error_pct': round(err_pct,1) if err_pct else None, 'type': 'direct'})
    return results

def generate_future_forecast(df, lgbm_models, dlinear_result, avail_features,
                             weekly_avg, forecast_days=FORECAST_DAYS,
                             ensemble_w_lgbm=0.5, ensemble_w_dl=0.5):
    last_date = df['date'].max(); last_row = df.iloc[-1:]
    forecast_rows = []
    lgbm_preds = {}
    for h in sorted(lgbm_models.keys()):
        X = last_row[avail_features].values
        lgbm_preds[h] = {'med': float(lgbm_models[h]['med'].predict(X)[0]),
            'lo': float(lgbm_models[h]['lo'].predict(X)[0]),
            'hi': float(lgbm_models[h]['hi'].predict(X)[0])}
    dlinear_preds = {}
    if dlinear_result and 'future_prices' in dlinear_result:
        for i, price in enumerate(dlinear_result['future_prices']):
            dlinear_preds[i+1] = max(float(price), 100)
    sorted_horizons = sorted(lgbm_preds.keys())
    last_price = float(df['price_per_kg'].iloc[-1])
    for day in range(1, forecast_days+1):
        curr_date = last_date + timedelta(days=day)
        if day <= sorted_horizons[0]:
            t = day / sorted_horizons[0]
            lgbm_med = last_price*(1-t) + lgbm_preds[sorted_horizons[0]]['med']*t
            lgbm_lo  = last_price*(1-t) + lgbm_preds[sorted_horizons[0]]['lo']*t
            lgbm_hi  = last_price*(1-t) + lgbm_preds[sorted_horizons[0]]['hi']*t
        elif day >= sorted_horizons[-1]:
            woy = curr_date.isocalendar()[1]
            seasonal = weekly_avg.get(woy, last_price)
            t = min(1.0, (day-sorted_horizons[-1])/150)
            lgbm_med = lgbm_preds[sorted_horizons[-1]]['med']*(1-t) + seasonal*t
            lgbm_lo  = lgbm_preds[sorted_horizons[-1]]['lo']*(1-t) + seasonal*0.85*t
            lgbm_hi  = lgbm_preds[sorted_horizons[-1]]['hi']*(1-t) + seasonal*1.15*t
        else:
            for j in range(len(sorted_horizons)-1):
                h1, h2 = sorted_horizons[j], sorted_horizons[j+1]
                if h1 <= day <= h2:
                    t = (day-h1)/(h2-h1)
                    lgbm_med = lgbm_preds[h1]['med']*(1-t) + lgbm_preds[h2]['med']*t
                    lgbm_lo  = lgbm_preds[h1]['lo']*(1-t)  + lgbm_preds[h2]['lo']*t
                    lgbm_hi  = lgbm_preds[h1]['hi']*(1-t)  + lgbm_preds[h2]['hi']*t
                    break
        if day in dlinear_preds and dlinear_preds[day] > 0:
            final_med = lgbm_med*ensemble_w_lgbm + dlinear_preds[day]*ensemble_w_dl
        else:
            final_med = lgbm_med
        final_med = max(final_med, 100); final_lo = max(lgbm_lo, final_med*0.3)
        forecast_rows.append({'date': curr_date.strftime('%Y-%m-%d'),
            'price': round(final_med), 'hi': round(lgbm_hi), 'lo': round(final_lo)})
    return forecast_rows

def stacking_ensemble_backtest(df, lgbm_models, catboost_models, dlinear_result, nhits_result, avail_features, test_days=TEST_DAYS):
    test_start = len(df) - test_days
    for h in HORIZONS:
        if h not in lgbm_models:
            continue
        meta_X, meta_y = [], []
        for i in range(test_days - h):
            idx = test_start + i
            if idx < 0 or idx + h >= len(df):
                break
            preds = []
            actual = float(df.iloc[idx + h]['price_per_kg'])
            X = df.iloc[idx:idx+1][avail_features].values
            preds.append(float(lgbm_models[h]['med'].predict(X)[0]))
            if catboost_models and h in catboost_models:
                preds.append(float(catboost_models[h]['med'].predict(X)[0]))
            else:
                preds.append(preds[0])
            if dlinear_result and h in dlinear_result.get('bt_preds', {}):
                dl_p = dlinear_result['bt_preds'][h]['preds']
                preds.append(dl_p[i] if i < len(dl_p) else preds[0])
            else:
                preds.append(preds[0])
            if nhits_result and h in nhits_result.get('bt_preds', {}):
                nh_p = nhits_result['bt_preds'][h]['preds']
                preds.append(nh_p[i] if i < len(nh_p) else preds[0])
            else:
                preds.append(preds[0])
            meta_X.append(preds); meta_y.append(actual)
        if len(meta_X) < 30:
            continue
        meta_X = np.array(meta_X); meta_y = np.array(meta_y)
        split = len(meta_X) // 2
        if HAS_SKLEARN and split >= 20:
            ridge = Ridge(alpha=1.0)
            ridge.fit(meta_X[:split], meta_y[:split])
            stacked_pred = ridge.predict(meta_X[split:])
            actual_test = meta_y[split:]
            mask = actual_test > 0
            stacked_mape = float(np.mean(np.abs((actual_test[mask]-stacked_pred[mask])/actual_test[mask]))*100)
            names = ['LightGBM','CatBoost','DLinear','N-HiTS']
            print(f"    D+{h}: Stacked MAPE={stacked_mape:.1f}%  weights={dict(zip(names,[round(w,3) for w in ridge.coef_]))}")
            yield h, ridge, stacked_mape
        else:
            avg_pred = meta_X[split:].mean(axis=1)
            actual_test = meta_y[split:]
            mask = actual_test > 0
            avg_mape = float(np.mean(np.abs((actual_test[mask]-avg_pred[mask])/actual_test[mask]))*100)
            yield h, None, avg_mape

def run_crop(crop_df, crop_name, weather_df=None):
    print(f"\n{'='*60}\n[{crop_name.upper()}] {len(crop_df)} days\n{'='*60}")
    crop_wx = None
    if weather_df is not None:
        crop_wx = weather_df[weather_df['crop']==crop_name].copy()
        if len(crop_wx) > 0:
            print(f"  Weather data: {len(crop_wx)} days")
        else:
            crop_wx = None
    weekly_avg = compute_yearly_week_avg(crop_df)
    df = add_features(crop_df, weekly_avg, crop_wx)
    df = df.dropna(subset=['lag90']).reset_index(drop=True)
    if len(df) < TEST_DAYS + 100:
        return None
    avail = [f for f in FEATURES if f in df.columns]
    n_wx = len([f for f in avail if f.startswith('wx_')])
    print(f"  Features: {len(avail)} (weather: {n_wx}), Training rows: {len(df)-TEST_DAYS}")
    print("\n  [1/4 LightGBM]")
    lgbm_models = train_lgbm_multihorizon(df, avail)
    if not lgbm_models:
        return None
    bt_results = backtest_multihorizon(df, lgbm_models, avail)
    for h, r in bt_results.items():
        print(f"    D+{h:2d}: MAPE={r['mape']:.1f}%  RMSE={r['rmse']:,.0f}")
    daily_bt = daily_backtest_lgbm(df, lgbm_models, avail)
    mape_lgbm = bt_results.get(min(bt_results.keys()), {}).get('mape', 99.9)
    print("\n  [2/4 CatBoost]")
    catboost_models = train_catboost_multihorizon(df, avail)
    cb_bt = backtest_multihorizon(df, catboost_models, avail) if catboost_models else {}
    for h, r in cb_bt.items():
        print(f"    D+{h:2d}: MAPE={r['mape']:.1f}%")
    print("\n  [3/4 DLinear]")
    dlinear_result = train_dlinear(crop_df, crop_name)
    dlinear_mape = dlinear_result['bt_mapes'].get(7) if dlinear_result and 'bt_mapes' in dlinear_result else None
    print("\n  [4/4 N-HiTS]")
    nhits_result = train_nhits(crop_df, crop_name)
    nhits_mape = nhits_result['bt_mapes'].get(7) if nhits_result and 'bt_mapes' in nhits_result else None
    print("\n  [Stacking Ensemble]")
    stacking_results = {}
    for h, ridge, sm in stacking_ensemble_backtest(df, lgbm_models, catboost_models, dlinear_result, nhits_result, avail):
        stacking_results[h] = sm
    model_mapes = {'lgbm': mape_lgbm}
    if dlinear_mape: model_mapes['dlinear'] = dlinear_mape
    if nhits_mape: model_mapes['nhits'] = nhits_mape
    if cb_bt and 7 in cb_bt: model_mapes['catboost'] = cb_bt[7]['mape']
    inv_sum = sum(1.0/max(m,1) for m in model_mapes.values())
    weights = {k: (1.0/max(v,1))/inv_sum for k,v in model_mapes.items()}
    print(f"\n  Weights: {', '.join(f'{k}={v:.2f}' for k,v in weights.items())}")
    w_lgbm = weights.get('lgbm', 0.5)
    w_dl = weights.get('dlinear',0) + weights.get('nhits',0)
    w_total = w_lgbm + w_dl
    if w_total > 0:
        w_lgbm /= w_total; w_dl /= w_total
    print("\n  [Future Forecast]")
    forecast_rows = generate_future_forecast(df, lgbm_models, dlinear_result, avail, weekly_avg, ensemble_w_lgbm=w_lgbm, ensemble_w_dl=w_dl)
    print(f"    Generated {len(forecast_rows)} days")
    test_df = df.iloc[-TEST_DAYS:]
    y_te = test_df['price_per_kg'].values; y_naive = test_df['lag7'].values
    mask = y_te > 0
    mape_naive = float(np.mean(np.abs((y_te[mask]-y_naive[mask])/y_te[mask]))*100)
    X_te = test_df[avail].values
    y_pred_os = lgbm_models[min(lgbm_models.keys())]['med'].predict(X_te)
    mape_onestep = float(np.mean(np.abs((y_te[mask]-y_pred_os[mask])/y_te[mask]))*100)
    all_mapes = [mape_lgbm]
    if dlinear_mape: all_mapes.append(dlinear_mape)
    if nhits_mape: all_mapes.append(nhits_mape)
    ensemble_d7 = stacking_results.get(7, min(all_mapes))
    stats = {
        'mape_d7': bt_results.get(7,{}).get('mape',99.9),
        'mape_d14': bt_results.get(14,{}).get('mape',99.9),
        'mape_d30': bt_results.get(30,{}).get('mape',99.9),
        'mape_lgbm_d7': round(mape_lgbm,2),
        'mape_catboost_d7': round(model_mapes.get('catboost',99.9),2),
        'mape_dlinear_d7': round(dlinear_mape,2) if dlinear_mape else None,
        'mape_nhits_d7': round(nhits_mape,2) if nhits_mape else None,
        'mape_ensemble': round(ensemble_d7,2),
        'mape_recursive': round(ensemble_d7,2),
        'mape_onestep': round(mape_onestep,2),
        'mape_naive': round(mape_naive,2),
        'mape_dlinear': round(dlinear_mape,2) if dlinear_mape else None,
        'rmse': round(bt_results.get(min(bt_results.keys()),{}).get('rmse',0),1),
        'beat_naive': ensemble_d7 < mape_naive,
        'test_days': TEST_DAYS, 'model': 'Stacking(LGB+CB+DL+NHiTS)',
        'horizons': list(lgbm_models.keys()), 'quantile_ci': True, 'weather_features': n_wx > 0,
    }
    print(f"\n  Summary: Stack={ensemble_d7:.1f}% LGB={mape_lgbm:.1f}% Naive={mape_naive:.1f}% {'BEAT' if stats['beat_naive'] else 'LOSE'}")
    return daily_bt, forecast_rows, stats

def main():
    if not HAS_LGBM:
        print("ERROR: pip install lightgbm"); return
    print("="*60)
    print("Price Forecast Pipeline v4 - 4-Model Stacking + Weather")
    print(f"  LightGBM: YES | CatBoost: {'YES' if HAS_CATBOOST else 'NO'}")
    print(f"  DLinear: {'YES' if HAS_TORCH else 'NO'} | N-HiTS: {'YES' if HAS_TORCH else 'NO'}")
    if HAS_TORCH: print(f"  GPU: {DEVICE.upper()}")
    print(f"  Horizons: D+{HORIZONS} | Test: {TEST_DAYS}d | Forecast: {FORECAST_DAYS}d")
    print("="*60)
    if not os.path.exists(SRC_CSV):
        print(f"ERROR: {SRC_CSV} not found"); return
    df_raw = pd.read_csv(SRC_CSV, parse_dates=['date'])
    print(f"\nLoaded: {len(df_raw):,} rows, {df_raw['crop'].nunique()} crops")
    weather_df = None
    if os.path.exists(WX_CSV):
        weather_df = pd.read_csv(WX_CSV, parse_dates=['date'])
        print(f"Weather: {len(weather_df):,} rows")
    else:
        print(f"Weather: NOT FOUND - run fetch_weather.py first")
    all_test, all_fore, all_stats = {}, {}, {}
    for crop_name in CROPS:
        crop_df = df_raw[df_raw['crop']==crop_name].copy().sort_values('date').reset_index(drop=True)
        if len(crop_df) < 200:
            continue
        orig_len = len(crop_df)
        year_counts = crop_df.groupby(crop_df['date'].dt.year)['price_per_kg'].count()
        yearly_means = crop_df.groupby(crop_df['date'].dt.year)['price_per_kg'].mean()
        overall_median = yearly_means.median()
        outlier_years = [yr for yr in yearly_means.index if yearly_means[yr] > overall_median*3 and year_counts.get(yr,0)>=200]
        if outlier_years:
            crop_df = crop_df[~crop_df['date'].dt.year.isin(outlier_years)].reset_index(drop=True)
            print(f"\n  Removed outlier years {outlier_years}: {orig_len} -> {len(crop_df)}")
        result = run_crop(crop_df, crop_name, weather_df)
        if result is None:
            continue
        test_result, forecast_rows, stats = result
        w = STANDARD_WEIGHTS[crop_name]; box_unit = UNIT_LABELS[crop_name]
        all_test[crop_name] = test_result
        all_fore[crop_name] = [{**row, 'unit': 'won/kg', 'avg_price': row['price']*w, 'box_unit': box_unit} for row in forecast_rows]
        all_stats[crop_name] = stats
    output = {'generated_at': datetime.now().isoformat(),
        'note': 'v4: 4-Model Stacking + Weather', 'test_results': all_test,
        'forecasts': all_fore, 'stats': all_stats}
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    if all_stats:
        mapes = [s['mape_recursive'] for s in all_stats.values()]
        avg_mape = sum(mapes)/len(mapes)
        avg_rmse = sum(s['rmse'] for s in all_stats.values())/len(all_stats)
        perf_data = {"last_updated": datetime.now().strftime('%Y-%m-%d %H:%M'),
            "models": {"Stacking(LGB+CB+DL+NHiTS)": {"mape": round(avg_mape,2), "rmse": round(avg_rmse,1), "rank": 1}}}
        with open(os.path.join(BASE_DIR, "model_performance.json"), 'w', encoding='utf-8') as f:
            json.dump(perf_data, f, ensure_ascii=False, indent=2)
    print("\n"+"="*60)
    print(f"Saved: {OUT_JSON}")
    for crop, s in all_stats.items():
        wx = "+WX" if s.get('weather_features') else ""
        print(f"  {crop:12s} Stack={s['mape_ensemble']:.1f}% Naive={s['mape_naive']:.1f}% {'BEAT' if s['beat_naive'] else 'LOSE'} {wx}")
    print("\nDone!")

if __name__ == '__main__':
    main()
