"""
앙상블 결과 처리 유틸리티
- TFT, LightGBM, CatBoost 결과를 통합
- 각 모델별 MAPE/MAE 계산
- 대시보드 업데이트용 성능 비교 데이터 생성
"""

import os
import pandas as pd
import numpy as np
import json


def load_results():
    """3개 모델의 결과 CSV 로드"""
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '농넷_과거파일')

    results = {}
    files = {
        'TFT': 'tft_result_final_v4.csv',
        'LightGBM': 'lgbm_result.csv',
        'CatBoost': 'catboost_result.csv',
    }

    for model_name, filename in files.items():
        path = os.path.join(base_dir, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            results[model_name] = df
            print(f"{model_name}: {len(df):,} rows loaded from {filename}")
        else:
            print(f"{model_name}: {filename} not found - skipping")

    return results


def evaluate(actual, predicted):
    """MAPE, MAE 계산"""
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)

    mae = np.mean(np.abs(actual - predicted))

    mask = actual > 0
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

    # RMSE
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    return {
        'mape': round(mape, 2),
        'mae': round(mae, 0),
        'rmse': round(rmse, 0),
    }


def generate_performance_summary(results):
    """모델별 성능 요약 생성"""
    summary = {}

    for model_name, df in results.items():
        metrics = evaluate(df['Actual'], df['Predicted'])
        summary[model_name] = {
            'samples': len(df),
            **metrics,
        }

    # MAPE 기준 정렬 (낮을수록 좋음)
    ranked = sorted(summary.items(), key=lambda x: x[1]['mape'])
    for rank, (name, data) in enumerate(ranked, 1):
        summary[name]['rank'] = rank

    return summary


def print_comparison(summary):
    """성능 비교 표 출력"""
    print("\n" + "=" * 70)
    print("\nModel Performance Comparison")
    print("=" * 70)
    # MAE/RMSE 단위: 모델 학습 기준인 원/kg (박스 단위 아님)
    print(f"{'Rank':<6}{'Model':<12}{'MAPE(%)':<12}{'MAE(원/kg)':<14}{'RMSE(원/kg)':<14}{'Samples':<10}")
    print("-" * 70)

    ranked = sorted(summary.items(), key=lambda x: x[1]['rank'])
    for name, data in ranked:
        medal = ['1', '2', '3'][data['rank'] - 1] if data['rank'] <= 3 else ' '
        print(f"{medal} {data['rank']:<4}{name:<12}{data['mape']:<12.2f}{data['mae']:<12,.0f}{data['rmse']:<12,.0f}{data['samples']:<10,}")

    print("=" * 70)


def generate_dashboard_data(summary):
    """대시보드용 JSON 데이터 생성"""
    dashboard_data = {
        'generated_at': pd.Timestamp.now().isoformat(),
        'models': {}
    }

    for model_name, data in summary.items():
        dashboard_data['models'][model_name] = {
            'mape': data['mape'],
            'mae': data['mae'],
            'rmse': data['rmse'],
            'mae_unit': '원/kg',   # 모델 학습 기준 단위 (박스 단위 아님)
            'rank': data['rank'],
            'samples': data['samples'],
        }

    # JSON 파일 저장
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    output_path = os.path.join(output_dir, 'model_performance.json')

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dashboard_data, f, ensure_ascii=False, indent=2)

    print(f"\nDashboard data saved: {output_path}")
    return dashboard_data


def generate_ensemble_predictions(results):
    """단순 평균 앙상블 예측 (공통 길이만큼)"""
    if len(results) < 2:
        print("⚠️ 앙상블을 위해 최소 2개 모델 결과가 필요합니다.")
        return None

    # 공통 길이 (가장 짧은 결과 기준)
    min_len = min(len(df) for df in results.values())

    actuals = []
    preds = []
    for name, df in results.items():
        actuals.append(df['Actual'].values[:min_len])
        preds.append(df['Predicted'].values[:min_len])

    # 앙상블 (단순 평균)
    ensemble_pred = np.mean(preds, axis=0)
    actual = actuals[0]  # 모든 모델의 Actual은 동일해야 함

    metrics = evaluate(actual, ensemble_pred)

    print("\nEnsemble (Simple Average):")
    print(f"   MAPE: {metrics['mape']:.2f}% | MAE: {metrics['mae']:,.0f}원/kg | RMSE: {metrics['rmse']:,.0f}원/kg")

    # 앙상블 결과 저장
    ensemble_df = pd.DataFrame({
        'Actual': actual,
        'Predicted': ensemble_pred
    })
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '농넷_과거파일')
    ensemble_df.to_csv(os.path.join(output_dir, 'ensemble_result.csv'), index=False, encoding='utf-8-sig')

    return metrics


def run():
    """전체 실행"""
    print("=" * 70)
    print("Ensemble results processing started")
    print("=" * 70)

    # 1. 결과 로드
    results = load_results()
    if not results:
        print("❌ 로드된 모델 결과가 없습니다.")
        return

    # 2. 성능 요약
    summary = generate_performance_summary(results)

    # 3. 비교 출력
    print_comparison(summary)

    # 4. 앙상블
    if len(results) >= 2:
        generate_ensemble_predictions(results)

    # 5. 대시보드 데이터 생성
    generate_dashboard_data(summary)


if __name__ == "__main__":
    run()
