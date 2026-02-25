# SOP: TFT 농산물 경락 단가 예측 모델 실행

## 1. 개요
미래 30일간의 농산물 kg당 경락 단가(`price_per_kg`)를 예측하는 TFT 모델을 학습하고 결과를 생성합니다.

## 2. 입력 데이터
- **파일**: `final_clean_dataset.csv`
- **위치**: 프로젝트 루트 또는 `C:\ai\농넷_과거데이터\`
- **주요 컬럼**: `date`, `item`, `kind`, `location`, `도매시장`, `도매법인`, `volume`, `price_per_kg` 및 기상 데이터

## 3. 실행 프로세스 (Execution Layer)
1. **데이터 전처리**:
    - `group_id = item + kind + location` 생성
    - `log_price_kg`, `log_volume` 로그 변환
    - 파생변수 생성: `volume_lag7`, `volume_ma14`
    - 결측치 처리: `ffill`, `bfill` (기상변수 대상)
2. **학습 및 예측**:
    - `pytorch-forecasting` 기반 `TimeSeriesDataSet` 구성
    - `max_encoder_length=60`, `max_prediction_length=30`
    - TFT 모델 학습 (Adam optimizer, QuantileLoss)
3. **결과 생성**:
    - `tft_result_final_v5.csv`: 상세 예측 데이터
    - `dashboard_data.json`: 대시보드 연동용 데이터

## 4. 대시보드 연동
- 생성된 `dashboard_data.json`은 `dashboard.html`에서 로드되어 시각화됩니다.
