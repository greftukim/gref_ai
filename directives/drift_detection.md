# Drift Detection Directive

예측선과 최근 실측치의 괴리가 커졌을 때 자동으로 재학습을 트리거하기 위한 SOP.

## Goal

일일 파이프라인이 `--skip-train` 으로만 동작하면 예측선이 고정되어 최근 가격 급변을 반영하지 못한다. 매 실행마다 드리프트를 측정해 필요 시에만 재학습을 돌려 연산 비용과 최신성의 균형을 맞춘다.

## Trigger & Outputs

- **Script**: `execution/detect_drift.py`
- **Inputs**:
  - `lgbm_forecast_dated.json` (기존 예측)
  - `data/<crop>_prices.csv` (최근 실측치)
- **Output**: `.tmp/drift_report.json` (작물별 MAPE, drift 여부, 사유)
- **Exit codes**:
  - `0` → 드리프트 없음, 재학습 불필요
  - `10` → 드리프트 감지, 재학습 권장
  - `2` → 오류 (보통 파일 부재)

## Parameters (CLI flags)

| flag | default | 의미 |
|------|---------|------|
| `--window`         | 10  | 최근 N일 실측치로 MAPE 계산 |
| `--mape-threshold` | 20  | 어떤 작물이든 MAPE (%) 가 이 값 초과 시 drift |
| `--stale-days`     | 14  | 예측 파일 `generated_at` 이 이 일수를 초과하면 drift |
| `--quiet`          | off | 로그 최소화 |

## CI integration

`.github/workflows/update.yml` 의 단계:

1. `Detect forecast drift` — `detect_drift.py` 실행, exit code 를 `steps.drift.outputs.retrain` 에 기록
2. `Generate prices.json` — `retrain=true` 면 `main_update.py` (전체, 재학습 포함), 아니면 `main_update.py --skip-train`
3. 커밋 대상에 `lgbm_forecast_dated.json` 과 `model_performance.json` 포함 (재학습 결과 반영)

## Self-annealing notes

- 특정 작물만 반복적으로 drift 로 잡히면:
  - 해당 작물의 피처(lag, weather) 재검토
  - `--mape-threshold` 를 작물별로 분리 필요하면 스크립트에 dict 기반 임계값 추가
- 예측 파일이 매일 재학습되며 비용이 커지면:
  - `--window` 를 늘리거나 (`30`) `--mape-threshold` 를 상향 (`25`)
  - 재학습 허용 빈도를 상위(workflow) 레벨에서 제한 (예: 24h 쿨다운)
- 추가 로그 히스토리가 필요하면 `.tmp/drift_report.json` 을 `logs/` 로 승격
