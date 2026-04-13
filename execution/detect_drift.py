"""
detect_drift.py
================
예측선과 실측치의 괴리(드리프트)를 감지해 재학습 필요 여부를 판정한다.

동작:
  1. 기존 예측 파일(lgbm_forecast_dated.json)을 읽는다.
  2. data/<crop>_prices.csv 의 최근 실측치를 읽는다.
  3. 예측-실측이 겹치는 최근 N일(기본 10일) 구간에서 MAPE 를 계산한다.
  4. 어떤 작물이든 최근 MAPE 가 임계값(기본 20%) 을 초과하거나
     예측 파일이 너무 오래됐으면(STALE_DAYS 초과) 드리프트로 판정한다.
  5. 결과를 .tmp/drift_report.json 으로 남기고
     드리프트면 exit code 10, 아니면 0 으로 종료한다.

CLI:
  python execution/detect_drift.py
    [--window 10] [--mape-threshold 20] [--stale-days 14] [--quiet]

Exit codes:
  0   드리프트 없음 (재학습 불필요)
  10  드리프트 감지 (재학습 권장)
  2   오류 (예측 파일 또는 실측 데이터 부재 등)
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FORECAST_JSON = os.path.join(BASE_DIR, "lgbm_forecast_dated.json")
REPORT_DIR = os.path.join(BASE_DIR, ".tmp")
REPORT_PATH = os.path.join(REPORT_DIR, "drift_report.json")

CROPS = {
    "strawberry": "strawberry_prices.csv",
    "cucumber":   "cucumber_prices.csv",
    "tomato":     "tomato_prices.csv",
    "paprika":    "paprika_prices.csv",
}

DRIFT_EXIT_CODE = 10


def _parse_date(s):
    return datetime.strptime(s, "%Y-%m-%d").date()


def _load_forecast():
    if not os.path.exists(FORECAST_JSON):
        return None
    with open(FORECAST_JSON, encoding="utf-8") as f:
        return json.load(f)


def _load_actuals(csv_path):
    """CSV 에서 {date(str): price_per_kg(float)} 맵 생성."""
    out = {}
    if not os.path.exists(csv_path):
        return out
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = row.get("DATE") or row.get("date")
            p = row.get("price_per_kg")
            if not d or not p:
                continue
            try:
                price = float(p)
            except ValueError:
                continue
            if price > 0:
                out[d] = price
    return out


def _mape(pairs):
    """pairs: list of (actual, predicted). 0 나눗셈 방지."""
    if not pairs:
        return None
    vals = [abs(a - p) / a for a, p in pairs if a > 0]
    if not vals:
        return None
    return sum(vals) / len(vals) * 100.0


def _forecast_age_days(forecast):
    gen = forecast.get("generated_at")
    if not gen:
        return None
    try:
        gen_dt = datetime.fromisoformat(gen)
    except ValueError:
        return None
    return (datetime.now() - gen_dt).days


def detect(window=10, mape_threshold=20.0, stale_days=14):
    forecast = _load_forecast()
    if not forecast:
        return {
            "drift": True,
            "reason": "forecast_missing",
            "details": {"path": FORECAST_JSON},
        }

    forecasts = forecast.get("forecasts", {})
    age = _forecast_age_days(forecast)

    per_crop = {}
    any_drift = False
    drift_reasons = []

    for crop, csv_name in CROPS.items():
        fc_list = forecasts.get(crop, []) or []
        fc_map = {item.get("date"): item.get("price") for item in fc_list if item.get("date")}
        actuals = _load_actuals(os.path.join(DATA_DIR, csv_name))

        if not actuals or not fc_map:
            per_crop[crop] = {
                "status": "skipped",
                "reason": "no_overlap_or_data",
                "n_compared": 0,
            }
            continue

        # 실측치 최신순 정렬 후 window 크기만큼 사용
        recent_actual_dates = sorted(actuals.keys())[-window:]
        pairs = []
        detail_rows = []
        for d in recent_actual_dates:
            if d in fc_map and fc_map[d] is not None:
                a = actuals[d]
                p = float(fc_map[d])
                pairs.append((a, p))
                detail_rows.append({"date": d, "actual": a, "forecast": p})

        mape = _mape(pairs)
        crop_entry = {
            "status": "ok",
            "window": window,
            "n_compared": len(pairs),
            "recent_mape_pct": round(mape, 2) if mape is not None else None,
            "threshold_pct": mape_threshold,
            "samples": detail_rows[-5:],  # 로그에 남길 최근 5개 샘플
        }
        if mape is not None and mape > mape_threshold:
            crop_entry["status"] = "drift"
            any_drift = True
            drift_reasons.append(f"{crop}: MAPE {mape:.2f}% > {mape_threshold}%")
        per_crop[crop] = crop_entry

    stale = age is not None and age > stale_days
    if stale:
        any_drift = True
        drift_reasons.append(f"forecast_stale: age {age}d > {stale_days}d")

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "forecast_generated_at": forecast.get("generated_at"),
        "forecast_age_days": age,
        "stale_threshold_days": stale_days,
        "mape_threshold_pct": mape_threshold,
        "window_days": window,
        "per_crop": per_crop,
        "drift": any_drift,
        "reasons": drift_reasons,
    }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=10,
                    help="최근 N일 실측치로 MAPE 계산 (default: 10)")
    ap.add_argument("--mape-threshold", type=float, default=20.0,
                    help="MAPE 임계값 %% (default: 20)")
    ap.add_argument("--stale-days", type=int, default=14,
                    help="예측 파일 최대 허용 나이 일수 (default: 14)")
    ap.add_argument("--quiet", action="store_true",
                    help="표준출력 최소화")
    args = ap.parse_args()

    report = detect(
        window=args.window,
        mape_threshold=args.mape_threshold,
        stale_days=args.stale_days,
    )

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if not args.quiet:
        print("[DRIFT] report →", REPORT_PATH)
        print(f"[DRIFT] forecast_age_days={report.get('forecast_age_days')} "
              f"stale_threshold={args.stale_days}")
        for crop, entry in report.get("per_crop", {}).items():
            if entry.get("status") == "skipped":
                print(f"  - {crop}: skipped ({entry.get('reason')})")
            else:
                print(f"  - {crop}: MAPE={entry.get('recent_mape_pct')}% "
                      f"(n={entry.get('n_compared')}) → {entry.get('status')}")
        if report.get("drift"):
            print("[DRIFT] ⚠ drift detected:")
            for r in report.get("reasons", []):
                print(f"  · {r}")
        else:
            print("[DRIFT] OK (재학습 불필요)")

    sys.exit(DRIFT_EXIT_CODE if report.get("drift") else 0)


if __name__ == "__main__":
    main()
