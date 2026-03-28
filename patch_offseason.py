import re

DASHBOARD = 'dashboard.html'

with open(DASHBOARD, encoding='utf-8') as f:
    content = f.read()

OLD = """      if (state.tftData && state.tftData[targetId]) {
        return state.tftData[targetId].forecast.slice(0, days).map(f => ({
          date: f.date,
          price: f.price,
          volume: f.volume || 0,
          hi: f.hi,
          lo: f.lo,
          isTFT: true
        }));
      }"""

NEW = """      if (state.tftData && state.tftData[targetId]) {
        var OFF_SEASON = { strawberry: [7, 8] };
        var _offMonths = OFF_SEASON[targetId] || [];
        return state.tftData[targetId].forecast.slice(0, days).filter(function(f) {
          var _m = new Date(f.date).getMonth() + 1;
          return _offMonths.indexOf(_m) === -1;
        }).map(function(f) {
          return { date: f.date, price: f.price, volume: f.volume || 0, hi: f.hi, lo: f.lo, isTFT: true };
        });
      }"""

if OLD in content:
    content = content.replace(OLD, NEW)
    with open(DASHBOARD, 'w', encoding='utf-8') as f:
        f.write(content)
    print("[OK] 비수기 필터 적용 완료")
else:
    print("[ERROR] 대상 코드를 찾지 못했습니다.")
