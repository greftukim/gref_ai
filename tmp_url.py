from datetime import date, timedelta
import urllib.parse

def generate_url(crop_name):
    base = "https://www.nongnet.or.kr/qlik/sso/single/"
    params = urllib.parse.urlencode({
        "appid": "551d7860-2a5d-49e5-915e-56517f3da2a3", 
        "sheet": "d89143e2-368a-4d41-9851-d4f58ce060dc", 
        "opt": "ctxmenu,currsel"
    })
    p_item = f"select=$::%ED%92%88%EB%AA%A9%EB%AA%85_%EC%84%A0%ED%83%9D,{urllib.parse.quote(crop_name)}"
    dates = [str((date.today() - timedelta(days=i) - date(1899, 12, 30)).days) for i in range(60)]
    p_date = f"select=$::%EA%B2%BD%EB%9D%BD%EC%9D%BC%EC%9E%90_%EC%84%A0%ED%83%9D,{','.join(dates)}"
    return f"{base}?{params}&{p_item}&{p_date}"

print(generate_url("딸기"))
