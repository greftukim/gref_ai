import requests
import json
import os

# ── .env 파일 로드 함수 ──────────────────────────────────────────────────
def load_env():
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, value = line.partition('=')
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")

load_env()

# ── 카카오 API 설정 ──────────────────────────────────────────────────────────
KAKAO_REST_API_KEY = os.environ.get("KAKAO_REST_API_KEY", "")
KAKAO_CLIENT_SECRET = os.environ.get("KAKAO_CLIENT_SECRET", "")
# 토큰 경로는 프로젝트 루트로 설정 (메인 스크립트들이 찾기 쉽게)
TOKEN_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kakao_token.json")

def save_tokens(tokens):
    with open(TOKEN_PATH, "w") as f:
        json.dump(tokens, f)

def load_tokens():
    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, "r") as f:
            return json.load(f)
    return None

def refresh_access_token():
    tokens = load_tokens()
    if not tokens or 'refresh_token' not in tokens:
        print("[Error] 리프레시 토큰이 없습니다. 먼저 인증을 진행해주세요.")
        return None

    url = "https://kauth.kakao.com/oauth/token"
    data = {
        "grant_type": "refresh_token",
        "client_id": KAKAO_REST_API_KEY,
        "client_secret": KAKAO_CLIENT_SECRET,
        "refresh_token": tokens['refresh_token']
    }
    
    response = requests.post(url, data=data)
    result = response.json()
    
    if response.status_code == 200:
        tokens.update(result)
        save_tokens(tokens)
        print("[OK] 카카오 액세스 토큰 갱신 완료")
        return tokens['access_token']
    else:
        print(f"[Error] 토큰 갱신 실패: {result}")
        return None

def send_kakao_memo(message_text):
    tokens = load_tokens()
    if not tokens:
        print("[Error] 토큰 파일이 없습니다. 인증 도우미를 실행해주세요.")
        return False

    access_token = tokens.get('access_token')
    url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
    
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    
    template = {
        "object_type": "text",
        "text": message_text,
        "link": {
            "web_url": "https://greftukim.github.io/gref_ai/dashboard.html",
            "mobile_web_url": "https://greftukim.github.io/gref_ai/dashboard.html"
        },
        "button_title": "대시보드 확인"
    }
    
    data = {
        "template_object": json.dumps(template)
    }
    
    res = requests.post(url, headers=headers, data=data)
    
    if res.status_code == 401:
        print("[Info] 토큰 만료됨. 갱신 후 재전송 시도...")
        new_token = refresh_access_token()
        if new_token:
            headers["Authorization"] = f"Bearer {new_token}"
            res = requests.post(url, headers=headers, data=data)
    
    if res.status_code == 200:
        print("[OK] 카카오톡 메시지가 전송되었습니다.")
        return True
    else:
        print(f"[Error] 메시지 전송 실패: {res.json()}")
        return False

if __name__ == "__main__":
    if not KAKAO_REST_API_KEY:
        print("[Error] .env 파일에서 KAKAO_REST_API_KEY를 찾을 수 없습니다.")
    else:
        send_kakao_memo("테스트 메시지입니다. GREF AI가 정상 작동 중입니다.")
