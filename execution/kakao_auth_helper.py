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

REST_API_KEY = os.environ.get("KAKAO_REST_API_KEY", "")
CLIENT_SECRET = os.environ.get("KAKAO_CLIENT_SECRET", "")
REDIRECT_URI = "http://localhost"

def get_auth_url():
    url = f"https://kauth.kakao.com/oauth/authorize?client_id={REST_API_KEY}&redirect_uri={REDIRECT_URI}&response_type=code&scope=talk_message"
    return url

def get_token(auth_code):
    url = "https://kauth.kakao.com/oauth/token"
    data = {
        "grant_type": "authorization_code",
        "client_id": REST_API_KEY,
        "client_secret": CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
        "code": auth_code
    }
    
    response = requests.post(url, data=data)
    if response.status_code == 200:
        tokens = response.json()
        # kakao_utils.py와 동일한 경로 (프로젝트 루트)에 저장
        token_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kakao_token.json")
        with open(token_path, "w") as f:
            json.dump(tokens, f)
        print(f"\n[성공] 토큰이 저장되었습니다: {token_path}")
        print("이제 kakao_utils.py를 통해 메시지를 보낼 수 있습니다.")
    else:
        print(f"\n[실패] 토큰 발급에 실패했습니다: {response.json()}")

if __name__ == "__main__":
    if not REST_API_KEY:
        print("[Error] .env 파일에서 KAKAO_REST_API_KEY를 찾을 수 없습니다. .env 파일을 먼저 확인해주세요.")
    else:
        print("=== 카카오 인증 도우미 ===")
        print("1. 아래 주소를 브라우저에 복사하여 접속하세요:")
        print(get_auth_url())
        print("\n2. 로그인을 완료하면 '연결할 수 없는 페이지'가 뜨면서 주소창에 'code=' 뒤에 긴 문자열이 생깁니다.")
        print("   그 문자열(코드)만 복사해서 아래에 입력해주세요.")
        
        code = input("\n코드 입력: ").strip()
        if code:
            get_token(code)
        else:
            print("코드가 입력되지 않았습니다.")
