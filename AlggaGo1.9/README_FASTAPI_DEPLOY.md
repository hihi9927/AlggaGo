# 🚀 AlggaGo FastAPI 배포 가이드

FastAPI 기반 AlggaGo 게임을 다양한 클라우드 플랫폼에 배포하는 방법을 안내합니다.

## 📋 배포 옵션

### 1. **Render** (추천) ⭐
- **무료 티어**: 월 750시간
- **자동 배포**: GitHub 연동
- **SSL 자동**: HTTPS 자동 제공
- **쉬운 설정**: 간단한 설정으로 바로 배포

### 2. **Railway**
- **빠른 배포**: 매우 빠른 배포 속도
- **확장성**: 트래픽에 따른 자동 스케일링
- **개발자 친화적**: 좋은 개발자 경험

### 3. **Fly.io**
- **글로벌 배포**: 전 세계 엣지 서버
- **성능**: 매우 빠른 응답 속도
- **무료 티어**: 월 3GB RAM, 3GB 스토리지

### 4. **Heroku**
- **안정성**: 검증된 플랫폼
- **쉬운 배포**: 간단한 Git 기반 배포
- **무료 티어**: 제한적 (2022년 11월 종료)

## 🎯 Render 배포 (추천)

### 1. GitHub 저장소 준비
```bash
# 로컬에서 Git 저장소 초기화
git init
git add .
git commit -m "Initial commit: AlggaGo FastAPI app"
git branch -M main
git remote add origin https://github.com/your-username/alggago.git
git push -u origin main
```

### 2. Render에서 서비스 생성
1. [Render](https://render.com)에 로그인
2. "New +" → "Web Service" 클릭
3. GitHub 저장소 연결
4. 설정:
   - **Name**: `alggago-game`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements_fastapi.txt`
   - **Start Command**: `uvicorn fastapi_app:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free

### 3. 환경 변수 설정 (선택사항)
- `ENVIRONMENT`: `production`
- `PORT`: `8000` (자동 설정됨)

### 4. 배포 완료
- 자동으로 배포가 시작됩니다
- 배포 완료 후 제공되는 URL로 접속 가능

## 🚂 Railway 배포

### 1. Railway CLI 설치
```bash
npm install -g @railway/cli
```

### 2. 프로젝트 배포
```bash
# Railway 로그인
railway login

# 프로젝트 초기화
railway init

# 배포
railway up

# 도메인 확인
railway domain
```

### 3. Railway 대시보드에서 설정
- **Build Command**: `pip install -r requirements_fastapi.txt`
- **Start Command**: `uvicorn fastapi_app:app --host 0.0.0.0 --port $PORT`

## 🦅 Fly.io 배포

### 1. Fly CLI 설치
```bash
# macOS
brew install flyctl

# Windows
powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"

# Linux
curl -L https://fly.io/install.sh | sh
```

### 2. 로그인 및 배포
```bash
# 로그인
fly auth login

# 앱 생성
fly launch

# 배포
fly deploy
```

### 3. fly.toml 설정 (자동 생성됨)
```toml
app = "alggago-game"
primary_region = "nrt"

[build]

[env]
  PORT = "8000"

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0
  processes = ["app"]

[[vm]]
  cpu_kind = "shared"
  cpus = 1
  memory_mb = 256
```

## 🐳 Docker 배포

### 1. 로컬 테스트
```bash
# Docker 이미지 빌드
docker build -t alggago-game .

# 컨테이너 실행
docker run -p 8000:8000 alggago-game

# 또는 docker-compose 사용
docker-compose up -d
```

### 2. 클라우드 플랫폼에 Docker 배포
- **Google Cloud Run**
- **AWS ECS**
- **Azure Container Instances**
- **DigitalOcean App Platform**

## 📁 배포 파일 구조

```
AlggaGo1.9/
├── fastapi_app.py              # 메인 FastAPI 앱
├── requirements_fastapi.txt    # FastAPI 의존성
├── Dockerfile                  # Docker 설정
├── docker-compose.yml          # Docker Compose 설정
├── static/                     # 정적 파일 (선택사항)
│   ├── css/
│   ├── js/
│   └── images/
└── README_FASTAPI_DEPLOY.md   # 이 파일
```

## 🔧 API 엔드포인트

### 기본 엔드포인트
- `GET /` - API 정보
- `GET /docs` - Swagger UI 문서
- `GET /redoc` - ReDoc 문서

### 게임 엔드포인트
- `GET /api/game/state` - 현재 게임 상태
- `POST /api/game/new` - 새 게임 시작
- `POST /api/game/ai-move` - AI 턴 실행
- `POST /api/game/auto` - 자동 게임 실행
- `GET /api/stats` - 게임 통계
- `GET /api/health` - 헬스 체크

### WebSocket
- `WS /ws` - 실시간 게임 업데이트

### 웹 인터페이스
- `GET /game` - 게임 웹 페이지

## 🚀 로컬 개발

### 1. 의존성 설치
```bash
pip install -r requirements_fastapi.txt
```

### 2. 개발 서버 실행
```bash
# 개발 모드
uvicorn fastapi_app:app --reload --host 0.0.0.0 --port 8000

# 프로덕션 모드
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
```

### 3. 접속
- API 문서: http://localhost:8000/docs
- 게임 페이지: http://localhost:8000/game
- API 정보: http://localhost:8000/

## 📊 모니터링

### 헬스 체크
```bash
curl http://your-app-url/api/health
```

### 로그 확인
- Render: 대시보드 → Logs
- Railway: 대시보드 → Deployments → Logs
- Fly.io: `fly logs`

### 성능 모니터링
- Render: 대시보드 → Metrics
- Railway: 대시보드 → Metrics
- Fly.io: `fly status`

## 🔒 보안 고려사항

### 프로덕션 설정
```python
# fastapi_app.py에 추가
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["your-domain.com"])

# 환경 변수로 설정
import os
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
```

### 환경 변수
```bash
# .env 파일 (로컬 개발용)
DEBUG=True
SECRET_KEY=your-secret-key
ALLOWED_HOSTS=localhost,127.0.0.1
```

## 🐛 문제 해결

### 일반적인 문제들

1. **포트 오류**
   ```bash
   # 환경 변수 확인
   echo $PORT
   
   # 포트 설정
   uvicorn fastapi_app:app --host 0.0.0.0 --port $PORT
   ```

2. **의존성 오류**
   ```bash
   # requirements.txt 확인
   cat requirements_fastapi.txt
   
   # 의존성 재설치
   pip install --upgrade -r requirements_fastapi.txt
   ```

3. **메모리 부족**
   - Render: Plan 업그레이드
   - Railway: Resources 증가
   - Fly.io: VM 크기 증가

### 로그 확인
```bash
# 로컬 로그
uvicorn fastapi_app:app --log-level debug

# Docker 로그
docker logs container-name

# 클라우드 플랫폼 로그
# 각 플랫폼 대시보드에서 확인
```

## 📈 성능 최적화

### 1. 캐싱
```python
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

@app.on_event("startup")
async def startup():
    redis = aioredis.from_url("redis://localhost", encoding="utf8")
    FastAPICache.init(RedisBackend(redis), prefix="alggago-cache")
```

### 2. 데이터베이스 (선택사항)
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./alggago.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
```

### 3. 백그라운드 작업
```python
from celery import Celery

celery_app = Celery("alggago", broker="redis://localhost:6379")

@celery_app.task
def process_game_result(game_data):
    # 게임 결과 처리
    pass
```

## 🎉 배포 완료 후

### 1. 테스트
- API 문서 확인: `https://your-app-url/docs`
- 게임 페이지 테스트: `https://your-app-url/game`
- WebSocket 연결 테스트

### 2. 모니터링 설정
- 로그 모니터링
- 성능 메트릭 확인
- 오류 알림 설정

### 3. 도메인 설정 (선택사항)
- 커스텀 도메인 연결
- SSL 인증서 확인

---

**🎯 AlggaGo FastAPI 배포 완료!**

이제 여러분의 AlggaGo 게임이 클라우드에서 실행됩니다! 🚀
