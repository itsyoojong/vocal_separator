---
title: 보컬 분리기 v2
emoji: 🎤
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.12.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Hugging Face UI와 프라이빗 백엔드를 지원하는 보컬 분리기 v2.
suggested_hardware: cpu-basic
---

# 보컬 분리기 v2

`v2`는 이 프로젝트의 분리 배포 버전입니다.

- `app.py`에서 Hugging Face Space 또는 로컬 Gradio UI 실행
- `backend.py`에서 프라이빗 분리 백엔드 실행
- YouTube URL 처리를 위한 백엔드의 선택적 인증 `yt-dlp` 설정

Gradio 앱 주요 기능:

- Demucs 기반 보컬/반주 분리
- 드럼, 베이스, 기타 스템 선택적 추출
- 곡 구조 분석 및 재배열
- 로컬 미리듣기 및 편집 결과 내보내기
- `HF Space UI + 프라이빗 백엔드 API` 분리 배포 지원

## 배포 모드

이 저장소는 두 가지 배포 방식을 지원합니다.

1. `app.py`를 Hugging Face Spaces 또는 로컬 머신에서 Gradio UI로 실행
2. `backend.py`를 Vultr 같은 프라이빗 서버에서 실행하여 YouTube 다운로드 및 빠른 분리 처리

운영 환경에서 YouTube URL 지원이 필요하다면 분리 배포가 현실적인 선택입니다. 공개 HF 런타임은 YouTube 측 rate-limit 또는 봇 차단에 걸릴 가능성이 더 높기 때문입니다.

## Hugging Face Spaces

UI 레이어 권장 시작 설정:

- SDK: `Gradio`
- Hardware: 무료 데모용 `CPU Basic`
- Visibility: 프론트엔드만 공개하려면 `Public`, 비공개 데모면 `Private`

실무 참고:

- 무료 CPU Space는 UI 호스팅에는 충분하지만, 분리까지 Space 내부에서 모두 처리하면 Demucs 연산 부담이 큽니다.
- `SEPARATOR_API_BASE`를 설정하면 UI는 파일 업로드와 YouTube URL 요청을 모두 프라이빗 백엔드로 전달합니다.
- Space 로컬 분리를 유지한다면, 무료 CPU 환경에서는 짧은 트랙 + `Fast` 프로필 조합이 가장 안정적입니다.

## 환경 변수

- 로그인 없이 분리 결과를 다운로드할 수 있습니다.
- 웹 쿠키 업로드/삭제 기능은 제공하지 않습니다. YouTube 인증이 필요하면 서버의 `YT_DLP_COOKIE_FILE` 경로에 쿠키 파일을 직접 설정하세요.
- `APP_HOST`: UI 바인드 호스트(선택)
- `APP_PORT`: UI 바인드 포트(선택)
- `ENABLE_REMOTE_URL`: `1`로 설정하면 YouTube URL UI를 강제로 활성화
- `SEPARATOR_API_BASE`: 프라이빗 백엔드 기본 URL (예: `https://api.example.com`)
- `SEPARATOR_API_TOKEN`: `app.py`와 `backend.py`가 공통으로 사용하는 API 키
- `SEPARATOR_API_TIMEOUT`: 백엔드 호출 타임아웃(초)
- `API_HOST`: `backend.py` 바인드 호스트
- `API_PORT`: `backend.py` 바인드 포트
- `SEPARATOR_PUBLIC_BASE_URL`: API가 리버스 프록시/도메인 뒤에 있을 때 artifact 링크에 사용할 공개 origin(선택)
- `ARTIFACT_MAX_ITEMS`: 메모리에 유지할 다운로드 가능 백엔드 artifact 최대 개수
- `YT_DLP_COOKIE_FILE`: 프라이빗 백엔드 `yt-dlp`용 `cookies.txt` 경로(선택)
- `YT_DLP_COOKIE_FILE`이 없으면 앱은 `.runtime/youtube_cookies.txt`를 기본 경로로 사용합니다. 쿠키는 선택 사항이며 저장소에 포함하지 않습니다. 필요한 경우 실행 환경에 직접 배치하세요.
- `YT_DLP_PROXY`: 프라이빗 백엔드 `yt-dlp` 프록시 URL(선택)
- `YT_DLP_USER_AGENT`: `yt-dlp` 사용자 에이전트(선택)
- `YT_DLP_IMPERSONATE`: `yt-dlp` impersonation 대상(선택). `curl_cffi` 사용 가능 시 기본값은 `chrome`
- `BASIC_PITCH_PYTHON`: Basic Pitch용 Python 인터프리터 오버라이드(선택)

## 권장 분리 배포

1. `backend.py`를 Vultr 같은 프라이빗 서버에 배포합니다.
2. `uvicorn backend:app --host 0.0.0.0 --port 8000`으로 백엔드를 실행합니다.
3. Nginx 또는 Caddy 같은 리버스 프록시로 백엔드 앞단에 HTTPS를 구성합니다.
4. 백엔드 서버와 HF Space 양쪽에 동일한 `SEPARATOR_API_TOKEN`을 설정합니다.
5. HF Space Secrets의 `SEPARATOR_API_BASE`를 백엔드 origin(예: `https://api.example.com`)으로 설정합니다.
6. 이 저장소를 HF Space 저장소에 푸시하면 `app.py`가 자동으로 백엔드를 사용합니다.

백엔드 환경 변수 예시:

```bash
export SEPARATOR_API_TOKEN='replace-this-with-a-long-random-secret'
export API_HOST='0.0.0.0'
export API_PORT='8000'
uvicorn backend:app --host "$API_HOST" --port "$API_PORT"
```

HF Space Secrets 예시:

```text
SEPARATOR_API_BASE=https://api.example.com
SEPARATOR_API_TOKEN=the-same-secret-as-backend
```

## 로컬 실행

UI만 실행:

```bash
python app.py
```

프라이빗 백엔드와 함께 UI 실행:

```bash
SEPARATOR_API_BASE=http://127.0.0.1:8000 \
SEPARATOR_API_TOKEN=dev-secret \
python app.py
```

백엔드 실행:

```bash
SEPARATOR_API_TOKEN=dev-secret \
python backend.py
```

## 참고 사항

- Hugging Face Spaces 무료 티어의 디스크는 일시적(ephemeral)이므로, 다운로드한 오디오와 생성된 스템은 재시작 시 유지되지 않습니다.
- 앱은 임시 파일을 런타임 파일시스템에 저장하며, 기본 데모 용도에서는 영구 스토리지가 필수는 아닙니다.
- 프라이빗 백엔드는 YouTube URL 실패 가능성을 낮춰주지만, 모든 봇 검사를 우회한다고 보장하지는 않습니다. YouTube 정책 집행 상태에 따라 쿠키, 프록시, 인증 기반 추출이 추가로 필요할 수 있습니다.
