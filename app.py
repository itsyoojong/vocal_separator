import gc
import html
import importlib.util
import inspect
import json
import os
import re
import secrets
import tempfile
import traceback
from collections import OrderedDict
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import gradio as gr
import requests
import soundfile as sf
import torch
import yt_dlp
from demucs.apply import apply_model
from demucs.audio import convert_audio
from demucs.pretrained import get_model
from analyzer import detect_segments
from arranger import export_arrangement

# 미리듣기 전용 엔드포인트에서 서빙할 토큰별 오디오 파일 경로
_preview_audio_files: OrderedDict[str, str] = OrderedDict()
APP_VERSION_LABEL = "v2"

SUPPORTED_FORMATS = [".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"]
YOUTUBE_URL_PATTERN = re.compile(
    r"(https?://)?(www\.)?"
    r"(youtube\.com/(watch\?v=|shorts/)|youtu\.be/|music\.youtube\.com/watch\?v=)"
    r"[\w\-]+"
)

IS_HF_SPACE = bool(os.getenv("SPACE_ID") or os.getenv("HF_SPACE_ID"))
APP_HOST = os.getenv("APP_HOST", "0.0.0.0" if IS_HF_SPACE else "127.0.0.1")
APP_PORT = int(os.getenv("APP_PORT", "7860"))
OUTPUT_HEADROOM = 0.98
CPU_THREADS = min(os.cpu_count() or 4, 4 if IS_HF_SPACE else 8)
DEFAULT_PROFILE = "fast" if IS_HF_SPACE else "balanced"
SEPARATOR_API_BASE = os.getenv("SEPARATOR_API_BASE", "").strip().rstrip("/")
SEPARATOR_API_TOKEN = os.getenv("SEPARATOR_API_TOKEN", "").strip()
SEPARATOR_API_TIMEOUT = int(os.getenv("SEPARATOR_API_TIMEOUT", "600"))
YT_DLP_COOKIE_FILE = os.getenv("YT_DLP_COOKIE_FILE", "").strip()
YT_DLP_PROXY = os.getenv("YT_DLP_PROXY", "").strip()
YT_DLP_USER_AGENT = os.getenv("YT_DLP_USER_AGENT", "").strip()
YT_DLP_IMPERSONATE = os.getenv("YT_DLP_IMPERSONATE", "chrome").strip()
RUNTIME_DIR = Path(__file__).resolve().parent / ".runtime"
REMOTE_URL_ENABLED = os.getenv(
    "ENABLE_REMOTE_URL",
    "1" if (SEPARATOR_API_BASE or not IS_HF_SPACE) else "0",
).strip().lower() in {
    "1", "true", "yes", "on"
}

torch.set_num_threads(CPU_THREADS)
if hasattr(torch, "set_num_interop_threads"):
    torch.set_num_interop_threads(max(1, min(2, CPU_THREADS)))

STRINGS = {
    "ko": {
        "mrmaker_desc": "음원 파일을 업로드하거나 YouTube URL을 입력하면 보컬·반주·드럼·베이스·기타 트랙으로 분리합니다.",
        "mrmaker_desc_upload_only": "음원 파일을 업로드하면 보컬·반주·드럼·베이스·기타 트랙으로 분리합니다.",
        "tab_file_upload": "파일 업로드",
        "audio_file_label": "음원 파일",
        "stems_label": "추출할 트랙",
        "profile_label": "분리 모드",
        "profile_choices": [("빠름", "fast"), ("균형", "balanced"), ("고음질", "quality")],
        "stem_choices": [("보컬 (Vocals)", "vocals"), ("반주 (Accompaniment)", "accompaniment"), ("드럼 (Drums)", "drums"), ("베이스 (Bass)", "bass"), ("기타 악기 (Other)", "other")],
        "start_btn": "분리 시작",
        "results_md": "### 분리 결과",
        "filename_label": "파일명 / BPM",
        "profile_help": {
            "fast": "속도 우선: 짧은 대기시간",
            "balanced": "균형: 기본 권장값",
            "quality": "음질 우선: 더 오래 걸릴 수 있음",
        },
        "quick_url_md": "또는 YouTube URL로 바로 분리",
        "remote_url_disabled": "현재 공개 Hugging Face 배포에서는 YouTube URL 다운로드를 지원하지 않습니다. 음원 파일 업로드를 사용해주세요.",
        "tab_youtube": "YouTube / YouTube Music URL",
        "url_label": "YouTube URL",
        "url_placeholder": "https://www.youtube.com/watch?v=...\nhttps://music.youtube.com/watch?v=...",
        "dl_separate_btn": "다운로드 후 분리",
        "song_title_label": "곡 제목 / BPM",
        "stem_vocals": "보컬", "stem_accomp": "반주", "stem_drums": "드럼", "stem_bass": "베이스", "stem_other": "기타 악기",
        "arranger_desc": "파일 또는 YouTube URL로 음원을 불러와 송폼을 분석하세요. 감지된 섹션을 드래그 앤 드롭으로 재배열해 새 편곡을 만들 수 있습니다. 섹션 경계는 비트에 스냅되고 이음새는 크로스페이드로 처리됩니다.",
        "arranger_desc_upload_only": "파일 업로드로 음원을 불러와 송폼을 분석하세요. 감지된 섹션을 드래그 앤 드롭으로 재배열해 새 편곡을 만들 수 있습니다. 섹션 경계는 비트에 스냅되고 이음새는 크로스페이드로 처리됩니다.",
        "load_btn": "불러오기", "arr_dl_btn": "다운로드",
        "status_label": "상태", "analyze_btn": "송폼 분석",
        "crossfade_label": "크로스페이드 (ms)", "crossfade_info": "0 = 하드 컷",
        "export_btn": "편곡 내보내기", "arr_result_label": "편곡 결과",
        "arr_state_label": "편곡 순서", "arr_state_info": "드래그앤드롭 결과가 자동 기록됩니다.",
        "arr_source_title": "오리지널 송폼", "arr_track_title": "편곡 트랙",
        "arr_track_subtitle": "(섹션을 드래그해서 추가)",
        "arr_empty_source": "분석 후 섹션이 표시됩니다.", "arr_empty_track": "여기에 섹션을 드래그하세요",
        "arr_preview_title": "미리 듣기", "arr_clear_btn": "전체 지우기",
        "arr_summary": lambda n, d: f"총 {n}개 섹션 · {d}",
        "analyzing": "분석 중...", "analysis_done": "분석 완료 — 섹션을 드래그해서 편곡을 만드세요",
        "err_no_file": "파일을 업로드해주세요.", "err_no_stems": "트랙을 하나 이상 선택해주세요.",
        "err_no_url": "URL을 입력해주세요.", "err_invalid_url": "유효하지 않은 YouTube URL입니다.",
        "err_unsupported_format": lambda formats: f"지원 형식: {formats}",
        "err_no_audio": "먼저 파일을 불러오거나 URL을 다운로드해주세요.",
        "err_no_analysis": "분석을 먼저 실행해주세요.", "err_no_sections": "편곡 트랙에 섹션을 추가해주세요.",
        "dl_fail": "오디오 다운로드에 실패했습니다.",
    },
    "en": {
        "mrmaker_desc": "Upload an audio file or enter a YouTube URL to separate it into vocal, accompaniment, drums, bass, and other tracks.",
        "mrmaker_desc_upload_only": "Upload an audio file to separate it into vocal, accompaniment, drums, bass, and other tracks.",
        "tab_file_upload": "File Upload",
        "audio_file_label": "Audio File",
        "stems_label": "Tracks to Extract",
        "profile_label": "Separation Mode",
        "profile_choices": [("Fast", "fast"), ("Balanced", "balanced"), ("High Quality", "quality")],
        "stem_choices": [("Vocals", "vocals"), ("Accompaniment", "accompaniment"), ("Drums", "drums"), ("Bass", "bass"), ("Other", "other")],
        "start_btn": "Start Separation",
        "results_md": "### Results",
        "filename_label": "Filename / BPM",
        "profile_help": {
            "fast": "Speed first: lower wait time",
            "balanced": "Balanced: recommended default",
            "quality": "Quality first: may take longer",
        },
        "quick_url_md": "Or separate directly from a YouTube URL",
        "remote_url_disabled": "YouTube URL download is disabled on this public Hugging Face deployment. Please use file upload instead.",
        "tab_youtube": "YouTube / YouTube Music URL",
        "url_label": "YouTube URL",
        "url_placeholder": "https://www.youtube.com/watch?v=...\nhttps://music.youtube.com/watch?v=...",
        "dl_separate_btn": "Download & Separate",
        "song_title_label": "Song Title / BPM",
        "stem_vocals": "Vocals", "stem_accomp": "Accompaniment", "stem_drums": "Drums", "stem_bass": "Bass", "stem_other": "Other",
        "arranger_desc": "Load audio from a file or YouTube URL to analyze the song structure. Rearrange detected sections via drag and drop to create a new arrangement. Section boundaries snap to beats and transitions are handled with crossfade.",
        "arranger_desc_upload_only": "Load audio from a file to analyze the song structure. Rearrange detected sections via drag and drop to create a new arrangement. Section boundaries snap to beats and transitions are handled with crossfade.",
        "load_btn": "Load", "arr_dl_btn": "Download",
        "status_label": "Status", "analyze_btn": "Analyze Structure",
        "crossfade_label": "Crossfade (ms)", "crossfade_info": "0 = Hard cut",
        "export_btn": "Export Arrangement", "arr_result_label": "Arrangement Result",
        "arr_state_label": "Arrangement Order", "arr_state_info": "Drag & drop results are recorded automatically.",
        "arr_source_title": "Original Structure", "arr_track_title": "Arrangement Track",
        "arr_track_subtitle": "(drag sections to add)",
        "arr_empty_source": "Sections will appear after analysis.", "arr_empty_track": "Drag sections here",
        "arr_preview_title": "Preview", "arr_clear_btn": "Clear All",
        "arr_summary": lambda n, d: f"{n} sections · {d}",
        "analyzing": "Analyzing...", "analysis_done": "Analysis complete — drag sections to create your arrangement",
        "err_no_file": "Please upload a file.", "err_no_stems": "Please select at least one track.",
        "err_no_url": "Please enter a URL.", "err_invalid_url": "Invalid YouTube URL.",
        "err_unsupported_format": lambda formats: f"Supported formats: {formats}",
        "err_no_audio": "Please load a file or download from URL first.",
        "err_no_analysis": "Please run analysis first.", "err_no_sections": "Please add sections to the arrangement track.",
        "dl_fail": "Failed to download audio.",
    },
}

LANG_ALIASES = {
    "ko": "ko",
    "korean": "ko",
    "한국어": "ko",
    "en": "en",
    "english": "en",
    "영어": "en",
}

# ── Demucs 모델 ────────────────────────────────────────────────────────────────

MODEL = None
WAV_CACHE: OrderedDict[tuple[str, int, int], tuple[torch.Tensor, float]] = OrderedDict()
PROFILE_PRESETS = {
    "fast": {"split_threshold_sec": 45, "split_overlap": 0.08, "shifts": 0},
    "balanced": {"split_threshold_sec": 35, "split_overlap": 0.10, "shifts": 0},
    "quality": {"split_threshold_sec": 25, "split_overlap": 0.20, "shifts": 1},
}
AUDIO_SUPPORTS_BUTTONS = "buttons" in inspect.signature(gr.Audio.__init__).parameters
LAUNCH_SUPPORTS_APP = "_app" in inspect.signature(gr.Blocks.launch).parameters


def _yt_dlp_cookie_path() -> Path:
    configured = os.getenv("YT_DLP_COOKIE_FILE", YT_DLP_COOKIE_FILE).strip()
    if configured:
        return Path(configured).expanduser()
    return RUNTIME_DIR / "youtube_cookies.txt"


def _get_ytdlp_cookie_file() -> str:
    cookie_path = _yt_dlp_cookie_path()
    return str(cookie_path) if cookie_path.is_file() else ""


def _register_preview_audio(audio_path: str) -> str:
    real_path = os.path.realpath(audio_path)
    token = secrets.token_urlsafe(24)
    _preview_audio_files[token] = real_path
    while len(_preview_audio_files) > 32:
        _preview_audio_files.popitem(last=False)
    return f"/preview-audio?token={token}"


def _cache_get(key):
    value = WAV_CACHE.get(key)
    if value is None:
        return None
    WAV_CACHE.move_to_end(key)
    return value


def _cache_put(key, value):
    WAV_CACHE[key] = value
    WAV_CACHE.move_to_end(key)
    while len(WAV_CACHE) > 6:
        WAV_CACHE.popitem(last=False)

def _best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS는 demucs 에서 CPU 보다 오히려 느림 (벤치마크 확인) → CPU 사용
    return torch.device("cpu")

def get_cached_model():
    global MODEL
    if MODEL is None:
        m = get_model("htdemucs")
        m.eval()
        m.to(_best_device())
        MODEL = m
    return MODEL


# ── 앱 시작 시 모델 사전 로드 (첫 요청 지연 방지) ──────────────────────────
import threading as _threading
def _preload():
    try:
        get_cached_model()
        print("[preload] htdemucs 모델 로드 완료", flush=True)
    except Exception as e:
        print(f"[preload] 모델 로드 실패: {e}", flush=True)
_threading.Thread(target=_preload, daemon=True).start()


def _profile_help_text(lang: str, profile: str) -> str:
    return STRINGS[_normalize_lang(lang)]["profile_help"][profile]


def _normalize_lang(lang: str | None) -> str:
    if not isinstance(lang, str):
        return "ko"
    normalized = LANG_ALIASES.get(lang.strip().lower())
    return normalized or "ko"


def _lang_button_update(lang: str, target_lang: str) -> dict:
    return gr.update(variant="primary" if lang == target_lang else "secondary")


def _mrmaker_desc_text(lang: str) -> str:
    s = STRINGS[_normalize_lang(lang)]
    return s["mrmaker_desc"] if REMOTE_URL_ENABLED else s["mrmaker_desc_upload_only"]


def _arranger_desc_text(lang: str) -> str:
    s = STRINGS[_normalize_lang(lang)]
    return s["arranger_desc"] if REMOTE_URL_ENABLED else s["arranger_desc_upload_only"]


def _separator_api_enabled() -> bool:
    return bool(SEPARATOR_API_BASE)


def _separator_api_auth_headers() -> dict[str, str]:
    headers = {}
    if SEPARATOR_API_TOKEN:
        headers["X-API-Key"] = SEPARATOR_API_TOKEN
    return headers


def _extract_api_error_detail(response: requests.Response) -> str:
    body = response.text.strip()
    try:
        payload = response.json()
    except Exception:
        return body or f"HTTP {response.status_code}"
    detail = payload.get("detail")
    if isinstance(detail, str):
        return detail
    return body or f"HTTP {response.status_code}"


def _separator_api_request(path: str, payload: dict | None = None, method: str = "POST") -> dict:
    if not _separator_api_enabled():
        raise RuntimeError("Separator backend is not configured.")

    try:
        response = requests.request(
            method,
            f"{SEPARATOR_API_BASE}{path}",
            json=payload,
            headers=_separator_api_auth_headers(),
            timeout=SEPARATOR_API_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise gr.Error(f"Separator backend request failed: {exc}") from exc
    if not response.ok:
        raise gr.Error(_extract_api_error_detail(response))
    return response.json()


def _separator_api_upload_file(audio_path: str, stems: list[str], profile: str, lang: str) -> dict:
    if not _separator_api_enabled():
        raise RuntimeError("Separator backend is not configured.")

    data = {
        "stems": json.dumps(stems, ensure_ascii=False),
        "profile": profile,
        "lang": lang,
    }
    with open(audio_path, "rb") as fh:
        files = {"file": (Path(audio_path).name, fh, "application/octet-stream")}
        try:
            response = requests.post(
                f"{SEPARATOR_API_BASE}/api/separate-file",
                data=data,
                files=files,
                headers=_separator_api_auth_headers(),
                timeout=SEPARATOR_API_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise gr.Error(f"Separator backend request failed: {exc}") from exc
    if not response.ok:
        raise gr.Error(_extract_api_error_detail(response))
    return response.json()


def _download_remote_artifact(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, prefix="backend_")
    try:
        with requests.get(url, timeout=SEPARATOR_API_TIMEOUT, stream=True) as response:
            response.raise_for_status()
            with open(tmp.name, "wb") as fh:
                for chunk in response.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        fh.write(chunk)
    except requests.RequestException as exc:
        raise gr.Error(f"Failed to download backend artifact: {exc}") from exc
    return tmp.name


def _remote_stem_updates(stem_urls: dict, output_stems: list[str]):
    values = [stem_urls.get(stem) if stem in output_stems else None for stem in STEM_KEYS]
    return paths_to_updates(values, output_stems)


def _canonicalize_youtube_url(url: str) -> str:
    raw = (url or "").strip()
    parsed = urlparse(raw)
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").strip("/")
    query = parse_qs(parsed.query or "")

    if host in {"music.youtube.com", "www.youtube.com", "youtube.com", "m.youtube.com"}:
        video_id = query.get("v", [None])[0]
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
        if path.startswith("shorts/"):
            short_id = path.split("/", 1)[1].split("/", 1)[0]
            if short_id:
                return f"https://www.youtube.com/watch?v={short_id}"

    if host == "youtu.be" and path:
        video_id = path.split("/", 1)[0]
        return f"https://www.youtube.com/watch?v={video_id}"

    return raw


def _download_error_message(lang: str, exc: Exception) -> str:
    lang = _normalize_lang(lang)
    raw = str(exc).strip()
    raw = re.sub(r"\x1b\[[0-9;]*m", "", raw)
    if "ERROR:" in raw:
        raw = raw.split("ERROR:", 1)[1].strip()
    lowered = raw.lower()
    if "sign in to confirm you" in lowered and "not a bot" in lowered:
        if lang == "ko":
            return (
                "YouTube가 현재 이 서버 요청을 봇으로 판단해 다운로드를 차단했습니다.\n"
                "공개 Hugging Face Space에서는 로그인 쿠키 또는 PO Token 없이 "
                "YouTube URL 다운로드가 막힐 수 있습니다.\n"
                "대안: 음원 파일 업로드를 사용하거나, 개인 서버/로컬 환경에서 인증 정보를 설정해 실행하세요."
            )
        return (
            "YouTube is blocking downloads from this server as a bot check.\n"
            "On a public Hugging Face Space, YouTube URL downloads may fail without "
            "cookies or a PO Token.\n"
            "Use file upload instead, or run the app on a private/local server with auth configured."
        )
    return f"{STRINGS[lang]['dl_fail']}\n{raw}" if raw else STRINGS[lang]["dl_fail"]


def _inline_error_text(lang: str, exc: Exception) -> str:
    label = "오류" if _normalize_lang(lang) == "ko" else "Error"
    message = str(exc).strip() or STRINGS[_normalize_lang(lang)]["dl_fail"]
    return f"[{label}] {message}"


def estimate_audio_bpm(audio_path: str) -> float | None:
    """Estimate track BPM for display. Returns None when detection is unreliable."""
    try:
        import librosa
    except Exception as exc:
        print(f"[bpm] librosa import 실패: {exc}", flush=True)
        return None

    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=180.0)
        if y.size < sr * 5:
            return None
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        bpm = float(np.atleast_1d(tempo)[0])
        if not np.isfinite(bpm) or bpm <= 0:
            return None
        return round(bpm, 1)
    except Exception as exc:
        print(f"[bpm] 추정 실패: {exc}", flush=True)
        return None


def _coerce_bpm(value) -> float | None:
    try:
        bpm = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(bpm) or bpm <= 0:
        return None
    return round(bpm, 1)


def _format_title_with_bpm(title: str, bpm: float | None, lang: str = "ko") -> str:
    lang = _normalize_lang(lang)
    name = (title or "").strip()
    missing_text = "BPM 미검출" if lang == "ko" else "BPM unavailable"
    if bpm is None:
        return f"{name} · {missing_text}" if name else missing_text
    bpm_text = f"BPM {bpm:.1f}"
    return f"{name} · {bpm_text}" if name else bpm_text


def _prepare_audio_tensor(audio_path: str, model):
    cache_key = (os.path.realpath(audio_path), model.samplerate, model.audio_channels)
    cached = _cache_get(cache_key)
    if cached is not None:
        wav_tensor, duration_sec = cached
        return wav_tensor.clone(), duration_sec

    data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    duration_sec = data.shape[0] / float(sr)
    wav = torch.from_numpy(data.T)
    del data
    wav = convert_audio(wav, sr, model.samplerate, model.audio_channels).contiguous()
    _cache_put(cache_key, (wav.cpu(), duration_sec))
    return wav, duration_sec


def _write_stem_audio(path: str, wav_out: np.ndarray, samplerate: int):
    peak = float(np.max(np.abs(wav_out))) if wav_out.size else 0.0
    if peak > OUTPUT_HEADROOM:
        wav_out = wav_out * (OUTPUT_HEADROOM / peak)
    sf.write(path, wav_out.T, samplerate)


def _audio_component(label: str, visible: bool, interactive: bool = False):
    kwargs = {"label": label, "visible": visible, "interactive": interactive}
    if AUDIO_SUPPORTS_BUTTONS:
        kwargs["buttons"] = ["download"]
    return gr.Audio(**kwargs)


_NO_VALUE = object()


def _audio_update(value=_NO_VALUE, visible: bool | None = None):
    kwargs = {}
    if value is not _NO_VALUE:
        kwargs["value"] = value
    if visible is not None:
        kwargs["visible"] = visible
    if AUDIO_SUPPORTS_BUTTONS:
        kwargs["buttons"] = ["download"]
    return gr.update(**kwargs)


def _has_curl_cffi() -> bool:
    return importlib.util.find_spec("curl_cffi") is not None


def _build_yt_dlp_opts(tmp_dir: str, include_missing_pot: bool = False) -> dict:
    extractor_args = {
        "youtube": {
            "player_client": ["mweb", "ios", "android", "web"],
        }
    }
    if include_missing_pot:
        extractor_args["youtube"]["formats"] = ["missing_pot"]

    ydl_opts = {
        "format": "bestaudio*/bestaudio/best",
        "outtmpl": str(Path(tmp_dir) / "%(title)s.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio",
                            "preferredcodec": "wav", "preferredquality": "0"}],
        "noplaylist": True,
        "quiet": True, "no_warnings": True,
        "extractor_args": extractor_args,
        "extractor_retries": 3,
        "retries": 3,
        "fragment_retries": 3,
        "sleep_interval_requests": 1,
    }
    if YT_DLP_IMPERSONATE and _has_curl_cffi():
        ydl_opts["impersonate"] = YT_DLP_IMPERSONATE
    return ydl_opts

# ── YouTube 다운로드 ───────────────────────────────────────────────────────────

def download_audio(url: str, lang: str = "ko") -> tuple[str, str]:
    lang = _normalize_lang(lang)
    url = _canonicalize_youtube_url(url)
    tmp_dir = tempfile.mkdtemp()
    ydl_opts = _build_yt_dlp_opts(tmp_dir, include_missing_pot=False)
    cookie_file = _get_ytdlp_cookie_file()
    if cookie_file:
        ydl_opts["cookiefile"] = cookie_file
    if YT_DLP_PROXY:
        ydl_opts["proxy"] = YT_DLP_PROXY
    if YT_DLP_USER_AGENT:
        ydl_opts["user_agent"] = YT_DLP_USER_AGENT
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title", "unknown")
    except Exception as exc:
        message = str(exc)
        if "Requested format is not available" in message:
            fallback_opts = _build_yt_dlp_opts(tmp_dir, include_missing_pot=True)
            if cookie_file:
                fallback_opts["cookiefile"] = cookie_file
            if YT_DLP_PROXY:
                fallback_opts["proxy"] = YT_DLP_PROXY
            if YT_DLP_USER_AGENT:
                fallback_opts["user_agent"] = YT_DLP_USER_AGENT
            try:
                with yt_dlp.YoutubeDL(fallback_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    title = info.get("title", "unknown")
            except Exception as fallback_exc:
                raise gr.Error(_download_error_message(lang, fallback_exc)) from fallback_exc
        else:
            raise gr.Error(_download_error_message(lang, exc)) from exc
    wav_files = list(Path(tmp_dir).glob("*.wav"))
    if not wav_files:
        raise gr.Error(STRINGS[lang]["dl_fail"])
    return str(wav_files[0]), title

# ── 트랙 분리 ─────────────────────────────────────────────────────────────────

STEM_KEYS = ["vocals", "accompaniment", "drums", "bass", "other"]


def separate_stems(audio_path: str, output_stems: list[str], profile: str = "balanced") -> list:
    import time
    t0 = time.time()
    gc.collect()

    model = get_cached_model()
    device = next(model.parameters()).device
    preset = PROFILE_PRESETS.get(profile, PROFILE_PRESETS["balanced"])
    print(f"[separate_stems] device={device}, profile={profile}, stems={output_stems}", flush=True)

    wav, duration_sec = _prepare_audio_tensor(audio_path, model)
    wav = wav.unsqueeze(0).to(device)
    use_split = duration_sec >= preset["split_threshold_sec"]
    autocast_enabled = device.type == "cuda"

    with torch.inference_mode():
        if autocast_enabled:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                sources = apply_model(
                    model, wav, device=device,
                    shifts=preset["shifts"], split=use_split,
                    segment=7.8,   # htdemucs 고정값 — 변경하면 shape 오류
                    overlap=preset["split_overlap"],
                    num_workers=0, progress=False,
                )[0]
        else:
            sources = apply_model(
                model, wav, device=device,
                shifts=preset["shifts"], split=use_split,
                segment=7.8,   # htdemucs 고정값 — 변경하면 shape 오류
                overlap=preset["split_overlap"],
                num_workers=0, progress=False,
            )[0]
    del wav
    gc.collect()

    stem_names = list(model.sources)
    stem_index = {name: idx for idx, name in enumerate(stem_names)}
    paths = []
    for stem in STEM_KEYS:
        if stem not in output_stems:
            paths.append(None); continue
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix=f"{stem}_")
        if stem == "accompaniment":
            keep_indices = [idx for name, idx in stem_index.items() if name != "vocals"]
            wav_out = sources[keep_indices].sum(dim=0).detach().cpu().numpy()
        else:
            wav_out = sources[stem_index[stem]].detach().cpu().numpy()
        _write_stem_audio(tmp.name, wav_out, model.samplerate)
        del wav_out
        paths.append(tmp.name)

    del stem_index
    del sources
    gc.collect()
    print(
        f"[separate_stems] 완료: {time.time()-t0:.1f}s "
        f"(duration={duration_sec:.1f}s, split={use_split}, overlap={preset['split_overlap']}, shifts={preset['shifts']})",
        flush=True,
    )
    return paths

def paths_to_updates(paths, output_stems):
    return [_audio_update(value=p, visible=(k in output_stems))
            for p, k in zip(paths, STEM_KEYS)]


def empty_output_updates(output_stems):
    return [_audio_update(value=None, visible=(k in output_stems))
            for k in STEM_KEYS]

# ── MR Maker 핸들러 ───────────────────────────────────────────────────────────

def mrmaker_from_file(audio_path, stems, profile="balanced", lang="ko"):
    lang = _normalize_lang(lang)
    s = STRINGS[lang]
    if not audio_path:
        raise gr.Error(s["err_no_file"])
    if Path(audio_path).suffix.lower() not in SUPPORTED_FORMATS:
        raise gr.Error(s["err_unsupported_format"](", ".join(SUPPORTED_FORMATS)))
    if not stems:
        raise gr.Error(s["err_no_stems"])
    if _separator_api_enabled():
        local_bpm = estimate_audio_bpm(audio_path)
        payload = _separator_api_upload_file(audio_path, stems, profile, lang)
        bpm = _coerce_bpm(payload.get("bpm"))
        if bpm is None:
            bpm = local_bpm
        title = _format_title_with_bpm(
            payload.get("title", Path(audio_path).name),
            bpm,
            lang,
        )
        return [title] + _remote_stem_updates(
            payload.get("stems", {}),
            stems,
        )
    bpm = estimate_audio_bpm(audio_path)
    title = _format_title_with_bpm(Path(audio_path).name, bpm, lang)
    return [title] + paths_to_updates(separate_stems(audio_path, stems, profile), stems)

def mrmaker_from_url(url, stems, profile="balanced", lang="ko"):
    lang = _normalize_lang(lang)
    s = STRINGS[lang]
    url = (url or "").strip()
    if not url:
        raise gr.Error(s["err_no_url"])
    if not YOUTUBE_URL_PATTERN.search(url):
        raise gr.Error(s["err_invalid_url"])
    if not stems:
        raise gr.Error(s["err_no_stems"])
    try:
        if _separator_api_enabled():
            payload = _separator_api_request(
                "/api/separate-url",
                {"url": url, "stems": stems, "profile": profile, "lang": lang},
            )
            title = _format_title_with_bpm(
                payload.get("title", ""),
                _coerce_bpm(payload.get("bpm")),
                lang,
            )
            return [title] + _remote_stem_updates(
                payload.get("stems", {}),
                stems,
            )
        audio_path, title = download_audio(url, lang)
        bpm = estimate_audio_bpm(audio_path)
        return [_format_title_with_bpm(title, bpm, lang)] + paths_to_updates(
            separate_stems(audio_path, stems, profile),
            stems,
        )
    except Exception as exc:
        print("[mrmaker_from_url] 실패", flush=True)
        print(traceback.format_exc(), flush=True)
        return [_inline_error_text(lang, exc)] + empty_output_updates(stems)

def visibility_update(stems):
    return [gr.update(visible=k in stems) for k in STEM_KEYS]


# ── 편곡 핸들러 ───────────────────────────────────────────────────────────────

def arranger_download_url(url, lang="ko"):
    lang = _normalize_lang(lang)
    s = STRINGS[lang]
    url = (url or "").strip()
    if not url:
        raise gr.Error(s["err_no_url"])
    if not YOUTUBE_URL_PATTERN.search(url):
        raise gr.Error(s["err_invalid_url"])
    if _separator_api_enabled():
        payload = _separator_api_request("/api/download-url", {"url": url, "lang": lang})
        local_path = _download_remote_artifact(payload["audio_url"])
        return local_path, f"✓ {payload.get('title', Path(local_path).name)}"
    path, title = download_audio(url, lang)
    return path, f"✓ {title}"

def arranger_load_file(file_path, lang="ko"):
    lang = _normalize_lang(lang)
    s = STRINGS[lang]
    if not file_path:
        raise gr.Error(s["err_no_file"])
    return file_path, f"✓ {Path(file_path).name}"

def run_analysis(audio_path, lang="ko"):
    lang = _normalize_lang(lang)
    s = STRINGS[lang]
    print(f"[run_analysis] path={repr(audio_path)}")
    if not audio_path:
        raise gr.Error(s["err_no_audio"])
    preview_url = _register_preview_audio(audio_path) if LAUNCH_SUPPORTS_APP else ""
    segments = detect_segments(audio_path)
    print(f"[run_analysis] {len(segments)} segments → preview: {preview_url}")
    segs_json = json.dumps(segments, ensure_ascii=False)
    trigger_html = (
        f'<div id="arr-trigger" data-segments=\'{segs_json}\' '
        f'data-preview-url="{html.escape(preview_url, quote=True)}" style="display:none"></div>'
        + get_arranger_html_shell(lang)
    )
    return segs_json, trigger_html

def run_export(audio_path, segs_json, arr_json, crossfade_ms, lang="ko"):
    lang = _normalize_lang(lang)
    s = STRINGS[lang]
    if not audio_path:
        raise gr.Error(s["err_no_audio"])
    try:
        segments = json.loads(segs_json)
        indices  = json.loads(arr_json)
    except Exception:
        raise gr.Error(s["err_no_analysis"])
    if not indices:
        raise gr.Error(s["err_no_sections"])
    result_path = export_arrangement(audio_path, segments, indices, int(crossfade_ms))
    return _audio_update(value=result_path)

# ── 편곡 UI 상수 ──────────────────────────────────────────────────────────────

def get_arranger_html_shell(lang="ko"):
    lang = _normalize_lang(lang)
    s = STRINGS[lang]
    return f"""
<div id="lang-trigger" data-lang="{lang}" style="display:none"></div>
<div id="arranger-root">
  <div class="arranger-cols">
    <div>
      <div class="panel-title">{s['arr_source_title']}</div>
      <div id="source-blocks" class="source-blocks">
        <p style="color:#aaa;font-size:13px;">{s['arr_empty_source']}</p>
      </div>
    </div>
    <div>
      <div class="panel-title">{s['arr_track_title']}
        <span style="font-weight:400;font-size:11px;">{s['arr_track_subtitle']}</span>
      </div>
      <div id="arrangement-track" class="arrangement-track">
        <div class="arrangement-empty">{s['arr_empty_track']}</div>
      </div>
      <div id="arr-summary" class="arr-summary"></div>
      <button class="clear-btn" onclick="window.arrClear()">{s['arr_clear_btn']}</button>
    </div>
  </div>
</div>
"""

ARRANGER_CSS = """
#arranger-root { font-family: sans-serif; color: #1a1a1a; }
.arranger-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.panel-title { font-weight:700; font-size:14px; margin-bottom:8px; color:#444;
               letter-spacing:.03em; text-transform:uppercase; }
.source-blocks { display:flex; flex-direction:column; gap:6px; min-height:80px; }
.seg-block { display:flex; align-items:center; gap:8px; padding:8px 12px;
             border-radius:8px; cursor:grab; user-select:none; font-size:13px;
             font-weight:600; color:#fff; box-shadow:0 1px 4px rgba(0,0,0,.18);
             transition:transform .1s; }
.seg-block:active { cursor:grabbing; transform:scale(1.02); }
.seg-label { font-size:18px; font-weight:800; min-width:22px; text-align:center; }
.seg-info { display:flex; flex-direction:column; font-size:11px; font-weight:400; opacity:.92; }
.arrangement-track { min-height:200px; border:2px dashed #bbb; border-radius:10px;
                     padding:10px; background:#f8f8f8; display:flex;
                     flex-direction:column; gap:6px;
                     transition:background .15s, border-color .15s; }
.arrangement-track.drag-over { background:#eef5ff; border-color:#4A90D9; }
.arrangement-empty { color:#aaa; font-size:13px; text-align:center; margin:auto; padding:20px 0; }
.arr-block { display:flex; align-items:center; gap:8px; padding:8px 12px;
             border-radius:8px; cursor:grab; user-select:none; font-size:13px;
             font-weight:600; color:#fff; box-shadow:0 1px 4px rgba(0,0,0,.18); }
.arr-block:active { cursor:grabbing; }
.arr-block.drag-over-top    { border-top:3px solid #fff; }
.arr-block.drag-over-bottom { border-bottom:3px solid #fff; }
.remove-btn { margin-left:auto; background:rgba(0,0,0,.25); border:none; color:#fff;
              border-radius:4px; width:22px; height:22px; font-size:14px;
              cursor:pointer; display:flex; align-items:center;
              justify-content:center; flex-shrink:0; }
.remove-btn:hover { background:rgba(0,0,0,.45); }
.arr-summary { margin-top:8px; font-size:12px; color:#666; text-align:right; }
.clear-btn { margin-top:6px; background:none; border:1px solid #ccc;
             border-radius:6px; padding:4px 10px; font-size:12px;
             cursor:pointer; color:#555; }
.clear-btn:hover { background:#f0f0f0; }
.preview-btn { margin-left:auto; background:rgba(255,255,255,.28); border:none;
               color:#fff; border-radius:50%; width:26px; height:26px; font-size:12px;
               cursor:pointer; display:flex; align-items:center; justify-content:center;
               flex-shrink:0; transition:background .15s, transform .1s; }
.preview-btn:hover { background:rgba(255,255,255,.55); transform:scale(1.1); }
.preview-btn.playing { background:rgba(255,255,255,.7); color:#333; }
"""

# ARRANGER_JS: 페이지 로드 시 실행. MutationObserver로 #arr-trigger 감지
# Gradio 6 injects this as a <script> tag — must be an IIFE to auto-execute.
ARRANGER_JS = """
(() => {
  window._arrSegs    = [];
  window._arrOrder   = [];
  window.arrAudioUrl = null;
  window._previewAudio = null;
  let dragSrcIdx = null;
  let dragArrPos = null;

  // Language strings
  window.LANG_STRINGS = {
    ko: {
      emptySource: '분석 후 섹션이 표시됩니다.',
      emptyArr: '여기에 섹션을 드래그하세요',
      previewTitle: '미리 듣기',
      summaryFn: function(n,d){ return '총 '+n+'개 섹션 · '+d; },
    },
    en: {
      emptySource: 'Sections will appear after analysis.',
      emptyArr: 'Drag sections here',
      previewTitle: 'Preview',
      summaryFn: function(n,d){ return n+' sections · '+d; },
    }
  };
  window._currentLang = 'ko';
  function LS(){ return window.LANG_STRINGS[window._currentLang] || window.LANG_STRINGS.ko; }

  function fmt(s){ const m=Math.floor(s/60),sec=Math.floor(s%60); return m+':'+String(sec).padStart(2,'0'); }
  function dur(s){ const m=Math.floor(s/60),sec=Math.floor(s%60); return m>0?m+'m '+sec+'s':sec+'s'; }

  function syncGradio(){
    // arrangement-state Textbox 를 직접 찾아 업데이트
    document.querySelectorAll('textarea').forEach(el => {
      if(el.closest('[data-testid="textbox"]') && el.value && el.value.startsWith('[')){
        const label = el.closest('.block')?.querySelector('label span');
        if(label && label.textContent.includes('arrangement')) {
          el.value = JSON.stringify(window._arrOrder);
          el.dispatchEvent(new Event('input',{bubbles:true}));
        }
      }
    });
    // elem_id 방식 폴백
    const fb = document.querySelector('#arrangement-state textarea');
    if(fb){ fb.value=JSON.stringify(window._arrOrder); fb.dispatchEvent(new Event('input',{bubbles:true})); }
  }

  // ── 미리듣기 ─────────────────────────────────────────────────────────────
  window.arrPreview = function(segIdx) {
    const seg = window._arrSegs[segIdx];
    if (!seg) return;
    const allBtns = document.querySelectorAll('.preview-btn');

    // If already playing this segment → stop
    if (window._previewAudio && window._previewAudio._segIdx === segIdx) {
      window._previewAudio.pause();
      window._previewAudio = null;
      allBtns.forEach(b=>{ b.textContent='▶'; b.classList.remove('playing'); });
      return;
    }

    // Stop any existing preview
    if (window._previewAudio) {
      window._previewAudio.pause();
      window._previewAudio = null;
    }
    allBtns.forEach(b=>{ b.textContent='▶'; b.classList.remove('playing'); });

    if (!window.arrAudioUrl) return;

    const audio = new Audio(window.arrAudioUrl);
    audio._segIdx = segIdx;
    window._previewAudio = audio;

    const thisBtn = document.querySelector('.preview-btn[data-idx="'+segIdx+'"]');
    if (thisBtn) { thisBtn.textContent='⏹'; thisBtn.classList.add('playing'); }

    const onReady = () => {
      audio.currentTime = seg.start;
      audio.play().catch(()=>{});
    };
    if (audio.readyState >= 2) { onReady(); }
    else { audio.addEventListener('canplay', onReady, {once:true}); }

    const stopPreview = () => {
      if (!window._previewAudio || window._previewAudio !== audio) return;
      if (audio.currentTime >= seg.end) {
        audio.pause();
        window._previewAudio = null;
        if (thisBtn) { thisBtn.textContent='▶'; thisBtn.classList.remove('playing'); }
      }
    };
    audio.addEventListener('timeupdate', stopPreview);
    audio.addEventListener('ended', ()=>{
      window._previewAudio=null;
      if(thisBtn){thisBtn.textContent='▶';thisBtn.classList.remove('playing');}
    });
    audio.addEventListener('error', ()=>{
      console.error('[arrPreview] 오디오 로드 실패 →', audio.src);
      window._previewAudio=null;
      if(thisBtn){thisBtn.textContent='⚠'; thisBtn.title=LS().previewTitle+' 실패';}
    });
  };

  function renderSource(){
    const c = document.getElementById('source-blocks');
    if(!c) return;
    c.innerHTML='';
    if(!window._arrSegs.length){
      c.innerHTML='<p style="color:#aaa;font-size:13px;">'+LS().emptySource+'</p>';
      return;
    }
    window._arrSegs.forEach(seg=>{
      const b=document.createElement('div');
      b.className='seg-block'; b.style.background=seg.color;
      b.draggable=true; b.dataset.idx=seg.index;
      b.innerHTML='<span class="seg-label">'+seg.label+'</span>'+
        '<span class="seg-info"><span>'+fmt(seg.start)+' – '+fmt(seg.end)+'</span>'+
        '<span>'+dur(seg.duration)+'</span></span>'+
        '<button class="preview-btn" data-idx="'+seg.index+'" title="'+LS().previewTitle+'">▶</button>';
      b.querySelector('.preview-btn').addEventListener('click', e=>{
        e.stopPropagation();   // prevent drag from triggering
        window.arrPreview(seg.index);
      });
      b.addEventListener('dragstart',e=>{
        dragSrcIdx=parseInt(e.currentTarget.dataset.idx); dragArrPos=null;
        e.dataTransfer.effectAllowed='copy';
      });
      c.appendChild(b);
    });
  }

  function renderArr(){
    const track=document.getElementById('arrangement-track');
    if(!track) return;
    track.innerHTML='';
    if(!window._arrOrder.length){
      track.innerHTML='<div class="arrangement-empty">'+LS().emptyArr+'</div>';
      const s=document.getElementById('arr-summary'); if(s) s.textContent='';
      return;
    }
    let totalDur=0;
    window._arrOrder.forEach((si,pos)=>{
      const seg=window._arrSegs[si]; if(!seg) return;
      totalDur+=seg.duration;
      const b=document.createElement('div');
      b.className='arr-block'; b.style.background=seg.color;
      b.draggable=true; b.dataset.pos=pos;
      b.innerHTML='<span class="seg-label">'+seg.label+'</span>'+
        '<span class="seg-info"><span>'+fmt(seg.start)+' – '+fmt(seg.end)+'</span>'+
        '<span>'+dur(seg.duration)+'</span></span>'+
        '<button class="remove-btn" data-pos="'+pos+'">✕</button>';
      b.querySelector('.remove-btn').addEventListener('click',e=>{
        e.stopPropagation();
        window._arrOrder.splice(parseInt(e.currentTarget.dataset.pos),1); renderArr();
      });
      b.addEventListener('dragstart',e=>{
        dragArrPos=parseInt(e.currentTarget.dataset.pos); dragSrcIdx=null;
        e.dataTransfer.effectAllowed='move'; e.stopPropagation();
      });
      b.addEventListener('dragover',e=>{
        e.preventDefault(); e.stopPropagation();
        const mid=e.currentTarget.getBoundingClientRect().top+e.currentTarget.getBoundingClientRect().height/2;
        e.currentTarget.classList.remove('drag-over-top','drag-over-bottom');
        e.currentTarget.classList.add(e.clientY<mid?'drag-over-top':'drag-over-bottom');
      });
      b.addEventListener('dragleave',e=>e.currentTarget.classList.remove('drag-over-top','drag-over-bottom'));
      b.addEventListener('drop',e=>{
        e.preventDefault(); e.stopPropagation();
        const t=e.currentTarget;
        t.classList.remove('drag-over-top','drag-over-bottom');
        const tPos=parseInt(t.dataset.pos);
        const after=e.clientY>=t.getBoundingClientRect().top+t.getBoundingClientRect().height/2;
        if(dragSrcIdx!==null){
          window._arrOrder.splice(after?tPos+1:tPos,0,dragSrcIdx);
        } else if(dragArrPos!==null && dragArrPos!==tPos){
          const[item]=window._arrOrder.splice(dragArrPos,1);
          const np=dragArrPos<tPos?tPos-1:tPos;
          window._arrOrder.splice(after?np+1:np,0,item);
        }
        renderArr();
      });
      track.appendChild(b);
    });
    const s=document.getElementById('arr-summary');
    if(s) s.textContent=LS().summaryFn(window._arrOrder.length, dur(totalDur));
    syncGradio();
  }

  function setupDrop(){
    const track=document.getElementById('arrangement-track');
    if(!track||track._dropReady) return;
    track._dropReady=true;
    track.addEventListener('dragover',e=>{
      if(e.target===track||e.target.classList.contains('arrangement-empty')){
        e.preventDefault(); track.classList.add('drag-over');
      }
    });
    track.addEventListener('dragleave',e=>{ if(!track.contains(e.relatedTarget)) track.classList.remove('drag-over'); });
    track.addEventListener('drop',e=>{
      if(e.target===track||e.target.classList.contains('arrangement-empty')){
        e.preventDefault(); track.classList.remove('drag-over');
        if(dragSrcIdx!==null){ window._arrOrder.push(dragSrcIdx); renderArr(); }
      }
    });
  }

    window.arrRender = function(segsJson, audioUrl){
    try{ window._arrSegs=JSON.parse(segsJson||'[]'); }catch(e){ console.error('arrRender parse error',e); return; }
    window._arrOrder=[];
    if (window._previewAudio) { window._previewAudio.pause(); window._previewAudio=null; }
    // 분석 시 발급된 토큰 기반 미리듣기 URL만 사용한다.
    window.arrAudioUrl = audioUrl || null;
    renderSource(); renderArr(); setupDrop();
  };

  window.arrClear = function(){ window._arrOrder=[]; renderArr(); };

  // ── Language switch observer ──────────────────────────────────────────────
  const langObserver = new MutationObserver(mutations => {
    for(const m of mutations){
      for(const node of m.addedNodes){
        if(node.nodeType!==1) continue;
        const t = node.id==='lang-trigger' ? node : (node.querySelector && node.querySelector('#lang-trigger'));
        if(t && t.dataset.lang){ window._currentLang = t.dataset.lang; renderSource(); renderArr(); return; }
      }
      if(m.type==='attributes' && m.target.id==='lang-trigger'){
        window._currentLang = m.target.dataset.lang; renderSource(); renderArr(); return;
      }
    }
  });
  langObserver.observe(document.body, {childList:true, subtree:true, attributes:true, attributeFilter:['data-lang']});

  // ── MutationObserver: #arr-trigger[data-segments] 감지 ───────────────────
  const observer = new MutationObserver(mutations => {
    for(const m of mutations){
      // 새 노드 추가 감지
      for(const node of m.addedNodes){
        if(node.nodeType!==1) continue;
        const trigger = node.id==='arr-trigger' ? node : node.querySelector && node.querySelector('#arr-trigger');
        if(trigger && trigger.dataset.segments){
          window.arrRender(trigger.dataset.segments, trigger.dataset.previewUrl || null);
          return;
        }
      }
      // data-segments 속성 변경 감지
      if(m.type==='attributes' && m.target.id==='arr-trigger'){
        window.arrRender(m.target.dataset.segments, m.target.dataset.previewUrl || null);
        return;
      }
    }
  });
  observer.observe(document.body, {childList:true, subtree:true, attributes:true, attributeFilter:['data-segments']});

  // 초기 드롭존 설정 (페이지 로드 시 이미 HTML 있을 경우)
  setTimeout(setupDrop, 800);
})();
"""

# ── UI ────────────────────────────────────────────────────────────────────────

DEFAULT_STEMS = ["vocals", "accompaniment"]


def make_outputs():
    return [
        _audio_component(label="보컬", visible=True),
        _audio_component(label="반주", visible=True),
        _audio_component(label="드럼", visible=False),
        _audio_component(label="베이스", visible=False),
        _audio_component(label="기타 악기", visible=False),
    ]


def build_ui():
    with gr.Blocks(title=f"MR Maker & Arranger {APP_VERSION_LABEL}", css=ARRANGER_CSS, js=ARRANGER_JS) as demo:

        # ── 최상단: 타이틀 + 언어 선택 ─────────────────────────
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown(f"# MR Maker & Arranger {APP_VERSION_LABEL}")
            with gr.Column(scale=0, min_width=130):
                with gr.Row():
                    lang_ko_btn = gr.Button("한국어", size="sm", variant="primary")
                    lang_en_btn = gr.Button("English", size="sm", variant="secondary")
        # 상태
        current_lang = gr.State("ko")

        with gr.Tabs():

            # ══════════════════════════════════════════════════════════════════
            # 탭 1: MR Maker (파일 업로드 + YouTube URL 통합)
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("MR Maker"):
                mrmaker_desc_md = gr.Markdown(_mrmaker_desc_text("ko"))

                with gr.Tabs():

                    # ── 파일 업로드 ───────────────────────────────────────
                    with gr.Tab("파일 업로드"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                file_input = gr.Audio(label="음원 파일", type="filepath", sources=["upload"])
                                file_stems = gr.CheckboxGroup(
                                    choices=STRINGS["ko"]["stem_choices"],
                                    value=DEFAULT_STEMS,
                                    label="추출할 트랙",
                                )
                                file_profile = gr.Radio(
                                    choices=STRINGS["ko"]["profile_choices"],
                                    value=DEFAULT_PROFILE,
                                    label=STRINGS["ko"]["profile_label"],
                                )
                                file_profile_help = gr.Markdown(_profile_help_text("ko", DEFAULT_PROFILE))
                                quick_url_disabled_md = gr.Markdown(
                                    STRINGS["ko"]["remote_url_disabled"],
                                    visible=not REMOTE_URL_ENABLED,
                                )
                                quick_url_md = gr.Markdown(
                                    STRINGS["ko"]["quick_url_md"],
                                    visible=REMOTE_URL_ENABLED,
                                )
                                quick_url_input = gr.Textbox(
                                    label=STRINGS["ko"]["url_label"],
                                    placeholder=STRINGS["ko"]["url_placeholder"],
                                    lines=2,
                                    visible=REMOTE_URL_ENABLED,
                                )
                                quick_url_btn = gr.Button(
                                    STRINGS["ko"]["dl_separate_btn"],
                                    variant="secondary",
                                    visible=REMOTE_URL_ENABLED,
                                )
                                file_btn = gr.Button("분리 시작", variant="primary", size="lg")
                            with gr.Column(scale=1):
                                file_results_md = gr.Markdown("### 분리 결과")
                                file_title = gr.Textbox(label="파일명", interactive=False, visible=True)
                                file_outputs = make_outputs()

                        file_stems.change(
                            fn=visibility_update,
                            inputs=file_stems,
                            outputs=file_outputs,
                            queue=False,
                        )
                        file_profile.change(
                            fn=_profile_help_text,
                            inputs=[current_lang, file_profile],
                            outputs=[file_profile_help],
                            queue=False,
                        )
                        file_btn.click(
                            fn=mrmaker_from_file,
                            inputs=[file_input, file_stems, file_profile, current_lang],
                            outputs=[file_title] + file_outputs,
                        )
                        quick_url_btn.click(
                            fn=mrmaker_from_url,
                            inputs=[quick_url_input, file_stems, file_profile, current_lang],
                            outputs=[file_title] + file_outputs,
                        )

                    # ── YouTube URL ───────────────────────────────────────
                    with gr.Tab("YouTube / YouTube Music URL", visible=REMOTE_URL_ENABLED):
                        with gr.Row():
                            with gr.Column(scale=1):
                                url_input = gr.Textbox(
                                    label="YouTube URL",
                                    placeholder="https://www.youtube.com/watch?v=...\nhttps://music.youtube.com/watch?v=...",
                                    lines=2,
                                )
                                url_stems = gr.CheckboxGroup(
                                    choices=STRINGS["ko"]["stem_choices"],
                                    value=DEFAULT_STEMS,
                                    label="추출할 트랙",
                                )
                                url_profile = gr.Radio(
                                    choices=STRINGS["ko"]["profile_choices"],
                                    value=DEFAULT_PROFILE,
                                    label=STRINGS["ko"]["profile_label"],
                                )
                                url_profile_help = gr.Markdown(_profile_help_text("ko", DEFAULT_PROFILE))
                                url_btn = gr.Button("다운로드 후 분리", variant="primary", size="lg")
                            with gr.Column(scale=1):
                                url_results_md = gr.Markdown("### 분리 결과")
                                url_title = gr.Textbox(label="곡 제목", interactive=False)
                                url_outputs = make_outputs()

                        url_stems.change(
                            fn=visibility_update,
                            inputs=url_stems,
                            outputs=url_outputs,
                            queue=False,
                        )
                        url_profile.change(
                            fn=_profile_help_text,
                            inputs=[current_lang, url_profile],
                            outputs=[url_profile_help],
                            queue=False,
                        )
                        url_btn.click(
                            fn=mrmaker_from_url,
                            inputs=[url_input, url_stems, url_profile, current_lang],
                            outputs=[url_title] + url_outputs,
                        )


            # ══════════════════════════════════════════════════════════════════
            # 탭 2: 편곡 (Arranger)
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("Arranger / 편곡"):
                arranger_desc_md = gr.Markdown(_arranger_desc_text("ko"))

                with gr.Row():
                    # ── 왼쪽: 소스 + 컨트롤 ──────────────────────────────
                    with gr.Column(scale=1):
                        arr_url_disabled_md = gr.Markdown(
                            STRINGS["ko"]["remote_url_disabled"],
                            visible=not REMOTE_URL_ENABLED,
                        )
                        with gr.Tabs():
                            with gr.Tab("파일 업로드"):
                                arr_file = gr.Audio(label="음원 파일", type="filepath", sources=["upload"])
                                arr_file_btn = gr.Button("불러오기", variant="secondary")
                            with gr.Tab("YouTube URL", visible=REMOTE_URL_ENABLED):
                                arr_url = gr.Textbox(
                                    label="YouTube URL",
                                    placeholder="https://www.youtube.com/watch?v=...",
                                    lines=2,
                                )
                                arr_url_btn = gr.Button("다운로드", variant="secondary")

                        arr_status = gr.Textbox(label="상태", interactive=False)

                        # gr.State: visible=False Textbox 대신 사용 (Gradio 6 호환)
                        arr_path   = gr.State("")    # 오디오 파일 경로
                        segs_state = gr.State("[]")  # 분석된 segments JSON

                        arr_analyze_btn = gr.Button("송폼 분석", variant="primary", size="lg")
                        gr.Markdown("---")
                        crossfade = gr.Slider(minimum=0, maximum=500, value=200, step=50,
                                              label="크로스페이드 (ms)", info="0 = 하드 컷")
                        arr_export_btn = gr.Button("편곡 내보내기", variant="secondary", size="lg")

                    # ── 오른쪽: 드래그앤드롭 UI ───────────────────────────
                    with gr.Column(scale=2):
                        arr_html = gr.HTML(value=get_arranger_html_shell("ko"))
                        # JS가 직접 업데이트하는 arrangement 순서 — visible Textbox 필수
                        arr_state = gr.Textbox(
                            value="[]", elem_id="arrangement-state",
                            label="편곡 순서", lines=1,
                            info="드래그앤드롭 결과가 자동 기록됩니다.",
                        )

                arr_result = _audio_component(label="편곡 결과", visible=True, interactive=False)

                # ── 이벤트 ────────────────────────────────────────────────
                arr_file.change(
                    fn=lambda p: (p or "", f"✓ {Path(p).name}" if p else ""),
                    inputs=[arr_file], outputs=[arr_path, arr_status],
                    queue=False,
                )
                arr_file_btn.click(
                    fn=arranger_load_file,
                    inputs=[arr_file, current_lang], outputs=[arr_path, arr_status],
                )
                arr_url_btn.click(
                    fn=arranger_download_url,
                    inputs=[arr_url, current_lang], outputs=[arr_path, arr_status],
                )
                arr_analyze_btn.click(
                    fn=lambda lang: STRINGS[_normalize_lang(lang)]["analyzing"],
                    inputs=[current_lang],
                    outputs=[arr_status],
                ).then(
                    fn=run_analysis,
                    inputs=[arr_path, current_lang],
                    outputs=[segs_state, arr_html],
                ).then(
                    fn=lambda lang: STRINGS[_normalize_lang(lang)]["analysis_done"],
                    inputs=[current_lang],
                    outputs=[arr_status],
                )
                arr_export_btn.click(
                    fn=run_export,
                    inputs=[arr_path, segs_state, arr_state, crossfade, current_lang],
                    outputs=[arr_result],
                )

        # ── 언어 변경 핸들러 ──────────────────────────────────────────────────
        def change_language(lang, cur_file_profile, cur_url_profile):
            lang = _normalize_lang(lang)
            s = STRINGS[lang]
            cur_file_profile = cur_file_profile or DEFAULT_PROFILE
            cur_url_profile = cur_url_profile or DEFAULT_PROFILE
            return (
                # current_lang state
                lang,
                # language buttons
                _lang_button_update(lang, "ko"),
                _lang_button_update(lang, "en"),
                # mrmaker_desc_md
                gr.update(value=_mrmaker_desc_text(lang)),
                # file_input label
                gr.update(label=s["audio_file_label"]),
                # file_stems label+choices
                gr.update(label=s["stems_label"], choices=s["stem_choices"]),
                # file_profile label+choices
                gr.update(label=s["profile_label"], choices=s["profile_choices"]),
                # file_profile_help
                gr.update(value=_profile_help_text(lang, cur_file_profile)),
                # quick_url_disabled_md
                gr.update(value=s["remote_url_disabled"]),
                # quick_url_md
                gr.update(value=s["quick_url_md"]),
                # quick_url_input label+placeholder
                gr.update(label=s["url_label"], placeholder=s["url_placeholder"]),
                # quick_url_btn
                gr.update(value=s["dl_separate_btn"]),
                # file_btn
                gr.update(value=s["start_btn"]),
                # file_results_md
                gr.update(value=s["results_md"]),
                # file_title label
                gr.update(label=s["filename_label"]),
                # file_outputs[0..4] labels
                gr.update(label=s["stem_vocals"]),
                gr.update(label=s["stem_accomp"]),
                gr.update(label=s["stem_drums"]),
                gr.update(label=s["stem_bass"]),
                gr.update(label=s["stem_other"]),
                # url_input label+placeholder
                gr.update(label=s["url_label"], placeholder=s["url_placeholder"]),
                # url_stems label+choices
                gr.update(label=s["stems_label"], choices=s["stem_choices"]),
                # url_profile label+choices
                gr.update(label=s["profile_label"], choices=s["profile_choices"]),
                # url_profile_help
                gr.update(value=_profile_help_text(lang, cur_url_profile)),
                # url_btn
                gr.update(value=s["dl_separate_btn"]),
                # url_results_md
                gr.update(value=s["results_md"]),
                # url_title label
                gr.update(label=s["song_title_label"]),
                # url_outputs[0..4] labels
                gr.update(label=s["stem_vocals"]),
                gr.update(label=s["stem_accomp"]),
                gr.update(label=s["stem_drums"]),
                gr.update(label=s["stem_bass"]),
                gr.update(label=s["stem_other"]),
                # arranger_desc_md
                gr.update(value=_arranger_desc_text(lang)),
                # arr_url_disabled_md
                gr.update(value=s["remote_url_disabled"]),
                # arr_file_btn
                gr.update(value=s["load_btn"]),
                # arr_url_btn
                gr.update(value=s["arr_dl_btn"]),
                # arr_status label
                gr.update(label=s["status_label"]),
                # arr_analyze_btn
                gr.update(value=s["analyze_btn"]),
                # crossfade label+info
                gr.update(label=s["crossfade_label"], info=s["crossfade_info"]),
                # arr_export_btn
                gr.update(value=s["export_btn"]),
                # arr_state label+info
                gr.update(label=s["arr_state_label"], info=s["arr_state_info"]),
                # arr_result label
                gr.update(label=s["arr_result_label"]),
                # arr_html — rebuild with new lang (includes lang-trigger div)
                gr.update(value=get_arranger_html_shell(lang)),
            )

        lang_ko_btn.click(
            fn=lambda cur_file_profile, cur_url_profile: change_language(
                "ko", cur_file_profile, cur_url_profile
            ),
            inputs=[file_profile, url_profile],
            outputs=[
                current_lang,
                lang_ko_btn,
                lang_en_btn,
                mrmaker_desc_md,
                file_input,
                file_stems,
                file_profile,
                file_profile_help,
                quick_url_disabled_md,
                quick_url_md,
                quick_url_input,
                quick_url_btn,
                file_btn,
                file_results_md,
                file_title,
                file_outputs[0], file_outputs[1], file_outputs[2], file_outputs[3], file_outputs[4],
                url_input,
                url_stems,
                url_profile,
                url_profile_help,
                url_btn,
                url_results_md,
                url_title,
                url_outputs[0], url_outputs[1], url_outputs[2], url_outputs[3], url_outputs[4],
                arranger_desc_md,
                arr_url_disabled_md,
                arr_file_btn,
                arr_url_btn,
                arr_status,
                arr_analyze_btn,
                crossfade,
                arr_export_btn,
                arr_state,
                arr_result,
                arr_html,
            ],
            queue=False,
        )

        lang_en_btn.click(
            fn=lambda cur_file_profile, cur_url_profile: change_language(
                "en", cur_file_profile, cur_url_profile
            ),
            inputs=[file_profile, url_profile],
            outputs=[
                current_lang,
                lang_ko_btn,
                lang_en_btn,
                mrmaker_desc_md,
                file_input,
                file_stems,
                file_profile,
                file_profile_help,
                quick_url_disabled_md,
                quick_url_md,
                quick_url_input,
                quick_url_btn,
                file_btn,
                file_results_md,
                file_title,
                file_outputs[0], file_outputs[1], file_outputs[2], file_outputs[3], file_outputs[4],
                url_input,
                url_stems,
                url_profile,
                url_profile_help,
                url_btn,
                url_results_md,
                url_title,
                url_outputs[0], url_outputs[1], url_outputs[2], url_outputs[3], url_outputs[4],
                arranger_desc_md,
                arr_url_disabled_md,
                arr_file_btn,
                arr_url_btn,
                arr_status,
                arr_analyze_btn,
                crossfade,
                arr_export_btn,
                arr_state,
                arr_result,
                arr_html,
            ],
            queue=False,
        )

    # default_concurrency_limit은 Gradio 5.x에서 제거됨 → 버전 방어 처리
    try:
        demo.queue(max_size=10, default_concurrency_limit=1)
    except TypeError:
        demo.queue(max_size=10)
    return demo


if __name__ == "__main__":
    demo = build_ui()

    if LAUNCH_SUPPORTS_APP:
        # launch()가 _app 인자를 지원하는 버전에서는 미리듣기용 라우트를 연결한다.
        from gradio.routes import App as _GrApp
        from fastapi.responses import FileResponse as _FResp
        from fastapi import Response as _Resp

        _base_app = _GrApp()

        @_base_app.get("/preview-audio")
        async def preview_audio(token: str | None = None):
            path = _preview_audio_files.get(token or "")
            if not path or not os.path.exists(path):
                return _Resp(status_code=404, content="분석 먼저 실행")
            import mimetypes
            mt = mimetypes.guess_type(path)[0] or "audio/wav"
            return _FResp(path, media_type=mt, headers={"Cache-Control": "no-store"})

        demo.launch(
            server_name=APP_HOST,
            server_port=APP_PORT,
            share=False,
            inbrowser=not IS_HF_SPACE,
            _app=_base_app,
        )
    else:
        # 구버전 Gradio(예: 5.x)는 _app 인자를 지원하지 않는다.
        demo.launch(
            server_name=APP_HOST,
            server_port=APP_PORT,
            share=False,
            inbrowser=not IS_HF_SPACE,
        )
