import json
import mimetypes
import os
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse, JSONResponse

from app import (
    STEM_KEYS,
    download_audio,
    estimate_audio_bpm,
    separate_stems,
)

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
SEPARATOR_API_TOKEN = os.getenv("SEPARATOR_API_TOKEN", "").strip()
SEPARATOR_PUBLIC_BASE_URL = os.getenv("SEPARATOR_PUBLIC_BASE_URL", "").strip().rstrip("/")
ARTIFACT_MAX_ITEMS = max(8, int(os.getenv("ARTIFACT_MAX_ITEMS", "96")))

app = FastAPI(title="Vocal Separator Backend v2", version="2.0.0")
_artifacts: OrderedDict[str, str] = OrderedDict()


def _http_error(detail: str, status_code: int = 400) -> HTTPException:
    return HTTPException(status_code=status_code, detail=detail)


def _require_api_key(x_api_key: str | None) -> None:
    if SEPARATOR_API_TOKEN and x_api_key != SEPARATOR_API_TOKEN:
        raise _http_error("Invalid API key.", status_code=401)


def _register_artifact(path: str) -> str:
    token = next(tempfile._get_candidate_names())
    _artifacts[token] = os.path.realpath(path)
    _artifacts.move_to_end(token)
    while len(_artifacts) > ARTIFACT_MAX_ITEMS:
        _, evicted_path = _artifacts.popitem(last=False)
        try:
            os.unlink(evicted_path)
        except OSError:
            pass
    return token


def _artifact_url(request: Request, token: str) -> str:
    if SEPARATOR_PUBLIC_BASE_URL:
        return f"{SEPARATOR_PUBLIC_BASE_URL}/artifacts/{token}"
    return str(request.url_for("get_artifact", token=token))


def _cleanup_download(audio_path: str) -> None:
    if not audio_path:
        return
    audio_file = Path(audio_path)
    try:
        if audio_file.exists():
            audio_file.unlink()
    except OSError:
        pass
    try:
        parent = audio_file.parent
        if parent.name.startswith("tmp") and parent.exists():
            parent.rmdir()
    except OSError:
        pass


def _normalize_stems(stems: list[str]) -> list[str]:
    unique = []
    for stem in stems:
        if stem in STEM_KEYS and stem not in unique:
            unique.append(stem)
    if not unique:
        raise _http_error("At least one valid stem is required.")
    return unique


def _parse_form_stems(raw: str) -> list[str]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _http_error("Invalid stems payload.") from exc
    if not isinstance(parsed, list):
        raise _http_error("Stems payload must be a JSON array.")
    return _normalize_stems(parsed)


def _separation_response(
    request: Request,
    title: str,
    paths: list[str],
    stems: list[str],
    bpm: float | None = None,
) -> dict:
    stem_urls = {}
    for stem, path in zip(STEM_KEYS, paths):
        if stem in stems and path:
            stem_urls[stem] = _artifact_url(request, _register_artifact(path))
    payload = {"title": title, "stems": stem_urls}
    if bpm is not None:
        payload["bpm"] = bpm
    return payload


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <title>MR Maker Backend v2</title>
        <style>
          body { font-family: sans-serif; max-width: 760px; margin: 48px auto; line-height: 1.6; color: #111; }
          code { background: #f3f4f6; padding: 2px 6px; border-radius: 6px; }
        </style>
      </head>
      <body>
        <h1>MR Maker Backend v2</h1>
        <p>The backend is online.</p>
        <p>Health check: <a href="/health"><code>/health</code></a></p>
        <p>This origin serves API requests and temporary artifacts for the MR Maker frontend.</p>
      </body>
    </html>
    """


@app.get("/health")
async def healthcheck():
    return {"ok": True}


@app.get("/artifacts/{token}", name="get_artifact")
async def get_artifact(token: str):
    path = _artifacts.get(token)
    if not path or not os.path.exists(path):
        raise _http_error("Artifact not found.", status_code=404)
    _artifacts.move_to_end(token)
    media_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, headers={"Cache-Control": "no-store"})


@app.post("/api/download-url")
async def api_download_url(
    payload: dict,
    request: Request,
    x_api_key: str | None = Header(default=None),
):
    _require_api_key(x_api_key)
    url = str(payload.get("url", "")).strip()
    lang = str(payload.get("lang", "ko"))
    if not url:
        raise _http_error("URL is required.")
    try:
        audio_path, title = download_audio(url, lang)
    except Exception as exc:
        raise _http_error(str(exc)) from exc
    return {"title": title, "audio_url": _artifact_url(request, _register_artifact(audio_path))}


@app.post("/api/separate-url")
async def api_separate_url(
    payload: dict,
    request: Request,
    x_api_key: str | None = Header(default=None),
):
    _require_api_key(x_api_key)
    url = str(payload.get("url", "")).strip()
    profile = str(payload.get("profile", "balanced"))
    lang = str(payload.get("lang", "ko"))
    stems = _normalize_stems(payload.get("stems") or [])
    if not url:
        raise _http_error("URL is required.")

    audio_path = ""
    try:
        audio_path, title = download_audio(url, lang)
        bpm = estimate_audio_bpm(audio_path)
        paths = separate_stems(audio_path, stems, profile)
        return _separation_response(request, title, paths, stems, bpm)
    except Exception as exc:
        raise _http_error(str(exc)) from exc
    finally:
        _cleanup_download(audio_path)


@app.post("/api/separate-file")
async def api_separate_file(
    request: Request,
    file: UploadFile = File(...),
    stems: str = Form(...),
    profile: str = Form("balanced"),
    lang: str = Form("ko"),
    x_api_key: str | None = Header(default=None),
):
    _require_api_key(x_api_key)
    requested_stems = _parse_form_stems(stems)

    suffix = Path(file.filename or "").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="upload_") as tmp:
        shutil.copyfileobj(file.file, tmp)
        upload_path = tmp.name

    try:
        bpm = estimate_audio_bpm(upload_path)
        paths = separate_stems(upload_path, requested_stems, profile)
        title = Path(file.filename or upload_path).stem
        return _separation_response(request, title, paths, requested_stems, bpm)
    except Exception as exc:
        raise _http_error(str(exc)) from exc
    finally:
        try:
            os.unlink(upload_path)
        except OSError:
            pass


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend:app", host=API_HOST, port=API_PORT, reload=False)
