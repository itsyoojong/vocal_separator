"""
Audio arrangement exporter.
Takes a list of segment dicts (with start/end in seconds) and an ordering,
joins them with beat-snapped crossfades.
"""
import tempfile

import numpy as np
import soundfile as sf
import librosa


def _crossfade_join(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """
    Join two (channels, samples) arrays with an n-sample crossfade.
    """
    if n <= 0 or a.shape[1] < n or b.shape[1] < n:
        return np.concatenate([a, b], axis=1)

    fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, n, dtype=np.float32)

    overlap = a[:, -n:] * fade_out + b[:, :n] * fade_in
    return np.concatenate([a[:, :-n], overlap, b[:, n:]], axis=1)


def export_arrangement(
    audio_path: str,
    segments: list[dict],
    arrangement_indices: list[int],
    crossfade_ms: int = 200,
) -> str:
    """
    Export arranged audio.

    Args:
        audio_path:          Source audio file path.
        segments:            Full segment list from analyzer.detect_segments().
        arrangement_indices: Ordered list of segment indices for the new arrangement.
        crossfade_ms:        Crossfade duration in milliseconds.

    Returns:
        Path to the output WAV file.
    """
    if not arrangement_indices:
        raise ValueError("편곡 트랙이 비어 있습니다.")

    y, sr = librosa.load(audio_path, sr=44100, mono=False)
    if y.ndim == 1:
        y = y[np.newaxis, :]  # (1, samples)

    crossfade_samples = int(crossfade_ms * sr / 1000)

    clips = []
    for idx in arrangement_indices:
        seg = segments[idx]
        s = int(seg["start"] * sr)
        e = int(seg["end"] * sr)
        e = min(e, y.shape[1])
        clip = y[:, s:e].copy()
        clips.append(clip)

    result = clips[0]
    for clip in clips[1:]:
        result = _crossfade_join(result, clip, crossfade_samples)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="arranged_")
    # (channels, samples) → (samples, channels)
    sf.write(tmp.name, result.T, sr)
    return tmp.name
