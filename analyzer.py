"""
Song structure analysis using librosa.
Returns segment boundaries snapped to beat positions,
then clusters similar segments into labeled sections (A, B, C...).
"""
import numpy as np
import librosa
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

LABEL_NAMES = list("ABCDEFGH")

SECTION_COLORS = {
    "A": "#4A90D9",
    "B": "#E67E22",
    "C": "#2ECC71",
    "D": "#9B59B6",
    "E": "#E74C3C",
    "F": "#1ABC9C",
    "G": "#F39C12",
    "H": "#95A5A6",
}


def _snap_to_nearest_beat(time: float, beat_times: np.ndarray) -> float:
    idx = int(np.argmin(np.abs(beat_times - time)))
    return float(beat_times[idx])


def detect_segments(
    audio_path: str,
    min_dur: float = 8.0,
    n_sections: int | None = None,
) -> list[dict]:
    """
    Detect structural segments in an audio file.

    Returns a list of dicts:
        {index, start, end, duration, label, color}
    """
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))
    hop = 512

    # ── Feature extraction ────────────────────────────────────────────────────
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
    features = np.vstack([chroma, mfcc])  # (25, T)

    # ── Beat tracking (for boundary snapping) ─────────────────────────────────
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)

    # ── Structural segmentation ───────────────────────────────────────────────
    if n_sections is None:
        n_sections = max(4, min(10, int(duration / 20)))

    try:
        bound_frames = librosa.segment.agglomerative(features, k=n_sections)
        bound_times = librosa.frames_to_time(bound_frames, sr=sr, hop_length=hop)
    except Exception:
        bound_times = np.linspace(0, duration, n_sections + 1)[1:-1]

    # ── Snap boundaries to nearest beats ─────────────────────────────────────
    snapped = []
    for t in bound_times:
        snapped_t = _snap_to_nearest_beat(t, beat_times) if len(beat_times) else t
        snapped.append(snapped_t)

    all_times = np.concatenate([[0.0], sorted(set(snapped)), [duration]])

    # ── Filter out segments shorter than min_dur ──────────────────────────────
    filtered = [all_times[0]]
    for t in all_times[1:]:
        if t - filtered[-1] >= min_dur:
            filtered.append(t)
    # Ensure the last boundary is exactly the end
    if filtered[-1] < duration:
        if duration - filtered[-1] < min_dur and len(filtered) > 1:
            filtered[-1] = duration  # extend last segment
        else:
            filtered.append(duration)

    # ── Build segment feature vectors ─────────────────────────────────────────
    segments = []
    for i in range(len(filtered) - 1):
        start, end = filtered[i], filtered[i + 1]
        sf = librosa.time_to_frames(start, sr=sr, hop_length=hop)
        ef = min(librosa.time_to_frames(end, sr=sr, hop_length=hop), features.shape[1])
        if ef <= sf:
            continue
        seg_feat = features[:, sf:ef].mean(axis=1)
        segments.append({"start": float(start), "end": float(end), "feature": seg_feat})

    if not segments:
        segments = [{"start": 0.0, "end": duration, "feature": np.zeros(25)}]

    # ── Cluster similar segments into labels A/B/C/... ───────────────────────
    n_clusters = min(len(segments), 5)
    if len(segments) > 1:
        feat_matrix = np.array([s["feature"] for s in segments])
        feat_matrix = StandardScaler().fit_transform(feat_matrix)
        raw_labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(feat_matrix)
    else:
        raw_labels = [0]

    result = []
    for i, (s, raw) in enumerate(zip(segments, raw_labels)):
        label = LABEL_NAMES[int(raw) % len(LABEL_NAMES)]
        result.append(
            {
                "index": i,
                "start": s["start"],
                "end": s["end"],
                "duration": s["end"] - s["start"],
                "label": label,
                "color": SECTION_COLORS.get(label, "#888"),
            }
        )

    return result


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"
