"""
music_db.py
───────────
ChromaDB-backed music library.

Stores every generated/converted tune with:
  - The exact prompt used
  - Audio feature embedding (MFCC + chroma, 35-dim) for similarity search
  - Full metadata: instrument, style, key, tempo, energy, source, timestamp
  - A permanent copy of the MP3 in ./music_library/audio/

Collections
───────────
  music_tracks  — one document per track, embedded with audio features
"""

import os
import uuid
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import librosa
import chromadb
from chromadb.config import Settings

# ─── Paths ────────────────────────────────────────────────────────────────────

DB_DIR    = Path("music_library") / "chroma"
AUDIO_DIR = Path("music_library") / "audio"

DB_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# ─── ChromaDB client (persistent, local) ─────────────────────────────────────

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(DB_DIR))
        _collection = _client.get_or_create_collection(
            name="music_tracks",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ─── Audio embedding (35-dim MFCC + chroma vector) ───────────────────────────

def _extract_embedding(wav_np: np.ndarray, sr: int) -> list[float]:
    """
    Extract a 35-dimensional audio feature vector from a waveform.
    Consistent, fixed-size, good for cosine similarity search.
    """
    try:
        # Ensure mono float32
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(axis=0)
        wav_np = wav_np.astype(np.float32)

        # 20 MFCC means
        mfcc = librosa.feature.mfcc(y=wav_np, sr=sr, n_mfcc=20)
        mfcc_mean = mfcc.mean(axis=1).tolist()

        # 12 Chroma means
        chroma = librosa.feature.chroma_cqt(y=wav_np, sr=sr)
        chroma_mean = chroma.mean(axis=1).tolist()

        # 3 Spectral features
        centroid = float(librosa.feature.spectral_centroid(y=wav_np, sr=sr).mean())
        rolloff  = float(librosa.feature.spectral_rolloff(y=wav_np, sr=sr).mean())
        rms      = float(librosa.feature.rms(y=wav_np).mean())

        embedding = mfcc_mean + chroma_mean + [centroid, rolloff, rms]
        return embedding                     # 20 + 12 + 3 = 35 floats

    except Exception:
        # Fallback: random-ish embedding so insert doesn't fail
        return [0.0] * 35


# ─── Public API ───────────────────────────────────────────────────────────────

def save_track(
    tmp_mp3_path: str,
    wav_np: np.ndarray,
    sr: int,
    prompt: str,
    metadata: dict,
) -> str:
    """
    Permanently save a generated/converted track to the library.

    Parameters
    ----------
    tmp_mp3_path : str    path to the temporary MP3 file
    wav_np       : np.ndarray   waveform array (used for embedding)
    sr           : int          sample rate
    prompt       : str          the MusicGen prompt used
    metadata     : dict         any extra fields (instrument, style, key, tempo…)

    Returns
    -------
    track_id : str   UUID of the saved track
    """
    col = _get_collection()

    track_id = str(uuid.uuid4())

    # Copy MP3 to permanent storage
    perm_path = AUDIO_DIR / f"{track_id}.mp3"
    shutil.copy2(tmp_mp3_path, perm_path)

    # Build ChromaDB metadata (only str / int / float / bool allowed)
    now = datetime.now()
    chroma_meta = {
        "filepath":   str(perm_path),
        "prompt":     prompt[:500],          # ChromaDB caps metadata strings
        "timestamp":  now.isoformat(),
        "date":       now.strftime("%d %b %Y  %H:%M"),
        "source":     str(metadata.get("source", "generated")),
        "instrument": str(metadata.get("instrument", "")),
        "style":      str(metadata.get("style", "")),
        "key":        str(metadata.get("key", "")),
        "mode":       str(metadata.get("mode", "")),
        "tempo":      float(metadata.get("tempo", 0)),
        "energy":     str(metadata.get("energy", "")),
        "duration":   float(metadata.get("duration", 0)),
    }

    embedding = _extract_embedding(wav_np, sr)

    col.add(
        ids=[track_id],
        embeddings=[embedding],
        documents=[prompt],          # for full-text search
        metadatas=[chroma_meta],
    )

    return track_id


def get_all_tracks() -> list[dict]:
    """Return all tracks sorted newest-first."""
    col = _get_collection()
    if col.count() == 0:
        return []
    result = col.get(include=["metadatas", "documents"])
    tracks = []
    for i, tid in enumerate(result["ids"]):
        meta = result["metadatas"][i]
        prompt = result["documents"][i]
        tracks.append({"id": tid, "prompt": prompt, **meta})
    # Sort newest first
    tracks.sort(key=lambda t: t.get("timestamp", ""), reverse=True)
    return tracks


def delete_track(track_id: str) -> bool:
    """Delete a track from DB and remove its MP3 file."""
    col = _get_collection()
    try:
        result = col.get(ids=[track_id], include=["metadatas"])
        if result["ids"]:
            fp = result["metadatas"][0].get("filepath", "")
            if fp and os.path.exists(fp):
                os.remove(fp)
        col.delete(ids=[track_id])
        return True
    except Exception:
        return False


def search_similar_tracks(wav_np: np.ndarray, sr: int, n: int = 5) -> list[dict]:
    """Find the N most similar tracks by audio feature similarity."""
    col = _get_collection()
    if col.count() == 0:
        return []
    embedding = _extract_embedding(wav_np, sr)
    results = col.query(
        query_embeddings=[embedding],
        n_results=min(n, col.count()),
        include=["metadatas", "documents", "distances"],
    )
    tracks = []
    for i, tid in enumerate(results["ids"][0]):
        meta     = results["metadatas"][0][i]
        prompt   = results["documents"][0][i]
        distance = results["distances"][0][i]
        similarity = round((1 - distance) * 100, 1)   # cosine → %
        tracks.append({"id": tid, "prompt": prompt, "similarity": similarity, **meta})
    return tracks


def search_by_prompt(query: str, n: int = 10) -> list[dict]:
    """Simple substring filter over all stored prompts."""
    query_lower = query.lower().strip()
    all_tracks = get_all_tracks()
    if not query_lower:
        return all_tracks[:n]
    return [t for t in all_tracks if query_lower in t.get("prompt", "").lower()][:n]


def library_stats() -> dict:
    """Return summary stats for the library."""
    col = _get_collection()
    count = col.count()
    size_mb = 0.0
    try:
        for f in AUDIO_DIR.glob("*.mp3"):
            size_mb += f.stat().st_size / (1024 * 1024)
    except Exception:
        pass
    return {"count": count, "size_mb": round(size_mb, 1)}
