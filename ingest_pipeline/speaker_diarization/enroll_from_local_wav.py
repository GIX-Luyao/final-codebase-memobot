import os
import time
import json
import uuid
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import requests
from dotenv import load_dotenv
# ----------------------------
# Config
# ----------------------------


load_dotenv()
PYANNOTE_API_KEY = os.getenv("PYANNOTE_API_KEY")
if not PYANNOTE_API_KEY:
    raise RuntimeError("Set env var PYANNOTE_API_KEY")

DATA_DIR = Path("./data")
STATE_DIR = Path("./state")
STATE_DIR.mkdir(exist_ok=True)

VOICEPRINTS_PATH = STATE_DIR / "voiceprints.json"

# pyannoteAI endpoints
MEDIA_INPUT = "https://api.pyannote.ai/v1/media/input"     # presigned PUT for media://...
DIARIZE = "https://api.pyannote.ai/v1/diarize"             # diarize (+ transcription)
IDENTIFY = "https://api.pyannote.ai/v1/identify"           # identify against voiceprints
VOICEPRINT = "https://api.pyannote.ai/v1/voiceprint"       # create voiceprint
JOBS = "https://api.pyannote.ai/v1/jobs/{jobId}"           # poll job results

HEADERS_JSON = {"Authorization": f"Bearer {PYANNOTE_API_KEY}", "Content-Type": "application/json"}
HEADERS_AUTH = {"Authorization": f"Bearer {PYANNOTE_API_KEY}"}

# Enrollment / matching knobs
MIN_TURN_SEC = 1.0
TARGET_VP_SEC = 20.0          # build ~20s clip (<=30s)
MAX_VP_SEC = 30.0
IDENTIFY_THRESHOLD = 50       # tune per your domain
MIN_CONFIDENCE_SCORE = 70.0   # minimum identification confidence (0-100) to accept a match, else create new speaker
                              # Lower values = more permissive (more false matches)
                              # Higher values = more strict (fewer false matches, more new speakers)


# ----------------------------
# Local audio helpers (ffmpeg)
# ----------------------------
def run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def to_wav_16k_mono(src_path: str, out_path: str) -> None:
    run(["ffmpeg", "-y", "-i", src_path, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out_path])

def cut_wav(src_wav: str, out_wav: str, start: float, end: float) -> None:
    run([
        "ffmpeg", "-y",
        "-i", src_wav,
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le",
        out_wav
    ])

def concat_wavs(parts: List[str], out_wav: str) -> None:
    lst = out_wav + ".txt"
    with open(lst, "w", encoding="utf-8") as f:
        for p in parts:
            f.write(f"file '{os.path.abspath(p)}'\n")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out_wav])


# ----------------------------
# Persistence
# ----------------------------
def load_voiceprints() -> Dict[str, Any]:
    if not VOICEPRINTS_PATH.exists():
        return {}
    return json.loads(VOICEPRINTS_PATH.read_text(encoding="utf-8"))

def save_voiceprints(vps: Dict[str, Any]) -> None:
    VOICEPRINTS_PATH.write_text(json.dumps(vps, indent=2), encoding="utf-8")


# ----------------------------
# pyannoteAI helpers
# ----------------------------
def media_put_url(media_key: str) -> str:
    r = requests.post(MEDIA_INPUT, headers=HEADERS_JSON, json={"url": f"media://{media_key}"}, timeout=60)
    r.raise_for_status()
    return r.json()["url"]

def upload_bytes(presigned_put_url: str, data: bytes) -> None:
    r = requests.put(presigned_put_url, data=data, headers={"Content-Type": "application/octet-stream"}, timeout=300)
    r.raise_for_status()

def submit_job(endpoint: str, payload: dict) -> dict:
    r = requests.post(endpoint, headers=HEADERS_JSON, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def poll_job(job_id: str, timeout_s: int = 240, sleep_s: float = 1.2) -> dict:
    t0 = time.time()
    while True:
        r = requests.get(JOBS.format(jobId=job_id), headers=HEADERS_AUTH, timeout=30)
        r.raise_for_status()
        job = r.json()
        st = job.get("status")

        if st == "succeeded":
            return job
        if st in ("failed", "canceled"):
            raise RuntimeError(f"Job {job_id} ended with status={st}: {json.dumps(job, indent=2)[:2000]}")

        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Timed out waiting for job {job_id} (last status={st})")

        time.sleep(sleep_s)

def upload_local_wav_to_media(wav_bytes: bytes, media_key: str) -> str:
    put = media_put_url(media_key)
    upload_bytes(put, wav_bytes)
    return f"media://{media_key}"


# ----------------------------
# Pipeline steps
# ----------------------------
def speech_to_text_diarization(audio_media_url: str) -> List[dict]:
    """
    Returns turnLevelTranscription:
      [{speaker, start, end, text}, ...]
    """
    job = submit_job(DIARIZE, {
        "url": audio_media_url,
        "model": "precision-2",
        "transcription": True,
        "exclusive": True,
        "turnLevelConfidence": True  # Enable turn-level confidence scores
        # optional:
        # "transcriptionConfig": {"model": "faster-whisper-large-v3-turbo"}
    })
    out = poll_job(job["jobId"])["output"]
    turns = out.get("turnLevelTranscription", []) or []
    # normalize floats + text
    norm = []
    for t in turns:
        norm.append({
            "speaker": t["speaker"],
            "start": float(t["start"]),
            "end": float(t["end"]),
            "text": (t.get("text") or "").strip(),
        })
    return norm

def build_single_speaker_clip_bytes(local_wav_16k: str, turns: List[dict], speaker: str) -> bytes:
    """
    Build a <=30s single-speaker enrollment/eval clip by concatenating that speaker's turns.
    """
    speaker_turns = [t for t in turns if t["speaker"] == speaker and (t["end"] - t["start"]) >= MIN_TURN_SEC]
    if not speaker_turns:
        raise ValueError(f"No usable turns for {speaker}")

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        parts = []
        total = 0.0
        for i, t in enumerate(speaker_turns):
            dur = float(t["end"] - t["start"])
            if total >= TARGET_VP_SEC or (total + dur) > MAX_VP_SEC:
                break
            part = str(Path(td) / f"p_{i:03d}.wav")
            cut_wav(local_wav_16k, part, t["start"], t["end"])
            parts.append(part)
            total += dur

        if not parts:
            raise ValueError(f"Turns too short to build clip for {speaker}")

        out_clip = str(Path(td) / "speaker_clip.wav")
        concat_wavs(parts, out_clip)
        return Path(out_clip).read_bytes()

def identify_clip(clip_media_url: str, voiceprints: Dict[str, Any]) -> Optional[Tuple[str, float]]:
    """
    Identify a single-speaker clip against existing voiceprints.
    Returns (matched label, confidence score) if confident match found, or None.
    Only returns a match if confidence meets MIN_CONFIDENCE_SCORE threshold.
    
    According to pyannote docs, identification output includes a 'voiceprints' array
    where each entry has a 'confidence' object with scores per voiceprint label.
    """
    vp_list = [{"label": v["label"], "voiceprint": v["voiceprint"]} for v in voiceprints.values()]
    if not vp_list:
        return None

    job = submit_job(IDENTIFY, {
        "url": clip_media_url,
        "model": "precision-2",
        "voiceprints": vp_list,
        "matching": {"threshold": IDENTIFY_THRESHOLD, "exclusive": True},
    })
    out = poll_job(job["jobId"])["output"]

    # According to docs, identification output has a 'voiceprints' array
    # Each entry: {"speaker": "SPEAKER_00", "match": "John Doe", "confidence": {"John Doe": 86, ...}}
    # Try both possible response structures
    voiceprints_result = out.get("voiceprints", [])
    if not voiceprints_result:
        # Fallback to 'identification' key if 'voiceprints' doesn't exist
        voiceprints_result = out.get("identification", [])
    
    if not voiceprints_result:
        return None
    
    # For single-speaker clip, we expect entries in the result array
    # Get the match and its confidence score
    best_match = None
    best_confidence = 0.0
    
    for vp_entry in voiceprints_result:
        match_label = vp_entry.get("match")
        if not match_label:
            continue
            
        # Get confidence - could be in a 'confidence' object or directly as 'score'
        confidence_obj = vp_entry.get("confidence", {})
        if isinstance(confidence_obj, dict) and match_label in confidence_obj:
            # Structure: {"confidence": {"John Doe": 86, ...}}
            confidence_score = float(confidence_obj[match_label])
        elif "score" in vp_entry:
            # Alternative structure with direct score field
            confidence_score = float(vp_entry.get("score", 0))
        else:
            # No confidence available, skip this entry
            continue
        
        # Take the match with highest confidence
        if confidence_score > best_confidence:
            best_confidence = confidence_score
            best_match = match_label
    
    # Only return match if confidence meets threshold
    if best_match and best_confidence >= MIN_CONFIDENCE_SCORE:
        return (best_match, best_confidence)
    else:
        return None

def create_voiceprint_from_clip(clip_media_url: str) -> str:
    """
    Create voiceprint string from a single-speaker clip.
    """
    job = submit_job(VOICEPRINT, {"url": clip_media_url, "model": "precision-2"})
    out = poll_job(job["jobId"])["output"]
    vp = out.get("voiceprint")
    if not vp:
        raise RuntimeError("Voiceprint job succeeded but output.voiceprint missing")
    return vp


# ----------------------------
# Entry point
# ----------------------------
def main(filename: str):
    # 0) load local wav
    src = DATA_DIR / filename
    if not src.exists():
        raise FileNotFoundError(f"Not found: {src}")

    # 0b) normalize to 16k mono locally
    norm_path = STATE_DIR / f"norm_{uuid.uuid4().hex}.wav"
    to_wav_16k_mono(str(src), str(norm_path))
    wav_bytes = norm_path.read_bytes()

    # 0c) upload full audio to pyannote media://
    audio_key = f"audio/{uuid.uuid4().hex}.wav"
    audio_media_url = upload_local_wav_to_media(wav_bytes, audio_key)

    voiceprints = load_voiceprints()

    # 1) speech-to-text diarization by turn
    turns = speech_to_text_diarization(audio_media_url)
    diar_speakers = sorted({t["speaker"] for t in turns})

    print("\n=== Turn-level transcription (diarization speakers) ===")
    for i, t in enumerate(turns, 1):
        print(f"[{i:03d}] {t['speaker']} {t['start']:.2f}-{t['end']:.2f}  {t['text']}")

    # 2) for each diarized speaker: identify clip, else enroll
    speaker_to_identity: Dict[str, str] = {}

    for spk in diar_speakers:
        print(f"\n--- Processing {spk} ---")

        # Build single-speaker clip from turns
        try:
            clip_bytes = build_single_speaker_clip_bytes(str(norm_path), turns, spk)
        except Exception as e:
            print(f"[skip] cannot build clip for {spk}: {e}")
            speaker_to_identity[spk] = "UNKNOWN"
            continue

        # Upload speaker clip to media://
        clip_key = f"clips/{uuid.uuid4().hex}_{spk}.wav"
        clip_media_url = upload_local_wav_to_media(clip_bytes, clip_key)

        # Identify against existing voiceprints
        match_result = identify_clip(clip_media_url, voiceprints)
        if match_result:
            match_label, confidence = match_result
            print(f"[match] {spk} -> {match_label} (confidence: {confidence:.1f})")
            speaker_to_identity[spk] = match_label
            continue
        else:
            print(f"[low confidence] {spk} match below threshold ({MIN_CONFIDENCE_SCORE:.1f}), creating new speaker")

        # No match => enroll new voiceprint
        new_label = f"person_{uuid.uuid4().hex[:8]}"
        print(f"[enroll] {spk} not identified -> creating new voiceprint '{new_label}'")

        vp_str = create_voiceprint_from_clip(clip_media_url)
        voiceprints[new_label] = {
            "label": new_label,
            "voiceprint": vp_str,
            "created_from": {
                "source_file": str(src),
                "diarization_speaker": spk,
            },
        }
        save_voiceprints(voiceprints)
        print(f"[saved] voiceprints.json updated with {new_label}")

        speaker_to_identity[spk] = new_label

    # Final: print turns with resolved identities
    print("\n=== Final turn-level transcript (with identities) ===")
    for i, t in enumerate(turns, 1):
        ident = speaker_to_identity.get(t["speaker"], "UNKNOWN")
        print(f"[{i:03d}] {ident} ({t['speaker']}) {t['start']:.2f}-{t['end']:.2f}  {t['text']}")

    print(f"\nVoiceprints stored at: {VOICEPRINTS_PATH}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python enroll_from_local_wav.py <file.wav>   (file must be in ./data/)")
        raise SystemExit(2)
    main(sys.argv[1])
