"""
Add the robot's voiceprint to voiceprints.json using robot_voice.wav.

If the "robot" voiceprint already exists in voiceprints.json, does nothing.
Otherwise creates a voiceprint via pyannote and saves it.

Usage (from project root):
  python -m memobot.ingest_pipeline.add_robot_voiceprint   # uses ingest_pipeline/data/robot_voice.wav
  python -m memobot.ingest_pipeline.add_robot_voiceprint /path/to/robot_voice.wav
"""
import sys
import uuid
from pathlib import Path

# Ensure package root is on path when run as script
_INGEST_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _INGEST_ROOT.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from memobot.ingest_pipeline.speaker_diarization.enroll_from_local_wav import (
    load_voiceprints,
    save_voiceprints,
    create_voiceprint_from_clip,
    upload_local_wav_to_media,
    to_wav_16k_mono,
)

ROBOT_LABEL = "robot"
DEFAULT_ROBOT_WAV = _INGEST_ROOT / "data" / "robot_voice.wav"


def main(wav_path: Path | None = None) -> None:
    wav_path = wav_path or DEFAULT_ROBOT_WAV
    wav_path = wav_path.resolve()

    if not wav_path.exists():
        raise FileNotFoundError(f"Robot voice WAV not found: {wav_path}")

    voiceprints = load_voiceprints()
    if ROBOT_LABEL in voiceprints:
        print("Robot voiceprint already exists in voiceprints.json; skipping.")
        return

    print(f"Creating robot voiceprint from {wav_path}...")
    norm_path = _INGEST_ROOT / f"robot_voice_16k_{uuid.uuid4().hex[:8]}.wav"
    try:
        to_wav_16k_mono(str(wav_path), str(norm_path))
        wav_bytes = norm_path.read_bytes()
    finally:
        if norm_path.exists():
            norm_path.unlink()

    media_key = f"robot_voice/{uuid.uuid4().hex}.wav"
    clip_media_url = upload_local_wav_to_media(wav_bytes, media_key)

    vp_str = create_voiceprint_from_clip(clip_media_url)
    voiceprints[ROBOT_LABEL] = {
        "label": ROBOT_LABEL,
        "voiceprint": vp_str,
        "created_from": {
            "source_file": str(wav_path),
            "note": "robot",
        },
    }
    save_voiceprints(voiceprints)
    print(f"Saved robot voiceprint to voiceprints.json.")


if __name__ == "__main__":
    wav_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    main(wav_path)
