#!/usr/bin/env python3
"""
Process Video:
1. Extract audio and run speaker diarization (Pyannote).
2. For each turn, check against voiceprints.json robot only; skip robot voice.
3. For non-robot turns: get frame when that person is talking; run TalkNet and
   match face to facial_database (DeepFace) in parallel across segments.
4. Output: JSON with audio_dialogue (person_id-to-voice) and array of person_ids.
   Person DB has no speaker_id; identity is by face_id -> person_id only.
"""

import os
import sys
import uuid
import subprocess
import tempfile
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

# #region agent log
print("DEBUG_LOG: process_video.py - Top of file")
# #endregion

import cv2
import numpy as np
from dotenv import load_dotenv
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import speaker diarization functions
sys.path.insert(0, str(Path(__file__).parent / "speaker_diarization"))
from enroll_from_local_wav import (
    to_wav_16k_mono,
    upload_local_wav_to_media,
    speech_to_text_diarization,
    is_robot_voice,
    build_single_speaker_clip_bytes,
)

# Import TalkNet functions
sys.path.insert(0, str(Path(__file__).parent / "talknet"))
try:
    from talknet_timestamp_speaker import (
        run_talknet_demo,
        get_faces_at_timestamp,
        ensure_video_in_demo,
        draw_boxes,
        Box,
    )
except ImportError as e:
    print(f"[Error] Failed to import TalkNet functions: {e}")
    raise

# Import face matching functions
sys.path.insert(0, str(Path(__file__).parent.parent / "deepface"))
from match_face import (
    set_mac_stability_env,
    ensure_db_cache,
    match_query_to_db,
)

# Import database functions (person DB has no speaker_id; identify by face_id only)
MEMOBOT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MEMOBOT_ROOT))
from utils.database import get_person_by_face_id

# Load .env
load_dotenv(dotenv_path=MEMOBOT_ROOT / ".env")
load_dotenv()
PYANNOTE_API_KEY = os.getenv("PYANNOTE_API_KEY")
if not PYANNOTE_API_KEY:
    raise RuntimeError("Set env var PYANNOTE_API_KEY")

# Configuration
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(__file__).parent / "data"
FACE_DB_DIR = REPO_ROOT / "face_database"
TALKNET_REPO = Path(__file__).parent / "talknet"
INTERMEDIATE_OUTPUTS_DIR = Path(__file__).parent / "intermediate_outputs"

# Optimization: Clip window size
CLIP_PRE_BUFFER = 1.0
CLIP_POST_BUFFER = 1.0
BUFFER_SEC = 0.2 

FACE_MODEL = "ArcFace"
FACE_DETECTOR = "opencv"
FACE_DISTANCE_METRIC = "cosine"
FACE_CACHE_PATH = Path(__file__).parent.parent / "deepface" / "db_embeddings.pkl"


def extract_audio_from_video(video_path: Path, output_wav: Path) -> None:
    to_wav_16k_mono(str(video_path), str(output_wav))


def crop_face_from_frame(frame: np.ndarray, box: Box) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    clamped = box.clamp(w, h)
    if clamped.x2 <= clamped.x1 or clamped.y2 <= clamped.y1:
        return None
    crop = frame[clamped.y1:clamped.y2, clamped.x1:clamped.x2]
    if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
        return None
    return crop


def save_face_crop(crop: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), crop)


def cut_mini_clip(input_path: Path, output_path: Path, start: float, duration: float):
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(input_path),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "ultrafast", 
        "-c:a", "aac",
        "-strict", "experimental",
        str(output_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)


def match_faces_to_database(
    face_crops: List[Tuple[str, np.ndarray]],
    face_db_dir: Path,
    cache_path: Path,
) -> Dict[str, Tuple[Optional[str], Optional[float]]]:
    set_mac_stability_env(max_threads=1)
    
    db_map = ensure_db_cache(
        db_dir=face_db_dir,
        cache_path=cache_path,
        model_name=FACE_MODEL,
        detector_backend=FACE_DETECTOR,
        enforce_detection=False,
        align=False,
        distance_metric=FACE_DISTANCE_METRIC,
        rebuild=True, 
    )
    
    if not db_map:
        return {}
    
    results = {}
    with tempfile.TemporaryDirectory() as td:
        for face_label, crop in face_crops:
            temp_path = Path(td) / f"{face_label}.jpg"
            cv2.imwrite(str(temp_path), crop)
            
            face_id, db_img, distance = match_query_to_db(
                query_img=temp_path,
                db_map=db_map,
                model_name=FACE_MODEL,
                detector_backend=FACE_DETECTOR,
                enforce_detection=False,
                align=False,
                distance_metric=FACE_DISTANCE_METRIC,
            )
            results[face_label] = (face_id, distance)
    return results


def process_video(video_filename: str) -> Tuple[List[Dict[str, Any]], Path]:
    video_path = DATA_DIR / video_filename
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    print(f"[Pipeline] Processing video: {video_path}")
    pipeline_start_time = time.perf_counter()
    
    run_id = uuid.uuid4().hex[:8]
    intermediate_dir = INTERMEDIATE_OUTPUTS_DIR / f"run_{run_id}"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Step 1: Extract Audio ---
    print("\n=== Step 1: Extracting audio ===")
    t_start = time.perf_counter()
    
    audio_output_dir = intermediate_dir / "audio"
    audio_output_dir.mkdir(exist_ok=True)
    audio_wav_path = audio_output_dir / "extracted_audio.wav"
    extract_audio_from_video(video_path, audio_wav_path)
    wav_bytes = audio_wav_path.read_bytes()
    
    # Upload to pyannote
    audio_key = f"audio/{uuid.uuid4().hex}.wav"
    audio_media_url = upload_local_wav_to_media(wav_bytes, audio_key)
    
    t_end = time.perf_counter()
    print(f"[Timing] Step 1 (Audio Extraction & Upload) took: {t_end - t_start:.2f}s")
    
    # --- Step 2: Speaker Diarization ---
    print("\n=== Step 2: Speaker diarization (Pyannote) ===")
    t_start = time.perf_counter()
    
    turns = speech_to_text_diarization(audio_media_url)
    print(f"[OK] Found {len(turns)} speaker turns")
    
    with open(intermediate_dir / "diarization.json", "w") as f:
        json.dump(turns, f, indent=2)

    t_end = time.perf_counter()
    print(f"[Timing] Step 2 (Diarization) took: {t_end - t_start:.2f}s")

    # --- Step 2b: Filter robot voice (check once per unique speaker, not per turn) ---
    print("\n=== Step 2b: Filter robot voice ===")
    t_start = time.perf_counter()
    
    unique_speakers = sorted({t["speaker"] for t in turns})
    speaker_is_robot = {}
    for spk in unique_speakers:
        print(f"[Robot check] {spk}...", flush=True)
        try:
            clip_bytes = build_single_speaker_clip_bytes(
                str(audio_wav_path), turns, spk
            )
        except Exception as e:
            print(f"[Skip] Cannot build clip for {spk}: {e}")
            speaker_is_robot[spk] = False
            continue
        clip_key = f"clips/{uuid.uuid4().hex}_{spk}.wav"
        clip_media_url = upload_local_wav_to_media(clip_bytes, clip_key)
        speaker_is_robot[spk] = is_robot_voice(clip_media_url)
        if speaker_is_robot[spk]:
            print(f"[Robot] {spk} is robot voice; skipping those turns.")
    
    human_turns = [t for t in turns if not speaker_is_robot.get(t["speaker"], False)]
    t_end = time.perf_counter()
    print(f"[Timing] Step 2b (Robot filter) took: {t_end - t_start:.2f}s")
    print(f"[Info] {len(human_turns)} non-robot turns to process (TalkNet + DeepFace).")
    
    if not human_turns:
        print("[Warning] No human turns after robot filter.")
        return [], intermediate_dir
    
    # --- Step 3: Targeted TalkNet + Face Matching (parallel per segment) ---
    print("\n=== Step 3: Active Speaker Detection + Face Match (parallel) ===")
    t_start = time.perf_counter()
    
    timestamps_to_process = []
    seen_starts = set()
    for turn in human_turns:
        ts = max(0.0, turn["start"] - BUFFER_SEC)
        if ts not in seen_starts:
            timestamps_to_process.append((ts, turn))
            seen_starts.add(ts)

    print(f"[Info] Processing {len(timestamps_to_process)} segments in parallel.")

    all_face_crops = []
    timestamp_to_faces = {}
    
    faces_output_dir = intermediate_dir / "faces"
    frames_output_dir = intermediate_dir / "frames"
    faces_output_dir.mkdir(exist_ok=True)
    frames_output_dir.mkdir(exist_ok=True)
    clips_dir = intermediate_dir / "temp_clips"
    clips_dir.mkdir(exist_ok=True)

    cap_check = cv2.VideoCapture(str(video_path))
    total_frames = cap_check.get(cv2.CAP_PROP_FRAME_COUNT)
    fps_check = cap_check.get(cv2.CAP_PROP_FPS)
    video_duration = total_frames / fps_check if fps_check > 0 else 0
    cap_check.release()

    def process_one_segment(args):
        idx, target_ts, turn = args
        clip_start = max(0, target_ts - CLIP_PRE_BUFFER)
        clip_end = min(video_duration, target_ts + CLIP_POST_BUFFER)
        duration = clip_end - clip_start
        if duration < 0.5:
            return (target_ts, turn, None, None, None, [])
        clip_name = f"clip_{idx}_{int(target_ts*1000)}"
        clip_path = clips_dir / f"{clip_name}.mp4"
        try:
            cut_mini_clip(video_path, clip_path, clip_start, duration)
            ensure_video_in_demo(TALKNET_REPO, clip_path, clip_name)
            annotated_video_path = run_talknet_demo(
                TALKNET_REPO, clip_name, force=False, confidence_threshold=-0.5
            )
            relative_ts = target_ts - clip_start
            boxes, speaker_box, orig_frame = get_faces_at_timestamp(
                demo_video=clip_path,
                annotated_video=annotated_video_path,
                timestamp=relative_ts,
            )
            crops = []
            if orig_frame is not None and boxes:
                for i, box in enumerate(boxes):
                    crop = crop_face_from_frame(orig_frame, box)
                    if crop is not None:
                        label = f"t{int(target_ts*1000)}_face_{i:02d}_{box.color}"
                        crops.append((label, crop))
                if speaker_box:
                    s_crop = crop_face_from_frame(orig_frame, speaker_box)
                    if s_crop is not None:
                        crops.append((f"t{int(target_ts*1000)}_speaker", s_crop))
            if clip_path.exists():
                clip_path.unlink()
            return (target_ts, turn, boxes, speaker_box, orig_frame, crops)
        except Exception as e:
            return (target_ts, turn, None, None, None, [])

    max_workers = min(4, len(timestamps_to_process))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_one_segment, (idx, target_ts, turn)): (idx, target_ts)
            for idx, (target_ts, turn) in enumerate(timestamps_to_process)
        }
        for future in as_completed(futures):
            idx, target_ts = futures[future]
            try:
                target_ts_r, turn, boxes, speaker_box, orig_frame, crops = future.result()
                timestamp_to_faces[target_ts_r] = (boxes, speaker_box, orig_frame)
                for label, crop in crops:
                    all_face_crops.append((label, crop))
                if orig_frame is not None:
                    frame_path = frames_output_dir / f"t{int(target_ts_r*1000)}_frame.jpg"
                    cv2.imwrite(str(frame_path), orig_frame)
                if crops:
                    ts_ms = int(target_ts_r * 1000)
                    for label, crop in crops:
                        stem = label.replace(f"t{ts_ms}_", "") + ".jpg"
                        save_face_crop(crop, faces_output_dir / f"t{ts_ms}" / stem)
            except Exception as e:
                print(f"  [Error] Segment {target_ts}: {e}")

    shutil.rmtree(clips_dir, ignore_errors=True)
    
    t_end = time.perf_counter()
    print(f"[Timing] Step 3 (Targeted TalkNet Loop) took: {t_end - t_start:.2f}s")

    if not all_face_crops:
        print("[Warning] No faces found in any active segments.")
        return [], intermediate_dir

    # --- Step 4: Face Matching ---
    print(f"\n=== Step 4: Matching {len(all_face_crops)} faces against database ===")
    t_start = time.perf_counter()
    
    face_matches = match_faces_to_database(
        face_crops=all_face_crops,
        face_db_dir=FACE_DB_DIR,
        cache_path=FACE_CACHE_PATH,
    )
    
    t_end = time.perf_counter()
    print(f"[Timing] Step 4 (Face Matching) took: {t_end - t_start:.2f}s")
    
    # --- Step 5: Combine Results (person_id from face_id; no speaker_id) ---
    print("\n=== Step 5: Combining results ===")
    results = []
    
    for timestamp, turn in timestamps_to_process:
        if timestamp not in timestamp_to_faces:
            continue
        
        boxes, speaker_box, _ = timestamp_to_faces[timestamp]
        
        speaker_face_id = None
        speaker_distance = None
        
        if speaker_box is not None:
            speaker_label = f"t{int(timestamp*1000)}_speaker"
            if speaker_label in face_matches:
                speaker_face_id, speaker_distance = face_matches[speaker_label]
        
        if speaker_face_id is None:
            for i, box in enumerate(boxes or []):
                if box.color == "green":
                    label = f"t{int(timestamp*1000)}_face_{i:02d}_{box.color}"
                    if label in face_matches:
                        fid, dist = face_matches[label]
                        if fid:
                            speaker_face_id, speaker_distance = fid, dist
                            break
                            
        if speaker_face_id is None:
            for i, box in enumerate(boxes or []):
                label = f"t{int(timestamp*1000)}_face_{i:02d}_{box.color}"
                if label in face_matches:
                    fid, dist = face_matches[label]
                    if fid:
                        speaker_face_id, speaker_distance = fid, dist
                        break

        person_id = None
        name = None
        if speaker_face_id:
            person = get_person_by_face_id(speaker_face_id)
            if person:
                person_id = person.get("person_id")
                name = person.get("name")
        
        results.append({
            "person_id": person_id,
            "face_id": speaker_face_id,
            "name": name,
            "text": turn["text"],
            "start": turn["start"],
            "end": turn["end"],
            "timestamp_processed": timestamp,
            "match_distance": speaker_distance,
        })

    final_output = intermediate_dir / "final_results.json"
    with open(final_output, "w") as f:
        json.dump(results, f, indent=2)
        
    pipeline_end_time = time.perf_counter()
    print(f"\n[Timing] TOTAL process_video execution time: {pipeline_end_time - pipeline_start_time:.2f}s")
    
    return results, intermediate_dir


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_video.py <video_filename>")
        sys.exit(1)
    process_video(sys.argv[1])