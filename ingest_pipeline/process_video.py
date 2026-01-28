#!/usr/bin/env python3
"""
Process Video:
1. Extract audio and run speaker diarization (with transcription)
2. For each speaker turn start (with 0.2s buffer), run TalkNet to extract faces
3. Match faces against face_database to identify face_id
4. Output: face_id matches which speaker_id (and what they said)

Usage:
    python process_video.py <video_filename>
    
The video should be in the data/ folder.
Face database images should be in face_database/ folder.
"""

import os
import sys
import uuid
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

import cv2
import numpy as np
from dotenv import load_dotenv
import requests
import asyncio
import sys
from pathlib import Path

# Allow importing from root directory (for Memobot package)
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from Memobot import MemobotService
except ImportError:
    print("[Warning] Memobot package not found. Knowledge Graph ingestion will be skipped.")
    MemobotService = None

# Import speaker diarization functions
sys.path.insert(0, str(Path(__file__).parent / "speaker_diarization"))
from enroll_from_local_wav import (
    to_wav_16k_mono,
    upload_local_wav_to_media,
    speech_to_text_diarization,
    enroll_speakers_from_turns,
    MEDIA_INPUT,
    HEADERS_JSON,
    HEADERS_AUTH,
    JOBS,
    submit_job,
    poll_job,
    media_put_url,
    upload_bytes,
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
    list_images,
    ensure_db_cache,
    match_query_to_db,
    person_id_from_path,
)

# Import database functions
MEMOBOT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MEMOBOT_ROOT))
from utils.database import (
    get_person_by_face_id,
    update_speaker_id,
    get_person_by_speaker_id,
)

# Load .env from memobot root
load_dotenv(dotenv_path=MEMOBOT_ROOT / ".env")
# Initialize MemobotService globally
# Allow importing from root directory (for Memobot package)
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from Memobot import MemobotService
    # Initialize service once at the global level
    # Configure from environment variables (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    memobot_service = MemobotService.from_env(group_id='tenant_001')
    print("[Info] MemobotService initialized globally")
except Exception as e:
    print(f"[Warning] Failed to initialize MemobotService globally: {e}")
    memobot_service = None

load_dotenv()
PYANNOTE_API_KEY = os.getenv("PYANNOTE_API_KEY")
if not PYANNOTE_API_KEY:
    raise RuntimeError("Set env var PYANNOTE_API_KEY")

# Configuration
DATA_DIR = Path(__file__).parent / "data"
FACE_DB_DIR = Path(__file__).parent.parent / "face_database"
TALKNET_REPO = Path(__file__).parent / "talknet"
INTERMEDIATE_OUTPUTS_DIR = Path(__file__).parent / "intermediate_outputs"
BUFFER_SEC = 0.2  # Buffer to add before speaker turn start

# Face matching config (from match_face.py defaults)
FACE_MODEL = "ArcFace"
FACE_DETECTOR = "opencv"
FACE_DISTANCE_METRIC = "cosine"
FACE_CACHE_PATH = Path(__file__).parent.parent / "deepface" / "db_embeddings.pkl"


def extract_audio_from_video(video_path: Path, output_wav: Path) -> None:
    """Extract audio from video and convert to 16k mono WAV."""
    to_wav_16k_mono(str(video_path), str(output_wav))


def crop_face_from_frame(frame: np.ndarray, box: Box) -> Optional[np.ndarray]:
    """Crop a face from a frame using a bounding box."""
    h, w = frame.shape[:2]
    clamped = box.clamp(w, h)
    if clamped.x2 <= clamped.x1 or clamped.y2 <= clamped.y1:
        return None
    crop = frame[clamped.y1:clamped.y2, clamped.x1:clamped.x2]
    if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
        return None
    return crop


def save_face_crop(crop: np.ndarray, output_path: Path) -> None:
    """Save a face crop image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), crop)


def extract_faces_at_timestamp(
    video_path: Path,
    talknet_repo: Path,
    video_name: str,
    timestamp: float,
    temp_dir: Path,
) -> Tuple[List[Box], Optional[Box], Optional[np.ndarray]]:
    """
    Extract faces at a specific timestamp using TalkNet.
    Returns: (all_face_boxes, speaker_box, original_frame)
    """
    # Ensure video is in demo folder
    demo_video = ensure_video_in_demo(talknet_repo, video_path, video_name)
    
    # Run TalkNet demo
    annotated_video = run_talknet_demo(
        talknet_repo, video_name, force=False, confidence_threshold=-0.5
    )
    
    # Read frames at timestamp
    cap_orig = open_video(demo_video)
    cap_ann = open_video(annotated_video)
    
    try:
        orig_frame, _ = read_frame_at_timestamp(cap_orig, timestamp)
        ann_frame, _ = read_frame_at_timestamp(cap_ann, timestamp)
        
        # Detect boxes from annotated frame
        boxes = find_colored_boxes(ann_frame)
        speaker_box = pick_speaker_box(boxes)
        
        return boxes, speaker_box, orig_frame
    finally:
        cap_orig.release()
        cap_ann.release()


def match_faces_to_database(
    face_crops: List[Tuple[str, np.ndarray]],
    face_db_dir: Path,
    cache_path: Path,
) -> Dict[str, Tuple[Optional[str], Optional[float]]]:
    """
    Match face crops against face database.
    face_crops: List of (face_label, crop_image_array)
    Returns: Dict mapping face_label -> (matched_face_id, distance)
    """
    set_mac_stability_env(max_threads=1)
    
    # Load face database
    db_map = ensure_db_cache(
        db_dir=face_db_dir,
        cache_path=cache_path,
        model_name=FACE_MODEL,
        detector_backend=FACE_DETECTOR,
        enforce_detection=False,
        align=False,
        distance_metric=FACE_DISTANCE_METRIC,
        rebuild=True,  # Always rebuild cache to ensure it matches current face_database directory
    )
    
    if not db_map:
        print(f"[Warning] No face database found in {face_db_dir}")
        print(f"[Info] Place face images (jpg/png) in {face_db_dir} to enable face matching")
        return {}
    
    results = {}
    
    # Save each crop temporarily and match
    total_start_time = time.time()
    print(f"[Timing] Total face matching started at {time.strftime('%H:%M:%S', time.localtime(total_start_time))} for {len(face_crops)} faces")
    
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
            print(f"[Match] {face_label} -> {face_id} (distance={distance})")
    
    total_end_time = time.time()
    total_elapsed = total_end_time - total_start_time
    print(f"[Timing] Total face matching ended at {time.strftime('%H:%M:%S', time.localtime(total_end_time))}")
    print(f"[Timing] Total time for {len(face_crops)} face recognitions: {total_elapsed:.3f}s (avg: {total_elapsed/len(face_crops):.3f}s per face)")
    
    return results


def process_video(video_filename: str) -> Tuple[List[Dict[str, Any]], Path]:
    """
    Main pipeline function.
    Returns (list of results, intermediate_outputs_dir):
    - results: [{speaker_id, face_id, text, start, end}, ...]
    - intermediate_outputs_dir: Path to where intermediate outputs were saved
    """
    video_path = DATA_DIR / video_filename
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    print(f"[Pipeline] Processing video: {video_path}")
    
    # Create intermediate outputs directory
    run_id = uuid.uuid4().hex[:8]
    intermediate_dir = INTERMEDIATE_OUTPUTS_DIR / f"run_{run_id}"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Info] Intermediate outputs will be saved to: {intermediate_dir}")
    
    # Generate unique video name for TalkNet
    video_name = f"ingest_{run_id}"
    
    # Step 1: Extract audio
    print("\n=== Step 1: Extracting audio ===")
    audio_output_dir = intermediate_dir / "audio"
    audio_output_dir.mkdir(exist_ok=True)
    audio_wav_path = audio_output_dir / "extracted_audio.wav"
    extract_audio_from_video(video_path, audio_wav_path)
    wav_bytes = audio_wav_path.read_bytes()
    print(f"[OK] Audio extracted and saved to: {audio_wav_path}")
    
    # Upload to pyannote
    audio_key = f"audio/{uuid.uuid4().hex}.wav"
    audio_media_url = upload_local_wav_to_media(wav_bytes, audio_key)
    print(f"[OK] Audio uploaded: {audio_media_url}")
    
    # Step 2: Speaker diarization with transcription
    print("\n=== Step 2: Speaker diarization ===")
    turns = speech_to_text_diarization(audio_media_url)
    print(f"[OK] Found {len(turns)} speaker turns")
    for i, t in enumerate(turns, 1):
        print(f"  [{i:03d}] {t['speaker']} {t['start']:.2f}-{t['end']:.2f}s: {t['text']}")
    
    # Save diarization results
    diarization_output = intermediate_dir / "diarization.json"
    with open(diarization_output, "w") as f:
        json.dump(turns, f, indent=2)
    print(f"[OK] Diarization results saved to: {diarization_output}")
    
    # Step 2b: Enroll speakers and map to voiceprint labels
    print("\n=== Step 2b: Speaker enrollment ===")
    speaker_to_voiceprint = enroll_speakers_from_turns(
        turns=turns,
        local_wav_16k_path=str(audio_wav_path),
        source_file=str(video_path)
    )
    print(f"[OK] Mapped {len(speaker_to_voiceprint)} speakers to voiceprints")
    for diar_speaker, vp_label in speaker_to_voiceprint.items():
        print(f"  {diar_speaker} -> {vp_label}")
    
    # Step 3: Extract faces at each speaker turn start (with buffer)
    print("\n=== Step 3: Extracting faces at speaker timestamps ===")
    
    # Collect unique timestamps (one per speaker turn start)
    timestamps_to_process = []
    seen_starts = set()
    for turn in turns:
        timestamp = max(0.0, turn["start"] - BUFFER_SEC)
        if timestamp not in seen_starts:
            timestamps_to_process.append((timestamp, turn))
            seen_starts.add(timestamp)
    
    print(f"[Info] Processing {len(timestamps_to_process)} unique timestamps (one per speaker turn start)")
    
    # Run TalkNet once for the whole video; then we only read frames at each timestamp
    demo_video = ensure_video_in_demo(TALKNET_REPO, video_path, video_name)
    annotated_video = run_talknet_demo(
        TALKNET_REPO, video_name, force=False, confidence_threshold=-0.5
    )
    
    # Extract faces at each timestamp (no TalkNet run—just read frame at that time)
    all_face_crops = []  # List of (label, crop_image)
    timestamp_to_faces = {}  # timestamp -> (boxes, speaker_box, frame)
    
    # Create directories for face outputs
    faces_output_dir = intermediate_dir / "faces"
    frames_output_dir = intermediate_dir / "frames"
    faces_output_dir.mkdir(exist_ok=True)
    frames_output_dir.mkdir(exist_ok=True)
    
    for idx, (timestamp, turn) in enumerate(timestamps_to_process):
        print(f"\n[Timestamp {idx+1}/{len(timestamps_to_process)}] t={timestamp:.2f}s (speaker: {turn['speaker']})")
        
        # Create timestamp-specific directory
        ts_dir = faces_output_dir / f"t{int(timestamp*1000):06d}"
        ts_dir.mkdir(exist_ok=True)
        
        try:
            boxes, speaker_box, orig_frame = get_faces_at_timestamp(
                demo_video=demo_video,
                annotated_video=annotated_video,
                timestamp=timestamp,
            )
            
            timestamp_to_faces[timestamp] = (boxes, speaker_box, orig_frame)
            
            # Save original frame
            frame_path = frames_output_dir / f"t{int(timestamp*1000):06d}_frame.jpg"
            cv2.imwrite(str(frame_path), orig_frame)
            
            # Draw boxes on frame and save
            frame_with_boxes = draw_boxes(orig_frame, boxes, speaker_box)
            frame_annotated_path = frames_output_dir / f"t{int(timestamp*1000):06d}_frame_annotated.jpg"
            cv2.imwrite(str(frame_annotated_path), frame_with_boxes)
            
            # Crop all faces and save
            for i, box in enumerate(boxes):
                crop = crop_face_from_frame(orig_frame, box)
                if crop is not None:
                    label = f"t{int(timestamp*1000)}_face_{i:02d}_{box.color}"
                    all_face_crops.append((label, crop))
                    
                    # Save face crop
                    face_crop_path = ts_dir / f"face_{i:02d}_{box.color}.jpg"
                    save_face_crop(crop, face_crop_path)
            
            # Crop speaker face if found and save
            if speaker_box is not None:
                speaker_crop = crop_face_from_frame(orig_frame, speaker_box)
                if speaker_crop is not None:
                    label = f"t{int(timestamp*1000)}_speaker"
                    all_face_crops.append((label, speaker_crop))
                    
                    # Save speaker face crop
                    speaker_crop_path = ts_dir / "speaker_face.jpg"
                    save_face_crop(speaker_crop, speaker_crop_path)
            
            print(f"  [OK] Found {len(boxes)} face boxes, speaker_box={'found' if speaker_box else 'not found'}")
            print(f"  [OK] Saved outputs to: {ts_dir}")
            
        except Exception as e:
            print(f"  [ERROR] Failed to extract faces at t={timestamp:.2f}s: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_face_crops:
        print("[Warning] No faces extracted from video")
        # Save empty results
        final_results_output = intermediate_dir / "final_results.json"
        with open(final_results_output, "w") as f:
            json.dump([], f, indent=2)
        return [], intermediate_dir
    
    # Step 4: Match faces against database
    print(f"\n=== Step 4: Matching {len(all_face_crops)} faces against database ===")
    face_matches = match_faces_to_database(
        face_crops=all_face_crops,
        face_db_dir=FACE_DB_DIR,
        cache_path=FACE_CACHE_PATH,
    )
    
    # Save face matching results
    face_matching_output = intermediate_dir / "face_matching.json"
    face_matching_data = {
        label: {
            "face_id": face_id,
            "distance": float(distance) if distance is not None else None
        }
        for label, (face_id, distance) in face_matches.items()
    }
    with open(face_matching_output, "w") as f:
        json.dump(face_matching_data, f, indent=2)
    print(f"[OK] Face matching results saved to: {face_matching_output}")
    
    # Step 5: Combine results
    print("\n=== Step 5: Combining results ===")
    results = []
    
    for timestamp, turn in timestamps_to_process:
        if timestamp not in timestamp_to_faces:
            continue
        
        boxes, speaker_box, _ = timestamp_to_faces[timestamp]
        
        # Find speaker face_id
        speaker_face_id = None
        speaker_distance = None
        
        if speaker_box is not None:
            speaker_label = f"t{int(timestamp*1000)}_speaker"
            if speaker_label in face_matches:
                speaker_face_id, speaker_distance = face_matches[speaker_label]
        
        # If speaker face not matched, try matching all green boxes
        if speaker_face_id is None and speaker_box is not None:
            for i, box in enumerate(boxes):
                if box.color == "green":
                    label = f"t{int(timestamp*1000)}_face_{i:02d}_{box.color}"
                    if label in face_matches:
                        face_id, distance = face_matches[label]
                        if face_id is not None:
                            speaker_face_id = face_id
                            speaker_distance = distance
                            break
        
        # If still no match, try any face at this timestamp
        if speaker_face_id is None:
            for i, box in enumerate(boxes):
                label = f"t{int(timestamp*1000)}_face_{i:02d}_{box.color}"
                if label in face_matches:
                    face_id, distance = face_matches[label]
                    if face_id is not None:
                        speaker_face_id = face_id
                        speaker_distance = distance
                        break
        
        # Get voiceprint label for this speaker
        diarization_speaker_id = turn["speaker"]
        voiceprint_label = speaker_to_voiceprint.get(diarization_speaker_id, "UNKNOWN")
        
        result = {
            "speaker_id": diarization_speaker_id,  # Keep diarization ID for reference
            "voiceprint_label": voiceprint_label,   # Voiceprint label (person_xxx)
            "face_id": speaker_face_id,
            "text": turn["text"],
            "start": turn["start"],
            "end": turn["end"],
            "timestamp_processed": timestamp,
            "match_distance": speaker_distance,
        }
        results.append(result)
    
    # Step 6: Update persons.db with speaker_id when we have both face_id and voiceprint_label
    print("\n=== Step 6: Updating persons.db with speaker IDs ===")
    updated_count = 0
    for result in results:
        face_id = result.get("face_id")
        voiceprint_label = result.get("voiceprint_label")
        
        if face_id and voiceprint_label and voiceprint_label != "UNKNOWN":
            # Check if person exists with this face_id
            person = get_person_by_face_id(face_id)
            if person:
                # Update speaker_id if not already set or if different
                if person.get("speaker_id") != voiceprint_label:
                    update_speaker_id(person["person_id"], voiceprint_label)
                    updated_count += 1
                    print(f"[DB] Updated person_id={person['person_id']} (face_id={face_id}) with speaker_id={voiceprint_label}")
            else:
                # Check if speaker_id already exists for a different face_id
                existing_person = get_person_by_speaker_id(voiceprint_label)
                if existing_person:
                    print(f"[DB] Warning: speaker_id {voiceprint_label} already assigned to face_id {existing_person['face_id']}, skipping face_id {face_id}")
                else:
                    print(f"[DB] Warning: face_id {face_id} not found in persons.db, cannot update speaker_id")
    
    if updated_count > 0:
        print(f"[OK] Updated {updated_count} person(s) in persons.db")
    else:
        print(f"[OK] No updates needed in persons.db")
    
    # Save final combined results
    final_results_output = intermediate_dir / "final_results.json"
    with open(final_results_output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Final results saved to: {final_results_output}")
    
    return results, intermediate_dir



async def main_async():
    if len(sys.argv) != 2:
        print("Usage: python ingest_pipeline.py <video_filename>")
        print(f"  Video should be in: {DATA_DIR}")
        print(f"  Face database should be in: {FACE_DB_DIR}")
        sys.exit(1)
    
    video_filename = sys.argv[1]
    
    try:
        # Run the processing pipeline (synchronous)
        results, intermediate_dir = process_video(video_filename)
        
        # Print results
        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        if not results:
            print("\n[Warning] No results generated. Check if video has audio and speakers.")
        else:
            print(f"\n{'Speaker':<12} {'Voiceprint':<15} {'Face ID':<20} {'Text':<40} {'Time':<15}")
            print("-" * 100)
            
            for r in results:
                speaker_id_str = r["speaker_id"]
                voiceprint_str = r.get("voiceprint_label", "UNKNOWN")
                face_id_str = r["face_id"] if r["face_id"] else "UNKNOWN"
                text_short = (r["text"][:37] + "...") if len(r["text"]) > 40 else r["text"]
                time_str = f"{r['start']:.1f}-{r['end']:.1f}s"
                print(f"{speaker_id_str:<12} {voiceprint_str:<15} {face_id_str:<20} {text_short:<40} {time_str:<15}")
        
        # Also save to root results.json for convenience
        output_json = Path(__file__).parent / "results.json"
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[OK] Results also saved to: {output_json}")
        print(f"[OK] All intermediate outputs saved to: {intermediate_dir}")
        
        # Build Knowledge Graph (and other parallel ingestions if any)
        if memobot_service and results:
            print("\n=== Step 6: Building Knowledge Graph ===")
            try:
                # Parallel ingestion could be added here if there were other services
                await memobot_service.build(results)
                print("[OK] Knowledge graph built successfully")
            except Exception as e:
                print(f"[ERROR] Failed to build knowledge graph: {e}")
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if memobot_service:
            await memobot_service.close()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
