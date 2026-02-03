#!/usr/bin/env python3
"""
Ingest pipeline entrypoint (data-folder based):
- By default: process every video in ingest_pipeline/data/, then run vector_db for each.
- Optional: pass a single video path/filename to process just that file.
"""

import os
import json
import shutil
import sys
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from dotenv import dotenv_values, load_dotenv

MEMOBOT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = MEMOBOT_ROOT.parent
DOTENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(DOTENV_PATH)
_ENV = dotenv_values(DOTENV_PATH)

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import your submodules
try:
    from process_video import process_video, DATA_DIR
    from vector_db import ingest_data
except ImportError as e:
    print(f"[Critical Error] Could not import local modules: {e}")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from Memobot import MemobotService
except ImportError:
    print("[Warning] Memobot package not found. Knowledge Graph ingestion will be skipped.")
    MemobotService = None
except Exception as e:
    print(f"[Error] Failed to import MemobotService: {e}")
    exit(1)

DEFAULT_INDEX_NAME = "twelve-labs"
DEFAULT_CLIP_LENGTH = 30
DEFAULT_MEMOBOT_GROUP_ID = os.getenv("MEMOBOT_GROUP_ID", "tenant_003")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _videos_in_data_dir() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    out = []
    for p in DATA_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            out.append(p)
    return sorted(out)


async def _process_one_video(
    video_path: Path,
    video_filename: str,
    index_name: str,
    clip_length: int,
    memobot_service: MemobotService = None,
    memobot_group_id: str = None,
) -> dict:
    
    if memobot_group_id is None:
        memobot_group_id = DEFAULT_MEMOBOT_GROUP_ID

    video_path_str = str(video_path.resolve())
    
    # --- Step 1: Process Video ---
    print("\n" + "=" * 60)
    print("Step 1: process_video (Diarization, TalkNet, Face Matching)")
    print("=" * 60)
    
    # Logic and timing are now handled inside process_video.py
    results, intermediate_dir = process_video(video_filename)
    
    print(f"Processed {len(results)} speaker turns.")

    # --- Step 2: Vector DB ---
    print("\n" + "=" * 60)
    print("Step 2: vector_db (Embeddings -> Pinecone)")
    print("=" * 60)
    
    ingest_result = ingest_data(
        video_source=video_path_str,
        index_name=index_name,
        clip_length=clip_length,
        process_video_results=results,
        memobot_group_id=memobot_group_id,
    )
    
    return ingest_result


async def ingest_to_graph(final_outputs: list[dict], memobot_service: MemobotService):
    if not memobot_service or not final_outputs:
        return

    print("\n" + "=" * 60)
    print("Step 3: Building Knowledge Graph")
    print("=" * 60)

    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    for item in final_outputs:
        pid = item.get("person_id")
        name = item.get("name")
        text_content = item.get("clip_summary")

        input_data = {
            "id": f"log_{int(datetime.now().timestamp())}_{pid}",
            "person_name": name,
            "person_id": pid,
            "text": text_content,
            "robot_pos_list": [],
            "timestamp": timestamp,
        }

        try:
            await memobot_service.build(input_data)
            print(f"[OK] Graph built for {name}")
        except Exception as e:
            print(f"[ERROR] Graph build failed for {name}: {e}")


async def main_async():
    # CLI Argument Parsing
    index_name = DEFAULT_INDEX_NAME
    clip_length = DEFAULT_CLIP_LENGTH
    videos_to_process: list[tuple[Path, str]] = []

    # Init Memobot
    memobot_service = None
    if MemobotService:
        try:
            memobot_service = MemobotService.from_env(group_id=DEFAULT_MEMOBOT_GROUP_ID)
            print(f"[Info] MemobotService initialized.")
        except Exception as e:
            print(f"[Warning] Failed to initialize MemobotService: {e}")

    # Determine Videos
    if len(sys.argv) >= 2:
        video_arg = sys.argv[1]
        if len(sys.argv) >= 3: index_name = sys.argv[2]
        if len(sys.argv) >= 4: clip_length = int(sys.argv[3])

        video_path = Path(video_arg) if (os.path.isabs(video_arg) or "/" in video_arg) else DATA_DIR / video_arg

        if not video_path.exists():
            print(f"Error: Video not found: {video_path}")
            sys.exit(1)

        video_path = video_path.resolve()
        try:
            video_path.relative_to(DATA_DIR)
            video_filename = video_path.name
        except ValueError:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            dest = DATA_DIR / video_path.name
            shutil.copy2(video_path, dest)
            video_filename = video_path.name
            video_path = dest
            print(f"Copied video to {dest}")

        videos_to_process = [(video_path, video_filename)]
    else:
        all_videos = _videos_in_data_dir()
        if not all_videos:
            print("No video files found.")
            sys.exit(1)
        videos_to_process = [(p, p.name) for p in all_videos]

    final_outputs: list[dict] = []

    # --- Main Loop ---
    for video_path, video_filename in videos_to_process:
        print(f"\n>>> Processing Video: {video_path.name}")
        
        try:
            ingest_result = await _process_one_video(
                video_path, video_filename, index_name, clip_length,
                memobot_service, DEFAULT_MEMOBOT_GROUP_ID
            )
            
            # Extract Results (audio_dialogue = person_id-to-voice; person_ids = all users who talked)
            clip_summary = ingest_result.get("summary")
            audio_dialogue = ingest_result.get("audio_dialogue")
            person_ids = ingest_result.get("person_ids") or []
            for p in ingest_result.get("persons", []) or []:
                if p.get("person_id") and p.get("name"):
                    final_outputs.append({
                        "person_id": p.get("person_id"),
                        "name": p.get("name"),
                        "clip_summary": clip_summary,
                        "audio_dialogue": audio_dialogue,
                        "person_ids": person_ids,
                    })
        except Exception as e:
            print(f"[ERROR] Failed processing {video_filename}: {e}")
            continue

    # --- Final Output (audio_dialogue + person_ids; no speaker_id) ---
    all_audio = []
    all_person_ids = set()
    for p in final_outputs:
        if p.get("audio_dialogue"):
            all_audio.append(p["audio_dialogue"])
        if p.get("person_id"):
            all_person_ids.add(p["person_id"])
    final_json = {
        "audio_dialogue": "\n---\n".join(all_audio) if all_audio else "",
        "person_ids": list(all_person_ids),
    }
    print("\n" + "=" * 60)
    print("FINAL JSON OUTPUT (audio_dialogue + person_ids)")
    print("=" * 60)
    print(json.dumps(final_json, indent=2))
    if final_outputs:
        print("\nPer-person (for graph):")
        print(json.dumps(final_outputs, indent=2))

    # --- Step 3: Graph ---
    await ingest_to_graph(final_outputs, memobot_service)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()