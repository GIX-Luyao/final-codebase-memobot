#!/usr/bin/env python3
"""
Ingest pipeline entrypoint (data-folder based):
- By default: process every video in ingest_pipeline/data/, then run vector_db for each.
- Optional: pass a single video path/filename to process just that file.

1. Run process_video (diarization, face extraction, face matching) → final_results
2. Run vector_db to get metadata JSON for vector embedding and upsert into Pinecone
   (Pegasus summaries use spoken words and speaker identity from process_video)
"""

import os
import json
import shutil
import sys
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from dotenv import dotenv_values

MEMOBOT_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = MEMOBOT_ROOT / ".env"
_ENV = dotenv_values(DOTENV_PATH)

# Ensure ingest_pipeline is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from process_video import process_video, DATA_DIR
from vector_db import ingest_data

# Allow importing from root directory (for Memobot package)
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from Memobot import MemobotService
except ImportError:
    print("[Warning] Memobot package not found. Knowledge Graph ingestion will be skipped.")
    MemobotService = None
    exit(1)
except Exception as e:
    print(f"[Error] Failed to import MemobotService: {e}")
    exit(1)

# Default Pinecone index and clip length (one embedding per 30s clip)
DEFAULT_INDEX_NAME = "twelve-labs"
DEFAULT_CLIP_LENGTH = 30

# Knowledge Graph tenant/group id (keep configurable so ingest + query share the same graph)
DEFAULT_MEMOBOT_GROUP_ID = os.getenv("MEMOBOT_GROUP_ID", "tenant_003")

# Video extensions to discover in data/
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _videos_in_data_dir() -> list[Path]:
    """Return sorted list of video paths in ingest_pipeline/data/."""
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
    print("=" * 60)
    print("Step 1: process_video (diarization, faces, face matching)")
    print("=" * 60)
    results, intermediate_dir = process_video(video_filename)
    print(f"Processed {len(results)} speaker turns. Intermediate outputs: {intermediate_dir}")

    print("\n" + "=" * 60)
    print("Step 2: vector_db (embeddings + Pegasus metadata → Pinecone)")
    print("=" * 60)
    ingest_result = ingest_data(
        video_source=video_path_str,
        index_name=index_name,
        clip_length=clip_length,
        process_video_results=results,
        memobot_group_id=memobot_group_id,
    )
    print("\nIngest complete.")

    return ingest_result


async def ingest_to_graph(final_outputs: list[dict], memobot_service: MemobotService):
    """
    Reformat final_outputs and build the knowledge graph.
    """
    if not memobot_service or not final_outputs:
        print("[Warning] MemobotService not available or no final outputs to ingest.")
        return

    print("\n" + "=" * 60)
    print("Step 3: Building Knowledge Graph from Final Outputs")
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
            "robot_pos_list": [],  # No robot position data available
            "timestamp": timestamp,
        }

        try:
            await memobot_service.build(input_data)
            print(f"[OK] Knowledge graph built successfully for {name} ({pid})")
        except Exception as e:
            print(f"[ERROR] Failed to build knowledge graph for {name}: {e}")


async def main_async():
    # Optional args: [video_path_or_filename] [index_name] [clip_length_sec]
    index_name = DEFAULT_INDEX_NAME
    clip_length = DEFAULT_CLIP_LENGTH
    videos_to_process: list[tuple[Path, str]] = []  # (path, filename for process_video)

    # Initialize MemobotService
    memobot_service = None
    if MemobotService:
        try:
            memobot_service = MemobotService.from_env(group_id=DEFAULT_MEMOBOT_GROUP_ID)
            print(f"[Info] MemobotService initialized (group_id={DEFAULT_MEMOBOT_GROUP_ID})")
        except Exception as e:
            print(f"[Warning] Failed to initialize MemobotService: {e}")
    if len(sys.argv) >= 2:
        # Explicit video: path or filename under data/
        video_arg = sys.argv[1]
        if len(sys.argv) >= 3:
            index_name = sys.argv[2]
        if len(sys.argv) >= 4:
            clip_length = int(sys.argv[3])

        if os.path.isabs(video_arg) or "/" in video_arg or "\\" in video_arg:
            video_path = Path(video_arg)
        else:
            video_path = DATA_DIR / video_arg

        if not video_path.exists():
            print(f"Error: Video not found: {video_path}", file=sys.stderr)
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
            print(f"Copied video to {dest} for process_video")

        videos_to_process = [(video_path, video_filename)]
    else:
        # No args: process everything in data/ with default index and clip length
        all_videos = _videos_in_data_dir()
        if not all_videos:
            print("Usage: python main.py [video_path_or_filename] [index_name] [clip_length_sec]")
            print("  No args: process all videos in ingest_pipeline/data/")
            print("  video_path_or_filename: path or filename under data/ to process a single video")
            print("  index_name: (optional) Pinecone index name, default: twelve-labs")
            print("  clip_length_sec: (optional) clip length for one embedding/summary, default: 30")
            print(f"\nNo video files found in {DATA_DIR} (extensions: {VIDEO_EXTENSIONS})")
            sys.exit(1)
        videos_to_process = [(p, p.name) for p in all_videos]
        print(f"Processing {len(videos_to_process)} video(s) from {DATA_DIR}")

    final_outputs: list[dict] = []

    for video_path, video_filename in videos_to_process:
        print(f"\n>>> Processing: {video_path.name}")
        ingest_result = await _process_one_video(
            video_path, video_filename, index_name, clip_length,
            memobot_service, DEFAULT_MEMOBOT_GROUP_ID,
        )

        # Per your request: output {person_id, name, clip_summary}.
        clip_summary = ingest_result.get("summary")
        audio_dialogue = ingest_result.get("audio_dialogue")
        for p in ingest_result.get("persons", []) or []:
            pid = p.get("person_id")
            name = p.get("name")
            if pid and name:
                final_outputs.append(
                    {
                        "person_id": pid,
                        "name": name,
                        "clip_summary": clip_summary,
                        "audio_dialogue": audio_dialogue,
                    }
                )
    print("\n" + "=" * 60)
    print("FINAL JSON OUTPUT")
    print("=" * 60)
    print(json.dumps(final_outputs, indent=2))

    print("================================================")

    # Step 3: Build Knowledge Graph using final_outputs
    await ingest_to_graph(final_outputs, memobot_service)

    
    
    # Note: memobot_service.close() is NOT called here anymore.
    # It should be managed by the caller of main_async if needed, 
    # or left open for the duration of the agent process.


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
