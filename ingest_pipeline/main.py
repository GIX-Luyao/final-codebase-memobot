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
from pathlib import Path

# Ensure ingest_pipeline is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from process_video import process_video, DATA_DIR
from vector_db import ingest_data

# Default Pinecone index and clip length (one embedding per 30s clip)
DEFAULT_INDEX_NAME = "twelve-labs"
DEFAULT_CLIP_LENGTH = 30

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


def _process_one_video(
    video_path: Path,
    video_filename: str,
    index_name: str,
    clip_length: int,
) -> dict:
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
    )
    print("\nIngest complete.")
    return ingest_result


def main():
    # Optional args: [video_path_or_filename] [index_name] [clip_length_sec]
    index_name = DEFAULT_INDEX_NAME
    clip_length = DEFAULT_CLIP_LENGTH
    videos_to_process: list[tuple[Path, str]] = []  # (path, filename for process_video)

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
        ingest_result = _process_one_video(video_path, video_filename, index_name, clip_length)

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


if __name__ == "__main__":
    main()
