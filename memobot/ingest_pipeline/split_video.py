"""
Script to split a video file into fixed time interval chunks.
Splits the first minute into 6-second intervals (10 chunks total).
Uses ffmpeg via subprocess for reliable video splitting.
"""

import os
import subprocess
import sys

# ORIGINAL VIDEO LINK: https://huggingface.co/datasets/ByteDance-Seed/M3-Bench/blob/main/videos/robot/kitchen_07.mp4

def get_video_size_mb(video_path):
    """Get the size of the video file in MB."""
    return os.path.getsize(video_path) / (1024 * 1024)

def get_video_duration_ffmpeg(video_path):
    """Get video duration using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None

def check_ffmpeg():
    """Check if ffmpeg is available."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
def split_video(video_path, output_dir="intermediate_data", chunk_duration_sec=6, max_duration_sec=60):
    # Check if ffmpeg is available
    if not check_ffmpeg():
        print("Error: ffmpeg is not installed or not in PATH")
        print("\nTo install ffmpeg:")
        print("  macOS: brew install ffmpeg")
        print("  Linux: sudo apt-get install ffmpeg")
        print("  Windows: Download from https://ffmpeg.org/download.html")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    total_duration_sec = get_video_duration_ffmpeg(video_path)
    if total_duration_sec is None:
        raise ValueError("Could not determine video duration. Make sure ffprobe is available.")

    process_duration = min(max_duration_sec, total_duration_sec)
    num_chunks = int(process_duration / chunk_duration_sec)

    print(f"Video info:")
    print(f"  Total duration: {total_duration_sec:.2f} seconds")
    print(f"  Processing first: {process_duration:.2f} seconds")
    print(f"  Chunk duration: {chunk_duration_sec} seconds")
    print(f"  Number of chunks: {num_chunks}\n")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    chunk_paths = []

    start_time = 0
    chunk_index = 0

    while start_time < process_duration and chunk_index < num_chunks:
        end_time = min(start_time + chunk_duration_sec, process_duration)
        output_path = os.path.join(output_dir, f"{video_name}_chunk_{chunk_index:03d}.mp4")

        print(f"Creating chunk {chunk_index + 1}/{num_chunks}: {os.path.basename(output_path)}")
        print(f"  Time range: {start_time:.2f}s - {end_time:.2f}s")

        # ✅ Better argument order:
        #   -y before outputs, -ss before -i, -t before output
        cmd = [
            'ffmpeg',
            '-y',                          # overwrite without asking
            '-ss', str(start_time),        # seek to start
            '-i', video_path,              # input file
            '-t', str(end_time - start_time),  # duration
            '-c', 'copy',                  # no re-encode
            '-avoid_negative_ts', 'make_zero',
            output_path
        ]

        try:
            # Don't hide stderr; it’s useful when debugging
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # print(result.stderr)  # uncomment if you want ffmpeg logs

            chunk_size = get_video_size_mb(output_path)
            print(f"  Created chunk size: {chunk_size:.2f} MB\n")

            chunk_paths.append(output_path)
            start_time = end_time
            chunk_index += 1
        except subprocess.CalledProcessError as e:
            print(f"  Error creating chunk: {e}")
            print("  ffmpeg stderr:\n", e.stderr)
            break

    print(f"✓ Successfully split video into {len(chunk_paths)} chunks")
    return chunk_paths



if __name__ == "__main__":
    video_path = "raw_data/kitchen_07.mp4"
    output_dir = "intermediate_data"
    chunk_duration_sec = 6  # 6-second intervals
    max_duration_sec = 60   # First minute only
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        exit(1)
    
    print(f"Splitting video: {video_path}")
    print(f"Output directory: {output_dir}")
    print(f"Chunk duration: {chunk_duration_sec} seconds")
    print(f"Processing first: {max_duration_sec} seconds\n")
    
    try:
        chunk_paths = split_video(video_path, output_dir, chunk_duration_sec=chunk_duration_sec, max_duration_sec=max_duration_sec)
        print(f"\nChunk files created:")
        for i, path in enumerate(chunk_paths):
            size_mb = get_video_size_mb(path)
            print(f"  {i+1}. {path} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"Error splitting video: {e}")
        import traceback
        traceback.print_exc()

