# Ingest Pipeline

A video processing pipeline that extracts speaker diarization, identifies faces, and matches them against a face database to create a comprehensive transcript with speaker and face identification.

## Overview

The ingest pipeline processes video files to:
1. Extract audio and perform speaker diarization with transcription
2. Extract faces at each speaker turn using TalkNet
3. Match detected faces against a face database using DeepFace
4. Combine results to identify which face_id corresponds to which speaker_id and what they said

## How It Works

The pipeline executes the following steps:

### Step 1: Audio Extraction
- Extracts audio from the input video
- Converts to 16kHz mono WAV format
- Uploads to pyannote.audio media storage for processing

### Step 2: Speaker Diarization
- Uses pyannote.audio API to perform speaker diarization with transcription
- Identifies speaker turns with timestamps and transcribed text
- Outputs speaker IDs (e.g., `SPEAKER_00`, `SPEAKER_01`) with start/end times

### Step 3: Face Extraction
- For each speaker turn start (with 0.2s buffer), extracts frames from the video
- Uses TalkNet to detect faces and identify the active speaker (green boxes = speaking, red boxes = not speaking)
- Crops all detected faces and the identified speaker face
- Saves intermediate outputs (frames, face crops, annotated frames)

### Step 4: Face Matching
- Matches all extracted face crops against the face database using DeepFace
- Uses ArcFace model with cosine distance metric
- Caches database embeddings for faster subsequent runs
- Returns face IDs (e.g., `person_1`, `person_2`) with match distances

### Step 5: Result Combination
- Combines speaker diarization results with face matching results
- Attempts to match the speaker face to a face_id using:
  1. Direct speaker face match (if TalkNet identified a speaker box)
  2. Green box matches (faces marked as speaking)
  3. Any face at the timestamp (fallback)
- Produces final results linking speaker_id → face_id → text

## Input

### Required Input
- **Video file**: Place your video file in the `data/` directory
  - Supported formats: Any video format supported by ffmpeg (MP4, AVI, MOV, etc.)
  - The video should contain audio for speaker diarization

### Face Database
- **Face images**: Place reference face images in the `face_database/` directory
  - Supported formats: JPG, PNG
  - Naming convention: `person_0.png`, `person_1.png`, `person_2.png`, etc.
  - Face IDs are derived from the filename (without extension)
  - These images should be clear, front-facing photos of the people you want to identify

### Environment Variables
- `PYANNOTE_API_KEY`: Required API key for pyannote.audio speaker diarization service
  - Set this in a `.env` file in the project root or as an environment variable

## Output

### Final Results
The pipeline generates a JSON file (`results.json` in the root directory) with the following structure:

```json
[
  {
    "speaker_id": "SPEAKER_00",
    "face_id": "person_1",
    "text": "Hello, how are you?",
    "start": 1.185,
    "end": 1.585,
    "timestamp_processed": 0.985,
    "match_distance": 0.267
  }
]
```

**Fields:**
- `speaker_id`: Speaker identifier from diarization (e.g., `SPEAKER_00`, `SPEAKER_01`)
- `face_id`: Matched face ID from database (e.g., `person_1`, `person_2`) or `null` if no match
- `text`: Transcribed text for this speaker turn
- `start`: Start time in seconds
- `end`: End time in seconds
- `timestamp_processed`: Timestamp used for face extraction (start - 0.2s buffer)
- `match_distance`: Cosine distance to matched face (lower = better match), or `null` if no match

### Intermediate Outputs
Each pipeline run creates a directory in `intermediate_outputs/run_<run_id>/` containing:

- `audio/extracted_audio.wav`: Extracted audio file
- `diarization.json`: Raw speaker diarization results
- `face_matching.json`: Face matching results for all detected faces
- `final_results.json`: Final combined results (same as root `results.json`)
- `faces/t<timestamp>/`: Face crops extracted at each timestamp
  - `face_XX_<color>.jpg`: Individual face crops (color = green/red)
  - `speaker_face.jpg`: Identified speaker face crop
- `frames/`: Original and annotated frames
  - `t<timestamp>_frame.jpg`: Original frame
  - `t<timestamp>_frame_annotated.jpg`: Frame with bounding boxes drawn

## Usage

### Basic Usage

```bash
python ingest_pipeline.py <video_filename>
```

**Example:**
```bash
python ingest_pipeline.py 30s_clip.mp4
```

The video file should be located in the `data/` directory:
```
ingest_pipeline/
  data/
    30s_clip.mp4  # Your video file here
```

### Output Location

- **Quick results**: `results.json` in the pipeline root directory
- **Detailed outputs**: `intermediate_outputs/run_<run_id>/` directory

## Configuration

Default configuration (can be modified in `ingest_pipeline.py`):

- `BUFFER_SEC = 0.2`: Buffer added before speaker turn start for face extraction
- `FACE_MODEL = "ArcFace"`: Face recognition model
- `FACE_DETECTOR = "opencv"`: Face detection backend
- `FACE_DISTANCE_METRIC = "cosine"`: Distance metric for face matching
- `DATA_DIR`: Directory containing input videos (default: `data/`)
- `FACE_DB_DIR`: Directory containing face database images (default: `face_database/`)

## Dependencies

The pipeline requires:
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- pyannote.audio API access
- TalkNet (included in `talknet/` directory)
- DeepFace (for face matching)
- ffmpeg (for audio extraction)

## Notes

- The pipeline processes each unique speaker turn start timestamp (deduplicates if multiple turns start at the same time)
- Face matching uses cached embeddings stored in `deepface/db_embeddings.pkl` for faster processing
- If no faces are detected or matched, `face_id` will be `null` in the results
- TalkNet may take some time to process the video on first run (it processes the entire video)
- Subsequent runs for the same video will be faster if TalkNet outputs are cached
