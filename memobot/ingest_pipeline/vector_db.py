"""
Pipeline to ingest video and create embeddings using TwelveLabs API + Pinecone,
with Pegasus summaries and importance scores as metadata.
"""

import os
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

from dotenv import load_dotenv
from twelvelabs import TwelveLabs
from typing import Any
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file at memobot root
MEMOBOT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MEMOBOT_ROOT))
load_dotenv(dotenv_path=MEMOBOT_ROOT / ".env")
TL_API_KEY = os.getenv("TWELVE_LABS_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Person DB helpers (stored in memobot/utils/database.py)
from utils.database import get_person_by_face_id, get_person_by_speaker_id

# TwelveLabs client (used by generate_embedding and ingest_data)
twelvelabs_client = TwelveLabs(api_key=TL_API_KEY)

# Default paths (can be overridden via command-line args)
DATA_DIR = Path(__file__).parent / "data"
VIDEO_PATH = str(DATA_DIR / "30s_clip.mp4")  # Default video
INDEX_NAME = "twelve-labs"
PEGASUS_INDEX_NAME = "pegasus-video-memories"  # can reuse this for many videos


def on_task_update(task: Any):
    """Callback function to monitor embedding task progress."""
    print(f"  Status={task.status}")


def on_pegasus_task_update(task: Any):
    """Callback for Pegasus indexing."""
    print(f"  Pegasus index status={task.status}")


# ---------- PEGASUS HELPERS (SUMMARY + IMPORTANCE) ----------

def ensure_pegasus_index(tl_client: TwelveLabs, index_name: str = PEGASUS_INDEX_NAME):
    """
    Ensure there is a Pegasus index for analysis.
    Creates it once if it doesn't exist.
    """
    # TwelveLabs SDK uses `client.indexes` (plural) in newer versions.
    # Keep a small compatibility shim for older SDKs that used `client.index`.
    indexes_client = getattr(tl_client, "indexes", None) or getattr(tl_client, "index", None)
    if indexes_client is None:
        raise AttributeError(
            "Your installed 'twelvelabs' SDK does not expose 'client.indexes' or 'client.index'. "
            "Try upgrading: pip install -U 'twelvelabs>=1.3.0'"
        )

    existing = list(indexes_client.list())
    for idx in existing:
        idx_name = getattr(idx, "index_name", None) or (idx.get("index_name") if isinstance(idx, dict) else None)
        if idx_name == index_name:
            return idx

    print(f"Creating Pegasus index: {index_name}")
    index = indexes_client.create(
        index_name=index_name,
        models=[
            {
                "model_name": "pegasus1.2",
                "model_options": ["visual", "audio"]
            }
        ],
    )
    print(f"Created Pegasus index: id={index.id}")
    return index


def upload_video_to_pegasus(
    tl_client: TwelveLabs,
    index_id: str,
    video_source: str,
) -> str:
    """
    Upload the same video to Pegasus index and return its video_id.
    """
    is_url = video_source.startswith(("http://", "https://"))

    # TwelveLabs SDK uses `client.tasks` (plural) in newer versions.
    # Keep a small compatibility shim for older SDKs that used `client.task`.
    tasks_client = getattr(tl_client, "tasks", None) or getattr(tl_client, "task", None)
    if tasks_client is None:
        raise AttributeError(
            "Your installed 'twelvelabs' SDK does not expose 'client.tasks' or 'client.task'. "
            "Try upgrading: pip install -U 'twelvelabs>=1.3.0'"
        )

    if is_url:
        task = tasks_client.create(
            index_id=index_id,
            video_url=video_source,
        )
    else:
        video_file_path = os.path.abspath(video_source)
        if not os.path.exists(video_file_path):
            raise FileNotFoundError(f"Video file not found for Pegasus: {video_file_path}")
        task = tasks_client.create(
            index_id=index_id,
            video_file=open(video_file_path, "rb"),
        )

    print(f"Pegasus task created: id={task.id}")
    task = tasks_client.wait_for_done(
        task_id=task.id,
        callback=on_pegasus_task_update
    )
    if task.status != "ready":
        raise RuntimeError(f"Pegasus indexing failed with status {task.status}")
    print(f"Pegasus upload complete. video_id={task.video_id}")
    return task.video_id


def analyze_segment_with_pegasus(
    tl_client: TwelveLabs,
    video_id: str,
    start_sec: float,
    end_sec: float,
    embedding_option=None,
    segment_dialogue=None,
):
    """
    Call Pegasus analyze_stream for a specific time window and
    return (summary, importance_score, talking_to_camera).

    We prompt Pegasus to respond with strict JSON:
    {
      "summary": "...",
      "importance_score": 1-10,
      "talking_to_camera": 0.0-1.0
    }

    When segment_dialogue is provided (from process_video), the summary MUST
    include the spoken words and the name of who spoke (face_id or speaker_id).

    Parameters
    ----------
    embedding_option : str or list, optional
        The embedding option type: "visual", "audio", "transcription", or a list
    segment_dialogue : list of dict, optional
        From process_video: [{speaker_id, face_id, text, start, end}, ...]
        for turns overlapping this segment. Used to include spoken words and
        speaker identity in the summary.

    Returns
    -------
    tuple
        (summary, importance_score, talking_to_camera)
    """
    # Determine the primary embedding option (handle both string and list)
    if isinstance(embedding_option, list):
        primary_option = embedding_option[0] if embedding_option else "visual"
    else:
        primary_option = embedding_option or "visual"

    # Build dialogue block when we have identified speech from process_video
    dialogue_block = ""
    if segment_dialogue:
        lines = []
        for t in segment_dialogue:
            name = t.get("face_id") or t.get("speaker_id") or "Unknown"
            text = (t.get("text") or "").strip()
            if text:
                lines.append(f"  - {name}: \"{text}\"")
        if lines:
            dialogue_block = """
The following spoken dialogue was identified in this segment (who said what).
Your summary MUST include this dialogue, attributing each quote to the speaker
(use the person name/face_id when available, otherwise speaker_id):

"""
            dialogue_block += "\n".join(lines)
            dialogue_block += "\n\n"

    # Create different prompts based on embedding option
    if primary_option == "transcription":
        instruction = f"""
1. Briefly summarize in ONE concise sentence what happens in this time range.
   FOCUS: This segment is based on transcription/audio. Include ALL spoken words,
   dialogue, and any text that appears. Quote or paraphrase exactly what is said.
   If there are multiple speakers, identify who says what (by name/face_id when given).
{dialogue_block}   When dialogue is provided above, your summary MUST include those spoken words
   and the name of who spoke each one. Include any on-screen text.
"""
    elif primary_option == "audio":
        instruction = f"""
1. Briefly summarize in ONE concise sentence what happens in this time range.
   FOCUS: This segment is based on audio. Include ALL spoken words, dialogue,
   sounds, music, and audio cues. Quote or paraphrase what is said. Describe
   any important sounds or audio events.
{dialogue_block}   When dialogue is provided above, your summary MUST include those spoken words
   and the name of who spoke each one. Include any on-screen text if visible.
"""
    else:  # visual (default)
        instruction = f"""
1. Briefly summarize in ONE concise sentence what happens in this time range.
   FOCUS: This segment is based on visual content. Describe what you see happening
   visually.
{dialogue_block}   When dialogue is provided above, your summary MUST include those spoken words
   and the name of who spoke each one. If there is any other spoken dialogue or
   on-screen text, include it as well.
"""

    prompt = f"""
You are analyzing a short memory segment from a longer video.

Only consider the portion of the video between {start_sec:.2f} and {end_sec:.2f} seconds.

{instruction}
2. Rate how important this event is for understanding the agent's day,
   on a scale from 1 to 10. Use the FULL range of scores - don't default to middle values.
   
   Scoring guidelines:
   - 1-2: Trivial background noise, idle waiting, meaningless filler (e.g., "um", "uh", silence)
   - 3-4: Minor routine actions, casual conversation without significance, ambient sounds
   - 5-6: Normal activities, standard interactions, routine tasks (e.g., opening a door, basic movement)
   - 7-8: Notable events, meaningful conversations, important actions, decisions being made
   - 9-10: Critical moments, major decisions, significant events that strongly impact the day,
           emotional moments, task completions, failures, or breakthroughs
   
   IMPORTANT: Just because someone is talking doesn't make it important. Consider:
   - Is this a meaningful conversation or just filler?
   - Does this action/event have consequences?
   - Would forgetting this change understanding of the day?
   - Is this routine vs. exceptional?
   
   Use scores across the full 1-10 range based on actual significance, not just presence of speech.

3. Determine if ANYONE is talking to or looking at the camera in this segment.
   This measures whether any person in the video is addressing the camera (looking at or speaking to the camera).
   Provide a confidence score strictly between 0.0 and 1.0 (never exactly 0.0 or 1.0):
   - 0.01-0.2: Very unlikely - no one appears to be addressing the camera (e.g., everyone looking away, talking to others, no one visible)
   - 0.3-0.4: Possibly looking at camera but unclear (e.g., brief glance, ambiguous direction, might be addressing camera)
   - 0.5: Uncertain or ambiguous (e.g., unclear who is being addressed, partial camera view, could go either way)
   - 0.6-0.7: Likely talking to/looking at camera (e.g., facing camera direction, using "you", but not fully clear)
   - 0.8-0.99: Very likely talking to or looking at the camera (e.g., direct eye contact with camera, clearly addressing camera, 
                sustained gaze at camera, direct speech to camera/viewer)
   
   IMPORTANT: The score must ALWAYS be between 0.0 and 1.0 (exclusive) - never exactly 0.0 or 1.0.
   Use values like 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, etc.
   
   KEY INDICATORS for high confidence (0.8-0.99):
   - Direct eye contact with the camera lens
   - Person facing the camera and speaking
   - Sustained gaze/look into the camera
   - Speech clearly directed at the camera/viewer
   - Using second person ("you") while looking at camera

Respond ONLY in valid JSON, with this exact schema:
{{
  "summary": "<one sentence>",
  "importance_score": <integer between 1 and 10>,
  "talking_to_camera": <float strictly between 0.0 and 1.0 (never exactly 0.0 or 1.0)>
}}
"""

    # TwelveLabs streaming analysis is rate-limited (e.g., 8 req / minute).
    # We only *intend* to call this once per 30s clip, but add a small retry
    # shim for transient 429s so reruns don't instantly fail.
    collected = ""
    max_attempts = 6
    for attempt in range(1, max_attempts + 1):
        try:
            text_stream = tl_client.analyze_stream(video_id=video_id, prompt=prompt)
            for chunk in text_stream:
                if chunk.event_type == "text_generation":
                    collected += chunk.text
            break
        except Exception as e:
            status_code = getattr(e, "status_code", None)
            headers = getattr(e, "headers", None) or {}
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
            if status_code == 429 and retry_after is not None and attempt < max_attempts:
                try:
                    sleep_s = float(retry_after)
                except Exception:
                    sleep_s = 10.0
                time.sleep(max(0.0, sleep_s))
                continue
            raise

    # Try to extract JSON object from the streamed text
    match = re.search(r"\{.*\}", collected, re.DOTALL)
    if not match:
        # Fallback: treat whole text as summary with unknown importance and confidence
        return collected.strip(), None, None

    try:
        data = json.loads(match.group(0))
        summary = data.get("summary", "").strip()
        importance = data.get("importance_score", None)
        talking_to_camera = data.get("talking_to_camera", None)
        # Ensure talking_to_camera is a float strictly between 0.0 and 1.0 (never exactly 0 or 1)
        if talking_to_camera is not None:
            talking_to_camera = float(talking_to_camera)
        return summary, importance, talking_to_camera
    except json.JSONDecodeError:
        return collected.strip(), None, None


# ---------- EMBEDDING + PINECONE PIPELINE ----------

def generate_embedding(video_source, clip_length=30):
    """
    Generate embeddings for a video using TwelveLabs Embed v_2 API.

    Notes (TwelveLabs v1.3):
    - "clip"-scope embeddings are constrained to 2–10s clip lengths.
    - To embed an entire 30s (or longer) file as a single vector, request
      the "video" scope.
    """
    # Ensure the installed SDK supports v1.3 Embed v_2 + Assets API.
    # Older SDKs expose only embed.task and will not work with the v1.3 quickstart.
    if not hasattr(twelvelabs_client, "assets") or not hasattr(getattr(twelvelabs_client, "embed", None), "v_2"):
        raise RuntimeError(
            "Your installed 'twelvelabs' SDK is too old (missing client.assets and/or client.embed.v_2).\n"
            "Upgrade it and retry:\n"
            "  pip install -U 'twelvelabs>=1.3.0'\n"
            "This repo's requirements.txt now pins twelvelabs>=1.3.0."
        )

    # Import request helper classes if available (SDK also accepts plain dicts).
    try:
        from twelvelabs import VideoInputRequest, MediaSource  # type: ignore
        have_models = True
    except Exception:
        VideoInputRequest = None  # type: ignore
        MediaSource = None  # type: ignore
        have_models = False

    is_url = video_source.startswith(("http://", "https://"))

    # 1) Create an asset (upload) from URL or local file
    if is_url:
        print(f"Uploading video via URL: {video_source}")
        asset = twelvelabs_client.assets.create(method="url", url=video_source)
    else:
        video_file_path = os.path.abspath(video_source)
        if not os.path.exists(video_file_path):
            raise FileNotFoundError(f"Video file not found: {video_file_path}")
        print(f"Uploading local video file: {video_file_path}")
        with open(video_file_path, "rb") as f:
            asset = twelvelabs_client.assets.create(method="direct", file=f)

    asset_id = getattr(asset, "id", None) or getattr(asset, "_id", None)
    if not asset_id:
        raise RuntimeError("Failed to create TwelveLabs asset (missing asset id).")
    print(f"Created asset: id={asset_id}")

    # 2) Request embeddings for the 30s clip as ONE vector per modality.
    #
    # In the current TwelveLabs SDK, this is done via:
    # - embedding_scope=["asset"] (one vector for the whole asset / requested range)
    # - embedding_option=["visual","audio","transcription"] (three modalities)
    # - start_sec/end_sec to constrain to the 30s clip
    if have_models:
        video_req = VideoInputRequest(
            media_source=MediaSource(asset_id=asset_id),
            start_sec=0.0,
            end_sec=float(clip_length),
            embedding_scope=["asset"],
            embedding_option=["visual", "audio", "transcription"],
        )
    else:
        video_req = {
            "media_source": {"asset_id": asset_id},
            "start_sec": 0.0,
            "end_sec": float(clip_length),
            "embedding_scope": ["asset"],
            "embedding_option": ["visual", "audio", "transcription"],
        }

    response = twelvelabs_client.embed.v_2.create(
        input_type="video",
        model_name="marengo3.0",
        video=video_req,
    )

    embeddings = []
    data = getattr(response, "data", None) or []
    for emb in data:
        # Normalize SDK field naming across versions
        start = getattr(emb, "start_sec", getattr(emb, "startSec", 0.0))
        end = getattr(emb, "end_sec", getattr(emb, "endSec", 0.0))
        option = getattr(emb, "embedding_option", getattr(emb, "embeddingOption", "visual"))
        scope = getattr(emb, "embedding_scope", getattr(emb, "embeddingScope", "video"))
        vec = getattr(emb, "embedding", None)
        if vec is None:
            continue
        embeddings.append(
            {
                "embedding": vec,
                "start_offset_sec": float(start or 0.0),
                "end_offset_sec": float(end or float(clip_length)),
                "embedding_scope": scope,
                "embedding_option": option,
            }
        )

    if not embeddings:
        raise RuntimeError("No embeddings returned from TwelveLabs embed.v_2.create().")

    return embeddings, response


def _turns_in_segment(process_video_results, start_sec: float, end_sec: float):
    """Return process_video turns that overlap [start_sec, end_sec]."""
    if not process_video_results:
        return []
    out = []
    for t in process_video_results:
        turn_start = float(t.get("start", 0))
        turn_end = float(t.get("end", 0))
        if turn_start < end_sec and turn_end > start_sec:
            out.append(t)
    return out


def _persons_in_turns(process_video_turns):
    """
    Resolve unique persons for the clip based on:
    - face_id -> persons.face_id
    - voiceprint_label (speaker id) -> persons.speaker_id

    Returns:
        (person_ids: list[str], persons: list[dict]) where persons are:
          {person_id, name, face_id, speaker_id}
    """
    if not process_video_turns:
        return [], []

    persons_by_id = {}

    for t in process_video_turns:
        face_id = t.get("face_id") or None
        speaker_id = t.get("voiceprint_label") or t.get("speaker_id") or None
        if speaker_id == "UNKNOWN":
            speaker_id = None

        # Prefer face match (more specific), then speaker_id mapping.
        person = None
        if face_id:
            person = get_person_by_face_id(face_id)
        if person is None and speaker_id:
            person = get_person_by_speaker_id(speaker_id)

        if person and person.get("person_id"):
            pid = person["person_id"]
            persons_by_id[pid] = {
                "person_id": pid,
                "name": person.get("name"),
                "face_id": person.get("face_id"),
                "speaker_id": person.get("speaker_id"),
            }

    persons = list(persons_by_id.values())
    person_ids = [p["person_id"] for p in persons]
    return person_ids, persons


def _dialogue_transcript(process_video_turns) -> str:
    """
    Build a spoken transcript for the clip, replacing raw speaker ids with person names when possible.

    Input turns come from process_video: [{speaker_id, voiceprint_label, face_id, text, start, end}, ...]
    Output is a human-readable transcript like:
        Alice: "..."
        Bob: "..."
    """
    if not process_video_turns:
        return ""

    # Ensure chronological order
    turns_sorted = sorted(process_video_turns, key=lambda t: float(t.get("start", 0.0)))

    lines: list[str] = []
    for t in turns_sorted:
        text = (t.get("text") or "").strip()
        if not text:
            continue

        face_id = t.get("face_id") or None
        speaker_id = t.get("voiceprint_label") or t.get("speaker_id") or None
        if speaker_id == "UNKNOWN":
            speaker_id = None

        person = None
        if face_id:
            person = get_person_by_face_id(face_id)
        if person is None and speaker_id:
            person = get_person_by_speaker_id(speaker_id)

        name = None
        if person:
            name = person.get("name") or None

        label = name or face_id or speaker_id or "Unknown"
        lines.append(f'{label}: "{text}"')

    return "\n".join(lines)


def ingest_data(video_source, index_name="twelve-labs", clip_length=30, process_video_results=None, memobot_group_id=None):
    """
    Generate one embedding and one Pegasus summary per video, then upsert to Pinecone.
    Input is assumed to be a 30-second clip: one segment (0 to clip_length), one vector.

    When process_video_results is provided (from process_video), the Pegasus summary
    will include the spoken words and who spoke (face_id or speaker_id) for the clip.
    memobot_group_id is stored in Pinecone metadata for filtering/tenant isolation.
    """
    process_video_results = process_video_results or []
    if memobot_group_id is None:
        memobot_group_id = os.getenv("MEMOBOT_GROUP_ID", "tenant_003")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Extract video name
    if video_source.startswith(('http://', 'https://')):
        video_name = os.path.splitext(os.path.basename(video_source.split('?')[0]))[0]
    else:
        video_name = os.path.splitext(os.path.basename(video_source))[0]
    print(f"Processing video: {video_name}")

    # Connect to / create Pinecone index
    if index_name not in pc.list_indexes().names():
        print(f"Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=512,  # TwelveLabs embedding dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    index = pc.Index(index_name)

    # 1) Generate embeddings with Marengo
    embeddings, task_result = generate_embedding(video_source, clip_length)

    # 2) Ensure Pegasus index + upload video once
    pegasus_index = ensure_pegasus_index(twelvelabs_client, PEGASUS_INDEX_NAME)
    pegasus_video_id = upload_video_to_pegasus(
        twelvelabs_client,
        pegasus_index.id,
        video_source
    )

    # 3) Common timestamp (ingestion time); you can swap this to "world time" later
    now_utc = datetime.now(timezone.utc).isoformat()

    # 4) Pegasus summary is intended to be computed ONCE per 30s clip
    # (not once per embedding modality), then reused for all vectors.
    clip_start_sec = 0.0
    clip_end_sec = float(clip_length)
    clip_dialogue = _turns_in_segment(process_video_results, clip_start_sec, clip_end_sec)
    person_ids, persons = _persons_in_turns(clip_dialogue)
    audio_dialogue = _dialogue_transcript(clip_dialogue)
    summary, importance, talking_to_camera = analyze_segment_with_pegasus(
        twelvelabs_client,
        pegasus_video_id,
        clip_start_sec,
        clip_end_sec,
        embedding_option="visual",
        segment_dialogue=clip_dialogue,
    )

    # 5) Prepare vectors with rich metadata (one per embedding option)
    vectors_to_upsert = []
    for i, emb in enumerate(embeddings):
        start_sec = emb["start_offset_sec"]
        end_sec = emb["end_offset_sec"]

        # Extract embedding option for vector ID (handle both string and list)
        emb_option = emb['embedding_option']
        if isinstance(emb_option, list):
            emb_option_str = emb_option[0] if emb_option else "visual"
        else:
            emb_option_str = emb_option or "visual"

        # Map embedding option to a shorter name for vector ID
        option_map = {
            "transcription": "text",
            "visual": "visual",
            "audio": "audio"
        }
        option_suffix = option_map.get(emb_option_str.lower(), emb_option_str.lower())

        vector_id = f"{video_name}_{i}_{option_suffix}"
        metadata = {
            'video_file': video_name,
            'start_time_sec': start_sec,
            'end_time_sec': end_sec,
            'scope': emb['embedding_scope'],
            'embedding_option': emb['embedding_option'],
            'timestamp_utc': now_utc,           # when this memory was stored
            'summary': summary,                 # one-sentence description
            'importance_score': importance,     # 1–10 (can be None if parsing fails)
            'talking_to_camera': talking_to_camera,  # 0.0-1.0 (can be None if parsing fails)
            'pegasus_video_id': pegasus_video_id,
            'person_ids': person_ids,           # all recognized person_ids in this 30s clip
            'audio_dialogue': audio_dialogue,   # transcript with person names when possible
            'memobot_group_id': memobot_group_id,  # tenant/group for filtering
        }

        vectors_to_upsert.append(
            (vector_id, emb['embedding'], metadata)
        )

    # 6) Upsert into Pinecone
    index.upsert(vectors=vectors_to_upsert)
    print(f"Ingested {len(embeddings)} embeddings for {video_source}")

    return {
        "video_source": video_source,
        "video_name": video_name,
        "clip_start_sec": clip_start_sec,
        "clip_end_sec": clip_end_sec,
        "upserted_vectors": len(embeddings),
        "summary": summary,
        "audio_dialogue": audio_dialogue,
        "person_ids": person_ids,
        "persons": persons,  # [{person_id,name,face_id,speaker_id}, ...]
    }


if __name__ == "__main__":
    # Initialize clients (twelvelabs_client created at module level)
    print("Initializing Pinecone client...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Ingest the video (one embedding + one summary per 30-second clip)
    print(f"\nIngesting video: {VIDEO_PATH}")
    print("One embedding and one summary for the full 30-second clip")
    result = ingest_data(
        VIDEO_PATH,
        index_name=INDEX_NAME,
        clip_length=30
    )
    print(f"\n{result}")
