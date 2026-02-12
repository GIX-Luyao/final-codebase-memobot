"""
Pipeline to ingest video and create embeddings using Google Vertex AI multimodal
embeddings + Pinecone, with Gemini video understanding (summary, importance, etc.) as metadata.
"""

import os
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

from dotenv import load_dotenv
from google.oauth2 import service_account
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file at memobot root
MEMOBOT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = MEMOBOT_ROOT.parent
sys.path.insert(0, str(MEMOBOT_ROOT))
load_dotenv(dotenv_path=MEMOBOT_ROOT / ".env")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Vertex AI multimodal embeddings (project/location from env or service account JSON)
VERTEX_AI_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("VERTEX_AI_PROJECT")
VERTEX_AI_LOCATION = os.getenv("VERTEX_AI_LOCATION", "us-central1")
VERTEX_AI_KEY_PATH = Path(
    os.getenv("VERTEX_AI_SERVICE_ACCOUNT_JSON", str(PROJECT_ROOT / "vertex_ai_service_account.json"))
)
# Pinecone index uses 512 dimensions; Vertex supports 128, 256, 512, 1408
EMBEDDING_DIMENSION = 1408

# Gemini API (video understanding + caption) - uses GOOGLE_API_KEY
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Person DB helpers (person DB has no speaker_id; identify by face_id only)
from utils.database import get_person_by_face_id

# Default paths (can be overridden via command-line args)
DATA_DIR = Path(__file__).parent / "data"
VIDEO_PATH = str(DATA_DIR / "30s_clip.mp4")  # Default video
INDEX_NAME = "memobot-memories"


# ---------- EMBEDDING + PINECONE PIPELINE ----------

def generate_embedding(video_source, clip_length=30):
    """
    Generate embeddings for a video using Google Vertex AI multimodal embeddings
    (multimodalembedding@001). Supports local file path or GCS URI (gs://...).

    Ref: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-multimodal-embeddings
    """
    import vertexai
    from vertexai.vision_models import MultiModalEmbeddingModel, Video
    from vertexai.vision_models import VideoSegmentConfig

    project = VERTEX_AI_PROJECT
    credentials = None
    if VERTEX_AI_KEY_PATH.exists():
        credentials = service_account.Credentials.from_service_account_file(str(VERTEX_AI_KEY_PATH))
        if not project:
            with open(VERTEX_AI_KEY_PATH) as f:
                project = json.load(f).get("project_id")
    if not project:
        raise RuntimeError(
            "Vertex AI project not set. Set GOOGLE_CLOUD_PROJECT or VERTEX_AI_PROJECT in .env, "
            "or use a service account JSON that contains project_id."
        )

    vertexai.init(
        project=project,
        location=VERTEX_AI_LOCATION,
        credentials=credentials,
    )
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")

    # Load video: GCS URI or local path (SDK handles base64 for local files)
    is_gcs = video_source.startswith("gs://")
    if is_gcs:
        print(f"Loading video from GCS: {video_source}")
        video = Video.load_from_file(video_source)
    else:
        video_path = os.path.abspath(video_source)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        print(f"Loading local video: {video_path}")
        video = Video.load_from_file(video_path)

    # One segment for the clip [0, clip_length]. Min interval_sec is 4.
    interval_sec = max(4.0, min(float(clip_length), 120.0))
    video_segment_config = VideoSegmentConfig(
        start_offset_sec=0,
        end_offset_sec=float(clip_length),
        interval_sec=interval_sec,
    )

    embeddings_response = model.get_embeddings(
        video=video,
        video_segment_config=video_segment_config,
        dimension=EMBEDDING_DIMENSION,
    )

    embeddings = []
    for seg in embeddings_response.video_embeddings:
        start_sec = getattr(seg, "start_offset_sec", 0.0)
        end_sec = getattr(seg, "end_offset_sec", float(clip_length))
        vec = getattr(seg, "embedding", None)
        if vec is None:
            continue
        embeddings.append(
            {
                "embedding": list(vec) if hasattr(vec, "__iter__") and not isinstance(vec, str) else vec,
                "start_offset_sec": float(start_sec),
                "end_offset_sec": float(end_sec),
                "embedding_scope": "asset",
                "embedding_option": "video",
            }
        )

    if not embeddings:
        raise RuntimeError("No video embeddings returned from Vertex AI multimodal embedding model.")

    return embeddings, embeddings_response


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
    Resolve unique persons for the clip from process_video turns.
    Person DB has no speaker_id; identity is by face_id -> person_id only.

    Returns:
        (person_ids: list[str], persons: list[dict]) where persons are:
          {person_id, name, face_id}
    """
    if not process_video_turns:
        return [], []

    persons_by_id = {}

    for t in process_video_turns:
        person_id = t.get("person_id")
        face_id = t.get("face_id") or None
        name = t.get("name")
        if not person_id and face_id:
            person = get_person_by_face_id(face_id)
            if person:
                person_id = person.get("person_id")
                name = person.get("name")

        if person_id:
            persons_by_id[person_id] = {
                "person_id": person_id,
                "name": name,
                "face_id": face_id,
            }

    persons = list(persons_by_id.values())
    person_ids = [p["person_id"] for p in persons]
    return person_ids, persons


def _dialogue_transcript(process_video_turns) -> str:
    """
    Build a spoken transcript for the clip using person_id only (no names).

    Input turns come from process_video: [{person_id, name, face_id, text, start, end}, ...]
    Output format: person_id: "..." or Unknown: "..."
    """
    if not process_video_turns:
        return ""

    turns_sorted = sorted(process_video_turns, key=lambda t: float(t.get("start", 0.0)))
    lines: list[str] = []
    for t in turns_sorted:
        text = (t.get("text") or "").strip()
        if not text:
            continue
        label = t.get("person_id") or t.get("face_id") or "Unknown"
        lines.append(f'{label}: "{text}"')
    return "\n".join(lines)


def _video_name_from_source(video_source: str) -> str:
    """Extract video name from URL or path."""
    if video_source.startswith(("http://", "https://")):
        return os.path.splitext(os.path.basename(video_source.split("?")[0]))[0]
    return os.path.splitext(os.path.basename(video_source))[0]


def run_embeddings_only(video_source: str, index_name: str, clip_length: float):
    """
    Memory builder — embeddings side only (Vertex AI multimodal).
    Safe to run in parallel with process_video and run_gemini_video_understanding.
    """
    t_start = time.perf_counter()
    video_name = _video_name_from_source(video_source)
    print(f"[Embeddings] Processing video: {video_name}")
    embeddings, _ = generate_embedding(video_source, clip_length)
    t_end = time.perf_counter()
    print(f"[Timing] Embeddings (Vertex AI multimodal) took: {t_end - t_start:.2f}s")
    return {
        "video_source": video_source,
        "video_name": video_name,
        "clip_length": float(clip_length),
        "clip_start_sec": 0.0,
        "clip_end_sec": float(clip_length),
        "embeddings": embeddings,
    }


# ---------- GEMINI VIDEO UNDERSTANDING (SUMMARY + IMPORTANCE + TALKING_TO_CAMERA) ----------

GEMINI_VIDEO_PROMPT = """You are analyzing a short memory segment from a video (audio + visual).

1. In ONE concise sentence, summarize what happens in this segment: what you see and hear, who appears, what is said or shown, and key actions or dialogue. Include timestamps (MM:SS) for salient moments if helpful.

2. Rate how important this event is for understanding the agent's day, on a scale from 1 to 10. Use the full range:
   - 1-2: Trivial (silence, filler, idle)
   - 3-4: Minor routine, casual chat
   - 5-6: Normal activities, routine tasks
   - 7-8: Notable events, meaningful conversation
   - 9-10: Critical moments, major decisions
   Consider: Is this meaningful or filler? Would forgetting this change understanding?

3. How likely is anyone talking to or looking at the camera? Give a confidence score strictly between 0.0 and 1.0 (never exactly 0.0 or 1.0):
   - 0.01-0.2: No one addressing camera
   - 0.5: Uncertain
   - 0.8-0.99: Very likely (eye contact, facing camera, speaking to viewer)

4. In a short paragraph (detailed_understanding), describe key events with both audio and visual details and timestamps.

Respond ONLY with valid JSON in this exact schema:
{
  "summary": "<one sentence>",
  "importance_score": <integer 1-10>,
  "talking_to_camera": <float strictly between 0.0 and 1.0>,
  "detailed_understanding": "<paragraph with key events, timestamps, audio/visual details>"
}"""


def run_gemini_video_understanding(video_source: str, clip_length: float):
    """
    Get video understanding from Gemini API: summary, importance_score, talking_to_camera, detailed_understanding.
    Uses File API for upload then generate_content. Supports local file path only.
    Ref: https://ai.google.dev/gemini-api/docs/video-understanding
    Returns dict with summary, importance_score, talking_to_camera, gemini_understanding (detailed_understanding).
    """
    empty = {
        "summary": "",
        "importance_score": None,
        "talking_to_camera": None,
        "gemini_understanding": "",
    }
    if not GOOGLE_API_KEY:
        print("[Gemini] Skipping video understanding: GOOGLE_API_KEY not set.")
        return empty

    is_url = video_source.startswith(("http://", "https://"))
    if is_url:
        print("[Gemini] Skipping video understanding: URL input not supported (use local file or GCS).")
        return empty

    video_path = os.path.abspath(video_source)
    if not os.path.exists(video_path):
        print(f"[Gemini] Skipping video understanding: file not found {video_path}")
        return empty

    t_start = time.perf_counter()
    try:
        from google import genai
    except ImportError:
        print("[Gemini] Skipping video understanding: google-genai not installed (pip install google-genai).")
        return empty

    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        from google.genai import types as genai_types
    except Exception as e:
        print(f"[Gemini] Skipping video understanding: client init failed: {e}")
        return empty

    try:
        myfile = client.files.upload(file=video_path)
        # File must be ACTIVE before use; upload returns while still PROCESSING.
        file_name = getattr(myfile, "name", None) or myfile
        if isinstance(file_name, str):
            poll_interval = 2
            max_wait_sec = 120
            waited = 0
            while waited < max_wait_sec:
                f = client.files.get(name=file_name)
                state = getattr(f, "state", None)
                state_str = str(state).split(".")[-1] if state is not None else ""
                if state == genai_types.FileState.ACTIVE or "ACTIVE" in state_str:
                    break
                if state == genai_types.FileState.FAILED or "FAILED" in state_str:
                    print(f"[Gemini] Uploaded file entered FAILED state: {file_name}")
                    return empty
                time.sleep(poll_interval)
                waited += poll_interval
                print(".", end="", flush=True)
            if waited >= max_wait_sec:
                print(f"[Gemini] File did not become ACTIVE within {max_wait_sec}s")
                return empty
            print(" ready.")
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[myfile, GEMINI_VIDEO_PROMPT],
        )
        text = (response.text or "").strip()
        if not text:
            print("[Gemini] Video understanding returned empty text.")
            return empty
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {"summary": text[:500], "importance_score": None, "talking_to_camera": None, "gemini_understanding": text}
        data = json.loads(match.group(0))
        summary = (data.get("summary") or "").strip()
        importance = data.get("importance_score")
        talking = data.get("talking_to_camera")
        if talking is not None:
            talking = float(talking)
        detailed = (data.get("detailed_understanding") or "").strip()
        t_end = time.perf_counter()
        print(f"[Timing] Gemini video understanding took: {t_end - t_start:.2f}s")
        return {
            "summary": summary or text[:500],
            "importance_score": importance,
            "talking_to_camera": talking,
            "gemini_understanding": detailed or text,
        }
    except Exception as e:
        print(f"[Gemini] Video understanding failed: {e}")
        return empty


def merge_and_upsert(
    video_source: str,
    index_name: str,
    clip_length: float,
    embeddings_result: dict,
    process_video_results: list,
    memobot_group_id: str = None,
    full_audio_dialogue: str = None,
    gemini_result: dict = None,
):
    """
    Merge results from process_video, embeddings, and Gemini; upsert to Pinecone.
    Call after running process_video, run_embeddings_only, and run_gemini_video_understanding
    in parallel. If full_audio_dialogue is provided (robot + person_id lines), it is
    used as audio_dialogue; otherwise built from process_video_results.
    """
    if memobot_group_id is None:
        memobot_group_id = os.getenv("MEMOBOT_GROUP_ID", "tenant_003")

    gemini_result = gemini_result or {}
    process_video_results = process_video_results or []
    video_name = embeddings_result["video_name"]
    embeddings = embeddings_result["embeddings"]
    clip_start_sec = embeddings_result["clip_start_sec"]
    clip_end_sec = embeddings_result["clip_end_sec"]

    summary = gemini_result.get("summary") or ""
    importance = gemini_result.get("importance_score")
    talking_to_camera = gemini_result.get("talking_to_camera")
    gemini_understanding = gemini_result.get("gemini_understanding") or ""

    clip_dialogue = _turns_in_segment(process_video_results, clip_start_sec, clip_end_sec)
    person_ids, persons = _persons_in_turns(clip_dialogue)
    audio_dialogue = (full_audio_dialogue if full_audio_dialogue is not None else _dialogue_transcript(clip_dialogue))

    t_start = time.perf_counter()
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    if index_name not in pc.list_indexes().names():
        print(f"Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=512,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    index = pc.Index(index_name)
    now_utc = datetime.now(timezone.utc).isoformat()

    vectors_to_upsert = []
    for i, emb in enumerate(embeddings):
        start_sec = emb["start_offset_sec"]
        end_sec = emb["end_offset_sec"]
        emb_option = emb["embedding_option"]
        emb_option_str = emb_option[0] if isinstance(emb_option, list) and emb_option else (emb_option or "visual")
        if not isinstance(emb_option_str, str):
            emb_option_str = "visual"
        option_map = {"transcription": "text", "visual": "visual", "audio": "audio"}
        option_suffix = option_map.get(emb_option_str.lower(), emb_option_str.lower())
        vector_id = f"{video_name}_{i}_{option_suffix}"
        metadata = {
            "video_file": video_name,
            "start_time_sec": start_sec,
            "end_time_sec": end_sec,
            "scope": emb["embedding_scope"],
            "embedding_option": emb["embedding_option"],
            "timestamp_utc": now_utc,
            "summary": summary,
            "importance_score": importance,
            "talking_to_camera": talking_to_camera,
            "person_ids": person_ids,
            "audio_dialogue": audio_dialogue,
            "memobot_group_id": memobot_group_id,
            "gemini_understanding": gemini_understanding[:40_000] if gemini_understanding else "",  # Pinecone metadata limit
        }
        vectors_to_upsert.append((vector_id, emb["embedding"], metadata))

    index.upsert(vectors=vectors_to_upsert)
    t_end = time.perf_counter()
    print(f"[Timing] Merge + Pinecone upsert took: {t_end - t_start:.2f}s")
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
        "persons": persons,
        "gemini_understanding": gemini_understanding,
    }


def ingest_data(video_source, index_name=None, clip_length=30, process_video_results=None, memobot_group_id=None):
    """
    Generate one embedding and Gemini video understanding per video, then upsert to Pinecone.
    Sequential fallback: runs embeddings and Gemini in sequence, then merge.
    For parallel execution use run_embeddings_only, run_gemini_video_understanding, and merge_and_upsert from main.
    """
    if index_name is None:
        index_name = INDEX_NAME
    process_video_results = process_video_results or []
    if memobot_group_id is None:
        memobot_group_id = os.getenv("MEMOBOT_GROUP_ID", "tenant_003")

    embeddings_result = run_embeddings_only(video_source, index_name, clip_length)
    gemini_result = run_gemini_video_understanding(video_source, clip_length)
    return merge_and_upsert(
        video_source,
        index_name,
        clip_length,
        embeddings_result,
        process_video_results,
        memobot_group_id,
        gemini_result=gemini_result,
    )


if __name__ == "__main__":
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
