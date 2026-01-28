"""
Pipeline to ingest video and create embeddings using TwelveLabs API + Pinecone,
with Pegasus summaries and importance scores as metadata.
"""

import os
import json
import re
from datetime import datetime, timezone

from dotenv import load_dotenv
from twelvelabs import TwelveLabs
from twelvelabs.embed.tasks.types.tasks_status_response import TasksStatusResponse
from twelvelabs.indexes import IndexesCreateRequestModelsItem
from twelvelabs.tasks import TasksRetrieveResponse
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file
load_dotenv()
TL_API_KEY = os.getenv("TWELVE_LABS_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Use the first video in intermediate_data
VIDEO_PATH = "intermediate_data/kitchen_07_chunk_000.mp4"
INDEX_NAME = "twelve-labs"
PEGASUS_INDEX_NAME = "pegasus-video-memories"  # can reuse this for many videos


def on_task_update(task: TasksStatusResponse):
    """Callback function to monitor embedding task progress."""
    print(f"  Status={task.status}")


def on_pegasus_task_update(task: TasksRetrieveResponse):
    """Callback for Pegasus indexing."""
    print(f"  Pegasus index status={task.status}")


# ---------- PEGASUS HELPERS (SUMMARY + IMPORTANCE) ----------

def ensure_pegasus_index(tl_client: TwelveLabs, index_name: str = PEGASUS_INDEX_NAME):
    """
    Ensure there is a Pegasus index for analysis.
    Creates it once if it doesn't exist.
    """
    existing = list(tl_client.indexes.list())
    for idx in existing:
        if idx.index_name == index_name:
            return idx

    print(f"Creating Pegasus index: {index_name}")
    index = tl_client.indexes.create(
        index_name=index_name,
        models=[
            IndexesCreateRequestModelsItem(
                model_name="pegasus1.2",
                model_options=["visual", "audio"]
            )
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

    if is_url:
        task = tl_client.tasks.create(
            index_id=index_id,
            video_url=video_source,
        )
    else:
        video_file_path = os.path.abspath(video_source)
        if not os.path.exists(video_file_path):
            raise FileNotFoundError(f"Video file not found for Pegasus: {video_file_path}")
        task = tl_client.tasks.create(
            index_id=index_id,
            video_file=open(video_file_path, "rb"),
        )

    print(f"Pegasus task created: id={task.id}")
    task = tl_client.tasks.wait_for_done(
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
    
    Parameters
    ----------
    embedding_option : str or list, optional
        The embedding option type: "visual", "audio", "transcription", or a list
    
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
    
    # Create different prompts based on embedding option
    if primary_option == "transcription":
        instruction = """
1. Briefly summarize in ONE concise sentence what happens in this time range.
   FOCUS: This segment is based on transcription/audio. Include ALL spoken words,
   dialogue, and any text that appears. Quote or paraphrase exactly what is said.
   If there are multiple speakers, identify who says what. Include any on-screen text.
"""
    elif primary_option == "audio":
        instruction = """
1. Briefly summarize in ONE concise sentence what happens in this time range.
   FOCUS: This segment is based on audio. Include ALL spoken words, dialogue,
   sounds, music, and audio cues. Quote or paraphrase what is said. Describe
   any important sounds or audio events. Include any on-screen text if visible.
"""
    else:  # visual (default)
        instruction = """
1. Briefly summarize in ONE concise sentence what happens in this time range.
   FOCUS: This segment is based on visual content. Describe what you see happening
   visually. If there is any spoken dialogue, audio, or on-screen text, include it
   in your summary as well.
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

    text_stream = tl_client.analyze_stream(
        video_id=video_id,
        prompt=prompt
    )

    collected = ""
    for chunk in text_stream:
        if chunk.event_type == "text_generation":
            collected += chunk.text

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

def generate_embedding(video_source, clip_length=6):
    """
    Generate embeddings for a video using TwelveLabs Embed API.
    The video will be split into segments of clip_length seconds.
    """
    is_url = video_source.startswith(('http://', 'https://'))
    
    task_kwargs = {
        'model_name': 'marengo3.0',
        'video_clip_length': clip_length
    }
    
    if is_url:
        task_kwargs['video_url'] = video_source
        print(f"Using video URL: {video_source}")
    else:
        video_file_path = os.path.abspath(video_source)
        if not os.path.exists(video_file_path):
            raise FileNotFoundError(f"Video file not found: {video_file_path}")
        task_kwargs['video_file'] = open(video_file_path, "rb")
        print(f"Using local video file: {video_file_path}")
    
    task = twelvelabs_client.embed.tasks.create(**task_kwargs)
    task_id = getattr(task, "id", getattr(task, "_id", None))
    print(f"Created embed task: id={task_id} status={getattr(task, 'status', 'pending')}")

    status_response = twelvelabs_client.embed.tasks.wait_for_done(
        task_id=task_id,
        sleep_interval=2,
        callback=on_task_update
    )
    print(f"Embedding done: {status_response.status}")

    task_result = twelvelabs_client.embed.tasks.retrieve(
        task_id=task_id,
        embedding_option=["visual", "audio", "transcription"]
    )

    embeddings = []
    if task_result.video_embedding and task_result.video_embedding.segments:
        for segment in task_result.video_embedding.segments:
            embeddings.append({
                'embedding': segment.float_,
                'start_offset_sec': segment.start_offset_sec,
                'end_offset_sec': segment.end_offset_sec,
                'embedding_scope': segment.embedding_scope,
                'embedding_option': segment.embedding_option
            })

    return embeddings, task_result


def ingest_data(video_source, index_name="twelve-labs", clip_length=6):
    """
    Generate embeddings for video, analyze each segment with Pegasus,
    and store everything in Pinecone (embedding + metadata).
    """
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

    # 4) Prepare vectors with rich metadata
    vectors_to_upsert = []
    for i, emb in enumerate(embeddings):
        start_sec = emb['start_offset_sec']
        end_sec = emb['end_offset_sec']

        # Call Pegasus to get summary + importance + talking_to_camera confidence for this segment
        summary, importance, talking_to_camera = analyze_segment_with_pegasus(
            twelvelabs_client,
            pegasus_video_id,
            start_sec,
            end_sec,
            embedding_option=emb['embedding_option'],
        )

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
        
        vector_id = f"{video_name}_{option_suffix}"
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
        }

        vectors_to_upsert.append(
            (vector_id, emb['embedding'], metadata)
        )

    # 5) Upsert into Pinecone
    index.upsert(vectors=vectors_to_upsert)
    print(f"Ingested {len(embeddings)} embeddings for {video_source}")

    return f"Ingested {len(embeddings)} embeddings for {video_source}"


if __name__ == "__main__":
    # Initialize clients
    print("Initializing TwelveLabs client...")
    twelvelabs_client = TwelveLabs(api_key=TL_API_KEY)
    
    print("Initializing Pinecone client...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Ingest the video
    print(f"\nIngesting video: {VIDEO_PATH}")
    print("Video will be split into 6-second intervals for embedding generation")
    result = ingest_data(
        VIDEO_PATH,
        index_name=INDEX_NAME,
        clip_length=6
    )
    print(f"\n{result}")
