# MemoBot Architecture

## User Journey

```
┌────────┐          ┌──────────────┐          ┌─────────────────┐          ┌─────────────────┐
│  User  │          │ Robot(Agent) │          │ MemoBot (API)   │          │ MemoBot Storage │
└───┬────┘          └──────┬───────┘          └────────┬────────┘          └────────┬────────┘
    │                      │                           │                            │
    │   ╔══════════════════════════════════════════════════════════════════════╗   │
    │   ║         Continuous Multimodal Memory Ingestion                       ║   │
    │   ╚══════════════════════════════════════════════════════════════════════╝   │
    │                      │                           │                            │
    │                      │  ┌─────────────────────┐  │                            │
    │                      │  │ loop [Realtime]     │  │                            │
    │                      │  │                     │  │                            │
    │                      │──┼─sendFrame({video,───┼─▶│                            │
    │                      │  │  audio, action, ts})│  │                            │
    │                      │  │                     │  │  store(embedding, metadata)│
    │                      │  │                     │  │───────────────────────────▶│
    │                      │  │                     │  │                            │
    │                      │  │                     │  │◀──────ack(memoryId)────────│
    │                      │  │                     │  │                            │
    │                      │◀─┼──ackStored(memoryId)┼──│                            │
    │                      │  │                     │  │                            │
    │                      │  └─────────────────────┘  │                            │
    │                      │                           │                            │
    │   ╔══════════════════════════════════════════════════════════════════════╗   │
    │   ║              User Asks About Past Event                              ║   │
    │   ╚══════════════════════════════════════════════════════════════════════╝   │
    │                      │                           │                            │
    │  "Where did I put    │                           │                            │
    │   my keys?"          │                           │                            │
    │─────────────────────▶│                           │                            │
    │                      │                           │                            │
    │                      │  retrieveMemory({query,   │                            │
    │                      │   optionalContext})       │                            │
    │                      │──────────────────────────▶│                            │
    │                      │                           │                            │
    │                      │                           │  search({query, embedding})│
    │                      │                           │───────────────────────────▶│
    │                      │                           │                            │
    │                      │                           │◀─memoryContext({clips,─────│
    │                      │                           │   events, objects})        │
    │                      │                           │                            │
    │                      │◀────memoryContext─────────│                            │
    │                      │                           │                            │
    │  "You put your keys  │                           │                            │
    │   on the desk at     │                           │                            │
    │   3:42 PM."          │                           │                            │
    │◀─────────────────────│                           │                            │
    │                      │                           │                            │
```

## System Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            Robot / Agent                                │
│                                                                         │
│   Camera ──┐                                                            │
│   Mic    ──┼──▶ sendFrame({video, audio, action, ts}) ──┐               │
│   Actions ─┘                                            │               │
│                                                         │ WebSocket     │
│   User Query ──▶ retrieveMemory({query}) ───────────────┼───┐           │
│                                                         │   │ REST      │
└─────────────────────────────────────────────────────────┼───┼───────────┘
                                                          │   │
                                                          ▼   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         MemoBot API (FastAPI)                           │
│                                                                         │
│   WebSocket Handler ─────▶ Extract Embeddings ─────▶ Store              │
│   /v1/ws/stream/{robot}      (Twelve Labs)            (PostgreSQL)      │
│                                                                         │
│   REST Endpoints ────────▶ Search ──────────────────▶ Return Context    │
│   /v1/memory/retrieve        (Vector + Twelve Labs)                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          MemoBot Storage                                │
│                                                                         │
│   ┌─────────────────────────┐    ┌─────────────────────────────────┐   │
│   │  PostgreSQL + pgvector  │    │  Twelve Labs (Video Index)      │   │
│   │                         │    │                                 │   │
│   │  • events (text embed)  │    │  • Multimodal embeddings        │   │
│   │  • memories (metadata)  │    │  • Visual + Audio search        │   │
│   │  • profiles             │    │  • Transcription                │   │
│   └─────────────────────────┘    └─────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Memory Ingestion (Continuous)

```
Robot sends complete video segments (~5 seconds each):
- Binary: raw MP4/WebM bytes
- OR JSON: {"type": "segment", "video": "<base64 MP4>", "action": "MOVING"}

The robot should use ffmpeg to create segments:
  ffmpeg -i /dev/video0 -c:v libx264 -f segment -segment_time 5 chunk_%03d.mp4

MemoBot:
1. Receives complete video segment
2. Uploads to Twelve Labs → gets multimodal embedding
3. Stores in PostgreSQL: {memory_id, robot_id, timestamp, twelve_labs_id, metadata}
4. Acks to robot: {"type": "ack_stored", "memory_id": "..."}
```

### 2. Memory Retrieval

```
Robot sends query:
{
  "query": "Where did I put my keys?",
  "context": {                    // Optional context
    "user_id": "john",
    "location": "living_room",
    "time_range": "today"
  }
}

MemoBot:
1. Searches Twelve Labs index (visual + audio)
2. Searches PostgreSQL events (text)
3. Combines and ranks results
4. Returns rich context:
{
  "clips": [
    {
      "memory_id": "abc123",
      "timestamp": "2024-01-15T15:42:00Z",
      "description": "User placed keys on wooden desk",
      "confidence": 0.92
    }
  ],
  "events": [
    {
      "timestamp": "2024-01-15T15:42:05Z",
      "type": "USER_SAID",
      "text": "I'll just leave my keys here"
    }
  ],
  "objects": ["keys", "desk", "living_room"]
}
```

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/ws/stream/{robot_id}` | WebSocket | Continuous multimodal ingestion |
| `/v1/memory/retrieve` | POST | Retrieve memories for a query |
| `/v1/memory/store` | POST | Store single event (text/action) |
| `/v1/memory/profile/{entity}` | GET | Get entity profile |

## Data Models

### Memory (stored in PostgreSQL)

```sql
memories (
  memory_id UUID PRIMARY KEY,
  robot_id VARCHAR NOT NULL,
  user_id VARCHAR,
  timestamp TIMESTAMPTZ NOT NULL,
  
  -- Content references
  twelve_labs_video_id VARCHAR,     -- Video in Twelve Labs
  text_content TEXT,                -- Transcribed/extracted text
  
  -- Extracted metadata
  objects JSONB,                    -- ["keys", "desk", "person"]
  actions JSONB,                    -- ["placed", "picked_up"]
  location VARCHAR,
  
  -- Embeddings for text search
  text_embedding VECTOR(384),
  
  -- Processing
  status VARCHAR DEFAULT 'processing'
)
```

### MemoryContext (returned from retrieval)

```python
class MemoryContext:
    clips: List[Clip]       # Relevant video moments
    events: List[Event]     # Related text events  
    objects: List[str]      # Detected objects
    summary: str            # LLM-generated summary (optional)
```

## Example Robot Integration

```python
import asyncio
import subprocess
from sdk import MemoBotClient

client = MemoBotClient("http://memobot:8000", "api-key")

# === Continuous Video Ingestion ===
async def stream_video():
    """
    Stream video segments to MemoBot.
    Uses ffmpeg to create 5-second MP4 segments from camera.
    """
    stream = client.create_stream_client("robot-123")
    await stream.connect(user_id="john")
    
    # Start ffmpeg to segment video
    # ffmpeg -i /dev/video0 -c:v libx264 -f segment -segment_time 5 /tmp/chunk_%03d.mp4
    
    segment_num = 0
    while True:
        segment_path = f"/tmp/chunk_{segment_num:03d}.mp4"
        
        # Wait for segment to be created by ffmpeg
        while not os.path.exists(segment_path):
            await asyncio.sleep(0.5)
        
        # Send to MemoBot
        with open(segment_path, "rb") as f:
            memory_id = await stream.send_segment(f.read())
            print(f"Stored: {memory_id}")
        
        os.remove(segment_path)
        segment_num += 1

# === Memory Retrieval ===
def answer_question(user_query: str) -> str:
    context = client.retrieve_memory(
        robot_id="robot-123",
        query=user_query,
        user_id="john"
    )
    
    # context["context"]["clips"] = video moments
    # context["context"]["events"] = text events
    # context["context"]["objects"] = detected objects
    
    # Use LLM to generate response
    answer = client.ask(
        robot_id="robot-123",
        question=user_query,
        user_id="john"
    )
    return answer["answer"]

# === Main Loop ===
async def main():
    # Start video streaming in background
    asyncio.create_task(stream_video())
    
    # Handle user queries
    while True:
        user_input = await get_user_speech()
        if "remember" in user_input or "where" in user_input:
            answer = answer_question(user_input)
            robot.speak(answer)
```

## Technology Stack

- **API**: FastAPI (REST + WebSocket)
- **Storage**: PostgreSQL + pgvector
- **Video AI**: Twelve Labs (multimodal embeddings)
- **Queue**: Redis + Celery
- **Embeddings**: OpenAI / sentence-transformers
