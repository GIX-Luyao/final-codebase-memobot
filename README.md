# MemoBot - Memory Layer for Robots

Semantic memory storage and retrieval for humanoid robots.

## How It Works

```
┌────────┐                    ┌─────────────────┐                    ┌─────────────────┐
│ Robot  │ ──sendSegment()──▶ │  MemoBot API    │ ──store()───────▶  │ MemoBot Storage │
│        │ (5-sec MP4 chunk)  │                 │ (embedding,        │ (PostgreSQL +   │
│        │                    │                 │  metadata)         │  Twelve Labs)   │
└────────┘                    └─────────────────┘                    └─────────────────┘
    │                               │                                       │
    │ "Where did I put my keys?"    │                                       │
    │─────────────────────────────▶ │ ──search()─────────────────────────▶  │
    │                               │                                       │
    │                               │ ◀──memoryContext({clips,events})────  │
    │ ◀──────────────────────────── │                                       │
    │ "You put them on the desk     │                                       │
    │  at 3:42 PM"                  │                                       │
```

## Quick Start

```bash
docker-compose up -d
```

## SDK Usage

```python
from sdk import MemoBotClient

client = MemoBotClient("http://localhost:8000", "your-api-key")

# === Store memories ===
client.log_speech(robot_id="robot-1", text="I put my keys on the desk", speaker="user")

# === Retrieve memories ===
context = client.retrieve_memory(
    robot_id="robot-1",
    query="Where did I put my keys?"
)

print(context["context"]["clips"])    # Video clips matching query
print(context["context"]["events"])   # Text events matching query  
print(context["context"]["objects"])  # Detected objects

# === Get LLM answer ===
answer = client.ask(robot_id="robot-1", question="Where are my keys?")
print(answer["answer"])  # "You put your keys on the desk."
```

## Stream Video Segments

Robots must send **complete video segments** (MP4/WebM), not raw frames.
Use ffmpeg on the robot to create segments:

```bash
ffmpeg -i /dev/video0 -c:v libx264 -f segment -segment_time 5 chunk_%03d.mp4
```

```python
import asyncio
import glob

async def stream_video_segments():
    stream = client.create_stream_client("robot-1")
    await stream.connect(user_id="john")
    
    for chunk_path in sorted(glob.glob("chunk_*.mp4")):
        with open(chunk_path, "rb") as f:
            memory_id = await stream.send_segment(f.read())
            print(f"Stored: {memory_id}")

asyncio.run(stream_video_segments())
```

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/ws/stream/{robot_id}` | WS | Stream video/audio/actions |
| `/v1/memory/retrieve` | POST | Get clips, events, objects for query |
| `/v1/memory/answer` | POST | Get LLM-generated answer |
| `/v1/events` | POST | Store text event |
| `/v1/memory/profile` | GET | Get user/location profile |

## Configuration

```bash
# .env
DATABASE_URL=postgresql://user:pass@localhost:5432/memobot
REDIS_URL=redis://localhost:6379/0
Knowledge_Graph=bolt://localhost:7687
OPENAI_API_KEY=sk-...           # For text embeddings + LLM
TWELVE_LABS_API_KEY=tlk_...     # For video embeddings
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design.
