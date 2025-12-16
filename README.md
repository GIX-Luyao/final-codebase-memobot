# MemoBot - Memory Layer for Robots

Semantic memory storage and retrieval for humanoid robots.

**Features:**
- Store speech, vision, and action events with embeddings
- Natural language memory search
- Real-time video streaming with Twelve Labs multimodal search
- Automatic user/location/object profiles
- LLM-powered question answering

## Quick Start

```bash
# Start services
docker-compose up -d

# Verify
curl http://localhost:8000/health
```

## SDK Usage

```python
from sdk import MemoBotClient

client = MemoBotClient("http://localhost:8000", "your-api-key")

# Log events
client.log_speech(robot_id="robot-1", text="I like quiet spaces", speaker="user", user_id="user-1")

# Search memory
results = client.search_memory(robot_id="robot-1", query="user preferences")

# Ask questions
answer = client.ask_memory(robot_id="robot-1", question="What does this user prefer?")

# Search video
results = client.search_video_memory(robot_id="robot-1", query="person waving")
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/events` | POST | Log event |
| `/v1/memory/search-events` | POST | Semantic search |
| `/v1/memory/answer` | POST | LLM-generated answer |
| `/v1/memory/profile` | GET | Get entity profile |
| `/v1/memory/video/upload` | POST | Upload video |
| `/v1/memory/video/search` | POST | Search video memories |
| `/v1/ws/video/{robot_id}` | WS | Stream video chunks |

Full docs: http://localhost:8000/docs

## Video Streaming

Robots send 5-second MP4 chunks via WebSocket. Each chunk is processed through Twelve Labs for multimodal search.

```python
# WebSocket protocol
ws = websocket.connect("ws://localhost:8000/v1/ws/video/robot-123")
ws.send(json.dumps({"type": "auth", "api_key": "your-key"}))
ws.send(mp4_chunk_bytes)  # Binary video data
```

## Configuration

```bash
# .env file
DATABASE_URL=postgresql://user:pass@localhost:5432/memobot
REDIS_URL=redis://localhost:6379/0
OPENAI_API_KEY=sk-...
TWELVE_LABS_API_KEY=tlk_...
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design.

```
Robot → REST/WebSocket → FastAPI → Celery Workers
                              ↓
              PostgreSQL (text) + Twelve Labs (video)
```

## License

MIT
