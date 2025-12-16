# MemoBot - Memory Layer for Robots

**Semantic memory storage and retrieval system for humanoid robots and AI agents.**

MemoBot provides a complete memory infrastructure that allows robots to:
- 🧠 Remember conversations, observations, and actions
- 🎥 **Stream and analyze real-time video** (Twelve Labs integration)
- 🔍 Search memories using natural language (text and video)
- 💡 Answer questions based on past experiences
- 👤 Build and maintain user profiles with preferences
- ⚡ Provide fast, contextual responses

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Video Streaming](#video-streaming)
- [SDK Usage](#sdk-usage)
- [Examples](#examples)
- [Deployment](#deployment)
- [Contributing](#contributing)

## Features

### Core Capabilities

- **Event Ingestion**: Capture speech, vision, actions, and system events
- **🆕 Real-Time Video Streaming**: WebSocket endpoint for robot camera feeds
- **🆕 Multimodal Video Search**: Semantic search across visual, audio, and text-in-video content
- **Vector Search**: Semantic search over all robot memories using embeddings
- **LLM-Powered Answers**: Get intelligent answers with supporting evidence
- **Profile Management**: Automatic profile building for users, locations, and objects
- **Session Summarization**: Background processing to group and summarize interactions
- **Flexible Storage**: PostgreSQL + pgvector for scalable vector search

### Technical Highlights

- FastAPI-based REST API with WebSocket support
- **Twelve Labs integration** for multimodal video embeddings
- Support for both OpenAI and local embeddings (sentence-transformers)
- Celery workers for background processing
- Docker-compose for easy deployment
- Python SDK with async video streaming client
- Full test coverage

## Architecture

```
┌──────────────────────────────────────────────────┐
│                Robot / SDK                        │
│            (humanoid client)                      │
└───────────┬──────────────────────┬───────────────┘
            │                      │
     HTTPS (REST)           WebSocket (Video)
            │                      │
            ▼                      ▼
┌───────────────────────────────────────────────────┐
│               API Gateway (FastAPI)               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │   Events    │ │   Memory    │ │   Video     │ │
│  │  Ingestion  │ │   Search    │ │  Streaming  │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ │
└───────────┬──────────────────────┬───────────────┘
            │                      │
            ▼                      ▼
┌────────────────────┐  ┌─────────────────────────┐
│   PostgreSQL       │  │     Twelve Labs         │
│   + pgvector       │  │   (Video Embeddings)    │
│                    │  │                         │
│ - events (384d)    │  │ - Multimodal search     │
│ - video_events     │  │ - Transcription         │
│ - profiles         │  │ - Scene understanding   │
└────────────────────┘  └─────────────────────────┘
            │
            ▼
┌────────────────────┐
│  Celery Workers    │
│  - Summarization   │
│  - Video Processing│
└────────────────────┘
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- OpenAI API key (or use local embeddings)

### 1. Clone and Setup

```bash
cd memobot
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 2. Start Services

```bash
docker-compose up -d
```

This starts:
- PostgreSQL with pgvector extension
- Redis for task queue
- FastAPI application server
- Celery worker for background tasks

### 3. Verify Installation

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "healthy"}
```

### 4. Use the SDK

```python
from sdk import MemoBotClient

# Initialize client
client = MemoBotClient(
    api_url="http://localhost:8000",
    api_key="your-api-key"
)

# Log an event
client.log_speech(
    robot_id="robot-123",
    text="I don't like loud noises.",
    speaker="user",
    user_id="user-456"
)

# Ask a question
answer = client.ask_memory(
    robot_id="robot-123",
    user_id="user-456",
    question="What are this user's preferences?"
)

print(answer['answer'])
print(f"Confidence: {answer['confidence']}")
```

## API Documentation

### Authentication

All API requests require an `Authorization` header:

```
Authorization: Bearer <API_KEY>
```

### Endpoints

#### POST /v1/events - Ingest Event

```json
{
  "robot_id": "robot-123",
  "user_id": "user-456",
  "source": "speech",
  "type": "USER_SAID",
  "text": "I don't like loud noises.",
  "metadata": {"location": "living_room"}
}
```

#### POST /v1/memory/search-events - Search Memory

```json
{
  "robot_id": "robot-123",
  "query": "What did this user say about noise?",
  "limit": 10
}
```

#### POST /v1/memory/answer - Get LLM Answer

```json
{
  "robot_id": "robot-123",
  "user_id": "user-456",
  "question": "What are this user's preferences?"
}
```

#### GET /v1/memory/profile - Get Profile

```
GET /v1/memory/profile?robot_id=robot-123&entity_type=user&entity_id=user-456
```

#### POST /v1/memory/video/search - Search Video Memory

```json
{
  "robot_id": "robot-123",
  "query": "person picking up a red cup",
  "search_options": ["visual", "audio"],
  "limit": 10
}
```

#### WebSocket /v1/ws/video/{robot_id} - Video Streaming

Real-time video streaming endpoint for robot cameras. See [Video Streaming](#video-streaming) section.

Full API documentation available at: http://localhost:8000/docs

## Video Streaming

MemoBot supports real-time video streaming from robot cameras with automatic multimodal analysis powered by Twelve Labs.

### How It Works

1. Robot connects via WebSocket to `/v1/ws/video/{robot_id}`
2. Robot sends binary video frames (JPEG, H264)
3. Server buffers frames into 5-second chunks
4. Chunks are processed asynchronously:
   - Uploaded to Twelve Labs
   - Multimodal embeddings generated (visual + audio)
   - Transcript extracted (speech-to-text)
   - Scene description generated
5. Results stored in PostgreSQL for semantic search

### WebSocket Protocol

```python
# 1. Connect
ws = websocket.connect("ws://localhost:8000/v1/ws/video/robot-123")

# 2. Authenticate
ws.send(json.dumps({"type": "auth", "api_key": "your-key"}))

# 3. Optional: Send metadata
ws.send(json.dumps({
    "type": "metadata",
    "metadata": {"user_id": "user-456", "frame_rate": 30}
}))

# 4. Stream frames (binary)
while True:
    frame = capture_camera_frame()
    ws.send(frame)  # Binary data
```

### Video Search Examples

```python
# Search for visual content
results = client.search_video_memory(
    robot_id="robot-123",
    query="person waving hello"
)

# Search with audio
results = client.search_video_memory(
    robot_id="robot-123",
    query="conversation about weather",
    search_options=["visual", "audio"]
)

# Search for text in video (signs, screens)
results = client.search_video_memory(
    robot_id="robot-123",
    query="whiteboard with diagrams",
    search_options=["text_in_video"]
)
```

### Configuration

```bash
# Twelve Labs API (required for video)
TWELVE_LABS_API_KEY=tlk_...
TWELVE_LABS_INDEX_NAME=memobot-videos

# Video processing
VIDEO_CHUNK_DURATION_SECONDS=5
ENABLE_VIDEO_PROCESSING=true
```

## SDK Usage

### Installation

```bash
# From the memobot directory
pip install -e .
```

### Basic Usage

```python
from sdk import MemoBotClient

client = MemoBotClient(
    api_url="http://localhost:8000",
    api_key="your-api-key"
)

# Log different types of events
client.log_speech(robot_id="robot-1", text="Hello", speaker="robot")
client.log_vision(robot_id="robot-1", description="Saw person", objects=["person"])
client.log_action(robot_id="robot-1", action="MOVED", description="Moved forward")

# Search memory
results = client.search_memory(
    robot_id="robot-1",
    query="interactions with users"
)

# Ask questions
answer = client.ask_memory(
    robot_id="robot-1",
    question="What did I do today?"
)

# Get profiles
profile = client.get_profile(
    robot_id="robot-1",
    entity_type="user",
    entity_id="user-123"
)
```

### Video Streaming

```python
import asyncio
from sdk import MemoBotClient

client = MemoBotClient(
    api_url="http://localhost:8000",
    api_key="your-api-key"
)

# Create video streaming client
video_client = client.create_video_stream_client(robot_id="robot-123")

async def stream_camera():
    """Stream camera frames to MemoBot."""
    import cv2
    
    async def frame_generator():
        cap = cv2.VideoCapture(0)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                _, buffer = cv2.imencode('.jpg', frame)
                yield buffer.tobytes()
                await asyncio.sleep(1/30)  # 30 FPS
        finally:
            cap.release()
    
    def on_chunk_created(msg):
        print(f"Video chunk saved: {msg['video_event_id']}")
    
    await video_client.stream_video(
        frame_generator(),
        on_chunk_created=on_chunk_created
    )

# Run the stream
asyncio.run(stream_camera())

# Search video memories
results = client.search_video_memory(
    robot_id="robot-123",
    query="person holding a cup"
)
```

## Examples

### Basic Usage

See [examples/basic_usage.py](examples/basic_usage.py) for a complete example showing:
- Event logging (speech, vision, actions)
- Memory search
- Question answering
- Profile retrieval

Run it:
```bash
python examples/basic_usage.py
```

### ROS Integration

See [examples/ros_integration.py](examples/ros_integration.py) for integrating with ROS-based robots.

## Deployment

### Development

```bash
docker-compose up
```

### Production

1. **Update environment variables**:
   - Set strong `API_SECRET_KEY`
   - Configure production database URL
   - Set appropriate CORS origins

2. **Scale workers**:
   ```bash
   docker-compose up -d --scale worker=3
   ```

3. **Use a reverse proxy** (nginx/traefik) for SSL termination

4. **Enable monitoring** with Prometheus/Grafana

### Configuration

Key environment variables:

```bash
# Database
DATABASE_URL=postgresql://user:pass@host:5432/memobot

# OpenAI (for text embeddings)
OPENAI_API_KEY=sk-...
USE_LOCAL_EMBEDDINGS=false

# Twelve Labs (for video embeddings)
TWELVE_LABS_API_KEY=tlk_...
TWELVE_LABS_INDEX_NAME=memobot-videos

# Redis
REDIS_URL=redis://localhost:6379/0

# Features
ENABLE_SUMMARIZATION=true
ENABLE_PROFILES=true
ENABLE_VIDEO_PROCESSING=true

# Video Processing
VIDEO_CHUNK_DURATION_SECONDS=5
VIDEO_TEMP_STORAGE_PATH=/tmp/memobot/videos
```

## Development

### Running Tests

```bash
pytest
```

### Local Development Without Docker

```bash
# Install dependencies
pip install -r requirements.txt

# Start PostgreSQL and Redis separately
# Then start the API
uvicorn backend.api.main:app --reload

# Start worker
celery -A backend.workers.celery_app worker --loglevel=info
```

### Database Migrations

```bash
# Create migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head
```

## System Requirements

### Minimum
- 2 CPU cores
- 4GB RAM
- 10GB storage

### Recommended
- 4+ CPU cores
- 8GB+ RAM
- 50GB+ storage (for large event history)

## Performance

### Text Events
- **Ingestion**: ~1000 events/second
- **Search latency**: <100ms for most queries
- **Embedding generation**: ~50ms per event (local) or ~200ms (OpenAI)
- **LLM answers**: 1-3 seconds depending on context size

### Video Events
- **Streaming**: Real-time via WebSocket (30+ FPS)
- **Processing**: 30-60 seconds per 5-second chunk (Twelve Labs)
- **Video search**: ~500ms for semantic query
- **Embedding dimension**: 1024 (multimodal)

## Security

- API key authentication required for all endpoints
- SQL injection protection via SQLAlchemy
- CORS configuration for web clients
- Rate limiting (configurable)
- Input validation via Pydantic

## Roadmap

### Completed ✅
- [x] Multi-modal embeddings (video) via Twelve Labs
- [x] Real-time video streaming (WebSocket)
- [x] Video semantic search (visual, audio, text-in-video)

### In Progress
- [ ] Knowledge graph integration (Neo4j) for entity/relations
- [ ] Streaming responses for long answers

### Planned
- [ ] GraphQL API
- [ ] Video thumbnail generation
- [ ] Face recognition integration
- [ ] More sophisticated session detection
- [ ] Multi-robot memory sharing
- [ ] Edge deployment (on-robot inference)
- [ ] Federation for distributed deployments

## License

MIT License - see LICENSE file

## Support

- Documentation: [ARCHITECTURE.md](ARCHITECTURE.md)
- Issues: GitHub Issues
- Discussions: GitHub Discussions

## Acknowledgments

- Built with FastAPI, SQLAlchemy, pgvector
- Text embeddings via OpenAI or sentence-transformers
- Video embeddings via [Twelve Labs](https://twelvelabs.io/) Marengo engine
- Inspired by MemGPT, LangChain, and various robotics memory systems

---

**Made with ❤️ for the robotics community**

