# MemoBot Architecture Documentation

## Overview

MemoBot is a semantic memory layer designed for humanoid robots and AI agents. It provides persistent, searchable memory with intelligent retrieval and summarization capabilities.

## System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         Robot / Client                          │
│                      (uses SDK or REST API)                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ HTTPS + Bearer Token
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                        API Gateway                              │
│                     (FastAPI Application)                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  │  Event Ingestion │  │  Memory Queries  │  │  Profiles    │ │
│  │   /v1/events     │  │  /v1/memory/*    │  │              │ │
│  └──────────────────┘  └──────────────────┘  └──────────────┘ │
└────────────────┬────────────────┬────────────────┬─────────────┘
                 │                │                │
                 ▼                ▼                ▼
┌────────────────────────────────────────────────────────────────┐
│                     Services Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │  Embedding   │  │ Vector Store │  │    LLM Service       │ │
│  │  Service     │  │   Service    │  │  (Summarization)     │ │
│  └──────────────┘  └──────────────┘  └──────────────────────┘ │
└────────────────┬────────────────┬────────────────┬─────────────┘
                 │                │                │
                 ▼                ▼                ▼
┌────────────────────────────────────────────────────────────────┐
│                     Storage Layer                              │
│  ┌──────────────────┐           ┌─────────────────────────┐   │
│  │   PostgreSQL     │           │      Redis              │   │
│  │  + pgvector      │           │  (Task Queue/Cache)     │   │
│  │                  │           └─────────────────────────┘   │
│  │  - events        │                                          │
│  │  - sessions      │                                          │
│  │  - profiles      │                                          │
│  │  - embeddings    │                                          │
│  └──────────────────┘                                          │
└────────────────────────────────────────────────────────────────┘
                 ▲
                 │ Periodic Tasks
                 │
┌────────────────┴────────────────────────────────────────────────┐
│                   Background Workers                            │
│                     (Celery Workers)                            │
│  ┌──────────────────┐           ┌─────────────────────────┐    │
│  │  Session         │           │  Profile                │    │
│  │  Summarization   │           │  Updates                │    │
│  └──────────────────┘           └─────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Data Model

### Core Tables

#### 1. Events Table

The central log of everything the robot observes and does.

```sql
events (
  event_id      UUID PRIMARY KEY,
  robot_id      TEXT NOT NULL,
  user_id       TEXT NULL,
  timestamp     TIMESTAMPTZ NOT NULL,
  source        TEXT NOT NULL,     -- 'speech', 'vision', 'system', 'action'
  type          TEXT NOT NULL,     -- 'USER_SAID', 'ROBOT_SAID', etc.
  text          TEXT NULL,
  metadata      JSONB,
  session_id    UUID NULL,
  embedding     VECTOR(384),       -- pgvector type
  created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_robot_user_timestamp ON events(robot_id, user_id, timestamp);
CREATE INDEX idx_robot_timestamp ON events(robot_id, timestamp);
CREATE INDEX idx_session_timestamp ON events(session_id, timestamp);
```

**Fields:**
- `event_id`: Unique identifier
- `robot_id`: Which robot created this event
- `user_id`: Associated user (if applicable)
- `timestamp`: When the event occurred
- `source`: Category of event source
- `type`: Specific event type
- `text`: Textual content (for embedding)
- `metadata`: Flexible JSON for extra data (location, objects, etc.)
- `session_id`: Groups related events
- `embedding`: Vector representation for semantic search

#### 2. Sessions Table

Groups of related events (conversations, interactions).

```sql
sessions (
  session_id    UUID PRIMARY KEY,
  robot_id      TEXT NOT NULL,
  user_id       TEXT NULL,
  start_time    TIMESTAMPTZ NOT NULL,
  end_time      TIMESTAMPTZ NOT NULL,
  summary       TEXT,
  metadata      JSONB,
  created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_robot_user_time ON sessions(robot_id, user_id, start_time);
```

**Purpose:**
- Group events that belong together
- Provide summaries of interactions
- Enable fast retrieval of conversation history

#### 3. Profiles Table

Persistent knowledge about entities (users, locations, objects).

```sql
profiles (
  profile_id    UUID PRIMARY KEY,
  robot_id      TEXT NOT NULL,
  entity_type   TEXT NOT NULL,    -- 'user', 'location', 'object'
  entity_id     TEXT NOT NULL,
  summary       TEXT,
  facts         JSONB,             -- [{subject, predicate, object, confidence}]
  last_updated  TIMESTAMPTZ NOT NULL,
  created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE UNIQUE INDEX idx_robot_entity ON profiles(robot_id, entity_type, entity_id);
```

**Purpose:**
- Cache stable knowledge about entities
- Fast lookup without searching all events
- Confidence-weighted facts

#### 4. Video Events Table

Stores video clips with multimodal embeddings from Twelve Labs.

```sql
video_events (
  video_event_id       UUID PRIMARY KEY,
  robot_id             TEXT NOT NULL,
  user_id              TEXT NULL,
  session_id           UUID NULL,
  start_timestamp      TIMESTAMPTZ NOT NULL,
  end_timestamp        TIMESTAMPTZ NOT NULL,
  duration_seconds     FLOAT NOT NULL,
  video_file_path      TEXT,              -- Local or cloud storage path
  twelve_labs_task_id  TEXT,              -- Twelve Labs upload task ID
  twelve_labs_video_id TEXT,              -- Video ID in Twelve Labs index
  video_embedding      VECTOR(1024),      -- Marengo-retrieval-2.6 embedding
  transcript           TEXT,              -- Speech-to-text from video
  scene_description    TEXT,              -- AI-generated scene description
  detected_objects     JSONB,             -- List of detected objects
  detected_actions     JSONB,             -- List of detected actions
  detected_text        JSONB,             -- OCR text from video frames
  metadata             JSONB,
  processing_status    TEXT DEFAULT 'pending',  -- pending, uploading, processing, completed, failed
  processing_error     TEXT,
  created_at           TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_video_robot_timestamp ON video_events(robot_id, start_timestamp);
CREATE INDEX idx_video_processing_status ON video_events(processing_status);
CREATE INDEX idx_video_twelve_labs_id ON video_events(twelve_labs_video_id);
```

**Purpose:**
- Store video clips captured by the robot
- Multimodal embeddings for semantic video search (visual, audio, text)
- Extract and store transcripts, detected objects, and scene descriptions
- Enable queries like "When did I see a red cup?" or "Find conversations about weather"

#### 5. Graph Database Schema (Future - Neo4j)

Planned schema for entity relationships extracted from video content:

```cypher
// Node Types
(:Person {id, name, robot_id, first_seen, last_seen, face_embedding})
(:Object {id, label, robot_id, first_seen, last_seen})
(:Location {id, name, robot_id, coordinates})
(:Action {id, type, timestamp})
(:VideoEvent {id, robot_id, timestamp, twelve_labs_video_id})

// Relationships
(:Person)-[:APPEARED_IN {confidence, timestamp, bounding_box}]->(:VideoEvent)
(:Object)-[:DETECTED_IN {confidence, bounding_box}]->(:VideoEvent)
(:Person)-[:INTERACTED_WITH {action_type, timestamp}]->(:Object)
(:Person)-[:WAS_AT {timestamp}]->(:Location)
(:Action)-[:PERFORMED_BY]->(:Person)
(:Action)-[:INVOLVED]->(:Object)
```

**Purpose:**
- Track entity appearances across video events
- Model relationships between people, objects, and locations
- Enable graph queries like "Who has interacted with this object?"
- Support entity timeline queries

## API Layer

### Authentication

All endpoints require Bearer token authentication:

```
Authorization: Bearer <API_KEY>
```

In production, implement proper API key management:
- Store hashed keys in database
- Rate limiting per key
- Key rotation
- Scope-based permissions

### Endpoint Design

#### Ingestion Endpoints

**POST /v1/events**
- Accepts single event
- Validates schema
- Stores in database
- Generates embedding (async)
- Returns event_id

**POST /v1/events/batch**
- Accepts multiple events
- Optimized for bulk ingestion
- Returns results array

#### Query Endpoints

**POST /v1/memory/search-events**
- RAG primitive
- Semantic similarity search
- Supports filters (time, source, type)
- Returns ranked events

**POST /v1/memory/answer**
- High-level query interface
- Retrieves relevant events
- Feeds to LLM
- Returns structured answer + evidence

**GET /v1/memory/profile**
- Fast profile lookup
- Creates on-demand if missing
- Returns summary + facts

#### Video Endpoints

**WebSocket /v1/ws/video/{robot_id}**
- Real-time video streaming from robots
- Binary frame transmission (JPEG, H264)
- Automatic chunking (5-second segments)
- Background processing via Twelve Labs
- Protocol:
  1. Connect with robot_id
  2. Send auth: `{"type": "auth", "api_key": "xxx"}`
  3. Stream binary frames
  4. Receive chunk acknowledgments

**POST /v1/memory/video/upload**
- Upload pre-recorded video files
- Supports MP4, MOV, AVI, MKV, WEBM
- Returns video_event_id
- Background processing queued

**POST /v1/memory/video/from-url**
- Create video event from URL
- Direct Twelve Labs ingestion
- Useful for cloud-stored videos

**POST /v1/memory/video/search**
- Semantic video search
- Natural language queries
- Search across visual, audio, text-in-video
- Returns ranked results with timestamps

**GET /v1/memory/video/{video_event_id}**
- Get video event details
- Includes transcript, detected objects
- Processing status

**GET /v1/memory/video/robot/{robot_id}/recent**
- List recent video events
- Filter by processing status

**GET /v1/memory/video/stats/{robot_id}**
- Video processing statistics
- Counts by status, total duration

## Services Layer

### Embedding Service

**Responsibilities:**
- Convert text to vector embeddings
- Support multiple backends (OpenAI, local models)
- Batch processing for efficiency

**Implementation:**
```python
class EmbeddingService:
    def embed(text: str) -> List[float]
    def embed_batch(texts: List[str]) -> List[List[float]]
```

**Backends:**
- **OpenAI**: `text-embedding-3-small` (384 dimensions)
- **Local**: `sentence-transformers/all-MiniLM-L6-v2`

**Performance:**
- OpenAI: ~200ms per request
- Local: ~50ms per text (GPU), ~200ms (CPU)

### Vector Store Service

**Responsibilities:**
- Store embeddings with metadata
- Perform similarity search
- Handle filtering

**Key Operations:**
```python
class VectorStoreService:
    def add_event_embedding(event_id, text) -> bool
    def search_similar_events(query, filters) -> List[Event]
    def get_recent_events(robot_id, limit) -> List[Event]
```

**Search Algorithm:**
- Uses pgvector's cosine distance operator
- Combines vector similarity with SQL filters
- Indexes for fast filtering before similarity computation

### LLM Service

**Responsibilities:**
- Generate answers from events
- Summarize sessions
- Extract facts

**Key Operations:**
```python
class LLMService:
    def generate_answer(question, events) -> Dict
    def summarize_session(events) -> str
    def extract_facts(events, entity_id) -> List[Fact]
```

**Prompting Strategy:**
- System prompt defines role
- Context: Top-K events formatted clearly
- Temperature: 0.3 for factual responses
- Max tokens: 200-500 depending on task

### Video Embedding Service (Twelve Labs)

**Responsibilities:**
- Upload video content to Twelve Labs index
- Generate multimodal embeddings (visual + audio)
- Extract transcripts and scene descriptions
- Semantic video search

**Implementation:**
```python
class TwelveLabsService:
    async def upload_video(video_path, metadata) -> str  # Returns task_id
    async def wait_for_task(task_id) -> Dict
    async def get_video_embedding(video_id) -> List[float]  # 1024 dimensions
    async def search_by_text(query, options) -> List[Dict]
    async def get_video_transcription(video_id) -> str
    async def generate_gist(video_id) -> Dict  # Topics, title
```

**Twelve Labs Capabilities:**
- **Marengo-retrieval-2.6**: Multimodal embedding engine (1024 dimensions)
- **Visual Search**: Find moments by visual content description
- **Audio Search**: Find moments by speech/sound
- **Text-in-Video**: OCR for text appearing in frames
- **Transcription**: Speech-to-text with timestamps

**Processing Pipeline:**
1. Robot streams frames via WebSocket
2. Server buffers into 5-second chunks
3. Chunks saved to temporary storage
4. Celery task uploads to Twelve Labs
5. Poll for indexing completion
6. Extract embedding, transcript, gist
7. Store results in PostgreSQL

**Performance:**
- Upload + Indexing: 30-60 seconds per 5-second clip
- Search: ~500ms for semantic query
- Embedding dimension: 1024 (vs 384 for text)

## Background Workers

### Session Summarization Task

**Frequency:** Hourly

**Algorithm:**
1. Find events without session_id (recent 7 days)
2. Group by (robot_id, user_id, time proximity)
3. Time gap threshold: 30 minutes
4. For each group:
   - Create session record
   - Generate LLM summary
   - Update events with session_id

**Why:**
- Reduces redundancy in searches
- Provides high-level view
- Enables conversation-level queries

### Profile Update Task

**Frequency:** Daily

**Algorithm:**
1. Find entities with recent activity (24 hours)
2. For each entity:
   - Retrieve recent events (limit 50)
   - Generate summary
   - Extract facts
   - Update or create profile

**Why:**
- Keeps profiles fresh
- Amortizes LLM cost
- Fast profile lookups

### Video Processing Task

**Trigger:** On video chunk creation (via WebSocket or upload)

**Algorithm:**
1. Receive video_event_id from queue
2. Upload video file to Twelve Labs
3. Poll task status until "ready"
4. Retrieve multimodal embedding
5. Extract transcript (speech-to-text)
6. Get text-in-video (OCR)
7. Generate scene gist (topics, title)
8. Update video_event with all extracted data

**Retry Logic:**
- Max 3 retries with exponential backoff
- On failure: mark status as "failed", store error

**Why:**
- Offloads expensive API calls from request path
- Handles rate limiting gracefully
- Provides async processing feedback

### Reprocess Failed Videos Task

**Frequency:** Hourly

**Algorithm:**
1. Find video_events with status="failed" (last 24 hours)
2. Reset status to "pending"
3. Requeue for processing

**Why:**
- Recovers from transient failures
- Handles API timeouts and rate limits
- Ensures data completeness

## Scaling Considerations

### Horizontal Scaling

**API Layer:**
- Stateless FastAPI instances
- Load balance with nginx/traefik
- Auto-scale based on request rate

**Workers:**
- Multiple Celery workers
- Task routing by type
- Priority queues

**Database:**
- Read replicas for queries
- Connection pooling
- Partition events table by time

### Vertical Scaling

**Memory:**
- Event volume: ~1KB per event
- 1M events ≈ 1GB (events + embeddings)
- Profiles: ~10KB each

**CPU:**
- Embedding generation (if local)
- Vector similarity computation
- LLM inference (if local)

### Caching Strategy

**Redis Caching:**
- Profile cache (TTL: 1 hour)
- Recent events cache (TTL: 5 minutes)
- Search result cache (TTL: 1 minute)

## Security

### API Security

1. **Authentication**: Bearer tokens
2. **Rate Limiting**: Per-key limits
3. **Input Validation**: Pydantic schemas
4. **SQL Injection**: Parameterized queries (SQLAlchemy)

### Data Privacy

1. **User Data**: Store only necessary fields
2. **Encryption**: At-rest (database level)
3. **Retention**: Configurable data retention policies
4. **GDPR**: Support for data export/deletion

## Monitoring

### Key Metrics

**API Metrics:**
- Request rate by endpoint
- Response time (p50, p95, p99)
- Error rate
- Authentication failures

**Storage Metrics:**
- Event count
- Database size
- Vector index size
- Query latency

**Worker Metrics:**
- Task queue length
- Task processing time
- Success/failure rate
- Worker health

### Logging

**Structured Logging:**
```python
{
  "timestamp": "2025-11-22T10:00:00Z",
  "level": "INFO",
  "service": "api",
  "robot_id": "robot-123",
  "endpoint": "/v1/events",
  "duration_ms": 45
}
```

## Testing Strategy

### Unit Tests
- Services (embedding, vector store, LLM)
- API endpoints
- Data models

### Integration Tests
- End-to-end API flows
- Database operations
- Worker tasks

### Performance Tests
- Load testing (k6, locust)
- Embedding generation throughput
- Search latency under load

## Future Enhancements

### Completed ✅
- [x] **Multi-modal embeddings (video)** - Twelve Labs integration for video understanding
- [x] **Real-time video streaming** - WebSocket endpoint for robot camera feeds
- [x] **Video semantic search** - Natural language queries across video content

### Short Term
- [ ] Webhook support for real-time notifications
- [ ] Streaming responses for long answers
- [ ] Multi-tenant support
- [ ] Advanced analytics dashboard
- [ ] Video thumbnail generation
- [ ] Video clip extraction API

### Medium Term
- [ ] **Knowledge graph integration (Neo4j)** - Entity/relation extraction from video
- [ ] Image frame embedding (individual frames)
- [ ] Federated learning across robots
- [ ] Compression for old events
- [ ] Video retention policies
- [ ] Face recognition integration

### Long Term
- [ ] Edge deployment (on-robot inference)
- [ ] Multi-robot memory sharing
- [ ] Causal reasoning from video sequences
- [ ] Continuous learning from feedback
- [ ] Real-time object tracking

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/memobot

# OpenAI (for text embeddings)
OPENAI_API_KEY=sk-...

# Twelve Labs (for video embeddings)
TWELVE_LABS_API_KEY=tlk_...
TWELVE_LABS_INDEX_ID=           # Optional, auto-created if not set
TWELVE_LABS_INDEX_NAME=memobot-videos

# Video Processing
VIDEO_CHUNK_DURATION_SECONDS=5
VIDEO_TEMP_STORAGE_PATH=/tmp/memobot/videos
ENABLE_VIDEO_PROCESSING=true

# Graph Database (future)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=
ENABLE_GRAPH_DB=false

# Redis
REDIS_URL=redis://localhost:6379/0
```

## References

- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [Celery Documentation](https://docs.celeryq.dev/)
- [Twelve Labs Documentation](https://docs.twelvelabs.io/)
- [Neo4j Documentation](https://neo4j.com/docs/)

