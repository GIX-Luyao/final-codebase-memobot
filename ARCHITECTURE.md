# MemoBot Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                         Robot/Client                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Speech/Text  │  │ Actions/      │  │ Video Camera         │   │
│  │ Events       │  │ Sensors       │  │ Stream (5s chunks)   │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘   │
└─────────┼─────────────────┼─────────────────────┼───────────────┘
          │ REST            │ REST                │ WebSocket
          ▼                 ▼                     ▼
┌──────────────────────────────────────────────────────────────────┐
│                       MemoBot API (FastAPI)                      │
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────────────┐│
│  │ /v1/events     │  │ /v1/memory     │  │ /v1/ws/video/{id}   ││
│  └───────┬────────┘  └───────┬────────┘  └──────────┬──────────┘│
└──────────┼───────────────────┼──────────────────────┼───────────┘
           │                   │                      │
           ▼                   ▼                      ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Celery Workers                              │
│  ┌──────────────────────┐  ┌────────────────────────────────┐   │
│  │ Summarization        │  │ Video Processing               │   │
│  │ Profile Updates      │  │ → Upload to Twelve Labs        │   │
│  │                      │  │ → Extract transcript/gist      │   │
│  └──────────────────────┘  └────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
           │                   │                      │
           ▼                   ▼                      ▼
┌──────────────────────────────────────────────────────────────────┐
│                         Storage                                  │
│  ┌────────────────────────────┐  ┌───────────────────────────┐  │
│  │ PostgreSQL + pgvector      │  │ Twelve Labs (Video Index) │  │
│  │ • Events, Sessions         │  │ • Multimodal Embeddings   │  │
│  │ • Profiles, VideoEvents    │  │ • Visual + Audio Search   │  │
│  │ • Text Embeddings          │  └───────────────────────────┘  │
│  └────────────────────────────┘                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Data Models

### Event
Stores all observations (speech, vision, actions):
```sql
events (
  event_id UUID PRIMARY KEY,
  robot_id VARCHAR NOT NULL,
  user_id VARCHAR,
  timestamp TIMESTAMPTZ NOT NULL,
  source VARCHAR NOT NULL,      -- 'speech', 'vision', 'action'
  type VARCHAR NOT NULL,        -- 'USER_SAID', 'ROBOT_SAID', etc.
  text TEXT,
  metadata JSONB,
  session_id UUID,
  embedding VECTOR(384)         -- all-MiniLM-L6-v2
)
```

### Session
Groups events into interactions:
```sql
sessions (
  session_id UUID PRIMARY KEY,
  robot_id VARCHAR NOT NULL,
  user_id VARCHAR,
  start_time TIMESTAMPTZ,
  end_time TIMESTAMPTZ,
  summary TEXT                  -- LLM-generated summary
)
```

### Profile
Persistent knowledge about entities:
```sql
profiles (
  profile_id UUID PRIMARY KEY,
  robot_id VARCHAR NOT NULL,
  entity_type VARCHAR,          -- 'user', 'location', 'object'
  entity_id VARCHAR,
  summary TEXT,
  facts JSONB                   -- [{subject, predicate, object}]
)
```

### VideoEvent
Video clips and Twelve Labs metadata:
```sql
video_events (
  video_event_id UUID PRIMARY KEY,
  robot_id VARCHAR NOT NULL,
  user_id VARCHAR,
  session_id UUID,
  start_timestamp TIMESTAMPTZ,
  end_timestamp TIMESTAMPTZ,
  duration_seconds FLOAT,
  video_file_path VARCHAR,      -- Local path or URL
  twelve_labs_video_id VARCHAR, -- ID in Twelve Labs index
  transcript TEXT,              -- Speech-to-text
  scene_description TEXT,       -- Generated description
  processing_status VARCHAR     -- pending, processing, completed, failed
)
```

## Components

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/events` | POST | Log events |
| `/v1/events/{robot_id}` | GET | Query events |
| `/v1/memory/search-events` | POST | Semantic search |
| `/v1/memory/answer` | POST | LLM-generated answer |
| `/v1/memory/profile` | GET | Get entity profile |
| `/v1/memory/video/upload` | POST | Upload video file |
| `/v1/memory/video/from-url` | POST | Upload from URL |
| `/v1/memory/video/search` | POST | Search video memories |
| `/v1/ws/video/{robot_id}` | WS | Stream video chunks |

### Services

| Service | Purpose |
|---------|---------|
| `EmbeddingService` | Generate text embeddings (OpenAI or local) |
| `VectorStore` | Similarity search on PostgreSQL/pgvector |
| `LLMService` | Answer generation using GPT-4 |
| `TwelveLabsService` | Video upload, indexing, search |

### Background Tasks

| Task | Schedule | Purpose |
|------|----------|---------|
| `summarize_sessions` | Periodic | Create session summaries |
| `update_profiles` | On new session | Extract facts to profiles |
| `process_video_chunk` | On upload | Process video via Twelve Labs |
| `reprocess_failed_videos` | Periodic | Retry failed video tasks |

## Video Processing Flow

1. **Client** sends 5-second MP4 chunks via WebSocket or uploads via REST
2. **API** saves file locally, creates `VideoEvent` with status=`pending`
3. **Celery worker** picks up task:
   - Uploads to Twelve Labs
   - Waits for indexing
   - Extracts transcript and gist
   - Updates `VideoEvent` with results
4. **Search** queries Twelve Labs index, matches results to local `VideoEvent`

## Technology Stack

- **Framework**: FastAPI
- **Database**: PostgreSQL + pgvector
- **Queue**: Redis + Celery
- **Video**: Twelve Labs API
- **LLM**: OpenAI GPT-4
- **Embeddings**: OpenAI or sentence-transformers
