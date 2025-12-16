"""Video event schemas."""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID
from enum import Enum


class VideoProcessingStatus(str, Enum):
    """Status of video processing."""
    PENDING = "pending"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoChunkMetadata(BaseModel):
    """Metadata for a video chunk received via WebSocket."""
    robot_id: str = Field(..., description="Robot identifier")
    user_id: Optional[str] = Field(None, description="User identifier if known")
    session_id: Optional[str] = Field(None, description="Session identifier")
    frame_rate: Optional[float] = Field(30.0, description="Video frame rate")
    resolution: Optional[str] = Field(None, description="Video resolution e.g. '1920x1080'")
    codec: Optional[str] = Field("h264", description="Video codec")
    location: Optional[Dict[str, Any]] = Field(None, description="Location metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "robot_id": "robot-123",
                "user_id": "user-456",
                "session_id": "sess-789",
                "frame_rate": 30.0,
                "resolution": "1280x720",
                "codec": "h264"
            }
        }


class VideoEventCreate(BaseModel):
    """Schema for creating a video event (for direct upload, not WebSocket)."""
    robot_id: str = Field(..., description="Robot identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    video_url: Optional[str] = Field(None, description="URL to video file")
    start_timestamp: Optional[datetime] = Field(None, description="Video start time")
    duration_seconds: Optional[float] = Field(None, description="Video duration")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "robot_id": "robot-123",
                "video_url": "https://storage.example.com/videos/clip.mp4",
                "start_timestamp": "2025-12-16T10:00:00Z",
                "duration_seconds": 5.0
            }
        }


class VideoEventResponse(BaseModel):
    """Response schema for video event creation."""
    video_event_id: UUID
    status: VideoProcessingStatus = VideoProcessingStatus.PENDING
    message: Optional[str] = None


class VideoEventDetail(BaseModel):
    """Detailed video event information."""
    video_event_id: UUID
    robot_id: str
    user_id: Optional[str]
    session_id: Optional[UUID]
    start_timestamp: datetime
    end_timestamp: datetime
    duration_seconds: float
    processing_status: VideoProcessingStatus
    transcript: Optional[str] = None
    scene_description: Optional[str] = None
    detected_objects: Optional[List[str]] = None
    detected_actions: Optional[List[str]] = None
    detected_text: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    score: Optional[float] = None  # For search results
    
    class Config:
        from_attributes = True


class VideoSearchRequest(BaseModel):
    """Request schema for video semantic search."""
    robot_id: str = Field(..., description="Robot identifier to search within")
    query: str = Field(..., description="Natural language search query")
    search_options: Optional[List[str]] = Field(
        default=["visual", "audio"],
        description="Search modalities: visual, audio, text_in_video"
    )
    time_from: Optional[datetime] = Field(None, description="Start of time range")
    time_to: Optional[datetime] = Field(None, description="End of time range")
    limit: int = Field(10, ge=1, le=100, description="Maximum results")
    threshold: Optional[float] = Field(None, ge=0, le=100, description="Confidence threshold")
    
    class Config:
        json_schema_extra = {
            "example": {
                "robot_id": "robot-123",
                "query": "person picking up a red cup",
                "search_options": ["visual", "audio"],
                "limit": 10
            }
        }


class VideoSearchResult(BaseModel):
    """Single result from video search."""
    video_event_id: UUID
    robot_id: str
    start_timestamp: datetime
    end_timestamp: datetime
    duration_seconds: float
    score: float = Field(..., description="Relevance score (0-100)")
    clip_start_sec: Optional[float] = Field(None, description="Relevant clip start within video")
    clip_end_sec: Optional[float] = Field(None, description="Relevant clip end within video")
    transcript: Optional[str] = None
    scene_description: Optional[str] = None
    thumbnail_url: Optional[str] = None
    
    class Config:
        from_attributes = True


class VideoSearchResponse(BaseModel):
    """Response schema for video search."""
    results: List[VideoSearchResult]
    total_count: int
    query: str


class WebSocketMessage(BaseModel):
    """Schema for WebSocket control messages."""
    type: str = Field(..., description="Message type: auth, metadata, ping, close")
    api_key: Optional[str] = Field(None, description="API key for auth message")
    metadata: Optional[VideoChunkMetadata] = Field(None, description="Chunk metadata")
    
    class Config:
        json_schema_extra = {
            "examples": [
                {"type": "auth", "api_key": "sk-xxx"},
                {"type": "ping"},
                {"type": "metadata", "metadata": {"robot_id": "robot-123"}}
            ]
        }


class WebSocketAck(BaseModel):
    """Schema for WebSocket acknowledgment messages."""
    type: str = Field(..., description="Message type: chunk_created, pong, error, connected")
    video_event_id: Optional[str] = None
    status: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None

