"""Video event schemas."""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID


class VideoEventCreate(BaseModel):
    """Schema for creating a video event from URL."""
    robot_id: str
    user_id: Optional[str] = None
    video_url: str
    metadata: Optional[Dict[str, Any]] = None


class VideoEventResponse(BaseModel):
    """Response after creating a video event."""
    video_event_id: UUID
    status: str = "pending"
    message: Optional[str] = None


class VideoEventDetail(BaseModel):
    """Full video event details."""
    video_event_id: UUID
    robot_id: str
    user_id: Optional[str]
    start_timestamp: datetime
    end_timestamp: datetime
    duration_seconds: float
    processing_status: str
    transcript: Optional[str] = None
    scene_description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True


class VideoSearchRequest(BaseModel):
    """Request for video search."""
    robot_id: str
    query: str
    search_options: List[str] = ["visual", "audio"]
    limit: int = Field(10, ge=1, le=100)


class VideoSearchResult(BaseModel):
    """Single video search result."""
    video_event_id: UUID
    robot_id: str
    start_timestamp: datetime
    duration_seconds: float
    score: float
    transcript: Optional[str] = None
    scene_description: Optional[str] = None


class VideoSearchResponse(BaseModel):
    """Response for video search."""
    results: List[VideoSearchResult]
    query: str
