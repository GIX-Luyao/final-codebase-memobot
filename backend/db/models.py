"""SQLAlchemy models for MemoBot."""
from sqlalchemy import Column, String, Text, DateTime, JSON, Float, Index
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
import uuid

from backend.db.database import Base


class Event(Base):
    """Event model - stores all robot observations and actions."""
    
    __tablename__ = "events"
    
    event_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    robot_id = Column(String(255), nullable=False, index=True)
    user_id = Column(String(255), nullable=True, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    source = Column(String(100), nullable=False)  # 'speech', 'vision', 'action', etc.
    type = Column(String(100), nullable=False)  # 'USER_SAID', 'ROBOT_SAID', etc.
    text = Column(Text, nullable=True)
    metadata = Column(JSON, nullable=True)
    session_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    embedding = Column(Vector(384), nullable=True)  # 384 for all-MiniLM-L6-v2
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Composite indexes for common queries
    __table_args__ = (
        Index('idx_robot_user_timestamp', 'robot_id', 'user_id', 'timestamp'),
        Index('idx_robot_timestamp', 'robot_id', 'timestamp'),
        Index('idx_session_timestamp', 'session_id', 'timestamp'),
    )


class VideoEvent(Base):
    """Video event model - stores video clips and their multimodal embeddings."""
    
    __tablename__ = "video_events"
    
    video_event_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    robot_id = Column(String(255), nullable=False, index=True)
    user_id = Column(String(255), nullable=True, index=True)
    session_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    
    # Timing
    start_timestamp = Column(DateTime(timezone=True), nullable=False)
    end_timestamp = Column(DateTime(timezone=True), nullable=False)
    duration_seconds = Column(Float, nullable=False)
    
    # Storage references
    video_file_path = Column(String(512), nullable=True)  # Local or S3 path
    twelve_labs_task_id = Column(String(255), nullable=True)  # Twelve Labs task ID
    twelve_labs_video_id = Column(String(255), nullable=True, index=True)  # Video ID in Twelve Labs index
    
    # Embeddings (from Twelve Labs Embed API - Marengo-retrieval-2.6)
    video_embedding = Column(Vector(1024), nullable=True)
    
    # Extracted content
    transcript = Column(Text, nullable=True)  # Speech-to-text from video
    scene_description = Column(Text, nullable=True)  # LLM description of scene
    detected_objects = Column(JSON, nullable=True)  # List of detected objects
    detected_actions = Column(JSON, nullable=True)  # List of detected actions
    detected_text = Column(JSON, nullable=True)  # OCR text from video
    
    # Metadata
    metadata = Column(JSON, nullable=True)
    processing_status = Column(String(50), default='pending')  # pending, processing, completed, failed
    processing_error = Column(Text, nullable=True)  # Error message if failed
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_video_robot_timestamp', 'robot_id', 'start_timestamp'),
        Index('idx_video_processing_status', 'processing_status'),
        Index('idx_video_session', 'session_id'),
    )


class Session(Base):
    """Session model - groups related events into conversations/interactions."""
    
    __tablename__ = "sessions"
    
    session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    robot_id = Column(String(255), nullable=False, index=True)
    user_id = Column(String(255), nullable=True, index=True)
    start_time = Column(DateTime(timezone=True), nullable=False)
    end_time = Column(DateTime(timezone=True), nullable=False)
    summary = Column(Text, nullable=True)
    metadata = Column(JSON, nullable=True)  # locations, topics, etc.
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_robot_user_time', 'robot_id', 'user_id', 'start_time'),
    )


class Profile(Base):
    """Profile model - persistent knowledge about users, locations, objects."""
    
    __tablename__ = "profiles"
    
    profile_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    robot_id = Column(String(255), nullable=False, index=True)
    entity_type = Column(String(100), nullable=False)  # 'user', 'location', 'object'
    entity_id = Column(String(255), nullable=False)
    summary = Column(Text, nullable=True)
    facts = Column(JSON, nullable=True)  # List of {subject, predicate, object, confidence}
    last_updated = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_robot_entity', 'robot_id', 'entity_type', 'entity_id', unique=True),
    )

