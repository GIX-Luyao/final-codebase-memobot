"""Video memory search and retrieval endpoints."""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from sqlalchemy import and_
from datetime import datetime
from typing import List, Optional
import os
import uuid
import aiofiles

from backend.db.database import get_db
from backend.db.models import VideoEvent
from backend.services.video_embedding import get_twelve_labs_service
from backend.api.dependencies import verify_api_key
from backend.config import get_settings
from backend.schemas.video import (
    VideoEventCreate,
    VideoEventResponse,
    VideoEventDetail,
    VideoSearchRequest,
    VideoSearchResult,
    VideoSearchResponse,
    VideoProcessingStatus
)

settings = get_settings()
router = APIRouter(prefix="/v1/memory/video", tags=["video-memory"])


@router.post("/upload", response_model=VideoEventResponse)
async def upload_video(
    robot_id: str = Form(...),
    video_file: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Upload a video file for processing.
    
    Alternative to WebSocket streaming for uploading pre-recorded video clips.
    The video will be processed asynchronously through Twelve Labs.
    
    Supported formats: MP4, MOV, AVI, MKV, WEBM
    """
    # Validate file type
    allowed_types = ["video/mp4", "video/quicktime", "video/x-msvideo", 
                     "video/x-matroska", "video/webm"]
    if video_file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
        )
    
    # Save file
    os.makedirs(settings.video_temp_storage_path, exist_ok=True)
    
    file_ext = os.path.splitext(video_file.filename)[1] or ".mp4"
    filename = f"{robot_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{file_ext}"
    filepath = os.path.join(settings.video_temp_storage_path, filename)
    
    async with aiofiles.open(filepath, 'wb') as f:
        content = await video_file.read()
        await f.write(content)
    
    # Create video event
    now = datetime.utcnow()
    video_event = VideoEvent(
        robot_id=robot_id,
        user_id=user_id,
        start_timestamp=now,
        end_timestamp=now,
        duration_seconds=0,  # Will be updated after processing
        video_file_path=filepath,
        processing_status="pending",
        metadata={"original_filename": video_file.filename}
    )
    
    db.add(video_event)
    db.commit()
    db.refresh(video_event)
    
    # Queue for processing
    from backend.workers.tasks import process_video_chunk
    process_video_chunk.delay(str(video_event.video_event_id))
    
    return VideoEventResponse(
        video_event_id=video_event.video_event_id,
        status=VideoProcessingStatus.PENDING,
        message="Video uploaded successfully. Processing queued."
    )


@router.post("/from-url", response_model=VideoEventResponse)
async def create_video_from_url(
    request: VideoEventCreate,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Create a video event from a URL.
    
    The video will be fetched and processed by Twelve Labs directly.
    Useful for videos stored in cloud storage (S3, GCS, etc.).
    """
    if not request.video_url:
        raise HTTPException(status_code=400, detail="video_url is required")
    
    now = datetime.utcnow()
    start_time = request.start_timestamp or now
    
    video_event = VideoEvent(
        robot_id=request.robot_id,
        user_id=request.user_id,
        start_timestamp=start_time,
        end_timestamp=start_time,
        duration_seconds=request.duration_seconds or 0,
        video_file_path=request.video_url,  # Store URL instead of local path
        processing_status="pending",
        metadata=request.metadata or {}
    )
    
    db.add(video_event)
    db.commit()
    db.refresh(video_event)
    
    # Queue for processing (will use upload_video_from_url)
    from backend.workers.tasks import process_video_chunk
    process_video_chunk.delay(str(video_event.video_event_id))
    
    return VideoEventResponse(
        video_event_id=video_event.video_event_id,
        status=VideoProcessingStatus.PENDING,
        message="Video event created. Processing queued."
    )


@router.post("/search", response_model=VideoSearchResponse)
async def search_video_memories(
    request: VideoSearchRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Search video memories using natural language.
    
    Uses Twelve Labs semantic search to find relevant video content
    based on visual, audio, and text-in-video analysis.
    
    Example queries:
    - "person picking up a red object"
    - "someone waving hello"
    - "conversation about the weather"
    - "whiteboard with diagrams"
    """
    service = get_twelve_labs_service()
    
    # Search via Twelve Labs
    search_results = await service.search_by_text(
        query=request.query,
        search_options=request.search_options or ["visual", "audio"],
        limit=request.limit,
        threshold=request.threshold
    )
    
    # Build filters for database query
    filters = [VideoEvent.robot_id == request.robot_id]
    
    if request.time_from:
        filters.append(VideoEvent.start_timestamp >= request.time_from)
    if request.time_to:
        filters.append(VideoEvent.end_timestamp <= request.time_to)
    
    # Match Twelve Labs results with our database records
    results = []
    
    for tl_result in search_results:
        video_id = tl_result.get("video_id")
        
        if not video_id:
            continue
        
        # Find matching video event
        video_event = db.query(VideoEvent).filter(
            and_(
                VideoEvent.twelve_labs_video_id == video_id,
                *filters
            )
        ).first()
        
        if video_event:
            # Get clip timing from Twelve Labs result
            clips = tl_result.get("clips", [])
            clip_start = clips[0].get("start") if clips else None
            clip_end = clips[0].get("end") if clips else None
            
            results.append(VideoSearchResult(
                video_event_id=video_event.video_event_id,
                robot_id=video_event.robot_id,
                start_timestamp=video_event.start_timestamp,
                end_timestamp=video_event.end_timestamp,
                duration_seconds=video_event.duration_seconds,
                score=tl_result.get("confidence", 0),
                clip_start_sec=clip_start,
                clip_end_sec=clip_end,
                transcript=video_event.transcript,
                scene_description=video_event.scene_description
            ))
    
    return VideoSearchResponse(
        results=results,
        total_count=len(results),
        query=request.query
    )


@router.get("/{video_event_id}", response_model=VideoEventDetail)
async def get_video_event(
    video_event_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Get details of a specific video event.
    
    Includes processing status, transcript, detected objects, etc.
    """
    try:
        event_uuid = uuid.UUID(video_event_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video_event_id format")
    
    video_event = db.query(VideoEvent).filter(
        VideoEvent.video_event_id == event_uuid
    ).first()
    
    if not video_event:
        raise HTTPException(status_code=404, detail="Video event not found")
    
    return VideoEventDetail(
        video_event_id=video_event.video_event_id,
        robot_id=video_event.robot_id,
        user_id=video_event.user_id,
        session_id=video_event.session_id,
        start_timestamp=video_event.start_timestamp,
        end_timestamp=video_event.end_timestamp,
        duration_seconds=video_event.duration_seconds,
        processing_status=VideoProcessingStatus(video_event.processing_status),
        transcript=video_event.transcript,
        scene_description=video_event.scene_description,
        detected_objects=video_event.detected_objects,
        detected_actions=video_event.detected_actions,
        detected_text=video_event.detected_text,
        metadata=video_event.metadata
    )


@router.get("/robot/{robot_id}/recent", response_model=List[VideoEventDetail])
async def get_recent_videos(
    robot_id: str,
    limit: int = 20,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Get recent video events for a robot.
    
    Optionally filter by processing status.
    """
    query = db.query(VideoEvent).filter(VideoEvent.robot_id == robot_id)
    
    if status:
        query = query.filter(VideoEvent.processing_status == status)
    
    videos = query.order_by(VideoEvent.start_timestamp.desc()).limit(limit).all()
    
    return [
        VideoEventDetail(
            video_event_id=v.video_event_id,
            robot_id=v.robot_id,
            user_id=v.user_id,
            session_id=v.session_id,
            start_timestamp=v.start_timestamp,
            end_timestamp=v.end_timestamp,
            duration_seconds=v.duration_seconds,
            processing_status=VideoProcessingStatus(v.processing_status),
            transcript=v.transcript,
            scene_description=v.scene_description,
            detected_objects=v.detected_objects,
            detected_actions=v.detected_actions,
            detected_text=v.detected_text,
            metadata=v.metadata
        )
        for v in videos
    ]


@router.delete("/{video_event_id}")
async def delete_video_event(
    video_event_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Delete a video event and its associated data.
    
    Also removes the video from Twelve Labs index if present.
    """
    try:
        event_uuid = uuid.UUID(video_event_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video_event_id format")
    
    video_event = db.query(VideoEvent).filter(
        VideoEvent.video_event_id == event_uuid
    ).first()
    
    if not video_event:
        raise HTTPException(status_code=404, detail="Video event not found")
    
    # Delete from Twelve Labs if indexed
    if video_event.twelve_labs_video_id:
        try:
            service = get_twelve_labs_service()
            await service.delete_video(video_event.twelve_labs_video_id)
        except Exception as e:
            # Log but don't fail if Twelve Labs deletion fails
            print(f"Warning: Failed to delete from Twelve Labs: {e}")
    
    # Delete local file if exists
    if video_event.video_file_path and os.path.exists(video_event.video_file_path):
        try:
            os.remove(video_event.video_file_path)
        except Exception as e:
            print(f"Warning: Failed to delete local file: {e}")
    
    # Delete database record
    db.delete(video_event)
    db.commit()
    
    return {"status": "deleted", "video_event_id": video_event_id}


@router.get("/stats/{robot_id}")
async def get_video_stats(
    robot_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Get video processing statistics for a robot.
    """
    from sqlalchemy import func
    
    # Count by status
    status_counts = db.query(
        VideoEvent.processing_status,
        func.count(VideoEvent.video_event_id)
    ).filter(
        VideoEvent.robot_id == robot_id
    ).group_by(VideoEvent.processing_status).all()
    
    # Total duration
    total_duration = db.query(
        func.sum(VideoEvent.duration_seconds)
    ).filter(
        VideoEvent.robot_id == robot_id,
        VideoEvent.processing_status == "completed"
    ).scalar() or 0
    
    return {
        "robot_id": robot_id,
        "status_counts": {status: count for status, count in status_counts},
        "total_duration_seconds": total_duration,
        "total_duration_hours": round(total_duration / 3600, 2)
    }

