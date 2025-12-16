"""Video memory search and retrieval endpoints."""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
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
    """Upload a video file for processing."""
    # Validate file type
    allowed_types = ["video/mp4", "video/quicktime", "video/webm"]
    if video_file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid file type. Use MP4, MOV, or WebM.")
    
    # Save file
    os.makedirs(settings.video_temp_storage_path, exist_ok=True)
    filename = f"{robot_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.mp4"
    filepath = os.path.join(settings.video_temp_storage_path, filename)
    
    async with aiofiles.open(filepath, 'wb') as f:
        await f.write(await video_file.read())
    
    # Create video event
    now = datetime.utcnow()
    video_event = VideoEvent(
        robot_id=robot_id,
        user_id=user_id,
        start_timestamp=now,
        end_timestamp=now,
        duration_seconds=0,
        video_file_path=filepath,
        processing_status="pending"
    )
    
    db.add(video_event)
    db.commit()
    db.refresh(video_event)
    
    # Queue for processing
    from backend.workers.tasks import process_video_chunk
    process_video_chunk.delay(str(video_event.video_event_id))
    
    return VideoEventResponse(
        video_event_id=video_event.video_event_id,
        status="pending",
        message="Video uploaded. Processing queued."
    )


@router.post("/from-url", response_model=VideoEventResponse)
async def create_video_from_url(
    request: VideoEventCreate,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Create a video event from a URL (e.g., S3, GCS)."""
    now = datetime.utcnow()
    video_event = VideoEvent(
        robot_id=request.robot_id,
        user_id=request.user_id,
        start_timestamp=now,
        end_timestamp=now,
        duration_seconds=0,
        video_file_path=request.video_url,
        processing_status="pending",
        metadata=request.metadata
    )
    
    db.add(video_event)
    db.commit()
    db.refresh(video_event)
    
    from backend.workers.tasks import process_video_chunk
    process_video_chunk.delay(str(video_event.video_event_id))
    
    return VideoEventResponse(
        video_event_id=video_event.video_event_id,
        status="pending",
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
    
    Examples: "person waving", "conversation about weather", "red cup on table"
    """
    service = get_twelve_labs_service()
    
    # Search via Twelve Labs
    tl_results = await service.search_by_text(
        query=request.query,
        search_options=request.search_options,
        limit=request.limit
    )
    
    # Match with our database
    results = []
    for tl_result in tl_results:
        video_id = tl_result.get("video_id")
        if not video_id:
            continue
        
        video_event = db.query(VideoEvent).filter(
            VideoEvent.twelve_labs_video_id == video_id,
            VideoEvent.robot_id == request.robot_id
        ).first()
        
        if video_event:
            results.append(VideoSearchResult(
                video_event_id=video_event.video_event_id,
                robot_id=video_event.robot_id,
                start_timestamp=video_event.start_timestamp,
                duration_seconds=video_event.duration_seconds,
                score=tl_result.get("confidence", 0),
                transcript=video_event.transcript,
                scene_description=video_event.scene_description
            ))
    
    return VideoSearchResponse(results=results, query=request.query)


@router.get("/{video_event_id}", response_model=VideoEventDetail)
async def get_video_event(
    video_event_id: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Get details of a video event."""
    try:
        event_uuid = uuid.UUID(video_event_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video_event_id")
    
    video_event = db.query(VideoEvent).filter(
        VideoEvent.video_event_id == event_uuid
    ).first()
    
    if not video_event:
        raise HTTPException(status_code=404, detail="Video event not found")
    
    return VideoEventDetail(
        video_event_id=video_event.video_event_id,
        robot_id=video_event.robot_id,
        user_id=video_event.user_id,
        start_timestamp=video_event.start_timestamp,
        end_timestamp=video_event.end_timestamp,
        duration_seconds=video_event.duration_seconds,
        processing_status=video_event.processing_status,
        transcript=video_event.transcript,
        scene_description=video_event.scene_description,
        metadata=video_event.metadata
    )


@router.get("/robot/{robot_id}/recent", response_model=List[VideoEventDetail])
async def get_recent_videos(
    robot_id: str,
    limit: int = 20,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Get recent video events for a robot."""
    videos = db.query(VideoEvent).filter(
        VideoEvent.robot_id == robot_id
    ).order_by(
        VideoEvent.start_timestamp.desc()
    ).limit(limit).all()
    
    return [
        VideoEventDetail(
            video_event_id=v.video_event_id,
            robot_id=v.robot_id,
            user_id=v.user_id,
            start_timestamp=v.start_timestamp,
            end_timestamp=v.end_timestamp,
            duration_seconds=v.duration_seconds,
            processing_status=v.processing_status,
            transcript=v.transcript,
            scene_description=v.scene_description,
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
    """Delete a video event."""
    try:
        event_uuid = uuid.UUID(video_event_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video_event_id")
    
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
        except Exception:
            pass  # Best effort
    
    # Delete local file
    if video_event.video_file_path and os.path.exists(video_event.video_file_path):
        try:
            os.remove(video_event.video_file_path)
        except Exception:
            pass
    
    db.delete(video_event)
    db.commit()
    
    return {"status": "deleted", "video_event_id": video_event_id}
