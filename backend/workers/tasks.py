"""Celery tasks for background processing."""
import asyncio
import os
from datetime import datetime, timedelta
from typing import List
import uuid

from backend.workers.celery_app import celery_app
from backend.db.database import SessionLocal
from backend.db.models import Event, Session as SessionModel, Profile, VideoEvent
from backend.services.llm import get_llm_service
from backend.config import get_settings

settings = get_settings()


def _run_async(coro):
    """Run async coroutine in sync context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# =============================================================================
# Video Processing Tasks
# =============================================================================

@celery_app.task(
    name="backend.workers.tasks.process_video_chunk",
    bind=True,
    max_retries=3,
    default_retry_delay=60
)
def process_video_chunk(self, video_event_id: str):
    """
    Process a video through Twelve Labs API.
    
    Handles both local files and URLs.
    """
    if not settings.enable_video_processing:
        return {"status": "disabled"}
    
    if not settings.twelve_labs_api_key:
        return {"status": "error", "message": "Twelve Labs API key not configured"}
    
    db = SessionLocal()
    
    try:
        video_event = db.query(VideoEvent).filter(
            VideoEvent.video_event_id == video_event_id
        ).first()
        
        if not video_event:
            return {"status": "error", "message": "Video event not found"}
        
        if video_event.processing_status == "completed":
            return {"status": "skipped", "message": "Already processed"}
        
        video_event.processing_status = "processing"
        db.commit()
        
        result = _run_async(_process_video(video_event, db))
        return result
    
    except Exception as e:
        db.rollback()
        
        # Mark as failed
        try:
            video_event = db.query(VideoEvent).filter(
                VideoEvent.video_event_id == video_event_id
            ).first()
            if video_event:
                video_event.processing_status = "failed"
                video_event.processing_error = str(e)
                db.commit()
        except:
            pass
        
        raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))
    
    finally:
        db.close()


async def _process_video(video_event: VideoEvent, db) -> dict:
    """Process video through Twelve Labs."""
    from backend.services.video_embedding import get_twelve_labs_service
    
    service = get_twelve_labs_service()
    video_path = video_event.video_file_path
    
    # Determine if it's a URL or local file
    is_url = video_path.startswith("http://") or video_path.startswith("https://")
    
    # Upload to Twelve Labs
    if is_url:
        task_id = await service.upload_video_from_url(video_path)
    else:
        if not os.path.exists(video_path):
            raise Exception(f"Video file not found: {video_path}")
        task_id = await service.upload_video(video_path)
    
    # Wait for indexing
    task_status = await service.wait_for_task(task_id)
    video_id = task_status.get("video_id")
    
    if not video_id:
        raise Exception("No video_id returned from Twelve Labs")
    
    video_event.twelve_labs_video_id = video_id
    
    # Get transcript
    transcript = await service.get_video_transcription(video_id)
    if transcript:
        video_event.transcript = transcript
    
    # Get scene description
    gist = await service.generate_gist(video_id, types=["topic", "title"])
    if gist:
        title = gist.get("title", "")
        topics = gist.get("topics", [])
        video_event.scene_description = f"{title}. {', '.join(topics)}" if topics else title
    
    video_event.processing_status = "completed"
    db.commit()
    
    return {
        "status": "completed",
        "video_event_id": str(video_event.video_event_id),
        "twelve_labs_video_id": video_id
    }


@celery_app.task(name="backend.workers.tasks.reprocess_failed_videos")
def reprocess_failed_videos():
    """Retry failed video processing (runs periodically)."""
    if not settings.enable_video_processing:
        return {"status": "disabled"}
    
    db = SessionLocal()
    
    try:
        cutoff = datetime.utcnow() - timedelta(hours=24)
        failed = db.query(VideoEvent).filter(
            VideoEvent.processing_status == "failed",
            VideoEvent.created_at >= cutoff
        ).limit(10).all()
        
        for video in failed:
            video.processing_status = "pending"
            video.processing_error = None
            db.commit()
            process_video_chunk.delay(str(video.video_event_id))
        
        return {"status": "success", "requeued": len(failed)}
    
    except Exception as e:
        return {"status": "error", "error": str(e)}
    
    finally:
        db.close()


@celery_app.task(name="backend.workers.tasks.summarize_sessions_task")
def summarize_sessions_task():
    """
    Periodic task to group events into sessions and summarize them.
    
    Looks for events without a session_id and groups them based on:
    - Same robot_id and user_id
    - Within 30 minutes of each other
    """
    if not settings.enable_summarization:
        return {"status": "disabled"}
    
    db = SessionLocal()
    llm_service = get_llm_service()
    
    try:
        # Find events without sessions (recent ones)
        cutoff_time = datetime.utcnow() - timedelta(days=7)
        unsessioned_events = db.query(Event).filter(
            Event.session_id.is_(None),
            Event.timestamp >= cutoff_time
        ).order_by(Event.robot_id, Event.user_id, Event.timestamp).all()
        
        # Group into sessions
        sessions_created = 0
        current_session_events = []
        current_robot_user = None
        last_event_time = None
        session_gap_minutes = 30
        
        for event in unsessioned_events:
            robot_user = (event.robot_id, event.user_id)
            
            # Check if we should start a new session
            should_start_new = (
                not current_session_events or
                robot_user != current_robot_user or
                (last_event_time and 
                 (event.timestamp - last_event_time).total_seconds() > session_gap_minutes * 60)
            )
            
            if should_start_new and current_session_events:
                # Create session for previous group
                _create_session(db, llm_service, current_session_events)
                sessions_created += 1
                current_session_events = []
            
            current_session_events.append(event)
            current_robot_user = robot_user
            last_event_time = event.timestamp
        
        # Create final session
        if current_session_events:
            _create_session(db, llm_service, current_session_events)
            sessions_created += 1
        
        db.commit()
        
        return {
            "status": "success",
            "sessions_created": sessions_created,
            "events_processed": len(unsessioned_events)
        }
    
    except Exception as e:
        db.rollback()
        return {"status": "error", "error": str(e)}
    
    finally:
        db.close()


def _create_session(db, llm_service, events: List[Event]):
    """Create a session from a list of events."""
    if not events:
        return
    
    # Create session
    session_id = uuid.uuid4()
    start_time = min(e.timestamp for e in events)
    end_time = max(e.timestamp for e in events)
    
    # Convert events for summarization
    event_dicts = [
        {
            "event_id": str(e.event_id),
            "text": e.text,
            "type": e.type,
            "timestamp": e.timestamp
        }
        for e in events
        if e.text
    ]
    
    # Generate summary
    summary = llm_service.summarize_session(event_dicts)
    
    # Extract metadata
    locations = set()
    for e in events:
        if e.metadata and "location" in e.metadata:
            locations.add(e.metadata["location"])
    
    metadata = {
        "locations": list(locations),
        "event_count": len(events)
    }
    
    # Create session record
    session = SessionModel(
        session_id=session_id,
        robot_id=events[0].robot_id,
        user_id=events[0].user_id,
        start_time=start_time,
        end_time=end_time,
        summary=summary,
        metadata=metadata
    )
    
    db.add(session)
    
    # Update events with session_id
    for event in events:
        event.session_id = session_id


@celery_app.task(name="backend.workers.tasks.update_profiles_task")
def update_profiles_task():
    """
    Periodic task to update user/location/object profiles.
    
    Looks for entities that have had new events and updates their profiles.
    """
    if not settings.enable_profiles:
        return {"status": "disabled"}
    
    db = SessionLocal()
    llm_service = get_llm_service()
    
    try:
        # Find profiles that need updating (haven't been updated in 24 hours)
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        # For now, just find active robot_id + user_id combinations
        recent_events = db.query(
            Event.robot_id,
            Event.user_id
        ).filter(
            Event.user_id.isnot(None),
            Event.timestamp >= cutoff_time
        ).distinct().all()
        
        profiles_updated = 0
        
        for robot_id, user_id in recent_events:
            if not user_id:
                continue
            
            # Get or create profile
            profile = db.query(Profile).filter(
                Profile.robot_id == robot_id,
                Profile.entity_type == "user",
                Profile.entity_id == user_id
            ).first()
            
            # Get recent events for this user
            events = db.query(Event).filter(
                Event.robot_id == robot_id,
                Event.user_id == user_id,
                Event.text.isnot(None)
            ).order_by(Event.timestamp.desc()).limit(50).all()
            
            if not events:
                continue
            
            # Convert to dict format
            event_dicts = [
                {
                    "event_id": str(e.event_id),
                    "text": e.text,
                    "type": e.type,
                    "timestamp": e.timestamp
                }
                for e in events
            ]
            
            # Generate summary and facts
            summary = llm_service.summarize_session(event_dicts)
            facts = llm_service.extract_facts(event_dicts, user_id)
            
            if profile:
                # Update existing profile
                profile.summary = summary
                profile.facts = facts
                profile.last_updated = datetime.utcnow()
            else:
                # Create new profile
                profile = Profile(
                    robot_id=robot_id,
                    entity_type="user",
                    entity_id=user_id,
                    summary=summary,
                    facts=facts,
                    last_updated=datetime.utcnow()
                )
                db.add(profile)
            
            profiles_updated += 1
        
        db.commit()
        
        return {
            "status": "success",
            "profiles_updated": profiles_updated
        }
    
    except Exception as e:
        db.rollback()
        return {"status": "error", "error": str(e)}
    
    finally:
        db.close()

