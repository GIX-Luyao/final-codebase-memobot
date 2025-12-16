"""WebSocket endpoint for real-time video streaming from robots."""
import os
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import aiofiles

from backend.db.database import SessionLocal
from backend.db.models import VideoEvent
from backend.config import get_settings

settings = get_settings()
router = APIRouter(prefix="/v1/ws", tags=["video-stream"])


class VideoStreamManager:
    """
    Manages WebSocket connections for video streaming.
    
    Expects robots to send complete video chunks (e.g., 5-second MP4 segments)
    rather than individual frames. This ensures valid video files for processing.
    """
    
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
        self.authenticated: Dict[str, bool] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, robot_id: str):
        """Accept a new connection."""
        await websocket.accept()
        self.connections[robot_id] = websocket
        self.authenticated[robot_id] = False
        self.metadata[robot_id] = {}
        
        await websocket.send_json({
            "type": "connected",
            "message": "Send auth message to begin."
        })
    
    def disconnect(self, robot_id: str):
        """Clean up connection."""
        self.connections.pop(robot_id, None)
        self.authenticated.pop(robot_id, None)
        self.metadata.pop(robot_id, None)
    
    async def authenticate(self, robot_id: str, api_key: str) -> bool:
        """Authenticate connection (validate against your auth system)."""
        # TODO: Validate against database in production
        if api_key:
            self.authenticated[robot_id] = True
            return True
        return False
    
    def is_authenticated(self, robot_id: str) -> bool:
        return self.authenticated.get(robot_id, False)
    
    def set_metadata(self, robot_id: str, user_id: str = None, session_id: str = None):
        """Store metadata for subsequent video chunks."""
        if user_id:
            self.metadata[robot_id]["user_id"] = user_id
        if session_id:
            self.metadata[robot_id]["session_id"] = session_id
    
    async def save_video_chunk(
        self,
        robot_id: str,
        video_data: bytes,
        db
    ) -> str:
        """
        Save a video chunk and queue for processing.
        
        Expects video_data to be a valid video file (MP4, WebM, etc.)
        """
        os.makedirs(settings.video_temp_storage_path, exist_ok=True)
        
        # Save with timestamp-based filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{robot_id}_{timestamp}_{uuid.uuid4().hex[:8]}.mp4"
        filepath = os.path.join(settings.video_temp_storage_path, filename)
        
        async with aiofiles.open(filepath, 'wb') as f:
            await f.write(video_data)
        
        # Get metadata
        meta = self.metadata.get(robot_id, {})
        session_uuid = None
        if meta.get("session_id"):
            try:
                session_uuid = uuid.UUID(meta["session_id"])
            except ValueError:
                pass
        
        # Create database record
        now = datetime.utcnow()
        video_event = VideoEvent(
            robot_id=robot_id,
            user_id=meta.get("user_id"),
            session_id=session_uuid,
            start_timestamp=now,
            end_timestamp=now,
            duration_seconds=settings.video_chunk_duration_seconds,
            video_file_path=filepath,
            processing_status="pending"
        )
        
        db.add(video_event)
        db.commit()
        db.refresh(video_event)
        
        # Queue for Twelve Labs processing
        from backend.workers.tasks import process_video_chunk
        process_video_chunk.delay(str(video_event.video_event_id))
        
        return str(video_event.video_event_id)


manager = VideoStreamManager()


@router.websocket("/video/{robot_id}")
async def video_stream_endpoint(websocket: WebSocket, robot_id: str):
    """
    WebSocket endpoint for video chunk streaming.
    
    ## Protocol
    
    1. Connect to `/v1/ws/video/{robot_id}`
    2. Send: `{"type": "auth", "api_key": "your-key"}`
    3. Optionally send: `{"type": "metadata", "user_id": "...", "session_id": "..."}`
    4. Send binary video chunks (complete MP4/WebM segments, ~5 seconds each)
    5. Receive: `{"type": "chunk_saved", "video_event_id": "..."}` for each chunk
    
    ## Important
    
    Video chunks must be complete, valid video files (not raw frames).
    Use tools like ffmpeg on the robot to segment video into chunks:
    
    ```bash
    ffmpeg -i input -c copy -f segment -segment_time 5 -reset_timestamps 1 chunk_%03d.mp4
    ```
    """
    db = SessionLocal()
    
    try:
        await manager.connect(websocket, robot_id)
        
        while True:
            message = await websocket.receive()
            
            if "bytes" in message:
                # Binary video chunk
                if not manager.is_authenticated(robot_id):
                    await websocket.send_json({"type": "error", "error": "Not authenticated"})
                    continue
                
                video_event_id = await manager.save_video_chunk(
                    robot_id, message["bytes"], db
                )
                
                await websocket.send_json({
                    "type": "chunk_saved",
                    "video_event_id": video_event_id,
                    "status": "processing"
                })
            
            elif "text" in message:
                # JSON control message
                try:
                    data = json.loads(message["text"])
                    msg_type = data.get("type", "")
                    
                    if msg_type == "auth":
                        if await manager.authenticate(robot_id, data.get("api_key", "")):
                            await websocket.send_json({"type": "authenticated"})
                        else:
                            await websocket.send_json({"type": "error", "error": "Auth failed"})
                    
                    elif msg_type == "metadata":
                        manager.set_metadata(
                            robot_id,
                            user_id=data.get("user_id"),
                            session_id=data.get("session_id")
                        )
                        await websocket.send_json({"type": "metadata_set"})
                    
                    elif msg_type == "ping":
                        await websocket.send_json({"type": "pong"})
                    
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "error": "Invalid JSON"})
    
    except WebSocketDisconnect:
        manager.disconnect(robot_id)
    except Exception as e:
        print(f"[VideoStream] Error: {e}")
        manager.disconnect(robot_id)
    finally:
        db.close()


@router.get("/video/status/{robot_id}")
async def get_stream_status(robot_id: str):
    """Check if a robot is connected."""
    return {
        "robot_id": robot_id,
        "connected": robot_id in manager.connections,
        "authenticated": manager.is_authenticated(robot_id)
    }
