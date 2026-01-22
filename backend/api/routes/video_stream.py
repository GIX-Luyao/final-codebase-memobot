"""WebSocket endpoint for continuous multimodal memory streaming."""
import os
import json
import uuid
import base64
from datetime import datetime
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import aiofiles

from backend.db.database import SessionLocal
from backend.db.models import VideoEvent
from backend.config import get_settings

settings = get_settings()
router = APIRouter(prefix="/v1/ws", tags=["stream"])


class VideoSegmentReceiver:
    """
    Receives complete video segments from robots.
    
    IMPORTANT: Robots must send pre-encoded video segments (MP4/WebM), NOT raw frames.
    Use ffmpeg on the robot to create segments:
        ffmpeg -i /dev/video0 -c:v libx264 -f segment -segment_time 5 chunk_%03d.mp4
    """
    
    def __init__(self, robot_id: str):
        self.robot_id = robot_id
        self.metadata: Dict[str, Any] = {}
        self.pending_segment: Optional[bytes] = None
        self.segment_actions: List[Dict] = []
        self.segment_start: Optional[datetime] = None
    
    def set_segment_start(self, timestamp: datetime = None):
        """Mark start of a new segment."""
        if not self.segment_start:
            self.segment_start = timestamp or datetime.utcnow()
    
    def add_action(self, action: str, timestamp: datetime = None):
        """Record an action during the segment."""
        self.segment_actions.append({
            "action": action, 
            "timestamp": (timestamp or datetime.utcnow()).isoformat()
        })
    
    async def save_segment(self, video_data: bytes, db) -> str:
        """
        Save a complete video segment and queue for processing.
        
        Args:
            video_data: Complete video file bytes (MP4, WebM, etc.)
            db: Database session
        
        Returns:
            memory_id of the created VideoEvent
        """
        os.makedirs(settings.video_temp_storage_path, exist_ok=True)
        
        # Detect format from magic bytes
        ext = ".mp4"  # default
        if video_data[:4] == b'\x1a\x45\xdf\xa3':  # WebM/MKV
            ext = ".webm"
        elif video_data[:3] == b'FLV':
            ext = ".flv"
        
        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.robot_id}_{timestamp_str}_{uuid.uuid4().hex[:8]}{ext}"
        filepath = os.path.join(settings.video_temp_storage_path, filename)
        
        async with aiofiles.open(filepath, 'wb') as f:
            await f.write(video_data)
        
        # Parse session_id safely
        session_uuid = None
        if self.metadata.get("session_id"):
            try:
                session_uuid = uuid.UUID(self.metadata["session_id"])
            except (ValueError, TypeError):
                pass
        
        now = datetime.utcnow()
        start = self.segment_start or now
        
        video_event = VideoEvent(
            robot_id=self.robot_id,
            user_id=self.metadata.get("user_id"),
            session_id=session_uuid,
            start_timestamp=start,
            end_timestamp=now,
            duration_seconds=(now - start).total_seconds(),
            video_file_path=filepath,
            processing_status="pending",
            metadata={"actions": self.segment_actions} if self.segment_actions else None
        )
        
        db.add(video_event)
        db.commit()
        db.refresh(video_event)
        
        # Queue for Twelve Labs processing
        from backend.workers.tasks import process_video_chunk
        process_video_chunk.delay(str(video_event.video_event_id))
        
        # Reset for next segment
        self.segment_actions = []
        self.segment_start = None
        
        return str(video_event.video_event_id)


class StreamManager:
    """Manages WebSocket connections for multimodal streaming."""
    
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
        self.receivers: Dict[str, VideoSegmentReceiver] = {}
        self.authenticated: Dict[str, bool] = {}
    
    async def connect(self, websocket: WebSocket, robot_id: str):
        await websocket.accept()
        self.connections[robot_id] = websocket
        self.receivers[robot_id] = VideoSegmentReceiver(robot_id)
        self.authenticated[robot_id] = False
        
        await websocket.send_json({"type": "connected", "message": "Send auth to begin"})
    
    def disconnect(self, robot_id: str):
        self.connections.pop(robot_id, None)
        self.receivers.pop(robot_id, None)
        self.authenticated.pop(robot_id, None)
    
    async def authenticate(self, robot_id: str, api_key: str) -> bool:
        # TODO: Validate against database
        if api_key:
            self.authenticated[robot_id] = True
            return True
        return False
    
    def set_metadata(self, robot_id: str, metadata: Dict):
        if robot_id in self.receivers:
            self.receivers[robot_id].metadata.update(metadata)


manager = StreamManager()


@router.websocket("/stream/{robot_id}")
async def multimodal_stream_endpoint(websocket: WebSocket, robot_id: str):
    """
    WebSocket endpoint for video segment streaming.
    
    ## Protocol
    
    1. Connect to `/v1/ws/stream/{robot_id}`
    2. Send: `{"type": "auth", "api_key": "your-key"}`
    3. Optionally: `{"type": "metadata", "user_id": "...", "session_id": "..."}`
    4. Send video segments as binary (complete MP4/WebM files, ~5 seconds each)
    5. Receive acks: `{"type": "ack_stored", "memory_id": "..."}`
    
    ## Important
    
    Send COMPLETE video segments (MP4/WebM), not raw frames.
    Use ffmpeg on the robot to create segments:
    
    ```bash
    ffmpeg -i /dev/video0 -c:v libx264 -f segment -segment_time 5 -reset_timestamps 1 chunk_%03d.mp4
    ```
    
    Or for JSON format with base64:
    ```json
    {"type": "segment", "video": "<base64 encoded MP4>", "action": "MOVING"}
    ```
    """
    db = SessionLocal()
    
    try:
        await manager.connect(websocket, robot_id)
        
        while True:
            message = await websocket.receive()
            
            if "text" in message:
                try:
                    data = json.loads(message["text"])
                    msg_type = data.get("type", "")
                    
                    if msg_type == "auth":
                        if await manager.authenticate(robot_id, data.get("api_key", "")):
                            await websocket.send_json({"type": "authenticated"})
                        else:
                            await websocket.send_json({"type": "error", "error": "Auth failed"})
                    
                    elif msg_type == "metadata":
                        manager.set_metadata(robot_id, {
                            "user_id": data.get("user_id"),
                            "session_id": data.get("session_id"),
                            "location": data.get("location")
                        })
                        await websocket.send_json({"type": "metadata_set"})
                    
                    elif msg_type == "segment":
                        # JSON-encoded video segment
                        if not manager.authenticated.get(robot_id):
                            await websocket.send_json({"type": "error", "error": "Not authenticated"})
                            continue
                        
                        video_data = base64.b64decode(data.get("video", "")) if data.get("video") else None
                        if not video_data:
                            await websocket.send_json({"type": "error", "error": "No video data"})
                            continue
                        
                        receiver = manager.receivers[robot_id]
                        if data.get("action"):
                            receiver.add_action(data["action"])
                        
                        memory_id = await receiver.save_segment(video_data, db)
                        await websocket.send_json({
                            "type": "ack_stored",
                            "memory_id": memory_id
                        })
                    
                    elif msg_type == "action":
                        # Record action without video
                        if robot_id in manager.receivers:
                            manager.receivers[robot_id].add_action(data.get("action", "UNKNOWN"))
                        await websocket.send_json({"type": "action_recorded"})
                    
                    elif msg_type == "ping":
                        await websocket.send_json({"type": "pong"})
                    
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "error": "Invalid JSON"})
            
            elif "bytes" in message:
                # Binary video segment (preferred for efficiency)
                if not manager.authenticated.get(robot_id):
                    await websocket.send_json({"type": "error", "error": "Not authenticated"})
                    continue
                
                video_data = message["bytes"]
                if len(video_data) < 100:
                    await websocket.send_json({"type": "error", "error": "Video segment too small"})
                    continue
                
                receiver = manager.receivers[robot_id]
                memory_id = await receiver.save_segment(video_data, db)
                await websocket.send_json({
                    "type": "ack_stored",
                    "memory_id": memory_id
                })
    
    except WebSocketDisconnect:
        manager.disconnect(robot_id)
    except Exception as e:
        print(f"[Stream] Error: {e}")
        manager.disconnect(robot_id)
    finally:
        db.close()


@router.get("/stream/status/{robot_id}")
async def get_stream_status(robot_id: str):
    """Check stream connection status for a robot."""
    return {
        "robot_id": robot_id,
        "connected": robot_id in manager.connections,
        "authenticated": manager.authenticated.get(robot_id, False)
    }
