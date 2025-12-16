"""WebSocket endpoint for real-time video streaming from robots."""
import asyncio
import os
import json
import tempfile
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query
import aiofiles
import struct

from backend.db.database import SessionLocal
from backend.db.models import VideoEvent
from backend.config import get_settings
from backend.schemas.video import (
    VideoChunkMetadata,
    WebSocketMessage,
    WebSocketAck,
    VideoProcessingStatus
)

settings = get_settings()
router = APIRouter(prefix="/v1/ws", tags=["video-stream"])


class VideoFrameBuffer:
    """
    Buffer for accumulating video frames into processable chunks.
    
    Collects incoming binary frames and saves them as video files
    when the configured chunk duration is reached.
    """
    
    def __init__(
        self,
        robot_id: str,
        chunk_duration: int = 5,
        frame_rate: float = 30.0
    ):
        """
        Initialize frame buffer.
        
        Args:
            robot_id: Robot identifier
            chunk_duration: Seconds per video chunk
            frame_rate: Expected frame rate
        """
        self.robot_id = robot_id
        self.chunk_duration = chunk_duration
        self.frame_rate = frame_rate
        
        # Buffer state
        self.frames: list[bytes] = []
        self.start_time: Optional[datetime] = None
        self.frame_count = 0
        
        # Metadata from client
        self.user_id: Optional[str] = None
        self.session_id: Optional[str] = None
        self.extra_metadata: Dict[str, Any] = {}
    
    def update_metadata(self, metadata: VideoChunkMetadata):
        """Update buffer metadata from client message."""
        self.user_id = metadata.user_id
        self.session_id = metadata.session_id
        if metadata.frame_rate:
            self.frame_rate = metadata.frame_rate
        self.extra_metadata = {
            "resolution": metadata.resolution,
            "codec": metadata.codec,
            "location": metadata.location
        }
    
    async def add_frame(self, frame_data: bytes) -> Optional[Dict[str, Any]]:
        """
        Add a frame to the buffer.
        
        Args:
            frame_data: Raw frame bytes (could be JPEG, H264 NAL, etc.)
            
        Returns:
            Dict with chunk info if chunk is complete, None otherwise
        """
        if self.start_time is None:
            self.start_time = datetime.utcnow()
        
        self.frames.append(frame_data)
        self.frame_count += 1
        
        # Check if we've collected enough for a chunk
        elapsed = (datetime.utcnow() - self.start_time).total_seconds()
        
        if elapsed >= self.chunk_duration:
            return await self._save_chunk()
        
        return None
    
    async def flush(self) -> Optional[Dict[str, Any]]:
        """Force save any remaining frames as a chunk."""
        if self.frames and self.start_time:
            return await self._save_chunk()
        return None
    
    async def _save_chunk(self) -> Dict[str, Any]:
        """Save buffered frames as a video file."""
        os.makedirs(settings.video_temp_storage_path, exist_ok=True)
        
        # Generate unique filename
        timestamp_str = self.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.robot_id}_{timestamp_str}_{uuid.uuid4().hex[:8]}.mp4"
        filepath = os.path.join(settings.video_temp_storage_path, filename)
        
        # Calculate duration
        end_time = datetime.utcnow()
        duration = (end_time - self.start_time).total_seconds()
        
        # Write frames to file
        # Note: In production, you'd use ffmpeg to properly encode frames
        # This writes raw frame data which needs to be in a container format
        async with aiofiles.open(filepath, 'wb') as f:
            # Write a simple header with frame count for reconstruction
            header = struct.pack('>I', len(self.frames))  # Frame count as 4-byte big-endian
            await f.write(header)
            
            for frame in self.frames:
                # Write frame length then frame data
                frame_header = struct.pack('>I', len(frame))
                await f.write(frame_header)
                await f.write(frame)
        
        chunk_info = {
            "filepath": filepath,
            "start_time": self.start_time,
            "end_time": end_time,
            "duration_seconds": duration,
            "frame_count": self.frame_count,
            "robot_id": self.robot_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "metadata": self.extra_metadata
        }
        
        # Reset buffer
        self.frames = []
        self.start_time = None
        self.frame_count = 0
        
        return chunk_info


class ConnectionManager:
    """
    Manages WebSocket connections from robots.
    
    Handles:
    - Connection lifecycle
    - Frame buffering per robot
    - Authentication
    """
    
    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.buffers: Dict[str, VideoFrameBuffer] = {}
        self.authenticated: Dict[str, bool] = {}
    
    async def connect(self, websocket: WebSocket, robot_id: str) -> bool:
        """
        Accept and register a new WebSocket connection.
        
        Args:
            websocket: The WebSocket connection
            robot_id: Robot identifier from URL
            
        Returns:
            True if connection accepted
        """
        await websocket.accept()
        self.active_connections[robot_id] = websocket
        self.buffers[robot_id] = VideoFrameBuffer(
            robot_id,
            chunk_duration=settings.video_chunk_duration_seconds
        )
        self.authenticated[robot_id] = False  # Require auth message
        
        print(f"[VideoStream] Robot {robot_id} connected")
        
        # Send connection confirmation
        await websocket.send_json(
            WebSocketAck(
                type="connected",
                message="Connection established. Send auth message to begin streaming."
            ).model_dump()
        )
        
        return True
    
    def disconnect(self, robot_id: str):
        """Clean up when a robot disconnects."""
        if robot_id in self.active_connections:
            del self.active_connections[robot_id]
        if robot_id in self.buffers:
            del self.buffers[robot_id]
        if robot_id in self.authenticated:
            del self.authenticated[robot_id]
        
        print(f"[VideoStream] Robot {robot_id} disconnected")
    
    def is_authenticated(self, robot_id: str) -> bool:
        """Check if robot has authenticated."""
        return self.authenticated.get(robot_id, False)
    
    async def authenticate(self, robot_id: str, api_key: str) -> bool:
        """
        Authenticate a robot connection.
        
        Args:
            robot_id: Robot identifier
            api_key: API key from auth message
            
        Returns:
            True if authentication successful
        """
        # In production, validate against database
        # For now, accept any non-empty key
        if api_key and len(api_key) > 0:
            self.authenticated[robot_id] = True
            return True
        return False
    
    async def process_frame(
        self,
        robot_id: str,
        frame_data: bytes,
        db
    ) -> Optional[str]:
        """
        Process an incoming video frame.
        
        Args:
            robot_id: Robot identifier
            frame_data: Raw frame bytes
            db: Database session
            
        Returns:
            Video event ID if a chunk was created, None otherwise
        """
        if robot_id not in self.buffers:
            return None
        
        if not self.is_authenticated(robot_id):
            return None
        
        buffer = self.buffers[robot_id]
        chunk_info = await buffer.add_frame(frame_data)
        
        if chunk_info:
            return await self._create_video_event(chunk_info, db)
        
        return None
    
    async def flush_buffer(self, robot_id: str, db) -> Optional[str]:
        """Flush remaining frames when connection closes."""
        if robot_id not in self.buffers:
            return None
        
        buffer = self.buffers[robot_id]
        chunk_info = await buffer.flush()
        
        if chunk_info:
            return await self._create_video_event(chunk_info, db)
        
        return None
    
    async def _create_video_event(
        self,
        chunk_info: Dict[str, Any],
        db
    ) -> str:
        """Create a VideoEvent record for a completed chunk."""
        # Parse session_id as UUID if provided
        session_uuid = None
        if chunk_info.get("session_id"):
            try:
                session_uuid = uuid.UUID(chunk_info["session_id"])
            except ValueError:
                pass
        
        video_event = VideoEvent(
            robot_id=chunk_info["robot_id"],
            user_id=chunk_info.get("user_id"),
            session_id=session_uuid,
            start_timestamp=chunk_info["start_time"],
            end_timestamp=chunk_info["end_time"],
            duration_seconds=chunk_info["duration_seconds"],
            video_file_path=chunk_info["filepath"],
            metadata={
                "frame_count": chunk_info["frame_count"],
                **chunk_info.get("metadata", {})
            },
            processing_status="pending"
        )
        
        db.add(video_event)
        db.commit()
        db.refresh(video_event)
        
        # Queue background task for Twelve Labs processing
        from backend.workers.tasks import process_video_chunk
        process_video_chunk.delay(str(video_event.video_event_id))
        
        return str(video_event.video_event_id)
    
    def update_metadata(self, robot_id: str, metadata: VideoChunkMetadata):
        """Update metadata for a robot's buffer."""
        if robot_id in self.buffers:
            self.buffers[robot_id].update_metadata(metadata)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/video/{robot_id}")
async def video_stream_endpoint(
    websocket: WebSocket,
    robot_id: str,
):
    """
    WebSocket endpoint for receiving real-time video frames from robots.
    
    ## Protocol
    
    1. **Connect**: Robot connects with robot_id in URL path
    2. **Authenticate**: Send JSON message: `{"type": "auth", "api_key": "your-key"}`
    3. **Optional Metadata**: Send `{"type": "metadata", "metadata": {...}}`
    4. **Stream**: Send binary frames (JPEG, H264 NAL units, etc.)
    5. **Keepalive**: Send `{"type": "ping"}` to receive `{"type": "pong"}`
    
    ## Frame Format
    
    Binary messages are treated as video frames. Supported formats:
    - JPEG frames (for compatibility)
    - H264 NAL units (recommended for efficiency)
    - Raw video container segments
    
    ## Server Responses
    
    - `{"type": "connected", "message": "..."}` - Connection accepted
    - `{"type": "authenticated", "message": "..."}` - Auth successful
    - `{"type": "chunk_created", "video_event_id": "...", "status": "processing"}` - Video chunk saved
    - `{"type": "pong"}` - Keepalive response
    - `{"type": "error", "error": "..."}` - Error message
    
    ## Example Client (Python)
    
    ```python
    import websockets
    import json
    
    async def stream_video(robot_id, api_key, frame_generator):
        uri = f"ws://localhost:8000/v1/ws/video/{robot_id}"
        
        async with websockets.connect(uri) as ws:
            # Authenticate
            await ws.send(json.dumps({"type": "auth", "api_key": api_key}))
            
            # Stream frames
            async for frame in frame_generator:
                await ws.send(frame)  # Binary frame data
    ```
    """
    # Get DB session for this connection
    db = SessionLocal()
    
    try:
        await manager.connect(websocket, robot_id)
        
        while True:
            # Receive data (can be binary frames or JSON control messages)
            message = await websocket.receive()
            
            if "bytes" in message:
                # Binary frame data
                frame_data = message["bytes"]
                
                if not manager.is_authenticated(robot_id):
                    await websocket.send_json(
                        WebSocketAck(
                            type="error",
                            error="Not authenticated. Send auth message first."
                        ).model_dump()
                    )
                    continue
                
                video_event_id = await manager.process_frame(robot_id, frame_data, db)
                
                if video_event_id:
                    # Acknowledge chunk was saved
                    await websocket.send_json(
                        WebSocketAck(
                            type="chunk_created",
                            video_event_id=video_event_id,
                            status="processing"
                        ).model_dump()
                    )
            
            elif "text" in message:
                # JSON control message
                try:
                    data = json.loads(message["text"])
                    msg_type = data.get("type", "")
                    
                    if msg_type == "auth":
                        api_key = data.get("api_key", "")
                        if await manager.authenticate(robot_id, api_key):
                            await websocket.send_json(
                                WebSocketAck(
                                    type="authenticated",
                                    message="Authentication successful. Begin streaming."
                                ).model_dump()
                            )
                        else:
                            await websocket.send_json(
                                WebSocketAck(
                                    type="error",
                                    error="Authentication failed. Invalid API key."
                                ).model_dump()
                            )
                    
                    elif msg_type == "metadata":
                        metadata_dict = data.get("metadata", {})
                        metadata_dict["robot_id"] = robot_id  # Ensure robot_id matches
                        metadata = VideoChunkMetadata(**metadata_dict)
                        manager.update_metadata(robot_id, metadata)
                        
                        await websocket.send_json(
                            WebSocketAck(
                                type="metadata_updated",
                                message="Metadata updated."
                            ).model_dump()
                        )
                    
                    elif msg_type == "ping":
                        await websocket.send_json(
                            WebSocketAck(type="pong").model_dump()
                        )
                    
                    elif msg_type == "flush":
                        # Force flush any buffered frames
                        video_event_id = await manager.flush_buffer(robot_id, db)
                        if video_event_id:
                            await websocket.send_json(
                                WebSocketAck(
                                    type="chunk_created",
                                    video_event_id=video_event_id,
                                    status="processing"
                                ).model_dump()
                            )
                    
                    else:
                        await websocket.send_json(
                            WebSocketAck(
                                type="error",
                                error=f"Unknown message type: {msg_type}"
                            ).model_dump()
                        )
                
                except json.JSONDecodeError:
                    await websocket.send_json(
                        WebSocketAck(
                            type="error",
                            error="Invalid JSON message"
                        ).model_dump()
                    )
    
    except WebSocketDisconnect:
        # Flush any remaining frames before disconnecting
        await manager.flush_buffer(robot_id, db)
        manager.disconnect(robot_id)
    
    except Exception as e:
        print(f"[VideoStream] Error for robot {robot_id}: {e}")
        await manager.flush_buffer(robot_id, db)
        manager.disconnect(robot_id)
    
    finally:
        db.close()


@router.get("/video/status/{robot_id}")
async def get_stream_status(robot_id: str):
    """
    Get the status of a video stream connection.
    
    Returns connection info if the robot is connected.
    """
    if robot_id in manager.active_connections:
        buffer = manager.buffers.get(robot_id)
        return {
            "robot_id": robot_id,
            "connected": True,
            "authenticated": manager.is_authenticated(robot_id),
            "buffered_frames": buffer.frame_count if buffer else 0,
            "buffer_start_time": buffer.start_time.isoformat() if buffer and buffer.start_time else None
        }
    
    return {
        "robot_id": robot_id,
        "connected": False
    }

