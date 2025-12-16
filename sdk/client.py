"""MemoBot client SDK."""
import asyncio
import json
import requests
from datetime import datetime
from typing import Optional, Dict, Any, List, AsyncGenerator, Callable


class MemoBotClient:
    """
    Client for interacting with MemoBot API.
    
    Example usage:
        client = MemoBotClient(
            api_url="https://api.memobot.ai",
            api_key="your-api-key"
        )
        
        # Log an event
        client.log_event(
            robot_id="robot-123",
            source="speech",
            type="USER_SAID",
            text="I don't like loud noises.",
            user_id="user-456"
        )
        
        # Query memory
        answer = client.ask_memory(
            robot_id="robot-123",
            question="What does this user like?",
            user_id="user-456"
        )
    """
    
    def __init__(self, api_url: str, api_key: str):
        """
        Initialize MemoBot client.
        
        Args:
            api_url: Base URL of the MemoBot API
            api_key: API authentication key
        """
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
    
    def log_event(
        self,
        robot_id: str,
        source: str,
        type: str,
        text: Optional[str] = None,
        user_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log a single event to memory.
        
        Args:
            robot_id: Unique identifier for the robot
            source: Event source (speech, vision, action, system)
            type: Event type (USER_SAID, ROBOT_SAID, etc.)
            text: Text content of the event
            user_id: User identifier (if known)
            timestamp: Event timestamp (defaults to now)
            metadata: Additional metadata
            
        Returns:
            Response with event_id and status
        """
        payload = {
            "robot_id": robot_id,
            "source": source,
            "type": type,
        }
        
        if text:
            payload["text"] = text
        if user_id:
            payload["user_id"] = user_id
        if timestamp:
            payload["timestamp"] = timestamp.isoformat()
        if metadata:
            payload["metadata"] = metadata
        
        response = self.session.post(
            f"{self.api_url}/v1/events",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def log_events_batch(
        self,
        events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Log multiple events in a single request.
        
        Args:
            events: List of event dictionaries
            
        Returns:
            Response with results for each event
        """
        response = self.session.post(
            f"{self.api_url}/v1/events/batch",
            json={"events": events}
        )
        response.raise_for_status()
        return response.json()
    
    def search_memory(
        self,
        robot_id: str,
        query: str,
        user_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Search memory for relevant events.
        
        Args:
            robot_id: Robot identifier
            query: Natural language query
            user_id: Optional user filter
            filters: Optional filters (time_from, time_to, sources, types)
            limit: Maximum results
            
        Returns:
            Response with matching events
        """
        payload = {
            "robot_id": robot_id,
            "query": query,
            "limit": limit
        }
        
        if user_id:
            payload["user_id"] = user_id
        if filters:
            payload["filters"] = filters
        
        response = self.session.post(
            f"{self.api_url}/v1/memory/search-events",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def ask_memory(
        self,
        robot_id: str,
        question: str,
        user_id: Optional[str] = None,
        time_window: Optional[Dict[str, str]] = None,
        max_context_events: int = 20
    ) -> Dict[str, Any]:
        """
        Ask a question and get an LLM-generated answer based on memory.
        
        Args:
            robot_id: Robot identifier
            question: Natural language question
            user_id: Optional user filter
            time_window: Optional time window (from, to)
            max_context_events: Maximum events for context
            
        Returns:
            Response with answer, confidence, and supporting events
        """
        payload = {
            "robot_id": robot_id,
            "question": question,
            "max_context_events": max_context_events
        }
        
        if user_id:
            payload["user_id"] = user_id
        if time_window:
            payload["time_window"] = time_window
        
        response = self.session.post(
            f"{self.api_url}/v1/memory/answer",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def get_profile(
        self,
        robot_id: str,
        entity_type: str,
        entity_id: str
    ) -> Dict[str, Any]:
        """
        Get profile for a user, location, or object.
        
        Args:
            robot_id: Robot identifier
            entity_type: Type of entity (user, location, object)
            entity_id: Entity identifier
            
        Returns:
            Profile with summary and facts
        """
        response = self.session.get(
            f"{self.api_url}/v1/memory/profile",
            params={
                "robot_id": robot_id,
                "entity_type": entity_type,
                "entity_id": entity_id
            }
        )
        response.raise_for_status()
        return response.json()
    
    # Convenience methods
    
    def log_speech(
        self,
        robot_id: str,
        text: str,
        speaker: str,
        user_id: Optional[str] = None,
        location: Optional[str] = None
    ) -> Dict[str, Any]:
        """Log a speech event (user or robot said something)."""
        event_type = "USER_SAID" if speaker == "user" else "ROBOT_SAID"
        metadata = {}
        if location:
            metadata["location"] = location
        
        return self.log_event(
            robot_id=robot_id,
            source="speech",
            type=event_type,
            text=text,
            user_id=user_id,
            metadata=metadata
        )
    
    def log_vision(
        self,
        robot_id: str,
        description: str,
        objects: Optional[List[str]] = None,
        location: Optional[str] = None
    ) -> Dict[str, Any]:
        """Log a vision event (robot saw something)."""
        metadata = {}
        if objects:
            metadata["objects"] = objects
        if location:
            metadata["location"] = location
        
        return self.log_event(
            robot_id=robot_id,
            source="vision",
            type="OBJECT_DETECTED",
            text=description,
            metadata=metadata
        )
    
    def log_action(
        self,
        robot_id: str,
        action: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Log a robot action."""
        return self.log_event(
            robot_id=robot_id,
            source="action",
            type=action,
            text=description,
            metadata=metadata
        )
    
    # Video Memory Methods
    
    def search_video_memory(
        self,
        robot_id: str,
        query: str,
        search_options: Optional[List[str]] = None,
        time_from: Optional[datetime] = None,
        time_to: Optional[datetime] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Search video memories using natural language.
        
        Args:
            robot_id: Robot identifier
            query: Natural language query (e.g., "person picking up a red cup")
            search_options: Search modalities: "visual", "audio", "text_in_video"
            time_from: Start of time range
            time_to: End of time range
            limit: Maximum results
            
        Returns:
            Response with matching video events
        """
        payload = {
            "robot_id": robot_id,
            "query": query,
            "limit": limit
        }
        
        if search_options:
            payload["search_options"] = search_options
        if time_from:
            payload["time_from"] = time_from.isoformat()
        if time_to:
            payload["time_to"] = time_to.isoformat()
        
        response = self.session.post(
            f"{self.api_url}/v1/memory/video/search",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def get_video_event(self, video_event_id: str) -> Dict[str, Any]:
        """
        Get details of a specific video event.
        
        Args:
            video_event_id: UUID of the video event
            
        Returns:
            Video event details including transcript, detected objects, etc.
        """
        response = self.session.get(
            f"{self.api_url}/v1/memory/video/{video_event_id}"
        )
        response.raise_for_status()
        return response.json()
    
    def get_recent_videos(
        self,
        robot_id: str,
        limit: int = 20,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent video events for a robot.
        
        Args:
            robot_id: Robot identifier
            limit: Maximum results
            status: Filter by processing status
            
        Returns:
            List of video events
        """
        params = {"limit": limit}
        if status:
            params["status"] = status
        
        response = self.session.get(
            f"{self.api_url}/v1/memory/video/robot/{robot_id}/recent",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    def upload_video(
        self,
        robot_id: str,
        video_path: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upload a video file for processing.
        
        Args:
            robot_id: Robot identifier
            video_path: Path to video file
            user_id: Optional user identifier
            
        Returns:
            Response with video_event_id and status
        """
        with open(video_path, 'rb') as f:
            files = {'video_file': f}
            data = {'robot_id': robot_id}
            if user_id:
                data['user_id'] = user_id
            
            # Remove Content-Type header for multipart upload
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            response = requests.post(
                f"{self.api_url}/v1/memory/video/upload",
                files=files,
                data=data,
                headers=headers
            )
        
        response.raise_for_status()
        return response.json()
    
    def get_video_stats(self, robot_id: str) -> Dict[str, Any]:
        """
        Get video processing statistics for a robot.
        
        Args:
            robot_id: Robot identifier
            
        Returns:
            Statistics including counts by status and total duration
        """
        response = self.session.get(
            f"{self.api_url}/v1/memory/video/stats/{robot_id}"
        )
        response.raise_for_status()
        return response.json()
    
    def create_video_stream_client(self, robot_id: str) -> "MemoBotVideoStreamClient":
        """
        Create a video streaming client for real-time video transmission.
        
        Args:
            robot_id: Robot identifier
            
        Returns:
            MemoBotVideoStreamClient instance
        """
        return MemoBotVideoStreamClient(
            api_url=self.api_url,
            api_key=self.api_key,
            robot_id=robot_id
        )


class MemoBotVideoStreamClient:
    """
    Client for streaming video to MemoBot via WebSocket.
    
    Example usage:
        async def stream_camera():
            client = MemoBotVideoStreamClient(
                api_url="http://localhost:8000",
                api_key="your-key",
                robot_id="robot-123"
            )
            
            async def frame_generator():
                import cv2
                cap = cv2.VideoCapture(0)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    _, buffer = cv2.imencode('.jpg', frame)
                    yield buffer.tobytes()
                    await asyncio.sleep(1/30)  # 30 FPS
            
            await client.stream_video(
                frame_generator(),
                on_chunk_created=lambda msg: print(f"Chunk: {msg['video_event_id']}")
            )
    """
    
    def __init__(self, api_url: str, api_key: str, robot_id: str):
        """
        Initialize video stream client.
        
        Args:
            api_url: Base URL of the MemoBot API
            api_key: API authentication key
            robot_id: Robot identifier
        """
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.robot_id = robot_id
        
        # Convert HTTP URL to WebSocket URL
        ws_url = self.api_url.replace("https://", "wss://").replace("http://", "ws://")
        self.ws_url = f"{ws_url}/v1/ws/video/{robot_id}"
        
        self.websocket = None
        self._connected = False
        self._authenticated = False
    
    async def connect(self) -> bool:
        """
        Establish WebSocket connection and authenticate.
        
        Returns:
            True if connected and authenticated successfully
        """
        try:
            import websockets
        except ImportError:
            raise ImportError("websockets package required. Install with: pip install websockets")
        
        self.websocket = await websockets.connect(self.ws_url)
        self._connected = True
        
        # Wait for connection confirmation
        msg = await self.websocket.recv()
        data = json.loads(msg)
        
        if data.get("type") != "connected":
            raise Exception(f"Unexpected response: {data}")
        
        # Authenticate
        await self.websocket.send(json.dumps({
            "type": "auth",
            "api_key": self.api_key
        }))
        
        msg = await self.websocket.recv()
        data = json.loads(msg)
        
        if data.get("type") == "authenticated":
            self._authenticated = True
            return True
        else:
            raise Exception(f"Authentication failed: {data.get('error', 'Unknown error')}")
    
    async def send_metadata(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        frame_rate: float = 30.0,
        resolution: Optional[str] = None
    ):
        """
        Send metadata about the video stream.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            frame_rate: Video frame rate
            resolution: Video resolution (e.g., "1920x1080")
        """
        if not self._connected:
            raise Exception("Not connected. Call connect() first.")
        
        metadata = {
            "robot_id": self.robot_id,
            "frame_rate": frame_rate
        }
        
        if user_id:
            metadata["user_id"] = user_id
        if session_id:
            metadata["session_id"] = session_id
        if resolution:
            metadata["resolution"] = resolution
        
        await self.websocket.send(json.dumps({
            "type": "metadata",
            "metadata": metadata
        }))
    
    async def send_frame(self, frame_data: bytes):
        """
        Send a single video frame.
        
        Args:
            frame_data: Binary frame data (JPEG, H264 NAL, etc.)
        """
        if not self._authenticated:
            raise Exception("Not authenticated. Call connect() first.")
        
        await self.websocket.send(frame_data)
    
    async def flush(self):
        """Force flush any buffered frames on the server."""
        if self._connected:
            await self.websocket.send(json.dumps({"type": "flush"}))
    
    async def ping(self) -> bool:
        """Send keepalive ping."""
        if not self._connected:
            return False
        
        await self.websocket.send(json.dumps({"type": "ping"}))
        msg = await self.websocket.recv()
        data = json.loads(msg)
        return data.get("type") == "pong"
    
    async def close(self):
        """Close the WebSocket connection."""
        if self.websocket:
            await self.flush()
            await self.websocket.close()
            self._connected = False
            self._authenticated = False
    
    async def stream_video(
        self,
        frame_generator: AsyncGenerator[bytes, None],
        on_chunk_created: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_error: Optional[Callable[[str], None]] = None
    ):
        """
        Stream video frames from an async generator.
        
        Args:
            frame_generator: Async generator yielding video frame bytes
            on_chunk_created: Callback when a video chunk is saved (receives message dict)
            on_error: Callback on error (receives error message)
            
        Example:
            async def camera_frames():
                import cv2
                cap = cv2.VideoCapture(0)
                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        _, buffer = cv2.imencode('.jpg', frame)
                        yield buffer.tobytes()
                        await asyncio.sleep(1/30)
                finally:
                    cap.release()
            
            await client.stream_video(camera_frames())
        """
        await self.connect()
        
        async def receive_responses():
            """Task to receive and process server responses."""
            try:
                async for message in self.websocket:
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type", "")
                        
                        if msg_type == "chunk_created" and on_chunk_created:
                            on_chunk_created(data)
                        elif msg_type == "error" and on_error:
                            on_error(data.get("error", "Unknown error"))
                        elif msg_type == "pong":
                            pass  # Keepalive response
                    except json.JSONDecodeError:
                        pass  # Ignore non-JSON messages
            except Exception:
                pass  # Connection closed
        
        # Start response receiver
        receive_task = asyncio.create_task(receive_responses())
        
        try:
            async for frame in frame_generator:
                await self.send_frame(frame)
        finally:
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass
            await self.close()
    
    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected
    
    @property
    def is_authenticated(self) -> bool:
        """Check if authenticated."""
        return self._authenticated

