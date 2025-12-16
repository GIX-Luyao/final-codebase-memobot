"""MemoBot SDK - Memory layer for robots."""
import json
import asyncio
import base64
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
import requests


class MemoBotClient:
    """
    Client for interacting with MemoBot memory API.
    
    Example:
        client = MemoBotClient("http://localhost:8000", "your-api-key")
        
        # Store memory
        client.store(robot_id="robot-1", text="User said hello", type="USER_SAID")
        
        # Retrieve memory
        context = client.retrieve_memory(robot_id="robot-1", query="What did the user say?")
        print(context["clips"])   # Video clips
        print(context["events"])  # Text events
        print(context["objects"]) # Detected objects
    """
    
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
    
    # ==========================================================================
    # Memory Storage
    # ==========================================================================
    
    def store(
        self,
        robot_id: str,
        text: str,
        type: str,
        source: str = "speech",
        user_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Store a memory event.
        
        Args:
            robot_id: Robot identifier
            text: Event text content
            type: Event type (USER_SAID, ROBOT_SAID, OBJECT_DETECTED, etc.)
            source: Event source (speech, vision, action)
            user_id: Optional user identifier
            timestamp: Event timestamp (defaults to now)
            metadata: Additional metadata
        
        Returns:
            Created event details
        """
        payload = {"robot_id": robot_id, "text": text, "type": type, "source": source}
        if user_id:
            payload["user_id"] = user_id
        if timestamp:
            payload["timestamp"] = timestamp.isoformat()
        if metadata:
            payload["metadata"] = metadata
        
        resp = self.session.post(f"{self.api_url}/v1/events", json=payload)
        resp.raise_for_status()
        return resp.json()
    
    def log_speech(self, robot_id: str, text: str, speaker: str, user_id: str = None) -> Dict:
        """Log a speech event."""
        return self.store(
            robot_id=robot_id,
            text=text,
            type="USER_SAID" if speaker == "user" else "ROBOT_SAID",
            source="speech",
            user_id=user_id
        )
    
    def log_vision(self, robot_id: str, description: str, objects: List[str] = None) -> Dict:
        """Log a vision event."""
        return self.store(
            robot_id=robot_id,
            text=description,
            type="OBJECT_DETECTED",
            source="vision",
            metadata={"objects": objects} if objects else None
        )
    
    # ==========================================================================
    # Memory Retrieval
    # ==========================================================================
    
    def retrieve_memory(
        self,
        robot_id: str,
        query: str,
        user_id: Optional[str] = None,
        time_from: Optional[datetime] = None,
        time_to: Optional[datetime] = None,
        limit: int = 10,
        include_summary: bool = False
    ) -> Dict[str, Any]:
        """
        Retrieve memories matching a query.
        
        Returns rich context with clips, events, and objects.
        
        Args:
            robot_id: Robot identifier
            query: Natural language query (e.g., "Where did I put my keys?")
            user_id: Filter by user
            time_from: Start of time range
            time_to: End of time range
            limit: Maximum results
            include_summary: Include LLM-generated summary
        
        Returns:
            {
                "query": "...",
                "context": {
                    "clips": [...],    # Video clips with timestamp, description, confidence
                    "events": [...],   # Text events with timestamp, type, text
                    "objects": [...],  # Detected objects
                    "summary": "..."   # Optional summary
                }
            }
        """
        payload = {
            "robot_id": robot_id,
            "query": query,
            "limit": limit,
            "include_summary": include_summary
        }
        if user_id:
            payload["user_id"] = user_id
        if time_from:
            payload["time_from"] = time_from.isoformat()
        if time_to:
            payload["time_to"] = time_to.isoformat()
        
        resp = self.session.post(f"{self.api_url}/v1/memory/retrieve", json=payload)
        resp.raise_for_status()
        return resp.json()
    
    def ask(
        self,
        robot_id: str,
        question: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ask a question and get an LLM-generated answer.
        
        Args:
            robot_id: Robot identifier
            question: Question to answer based on memories
            user_id: Optional user context
        
        Returns:
            {"answer": "...", "confidence": 0.9, "supporting_events": [...]}
        """
        payload = {"robot_id": robot_id, "question": question}
        if user_id:
            payload["user_id"] = user_id
        
        resp = self.session.post(f"{self.api_url}/v1/memory/answer", json=payload)
        resp.raise_for_status()
        return resp.json()
    
    def get_profile(self, robot_id: str, entity_type: str, entity_id: str) -> Dict:
        """Get profile for a user, location, or object."""
        resp = self.session.get(
            f"{self.api_url}/v1/memory/profile",
            params={"robot_id": robot_id, "entity_type": entity_type, "entity_id": entity_id}
        )
        resp.raise_for_status()
        return resp.json()
    
    # ==========================================================================
    # Video Upload
    # ==========================================================================
    
    def upload_video(self, robot_id: str, video_path: str, user_id: str = None) -> Dict:
        """Upload a video file for processing."""
        with open(video_path, 'rb') as f:
            data = {'robot_id': robot_id}
            if user_id:
                data['user_id'] = user_id
            resp = requests.post(
                f"{self.api_url}/v1/memory/video/upload",
                files={'video_file': f},
                data=data,
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
        resp.raise_for_status()
        return resp.json()
    
    def create_stream_client(self, robot_id: str) -> "MemoBotStreamClient":
        """Create a client for streaming multimodal data."""
        return MemoBotStreamClient(self.api_url, self.api_key, robot_id)


class MemoBotStreamClient:
    """
    Client for streaming video segments to MemoBot.
    
    IMPORTANT: Send complete video segments (MP4/WebM), not raw frames.
    Use ffmpeg to create segments on the robot:
        ffmpeg -i /dev/video0 -c:v libx264 -f segment -segment_time 5 chunk_%03d.mp4
    
    Example:
        stream = client.create_stream_client("robot-123")
        await stream.connect(user_id="john")
        
        # Send pre-encoded video segments
        for chunk_path in glob.glob("chunk_*.mp4"):
            with open(chunk_path, "rb") as f:
                memory_id = await stream.send_segment(f.read())
                print(f"Stored: {memory_id}")
    """
    
    def __init__(self, api_url: str, api_key: str, robot_id: str):
        self.api_url = api_url
        self.api_key = api_key
        self.robot_id = robot_id
        
        ws_url = api_url.replace("https://", "wss://").replace("http://", "ws://")
        self.ws_url = f"{ws_url}/v1/ws/stream/{robot_id}"
        
        self.websocket = None
        self._authenticated = False
    
    async def connect(self, user_id: str = None, session_id: str = None):
        """Connect and authenticate."""
        try:
            import websockets
        except ImportError:
            raise ImportError("Install websockets: pip install websockets")
        
        self.websocket = await websockets.connect(self.ws_url)
        
        # Wait for connection
        msg = json.loads(await self.websocket.recv())
        if msg.get("type") != "connected":
            raise Exception(f"Connection failed: {msg}")
        
        # Authenticate
        await self.websocket.send(json.dumps({"type": "auth", "api_key": self.api_key}))
        msg = json.loads(await self.websocket.recv())
        if msg.get("type") != "authenticated":
            raise Exception(f"Auth failed: {msg}")
        
        self._authenticated = True
        
        # Set metadata
        if user_id or session_id:
            await self.websocket.send(json.dumps({
                "type": "metadata",
                "user_id": user_id,
                "session_id": session_id
            }))
            await self.websocket.recv()
    
    async def send_segment(self, video_data: bytes, action: str = None) -> str:
        """
        Send a complete video segment.
        
        Args:
            video_data: Complete video file bytes (MP4, WebM, etc.)
            action: Optional robot action during this segment
        
        Returns:
            memory_id of the stored segment
        """
        if not self._authenticated:
            raise Exception("Not connected. Call connect() first.")
        
        # Send as binary for efficiency
        await self.websocket.send(video_data)
        
        # Wait for ack
        msg = json.loads(await self.websocket.recv())
        if msg.get("type") == "ack_stored":
            return msg.get("memory_id")
        elif msg.get("type") == "error":
            raise Exception(f"Server error: {msg.get('error')}")
        else:
            raise Exception(f"Unexpected response: {msg}")
    
    async def record_action(self, action: str):
        """Record a robot action (without video)."""
        if not self._authenticated:
            raise Exception("Not connected. Call connect() first.")
        
        await self.websocket.send(json.dumps({"type": "action", "action": action}))
        await self.websocket.recv()
    
    async def close(self):
        """Close connection."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            self._authenticated = False
