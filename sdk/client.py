"""MemoBot client SDK."""
import requests
from datetime import datetime
from typing import Optional, Dict, Any, List


class MemoBotClient:
    """
    Client for interacting with MemoBot API.
    
    Example:
        client = MemoBotClient("http://localhost:8000", "your-api-key")
        
        # Log an event
        client.log_speech(
            robot_id="robot-123",
            text="I don't like loud noises.",
            speaker="user",
            user_id="user-456"
        )
        
        # Query memory
        answer = client.ask_memory(
            robot_id="robot-123",
            question="What does this user like?"
        )
    """
    
    def __init__(self, api_url: str, api_key: str):
        """Initialize client."""
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
    
    # ==========================================================================
    # Event Logging
    # ==========================================================================
    
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
        """Log an event to memory."""
        payload = {"robot_id": robot_id, "source": source, "type": type}
        if text:
            payload["text"] = text
        if user_id:
            payload["user_id"] = user_id
        if timestamp:
            payload["timestamp"] = timestamp.isoformat()
        if metadata:
            payload["metadata"] = metadata
        
        resp = self.session.post(f"{self.api_url}/v1/events", json=payload)
        resp.raise_for_status()
        return resp.json()
    
    def log_speech(
        self,
        robot_id: str,
        text: str,
        speaker: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Log a speech event."""
        return self.log_event(
            robot_id=robot_id,
            source="speech",
            type="USER_SAID" if speaker == "user" else "ROBOT_SAID",
            text=text,
            user_id=user_id
        )
    
    def log_vision(
        self,
        robot_id: str,
        description: str,
        objects: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Log a vision event."""
        return self.log_event(
            robot_id=robot_id,
            source="vision",
            type="OBJECT_DETECTED",
            text=description,
            metadata={"objects": objects} if objects else None
        )
    
    # ==========================================================================
    # Memory Queries
    # ==========================================================================
    
    def search_memory(
        self,
        robot_id: str,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Search memory for relevant events."""
        payload = {"robot_id": robot_id, "query": query, "limit": limit}
        if user_id:
            payload["user_id"] = user_id
        
        resp = self.session.post(f"{self.api_url}/v1/memory/search-events", json=payload)
        resp.raise_for_status()
        return resp.json()
    
    def ask_memory(
        self,
        robot_id: str,
        question: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Ask a question and get an LLM-generated answer."""
        payload = {"robot_id": robot_id, "question": question}
        if user_id:
            payload["user_id"] = user_id
        
        resp = self.session.post(f"{self.api_url}/v1/memory/answer", json=payload)
        resp.raise_for_status()
        return resp.json()
    
    def get_profile(
        self,
        robot_id: str,
        entity_type: str,
        entity_id: str
    ) -> Dict[str, Any]:
        """Get profile for a user, location, or object."""
        resp = self.session.get(
            f"{self.api_url}/v1/memory/profile",
            params={"robot_id": robot_id, "entity_type": entity_type, "entity_id": entity_id}
        )
        resp.raise_for_status()
        return resp.json()
    
    # ==========================================================================
    # Video Memory
    # ==========================================================================
    
    def search_video_memory(
        self,
        robot_id: str,
        query: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Search video memories using natural language.
        
        Example queries: "person waving", "red cup on table"
        """
        resp = self.session.post(
            f"{self.api_url}/v1/memory/video/search",
            json={"robot_id": robot_id, "query": query, "limit": limit}
        )
        resp.raise_for_status()
        return resp.json()
    
    def upload_video(
        self,
        robot_id: str,
        video_path: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
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
    
    def upload_video_from_url(
        self,
        robot_id: str,
        video_url: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Upload a video from URL."""
        payload = {"robot_id": robot_id, "video_url": video_url}
        if user_id:
            payload["user_id"] = user_id
        
        resp = self.session.post(f"{self.api_url}/v1/memory/video/from-url", json=payload)
        resp.raise_for_status()
        return resp.json()
    
    def get_video_event(self, video_event_id: str) -> Dict[str, Any]:
        """Get details of a video event."""
        resp = self.session.get(f"{self.api_url}/v1/memory/video/{video_event_id}")
        resp.raise_for_status()
        return resp.json()
    
    def get_recent_videos(self, robot_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent video events for a robot."""
        resp = self.session.get(
            f"{self.api_url}/v1/memory/video/robot/{robot_id}/recent",
            params={"limit": limit}
        )
        resp.raise_for_status()
        return resp.json()
