"""Video embedding service using Twelve Labs API."""
import os
import asyncio
from typing import Optional, Dict, Any, List
import httpx

from backend.config import get_settings

settings = get_settings()


class TwelveLabsService:
    """
    Service for video processing via Twelve Labs.
    
    Provides:
    - Video upload (file or URL)
    - Semantic video search
    - Transcript extraction
    - Scene description generation
    """
    
    BASE_URL = "https://api.twelvelabs.io/v1.2"
    
    def __init__(self):
        self.api_key = settings.twelve_labs_api_key
        self.index_name = settings.twelve_labs_index_name
        self._index_id: Optional[str] = None
        self._index_lock = asyncio.Lock()
    
    @property
    def headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }
    
    async def _get_index_id(self) -> str:
        """Get or create index ID (thread-safe)."""
        if self._index_id:
            return self._index_id
        
        async with self._index_lock:
            # Double-check after acquiring lock
            if self._index_id:
                return self._index_id
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Check existing indexes
                resp = await client.get(f"{self.BASE_URL}/indexes", headers=self.headers)
                if resp.status_code == 200:
                    for idx in resp.json().get("data", []):
                        if idx.get("name") == self.index_name:
                            self._index_id = idx.get("_id")
                            if self._index_id:
                                return self._index_id
                
                # Create new index
                resp = await client.post(
                    f"{self.BASE_URL}/indexes",
                    headers=self.headers,
                    json={
                        "name": self.index_name,
                        "engines": [{"name": "marengo2.6", "options": ["visual", "audio"]}]
                    }
                )
                if resp.status_code == 201:
                    self._index_id = resp.json().get("_id")
                    if self._index_id:
                        return self._index_id
                
                raise Exception(f"Failed to create index: {resp.text}")
    
    async def upload_video(self, video_path: str) -> str:
        """Upload a local video file. Returns task_id."""
        index_id = await self._get_index_id()
        
        async with httpx.AsyncClient(timeout=600.0) as client:
            with open(video_path, "rb") as f:
                resp = await client.post(
                    f"{self.BASE_URL}/tasks",
                    headers={"x-api-key": self.api_key},
                    files={"video_file": (os.path.basename(video_path), f, "video/mp4")},
                    data={"index_id": index_id}
                )
                if resp.status_code == 201:
                    return resp.json()["_id"]
                raise Exception(f"Upload failed: {resp.text}")
    
    async def upload_video_from_url(self, video_url: str) -> str:
        """Upload video from URL. Returns task_id."""
        index_id = await self._get_index_id()
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.BASE_URL}/tasks/external-provider",
                headers=self.headers,
                json={"index_id": index_id, "url": video_url}
            )
            if resp.status_code == 201:
                return resp.json()["_id"]
            raise Exception(f"URL upload failed: {resp.text}")
    
    async def wait_for_task(self, task_id: str, max_wait: int = 600) -> Dict[str, Any]:
        """Wait for task completion. Returns task info with video_id."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            for _ in range(max_wait // 5):
                resp = await client.get(f"{self.BASE_URL}/tasks/{task_id}", headers=self.headers)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "ready":
                        return data
                    if data.get("status") == "failed":
                        raise Exception(f"Task failed: {data.get('error')}")
                await asyncio.sleep(5)
        raise Exception("Task timed out")
    
    async def search_by_text(
        self,
        query: str,
        search_options: List[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search videos by natural language query."""
        index_id = await self._get_index_id()
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.BASE_URL}/search",
                headers=self.headers,
                json={
                    "index_id": index_id,
                    "query": query,
                    "search_options": search_options or ["visual", "audio"],
                    "limit": limit
                }
            )
            if resp.status_code == 200:
                return resp.json().get("data", [])
            return []
    
    async def get_video_transcription(self, video_id: str) -> Optional[str]:
        """Get transcript from video."""
        index_id = await self._get_index_id()
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(
                f"{self.BASE_URL}/indexes/{index_id}/videos/{video_id}/transcription",
                headers=self.headers
            )
            if resp.status_code == 200:
                segments = [s.get("value", "") for s in resp.json().get("data", [])]
                return " ".join(segments) if segments else None
            return None
    
    async def generate_gist(self, video_id: str, types: List[str] = None) -> Optional[Dict]:
        """Generate summary/gist of video."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.BASE_URL}/generate",
                headers=self.headers,
                json={"video_id": video_id, "types": types or ["topic", "title"]}
            )
            if resp.status_code == 200:
                return resp.json()
            return None
    
    async def delete_video(self, video_id: str) -> bool:
        """Delete video from index."""
        index_id = await self._get_index_id()
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.delete(
                f"{self.BASE_URL}/indexes/{index_id}/videos/{video_id}",
                headers=self.headers
            )
            return resp.status_code == 204


# Singleton
_service: Optional[TwelveLabsService] = None


def get_twelve_labs_service() -> TwelveLabsService:
    global _service
    if _service is None:
        _service = TwelveLabsService()
    return _service
