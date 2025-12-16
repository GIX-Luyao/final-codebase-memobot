"""Video embedding service using Twelve Labs API."""
import os
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
import httpx

from backend.config import get_settings

settings = get_settings()


class TwelveLabsService:
    """Service for video embeddings using Twelve Labs API.
    
    Twelve Labs provides multimodal video understanding including:
    - Visual content analysis (objects, actions, scenes)
    - Audio/speech transcription
    - Text-in-video (OCR)
    - Semantic search across video content
    
    Uses Marengo-retrieval-2.6 engine for embedding generation.
    """
    
    BASE_URL = "https://api.twelvelabs.io/v1.2"
    
    def __init__(self):
        """Initialize Twelve Labs service."""
        self.api_key = settings.twelve_labs_api_key
        self.index_id = settings.twelve_labs_index_id
        self.index_name = settings.twelve_labs_index_name
        self._headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }
    
    @property
    def headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        return self._headers.copy()
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get headers without Content-Type for file uploads."""
        return {"x-api-key": self.api_key}
    
    async def create_index(self, index_name: Optional[str] = None) -> str:
        """
        Create a new Twelve Labs index for storing videos.
        
        Args:
            index_name: Optional name for the index (defaults to settings)
            
        Returns:
            Index ID
        """
        name = index_name or self.index_name
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.BASE_URL}/indexes",
                headers=self.headers,
                json={
                    "name": name,
                    "engines": [
                        {
                            "name": "marengo2.6",
                            "options": ["visual", "audio"]
                        }
                    ]
                }
            )
            
            if response.status_code == 201:
                result = response.json()
                return result["_id"]
            
            raise Exception(f"Failed to create index: {response.status_code} - {response.text}")
    
    async def get_or_create_index(self, index_name: Optional[str] = None) -> str:
        """
        Get existing index or create a new one.
        
        Args:
            index_name: Optional name for the index (defaults to settings)
            
        Returns:
            Index ID
        """
        name = index_name or self.index_name
        
        # First check if index_id is configured
        if self.index_id:
            return self.index_id
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # List existing indexes
            response = await client.get(
                f"{self.BASE_URL}/indexes",
                headers=self.headers
            )
            
            if response.status_code == 200:
                indexes = response.json().get("data", [])
                for idx in indexes:
                    if idx["name"] == name:
                        self.index_id = idx["_id"]
                        return idx["_id"]
            
            # Create new index if not found
            return await self.create_index(name)
    
    async def upload_video(
        self,
        video_path: str,
        metadata: Optional[Dict] = None,
        language: str = "en"
    ) -> str:
        """
        Upload a video file to Twelve Labs index.
        
        Args:
            video_path: Path to the video file
            metadata: Optional metadata to associate with the video
            language: Language code for transcription (default: en)
            
        Returns:
            Task ID for tracking upload progress
        """
        # Ensure we have an index
        index_id = await self.get_or_create_index()
        
        async with httpx.AsyncClient(timeout=600.0) as client:
            with open(video_path, "rb") as f:
                files = {"video_file": (os.path.basename(video_path), f, "video/mp4")}
                data = {
                    "index_id": index_id,
                    "language": language
                }
                
                response = await client.post(
                    f"{self.BASE_URL}/tasks",
                    headers=self._get_auth_headers(),
                    files=files,
                    data=data
                )
                
                if response.status_code == 201:
                    return response.json()["_id"]
                
                raise Exception(f"Failed to upload video: {response.status_code} - {response.text}")
    
    async def upload_video_from_url(
        self,
        video_url: str,
        metadata: Optional[Dict] = None,
        language: str = "en"
    ) -> str:
        """
        Upload a video from URL to Twelve Labs index.
        
        Args:
            video_url: URL of the video file
            metadata: Optional metadata
            language: Language code for transcription
            
        Returns:
            Task ID for tracking upload progress
        """
        index_id = await self.get_or_create_index()
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.BASE_URL}/tasks/external-provider",
                headers=self.headers,
                json={
                    "index_id": index_id,
                    "url": video_url,
                    "language": language
                }
            )
            
            if response.status_code == 201:
                return response.json()["_id"]
            
            raise Exception(f"Failed to upload video from URL: {response.status_code} - {response.text}")
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Check the status of a video upload/indexing task.
        
        Args:
            task_id: The task ID returned from upload
            
        Returns:
            Task status dict with 'status' field: 'pending', 'indexing', 'ready', 'failed'
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.BASE_URL}/tasks/{task_id}",
                headers=self.headers
            )
            
            if response.status_code == 200:
                return response.json()
            
            raise Exception(f"Failed to get task status: {response.status_code} - {response.text}")
    
    async def wait_for_task(
        self,
        task_id: str,
        max_attempts: int = 120,
        poll_interval: float = 5.0
    ) -> Dict[str, Any]:
        """
        Wait for a task to complete.
        
        Args:
            task_id: The task ID to wait for
            max_attempts: Maximum number of polling attempts
            poll_interval: Seconds between polls
            
        Returns:
            Final task status
            
        Raises:
            Exception if task fails or times out
        """
        for attempt in range(max_attempts):
            status = await self.get_task_status(task_id)
            task_status = status.get("status")
            
            if task_status == "ready":
                return status
            elif task_status == "failed":
                raise Exception(f"Task failed: {status.get('error', 'Unknown error')}")
            
            await asyncio.sleep(poll_interval)
        
        raise Exception(f"Task timed out after {max_attempts * poll_interval} seconds")
    
    async def get_video_embedding(
        self,
        video_id: str,
        start_offset_sec: Optional[float] = None,
        end_offset_sec: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get embedding for a video or video segment using Twelve Labs Embed API.
        
        Uses Marengo-retrieval-2.6 for multimodal embeddings.
        
        Args:
            video_id: The video ID in Twelve Labs index
            start_offset_sec: Optional start time for segment
            end_offset_sec: Optional end time for segment
            
        Returns:
            Dict with 'embedding' (list of floats) and metadata
        """
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "video_id": video_id,
                "model_name": "Marengo-retrieval-2.6"
            }
            
            if start_offset_sec is not None:
                payload["start_offset_sec"] = start_offset_sec
            if end_offset_sec is not None:
                payload["end_offset_sec"] = end_offset_sec
            
            response = await client.post(
                f"{self.BASE_URL}/embed",
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "embedding": result.get("video_embedding", {}).get("embedding"),
                    "segments": result.get("video_embedding", {}).get("segments", [])
                }
            
            return None
    
    async def create_text_embedding(self, text: str) -> Optional[List[float]]:
        """
        Create embedding for text query (for cross-modal search).
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.BASE_URL}/embed",
                headers=self.headers,
                json={
                    "text": text,
                    "model_name": "Marengo-retrieval-2.6"
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("text_embedding", {}).get("embedding")
            
            return None
    
    async def search_by_text(
        self,
        query: str,
        index_id: Optional[str] = None,
        search_options: Optional[List[str]] = None,
        limit: int = 10,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search videos by natural language query.
        
        Args:
            query: Natural language search query
            index_id: Optional index ID (defaults to configured index)
            search_options: List of search modalities: "visual", "audio", "text_in_video"
            limit: Maximum results to return
            threshold: Optional confidence threshold (0-100)
            
        Returns:
            List of search results with video_id, score, and metadata
        """
        idx = index_id or self.index_id or await self.get_or_create_index()
        options = search_options or ["visual", "audio"]
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "index_id": idx,
                "query": query,
                "search_options": options,
                "limit": limit
            }
            
            if threshold is not None:
                payload["threshold"] = threshold
            
            response = await client.post(
                f"{self.BASE_URL}/search",
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 200:
                return response.json().get("data", [])
            
            return []
    
    async def get_video_transcription(
        self,
        video_id: str,
        index_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Get transcription (speech-to-text) from a video.
        
        Args:
            video_id: The video ID
            index_id: Optional index ID
            
        Returns:
            Full transcript text or None
        """
        idx = index_id or self.index_id or await self.get_or_create_index()
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(
                f"{self.BASE_URL}/indexes/{idx}/videos/{video_id}/transcription",
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json().get("data", [])
                # Combine all transcript segments
                segments = [seg.get("value", "") for seg in data if seg.get("value")]
                return " ".join(segments) if segments else None
            
            return None
    
    async def get_video_text_in_video(
        self,
        video_id: str,
        index_id: Optional[str] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get OCR text detected in video frames.
        
        Args:
            video_id: The video ID
            index_id: Optional index ID
            
        Returns:
            List of detected text with timestamps
        """
        idx = index_id or self.index_id or await self.get_or_create_index()
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(
                f"{self.BASE_URL}/indexes/{idx}/videos/{video_id}/text-in-video",
                headers=self.headers
            )
            
            if response.status_code == 200:
                return response.json().get("data", [])
            
            return None
    
    async def generate_gist(
        self,
        video_id: str,
        types: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate summary/gist of video content.
        
        Args:
            video_id: The video ID
            types: Types of gist to generate: "topic", "hashtag", "title"
            
        Returns:
            Dict with generated content
        """
        gist_types = types or ["topic", "hashtag", "title"]
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.BASE_URL}/generate",
                headers=self.headers,
                json={
                    "video_id": video_id,
                    "types": gist_types
                }
            )
            
            if response.status_code == 200:
                return response.json()
            
            return None
    
    async def delete_video(
        self,
        video_id: str,
        index_id: Optional[str] = None
    ) -> bool:
        """
        Delete a video from the index.
        
        Args:
            video_id: The video ID to delete
            index_id: Optional index ID
            
        Returns:
            True if successful
        """
        idx = index_id or self.index_id or await self.get_or_create_index()
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.delete(
                f"{self.BASE_URL}/indexes/{idx}/videos/{video_id}",
                headers=self.headers
            )
            
            return response.status_code == 204


# Global service instance
_twelve_labs_service: Optional[TwelveLabsService] = None


def get_twelve_labs_service() -> TwelveLabsService:
    """Get or create the global Twelve Labs service instance."""
    global _twelve_labs_service
    if _twelve_labs_service is None:
        _twelve_labs_service = TwelveLabsService()
    return _twelve_labs_service

