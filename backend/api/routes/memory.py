"""Memory query endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

from backend.db.database import get_db
from backend.db.models import VideoEvent
from backend.schemas.memory import (
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryAnswerRequest,
    MemoryAnswerResponse
)
from backend.schemas.event import EventDetail
from backend.services.vector_store import VectorStoreService
from backend.services.llm import get_llm_service
from backend.api.dependencies import verify_api_key
from backend.config import get_settings

settings = get_settings()
router = APIRouter(prefix="/v1/memory", tags=["memory"])


# === Schemas for unified memory retrieval ===

class MemoryClip(BaseModel):
    """A video memory clip."""
    memory_id: str
    timestamp: datetime
    description: Optional[str] = None
    transcript: Optional[str] = None
    confidence: float = 0.0


class MemoryEvent(BaseModel):
    """A text/action event."""
    timestamp: datetime
    type: str
    text: Optional[str] = None
    source: str


class MemoryContext(BaseModel):
    """Rich memory context returned from retrieval."""
    clips: List[MemoryClip] = []
    events: List[MemoryEvent] = []
    objects: List[str] = []
    summary: Optional[str] = None


class RetrieveMemoryRequest(BaseModel):
    """Request to retrieve memories."""
    robot_id: str
    query: str
    user_id: Optional[str] = None
    location: Optional[str] = None
    time_from: Optional[datetime] = None
    time_to: Optional[datetime] = None
    limit: int = 10
    include_summary: bool = False


class RetrieveMemoryResponse(BaseModel):
    """Response with rich memory context."""
    query: str
    context: MemoryContext


@router.post("/search-events", response_model=MemorySearchResponse)
async def search_events(
    request: MemorySearchRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Search for events using semantic similarity.
    
    This is the RAG primitive - returns relevant events that can be
    used as context for LLM queries.
    """
    vector_store = VectorStoreService(db)
    
    # Extract filters
    time_from = None
    time_to = None
    sources = None
    types = None
    
    if request.filters:
        time_from = request.filters.time_from
        time_to = request.filters.time_to
        sources = request.filters.sources
        types = request.filters.types
    
    # Search for similar events
    results = vector_store.search_similar_events(
        query_text=request.query,
        robot_id=request.robot_id,
        user_id=request.user_id,
        time_from=time_from,
        time_to=time_to,
        sources=sources,
        types=types,
        limit=request.limit
    )
    
    # Convert to response format
    items = []
    for result in results:
        item = EventDetail(
            event_id=result["event_id"],
            robot_id=result["robot_id"],
            user_id=result["user_id"],
            timestamp=result["timestamp"],
            source=result["source"],
            type=result["type"],
            text=result["text"],
            metadata=result["metadata"] if request.include_metadata else None,
            score=result["score"]
        )
        items.append(item)
    
    return MemorySearchResponse(items=items)


@router.post("/answer", response_model=MemoryAnswerResponse)
async def get_memory_answer(
    request: MemoryAnswerRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Get an LLM-generated answer based on memory.
    
    This endpoint:
    1. Searches for relevant events
    2. Feeds them to an LLM
    3. Returns a structured answer with supporting evidence
    """
    vector_store = VectorStoreService(db)
    llm_service = get_llm_service()
    
    # Extract time window
    time_from = None
    time_to = None
    if request.time_window:
        time_from = request.time_window.from_
        time_to = request.time_window.to
    
    # Search for relevant events
    events = vector_store.search_similar_events(
        query_text=request.question,
        robot_id=request.robot_id,
        user_id=request.user_id,
        time_from=time_from,
        time_to=time_to,
        limit=request.max_context_events
    )
    
    # Generate answer using LLM
    result = llm_service.generate_answer(
        question=request.question,
        events=events
    )
    
    return MemoryAnswerResponse(
        answer=result["answer"],
        confidence=result["confidence"],
        supporting_events=result["supporting_events"]
    )


@router.post("/retrieve", response_model=RetrieveMemoryResponse)
async def retrieve_memory(
    request: RetrieveMemoryRequest,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """
    Retrieve memories matching a query.
    
    Returns rich context including:
    - **clips**: Relevant video moments from robot's camera
    - **events**: Related text events (speech, actions)
    - **objects**: Detected objects across all memories
    - **summary**: Optional LLM-generated summary
    
    Example queries:
    - "Where did I put my keys?"
    - "What did we talk about yesterday?"
    - "Show me when someone entered the room"
    """
    vector_store = VectorStoreService(db)
    
    # 1. Search text events
    text_events = vector_store.search_similar_events(
        query_text=request.query,
        robot_id=request.robot_id,
        user_id=request.user_id,
        time_from=request.time_from,
        time_to=request.time_to,
        limit=request.limit
    )
    
    # 2. Search video memories via Twelve Labs
    clips = []
    objects_set = set()
    
    if settings.enable_video_processing and settings.twelve_labs_api_key:
        from backend.services.video_embedding import get_twelve_labs_service
        service = get_twelve_labs_service()
        
        try:
            tl_results = await service.search_by_text(
                query=request.query,
                search_options=["visual", "audio"],
                limit=request.limit
            )
            
            # Match with our database
            for tl_result in tl_results:
                video_id = tl_result.get("video_id")
                if not video_id:
                    continue
                
                video_event = db.query(VideoEvent).filter(
                    VideoEvent.twelve_labs_video_id == video_id,
                    VideoEvent.robot_id == request.robot_id
                ).first()
                
                if video_event:
                    clips.append(MemoryClip(
                        memory_id=str(video_event.video_event_id),
                        timestamp=video_event.start_timestamp,
                        description=video_event.scene_description,
                        transcript=video_event.transcript,
                        confidence=tl_result.get("confidence", 0)
                    ))
                    
                    # Extract objects from metadata
                    if video_event.metadata and video_event.metadata.get("objects"):
                        objects_set.update(video_event.metadata["objects"])
        except Exception as e:
            print(f"[Memory] Video search error: {e}")
    
    # 3. Build events list
    events = [
        MemoryEvent(
            timestamp=e["timestamp"],
            type=e["type"],
            text=e["text"],
            source=e["source"]
        )
        for e in text_events
    ]
    
    # 4. Extract objects from events
    for e in text_events:
        if e.get("metadata") and e["metadata"].get("objects"):
            objects_set.update(e["metadata"]["objects"])
    
    # 5. Optional: Generate summary
    summary = None
    if request.include_summary and (clips or events):
        try:
            llm_service = get_llm_service()
            context_text = "\n".join([
                f"- {e['timestamp']}: {e['text']}" for e in text_events if e.get('text')
            ])
            for clip in clips:
                if clip.transcript:
                    context_text += f"\n- {clip.timestamp}: [Video] {clip.transcript}"
            
            result = llm_service.generate_answer(
                question=f"Summarize these memories related to: {request.query}",
                events=text_events
            )
            summary = result.get("answer")
        except:
            pass
    
    return RetrieveMemoryResponse(
        query=request.query,
        context=MemoryContext(
            clips=clips,
            events=events,
            objects=list(objects_set),
            summary=summary
        )
    )

