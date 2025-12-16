"""Main FastAPI application for MemoBot."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.db.database import init_db
from backend.api.routes import events, memory, profiles, video_stream, video_memory
from backend.config import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the application."""
    # Startup
    print("Initializing database...")
    init_db()
    print("Database initialized!")
    
    # Initialize video storage directory
    if settings.enable_video_processing:
        import os
        os.makedirs(settings.video_temp_storage_path, exist_ok=True)
        print(f"Video storage initialized at: {settings.video_temp_storage_path}")
    
    yield
    
    # Shutdown
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="MemoBot API",
    description="""
    Memory Layer for Robots - Semantic memory storage and retrieval.
    
    ## Features
    
    - **Event Ingestion**: Store observations, speech, actions, and other events
    - **Semantic Search**: Find relevant memories using natural language
    - **Video Processing**: Real-time video streaming with multimodal embeddings (Twelve Labs)
    - **Profile Management**: Persistent knowledge about users, locations, objects
    
    ## Video Streaming
    
    Use the WebSocket endpoint `/v1/ws/video/{robot_id}` for real-time video streaming.
    See the endpoint documentation for the protocol details.
    """,
    version="0.2.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(events.router)
app.include_router(memory.router)
app.include_router(profiles.router)
app.include_router(video_stream.router)  # WebSocket for video streaming
app.include_router(video_memory.router)  # Video search and management


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "MemoBot API",
        "version": "0.2.0",
        "status": "online",
        "docs": "/docs",
        "features": {
            "events": True,
            "memory_search": True,
            "profiles": True,
            "video_processing": settings.enable_video_processing,
            "video_websocket": "/v1/ws/video/{robot_id}"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "video_processing_enabled": settings.enable_video_processing,
        "twelve_labs_configured": bool(settings.twelve_labs_api_key)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

