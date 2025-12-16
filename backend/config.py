"""Configuration management for MemoBot."""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings."""
    
    # Database
    database_url: str = "postgresql://postgres:password@localhost:5432/memobot"
    
    # OpenAI
    openai_api_key: str = ""
    
    # Embedding configuration
    use_local_embeddings: bool = False
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384  # for all-MiniLM-L6-v2
    
    # Twelve Labs Configuration
    twelve_labs_api_key: str = ""
    twelve_labs_index_id: str = ""  # Pre-created index for video embeddings
    twelve_labs_index_name: str = "memobot-videos"  # Name for auto-created index
    
    # Video Processing
    video_frame_buffer_size: int = 30  # Buffer frames before processing
    video_chunk_duration_seconds: int = 5  # Process video in 5-second chunks
    video_temp_storage_path: str = "/tmp/memobot/videos"
    video_embedding_dimension: int = 1024  # Marengo-retrieval-2.6 dimension
    
    # Graph Database (future)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    enable_graph_db: bool = False  # Disabled by default
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    
    # API Security
    api_secret_key: str = "your-secret-key-change-this-in-production"
    api_algorithm: str = "HS256"
    api_access_token_expire_minutes: int = 30
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # Features
    enable_summarization: bool = True
    enable_profiles: bool = True
    enable_video_processing: bool = True
    summarization_batch_size: int = 100
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

