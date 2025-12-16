"""Configuration management for MemoBot."""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings."""
    
    # Database
    database_url: str = "postgresql://postgres:password@localhost:5432/memobot"
    redis_url: str = "redis://localhost:6379/0"
    
    # OpenAI (for text embeddings and LLM)
    openai_api_key: str = ""
    use_local_embeddings: bool = False
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Twelve Labs (for video processing)
    twelve_labs_api_key: str = ""
    twelve_labs_index_name: str = "memobot-videos"
    
    # Video Processing
    video_chunk_duration_seconds: int = 5
    video_temp_storage_path: str = "/tmp/memobot/videos"
    
    # API Security
    api_secret_key: str = "change-this-in-production"
    
    # Feature Flags
    enable_summarization: bool = True
    enable_profiles: bool = True
    enable_video_processing: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

