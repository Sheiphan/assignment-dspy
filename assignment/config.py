"""
Configuration module for the Data Transformation System
"""

import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

@dataclass
class SystemConfig:
    """System configuration settings"""
    
    # OpenAI API Configuration
    openai_api_key: str = ""
    openai_llm_model: str = "gpt-4"
    openai_embedding_model: str = "text-embedding-3-small"
    
    # Processing Configuration
    confidence_threshold: float = 0.7
    max_chunk_size: int = 1000
    chunk_overlap: int = 100
    vector_search_top_k: int = 5
    
    # System Behavior
    log_level: str = "INFO"
    save_results: bool = True
    output_directory: str = "./results"
    
    # Performance Tuning
    max_concurrent_extractions: int = 5
    cache_embeddings: bool = True
    
    @classmethod
    def from_environment(cls) -> "SystemConfig":
        """Load configuration from environment variables"""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_llm_model=os.getenv("OPENAI_LLM_MODEL", "gpt-4"),
            openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.7")),
            max_chunk_size=int(os.getenv("MAX_CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
            vector_search_top_k=int(os.getenv("VECTOR_SEARCH_TOP_K", "5")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            save_results=os.getenv("SAVE_RESULTS", "true").lower() == "true",
            output_directory=os.getenv("OUTPUT_DIRECTORY", "./results"),
            max_concurrent_extractions=int(os.getenv("MAX_CONCURRENT_EXTRACTIONS", "5")),
            cache_embeddings=os.getenv("CACHE_EMBEDDINGS", "true").lower() == "true",
        )

# Global configuration instance
config = SystemConfig.from_environment() 