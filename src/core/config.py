"""Application configuration and constants."""
from pathlib import Path
from dataclasses import dataclass, field


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class ModelPaths:
    """Paths to AI model files."""
    llm: str = "models/Qwen3VL-4B-Instruct-Q4_K_M.gguf"
    embedding: str = "models/bge-m3-Q8_0.gguf"
    reranker: str = "models/bge-reranker-v2-m3-Q6_K.gguf"
    
    def get_llm_path(self) -> str:
        return str(PROJECT_ROOT / self.llm)
    
    def get_embedding_path(self) -> str:
        return str(PROJECT_ROOT / self.embedding)
    
    def get_reranker_path(self) -> str:
        return str(PROJECT_ROOT / self.reranker)


@dataclass
class Config:
    """Application configuration."""
    # Directory paths
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    uploads_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "uploads")
    history_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "history")
    corpus_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "corpus")
    artifacts_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts")
    
    # Data files
    corpus_file: str = field(default_factory=lambda: str(PROJECT_ROOT / "data" / "corpus" / "corpus.json"))
    chat_history_file: str = field(default_factory=lambda: str(PROJECT_ROOT / "data" / "history" / "chat_history.json"))
    
    # Model paths
    models: ModelPaths = field(default_factory=ModelPaths)
    
    # LLM settings
    llm_context_length: int = 12288
    llm_batch_size: int = 3072
    llm_ubatch_size: int = 512
    
    # Retrieval settings
    retrieval_top_k: int = 5
    rrf_alpha: int = 60
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    def __post_init__(self):
        """Ensure directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()

