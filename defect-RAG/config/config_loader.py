"""Configuration loader for Defect RAG System."""
import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    """LLM configuration."""
    base_url: str = "http://localhost:8000/v1"
    api_key: str = ""
    model: str = "gpt-3.5-turbo"
    verify_ssl: bool = True
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


@dataclass
class EmbeddingConfig:
    """Embedding configuration."""
    base_url: str = "http://localhost:8000/v1"
    api_key: str = ""
    model: str = "text-embedding-ada-002"
    verify_ssl: bool = True
    vector_dimension: int = 1024
    batch_size: int = 32


@dataclass
class RetrievalConfig:
    """Retrieval configuration."""
    top_k: int = 5
    similarity_threshold: float = 0.7
    search_type: str = "similarity"
    enable_rerank: bool = True
    rerank_top_k: int = 10
    rerank_type: str = "simple"  # 'simple' or 'llm'


@dataclass
class VectorStoreConfig:
    """Vector store configuration."""
    persist_directory: str = "./vector_db"
    collection_name: str = "defects"


@dataclass
class DataConfig:
    """Data configuration."""
    default_data_path: str = "./data (6).json"
    text_fields: list = field(default_factory=lambda: ["Summary", "PreClarification"])
    metadata_fields: list = field(default_factory=list)


@dataclass
class UIConfig:
    """UI configuration."""
    default_language: str = "zh"
    page_title: str = "软件缺陷分析RAG系统"
    page_icon: str = "🔍"


class Config:
    """Main configuration class."""

    def __init__(self, config_path: str = None):
        """Initialize configuration.

        Args:
            config_path: Path to YAML config file. If None, uses default.
        """
        self.config_path = config_path or self._find_config_file()
        self._config_data = self._load_config()

        # Initialize sub-configs
        self.llm = self._init_llm_config()
        self.embedding = self._init_embedding_config()
        self.retrieval = self._init_retrieval_config()
        self.vector_store = self._init_vector_store_config()
        self.data = self._init_data_config()
        self.ui = self._init_ui_config()

    def _find_config_file(self) -> str:
        """Find configuration file."""
        # Check environment variable first
        env_path = os.getenv("DEFECT_RAG_CONFIG")
        if env_path and os.path.exists(env_path):
            return env_path

        # Default location
        default_path = Path(__file__).parent / "settings.yaml"
        if default_path.exists():
            return str(default_path)

        raise FileNotFoundError("Configuration file not found. Please create config/settings.yaml")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _init_llm_config(self) -> LLMConfig:
        """Initialize LLM configuration."""
        llm_data = self._config_data.get('llm', {})
        return LLMConfig(
            base_url=os.getenv('LLM_BASE_URL', llm_data.get('base_url', 'http://localhost:8000/v1')),
            api_key=os.getenv('LLM_API_KEY', llm_data.get('api_key', '')),
            model=os.getenv('LLM_MODEL', llm_data.get('model', 'gpt-3.5-turbo')),
            verify_ssl=llm_data.get('verify_ssl', True),
            temperature=llm_data.get('temperature', 0.7),
            max_tokens=llm_data.get('max_tokens', 2048),
            top_p=llm_data.get('top_p', 0.9),
            frequency_penalty=llm_data.get('frequency_penalty', 0.0),
            presence_penalty=llm_data.get('presence_penalty', 0.0)
        )

    def _init_embedding_config(self) -> EmbeddingConfig:
        """Initialize Embedding configuration."""
        emb_data = self._config_data.get('embedding', {})
        return EmbeddingConfig(
            base_url=os.getenv('EMBEDDING_BASE_URL', emb_data.get('base_url', 'http://localhost:8000/v1')),
            api_key=os.getenv('EMBEDDING_API_KEY', emb_data.get('api_key', '')),
            model=os.getenv('EMBEDDING_MODEL', emb_data.get('model', 'text-embedding-ada-002')),
            verify_ssl=emb_data.get('verify_ssl', True),
            vector_dimension=emb_data.get('vector_dimension', 1024),
            batch_size=emb_data.get('batch_size', 32)
        )

    def _init_retrieval_config(self) -> RetrievalConfig:
        """Initialize retrieval configuration."""
        ret_data = self._config_data.get('retrieval', {})
        return RetrievalConfig(
            top_k=ret_data.get('top_k', 5),
            similarity_threshold=ret_data.get('similarity_threshold', 0.7),
            search_type=ret_data.get('search_type', 'similarity'),
            enable_rerank=ret_data.get('enable_rerank', True),
            rerank_top_k=ret_data.get('rerank_top_k', 10),
            rerank_type=ret_data.get('rerank_type', 'simple')
        )

    def _init_vector_store_config(self) -> VectorStoreConfig:
        """Initialize vector store configuration."""
        vs_data = self._config_data.get('vector_store', {})
        return VectorStoreConfig(
            persist_directory=vs_data.get('persist_directory', './vector_db'),
            collection_name=vs_data.get('collection_name', 'defects')
        )

    def _init_data_config(self) -> DataConfig:
        """Initialize data configuration."""
        data_cfg = self._config_data.get('data', {})
        return DataConfig(
            default_data_path=data_cfg.get('default_data_path', './data (6).json'),
            text_fields=data_cfg.get('text_fields', ['Summary', 'PreClarification']),
            metadata_fields=data_cfg.get('metadata_fields', [])
        )

    def _init_ui_config(self) -> UIConfig:
        """Initialize UI configuration."""
        ui_data = self._config_data.get('ui', {})
        return UIConfig(
            default_language=ui_data.get('default_language', 'zh'),
            page_title=ui_data.get('page_title', '软件缺陷分析RAG系统'),
            page_icon=ui_data.get('page_icon', '🔍')
        )

    def get_llm_params(self, **overrides) -> Dict[str, Any]:
        """Get LLM parameters with optional overrides."""
        params = {
            'temperature': self.llm.temperature,
            'max_tokens': self.llm.max_tokens,
            'top_p': self.llm.top_p,
            'frequency_penalty': self.llm.frequency_penalty,
            'presence_penalty': self.llm.presence_penalty
        }
        params.update(overrides)
        return params


# Global config instance
_config_instance = None


def get_config(config_path: str = None) -> Config:
    """Get or create global configuration instance."""
    global _config_instance
    if _config_instance is None or config_path is not None:
        _config_instance = Config(config_path)
    return _config_instance


def reload_config(config_path: str = None) -> Config:
    """Reload configuration."""
    global _config_instance
    _config_instance = Config(config_path)
    return _config_instance
