"""
Embedding model factory for LangChain embeddings
LangChainのEmbeddingモデルのファクトリ

This module provides a factory class for creating various LangChain embedding models.
このモジュールは様々なLangChainの埋め込みモデルを作成するファクトリクラスを提供します。
"""

from typing import Optional, Dict, Any, Union
from enum import Enum, auto

from langchain.embeddings.base import Embeddings
from langchain_community.embeddings import OpenAIEmbeddings, OllamaEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

class EmbeddingModelType(Enum):
    """
    Types of embedding models supported
    サポートされている埋め込みモデルの種類
    """
    OPENAI = auto()  # OpenAI's text-embedding-ada-002
    OLLAMA = auto()  # Ollama's embeddings

class EmbeddingModelFactory:
    """
    Factory class for creating embedding models
    埋め込みモデルを作成するファクトリクラス
    """

    @staticmethod
    def create_embeddings(
        model_type: EmbeddingModelType,
        cache_dir: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs: Any
    ) -> Embeddings:
        """
        Create an embedding model instance
        埋め込みモデルのインスタンスを作成

        Args:
            model_type (EmbeddingModelType): Type of embedding model to create
                作成する埋め込みモデルの種類
            cache_dir (Optional[str], optional): Directory to cache embeddings. If provided,
                embeddings will be cached. Defaults to None.
                埋め込みをキャッシュするディレクトリ。指定された場合、埋め込みがキャッシュされます。
                デフォルトはNone。
            model_name (Optional[str], optional): Name of the specific model to use.
                For Ollama, this could be 'llama2' etc. Defaults to None.
                使用する特定のモデル名。Ollamaの場合、'llama2'などが指定可能。デフォルトはNone。
            **kwargs: Additional arguments for the embedding model
                埋め込みモデルの追加引数

        Returns:
            Embeddings: An instance of LangChain Embeddings
                LangChain Embeddingsのインスタンス

        Raises:
            ValueError: If model_type is not supported or required parameters are missing
                model_typeがサポートされていないか、必要なパラメータが不足している場合
        """
        # Create base embeddings based on model type
        base_embeddings = EmbeddingModelFactory._create_base_embeddings(
            model_type, model_name, **kwargs
        )

        # Wrap with cache if cache_dir is provided
        if cache_dir:
            store = LocalFileStore(cache_dir)
            return CacheBackedEmbeddings.from_bytes_store(
                base_embeddings,
                store,
                namespace=base_embeddings.model if hasattr(base_embeddings, 'model') else str(model_type)
            )

        return base_embeddings

    @staticmethod
    def _create_base_embeddings(
        model_type: EmbeddingModelType,
        model_name: Optional[str] = None,
        **kwargs: Any
    ) -> Embeddings:
        """
        Create base embedding model without caching
        キャッシュなしの基本埋め込みモデルを作成

        Args:
            model_type (EmbeddingModelType): Type of embedding model
                埋め込みモデルの種類
            model_name (Optional[str], optional): Name of the specific model
                特定のモデル名
            **kwargs: Additional arguments for the embedding model
                埋め込みモデルの追加引数

        Returns:
            Embeddings: Base embedding model instance
                基本埋め込みモデルのインスタンス

        Raises:
            ValueError: If model_type is not supported
                model_typeがサポートされていない場合
        """
        if model_type == EmbeddingModelType.OPENAI:
            if not kwargs.get('openai_api_key'):
                raise ValueError(
                    "OpenAI API key is required for OpenAI embeddings"
                    "OpenAIの埋め込みにはOpenAI APIキーが必要です"
                )
            return OpenAIEmbeddings(**kwargs)

        elif model_type == EmbeddingModelType.OLLAMA:
            model_kwargs = {
                'model': model_name or 'llama2',
                **kwargs
            }
            return OllamaEmbeddings(**model_kwargs)

        raise ValueError(
            f"Unsupported model type: {model_type}"
            f"サポートされていないモデルタイプです: {model_type}"
        )

# 使用例 / Usage examples:
"""
# OpenAIの埋め込みモデルを作成
openai_embeddings = EmbeddingModelFactory.create_embeddings(
    model_type=EmbeddingModelType.OPENAI,
    openai_api_key="your-api-key",
    cache_dir="./cache/embeddings"
)

# Ollamaの埋め込みモデルを作成（llama2を使用）
ollama_embeddings = EmbeddingModelFactory.create_embeddings(
    model_type=EmbeddingModelType.OLLAMA,
    model_name="llama2",
    cache_dir="./cache/embeddings"
)

# キャッシュなしでOpenAIの埋め込みモデルを作成
openai_embeddings_no_cache = EmbeddingModelFactory.create_embeddings(
    model_type=EmbeddingModelType.OPENAI,
    openai_api_key="your-api-key"
)
""" 