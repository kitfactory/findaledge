"""
Embedding model factory for LangChain embeddings
LangChainのEmbeddingモデルのファクトリ

This module provides a factory class for creating various LangChain embedding models.
このモジュールは様々なLangChainの埋め込みモデルを作成するファクトリクラスを提供します。
"""

import os
from typing import Optional, Dict, Any, Union
from enum import Enum, auto

from langchain.embeddings.base import Embeddings
from langchain_community.embeddings import OpenAIEmbeddings, OllamaEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

class ModelProvider(Enum): # Enum名を変更
    """
    Types of embedding model providers supported
    サポートされている埋め込みモデルプロバイダーの種類
    """
    OPENAI = auto()
    OLLAMA = auto()

class EmbeddingModelFactory:
    """
    Factory class for creating embedding models
    埋め込みモデルを作成するファクトリクラス
    """

    @staticmethod
    def create_embeddings(
        model_provider: Optional[Union[ModelProvider, str]] = None, # Allow str input
        model_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs: Any
    ) -> Embeddings:
        """
        Create an embedding model instance
        埋め込みモデルのインスタンスを作成

        Args:
            model_provider (Optional[Union[ModelProvider, str]], optional): Type of embedding model provider to create.
                If None, reads from FINDERLEDGE_MODEL_PROVIDER environment variable (default: 'openai').
                作成する埋め込みモデルプロバイダーの種類。Noneの場合、環境変数 FINDERLEDGE_MODEL_PROVIDER
                から読み込みます (デフォルト: 'openai').
            model_name (Optional[str], optional): Name of the specific model to use.
                For Ollama, this could be 'llama2' etc. Defaults to None.
                使用する特定のモデル名。Ollamaの場合、'llama2'などが指定可能。デフォルトはNone。
            cache_dir (Optional[str], optional): Directory to cache embeddings. If provided,
                embeddings will be cached. Defaults to None.
                埋め込みをキャッシュするディレクトリ。指定された場合、埋め込みがキャッシュされます。
                デフォルトはNone。
            **kwargs: Additional arguments for the embedding model
                埋め込みモデルの追加引数

        Returns:
            Embeddings: An instance of LangChain Embeddings
                LangChain Embeddingsのインスタンス

        Raises:
            ValueError: If model_provider is not supported or required parameters are missing
                model_providerがサポートされていないか、必要なパラメータが不足している場合
        """
        # Load environment variables if not already loaded (optional but good practice)
        # oneenv.load() # <- 削除

        provider_to_use: ModelProvider
        if model_provider is None:
            # Determine provider from environment variable
            provider_str = os.getenv("FINDERLEDGE_MODEL_PROVIDER", "openai").upper()
            try:
                provider_to_use = ModelProvider[provider_str]
            except KeyError:
                raise ValueError(
                    f"Unsupported model provider in environment variable FINDERLEDGE_MODEL_PROVIDER: {provider_str}. "
                    f"Supported: {[p.name for p in ModelProvider]}."
                )
        elif isinstance(model_provider, str):
            # Determine provider from string input
            provider_str = model_provider.upper()
            try:
                provider_to_use = ModelProvider[provider_str]
            except KeyError:
                raise ValueError(
                    f"Unsupported model provider string: {model_provider}. "
                    f"Supported: {[p.name for p in ModelProvider]}."
                )
        elif isinstance(model_provider, ModelProvider):
            # Provider is already an enum
            provider_to_use = model_provider
        else:
             # Handle unexpected type for model_provider
            raise TypeError(f"Invalid type for model_provider: {type(model_provider)}. Expected ModelProvider enum or str.")

        # Now call _create_base_embeddings with the determined ModelProvider enum
        base_embeddings = EmbeddingModelFactory._create_base_embeddings(
            model_provider=provider_to_use,
            model_name=model_name,
            **kwargs
        )

        # Wrap with cache if cache_dir is provided
        if cache_dir:
            store = LocalFileStore(cache_dir)
            namespace_str = (
                base_embeddings.model
                if hasattr(base_embeddings, 'model')
                else f"{provider_to_use.name}_{model_name or 'default'}"
            )
            return CacheBackedEmbeddings.from_bytes_store(
                base_embeddings,
                store,
                namespace=namespace_str
            )

        return base_embeddings

    @staticmethod
    def _create_base_embeddings(
        model_provider: ModelProvider, # Expects enum now
        model_name: Optional[str] = None,
        **kwargs: Any
    ) -> Embeddings:
        """
        Create base embedding model without caching
        キャッシュなしの基本埋め込みモデルを作成

        Args:
            model_provider (ModelProvider): Type of embedding model provider
                埋め込みモデルプロバイダーの種類
            model_name (Optional[str], optional): Name of the specific model
                特定のモデル名
            **kwargs: Additional arguments for the embedding model
                埋め込みモデルの追加引数

        Returns:
            Embeddings: Base embedding model instance
                基本埋め込みモデルのインスタンス

        Raises:
            ValueError: If model_provider is not supported
                model_providerがサポートされていない場合
        """
        # Load necessary API keys/URLs from environment variables if not provided in kwargs
        # oneenv.load() # <- 削除

        if model_provider == ModelProvider.OPENAI:
            # Allow overriding via kwargs, otherwise use environment variables
            final_kwargs = {
                'openai_api_key': os.getenv('OPENAI_API_KEY'),
                'base_url': os.getenv('OPENAI_BASE_URL'), # Use OPENAI_BASE_URL
                'model': model_name or os.getenv('FINDERLEDGE_EMBEDDING_MODEL_NAME') or 'text-embedding-3-small', # Use FINDERLEDGE_EMBEDDING_MODEL_NAME
                **kwargs
            }
            # Remove None values so Langchain uses its defaults if necessary
            final_kwargs = {k: v for k, v in final_kwargs.items() if v is not None}

            if not final_kwargs.get('openai_api_key'):
                raise ValueError(
                    "OpenAI API key is required. Provide it via OPENAI_API_KEY environment variable or openai_api_key argument."
                )
            return OpenAIEmbeddings(**final_kwargs)

        elif model_provider == ModelProvider.OLLAMA:
            final_kwargs = {
                'base_url': os.getenv('OLLAMA_BASE_URL') or 'http://localhost:11434', # Use OLLAMA_BASE_URL
                'model': model_name or os.getenv('FINDERLEDGE_EMBEDDING_MODEL_NAME') or 'llama2', # Use FINDERLEDGE_EMBEDDING_MODEL_NAME
                **kwargs
            }
            # Remove None values
            final_kwargs = {k: v for k, v in final_kwargs.items() if v is not None}
            return OllamaEmbeddings(**final_kwargs)

        raise ValueError(f"Unsupported model provider: {model_provider}")

# 使用例 / Usage examples:
"""
# 環境変数 FINDERLEDGE_MODEL_PROVIDER に従い、キャッシュ付きで作成
# (事前に oneenv.load() が実行されているか、.env ファイルがある想定)
embeddings = EmbeddingModelFactory.create_embeddings(
    cache_dir="./cache/embeddings"
)

# 明示的にOpenAIを指定し、キャッシュ付きで作成
openai_embeddings = EmbeddingModelFactory.create_embeddings(
    model_provider=ModelProvider.OPENAI,
    cache_dir="./cache/embeddings"
    # APIキーは環境変数 OPENAI_API_KEY から読み込まれる想定
)

# 明示的にOllamaを指定し、特定のモデル名で作成
ollama_embeddings = EmbeddingModelFactory.create_embeddings(
    model_provider=ModelProvider.OLLAMA,
    model_name="mistral",
    cache_dir="./cache/embeddings"
    # ベースURLは環境変数 OLLAMA_BASE_URL から読み込まれる想定
)

# キャッシュなしでOpenAIの特定のモデルを作成（APIキーは引数で渡す）
openai_specific_model = EmbeddingModelFactory.create_embeddings(
    model_provider=ModelProvider.OPENAI,
    model_name="text-embedding-3-large",
    openai_api_key="your-explicit-key"
)
""" 