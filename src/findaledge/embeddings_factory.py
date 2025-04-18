"""
Embedding model factory for LangChain embeddings
LangChainのEmbeddingモデルのファクトリ

This module provides a factory class for creating various LangChain embedding models.
このモジュールは様々なLangChainの埋め込みモデルを作成するファクトリクラスを提供します。
"""

import os
from typing import Optional, Dict, Any, Union
from enum import Enum, auto
import inspect
from pathlib import Path
import logging

from langchain.embeddings.base import Embeddings
from langchain_community.embeddings import OpenAIEmbeddings, OllamaEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
from typing import Type # Added for type hinting
# from oneenv import load # <- DELETED: Should be called at application entry point

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
    default_provider = ModelProvider.OPENAI
    default_model_name = {
        ModelProvider.OPENAI: "text-embedding-3-small",
        ModelProvider.OLLAMA: "llama2" # Example default for Ollama
    }

    def __init__(self):
        """
        Initialize the factory with an empty cache.
        ファクトリを空のキャッシュで初期化します。
        """
        self.cache: Dict[str, Embeddings] = {}

    def create_embeddings(
        self,
        provider: Optional[Union[ModelProvider, str]] = None,
        model_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Embeddings:
        """
        Creates or retrieves cached embedding model instances.
        埋め込みモデルのインスタンスを作成またはキャッシュから取得します。

        Parameter Priority:
        パラメータの優先順位:
        1. Explicit arguments to this method (provider, model_name, items in kwargs like api_key, openai_api_key, base_url).
           このメソッドへの明示的な引数（provider, model_name、api_key, openai_api_key, base_urlなどのkwargs内の項目）。
        2. Environment variables (FINDALEDGE_MODEL_PROVIDER, FINDALEDGE_EMBEDDING_MODEL_NAME, OPENAI_API_KEY, OPENAI_BASE_URL, OLLAMA_BASE_URL).
           環境変数。
        3. Factory defaults (default_provider, default_model_name map).
           ファクトリのデフォルト値。

        Args:
            provider: The embedding model provider (enum member or string name).
            model_name: The specific model name.
            cache_dir: Directory to cache embeddings. If provided, wraps the base embedding.
                       埋め込みをキャッシュするディレクトリ。指定された場合、基本埋め込みをラップします。
            model_kwargs: Additional keyword arguments for the specific model constructor.
                         特定のモデルコンストラクタのための追加キーワード引数。
            **kwargs: Additional arguments like `api_key`, `openai_api_key`, `base_url`.

        Returns:
            An instance of the requested Langchain Embeddings class (potentially cache-backed).
            要求されたLangchain Embeddingsクラスのインスタンス（キャッシュバックされる可能性あり）。

        Raises:
            ValueError: If the provider is unsupported, required API keys are missing,
                      or a model name cannot be determined.
                      プロバイダーがサポートされていない、必要なAPIキーがない、
                      またはモデル名を決定できない場合。
        """
        # Determine resolved provider and model name
        resolved_provider = provider if isinstance(provider, ModelProvider) else self.resolve_provider(
            provider or os.getenv('FINDALEDGE_MODEL_PROVIDER'))
        resolved_model_name = model_name or os.getenv('FINDALEDGE_EMBEDDING_MODEL_NAME') or self.default_model_name.get(resolved_provider)
        if not resolved_model_name:
             raise ValueError(f"Default model name not found for provider {resolved_provider} and no model_name provided.")

        # Generate a unique cache key based on provider, model, and critical kwargs
        # Add api_key presence to cache key if relevant (e.g., OpenAI)
        key_present_str = ""
        if resolved_provider == ModelProvider.OPENAI:
            api_key_present = bool(kwargs.get('openai_api_key') or kwargs.get('api_key') or os.getenv('OPENAI_API_KEY'))
            key_present_str = f"|key_present={api_key_present}"

        # Include model_kwargs in the cache key (convert dict to sorted tuple of items for consistency)
        model_kwargs_str = ""
        if model_kwargs:
            # Sort by key to ensure consistent order
            sorted_kwargs = sorted(model_kwargs.items())
            model_kwargs_str = f"|model_kwargs={str(sorted_kwargs)}" # Simple string representation

        cache_key = f"provider={resolved_provider.name}|model={resolved_model_name}{key_present_str}{model_kwargs_str}"

        logging.debug(f"Using Cache Key: {cache_key}")

        if cache_key in self.cache:
            logging.debug("Returning cached embedding model instance.")
            return self.cache[cache_key]

        logging.debug(f"Creating new embedding model instance: Provider={resolved_provider.name}, Model={resolved_model_name}")

        # --- Base Embedding Creation ---
        # Pass model_kwargs explicitly, and other necessary args from kwargs
        base_args = {
            "model_provider": resolved_provider,
            "model_name": resolved_model_name,
            "model_kwargs": model_kwargs,
            "openai_api_key": kwargs.get('openai_api_key'),
            "api_key": kwargs.get('api_key'),
            "base_url": kwargs.get('base_url'),
            "dimensions": kwargs.get('dimensions'),
        }
        # Noneの値は除外
        filtered_args = {k: v for k, v in base_args.items() if v is not None}
        # 実際の呼び出し
        base_embedder = self._create_base_embeddings(**filtered_args)

        # --- Cache Wrapping ---
        if cache_dir:
            logging.debug(f"Applying local file cache: {cache_dir}")
            # Ensure cache directory exists
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            store = LocalFileStore(cache_dir)
            cached_embedder = CacheBackedEmbeddings.from_bytes_store(
                base_embedder, store, namespace=f"{resolved_provider.name}_{resolved_model_name}"
            )
            self.cache[cache_key] = cached_embedder
            return cached_embedder
        else:
            # If no cache_dir, store the base embedding directly (or maybe don't cache?)
            # For now, let's cache the base embedding instance as well
            logging.debug("No cache directory specified, caching base embedding instance.")
            self.cache[cache_key] = base_embedder
            return base_embedder

    @staticmethod
    def _create_base_embeddings(
        model_provider: ModelProvider,
        model_name: str, # model_name is required here
        model_kwargs: Optional[Dict[str, Any]] = None, # Accept model_kwargs explicitly
        **other_kwargs: Any # Catch other args like openai_api_key, base_url, dimensions etc.
    ) -> Embeddings:
        """
        (Internal Static Method) Create base embedding model without caching.
        (内部静的メソッド) キャッシュなしの基本埋め込みモデルを作成します。
        Relies on arguments passed; uses environment variables as fallbacks for keys/URLs if not in kwargs.
        渡された引数に依存します。kwargsにない場合のキー/URLのフォールバックとして環境変数を使用します。

        Args:
            model_provider (ModelProvider): Type of embedding model provider.
            model_name (str): Name of the specific model.
            model_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments for the specific model constructor.
                                                    特定のモデルコンストラクタのための追加キーワード引数。
            **other_kwargs: Additional arguments like openai_api_key, base_url, etc.
                           openai_api_key, base_url などの追加引数。

        Returns:
            Embeddings: Base embedding model instance.

        Raises:
            ValueError: If model_provider is not supported or required keys are missing.
        """
        if model_provider == ModelProvider.OPENAI:
            # Priority for key: kwargs['openai_api_key'] > kwargs['api_key'] > env
            # Use other_kwargs here
            openai_api_key = other_kwargs.get('openai_api_key') or other_kwargs.get('api_key') or os.getenv('OPENAI_API_KEY')
            # Priority for base_url: kwargs['base_url'] > env
            base_url = other_kwargs.get('base_url', os.getenv('OPENAI_BASE_URL'))

            if not openai_api_key:
                raise ValueError(
                    "OpenAI API key is required. Provide it via OPENAI_API_KEY environment variable or openai_api_key/api_key argument."
                )

            constructor_args = {
                "model": model_name,
                "openai_api_key": openai_api_key,
                "base_url": base_url,
                "model_kwargs": model_kwargs or {}, # Pass model_kwargs, default to {} if None
                # Add other potential args from other_kwargs if needed by OpenAIEmbeddings
                "dimensions": other_kwargs.get("dimensions")
                # "show_progress_bar": other_kwargs.get("show_progress_bar") # Example
            }
            # Filter out None values before passing to constructor
            constructor_args = {k: v for k, v in constructor_args.items() if v is not None}

            logging.debug(f"[_create_base_embeddings] OpenAI final args: { {k: ('****' if 'key' in k else v) for k,v in constructor_args.items()} }") # Mask key in log
            return OpenAIEmbeddings(**constructor_args)

        elif model_provider == ModelProvider.OLLAMA:
            # Priority for base_url: kwargs['base_url'] > env
            base_url = other_kwargs.get('base_url', os.getenv('OLLAMA_BASE_URL'))

            if not base_url:
                 # Defaulting Ollama URL if not provided
                 base_url = "http://localhost:11434"
                 logging.warning(f"Ollama base URL not provided via argument or OLLAMA_BASE_URL env var. Defaulting to {base_url}")


            constructor_args = {
                "model": model_name,
                "base_url": base_url,
                 # Pass model_kwargs, default to {} if None
                "model_kwargs": model_kwargs or {},
                 # Add other potential args from other_kwargs if needed by OllamaEmbeddings
                 "temperature": other_kwargs.get("temperature"),
                 "top_k": other_kwargs.get("top_k"),
                 "top_p": other_kwargs.get("top_p"),
                 # "show_progress": other_kwargs.get("show_progress") # Example
            }
            # Filter out None values before passing to constructor
            constructor_args = {k: v for k, v in constructor_args.items() if v is not None}

            logging.debug(f"[_create_base_embeddings] Ollama final args: {constructor_args}")
            return OllamaEmbeddings(**constructor_args)

        else:
            raise ValueError(f"Unsupported embedding model provider: {model_provider}")

    def resolve_provider(self, provider_str: Optional[str]) -> ModelProvider:
        """
        Gets provider from argument or environment variable, falling back to default.

        引数または環境変数からプロバイダーを取得し、デフォルトにフォールバックします。     
        Raises ValueError if set but invalid.
        設定されているが無効な場合はValueErrorを発生させます。
        """
        if provider_str:
            try:
                return ModelProvider[provider_str.upper()]
            except KeyError:
                raise ValueError(
                    f"Unsupported model provider string: {provider_str.upper()}. "
                    f"Supported: {[p.name for p in ModelProvider]}"
                )
        # 環境変数もFINDALEDGE_...に統一
        env_provider = os.getenv('FINDALEDGE_MODEL_PROVIDER')
        if env_provider:
            try:
                return ModelProvider[env_provider.upper()]
            except KeyError:
                raise ValueError(
                    f"Unsupported model provider string: {env_provider.upper()}. "
                    f"Supported: {[p.name for p in ModelProvider]}"
                )
        return self.default_provider

    def _get_model_from_env(self) -> Optional[str]:
        """Gets model name from env var."""
        return os.getenv("FINDERLEDGE_EMBEDDING_MODEL_NAME")

    def _validate_provider(self, provider_input: Any) -> ModelProvider:
        """Validates the provider input and returns a ModelProvider enum member or raises ValueError."""
        if isinstance(provider_input, ModelProvider):
            return provider_input
        if isinstance(provider_input, str):
            try:
                return ModelProvider[provider_input.upper()]
            except KeyError:
                raise ValueError(
                    f"Unsupported model provider string: {provider_input}. "
                    f"Supported: {[p.name for p in ModelProvider]}"
                )
        # Handle other invalid types
        raise ValueError(f"Unsupported model provider type: {type(provider_input)}. Use ModelProvider enum or string.")

# 使用例 / Usage examples:
"""
# (Example usage assuming factory instance `emb_factory` exists)
# emb_factory = EmbeddingModelFactory()

# 環境変数 FINDERLEDGE_MODEL_PROVIDER に従い、キャッシュ付きで作成
embeddings = emb_factory.create_embeddings(
    cache_dir="./cache/embeddings"
)

# 明示的にOpenAIを指定し、キャッシュ付きで作成
openai_embeddings = emb_factory.create_embeddings(
    provider="openai",
    cache_dir="./cache/embeddings"
    # APIキーは環境変数 OPENAI_API_KEY から読み込まれる想定
)

# 明示的にOllamaを指定し、特定のモデル名で作成
ollama_embeddings = emb_factory.create_embeddings(
    provider=ModelProvider.OLLAMA,
    model_name="mistral",
    cache_dir="./cache/embeddings"
    # ベースURLは環境変数 OLLAMA_BASE_URL から読み込まれる想定
)

# キャッシュなしでOpenAIの特定のモデルを作成（APIキーは引数で渡す）
openai_specific_model = emb_factory.create_embeddings(
    provider=ModelProvider.OPENAI,
    model_name="text-embedding-3-large",
    openai_api_key="your-explicit-key"
)
""" 