"""
Test module for embeddings_factory.py
embeddings_factory.pyのテストモジュール
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock, ANY
from langchain.embeddings.base import Embeddings
from langchain_community.embeddings import OpenAIEmbeddings, OllamaEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.vectorstores.utils import DistanceStrategy
from langchain.storage import InMemoryStore, LocalFileStore
from langchain_core.embeddings import Embeddings
from pathlib import Path # Added for cache testing

from findaledge.embeddings_factory import EmbeddingModelFactory, ModelProvider

# Fixtures
@pytest.fixture
def mock_openai_embeddings():
    """
    Mock OpenAIEmbeddings instance
    OpenAIEmbeddingsのモックインスタンス
    """
    with patch('findaledge.embeddings_factory.OpenAIEmbeddings') as mock:
        yield mock

@pytest.fixture
def mock_ollama_embeddings():
    """
    Mock OllamaEmbeddings instance
    OllamaEmbeddingsのモックインスタンス
    """
    with patch('findaledge.embeddings_factory.OllamaEmbeddings') as mock:
        yield mock

@pytest.fixture
def factory():
    """Fixture for EmbeddingModelFactory instance."""
    factory_instance = EmbeddingModelFactory()
    factory_instance.cache.clear() # Clear instance cache before returning
    return factory_instance

@pytest.fixture(autouse=True)
def reset_factory_cache(factory):
    """Ensure factory cache is clear before each test using the factory."""
    factory.cache.clear() # Access instance cache

# Fixture to set environment variables
@pytest.fixture
def set_env_vars(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key_from_env")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://test-ollama-env:11434")
    monkeypatch.setenv("FINDERLEDGE_MODEL_PROVIDER", "openai") # Default provider env
    monkeypatch.setenv("FINDERLEDGE_EMBEDDING_MODEL_NAME", "text-embedding-env-default") # Default model env

# Test with default settings (using environment variables)
@patch('findaledge.embeddings_factory.EmbeddingModelFactory._create_base_embeddings')
def test_create_embeddings_default(mock_create_base, factory, monkeypatch):
    """Test creating embeddings with default settings (OpenAI). Assume API key in env."""
    # Simulate API key presence using environment variable
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-default")
    factory.create_embeddings()
    mock_create_base.assert_called_once_with(
        model_provider=ModelProvider.OPENAI,
        model_name="text-embedding-3-small",
    )

# Test explicitly specifying OpenAI
def test_create_embeddings_openai_explicit(factory, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key") # Still need key potentially
    with patch('findaledge.embeddings_factory.EmbeddingModelFactory._create_base_embeddings') as mock_create_base:
        embeddings = factory.create_embeddings(provider="openai", model_name="explicit-model", api_key="explicit-key") # Use factory instance
        mock_create_base.assert_called_once_with(
            model_provider=ModelProvider.OPENAI,
            model_name="explicit-model",
            api_key="explicit-key" # Check that kwargs are passed through
        )
        assert embeddings == mock_create_base.return_value

# Test explicitly specifying Ollama
def test_create_embeddings_ollama_explicit(factory):
    with patch('findaledge.embeddings_factory.EmbeddingModelFactory._create_base_embeddings') as mock_create_base:
        embeddings = factory.create_embeddings(provider="ollama", model_name="explicit-ollama", base_url="http://host.docker.internal:11434") # Use factory instance
        mock_create_base.assert_called_once_with(
            model_provider=ModelProvider.OLLAMA,
            model_name="explicit-ollama",
            base_url="http://host.docker.internal:11434"
        )
        assert embeddings == mock_create_base.return_value

# Test specifying Ollama model name
def test_create_embeddings_ollama_model_name(factory):
     with patch('findaledge.embeddings_factory.EmbeddingModelFactory._create_base_embeddings') as mock_create_base:
        # Provider explicitly ollama, model from args
        embeddings = factory.create_embeddings(provider="ollama", model_name="specific-ollama-model") # Use factory instance
        mock_create_base.assert_called_once_with(
            model_provider=ModelProvider.OLLAMA,
            model_name="specific-ollama-model"
            # base_url will use default inside _create_base_embeddings if not passed
        )
        assert embeddings == mock_create_base.return_value

# Test specifying OpenAI model name
@patch('findaledge.embeddings_factory.EmbeddingModelFactory._create_base_embeddings')
def test_create_embeddings_openai_model_name(mock_create_base, factory, monkeypatch):
    """Test creating OpenAI embeddings with a specific model name."""
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-specific-model")
    factory.create_embeddings(model_name="test-openai-model")
    mock_create_base.assert_called_once_with(
        model_provider=ModelProvider.OPENAI,
        model_name="test-openai-model", # Explicit model name
    )

# Test with caching enabled
@patch('findaledge.embeddings_factory.EmbeddingModelFactory._create_base_embeddings')
def test_create_embeddings_with_cache(mock_create_base, factory, tmp_path, monkeypatch):
    """Test creating embeddings with a specified cache directory."""
    cache_dir = tmp_path / "embedding_cache"
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-cache")
    factory.create_embeddings(cache_dir=str(cache_dir))
    mock_create_base.assert_called_once_with(
        model_provider=ModelProvider.OPENAI,
        model_name="text-embedding-3-small",
    )
    # Assert that the *returned* object IS a CacheBackedEmbeddings instance
    embeddings = factory.create_embeddings(cache_dir=str(cache_dir))
    assert isinstance(embeddings, CacheBackedEmbeddings)

# Test with explicit Ollama and caching
def test_create_embeddings_ollama_with_cache(factory, tmp_path):
    cache_dir = tmp_path / "ollama_cache"
    with patch('findaledge.embeddings_factory.EmbeddingModelFactory._create_base_embeddings') as mock_create_base, \
         patch('findaledge.embeddings_factory.CacheBackedEmbeddings.from_bytes_store') as mock_cache_from_store:

        mock_base_embedder_instance = MagicMock(spec=OllamaEmbeddings)
        mock_create_base.return_value = mock_base_embedder_instance

        embeddings = factory.create_embeddings(
            provider="ollama",
            model_name="cached-ollama",
            cache_dir=str(cache_dir) # Use factory instance
        )
        mock_create_base.assert_called_once_with(
            model_provider=ModelProvider.OLLAMA,
            model_name="cached-ollama"
            # base_url uses default
        )
        mock_cache_from_store.assert_called_once()
        call_args, call_kwargs = mock_cache_from_store.call_args
        assert call_args[0] == mock_base_embedder_instance
        assert isinstance(call_args[1], LocalFileStore)
        assert str(cache_dir) in str(call_args[1].root_path)
        assert call_kwargs['namespace'] == "OLLAMA_cached-ollama"
        assert embeddings == mock_cache_from_store.return_value

# Test invalid provider string from environment
def test_invalid_provider_env(factory, monkeypatch):
    monkeypatch.setenv("FINDALEDGE_MODEL_PROVIDER", "invalid_provider")
    # Now expect ValueError because _get_provider_from_env raises it
    with pytest.raises(ValueError, match=r"Unsupported model provider string: INVALID_PROVIDER"):
        factory.create_embeddings() # Use factory instance

# Test missing OpenAI API key
def test_missing_openai_key(factory, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Expect ValueError from _create_base_embeddings
    with pytest.raises(ValueError, match="OpenAI API key is required"):
        factory.create_embeddings(provider="openai", model_name="test-model") # Use factory instance

# Test overriding OpenAI key via kwargs
@patch('findaledge.embeddings_factory.EmbeddingModelFactory._create_base_embeddings')
def test_override_openai_key_kwargs(mock_create_base, factory, monkeypatch):
    """Test overriding OpenAI API key via kwargs."""
    monkeypatch.setenv("OPENAI_API_KEY", "env-key-override-test")
    factory.create_embeddings(openai_api_key="kwarg-key")
    mock_create_base.assert_called_once_with(
        model_provider=ModelProvider.OPENAI,
        model_name="text-embedding-3-small",
        openai_api_key="kwarg-key" # Key passed via kwargs *should* be passed down
    )

# Tests for EmbeddingModelFactory
class TestEmbeddingModelFactory:
    """
    Test cases for EmbeddingModelFactory
    EmbeddingModelFactoryのテストケース
    """

    @patch('findaledge.embeddings_factory.EmbeddingModelFactory._create_base_embeddings')
    def test_create_openai_embeddings(self, mock_create_base, factory, monkeypatch):
        """
        Test creating OpenAI embeddings
        OpenAI埋め込みの作成をテスト
        """
        api_key = "test-api-key"
        factory.create_embeddings(
            provider=ModelProvider.OPENAI,
            openai_api_key=api_key
        )
        mock_create_base.assert_called_once_with(
            model_provider=ModelProvider.OPENAI,
            model_name='text-embedding-3-small', # Default model
            openai_api_key=api_key
        )

    @patch('findaledge.embeddings_factory.EmbeddingModelFactory._create_base_embeddings')
    def test_create_ollama_embeddings(self, mock_create_base, factory):
        """
        Test creating Ollama embeddings
        Ollama埋め込みの作成をテスト
        """
        model_name = "llama2-test"
        factory.create_embeddings(
            provider=ModelProvider.OLLAMA, # Use provider instead of model_provider
            model_name=model_name
        )
        mock_create_base.assert_called_once_with(
            model_provider=ModelProvider.OLLAMA,
            model_name=model_name,
            # base_url uses default inside _create_base
        )

    @patch('findaledge.embeddings_factory.CacheBackedEmbeddings.from_bytes_store')
    @patch('findaledge.embeddings_factory.EmbeddingModelFactory._create_base_embeddings')
    def test_create_cached_embeddings(self, mock_create_base, mock_cache_from_store, factory, tmp_path, monkeypatch):
        """
        Test creating cached embeddings
        キャッシュ付き埋め込みの作成をテスト
        """
        api_key = "test-api-key"
        cache_dir = str(tmp_path / "embeddings_cache")
        mock_base_instance = MagicMock()
        mock_create_base.return_value = mock_base_instance

        embeddings = factory.create_embeddings(
            provider=ModelProvider.OPENAI,
            openai_api_key=api_key,
            cache_dir=cache_dir
        )

        assert embeddings == mock_cache_from_store.return_value
        mock_create_base.assert_called_once_with(
             model_provider=ModelProvider.OPENAI,
             model_name='text-embedding-3-small', # Default model
             openai_api_key=api_key
        )
        mock_cache_from_store.assert_called_once()
        call_args, call_kwargs = mock_cache_from_store.call_args
        assert call_args[0] == mock_base_instance
        assert isinstance(call_args[1], LocalFileStore)
        assert call_kwargs['namespace'] == "OPENAI_text-embedding-3-small"

    @patch.dict(os.environ, {}, clear=True)
    def test_create_openai_embeddings_without_api_key(self, factory):
        """
        Test creating OpenAI embeddings without API key raises error
        APIキーなしでOpenAI埋め込みを作成するとエラーが発生することをテスト
        """
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            factory.create_embeddings(provider="openai")

    @patch('findaledge.embeddings_factory.EmbeddingModelFactory._create_base_embeddings')
    def test_create_embeddings_with_unsupported_model(self, mock_create_base, factory):
        """
        Test creating embeddings with unsupported model type raises error
        サポートされていないモデルタイプでの埋め込み作成がエラーを発生させることをテスト
        """
        with pytest.raises(ValueError, match=r"Unsupported model provider string: INVALID_PROVIDER"):
            factory.create_embeddings(provider="invalid_provider")

@patch('findaledge.embeddings_factory.EmbeddingModelFactory._create_base_embeddings')
def test_create_embeddings_from_env(mock_create_base, mocker):
    # Simulate loading from env vars using mocker.patch for os.getenv
    def mock_getenv(key, default=None):
        if key == "FINDALEDGE_MODEL_PROVIDER":
            return "openai"
        elif key == "FINDALEDGE_EMBEDDING_MODEL_NAME":
            return "text-embedding-env"
        elif key == "OPENAI_API_KEY":
            return "fake-api-key"
        else:
            return default # Return default for other keys

    mocker.patch('findaledge.embeddings_factory.os.getenv', side_effect=mock_getenv)

    factory = EmbeddingModelFactory() # Re-initialize to pick up env vars via patched getenv
    embeddings = factory.create_embeddings() # Call without args to trigger env var usage

    # Assert _create_base was called with values derived from environment variables
    mock_create_base.assert_called_once_with(
        model_provider=ModelProvider.OPENAI, # Resolved from env
        model_name="text-embedding-env",     # Resolved from env
        # API key is loaded internally in _create_base
    )

@patch('findaledge.embeddings_factory.EmbeddingModelFactory._create_base_embeddings')
def test_create_embeddings_provider_priority(mock_create_base, monkeypatch):
    monkeypatch.setenv("FINDALEDGE_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("FINDALEDGE_EMBEDDING_MODEL_NAME", "env-model")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://env-ollama")

    factory = EmbeddingModelFactory()

    # Function args should override env vars
    embeddings = factory.create_embeddings(
        provider=ModelProvider.OLLAMA,
        model_name="arg-model",
        base_url="http://arg-ollama"
    )
    mock_create_base.assert_called_once_with(
        model_provider=ModelProvider.OLLAMA,
        model_name="arg-model",
        base_url="http://arg-ollama"
    )
    assert embeddings == mock_create_base.return_value

def test_create_embeddings_unsupported_provider_in_args(factory):
    with pytest.raises(ValueError, match=r"Unsupported model provider string: FAKE"):
        factory.create_embeddings(provider="fake")

def test_create_embeddings_unsupported_provider_in_env(monkeypatch):
    monkeypatch.setenv("FINDALEDGE_MODEL_PROVIDER", "invalid_provider")
    factory = EmbeddingModelFactory()
    # Expect ValueError directly from _get_provider_from_env called by create_embeddings
    with pytest.raises(ValueError, match=r"Unsupported model provider string: INVALID_PROVIDER"):
        factory.create_embeddings()

# Test caching functionality - mock the helper
@patch('findaledge.embeddings_factory.EmbeddingModelFactory._create_base_embeddings')
def test_embedding_model_caching(mock_create_base, factory, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-caching")
    # Mock return values for the base embedder
    mock_instance1 = MagicMock()
    mock_instance2 = MagicMock()
    mock_create_base.side_effect = [mock_instance1, mock_instance2]

    # Create first instance
    emb1 = factory.create_embeddings(provider=ModelProvider.OPENAI, model_name="model1")
    mock_create_base.assert_called_once_with(
        model_provider=ModelProvider.OPENAI,
        model_name="model1",
        # cache_dir=None # Removed assertion
        # openai_api_key="test-key" # Not passed explicitly here
    )
    assert emb1 == mock_instance1

    # Create second instance (should be cached)
    emb2 = factory.create_embeddings(provider=ModelProvider.OPENAI, model_name="model1")
    # _create_base should NOT be called again for the same key
    mock_create_base.assert_called_once() # Assert still called only once
    assert emb2 == mock_instance1 # Should return the cached instance

    # Create third instance with different model (should not be cached)
    emb3 = factory.create_embeddings(provider=ModelProvider.OPENAI, model_name="model2")
    assert mock_create_base.call_count == 2 # Now called twice
    mock_create_base.assert_called_with(
        model_provider=ModelProvider.OPENAI,
        model_name="model2",
        # cache_dir=None # Removed assertion
        # openai_api_key="test-key" # Not passed explicitly here
    ) # Check the second call args
    assert emb3 == mock_instance2

# Test creating embeddings when API key is missing (for OpenAI)
def test_create_openai_embeddings_without_api_key(monkeypatch, factory):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OpenAI API key is required"):
        factory.create_embeddings(provider=ModelProvider.OPENAI)

# Test default provider/model selection when env vars are not set
@patch('findaledge.embeddings_factory.EmbeddingModelFactory._create_base_embeddings')
def test_default_provider_and_model(mock_create_base, factory, monkeypatch):
    # Clear env vars that might influence defaults
    monkeypatch.delenv("FINDALEDGE_MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("FINDALEDGE_EMBEDDING_MODEL_NAME", raising=False)
    # Ensure API key is present for default OpenAI
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-default-provider")

    emb = factory.create_embeddings() # Call without args

    mock_create_base.assert_called_once_with(
        model_provider=ModelProvider.OPENAI,         # Default provider
        model_name="text-embedding-3-small", # Default model
        # cache_dir=None # Removed assertion
        # openai_api_key="test-key" # Not passed explicitly here
    )

# Test different providers mixed with caching - mock the helper
@patch('findaledge.embeddings_factory.EmbeddingModelFactory._create_base_embeddings')
def test_mixed_provider_caching(mock_create_base, factory, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-mixed-cache")
    # Mock return values
    mock_openai_m1 = MagicMock()
    mock_ollama_m1 = MagicMock()
    mock_ollama_m2 = MagicMock()
    mock_openai_m2 = MagicMock()
    mock_create_base.side_effect = [mock_openai_m1, mock_ollama_m1, mock_ollama_m2, mock_openai_m2]

    # OpenAI instance
    emb_openai1 = factory.create_embeddings(provider=ModelProvider.OPENAI, model_name="openai-m1")
    mock_create_base.assert_called_with(model_provider=ModelProvider.OPENAI, model_name="openai-m1") # Removed cache_dir
    assert emb_openai1 == mock_openai_m1

    # Ollama instance (different provider, same model name)
    emb_ollama1 = factory.create_embeddings(provider=ModelProvider.OLLAMA, model_name="ollama-m1")
    mock_create_base.assert_called_with(model_provider=ModelProvider.OLLAMA, model_name="ollama-m1") # Removed cache_dir
    assert emb_ollama1 == mock_ollama_m1

    # Ollama instance (different model name)
    emb_ollama2 = factory.create_embeddings(provider=ModelProvider.OLLAMA, model_name="ollama-m2")
    mock_create_base.assert_called_with(model_provider=ModelProvider.OLLAMA, model_name="ollama-m2") # Removed cache_dir
    assert emb_ollama2 == mock_ollama_m2

    # OpenAI instance (different model name)
    emb_openai2 = factory.create_embeddings(provider=ModelProvider.OPENAI, model_name="openai-m2")
    mock_create_base.assert_called_with(model_provider=ModelProvider.OPENAI, model_name="openai-m2") # Removed cache_dir
    assert emb_openai2 == mock_openai_m2

    # Re-request OpenAI M1 (should be cached)
    emb_openai1_cached = factory.create_embeddings(provider=ModelProvider.OPENAI, model_name="openai-m1")
    assert emb_openai1_cached == mock_openai_m1

    # Re-request Ollama M1 (should be cached)
    emb_ollama1_cached = factory.create_embeddings(provider=ModelProvider.OLLAMA, model_name="ollama-m1")
    assert emb_ollama1_cached == mock_ollama_m1

    assert mock_create_base.call_count == 4 # Should only have been called 4 times for the unique instances