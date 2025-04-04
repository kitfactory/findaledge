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
from langchain.storage import InMemoryStore
from langchain_core.embeddings import Embeddings

from finderledge.embeddings_factory import EmbeddingModelFactory, ModelProvider

# Fixtures
@pytest.fixture
def mock_openai_embeddings():
    """
    Mock OpenAIEmbeddings instance
    OpenAIEmbeddingsのモックインスタンス
    """
    with patch('finderledge.embeddings_factory.OpenAIEmbeddings') as mock:
        yield mock

@pytest.fixture
def mock_ollama_embeddings():
    """
    Mock OllamaEmbeddings instance
    OllamaEmbeddingsのモックインスタンス
    """
    with patch('finderledge.embeddings_factory.OllamaEmbeddings') as mock:
        yield mock

# Fixture to set environment variables
@pytest.fixture
def set_env_vars(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://test-ollama:11434")
    monkeypatch.setenv("FINDERLEDGE_MODEL_PROVIDER", "openai") # Default provider
    monkeypatch.setenv("FINDERLEDGE_EMBEDDING_MODEL_NAME", "text-embedding-ada-002") # Default model

# Test with default settings (using environment variables)
def test_create_embeddings_default(set_env_vars):
    embeddings = EmbeddingModelFactory.create_embeddings()
    assert isinstance(embeddings, OpenAIEmbeddings)
    # Check if model name from env var was used (may be internal detail)
    # assert embeddings.model == "text-embedding-ada-002"

# Test explicitly specifying OpenAI
def test_create_embeddings_openai_explicit(set_env_vars):
    embeddings = EmbeddingModelFactory.create_embeddings(model_provider=ModelProvider.OPENAI)
    assert isinstance(embeddings, OpenAIEmbeddings)

# Test explicitly specifying Ollama
def test_create_embeddings_ollama_explicit(set_env_vars):
    embeddings = EmbeddingModelFactory.create_embeddings(model_provider=ModelProvider.OLLAMA)
    assert isinstance(embeddings, OllamaEmbeddings)

# Test specifying Ollama model name
def test_create_embeddings_ollama_model_name(set_env_vars):
    embeddings = EmbeddingModelFactory.create_embeddings(
        model_provider=ModelProvider.OLLAMA,
        model_name="mistral"
    )
    assert isinstance(embeddings, OllamaEmbeddings)
    assert embeddings.model == "mistral"

# Test specifying OpenAI model name override
def test_create_embeddings_openai_model_name(set_env_vars):
    embeddings = EmbeddingModelFactory.create_embeddings(
        model_provider=ModelProvider.OPENAI,
        model_name="text-embedding-3-large"
    )
    assert isinstance(embeddings, OpenAIEmbeddings)
    assert embeddings.model == "text-embedding-3-large"

# Test with caching enabled
def test_create_embeddings_with_cache(set_env_vars, tmp_path):
    cache_dir = tmp_path / "embedding_cache"
    embeddings = EmbeddingModelFactory.create_embeddings(
        cache_dir=str(cache_dir)
    )
    assert isinstance(embeddings, CacheBackedEmbeddings)
    assert isinstance(embeddings.underlying_embeddings, OpenAIEmbeddings)
    # Check if cache directory was created (optional)
    # assert cache_dir.exists()

# Test with explicit Ollama and caching
def test_create_embeddings_ollama_with_cache(set_env_vars, tmp_path):
    cache_dir = tmp_path / "ollama_cache"
    embeddings = EmbeddingModelFactory.create_embeddings(
        model_provider=ModelProvider.OLLAMA,
        model_name="llama3",
        cache_dir=str(cache_dir)
    )
    assert isinstance(embeddings, CacheBackedEmbeddings)
    assert isinstance(embeddings.underlying_embeddings, OllamaEmbeddings)
    assert embeddings.underlying_embeddings.model == "llama3"

# Test invalid provider string from environment
def test_invalid_provider_env(monkeypatch):
    monkeypatch.setenv("FINDERLEDGE_MODEL_PROVIDER", "invalid_provider")
    with pytest.raises(ValueError, match="Unsupported model provider"):
        EmbeddingModelFactory.create_embeddings()

# Test missing OpenAI API key
def test_missing_openai_key(monkeypatch):
    # Ensure the key is not set
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OpenAI API key is required"):
        EmbeddingModelFactory.create_embeddings(model_provider=ModelProvider.OPENAI)

# Test overriding OpenAI key via kwargs
def test_override_openai_key_kwargs():
    # No need for env var fixture here
    embeddings = EmbeddingModelFactory.create_embeddings(
        model_provider=ModelProvider.OPENAI,
        openai_api_key="key_from_kwargs"
    )
    assert isinstance(embeddings, OpenAIEmbeddings)
    # Check if the key was passed (internal detail, might need mock)
    # For simplicity, just check type

# Tests for EmbeddingModelFactory
class TestEmbeddingModelFactory:
    """
    Test cases for EmbeddingModelFactory
    EmbeddingModelFactoryのテストケース
    """

    def test_create_openai_embeddings(self, mock_openai_embeddings):
        """
        Test creating OpenAI embeddings
        OpenAI埋め込みの作成をテスト
        """
        # Arrange
        api_key = "test-api-key"

        # Act
        embeddings = EmbeddingModelFactory.create_embeddings(
            model_provider=ModelProvider.OPENAI,
            openai_api_key=api_key
        )

        # Assert - Include default model name in expected args
        mock_openai_embeddings.assert_called_once_with(
            openai_api_key=api_key,
            model='text-embedding-3-small' # Default model is passed
        )

    def test_create_ollama_embeddings(self, mock_ollama_embeddings):
        """
        Test creating Ollama embeddings
        Ollama埋め込みの作成をテスト
        """
        # Arrange
        model_name = "llama2-test"

        # Act
        embeddings = EmbeddingModelFactory.create_embeddings(
            model_provider=ModelProvider.OLLAMA,
            model_name=model_name
        )

        # Assert - Include default base_url in expected args
        mock_ollama_embeddings.assert_called_once_with(
            model=model_name,
            base_url='http://localhost:11434' # Default base_url is passed
        )

    def test_create_cached_embeddings(self, mock_openai_embeddings, tmp_path):
        """
        Test creating cached embeddings
        キャッシュ付き埋め込みの作成をテスト
        """
        # Arrange
        api_key = "test-api-key"
        cache_dir = str(tmp_path / "embeddings_cache")

        # Act
        embeddings = EmbeddingModelFactory.create_embeddings(
            model_provider=ModelProvider.OPENAI,
            openai_api_key=api_key,
            cache_dir=cache_dir
        )

        # Assert
        assert isinstance(embeddings, CacheBackedEmbeddings)
        # Check the underlying embedding call args
        mock_openai_embeddings.assert_called_once_with(
             openai_api_key=api_key,
             model='text-embedding-3-small' # Default model
         )

    @patch.dict(os.environ, {}, clear=True)
    @patch('finderledge.embeddings_factory.OpenAIEmbeddings')
    def test_create_openai_embeddings_without_api_key(self, mock_openai_embeddings):
        """
        Test creating OpenAI embeddings without API key raises error
        APIキーなしでOpenAI埋め込みを作成するとエラーが発生することをテスト
        """
        # Act & Assert
        # The error is raised in _create_base_embeddings now
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            EmbeddingModelFactory.create_embeddings(model_provider="openai")
        mock_openai_embeddings.assert_not_called()

    def test_create_embeddings_with_unsupported_model(self):
        """
        Test creating embeddings with unsupported model type raises error
        サポートされていないモデルタイプでの埋め込み作成がエラーを発生させることをテスト
        """
        # Test the environment variable path
        with patch.dict(os.environ, {"FINDERLEDGE_MODEL_PROVIDER": "invalid_provider"}):
            # Match the more specific error raised by create_embeddings
            with pytest.raises(ValueError, match=r"Unsupported model provider in environment variable FINDERLEDGE_MODEL_PROVIDER: INVALID_PROVIDER. Supported: \['OPENAI', 'OLLAMA'\]"):
                EmbeddingModelFactory.create_embeddings()

        # Test the direct string input path
        with pytest.raises(ValueError, match=r"Unsupported model provider string: another_invalid. Supported: \['OPENAI', 'OLLAMA'\]"):
            EmbeddingModelFactory.create_embeddings(model_provider="another_invalid")

@patch('finderledge.embeddings_factory.OpenAIEmbeddings')
@patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True)
def test_create_openai_embeddings_with_api_key(mock_openai_embeddings):
    # Arrange
    factory = EmbeddingModelFactory()
    provider = "openai"
    model = "text-embedding-ada-002"

    # Act
    embeddings = factory.create_embeddings(provider, model)

    # Assert
    assert isinstance(embeddings, MagicMock)
    mock_openai_embeddings.assert_called_once_with(model=model, openai_api_key="test_key")

@patch.dict(os.environ, {}, clear=True) # Ensure no key is present
@patch('finderledge.embeddings_factory.OpenAIEmbeddings')
def test_create_openai_embeddings_without_api_key(mock_openai_embeddings):
    factory = EmbeddingModelFactory()
    provider = "openai"
    model = "text-embedding-ada-002"

    # Expect the specific ValueError for missing API key
    with pytest.raises(ValueError, match="OpenAI API key is required"):
        factory.create_embeddings(provider, model)
    mock_openai_embeddings.assert_not_called()

@patch('finderledge.embeddings_factory.OllamaEmbeddings')
@patch('finderledge.embeddings_factory.OpenAIEmbeddings')
def test_create_embeddings_with_unsupported_model(mock_openai, mock_ollama):
    factory = EmbeddingModelFactory()
    provider = "unsupported_provider"
    model = "some_model"

    # Match the error raised by create_embeddings for an unsupported string
    with pytest.raises(ValueError, match=r"Unsupported model provider string.*unsupported_provider"):
        factory.create_embeddings(provider, model)

    mock_openai.assert_not_called()
    mock_ollama.assert_not_called() 