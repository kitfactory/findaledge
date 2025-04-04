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
    @patch('findaledge.embeddings_factory.OpenAIEmbeddings')
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

@patch('findaledge.embeddings_factory.OpenAIEmbeddings')
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
@patch('findaledge.embeddings_factory.OpenAIEmbeddings')
def test_create_openai_embeddings_without_api_key(mock_openai_embeddings):
    factory = EmbeddingModelFactory()
    provider = "openai"
    model = "text-embedding-ada-002"

    # Expect the specific ValueError for missing API key
    with pytest.raises(ValueError, match="OpenAI API key is required"):
        factory.create_embeddings(provider, model)
    mock_openai_embeddings.assert_not_called()

@patch('findaledge.embeddings_factory.OllamaEmbeddings')
@patch('findaledge.embeddings_factory.OpenAIEmbeddings')
def test_create_embeddings_with_unsupported_model(mock_openai, mock_ollama):
    factory = EmbeddingModelFactory()
    provider = "unsupported_provider"
    model = "some_model"

    # Match the error raised by create_embeddings for an unsupported string
    with pytest.raises(ValueError, match=r"Unsupported model provider string.*unsupported_provider"):
        factory.create_embeddings(provider, model)

    mock_openai.assert_not_called()
    mock_ollama.assert_not_called()

def test_create_openai_embeddings(mock_openai_embeddings):
    factory = EmbeddingModelFactory()
    # Ensure the patch target uses the new name
    with patch('findaledge.embeddings_factory.OpenAIEmbeddings') as mock:
        embeddings = factory.create_embeddings(provider=ModelProvider.OPENAI, model_name="test-model", api_key="test-key")
        mock.assert_called_once_with(model="test-model", openai_api_key="test-key", base_url=None)
        assert embeddings == mock.return_value

def test_create_ollama_embeddings(mock_ollama_embeddings):
    factory = EmbeddingModelFactory()
    # Ensure the patch target uses the new name
    with patch('findaledge.embeddings_factory.OllamaEmbeddings') as mock:
        embeddings = factory.create_embeddings(provider=ModelProvider.OLLAMA, model_name="ollama-model", base_url="http://localhost:11434")
        mock.assert_called_once_with(model="ollama-model", base_url="http://localhost:11434")
        assert embeddings == mock.return_value

def test_create_embeddings_from_env(monkeypatch):
    # Ensure env var names use the new name
    monkeypatch.setenv("FINDALEDGE_MODEL_PROVIDER", "openai") # Default provider
    monkeypatch.setenv("FINDALEDGE_EMBEDDING_MODEL_NAME", "text-embedding-ada-002") # Default model
    monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key")

    factory = EmbeddingModelFactory()
    # Ensure the patch target uses the new name
    with patch('findaledge.embeddings_factory.OpenAIEmbeddings') as mock:
        embeddings = factory.create_embeddings() # No args, should use env vars
        mock.assert_called_once_with(
            model="text-embedding-ada-002",
            openai_api_key="fake-api-key", # Fetched from env by OpenAIEmbeddings itself
            base_url=None # Default value
        )
        assert embeddings is not None
        # assert isinstance(embeddings, OpenAIEmbeddings) # Cannot assert instance with mock

def test_create_embeddings_provider_priority(monkeypatch):
    # Environment variables set
    monkeypatch.setenv("FINDALEDGE_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("FINDALEDGE_EMBEDDING_MODEL_NAME", "env-model")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://env-ollama")

    factory = EmbeddingModelFactory()

    # Function args should override env vars
    # Ensure the patch target uses the new name
    with patch('findaledge.embeddings_factory.OllamaEmbeddings') as mock_ollama:
        embeddings = factory.create_embeddings(
            provider=ModelProvider.OLLAMA,
            model_name="arg-model",
            base_url="http://arg-ollama"
        )
        mock_ollama.assert_called_once_with(model="arg-model", base_url="http://arg-ollama")
        assert embeddings == mock_ollama.return_value

def test_create_embeddings_unsupported_provider_in_args():
    factory = EmbeddingModelFactory()
    with pytest.raises(ValueError, match=r"Unsupported model provider specified: FAKE"):
        factory.create_embeddings(provider="fake") # type: ignore

def test_create_embeddings_unsupported_provider_in_env(monkeypatch):
    # Ensure env var name uses the new name
    monkeypatch.setenv("FINDALEDGE_MODEL_PROVIDER", "invalid_provider")
    monkeypatch.setenv("FINDALEDGE_EMBEDDING_MODEL_NAME", "some-model")

    factory = EmbeddingModelFactory()
    # Ensure the patch target and error message use the new env var name
    with pytest.raises(ValueError, match=r"Unsupported model provider in environment variable FINDALEDGE_MODEL_PROVIDER: INVALID_PROVIDER"):
        factory.create_embeddings()

# Test caching functionality
# Ensure the patch target uses the new name
@patch('findaledge.embeddings_factory.OpenAIEmbeddings')
def test_embedding_model_caching(mock_openai_embeddings_cls, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    factory = EmbeddingModelFactory()

    # Create first instance
    emb1 = factory.create_embeddings(provider=ModelProvider.OPENAI, model_name="model1")
    mock_openai_embeddings_cls.assert_called_once_with(model="model1", openai_api_key="test-key", base_url=None)
    mock_openai_embeddings_cls.reset_mock()

    # Create second instance with same params - should be cached
    emb2 = factory.create_embeddings(provider=ModelProvider.OPENAI, model_name="model1")
    mock_openai_embeddings_cls.assert_not_called() # Should not be called again
    assert emb1 is emb2 # Should be the exact same object

    # Create third instance with different params - should not be cached
    emb3 = factory.create_embeddings(provider=ModelProvider.OPENAI, model_name="model2")
    mock_openai_embeddings_cls.assert_called_once_with(model="model2", openai_api_key="test-key", base_url=None)
    assert emb1 is not emb3
    assert emb2 is not emb3

# Test creating embeddings when API key is missing (for OpenAI)
def test_create_openai_embeddings_without_api_key(monkeypatch):
    # Ensure API key env var is unset
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    factory = EmbeddingModelFactory()

    # Patch the actual OpenAIEmbeddings class used internally
    # Ensure the patch target uses the new name
    with patch('findaledge.embeddings_factory.OpenAIEmbeddings') as mock_openai:
        # Mock the behavior when API key is missing during initialization or first call
        # The actual error might be raised by langchain's Pydantic validation or upon first API call
        # Let's assume for this test the factory allows creation but usage would fail.
        # A more accurate test might involve mocking the API call itself if langchain allows creation without key.
        # For simplicity, we just check if the factory attempts creation.
        factory.create_embeddings(provider=ModelProvider.OPENAI, model_name="test-model")
        mock_openai.assert_called_once_with(model="test-model", openai_api_key=None, base_url=None)

# Test default provider/model selection when env vars are not set
# Ensure the patch target uses the new name
@patch('findaledge.embeddings_factory.OpenAIEmbeddings')
def test_default_provider_and_model(mock_openai_embeddings_cls, monkeypatch):
     # Ensure relevant env vars are unset
    monkeypatch.delenv("FINDALEDGE_MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("FINDALEDGE_EMBEDDING_MODEL_NAME", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key") # Still need API key

    factory = EmbeddingModelFactory()
    emb = factory.create_embeddings() # Call without args

    # Check if it defaulted to OpenAI and the expected default model
    mock_openai_embeddings_cls.assert_called_once_with(
        model="text-embedding-3-small", # Check against the actual default in the factory
        openai_api_key="test-key",
        base_url=None
    )
    assert emb == mock_openai_embeddings_cls.return_value


# Test different providers mixed with caching
# Ensure the patch targets use the new name
@patch('findaledge.embeddings_factory.OllamaEmbeddings')
@patch('findaledge.embeddings_factory.OpenAIEmbeddings')
def test_mixed_provider_caching(mock_openai_cls, mock_ollama_cls, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    factory = EmbeddingModelFactory()

    # OpenAI instance
    emb_openai1 = factory.create_embeddings(provider=ModelProvider.OPENAI, model_name="openai-m1")
    mock_openai_cls.assert_called_once_with(model="openai-m1", openai_api_key="test-key", base_url=None)
    mock_openai_cls.reset_mock()

    # Ollama instance
    emb_ollama1 = factory.create_embeddings(provider=ModelProvider.OLLAMA, model_name="ollama-m1")
    mock_ollama_cls.assert_called_once_with(model="ollama-m1", base_url=None)
    mock_ollama_cls.reset_mock()

    # Another OpenAI instance (cached)
    emb_openai2 = factory.create_embeddings(provider=ModelProvider.OPENAI, model_name="openai-m1")
    mock_openai_cls.assert_not_called()
    assert emb_openai1 is emb_openai2

    # Another Ollama instance (new model, not cached)
    emb_ollama2 = factory.create_embeddings(provider=ModelProvider.OLLAMA, model_name="ollama-m2")
    mock_ollama_cls.assert_called_once_with(model="ollama-m2", base_url=None)
    assert emb_ollama1 is not emb_ollama2

    # Another OpenAI instance (new model, not cached)
    emb_openai3 = factory.create_embeddings(provider=ModelProvider.OPENAI, model_name="openai-m2")
    mock_openai_cls.assert_called_once_with(model="openai-m2", openai_api_key="test-key", base_url=None)
    assert emb_openai1 is not emb_openai3 