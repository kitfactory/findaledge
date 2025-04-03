"""
Test module for embeddings_factory.py
embeddings_factory.pyのテストモジュール
"""

import os
import pytest
from unittest.mock import Mock, patch
from langchain.embeddings.base import Embeddings
from langchain_community.embeddings import OpenAIEmbeddings, OllamaEmbeddings
from langchain.embeddings import CacheBackedEmbeddings

from finderledge.embeddings_factory import EmbeddingModelFactory, EmbeddingModelType

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
            model_type=EmbeddingModelType.OPENAI,
            openai_api_key=api_key
        )

        # Assert
        mock_openai_embeddings.assert_called_once_with(openai_api_key=api_key)

    def test_create_ollama_embeddings(self, mock_ollama_embeddings):
        """
        Test creating Ollama embeddings
        Ollama埋め込みの作成をテスト
        """
        # Arrange
        model_name = "llama2-test"

        # Act
        embeddings = EmbeddingModelFactory.create_embeddings(
            model_type=EmbeddingModelType.OLLAMA,
            model_name=model_name
        )

        # Assert
        mock_ollama_embeddings.assert_called_once_with(model=model_name)

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
            model_type=EmbeddingModelType.OPENAI,
            openai_api_key=api_key,
            cache_dir=cache_dir
        )

        # Assert
        assert isinstance(embeddings, CacheBackedEmbeddings)
        mock_openai_embeddings.assert_called_once_with(openai_api_key=api_key)

    def test_create_openai_embeddings_without_api_key(self):
        """
        Test creating OpenAI embeddings without API key raises error
        APIキーなしでOpenAI埋め込みを作成するとエラーが発生することをテスト
        """
        # Act & Assert
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            EmbeddingModelFactory.create_embeddings(
                model_type=EmbeddingModelType.OPENAI
            )

    def test_create_embeddings_with_unsupported_model(self):
        """
        Test creating embeddings with unsupported model type raises error
        サポートされていないモデルタイプでの埋め込み作成がエラーを発生させることをテスト
        """
        # Arrange
        mock_unsupported_type = Mock()
        mock_unsupported_type.name = "UNSUPPORTED"

        # Act & Assert
        with pytest.raises(ValueError, match="Unsupported model type"):
            EmbeddingModelFactory.create_embeddings(
                model_type=mock_unsupported_type
            ) 