"""
Test embedding model implementation
埋め込みモデル実装のテスト

This module contains tests for the embedding model implementation.
このモジュールには埋め込みモデル実装のテストが含まれています。
"""

import pytest
from unittest.mock import patch, MagicMock
from finderledge.embedding_model import OpenAIEmbeddingModel

@pytest.fixture
def embedding_model():
    """
    Create a test embedding model
    テスト用の埋め込みモデルを作成
    """
    return OpenAIEmbeddingModel(model="text-embedding-3-small")

def test_embedding_model_initialization(embedding_model):
    """
    Test embedding model initialization
    埋め込みモデルの初期化をテスト
    """
    assert embedding_model.model == "text-embedding-3-small"
    assert embedding_model.client is not None

def test_generate_embedding(embedding_model):
    """
    Test single text embedding generation
    単一テキストの埋め込み生成をテスト
    """
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    
    with patch.object(embedding_model.client.embeddings, 'create', return_value=mock_response):
        embedding = embedding_model.embed_query("test text")
        assert len(embedding) == 3
        assert embedding == [0.1, 0.2, 0.3]

def test_generate_embeddings(embedding_model):
    """
    Test multiple text embeddings generation
    複数テキストの埋め込み生成をテスト
    """
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1, 0.2, 0.3]),
        MagicMock(embedding=[0.4, 0.5, 0.6])
    ]
    
    with patch.object(embedding_model.client.embeddings, 'create', return_value=mock_response):
        embeddings = embedding_model.embed_documents(["text1", "text2"])
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]

def test_embedding_model_serialization(embedding_model):
    """
    Test embedding model serialization
    埋め込みモデルのシリアライズをテスト
    """
    # Test to_dict
    model_dict = embedding_model.to_dict()
    assert model_dict["model"] == "text-embedding-3-small"
    
    # Test from_dict
    new_model = OpenAIEmbeddingModel.from_dict(model_dict)
    assert new_model.model == "text-embedding-3-small"

def test_embedding_model_empty_text(embedding_model):
    """
    Test embedding model with empty text
    空のテキストでの埋め込みモデルをテスト
    """
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    
    with patch.object(embedding_model.client.embeddings, 'create', return_value=mock_response):
        embedding = embedding_model.embed_query("")
        assert len(embedding) == 3
        assert embedding == [0.1, 0.2, 0.3]

def test_embedding_model_long_text(embedding_model):
    """
    Test embedding model with long text
    長いテキストでの埋め込みモデルをテスト
    """
    long_text = "test " * 1000
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    
    with patch.object(embedding_model.client.embeddings, 'create', return_value=mock_response):
        embedding = embedding_model.embed_query(long_text)
        assert len(embedding) == 3
        assert embedding == [0.1, 0.2, 0.3]

def test_embedding_model_special_characters(embedding_model):
    """
    Test embedding model with special characters
    特殊文字での埋め込みモデルをテスト
    """
    special_text = "!@#$%^&*()_+{}|:\"<>?`-=[]\\;',./"
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    
    with patch.object(embedding_model.client.embeddings, 'create', return_value=mock_response):
        embedding = embedding_model.embed_query(special_text)
        assert len(embedding) == 3
        assert embedding == [0.1, 0.2, 0.3] 