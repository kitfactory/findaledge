"""
Tests for the EmbeddingModel classes
EmbeddingModelクラスのテスト
"""

import pytest
from unittest.mock import patch, MagicMock
from finderledge import OpenAIEmbeddingModel
import numpy as np

def test_openai_embedding_model():
    """
    Test OpenAI embedding model
    OpenAI埋め込みモデルのテスト
    """
    # Mock OpenAI API response
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1, 0.2, 0.3]),
        MagicMock(embedding=[0.4, 0.5, 0.6])
    ]
    
    with patch('openai.embeddings.create', return_value=mock_response):
        model = OpenAIEmbeddingModel(api_key="dummy_key", model_name="text-embedding-ada-002")
        
        # Test document embedding
        documents = ["First document", "Second document"]
        embeddings = model.embed_documents(documents)
        
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]
        
        # Test query embedding
        query = "Test query"
        query_embedding = model.embed_query(query)
        
        assert query_embedding == [0.1, 0.2, 0.3]

def test_embedding_model_errors():
    """
    Test error handling in embedding models
    埋め込みモデルのエラーハンドリングをテスト
    """
    # Reset OpenAI API key
    with patch('openai.api_key', None):
        # Test OpenAI model with no API key
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            OpenAIEmbeddingModel() 