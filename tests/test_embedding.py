"""
Tests for the EmbeddingModel classes
EmbeddingModelクラスのテスト
"""

import pytest
from unittest.mock import patch, MagicMock
from finderledge import OpenAIEmbeddingModel, HuggingFaceEmbeddingModel

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
    
    with patch('openai.Embedding.create', return_value=mock_response):
        model = OpenAIEmbeddingModel(model_name="text-embedding-ada-002")
        
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

def test_huggingface_embedding_model():
    """
    Test HuggingFace embedding model
    HuggingFace埋め込みモデルのテスト
    """
    # Mock HuggingFace model and tokenizer
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    
    # Mock model output
    mock_output = MagicMock()
    mock_output.last_hidden_state = MagicMock()
    mock_output.last_hidden_state.mean.return_value = [[0.1, 0.2, 0.3]]
    
    with patch('transformers.AutoModel.from_pretrained', return_value=mock_model), \
         patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
        
        model = HuggingFaceEmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Test document embedding
        documents = ["First document", "Second document"]
        embeddings = model.embed_documents(documents)
        
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.1, 0.2, 0.3]
        
        # Test query embedding
        query = "Test query"
        query_embedding = model.embed_query(query)
        
        assert query_embedding == [0.1, 0.2, 0.3]

def test_embedding_model_errors():
    """
    Test error handling in embedding models
    埋め込みモデルのエラーハンドリングをテスト
    """
    # Test OpenAI model with invalid API key
    with pytest.raises(ValueError):
        OpenAIEmbeddingModel(model_name="text-embedding-ada-002", api_key="invalid_key")
    
    # Test HuggingFace model with invalid model name
    with pytest.raises(ValueError):
        HuggingFaceEmbeddingModel(model_name="invalid_model") 