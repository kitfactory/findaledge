"""
Tests for the EmbeddingModel class
EmbeddingModelクラスのテスト
"""

import pytest
import numpy as np
import torch
from finderledge.embedding_model import EmbeddingModel

@pytest.fixture
def embedding_model():
    """
    Create a test embedding model
    テスト用の埋め込みモデルを作成
    """
    return EmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")

def test_embedding_model_initialization(embedding_model):
    """
    Test embedding model initialization
    埋め込みモデルの初期化テスト
    """
    assert embedding_model.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert isinstance(embedding_model.device, torch.device)
    assert embedding_model.tokenizer is not None
    assert embedding_model.model is not None

def test_generate_embedding(embedding_model):
    """
    Test generating embedding for a single text
    単一テキストの埋め込み生成テスト
    """
    text = "This is a test document."
    embedding = embedding_model.generate_embedding(text)

    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert embedding.shape[0] > 0

def test_generate_embeddings(embedding_model):
    """
    Test generating embeddings for multiple texts
    複数テキストの埋め込み生成テスト
    """
    texts = [
        "This is a test document.",
        "This is another test document.",
        "This is a different document."
    ]
    embeddings = embedding_model.generate_embeddings(texts)

    assert len(embeddings) == 3
    assert all(isinstance(embedding, np.ndarray) for embedding in embeddings)
    assert all(embedding.ndim == 1 for embedding in embeddings)
    assert all(embedding.shape[0] > 0 for embedding in embeddings)

def test_embedding_model_serialization(embedding_model):
    """
    Test embedding model serialization and deserialization
    埋め込みモデルのシリアライズとデシリアライズのテスト
    """
    # Test serialization
    model_dict = embedding_model.to_dict()
    assert isinstance(model_dict, dict)
    assert model_dict["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert isinstance(model_dict["device"], str)

    # Test deserialization
    new_model = EmbeddingModel.from_dict(model_dict)
    assert new_model.model_name == embedding_model.model_name
    assert str(new_model.device) == str(embedding_model.device)

def test_embedding_model_empty_text(embedding_model):
    """
    Test embedding model with empty text
    空のテキストを使用した埋め込みモデルのテスト
    """
    text = ""
    embedding = embedding_model.generate_embedding(text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert embedding.shape[0] > 0

def test_embedding_model_long_text(embedding_model):
    """
    Test embedding model with long text
    長いテキストを使用した埋め込みモデルのテスト
    """
    text = "This is a very long text. " * 100
    embedding = embedding_model.generate_embedding(text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert embedding.shape[0] > 0

def test_embedding_model_special_characters(embedding_model):
    """
    Test embedding model with special characters
    特殊文字を使用した埋め込みモデルのテスト
    """
    text = "This is a test document with special characters: @#$%^&*"
    embedding = embedding_model.generate_embedding(text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert embedding.shape[0] > 0 