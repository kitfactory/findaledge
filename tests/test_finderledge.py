"""
Tests for the FinderLedge class
FinderLedgeクラスのテスト
"""

import pytest
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from finderledge import FinderLedge, Document, OpenAIEmbeddingModel

@pytest.fixture
def temp_dir(tmp_path):
    """
    Create a temporary directory for test files
    テストファイル用の一時ディレクトリを作成
    """
    return tmp_path

@pytest.fixture
def mock_embedding_model():
    """
    Create a mock embedding model
    モックの埋め込みモデルを作成
    """
    model = MagicMock(spec=OpenAIEmbeddingModel)
    model.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    model.embed_query.return_value = [0.1, 0.2, 0.3]
    return model

@pytest.fixture
def finderledge(temp_dir, mock_embedding_model):
    """
    Create a FinderLedge instance for testing
    テスト用のFinderLedgeインスタンスを作成
    """
    return FinderLedge(
        db_name="test_db",
        persist_dir=str(temp_dir),
        embedding_model=mock_embedding_model
    )

def test_finderledge_initialization(finderledge):
    """
    Test FinderLedge initialization
    FinderLedgeの初期化をテスト
    """
    assert finderledge.db_name == "test_db"
    assert finderledge.documents == {}
    assert finderledge.bm25_index is not None

def test_add_document(finderledge):
    """
    Test adding a document
    文書の追加をテスト
    """
    doc = Document(
        id="test1",
        title="Test Document",
        content="This is a test document."
    )
    
    finderledge.add_document(doc)
    
    assert "test1" in finderledge.documents
    assert finderledge.documents["test1"] == doc
    assert len(doc.chunks) > 0
    assert len(doc.chunk_embeddings) > 0

def test_remove_document(finderledge):
    """
    Test removing a document
    文書の削除をテスト
    """
    doc = Document(
        id="test1",
        title="Test Document",
        content="This is a test document."
    )
    
    finderledge.add_document(doc)
    finderledge.remove_document("test1")
    
    assert "test1" not in finderledge.documents

def test_find_related_documents(finderledge):
    """
    Test finding related documents
    関連文書の検索をテスト
    """
    # Add test documents
    doc1 = Document(
        id="test1",
        title="First Document",
        content="This is the first test document."
    )
    doc2 = Document(
        id="test2",
        title="Second Document",
        content="This is the second test document."
    )
    
    finderledge.add_document(doc1)
    finderledge.add_document(doc2)
    
    # Test different search modes
    query = "test document"
    
    # Test hybrid search
    results = finderledge.find_related_documents(query, search_mode="hybrid")
    assert len(results) > 0
    
    # Test vector search
    results = finderledge.find_related_documents(query, search_mode="vector")
    assert len(results) > 0
    
    # Test keyword search
    results = finderledge.find_related_documents(query, search_mode="keyword")
    assert len(results) > 0

def test_get_context(finderledge):
    """
    Test getting context for a query
    クエリのコンテキスト取得をテスト
    """
    # Add test documents
    doc1 = Document(
        id="test1",
        title="First Document",
        content="This is the first test document."
    )
    doc2 = Document(
        id="test2",
        title="Second Document",
        content="This is the second test document."
    )
    
    finderledge.add_document(doc1)
    finderledge.add_document(doc2)
    
    query = "test document"
    context = finderledge.get_context(query)
    
    assert isinstance(context, str)
    assert len(context) > 0
    assert "test document" in context.lower()

def test_persist_and_load_state(finderledge, temp_dir):
    """
    Test persisting and loading state
    状態の永続化と読み込みをテスト
    """
    # Add test document
    doc = Document(
        id="test1",
        title="Test Document",
        content="This is a test document."
    )
    
    finderledge.add_document(doc)
    
    # Persist state
    finderledge._persist_state()
    
    # Create new instance and load state
    new_finderledge = FinderLedge(
        db_name="test_db",
        persist_dir=str(temp_dir),
        embedding_model=finderledge.embedding_model
    )
    new_finderledge._load_state()
    
    assert "test1" in new_finderledge.documents
    assert new_finderledge.documents["test1"].id == doc.id
    assert new_finderledge.documents["test1"].title == doc.title
    assert new_finderledge.documents["test1"].content == doc.content

def test_close(finderledge):
    """
    Test closing FinderLedge instance
    FinderLedgeインスタンスのクローズをテスト
    """
    finderledge.close()
    # Add assertions for cleanup if needed

def test_get_langchain_retriever(finderledge):
    """
    Test getting LangChain retriever
    LangChainレトリーバーの取得をテスト
    """
    with pytest.raises(NotImplementedError):
        finderledge.get_langchain_retriever() 