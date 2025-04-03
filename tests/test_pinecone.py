"""
Test module for Pinecone document store
Pineconeドキュメントストアのテストモジュール
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from finderledge.pinecone import PineconeDocumentStore

class MockEmbeddings(Embeddings):
    """Mock embeddings for testing"""
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]
    
    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

@pytest.fixture
def mock_pinecone():
    """Create mock Pinecone store"""
    with patch('pinecone.init'), \
         patch('langchain_community.vectorstores.Pinecone') as mock_store:
        mock_instance = MagicMock()
        mock_store.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def store(mock_pinecone):
    """Create test document store instance"""
    return PineconeDocumentStore(
        api_key="test-key",
        environment="test-env",
        index_name="test-index",
        embedding_function=MockEmbeddings()
    )

def test_initialization(store, mock_pinecone):
    """Test store initialization"""
    assert store is not None
    assert store.pinecone_store == mock_pinecone

def test_add_documents(store, mock_pinecone):
    """Test adding documents"""
    # Prepare test data
    docs = [
        Document(
            page_content="Test document 1",
            metadata={"source": "test1.txt"}
        ),
        Document(
            page_content="Test document 2",
            metadata={"source": "test2.txt"}
        )
    ]

    # Add documents
    doc_ids = store.add_documents(docs)

    # Verify documents were added
    assert len(doc_ids) == 2
    mock_pinecone.add_documents.assert_called_once()

def test_get_document(store, mock_pinecone):
    """Test getting a document"""
    # Mock search results
    mock_doc = Document(
        page_content="Test document",
        metadata={"id": "test-id", "source": "test.txt"}
    )
    mock_pinecone.similarity_search.return_value = [mock_doc]

    # Get document
    doc = store.get_document("test-id")

    # Verify document was retrieved
    assert doc == mock_doc
    mock_pinecone.similarity_search.assert_called_once_with(
        query="",
        k=1,
        filter={"id": "test-id"}
    )

def test_get_split_documents(store, mock_pinecone):
    """Test getting split documents"""
    # Mock search results
    mock_splits = [
        Document(
            page_content="Split 1",
            metadata={"parent_id": "parent-id", "split_index": 0, "is_split": True}
        ),
        Document(
            page_content="Split 2",
            metadata={"parent_id": "parent-id", "split_index": 1, "is_split": True}
        )
    ]
    mock_pinecone.similarity_search.return_value = mock_splits

    # Get split documents
    splits = store.get_split_documents("parent-id")

    # Verify splits were retrieved
    assert len(splits) == 2
    assert splits == mock_splits
    mock_pinecone.similarity_search.assert_called_once_with(
        query="",
        filter={"parent_id": "parent-id", "is_split": True}
    )

def test_delete_document(store, mock_pinecone):
    """Test deleting a document"""
    # Mock split documents
    mock_splits = [
        Document(
            page_content="Split 1",
            metadata={"id": "split-1", "parent_id": "test-id", "is_split": True}
        ),
        Document(
            page_content="Split 2",
            metadata={"id": "split-2", "parent_id": "test-id", "is_split": True}
        )
    ]
    mock_pinecone.similarity_search.return_value = mock_splits

    # Delete document
    store.delete_document("test-id")

    # Verify document and splits were deleted
    assert mock_pinecone.delete.call_count == 2
    mock_pinecone.delete.assert_any_call(["test-id"])
    mock_pinecone.delete.assert_any_call(["split-1", "split-2"])

def test_as_retriever(store, mock_pinecone):
    """Test getting retriever interface"""
    # Get retriever
    retriever = store.as_retriever()

    # Verify retriever was created
    assert retriever is not None
    mock_pinecone.as_retriever.assert_called_once() 