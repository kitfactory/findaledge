"""
Test module for chroma.py
chroma.pyのテストモジュール
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings

from finderledge.chroma import ChromaDocumentStore

@pytest.fixture
def mock_chroma():
    """
    Mock Chroma vector document store
    Chromaベクトルドキュメントストアのモック
    """
    with patch('finderledge.chroma.Chroma') as mock:
        mock_instance = mock.return_value
        # Mock similarity_search method
        mock_instance.similarity_search.return_value = []
        # Mock add_documents method
        mock_instance.add_documents = Mock()
        yield mock_instance

@pytest.fixture
def mock_embeddings():
    """
    Mock OpenAI embeddings
    OpenAI埋め込みのモック
    """
    return Mock(spec=OpenAIEmbeddings)

@pytest.fixture
def test_store(tmp_path, mock_chroma, mock_embeddings):
    """
    Test ChromaDocumentStore instance
    ChromaDocumentStoreのテストインスタンス
    """
    store = ChromaDocumentStore(
        persist_directory=str(tmp_path / "test_chroma"),
        embedding_function=mock_embeddings,
        collection_name="test_collection",
        chunk_size=100,
        chunk_overlap=20
    )
    store.chroma_store = mock_chroma  # Ensure we use the mock
    return store

class TestChromaDocumentStore:
    """
    Test cases for ChromaDocumentStore
    ChromaDocumentStoreのテストケース
    """

    def test_initialization(self, test_store):
        """
        Test store initialization
        ストアの初期化をテスト
        """
        assert test_store.chunk_size == 100
        assert test_store.chunk_overlap == 20

    def test_add_documents(self, test_store):
        """
        Test adding documents
        ドキュメントの追加をテスト
        """
        # Prepare test documents
        docs = [
            Document(
                page_content="Test document 1\nWith multiple lines\nFor splitting",
                metadata={"source": "test1.txt"}
            ),
            Document(
                page_content="Test document 2\nAlso with\nMultiple lines",
                metadata={"source": "test2.txt"}
            )
        ]

        # Add documents
        parent_ids = test_store.add_documents(docs)

        # Verify parent IDs were returned
        assert len(parent_ids) == 2
        assert all(isinstance(id_, str) for id_ in parent_ids)

        # Verify documents were added to Chroma
        add_docs_call = test_store.chroma_store.add_documents.call_args
        assert add_docs_call is not None
        added_docs = add_docs_call.args[0]

        # Verify split documents were added
        assert len(added_docs) > len(docs)  # Should have splits
        
        # Verify metadata structure
        split_docs = [doc for doc in added_docs if doc.metadata.get('is_split')]
        parent_docs = [doc for doc in added_docs if not doc.metadata.get('is_split')]
        
        assert len(parent_docs) == len(docs)
        assert len(split_docs) > 0
        
        # Check parent docs metadata
        for doc in parent_docs:
            assert doc.metadata.get('is_parent') is True
            assert 'split_count' in doc.metadata
            assert doc.metadata['split_count'] > 0
            
        # Check split docs metadata
        for doc in split_docs:
            assert 'parent_id' in doc.metadata
            assert doc.metadata['is_split'] is True

    def test_get_document(self, test_store):
        """
        Test retrieving document by ID
        IDによる文書取得をテスト
        """
        # Mock document
        mock_doc = Document(
            page_content="Test document",
            metadata={"id": "doc1"}
        )
        test_store.chroma_store.similarity_search.return_value = [mock_doc]

        # Test get document
        result = test_store.get_document("doc1")
        
        # Verify similarity search was called correctly
        test_store.chroma_store.similarity_search.assert_called_with(
            query="",
            k=1,
            filter={"id": "doc1"}
        )
        
        assert result == mock_doc

    def test_get_parent_document(self, test_store):
        """
        Test retrieving parent document
        親ドキュメントの取得をテスト
        """
        # Add a test document
        doc = Document(
            page_content="Test document\nWith multiple lines\nFor splitting",
            metadata={"source": "test.txt"}
        )
        
        # Add document and get parent ID
        [parent_id] = test_store.add_documents([doc])
        
        # Mock parent document for retrieval
        mock_parent = Document(
            page_content=doc.page_content,
            metadata={
                "id": parent_id,
                "source": "test.txt",
                "is_parent": True,
                "split_count": 3  # Example split count
            }
        )
        test_store.chroma_store.similarity_search.return_value = [mock_parent]
        
        # Get parent document
        result = test_store.get_parent_document(parent_id)
        
        # Verify result
        assert result is not None
        assert result.metadata['id'] == parent_id
        assert result.metadata['split_count'] > 0

    def test_get_split_documents(self, test_store):
        """
        Test retrieving split documents
        分割ドキュメントの取得をテスト
        """
        parent_id = "test_parent_1"
        
        # Mock split documents
        mock_splits = [
            Document(
                page_content=f"Split {i}",
                metadata={
                    "id": f"{parent_id}_split_{i}",
                    "parent_id": parent_id,
                    "is_split": True
                }
            )
            for i in range(3)
        ]
        test_store.chroma_store.similarity_search.return_value = mock_splits

        # Get splits
        results = test_store.get_split_documents(parent_id)
        
        # Verify similarity search was called correctly
        test_store.chroma_store.similarity_search.assert_called_with(
            query="",
            k=100,
            filter={
                "parent_id": parent_id,
                "is_split": True
            }
        )
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.metadata['parent_id'] == parent_id
            assert result.metadata['is_split'] is True

    def test_as_retriever(self, test_store):
        """
        Test getting retriever interface
        リトリーバーインターフェースの取得をテスト
        """
        # Mock retriever
        mock_retriever = MagicMock()
        test_store.chroma_store.as_retriever.return_value = mock_retriever
        
        # Get retriever with custom parameters
        search_kwargs = {"k": 5}
        retriever = test_store.as_retriever(**search_kwargs)
        
        # Verify as_retriever was called with parameters
        test_store.chroma_store.as_retriever.assert_called_with(**search_kwargs)
        assert retriever == mock_retriever 