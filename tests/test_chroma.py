"""
Test module for chroma.py
chroma.pyのテストモジュール
"""

import pytest
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import unittest
import shutil
import time
from typing import List

# Use correct Document import based on langchain version
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document # Older langchain
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStoreRetriever # For type hint
# Import Chroma from the correct community path
from langchain_community.vectorstores import Chroma

from finderledge.document_store.chroma import ChromaDocumentStore

# Mock Chroma client directly if needed for specific tests, otherwise rely on patching Chroma class
# from chromadb import Client, Collection

# Constants for testing
PERSIST_DIR = "./test_chroma_persist"
COLLECTION_NAME = "test_collection"
DOC_ID_1 = "doc1"
DOC_ID_2 = "doc2"
DOC_CONTENT_1 = "This is document 1."
DOC_CONTENT_2 = "This is document 2."
EMBEDDING_1 = [0.1, 0.2, 0.3]
EMBEDDING_2 = [0.4, 0.5, 0.6]
METADATA_1 = {"source": "file1.txt"}
METADATA_2 = {"source": "file2.txt"}

@pytest.fixture
def mock_embeddings(mocker):
    """Fixture for mocking Embeddings.
    文書の埋め込みをモックするフィクスチャ。
    """
    mock = MagicMock(spec=Embeddings)
    # Setup mock embeddings to return different vectors for different texts
    def embed_documents(texts):
        embeddings = []
        for text in texts:
            if DOC_CONTENT_1 in text:
                embeddings.append(EMBEDDING_1)
            elif DOC_CONTENT_2 in text:
                embeddings.append(EMBEDDING_2)
            else:
                embeddings.append([0.0] * 3) # Default embedding
        return embeddings

    mock.embed_documents.side_effect = embed_documents
    mock.embed_query.return_value = [0.7, 0.8, 0.9] # Consistent query embedding
    return mock

@pytest.fixture
def mock_chroma_client(mocker):
    """Fixture for mocking the chromadb Client and Collection.
    chromadb Client と Collection をモックするフィクスチャ。
    """
    mock_collection = MagicMock() # spec=Collection removed as import is complex
    mock_collection.add.return_value = None
    mock_collection.delete.return_value = None
    # Simulate query results including embeddings (optional but useful)
    mock_collection.query.return_value = {
        "ids": [[DOC_ID_1]],
        "documents": [[DOC_CONTENT_1]],
        "metadatas": [[METADATA_1]],
        "embeddings": [[EMBEDDING_1]], # Optional: Include if needed
        "distances": [[0.1]]
    }
    mock_collection.get.return_value = {
        "ids": [DOC_ID_1, DOC_ID_2],
        "documents": [DOC_CONTENT_1, DOC_CONTENT_2],
        "metadatas": [METADATA_1, METADATA_2],
        "embeddings": [EMBEDDING_1, EMBEDDING_2], # Optional
    }

    mock_client = MagicMock() # spec=Client removed
    # Configure client methods if ChromaDocumentStore interacts with them directly (e.g., list_collections)
    mock_client.get_or_create_collection.return_value = mock_collection

    # Patch the chromadb Client constructor
    mocker.patch('chromadb.Client', return_value=mock_client)
    return mock_client, mock_collection

@pytest.fixture
def mock_chroma():
    """
    Mock Chroma vector document store
    Chromaベクトルドキュメントストアのモック
    """
    with patch('finderledge.document_store.chroma.Chroma') as mock:
        mock_instance = mock.return_value
        # Mock similarity_search method
        mock_instance.similarity_search.return_value = []
        # Mock add_documents method
        mock_instance.add_documents = Mock()
        yield mock_instance

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

class TestChromaDocumentStore(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_chroma_persist"
        # Ensure directory is removed before creating
        if os.path.exists(self.test_dir):
            # Retry removal with a small delay on Windows
            try:
                shutil.rmtree(self.test_dir)
            except OSError:
                time.sleep(0.1)
                try:
                    shutil.rmtree(self.test_dir, ignore_errors=True)
                except Exception as e:
                    print(f"Warning: Could not remove test dir {self.test_dir} during setup: {e}")
        # Create directory, allowing it to exist
        os.makedirs(self.test_dir, exist_ok=True)
        self.mock_embeddings = MagicMock(spec=Embeddings)
        
        # Configure mock to return embeddings based on input length
        def mock_embed_documents(texts: List[str]) -> List[List[float]]:
            # Return a list of dummy embeddings with the same length as input texts
            return [[0.1, 0.2, 0.3]] * len(texts)
        
        self.mock_embeddings.embed_documents.side_effect = mock_embed_documents
        self.mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        
        self.store = ChromaDocumentStore(
            persist_directory=self.test_dir,
            embedding_function=self.mock_embeddings,
            collection_name=f"test_collection_{time.time_ns()}"
        )

    def tearDown(self):
        # Try to clean up the store and directory
        if hasattr(self, 'store') and self.store and hasattr(self.store, 'chroma_store'):
            try:
                # Accessing private _client might be fragile, consider if Chroma offers a close() method
                if hasattr(self.store.chroma_store, '_client'):
                     # Attempt to reset/close client if possible (adapt based on actual Chroma API)
                     # self.store.chroma_store._client.reset() # Example, adjust based on API
                     pass # No standard close method found easily
            except Exception as e:
                print(f"Warning: Error closing Chroma client: {e}")
        self.store = None 
        if os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir)
            except OSError:
                time.sleep(0.1) # Small delay before retrying removal
                try:
                    shutil.rmtree(self.test_dir, ignore_errors=True)
                except Exception as e:
                    print(f"Warning: Could not remove test dir {self.test_dir} during teardown: {e}")

    def test_initialization(self):
        """Test ChromaDocumentStore initialization"""
        self.assertIsNotNone(self.store)
        self.assertEqual(self.store.persist_directory, self.test_dir)
        self.assertEqual(self.store.embedding_function, self.mock_embeddings)
        # Check if the actual Chroma store was initialized (no longer mocking by default)
        self.assertIsNotNone(self.store.chroma_store) 
        # Optionally check collection name if needed
        # self.assertTrue(self.store.collection_name.startswith("test_collection_"))

    def test_add_documents(self):
        """Test adding documents and verify splitting/metadata"""
        # Use longer content to ensure splitting with default chunk size
        long_content = "This is document 1. " * 200 + "Split point. " + "More content for doc 1. " * 200
        short_content = "Doc 2 Content."
        docs = [
            Document(page_content=long_content, metadata={"source": "s1", "id": "doc1"}),
            Document(page_content=short_content, metadata={"source": "s2", "id": "doc2"})
        ]

        parent_ids = self.store.add_documents(docs)

        self.assertIsInstance(parent_ids, list)
        self.assertEqual(len(parent_ids), 2)
        self.assertIn("doc1", parent_ids)
        self.assertIn("doc2", parent_ids)

        # Verify splits for doc1 were created
        splits_doc1 = self.store.get_split_documents("doc1")
        self.assertTrue(len(splits_doc1) >= 2) # Expect at least two splits now
        for split in splits_doc1:
            self.assertEqual(split.metadata['parent_id'], "doc1")
            self.assertTrue(split.metadata['is_split'])
            self.assertEqual(split.metadata['source'], "s1") # Original metadata preserved

        # Verify doc2 (original doc should be retrievable)
        retrieved_doc2 = self.store.get_document("doc2")
        self.assertIsNotNone(retrieved_doc2)
        self.assertEqual(retrieved_doc2.metadata.get("source"), "s2")
        # Verify doc2 was not split
        splits_doc2 = self.store.get_split_documents("doc2")
        self.assertEqual(len(splits_doc2), 0)

    def test_get_document(self):
        """Test getting a document (parent)"""
        doc_id = "get_doc_test_1"
        doc_content = "Get me!"
        docs = [Document(page_content=doc_content, metadata={"id": doc_id})]
        added_ids = self.store.add_documents(docs)
        self.assertIn(doc_id, added_ids)

        retrieved_doc = self.store.get_document(doc_id)

        self.assertIsNotNone(retrieved_doc)
        self.assertEqual(retrieved_doc.page_content, doc_content)
        self.assertEqual(retrieved_doc.metadata['id'], doc_id)
        # It should retrieve the parent doc, which might have is_parent=True if splits happened
        # self.assertTrue(retrieved_doc.metadata.get('is_parent', False) or not retrieved_doc.metadata.get('is_split'))

    def test_get_parent_document(self):
        """Test getting a parent document explicitly"""
        parent_id = "get_parent_test_1"
        # Use longer content to ensure splitting
        doc_content = "This is the parent content that needs splitting. " * 200 + "End of parent."
        docs = [Document(page_content=doc_content, metadata={"id": parent_id})]
        self.store.add_documents(docs)

        # Wait briefly to ensure persistence (might help in some race conditions)
        time.sleep(0.1)

        retrieved_doc = self.store.get_parent_document(parent_id)

        self.assertIsNotNone(retrieved_doc, f"Parent document {parent_id} not found.")
        if retrieved_doc: # Check added to avoid AttributeError on None
            self.assertEqual(retrieved_doc.page_content, doc_content)
            self.assertEqual(retrieved_doc.metadata['id'], parent_id)
            self.assertTrue(retrieved_doc.metadata.get('is_parent'), "is_parent metadata missing or False")

    def test_get_split_documents(self):
        """Test getting split documents"""
        parent_id = "get_splits_test_1"
        # Use longer content to ensure splitting
        doc_content = "Content designed to be split into multiple chunks. " * 300 + "Final chunk."
        docs = [Document(page_content=doc_content, metadata={"id": parent_id})]
        self.store.add_documents(docs)

        retrieved_splits = self.store.get_split_documents(parent_id)

        self.assertTrue(len(retrieved_splits) > 1, "Document was not split as expected.") # Expect more than 1 split
        for split in retrieved_splits:
            self.assertEqual(split.metadata['parent_id'], parent_id)
            self.assertTrue(split.metadata['is_split'])

    # Patch the internal store's as_retriever for this specific test
    @patch.object(Chroma, 'as_retriever') 
    def test_as_retriever(self, mock_as_retriever):
        """Test getting the store as a retriever"""
        mock_retriever_instance = MagicMock(spec=VectorStoreRetriever)
        mock_as_retriever.return_value = mock_retriever_instance

        # Replace the store's internal chroma_store's as_retriever with the mock
        # This is tricky as ChromaDocumentStore initializes its own Chroma instance.
        # Instead, let's patch Chroma.as_retriever directly

        retriever = self.store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        
        # Assert that the *Chroma class's* as_retriever was called with the store's instance
        mock_as_retriever.assert_called_once_with(search_type="similarity", search_kwargs={"k": 5})
        self.assertEqual(retriever, mock_retriever_instance) 