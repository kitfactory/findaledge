"""
Test module for FAISS document store
FAISSドキュメントストアのテストモジュール
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever

# Use src prefix for imports
from src.findaledge.document_store.faiss import FAISSDocumentStore

class MockEmbeddings(Embeddings):
    """Mock embeddings for testing"""
    def embed_documents(self, texts):
        """Mock embed_documents method"""
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text):
        """Mock embed_query method"""
        return [0.1, 0.2, 0.3]

@pytest.fixture
def mock_faiss(mocker):
    """Mock the FAISS class methods used during initialization"""
    mock = mocker.MagicMock(spec=FAISS) # Use spec for better mocking

    # Configure mock attributes needed by the tests
    # Configure docstore attribute and its _dict
    mock.docstore = mocker.MagicMock()
    mock.docstore._dict = {} # Mock the internal docstore dictionary

    # Configure index_to_docstore_id
    mock.index_to_docstore_id = {} # Mock the index mapping

    # Configure methods if they need specific return values or side effects for tests
    mock.add_embeddings = mocker.MagicMock()
    mock.delete = mocker.MagicMock()
    mock.save_local = mocker.MagicMock()
    mock_retriever = mocker.MagicMock(spec=BaseRetriever) # Add spec for retriever
    mock.as_retriever.return_value = mock_retriever

    return mock

@pytest.fixture
def store(tmp_path):
    """Creates a basic FAISSDocumentStore instance for testing."""
    # This fixture now *only* creates the store instance.
    # Patching and assigning the internal mock is done in tests.
    persist_dir = tmp_path / "faiss_test_store"
    store_instance = FAISSDocumentStore(
        embedding_function=MockEmbeddings(),
        persist_directory=str(persist_dir)
    )
    # NOTE: store_instance.faiss_store will be initialized with a real FAISS object
    # (likely from the dummy data in _ensure_storage) by default.
    # Tests needing to mock the internal store must set store.faiss_store = mock_faiss.
    return store_instance

# --- Test Cases ---

def test_initialization(mock_faiss, tmp_path):
    """Test store initialization without persistence (mocking FAISS class)."""
    persist_dir = tmp_path / "faiss_init_no_persist"
    assert not persist_dir.exists()

    with patch('src.findaledge.document_store.faiss.FAISS', autospec=True) as mock_faiss_class:
        # Configure the class mock
        mock_faiss_class.from_embeddings.return_value = mock_faiss # Return our instance mock

        # Initialize the store *within the patch context*
        store = FAISSDocumentStore(
            embedding_function=MockEmbeddings(),
            persist_directory=str(persist_dir)
        )

        # Check that _ensure_storage called from_embeddings
        mock_faiss_class.from_embeddings.assert_called_once()
        mock_faiss_class.load_local.assert_not_called()
        # Check that the store instance holds the mock returned by the patched class
        assert store.faiss_store is mock_faiss

def test_initialization_with_persistence(mock_faiss, tmp_path):
    """Test store initialization with persistence (mocking FAISS class)."""
    persist_dir = tmp_path / "faiss_init_persist"
    persist_dir.mkdir() # Directory exists

    with patch('src.findaledge.document_store.faiss.FAISS', autospec=True) as mock_faiss_class:
        # Configure the class mock
        mock_faiss_class.load_local.return_value = mock_faiss # Return our instance mock

        # Initialize the store *within the patch context*
        store = FAISSDocumentStore(
            embedding_function=MockEmbeddings(),
            persist_directory=str(persist_dir),
            allow_dangerous_deserialization=False
        )

        # Check that _ensure_storage called load_local
        mock_faiss_class.load_local.assert_called_once_with(
            str(persist_dir),
            store.embedding_function,
            allow_dangerous_deserialization=False
        )
        mock_faiss_class.from_embeddings.assert_not_called()
        # Check that the store instance holds the mock returned by the patched class
        assert store.faiss_store is mock_faiss

def test_add_documents(store, mock_faiss):
    """Test adding documents"""
    # Assign the mock to the store instance for this test
    store.faiss_store = mock_faiss

    docs = [
        Document(page_content="doc1", metadata={"id": "d1"}),
        Document(page_content="doc2", metadata={"id": "d2", "source": "s2"}),
    ]
    added_ids = store.add_documents(docs)

    assert added_ids == ["d1", "d2"]
    mock_faiss.add_embeddings.assert_called_once()
    # Inspect call arguments
    # FAISS add_embeddings args: text_embeddings, metadatas
    call_args, call_kwargs = mock_faiss.add_embeddings.call_args
    assert len(call_kwargs['text_embeddings']) == 2
    assert call_kwargs['text_embeddings'][0][0] == "doc1"
    assert isinstance(call_kwargs['text_embeddings'][0][1], list)
    assert len(call_kwargs['metadatas']) == 2
    assert call_kwargs['metadatas'][0]["id"] == "d1"
    assert call_kwargs['metadatas'][1]["id"] == "d2"

    if store.persist_directory:
        mock_faiss.save_local.assert_called_once_with(store.persist_directory)
    else:
        mock_faiss.save_local.assert_not_called()

def test_get_document(store, mock_faiss):
    """Test getting a document"""
    store.faiss_store = mock_faiss # Assign mock
    doc_id = "doc123"
    expected_doc = Document(page_content="content", metadata={"id": doc_id})
    mock_faiss.docstore._dict = {0: expected_doc}

    retrieved_doc = store.get_document(doc_id)

    assert retrieved_doc is expected_doc

def test_get_document_not_found(store, mock_faiss):
    """Test getting a non-existent document"""
    store.faiss_store = mock_faiss # Assign mock
    mock_faiss.docstore._dict = {0: Document(page_content="other", metadata={"id": "other_id"})}

    retrieved_doc = store.get_document("not_found_id")

    assert retrieved_doc is None

def test_get_split_documents(store, mock_faiss):
    """Test getting split documents"""
    store.faiss_store = mock_faiss # Assign mock
    parent_id = "parent1"
    split1 = Document(page_content="split1", metadata={"id": "s1", "parent_id": parent_id, "is_split": True})
    split2 = Document(page_content="split2", metadata={"id": "s2", "parent_id": parent_id, "is_split": True})
    other_doc = Document(page_content="other", metadata={"id": "other", "parent_id": "other_parent"})
    parent_doc = Document(page_content="parent", metadata={"id": parent_id, "is_parent": True})

    mock_faiss.docstore._dict = {
        0: split1,
        1: split2,
        2: other_doc,
        3: parent_doc
    }

    splits = store.get_split_documents(parent_id)

    assert len(splits) == 2
    assert split1 in splits
    assert split2 in splits

def test_delete_document(store, mock_faiss):
    """Test deleting a document and its splits"""
    store.faiss_store = mock_faiss # Assign mock
    parent_id = "parent_del"
    split_id = "split_del"
    other_id = "other_del"

    parent_doc = Document(page_content="parent", metadata={"id": parent_id})
    split_doc = Document(page_content="split", metadata={"id": split_id, "parent_id": parent_id})
    other_doc = Document(page_content="other", metadata={"id": other_id})

    mock_faiss.docstore._dict = {
        0: parent_doc,
        1: split_doc,
        2: other_doc
    }
    mock_faiss.index_to_docstore_id = {0: "faiss_id_0", 1: "faiss_id_1", 2: "faiss_id_2"}

    store.delete_document(parent_id)

    mock_faiss.delete.assert_called_once_with(["faiss_id_0", "faiss_id_1"])
    if store.persist_directory:
        mock_faiss.save_local.assert_called_once_with(store.persist_directory)
    else:
        mock_faiss.save_local.assert_not_called()

def test_as_retriever(store, mock_faiss):
    """Test getting the retriever"""
    store.faiss_store = mock_faiss # Assign mock
    retriever = store.as_retriever(search_type="mmr")
    mock_faiss.as_retriever.assert_called_once_with(search_type="mmr")
    assert retriever is mock_faiss.as_retriever.return_value

def test_list_documents(store, mock_faiss):
    """Test listing documents (should return only parents or unsplit docs)"""
    store.faiss_store = mock_faiss # Assign mock
    parent1 = Document(page_content="p1", metadata={"id": "p1", "is_parent": True})
    split1 = Document(page_content="s1", metadata={"id": "s1", "parent_id": "p1", "is_split": True})
    unsplit1 = Document(page_content="u1", metadata={"id": "u1"})

    mock_faiss.docstore._dict = {0: parent1, 1: split1, 2: unsplit1}

    listed_docs = store.list_documents()

    assert len(listed_docs) == 2
    assert parent1 in listed_docs
    assert unsplit1 in listed_docs

# Commented out test_update_document
# def test_update_document(store, mock_faiss):
# ...

def test_persistence_error_handling(tmp_path):
    """Test handling of persistence load errors (mocking FAISS class)."""
    persist_dir = tmp_path / "faiss_error_persist"
    persist_dir.mkdir() # Directory exists

    with patch('src.findaledge.document_store.faiss.FAISS', autospec=True) as mock_faiss_class_local:
        # Configure mocks for this test's FAISSDocumentStore creation
        mock_faiss_class_local.load_local.side_effect = Exception("Simulated load error")
        # Configure from_embeddings to return a specific mock
        mock_fallback_instance = MagicMock(spec=FAISS)
        mock_faiss_class_local.from_embeddings.return_value = mock_fallback_instance

        store_instance = FAISSDocumentStore(
            embedding_function=MockEmbeddings(),
            persist_directory=str(persist_dir),
            allow_dangerous_deserialization=True
        )

        # Assert the behavior after the failed load attempt
        assert store_instance.faiss_store is mock_fallback_instance # Should have fallen back
        mock_faiss_class_local.load_local.assert_called_once() # Verify load was attempted
        mock_faiss_class_local.from_embeddings.assert_called_once() # Verify fallback occurred 