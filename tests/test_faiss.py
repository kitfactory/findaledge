"""
Test module for FAISS document store
FAISSドキュメントストアのテストモジュール
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY
from langchain.schema import Document
from findaledge.document_store.faiss import FAISSDocumentStore

class MockEmbeddings:
    """Mock embeddings for testing"""
    def embed_documents(self, texts):
        """Mock embed_documents method"""
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text):
        """Mock embed_query method"""
        return [0.1, 0.2, 0.3]

@pytest.fixture
def mock_faiss():
    """Create mock FAISS instance"""
    mock = MagicMock()
    mock.docstore = MagicMock()
    mock.docstore._dict = {}
    mock.index_to_docstore_id = {}
    mock.add_embeddings = MagicMock()
    mock.delete = MagicMock()
    mock.save_local = MagicMock()
    mock_retriever = MagicMock()
    mock.as_retriever.return_value = mock_retriever
    return mock

@pytest.fixture
def store(mock_faiss, tmp_path):
    """Create test document store instance"""
    persist_dir = tmp_path / "faiss_index"
    with patch('findaledge.document_store.faiss.FAISS', autospec=True) as mock_faiss_class:
        mock_faiss_class.load_local.return_value = mock_faiss
        mock_faiss_class.from_embeddings.return_value = mock_faiss

        store = FAISSDocumentStore(
            embedding_function=MockEmbeddings(),
            persist_directory=str(persist_dir)
        )
        assert store.faiss_store is mock_faiss
        return store

def test_initialization(mock_faiss, tmp_path):
    """Test store initialization without persistence"""
    persist_dir = tmp_path / "faiss_index"
    assert not persist_dir.exists()

    with patch('findaledge.document_store.faiss.FAISS', autospec=True) as mock_faiss_class:
        mock_faiss_class.from_embeddings.return_value = mock_faiss
        store = FAISSDocumentStore(
            embedding_function=MockEmbeddings(),
            persist_directory=str(persist_dir)
        )
        assert store.faiss_store is mock_faiss
        assert store.persist_directory == str(persist_dir)
        mock_faiss_class.from_embeddings.assert_called_once()
        mock_faiss_class.load_local.assert_not_called()

def test_initialization_with_persistence(mock_faiss, tmp_path):
    """Test store initialization with persistence"""
    persist_dir = tmp_path / "faiss_index"
    persist_dir.mkdir()

    with patch('findaledge.document_store.faiss.FAISS', autospec=True) as mock_faiss_class:
        mock_faiss_class.load_local.return_value = mock_faiss
        store = FAISSDocumentStore(
            embedding_function=MockEmbeddings(),
            persist_directory=str(persist_dir),
            allow_dangerous_deserialization=True
        )
        assert store.faiss_store is mock_faiss
        assert store.persist_directory == str(persist_dir)
        mock_faiss_class.load_local.assert_called_once_with(
            str(persist_dir),
            ANY,
            allow_dangerous_deserialization=True
        )
        mock_faiss_class.from_embeddings.assert_not_called()

def test_add_documents(store):
    """Test adding documents"""
    docs = [
        Document(page_content="content1", metadata={"id": "test1"}),
        Document(page_content="content2", metadata={"id": "test2"})
    ]
    expected_ids = ["test1", "test2"]
    returned_ids = store._add_documents(docs)
    assert returned_ids == expected_ids

    store.faiss_store.add_embeddings.assert_called_once()
    call_args, call_kwargs = store.faiss_store.add_embeddings.call_args
    assert len(call_kwargs['text_embeddings']) == 2
    assert call_kwargs['text_embeddings'][0][0] == "content1"
    assert call_kwargs['text_embeddings'][1][0] == "content2"
    assert len(call_kwargs['metadatas']) == 2
    assert call_kwargs['metadatas'][0] == {"id": "test1", "parent_id": "test1"}
    assert call_kwargs['metadatas'][1] == {"id": "test2", "parent_id": "test2"}

    store.faiss_store.save_local.assert_called_once_with(store.persist_directory)

def test_get_document(store):
    """Test getting a document by ID"""
    doc_id = "test1"
    target_doc = Document(page_content="content1", metadata={"id": doc_id, "parent_id": doc_id})
    other_doc = Document(page_content="content2", metadata={"id": "other", "parent_id": "other"})
    store.faiss_store.docstore._dict = {
        0: target_doc,
        1: other_doc
    }
    result = store._get_document(doc_id)
    assert result is target_doc

def test_get_document_not_found(store):
    """Test getting a non-existent document"""
    store.faiss_store.docstore._dict = {
        0: Document(page_content="content1", metadata={"id": "other", "parent_id": "other"})
    }
    result = store._get_document("nonexistent")
    assert result is None

def test_get_split_documents(store):
    """Test getting split documents by parent ID"""
    parent_id = "parent1"
    parent_doc = Document(page_content="parent content", metadata={"id": parent_id, "parent_id": parent_id})
    split1 = Document(page_content="split content 1", metadata={"id": "split1", "parent_id": parent_id})
    split2 = Document(page_content="split content 2", metadata={"id": "split2", "parent_id": parent_id})
    other_doc = Document(page_content="other content", metadata={"id": "other", "parent_id": "other"})

    store.faiss_store.docstore._dict = {
        0: parent_doc,
        1: split1,
        2: split2,
        3: other_doc
    }

    result = store._get_split_documents(parent_id)
    assert len(result) == 2
    assert split1 in result
    assert split2 in result
    assert parent_doc not in result
    assert other_doc not in result

def test_delete_document(store):
    """Test deleting a document and its splits"""
    parent_id = "parent_to_delete"
    split_id = "split_to_delete"

    parent_doc = Document(page_content="parent content", metadata={"id": parent_id, "parent_id": parent_id})
    split_doc = Document(page_content="split content", metadata={"id": split_id, "parent_id": parent_id})
    other_doc = Document(page_content="other content", metadata={"id": "other", "parent_id": "other"})

    store.faiss_store.docstore._dict = {
        0: parent_doc,
        1: split_doc,
        2: other_doc
    }
    store.faiss_store.index_to_docstore_id = {
        0: "faiss_internal_0",
        1: "faiss_internal_1",
        2: "faiss_internal_2"
    }

    store.delete_document(parent_id)

    store.faiss_store.delete.assert_called_once_with(["faiss_internal_0", "faiss_internal_1"])
    store.faiss_store.save_local.assert_called_once_with(store.persist_directory)

def test_as_retriever(store):
    """Test getting the retriever"""
    retriever = store.as_retriever(search_type="similarity")
    assert retriever is store.faiss_store.as_retriever.return_value
    store.faiss_store.as_retriever.assert_called_once_with(search_type="similarity")

def test_list_documents(store):
    """Test listing parent documents"""
    parent1 = Document(page_content="p1", metadata={"id": "p1", "parent_id": "p1"})
    split1 = Document(page_content="s1", metadata={"id": "s1", "parent_id": "p1"})
    parent2 = Document(page_content="p2", metadata={"id": "p2", "parent_id": "p2"})

    store.faiss_store.docstore._dict = {
        0: parent1,
        1: split1,
        2: parent2
    }
    result = store.list_documents()
    assert len(result) == 2
    assert parent1 in result
    assert parent2 in result
    assert split1 not in result

def test_update_document(store):
    """Test updating a document"""
    doc_id_to_update = "doc_to_update"
    old_doc = Document(page_content="old content", metadata={"id": doc_id_to_update, "parent_id": doc_id_to_update})
    new_doc_data = Document(page_content="new content", metadata={})

    store.delete_document = MagicMock()
    store._add_documents = MagicMock(return_value=[doc_id_to_update])

    store.update_document(doc_id_to_update, new_doc_data)

    store.delete_document.assert_called_once_with(doc_id_to_update)
    store._add_documents.assert_called_once()
    call_args, call_kwargs = store._add_documents.call_args
    added_doc = call_args[0][0]
    assert isinstance(added_doc, Document)
    assert added_doc.page_content == "new content"
    assert added_doc.metadata["id"] == doc_id_to_update

def test_persistence_error_handling(mock_faiss, tmp_path):
    """Test handling of persistence load errors"""
    persist_dir = tmp_path / "faiss_index"
    persist_dir.mkdir()

    with patch('findaledge.document_store.faiss.FAISS', autospec=True) as mock_faiss_class:
        mock_faiss_class.load_local.side_effect = Exception("Load error")
        mock_faiss_class.from_embeddings.return_value = mock_faiss

        store = FAISSDocumentStore(
            embedding_function=MockEmbeddings(),
            persist_directory=str(persist_dir),
            allow_dangerous_deserialization=True
        )
        assert store.faiss_store is mock_faiss
        mock_faiss_class.load_local.assert_called_once()
        mock_faiss_class.from_embeddings.assert_called_once() 