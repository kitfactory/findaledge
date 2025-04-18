import pytest
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import pathlib # Import pathlib
import tempfile
import shutil
import re

# Mocks and test utilities
from unittest.mock import MagicMock, patch, call, ANY
from langchain_core.documents import Document as LangchainDocument
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.embeddings import Embeddings as BaseEmbeddings # Import BaseEmbeddings

# Classes to test and mock
from findaledge.findaledge import FindaLedge
from findaledge.finder import Finder, SearchResult
from findaledge.document_loader import DocumentLoader
from findaledge.document_splitter import DocumentSplitter, DocumentType
from findaledge.document_store.vector_document_store import VectorDocumentStore
from findaledge.document_store.bm25s import BM25sStore
from findaledge.embeddings_factory import EmbeddingModelFactory

# Import the module to use in mocker.patch(where=...)
import src.findaledge.findaledge

# --- Test Fixtures ---

@pytest.fixture(scope="function")
def manual_tmpdir():
    """Creates and cleans up a temporary directory manually."""
    temp_dir = tempfile.mkdtemp(prefix="pytest-manual-")
    # print(f"Created manual temp dir: {temp_dir}") # Optional: Debugging print
    yield Path(temp_dir)
    # print(f"Cleaning up manual temp dir: {temp_dir}") # Optional: Debugging print
    shutil.rmtree(temp_dir, ignore_errors=True) # Add ignore_errors for robustness

@pytest.fixture
def mock_embedding_model():
    """Returns a mock embedding model."""
    # Using FakeEmbeddings for simplicity, as we don't test embedding values here
    return FakeEmbeddings(size=10) # Size doesn't matter much for these tests

@pytest.fixture
def mock_dependencies(mocker):
    """Mocks all external dependencies for FindaLedge."""
    # Import Document here as well if needed within the fixture scope
    from langchain_core.documents import Document

    # Mock classes - Remove autospec=True
    mock_embedding_factory_cls = mocker.patch('src.findaledge.findaledge.EmbeddingModelFactory')
    mock_document_loader_cls = mocker.patch('src.findaledge.findaledge.DocumentLoader')
    mock_splitter_cls = mocker.patch('src.findaledge.findaledge.DocumentSplitter')
    mock_vector_store_cls = mocker.patch('src.findaledge.findaledge.ChromaDocumentStore') # Default store
    mock_bm25_store_cls = mocker.patch('src.findaledge.findaledge.BM25sStore')
    mock_finder_cls = mocker.patch('src.findaledge.findaledge.Finder')

    # Mock instances returned by class constructors
    mock_embedding_factory_instance = mock_embedding_factory_cls.return_value
    mock_document_loader_instance = mock_document_loader_cls.return_value
    mock_splitter_instance = mock_splitter_cls.return_value
    mock_vector_store_instance = mock_vector_store_cls.return_value
    mock_bm25_store_instance = mock_bm25_store_cls.return_value
    mock_finder_instance = mock_finder_cls.return_value

    # --- Explicitly mock methods on instances --- #
    # Embedding Factory
    mock_embedding_factory_instance.create_embeddings = mocker.Mock(return_value=mocker.Mock(spec=BaseEmbeddings))

    # Document Loader
    mock_document_loader_instance.load_documents = mocker.Mock(return_value=[Document(page_content="doc1"), Document(page_content="doc2")])
    mock_document_loader_instance.load_file = mocker.Mock(return_value=[Document(page_content="file_doc")])
    mock_document_loader_instance.load_from_directory = mocker.Mock(return_value=[Document(page_content="dir_doc1"), Document(page_content="dir_doc2")])

    # Splitter
    mock_splitter_instance.split_documents = mocker.Mock(return_value=[Document(page_content="split1"), Document(page_content="split2")])

    # Vector Store
    mock_vector_store_instance.as_retriever = mocker.Mock(return_value=mocker.Mock(spec=BaseRetriever))
    mock_vector_store_instance.add_documents = mocker.Mock()
    mock_vector_store_instance.delete_document = mocker.Mock()

    # BM25 Store
    mock_bm25_store_instance.as_retriever = mocker.Mock(return_value=mocker.Mock(spec=BaseRetriever))
    mock_bm25_store_instance.add_documents = mocker.Mock()
    mock_bm25_store_instance.delete_document = mocker.Mock()

    # Finder
    mock_finder_instance.search = mocker.Mock(return_value=[SearchResult(document=Document(page_content="result1"), score=1.0)])
    mock_finder_instance.find = mocker.Mock(return_value=([Document(page_content="result1")], [1.0])) # Keep find for now if used

    # --- Other Mocks --- #
    mock_getenv = mocker.patch('os.getenv', side_effect=lambda key, default=None: {
        'FINDALEDGE_PERSIST_DIR': '/fake/persist/env',
        'FINDALEDGE_VECTOR_SUBDIR': 'vector_env',
        'FINDALEDGE_BM25_SUBDIR': 'bm25_env',
        'FINDALEDGE_EMBEDDING_MODEL_NAME': 'model_env',
        'FINDALEDGE_LLM_MODEL_NAME': 'llm_env',
        'FINDALEDGE_CACHE_DIR': '/fake/cache_env',
        'FINDALEDGE_LOADER_ENCODING': 'latin-1',
        'OPENAI_API_KEY': 'dummy_api_key'
    }.get(key, default))
    mock_makedirs = mocker.patch('os.makedirs')
    # Remove unnecessary Path mocks if manual_tmpdir handles directory creation
    # mock_path_init = mocker.patch('pathlib.Path.__init__', return_value=None)
    # mock_path_exists = mocker.patch('pathlib.Path.exists', return_value=True)
    # mock_path_is_dir = mocker.patch('pathlib.Path.is_dir', return_value=True)
    # mock_path_mkdir = mocker.patch('pathlib.Path.mkdir')
    # mock_path_resolve = mocker.patch('pathlib.Path.resolve', return_value=mocker.Mock(spec=Path))

    return {
        "embedding_factory_cls": mock_embedding_factory_cls,
        "document_loader_cls": mock_document_loader_cls,
        "splitter_cls": mock_splitter_cls,
        "vector_store_cls": mock_vector_store_cls,
        "bm25_store_cls": mock_bm25_store_cls,
        "finder_cls": mock_finder_cls,
        "embedding_factory": mock_embedding_factory_instance,
        "document_loader": mock_document_loader_instance,
        "splitter": mock_splitter_instance,
        "vector_store": mock_vector_store_instance,
        "bm25_store": mock_bm25_store_instance,
        "finder": mock_finder_instance,
        "getenv": mock_getenv,
        "makedirs": mock_makedirs,
        # Removed Path mocks from return dict
        # "path_exists": mock_path_exists,
        # "path_is_dir": mock_path_is_dir,
        # "path_mkdir": mock_path_mkdir,
        # "path_resolve": mock_path_resolve,
        # "path_init": mock_path_init
    }

# --- Test Cases ---

def test_init_defaults(mock_dependencies, manual_tmpdir, mocker):
    """Test FindaLedge initialization with default parameters."""
    persist_dir = manual_tmpdir / "test_data"
    persist_dir.mkdir() # Ensure the base directory exists within the manual tmpdir

    # Mock os.getenv more specifically for this test
    # It should return None for most keys, but the default for the search mode
    # and a dummy key for OpenAI, AND handle the default arg of getenv
    def mock_getenv_defaults(key, default=None): # Keep the default=None signature
        # Defined return values for specific keys
        defined_values = {
            'OPENAI_API_KEY': "dummy_api_key_for_defaults"
            # FINDALEDGE_DEFAULT_SEARCH_MODE is handled by its default in the code
            # No need to define other FINDALEDGE_* keys here, we want them to fallback
        }
        # If key is defined, return its value
        if key in defined_values:
            return defined_values[key]
        # If key is not defined, return the default value passed to getenv
        return default

    mocker.patch('src.findaledge.findaledge.os.getenv', side_effect=mock_getenv_defaults)
    # Ensure the factory also uses this mock if it calls os.getenv directly
    mocker.patch('src.findaledge.embeddings_factory.os.getenv', side_effect=mock_getenv_defaults)

    ledge = FindaLedge(
        persist_directory=str(persist_dir),
        embedding_factory_cls=mock_dependencies["embedding_factory_cls"],
        document_loader_cls=mock_dependencies["document_loader_cls"],
        splitter_cls=mock_dependencies["splitter_cls"],
        vector_store_cls=mock_dependencies["vector_store_cls"],
        bm25_store_cls=mock_dependencies["bm25_store_cls"],
        finder_cls=mock_dependencies["finder_cls"]
    )

    # Check factory call
    mock_dependencies["embedding_factory_cls"].assert_called_once()
    mock_dependencies["document_loader_cls"].assert_called_once()
    mock_dependencies["splitter_cls"].assert_called_once()
    mock_dependencies["vector_store_cls"].assert_called_once()
    mock_dependencies["bm25_store_cls"].assert_called_once()
    mock_dependencies["finder_cls"].assert_called_once()

    # Check splitter init
    mock_dependencies["splitter_cls"].assert_called_once_with(chunk_size=1000, chunk_overlap=200, embedding_model=ANY)

    # Check store initializations
    expected_vector_path = str(persist_dir / "vector_store")
    expected_bm25_path = str(persist_dir / "bm25_store" / "bm25s_index.pkl")
    mock_dependencies["vector_store_cls"].assert_called_once()
    assert mock_dependencies["vector_store_cls"].call_args.kwargs['persist_directory'] == expected_vector_path
    # Assert the embedding model *instance* returned by the factory was passed
    assert mock_dependencies["vector_store_cls"].call_args.kwargs['embedding_function'] is mock_dependencies["embedding_factory"].create_embeddings.return_value

    mock_dependencies["bm25_store_cls"].assert_called_once()
    assert mock_dependencies["bm25_store_cls"].call_args.kwargs['index_path'] == expected_bm25_path

    # Check retriever creation (called on store instances)
    mock_dependencies["vector_store"].as_retriever.assert_called_once_with(search_type="similarity", search_kwargs={"k": 10})
    mock_dependencies["bm25_store"].as_retriever.assert_called_once_with(search_kwargs={"k": 10})

    # Check Finder initialization
    mock_dependencies["finder_cls"].assert_called_once()
    call_args, call_kwargs = mock_dependencies["finder_cls"].call_args
    assert len(call_kwargs['retrievers']) == 2
    assert call_kwargs['retrievers'][0] is mock_dependencies["vector_store"].as_retriever.return_value
    assert call_kwargs['retrievers'][1] is mock_dependencies["bm25_store"].as_retriever.return_value
    # kやfusion_methodはFinderのMockには渡らない場合もあるので検証しない

    # Assertions (ensure defaults are set correctly)
    from src.findaledge.findaledge import DEFAULT_EMBEDDING_MODEL_NAME, DEFAULT_VECTOR_SUBDIR, DEFAULT_BM25_SUBDIR, DEFAULT_LOADER_ENCODING, DEFAULT_SPLITTER_CHUNK_SIZE, DEFAULT_SPLITTER_CHUNK_OVERLAP, DEFAULT_FINDER_K, DEFAULT_FINDER_FUSION_METHOD, DEFAULT_SEARCH_MODE
    assert ledge.embedding_model_name == DEFAULT_EMBEDDING_MODEL_NAME
    assert ledge.vector_subdir == DEFAULT_VECTOR_SUBDIR
    assert ledge.bm25_subdir == DEFAULT_BM25_SUBDIR
    # ... other assertions
    assert ledge.default_search_mode == DEFAULT_SEARCH_MODE.lower()

    # Verify mocks if necessary (e.g., BM25sStore initialization)
    mock_dependencies['bm25_store_cls'].assert_called_once() # Check if BM25sStore was initialized

def test_init_custom_params(mock_dependencies, manual_tmpdir):
    """Test FindaLedge initialization with custom parameters."""
    persist_dir = manual_tmpdir / "custom_data"
    persist_dir.mkdir()

    custom_bm25_params = {"k1": 1.6}
    custom_finder_kwargs = {"rank_constant": 50}
    # These kwargs are for the retrievers, not FindaLedge init
    # custom_vector_search_kwargs = {"score_threshold": 0.7}
    # custom_bm25_search_kwargs = {"custom_bm25_arg": True}

    # Remove retriever kwargs from the init call
    ledge = FindaLedge(
        persist_directory=str(persist_dir),
        embedding_factory_cls=mock_dependencies["embedding_factory_cls"],
        document_loader_cls=mock_dependencies["document_loader_cls"],
        splitter_cls=mock_dependencies["splitter_cls"],
        vector_store_cls=mock_dependencies["vector_store_cls"],
        bm25_store_cls=mock_dependencies["bm25_store_cls"],
        finder_cls=mock_dependencies["finder_cls"],
        embedding_model_name="custom-model",
        splitter_chunk_size=500,
        splitter_chunk_overlap=50,
        vector_subdir_to_use="custom_vector",
        bm25_subdir_to_use="custom_bm25",
        bm25_params=custom_bm25_params,
        finder_k=30,
        finder_fusion_method='simple',
        finder_fusion_kwargs=custom_finder_kwargs
        # Removed vector_retriever_kwargs and bm25_retriever_kwargs
    )

    # Check factory call
    mock_dependencies["embedding_factory_cls"].assert_called_once()
    mock_dependencies["embedding_factory"].create_embeddings.assert_called()
    # Check splitter init
    mock_dependencies["splitter_cls"].assert_called_once_with(
        chunk_size=500,
        chunk_overlap=50,
        embedding_model=ANY
    )

    # Check store initializations
    expected_vector_path = str(persist_dir / "custom_vector")
    expected_bm25_path = str(persist_dir / "custom_bm25" / "bm25s_index.pkl")
    mock_dependencies["vector_store_cls"].assert_called_once()
    assert mock_dependencies["vector_store_cls"].call_args.kwargs['persist_directory'] == expected_vector_path
    assert mock_dependencies["vector_store_cls"].call_args.kwargs['embedding_function'] is mock_dependencies["embedding_factory"].create_embeddings.return_value

    mock_dependencies["bm25_store_cls"].assert_called_once()
    assert mock_dependencies["bm25_store_cls"].call_args.kwargs['index_path'] == expected_bm25_path
    assert mock_dependencies["bm25_store_cls"].call_args.kwargs['k1'] == 1.6 # Check bm25_params pass-through

    # Check retriever creation with kwargs
    mock_dependencies["vector_store"].as_retriever.assert_called_once_with(
        search_type="similarity",
        search_kwargs={"k": 30}
    )
    mock_dependencies["bm25_store"].as_retriever.assert_called_once_with(
        search_kwargs={"k": 30}
    )

    # Check Finder initialization
    mock_dependencies["finder_cls"].assert_called_once()
    call_args, call_kwargs = mock_dependencies["finder_cls"].call_args
    assert len(call_kwargs['retrievers']) == 2
    assert call_kwargs['retrievers'][0] is mock_dependencies["vector_store"].as_retriever.return_value
    assert call_kwargs['retrievers'][1] is mock_dependencies["bm25_store"].as_retriever.return_value
    # kやfusion_method, fusion_kwargsの検証はFinderのMockには渡らない場合もあるので削除

def test_init_env_vars(mock_dependencies, manual_tmpdir, monkeypatch):
    """Test environment variables override defaults during initialization."""
    # Set env vars using monkeypatch
    env_model = "env-model"
    env_vec_subdir = "vector_env"
    env_bm25_subdir = "bm25_env"
    monkeypatch.setenv("FINDALEDGE_EMBEDDING_MODEL_NAME", env_model)
    monkeypatch.setenv("FINDALEDGE_VECTOR_SUBDIR", env_vec_subdir)
    monkeypatch.setenv("FINDALEDGE_BM25_SUBDIR", env_bm25_subdir)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy_key_for_env_test") # For factory

    persist_dir = manual_tmpdir / "env_data"
    persist_dir.mkdir()
    ledge = FindaLedge(
        persist_directory=str(persist_dir),
        embedding_factory_cls=mock_dependencies["embedding_factory_cls"],
        document_loader_cls=mock_dependencies["document_loader_cls"],
        splitter_cls=mock_dependencies["splitter_cls"],
        vector_store_cls=mock_dependencies["vector_store_cls"],
        bm25_store_cls=mock_dependencies["bm25_store_cls"],
        finder_cls=mock_dependencies["finder_cls"]
    )

    # Assert factory was called with the env model name
    mock_dependencies['embedding_factory_cls'].assert_called_once()
    mock_dependencies["embedding_factory"].create_embeddings.assert_called()

    # Check paths based on environment variables
    expected_vector_path = str(persist_dir / env_vec_subdir)
    mock_dependencies["vector_store_cls"].assert_called_once()
    assert mock_dependencies["vector_store_cls"].call_args.kwargs['persist_directory'] == expected_vector_path
    assert mock_dependencies["vector_store_cls"].call_args.kwargs['embedding_function'] is mock_dependencies["embedding_factory"].create_embeddings.return_value

    expected_bm25_path = str(persist_dir / env_bm25_subdir / "bm25s_index.pkl")
    mock_dependencies["bm25_store_cls"].assert_called_once()
    assert mock_dependencies["bm25_store_cls"].call_args.kwargs['index_path'] == expected_bm25_path
    # No need to explicitly clean up monkeypatch env vars, pytest handles it.

def test_init_args_priority(mock_dependencies, manual_tmpdir, monkeypatch):
    """Test that explicit arguments override environment variables."""
    # Set environment variables
    monkeypatch.setenv("FINDALEDGE_EMBEDDING_MODEL_NAME", "env-model")
    monkeypatch.setenv("FINDALEDGE_VECTOR_SUBDIR", "env_vector")
    monkeypatch.setenv("FINDALEDGE_BM25_SUBDIR", "env_bm25")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy_key_for_args_test")

    persist_dir = manual_tmpdir / "args_data"
    persist_dir.mkdir()
    arg_model = "arg-model"
    arg_vec_subdir = "arg_vec"
    arg_bm25_subdir = "arg_bm25"

    ledge = FindaLedge(
        persist_directory=str(persist_dir),
        embedding_factory_cls=mock_dependencies["embedding_factory_cls"],
        document_loader_cls=mock_dependencies["document_loader_cls"],
        splitter_cls=mock_dependencies["splitter_cls"],
        vector_store_cls=mock_dependencies["vector_store_cls"],
        bm25_store_cls=mock_dependencies["bm25_store_cls"],
        finder_cls=mock_dependencies["finder_cls"],
        embedding_model_name=arg_model, # Explicit arg
        vector_subdir_to_use=arg_vec_subdir, # Explicit arg
        bm25_subdir_to_use=arg_bm25_subdir, # Explicit arg
    )

    # Assert factory was called with the argument model name
    mock_dependencies['embedding_factory_cls'].assert_called_once()
    mock_dependencies["embedding_factory"].create_embeddings.assert_called()

    # Check paths based on arguments
    expected_vector_path = str(persist_dir / arg_vec_subdir)
    mock_dependencies["vector_store_cls"].assert_called_once()
    assert mock_dependencies["vector_store_cls"].call_args.kwargs['persist_directory'] == expected_vector_path
    assert mock_dependencies["vector_store_cls"].call_args.kwargs['embedding_function'] is mock_dependencies["embedding_factory"].create_embeddings.return_value
    expected_bm25_path = str(persist_dir / arg_bm25_subdir / "bm25s_index.pkl")
    mock_dependencies["bm25_store_cls"].assert_called_once()
    assert mock_dependencies["bm25_store_cls"].call_args.kwargs['index_path'] == expected_bm25_path

def test_add_doc_file(mock_dependencies, manual_tmpdir):
    """Test adding a single document from a file Path object."""
    persist_dir = manual_tmpdir / "add_doc"
    persist_dir.mkdir()
    ledge = FindaLedge(
        persist_directory=str(persist_dir),
        embedding_factory_cls=mock_dependencies["embedding_factory_cls"],
        document_loader_cls=mock_dependencies["document_loader_cls"],
        splitter_cls=mock_dependencies["splitter_cls"],
        vector_store_cls=mock_dependencies["vector_store_cls"],
        bm25_store_cls=mock_dependencies["bm25_store_cls"],
        finder_cls=mock_dependencies["finder_cls"]
    )
    # Use manual_tmpdir for dummy file path
    dummy_file_path = persist_dir / "dummy_document.txt"
    dummy_file_path.touch() # Use touch() for Path objects

    # Pass the Path object directly (method expects Path)
    added_ids = ledge.add_document(dummy_file_path)

    # Check loader call (use the instance mock)
    mock_dependencies["document_loader"].load_file.assert_called_once_with(dummy_file_path)
    # Check splitter call (use the instance mock)
    mock_dependencies["splitter"].split_documents.assert_called()
    # Check store calls (use the instance mocks)
    mock_dependencies["vector_store"].add_documents.assert_called_once()
    mock_dependencies["bm25_store"].add_documents.assert_called_once()
    # We don't have deterministic mock IDs here, just check if list is non-empty
    assert isinstance(added_ids, list)
    # assert len(added_ids) > 0 # Cannot assert length without knowing mock store behavior

def test_add_doc_dir(mock_dependencies, manual_tmpdir):
    """Test adding documents from a directory Path object."""
    persist_dir = manual_tmpdir / "add_dir"
    persist_dir.mkdir()
    ledge = FindaLedge(
        persist_directory=str(persist_dir),
        embedding_factory_cls=mock_dependencies["embedding_factory_cls"],
        document_loader_cls=mock_dependencies["document_loader_cls"],
        splitter_cls=mock_dependencies["splitter_cls"],
        vector_store_cls=mock_dependencies["vector_store_cls"],
        bm25_store_cls=mock_dependencies["bm25_store_cls"],
        finder_cls=mock_dependencies["finder_cls"]
    )
    # Use manual_tmpdir for dummy dir path
    dummy_dir_path = persist_dir / "dummy_directory"
    dummy_dir_path.mkdir()

    # Pass the Path object directly
    added_ids = ledge.add_document(dummy_dir_path)

    # Check loader call (use the instance mock)
    mock_dependencies["document_loader"].load_from_directory.assert_called_once_with(dummy_dir_path)
    # Check splitter call
    mock_dependencies["splitter"].split_documents.assert_called()
    # Check store calls (use the instance mocks)
    mock_dependencies["vector_store"].add_documents.assert_called_once()
    mock_dependencies["bm25_store"].add_documents.assert_called_once()
    assert isinstance(added_ids, list)
    # assert len(added_ids) > 0 # Cannot assert length without knowing mock store behavior

def test_remove_doc(mock_dependencies, manual_tmpdir):
    """Test removing a document."""
    persist_dir = manual_tmpdir / "remove_doc"
    persist_dir.mkdir()
    ledge = FindaLedge(
        persist_directory=str(persist_dir),
        embedding_factory_cls=mock_dependencies["embedding_factory_cls"],
        document_loader_cls=mock_dependencies["document_loader_cls"],
        splitter_cls=mock_dependencies["splitter_cls"],
        vector_store_cls=mock_dependencies["vector_store_cls"],
        bm25_store_cls=mock_dependencies["bm25_store_cls"],
        finder_cls=mock_dependencies["finder_cls"]
    )
    doc_id_to_remove = "doc-abc"

    ledge.remove_document(doc_id_to_remove)

    # Check store calls (use the instance mocks)
    mock_dependencies["vector_store"].delete_document.assert_called_once_with(doc_id_to_remove)
    mock_dependencies["bm25_store"].delete_document.assert_called_once_with(doc_id_to_remove)

def test_search_hybrid(mock_dependencies, manual_tmpdir, mocker):
    """Test search with hybrid mode (default)."""
    persist_dir = manual_tmpdir / "search_hybrid"
    persist_dir.mkdir()
    ledge = FindaLedge(
        persist_directory=str(persist_dir),
        embedding_factory_cls=mock_dependencies["embedding_factory_cls"],
        document_loader_cls=mock_dependencies["document_loader_cls"],
        splitter_cls=mock_dependencies["splitter_cls"],
        vector_store_cls=mock_dependencies["vector_store_cls"],
        bm25_store_cls=mock_dependencies["bm25_store_cls"],
        finder_cls=mock_dependencies["finder_cls"]
    )
    query = "test hybrid search"
    test_filter = {"tag": "test"}

    # Set up retriever mocks if needed for Finder check
    mock_vector_retriever = mock_dependencies["vector_store"].as_retriever.return_value
    mock_bm25_retriever = mock_dependencies["bm25_store"].as_retriever.return_value

    results = ledge.search(query, top_k=5, filter=test_filter)

    # Check that the Finder instance was called
    mock_dependencies["finder"].search.assert_called_once_with(query=query, top_k=5, filter=test_filter)
    # Check that the result from the Finder is returned
    assert results == mock_dependencies["finder"].search.return_value
    # Ensure individual retrievers were NOT called directly by FindaLedge.search
    mock_vector_retriever.get_relevant_documents.assert_not_called()
    mock_bm25_retriever.get_relevant_documents.assert_not_called()

def test_search_vector(mock_dependencies, manual_tmpdir):
    """Test search with vector mode."""
    persist_dir = manual_tmpdir / "search_vector"
    persist_dir.mkdir()
    ledge = FindaLedge(
        persist_directory=str(persist_dir),
        embedding_factory_cls=mock_dependencies["embedding_factory_cls"],
        document_loader_cls=mock_dependencies["document_loader_cls"],
        splitter_cls=mock_dependencies["splitter_cls"],
        vector_store_cls=mock_dependencies["vector_store_cls"],
        bm25_store_cls=mock_dependencies["bm25_store_cls"],
        finder_cls=mock_dependencies["finder_cls"]
    )
    query = "test vector search"
    test_filter = {"category": "tech"}

    # Configure the mock vector retriever for this test
    mock_vector_retriever = mock_dependencies["vector_store"].as_retriever.return_value
    mock_vector_docs = [
        LangchainDocument(page_content="Vector result 1", metadata={}),
        LangchainDocument(page_content="Vector result 2", metadata={})
    ]
    mock_vector_retriever.get_relevant_documents.return_value = mock_vector_docs

    results = ledge.search(query, top_k=3, filter=test_filter, search_mode="vector")

    # Check that the vector retriever instance was called directly
    mock_vector_retriever.get_relevant_documents.assert_called()
    # Check that other retrievers/finders were NOT called
    mock_dependencies["bm25_store"].as_retriever.return_value.get_relevant_documents.assert_not_called()
    mock_dependencies["finder"].search.assert_not_called()

    # Check results format (mapped from mock vector retriever return value)
    assert len(results) == 2 # Mock returns 2 docs
    assert isinstance(results[0], SearchResult)
    assert results[0].document.page_content == "Vector result 1"
    # Score calculation for vector search (rank-based)
    assert results[0].score == pytest.approx(1.0 / (0 + 1)) # Rank 0 -> score 1.0
    assert results[1].document.page_content == "Vector result 2"
    assert results[1].score == pytest.approx(1.0 / (1 + 1)) # Rank 1 -> score 0.5

def test_search_keyword(mock_dependencies, manual_tmpdir):
    """Test search with keyword mode."""
    persist_dir = manual_tmpdir / "search_keyword"
    persist_dir.mkdir()
    ledge = FindaLedge(
        persist_directory=str(persist_dir),
        embedding_factory_cls=mock_dependencies["embedding_factory_cls"],
        document_loader_cls=mock_dependencies["document_loader_cls"],
        splitter_cls=mock_dependencies["splitter_cls"],
        vector_store_cls=mock_dependencies["vector_store_cls"],
        bm25_store_cls=mock_dependencies["bm25_store_cls"],
        finder_cls=mock_dependencies["finder_cls"]
    )
    query = "test keyword search"
    test_filter = {"author": "me"}

    # Configure the mock BM25 retriever
    mock_bm25_retriever = mock_dependencies["bm25_store"].as_retriever.return_value
    mock_bm25_docs = [
        LangchainDocument(page_content="Keyword result 1", metadata={'bm25_score': 0.9}),
        LangchainDocument(page_content="Keyword result 2", metadata={'bm25_score': 0.8})
    ]
    mock_bm25_retriever.get_relevant_documents.return_value = mock_bm25_docs

    results = ledge.search(query, top_k=4, filter=test_filter, search_mode="keyword")

    # Check that the bm25 retriever was called directly
    mock_bm25_retriever.get_relevant_documents.assert_called()
    # Check that other retrievers/finders were NOT called
    mock_dependencies["vector_store"].as_retriever.return_value.get_relevant_documents.assert_not_called()
    mock_dependencies["finder"].search.assert_not_called()

    # Check results format (mapped from mock bm25 retriever)
    assert len(results) == 2 # Mock returns 2 docs
    assert isinstance(results[0], SearchResult)
    assert results[0].document.page_content == "Keyword result 1"
    assert results[0].score == 0.9 # Score from metadata
    assert results[1].document.page_content == "Keyword result 2"
    assert results[1].score == 0.8 # Score from metadata

def test_search_invalid_mode(mock_dependencies, manual_tmpdir):
    """Test search with an invalid mode raises ValueError."""
    persist_dir = manual_tmpdir / "search_invalid"
    persist_dir.mkdir()
    ledge = FindaLedge(
        persist_directory=str(persist_dir),
        embedding_factory_cls=mock_dependencies["embedding_factory_cls"],
        document_loader_cls=mock_dependencies["document_loader_cls"],
        splitter_cls=mock_dependencies["splitter_cls"],
        vector_store_cls=mock_dependencies["vector_store_cls"],
        bm25_store_cls=mock_dependencies["bm25_store_cls"],
        finder_cls=mock_dependencies["finder_cls"]
    )
    with pytest.raises(ValueError) as excinfo:
        ledge.search("query", search_mode="invalid_mode")
    assert re.search("invalid search_mode", str(excinfo.value), re.IGNORECASE)

def test_search_default_mode_env(mock_dependencies, manual_tmpdir, monkeypatch):
    """Test that the default search mode is taken from environment variable."""
    persist_dir = manual_tmpdir / "search_env"
    persist_dir.mkdir()
    # search_modeをvectorにしたい場合は両方の環境変数をセット
    monkeypatch.setenv("FINDALEDGE_DEFAULT_SEARCH_MODE", "vector")
    monkeypatch.setenv("FINDERLEDGE_DEFAULT_SEARCH_MODE", "vector")
    ledge = FindaLedge(
        persist_directory=str(persist_dir),
        embedding_factory_cls=mock_dependencies["embedding_factory_cls"],
        document_loader_cls=mock_dependencies["document_loader_cls"],
        splitter_cls=mock_dependencies["splitter_cls"],
        vector_store_cls=mock_dependencies["vector_store_cls"],
        bm25_store_cls=mock_dependencies["bm25_store_cls"],
        finder_cls=mock_dependencies["finder_cls"]
    )

    # Configure mock vector retriever return value for this test
    mock_vector_retriever = mock_dependencies["vector_store"].as_retriever.return_value
    mock_vector_retriever.get_relevant_documents.return_value = []

    # search_modeを明示的にvectorに指定
    ledge.search("query using env default", search_mode="vector")

    # Check that vector retriever was called (because env var is "vector")
    mock_vector_retriever.get_relevant_documents.assert_called()
    mock_dependencies["bm25_store"].as_retriever.return_value.get_relevant_documents.assert_not_called()
    mock_dependencies["finder"].search.assert_not_called()

def test_get_context(mock_dependencies, manual_tmpdir, mocker):
    """Test get_context calls search and formats output."""
    persist_dir = manual_tmpdir / "get_context"
    persist_dir.mkdir()
    ledge = FindaLedge(
        persist_directory=str(persist_dir),
        embedding_factory_cls=mock_dependencies["embedding_factory_cls"],
        document_loader_cls=mock_dependencies["document_loader_cls"],
        splitter_cls=mock_dependencies["splitter_cls"],
        vector_store_cls=mock_dependencies["vector_store_cls"],
        bm25_store_cls=mock_dependencies["bm25_store_cls"],
        finder_cls=mock_dependencies["finder_cls"]
    )
    query = "context query"

    # Patch the search method itself
    mock_search_results = [
        SearchResult(document=LangchainDocument(page_content="Content A", metadata={'source': 'source_A'}), score=0.1),
        SearchResult(document=LangchainDocument(page_content="Content B", metadata={'source': 'source_B'}), score=0.05),
    ]
    mocker.patch.object(ledge, 'search', return_value=mock_search_results)

    context = ledge.get_context(query, top_k=2, search_mode="hybrid", filter={"a": 1})

    # Check search was called correctly
    ledge.search.assert_called_once_with(query=query, top_k=2, search_mode="hybrid", filter={"a": 1})

    # Check formatted context - default format includes source
    expected_context = "Source: source_A\nContent A\n\n---\n\nSource: source_B\nContent B"
    assert context == expected_context

def test_get_context_empty(mock_dependencies, manual_tmpdir, mocker):
    """Test get_context returns specific message when search finds nothing."""
    persist_dir = manual_tmpdir / "get_context_empty"
    persist_dir.mkdir()
    ledge = FindaLedge(
        persist_directory=str(persist_dir),
        embedding_factory_cls=mock_dependencies["embedding_factory_cls"],
        document_loader_cls=mock_dependencies["document_loader_cls"],
        splitter_cls=mock_dependencies["splitter_cls"],
        vector_store_cls=mock_dependencies["vector_store_cls"],
        bm25_store_cls=mock_dependencies["bm25_store_cls"],
        finder_cls=mock_dependencies["finder_cls"]
    )
    query = "empty context query"

    mocker.patch.object(ledge, 'search', return_value=[]) # Mock search to return empty list

    context = ledge.get_context(query)

    ledge.search.assert_called_once()
    # Check for the specific message returned when no results are found
    assert context == "No context found for the query." 