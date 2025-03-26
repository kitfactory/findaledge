"""
Tests for OpenAI API integration
OpenAI APIとの連携のテスト

This module contains tests for the integration between FinderLedge and OpenAI API.
このモジュールには、FinderLedgeとOpenAI APIの統合テストが含まれています。
"""

import os
import shutil
import pytest
from unittest.mock import Mock, patch, MagicMock
from openai import OpenAI
from finderledge import Finder, Tokenizer, OpenAIEmbeddingModel, DocumentStore, EmbeddingStore, BM25, Document
from examples.agents_integration import FinderTool, get_tool_functions, create_finder_assistant
import numpy as np

@pytest.fixture(autouse=True)
def setup_env():
    """
    Set up environment variables for testing
    テスト用の環境変数を設定する
    """
    # バックアップを取る
    old_api_key = os.environ.get("OPENAI_API_KEY")
    
    # テスト用のAPIキーを設定
    os.environ["OPENAI_API_KEY"] = "test-api-key"
    
    yield
    
    # 環境変数を元に戻す
    if old_api_key:
        os.environ["OPENAI_API_KEY"] = old_api_key
    else:
        del os.environ["OPENAI_API_KEY"]

@pytest.fixture
def mock_openai_client():
    """
    Create a mock OpenAI client
    モックのOpenAIクライアントを作成する
    """
    mock_client = Mock(spec=OpenAI)
    mock_client.beta = MagicMock()
    mock_client.beta.assistants = MagicMock()
    return mock_client

@pytest.fixture
def mock_embedding_model():
    """
    Create a mock embedding model
    モックの埋め込みモデルを作成する
    """
    mock_model = Mock(spec=OpenAIEmbeddingModel)
    mock_model.embed_documents.return_value = [np.array([0.1] * 1536)]  # OpenAIの埋め込みベクトルの次元
    mock_model.embed_query.return_value = np.array([0.1] * 1536)
    mock_model.embed_text.return_value = np.array([0.1] * 1536)
    return mock_model

@pytest.fixture
def finder(mock_embedding_model):
    """
    Create a Finder instance for testing
    テスト用のFinderインスタンスを作成する
    """
    # Clean up test directory if it exists
    test_dir = "test_finder"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    # Initialize components
    tokenizer = Tokenizer()
    document_store = DocumentStore(test_dir)
    embedding_store = EmbeddingStore(test_dir)
    bm25 = BM25(test_dir)

    # Create Finder instance
    finder = Finder(
        tokenizer=tokenizer,
        embedding_model=mock_embedding_model,
        document_store=document_store,
        embedding_store=embedding_store,
        bm25=bm25,
        storage_dir=test_dir
    )

    # Add test documents
    doc1 = Document(id="1", content="This is a test document about AI.")
    doc2 = Document(id="2", content="Another document about machine learning.")
    finder.add_document(doc1)
    finder.add_document(doc2)

    return finder

@pytest.fixture(autouse=True)
def cleanup():
    """
    Clean up test directory after each test
    各テスト後にテストディレクトリをクリーンアップする
    """
    yield
    test_dir = "test_finder"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

@pytest.fixture
def finder_tool(finder):
    """
    Create a FinderTool instance for testing
    テスト用のFinderToolインスタンスを作成する
    """
    return FinderTool(finder)

def test_search_documents(finder_tool):
    """
    Test document search functionality
    文書検索機能をテストする
    """
    results = finder_tool.search_documents("AI", mode="hybrid", top_k=2)
    assert isinstance(results, list)
    assert len(results) > 0
    assert all(isinstance(result, dict) for result in results)
    assert all(key in result for result in results for key in ["id", "content", "score"])

def test_add_document(finder_tool):
    """
    Test document addition functionality
    文書追加機能をテストする
    """
    doc_id = finder_tool.add_document("New test document")
    assert isinstance(doc_id, str)
    assert len(doc_id) > 0

def test_remove_document(finder_tool):
    """
    Test document removal functionality
    文書削除機能をテストする
    """
    doc_id = finder_tool.add_document("Document to remove")
    result = finder_tool.remove_document(doc_id)
    assert result is True

def test_error_handling(finder_tool):
    """
    Test error handling in FinderTool methods
    FinderToolメソッドのエラーハンドリングをテストする
    """
    result = finder_tool.search_documents("test", mode="invalid_mode")
    assert isinstance(result, dict)
    assert "error" in result
    assert "Invalid search mode" in result["error"]

def test_get_tool_functions():
    """
    Test tool function definitions
    ツール関数の定義をテストする
    """
    functions = get_tool_functions()
    assert len(functions) == 3
    
    function_names = {func["name"] for func in functions}
    assert function_names == {"search_documents", "add_document", "remove_document"}
    
    for func in functions:
        assert "description" in func
        assert "parameters" in func
        assert "type" in func["parameters"]
        assert "properties" in func["parameters"]
        assert "required" in func["parameters"]

def test_create_finder_assistant(finder, mock_openai_client):
    """
    Test assistant creation with FinderLedge tools
    FinderLedgeツールを持つアシスタントの作成をテストする
    """
    # Mock the assistant creation
    # アシスタントの作成をモック化
    mock_assistant = {
        "id": "test_assistant_id",
        "name": "finder_assistant",
        "description": "An assistant that can search, add, and remove documents using FinderLedge",
        "model": "gpt-4-turbo-preview",
        "tools": [{"type": "function", "function": func} for func in get_tool_functions()]
    }
    mock_openai_client.beta.assistants.create.return_value = mock_assistant

    # Create assistant
    # アシスタントを作成
    assistant = create_finder_assistant(mock_openai_client, finder)
    
    # Verify the assistant configuration
    # アシスタントの設定を検証
    assert assistant["name"] == "finder_assistant"
    assert assistant["model"] == "gpt-4-turbo-preview"
    assert len(assistant["tools"]) == 3 