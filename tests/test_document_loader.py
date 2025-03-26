"""
Tests for document loader
文書ローダーのテスト
"""

import pytest
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from finderledge import DocumentLoader
from finderledge.document import Document

@pytest.fixture
def temp_dir(tmp_path):
    """
    Create a temporary directory for test files
    テストファイル用の一時ディレクトリを作成
    """
    return tmp_path

@pytest.fixture
def sample_text_file(temp_dir):
    """
    Create a sample text file for testing
    テスト用のサンプルテキストファイルを作成
    """
    file_path = temp_dir / "test.txt"
    file_path.write_text("This is a test document.\nIt has multiple lines.")
    return file_path

@pytest.fixture
def sample_json_file(temp_dir):
    """
    Create a sample JSON file for testing
    テスト用のサンプルJSONファイルを作成
    """
    file_path = temp_dir / "test.json"
    data = {"key": "value", "text": "Test JSON content"}
    file_path.write_text(json.dumps(data))
    return file_path

@pytest.fixture
def sample_markdown_file(temp_dir):
    """
    Create a sample Markdown file for testing
    テスト用のサンプルMarkdownファイルを作成
    """
    file_path = temp_dir / "test.md"
    file_path.write_text("# Test Document\n\nThis is a test markdown document.")
    return file_path

def test_document_loader_initialization():
    """
    Test document loader initialization
    文書ローダーの初期化テスト
    """
    loader = DocumentLoader()
    assert loader.text_splitter is not None
    assert loader.text_splitter.chunk_size == 1000
    assert loader.text_splitter.chunk_overlap == 200

def test_load_text_file():
    """
    Test loading text file
    テキストファイルの読み込みテスト
    """
    loader = DocumentLoader()
    test_file = Path("tests/data/test.txt")
    test_file.parent.mkdir(exist_ok=True)
    test_file.write_text("This is a test document.\nIt has multiple lines.")

    documents = loader.load_file(test_file)
    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert documents[0].content == "This is a test document.\nIt has multiple lines."
    assert documents[0].title == "test.txt"

def test_load_pdf_file():
    """
    Test loading PDF file
    PDFファイルの読み込みテスト
    """
    loader = DocumentLoader()
    test_file = Path("tests/data/test.pdf")
    test_file.parent.mkdir(exist_ok=True)
    # PDFファイルの作成は複雑なので、このテストはスキップ
    pytest.skip("PDF file creation is complex, skipping this test")

def test_load_docx_file():
    """
    Test loading DOCX file
    DOCXファイルの読み込みテスト
    """
    loader = DocumentLoader()
    test_file = Path("tests/data/test.docx")
    test_file.parent.mkdir(exist_ok=True)
    # DOCXファイルの作成は複雑なので、このテストはスキップ
    pytest.skip("DOCX file creation is complex, skipping this test")

def test_load_nonexistent_file():
    """
    Test loading nonexistent file
    存在しないファイルの読み込みテスト
    """
    loader = DocumentLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_file(Path("nonexistent.txt"))

def test_load_directory():
    """
    Test loading directory
    ディレクトリの読み込みテスト
    """
    loader = DocumentLoader()
    test_dir = Path("tests/data")
    test_dir.mkdir(exist_ok=True)
    
    # テストファイルを作成
    (test_dir / "test1.txt").write_text("First test document")
    (test_dir / "test2.txt").write_text("Second test document")
    
    documents = loader.load_directory(test_dir)
    assert len(documents) == 2
    assert all(isinstance(doc, Document) for doc in documents)
    assert {doc.title for doc in documents} == {"test1.txt", "test2.txt"}

def test_load_directory_with_filter():
    """
    Test loading directory with file filter
    ファイルフィルター付きディレクトリ読み込みテスト
    """
    loader = DocumentLoader()
    test_dir = Path("tests/data")
    test_dir.mkdir(exist_ok=True)
    
    # テストファイルを作成
    (test_dir / "test1.txt").write_text("First test document")
    (test_dir / "test2.pdf").write_text("Second test document")
    
    documents = loader.load_directory(test_dir, file_filter=lambda x: x.suffix == ".txt")
    assert len(documents) == 1
    assert documents[0].title == "test1.txt"

def test_load_directory_nonexistent():
    """
    Test loading nonexistent directory
    存在しないディレクトリの読み込みテスト
    """
    loader = DocumentLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_directory(Path("nonexistent_dir"))

def test_load_text_with_metadata():
    """
    Test loading text with metadata
    メタデータ付きテキストの読み込みテスト
    """
    loader = DocumentLoader()
    text = "This is a test document."
    metadata = {"source": "test", "author": "test_user"}
    
    documents = loader.load_text(text, metadata=metadata)
    assert len(documents) == 1
    assert documents[0].content == text
    assert documents[0].metadata == metadata

def test_load_text_with_title():
    """
    Test loading text with title
    タイトル付きテキストの読み込みテスト
    """
    loader = DocumentLoader()
    text = "This is a test document."
    title = "Test Document"
    
    documents = loader.load_text(text, title=title)
    assert len(documents) == 1
    assert documents[0].content == text
    assert documents[0].title == title

def test_load_text_file(sample_text_file):
    """
    Test loading text files
    テキストファイルの読み込みをテスト
    """
    loader = DocumentLoader()
    content = loader.load_document(sample_text_file)
    
    assert isinstance(content, str)
    assert "This is a test document" in content
    assert "It has multiple lines" in content

def test_load_json_file(sample_json_file):
    """
    Test loading JSON files
    JSONファイルの読み込みをテスト
    """
    loader = DocumentLoader()
    content = loader.load_json(sample_json_file)
    
    assert isinstance(content, dict)
    assert content["key"] == "value"
    assert content["text"] == "Test JSON content"

def test_load_markdown_file(sample_markdown_file):
    """
    Test loading Markdown files
    Markdownファイルの読み込みをテスト
    """
    loader = DocumentLoader()
    content = loader.load_markdown(sample_markdown_file)
    
    assert isinstance(content, str)
    assert "Test Document" in content
    assert "This is a test markdown document" in content

def test_load_multiple_documents(temp_dir):
    """
    Test loading multiple documents
    複数文書の読み込みをテスト
    """
    # Create multiple test files
    files = []
    for i in range(3):
        file_path = temp_dir / f"test{i}.txt"
        file_path.write_text(f"Test document {i}")
        files.append(file_path)
    
    loader = DocumentLoader()
    contents = loader.load_documents(files)
    
    assert len(contents) == 3
    assert all(isinstance(content, str) for content in contents)
    assert all(f"Test document {i}" in content for i, content in enumerate(contents))

def test_load_invalid_json(temp_dir):
    """
    Test loading invalid JSON file
    無効なJSONファイルの読み込みをテスト
    """
    file_path = temp_dir / "invalid.json"
    file_path.write_text("Invalid JSON content")
    
    loader = DocumentLoader()
    with pytest.raises(json.JSONDecodeError):
        loader.load_json(file_path)

def test_load_unsupported_file_type(temp_dir):
    """
    Test loading unsupported file type
    サポートされていないファイルタイプの読み込みをテスト
    """
    file_path = temp_dir / "test.xyz"
    file_path.write_text("Test content")
    
    loader = DocumentLoader()
    with pytest.raises(ValueError):
        loader.load_document(file_path) 