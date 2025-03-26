"""
Tests for the DocumentLoader class
DocumentLoaderクラスのテスト
"""

import pytest
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from finderledge import DocumentLoader

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

def test_load_nonexistent_file():
    """
    Test loading nonexistent file
    存在しないファイルの読み込みをテスト
    """
    loader = DocumentLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_document("nonexistent.txt")

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