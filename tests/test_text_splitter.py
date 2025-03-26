"""
Tests for the TextSplitter class
TextSplitterクラスのテスト
"""

import pytest
from finderledge import TextSplitter

def test_text_splitter_basic():
    """
    Test basic text splitting functionality
    基本的なテキスト分割機能をテスト
    """
    splitter = TextSplitter(chunk_size=10, chunk_overlap=3)
    text = "This is a test document for splitting."

    chunks = splitter.split_text(text)
    assert len(chunks) > 0
    assert all(len(chunk) <= 10 for chunk in chunks)

def test_text_splitter_custom_overlap():
    """
    Test text splitter with custom overlap
    カスタムオーバーラップでのテキスト分割をテスト
    """
    splitter = TextSplitter(chunk_size=20, chunk_overlap=5)
    text = "This is a test document. It has multiple sentences. Testing overlap."

    chunks = splitter.split_text(text)
    assert len(chunks) > 1

    # Check that consecutive chunks have some overlap
    for i in range(len(chunks) - 1):
        # Get words from the end of the current chunk and start of the next chunk
        current_words = set(chunks[i].split()[-3:])  # Last 3 words
        next_words = set(chunks[i + 1].split()[:3])  # First 3 words
        # Check if there is any overlap in words
        overlap = current_words & next_words
        assert len(overlap) > 0, f"No overlap found between chunks {i} and {i+1}"

def test_text_splitter_empty_text():
    """
    Test splitting empty text
    空のテキストの分割をテスト
    """
    splitter = TextSplitter()
    chunks = splitter.split_text("")
    assert len(chunks) == 0

def test_text_splitter_single_chunk():
    """
    Test text that fits in a single chunk
    1つのチャンクに収まるテキストをテスト
    """
    splitter = TextSplitter(chunk_size=100)
    text = "Short text"
    chunks = splitter.split_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text

def test_text_splitter_multiple_documents():
    """
    Test splitting multiple documents
    複数の文書の分割をテスト
    """
    splitter = TextSplitter(chunk_size=10, chunk_overlap=3)
    documents = ["First document", "Second document", "Third document"]
    chunks = splitter.split_documents(documents)
    assert len(chunks) >= len(documents)
    assert all(len(chunk) <= 10 for chunk in chunks)

def test_text_splitter_invalid_params():
    """
    Test text splitter with invalid parameters
    不正なパラメータでのテキスト分割をテスト
    """
    # Test chunk_overlap >= chunk_size
    splitter = TextSplitter(chunk_size=10, chunk_overlap=15)
    text = "This is a test document."
    chunks = splitter.split_text(text)
    assert len(chunks) > 0  # Should still work with adjusted overlap

def test_text_splitter_custom_chunk_size():
    """
    Test text splitter with custom chunk size
    カスタムチャンクサイズでのテキスト分割をテスト
    """
    splitter = TextSplitter(chunk_size=100)
    text = "This is a test document.\nIt has multiple lines.\nAnd some content."
    
    chunks = splitter.split_text(text)
    
    assert len(chunks) > 0
    assert all(len(chunk) <= 100 for chunk in chunks)

def test_text_splitter_custom_separators():
    """
    Test text splitter with custom separators
    カスタムセパレータでのテキスト分割をテスト
    """
    splitter = TextSplitter(separators=["\n", ".", " "])
    text = "This is a test document.\nIt has multiple lines.\nAnd some content."
    
    chunks = splitter.split_text(text)
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)

def test_text_splitter_documents():
    """
    Test text splitter with document list
    文書リストでのテキスト分割をテスト
    """
    splitter = TextSplitter()
    documents = [
        "First document content.",
        "Second document content.",
        "Third document content."
    ]
    
    chunks = splitter.split_documents(documents)
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert all(len(chunk) <= 1000 for chunk in chunks) 