"""
Tests for the TextSplitter class
TextSplitterクラスのテスト
"""

import pytest
from finderledge import TextSplitter

def test_text_splitter_default():
    """
    Test text splitter with default settings
    デフォルト設定でのテキスト分割をテスト
    """
    splitter = TextSplitter()
    text = "This is a test document.\nIt has multiple lines.\nAnd some content."
    
    chunks = splitter.split_text(text)
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert all(len(chunk) <= 1000 for chunk in chunks)

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

def test_text_splitter_custom_overlap():
    """
    Test text splitter with custom overlap
    カスタムオーバーラップでのテキスト分割をテスト
    """
    splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
    text = "This is a test document.\nIt has multiple lines.\nAnd some content."
    
    chunks = splitter.split_text(text)
    
    assert len(chunks) > 0
    # Check overlap between consecutive chunks
    for i in range(len(chunks) - 1):
        overlap = set(chunks[i].split()) & set(chunks[i + 1].split())
        assert len(overlap) >= 2  # At least 2 words should overlap

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

def test_text_splitter_empty_text():
    """
    Test text splitter with empty text
    空のテキストでのテキスト分割をテスト
    """
    splitter = TextSplitter()
    chunks = splitter.split_text("")
    
    assert len(chunks) == 0

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