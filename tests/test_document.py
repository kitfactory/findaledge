"""
Tests for the Document class
Documentクラスのテスト
"""

import pytest
from datetime import datetime
from finderledge.document import Document

def test_document_creation():
    """
    Test document creation
    文書作成のテスト
    """
    doc = Document(
        id="test1",
        title="Test Document",
        content="This is a test document.",
        metadata={"author": "Test Author"}
    )

    assert doc.id == "test1"
    assert doc.title == "Test Document"
    assert doc.content == "This is a test document."
    assert doc.metadata == {"author": "Test Author"}
    assert isinstance(doc.created_at, datetime)
    assert isinstance(doc.updated_at, datetime)
    assert doc.chunks == {}

def test_document_add_chunks():
    """
    Test adding chunks to document
    文書へのチャンク追加のテスト
    """
    doc = Document(
        id="test2",
        title="Test Document",
        content="This is a test document."
    )

    chunks = ["This is", "a test", "document."]
    doc.add_chunks(chunks)

    assert len(doc.chunks) == 3
    assert all(isinstance(chunk_id, str) for chunk_id in doc.chunks.keys())
    assert all(chunk in chunks for chunk in doc.chunks.values())

def test_document_add_chunk_embeddings():
    """
    Test adding chunk embeddings to document
    文書へのチャンク埋め込み追加のテスト
    """
    doc = Document(
        id="test3",
        title="Test Document",
        content="This is a test document."
    )

    chunks = ["This is", "a test", "document."]
    doc.add_chunks(chunks)

    embeddings = {
        list(doc.chunks.keys())[0]: [0.1, 0.2, 0.3],
        list(doc.chunks.keys())[1]: [0.4, 0.5, 0.6],
        list(doc.chunks.keys())[2]: [0.7, 0.8, 0.9]
    }
    doc.add_chunk_embeddings(embeddings)

    assert len(doc.chunk_embeddings) == 3
    assert all(isinstance(embedding, list) for embedding in doc.chunk_embeddings.values())
    assert all(len(embedding) == 3 for embedding in doc.chunk_embeddings.values())

def test_document_clear_chunks():
    """
    Test clearing chunks from document
    文書からのチャンク削除のテスト
    """
    doc = Document(
        id="test4",
        title="Test Document",
        content="This is a test document."
    )

    chunks = ["This is", "a test", "document."]
    doc.add_chunks(chunks)

    embeddings = {
        list(doc.chunks.keys())[0]: [0.1, 0.2, 0.3],
        list(doc.chunks.keys())[1]: [0.4, 0.5, 0.6],
        list(doc.chunks.keys())[2]: [0.7, 0.8, 0.9]
    }
    doc.add_chunk_embeddings(embeddings)

    doc.clear_chunks()

    assert doc.chunks == {}
    assert doc.chunk_embeddings == {}

def test_document_serialization():
    """
    Test document serialization and deserialization
    文書のシリアライズとデシリアライズのテスト
    """
    doc = Document(
        id="test5",
        title="Test Document",
        content="This is a test document.",
        metadata={"author": "Test Author"}
    )

    chunks = ["This is", "a test", "document."]
    doc.add_chunks(chunks)

    embeddings = {
        list(doc.chunks.keys())[0]: [0.1, 0.2, 0.3],
        list(doc.chunks.keys())[1]: [0.4, 0.5, 0.6],
        list(doc.chunks.keys())[2]: [0.7, 0.8, 0.9]
    }
    doc.add_chunk_embeddings(embeddings)

    # Test serialization
    doc_dict = doc.to_dict()
    assert isinstance(doc_dict, dict)
    assert doc_dict["id"] == "test5"
    assert doc_dict["title"] == "Test Document"
    assert doc_dict["content"] == "This is a test document."
    assert doc_dict["metadata"] == {"author": "Test Author"}
    assert "created_at" in doc_dict
    assert "updated_at" in doc_dict
    assert "chunks" in doc_dict
    assert "chunk_embeddings" in doc_dict

    # Test deserialization
    new_doc = Document.from_dict(doc_dict)
    assert new_doc.id == doc.id
    assert new_doc.title == doc.title
    assert new_doc.content == doc.content
    assert new_doc.metadata == doc.metadata
    assert new_doc.chunks == doc.chunks
    assert new_doc.chunk_embeddings == doc.chunk_embeddings 