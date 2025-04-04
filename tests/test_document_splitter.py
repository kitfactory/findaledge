"""
Test cases for DocumentSplitter class
DocumentSplitterクラスのテストケース
"""

import pytest
from pathlib import Path
from langchain.schema import Document
from finderledge import DocumentSplitter, DocumentType

@pytest.fixture
def splitter():
    """
    Create DocumentSplitter instance for testing
    テスト用のDocumentSplitterインスタンスを作成
    """
    return DocumentSplitter(chunk_size=100, chunk_overlap=20)

def test_init_splitters():
    """
    Test splitter initialization with custom parameters
    カスタムパラメータでのスプリッター初期化をテスト
    """
    splitter = DocumentSplitter(chunk_size=500, chunk_overlap=50)
    assert splitter.chunk_size == 500
    assert splitter.chunk_overlap == 50
    assert len(splitter.splitters) == len(DocumentType)

def test_get_document_type_from_metadata(splitter):
    """
    Test document type detection from metadata
    メタデータからのドキュメントタイプ検出をテスト
    """
    doc = Document(
        page_content="Test content",
        metadata={"type": "python"}
    )
    assert splitter._get_document_type(doc) == DocumentType.PYTHON

def test_get_document_type_from_extension(splitter):
    """
    Test document type detection from file extension
    ファイル拡張子からのドキュメントタイプ検出をテスト
    """
    doc = Document(
        page_content="Test content",
        metadata={"source": "test.md"}
    )
    assert splitter._get_document_type(doc) == DocumentType.MARKDOWN

def test_get_document_type_default(splitter):
    """
    Test default document type when no type info is available
    タイプ情報がない場合のデフォルトドキュメントタイプをテスト
    """
    doc = Document(page_content="Test content")
    assert splitter._get_document_type(doc) == DocumentType.TEXT

def test_split_document_markdown():
    """
    Test splitting markdown document
    Markdownドキュメントの分割をテスト
    """
    splitter = DocumentSplitter(chunk_size=20, chunk_overlap=5)
    doc = Document(
        page_content="# Title\n\nFirst paragraph.\n\nSecond paragraph.",
        metadata={"source": "test.md"}
    )
    split_docs = splitter.split_document(doc)
    
    assert len(split_docs) > 1
    assert all(isinstance(d, Document) for d in split_docs)
    assert all(d.metadata["doc_type_detected"] == "MARKDOWN" for d in split_docs)
    assert all(d.metadata["source"] == "test.md" for d in split_docs)

def test_split_document_python():
    """
    Test splitting Python code document
    Pythonコードドキュメントの分割をテスト
    """
    splitter = DocumentSplitter(chunk_size=20, chunk_overlap=5)
    doc = Document(
        page_content="def test():\n    print('hello')\n\ndef another():\n    print('world')",
        metadata={"type": "python"}
    )
    split_docs = splitter.split_document(doc)
    
    assert len(split_docs) > 1
    assert all(isinstance(d, Document) for d in split_docs)
    assert all(d.metadata["type"] == "python" for d in split_docs)
    assert all(d.metadata["doc_type_detected"] == "PYTHON" for d in split_docs)

def test_split_document_html():
    """
    Test splitting HTML document
    HTMLドキュメントの分割をテスト
    """
    splitter = DocumentSplitter(chunk_size=20, chunk_overlap=5)
    doc = Document(
        page_content="""
        <!DOCTYPE html>
        <html>
        <body>
            <h1>Title</h1>
            <p>First paragraph.</p>
            <h2>Subtitle</h2>
            <p>Second paragraph.</p>
        </body>
        </html>
        """,
        metadata={"source": "test.html"}
    )
    split_docs = splitter.split_document(doc)
    
    assert len(split_docs) > 1
    assert all(isinstance(d, Document) for d in split_docs)
    assert all(d.metadata["doc_type_detected"] == "HTML" for d in split_docs)
    assert all(d.metadata["source"] == "test.html" for d in split_docs)

def test_split_document_json():
    """
    Test splitting JSON document
    JSONドキュメントの分割をテスト
    """
    splitter = DocumentSplitter(chunk_size=20, chunk_overlap=5)
    doc = Document(
        page_content="""
        {
            "title": "Test",
            "sections": [
                {"name": "Section 1", "content": "First content"},
                {"name": "Section 2", "content": "Second content"}
            ]
        }
        """,
        metadata={"source": "test.json"}
    )
    split_docs = splitter.split_document(doc)
    
    assert len(split_docs) > 1
    assert all(isinstance(d, Document) for d in split_docs)
    assert all(d.metadata["doc_type_detected"] == "JSON" for d in split_docs)
    assert all(d.metadata["source"] == "test.json" for d in split_docs)

def test_split_documents_multiple_types():
    """
    Test splitting multiple documents of different types
    異なるタイプの複数ドキュメントの分割をテスト
    """
    splitter = DocumentSplitter(chunk_size=20, chunk_overlap=5)
    docs = [
        Document(
            page_content="# Markdown\n\nContent",
            metadata={"source": "test.md"}
        ),
        Document(
            page_content="def test(): pass",
            metadata={"type": "python"}
        ),
        Document(
            page_content="<!DOCTYPE html><h1>Title</h1><p>Content</p>",
            metadata={"source": "test.html"}
        ),
        Document(
            page_content='{"key": "value", "array": [1, 2, 3]}',
            metadata={"source": "test.json"}
        )
    ]
    split_docs = splitter.split_documents(docs)
    
    assert len(split_docs) > len(docs)
    assert any(d.metadata.get("doc_type_detected") == "MARKDOWN" and d.metadata.get("source") == "test.md" for d in split_docs)
    assert any(d.metadata.get("doc_type_detected") == "PYTHON" and d.metadata.get("type") == "python" for d in split_docs)
    assert any(d.metadata.get("doc_type_detected") == "HTML" and d.metadata.get("source") == "test.html" for d in split_docs)
    assert any(d.metadata.get("doc_type_detected") == "JSON" and d.metadata.get("source") == "test.json" for d in split_docs)

def test_metadata_preservation():
    """
    Test metadata preservation after splitting
    分割後のメタデータ保持をテスト
    """
    splitter = DocumentSplitter(chunk_size=20, chunk_overlap=5)
    original_metadata = {
        "id": "test123",
        "source": "test.md",
        "custom": "value"
    }
    doc = Document(
        page_content="# Title\n\nContent\n\nMore content",
        metadata=original_metadata
    )
    split_docs = splitter.split_document(doc)
    
    for split_doc in split_docs:
        assert split_doc.metadata["source"] == original_metadata["source"]
        assert split_doc.metadata["custom"] == original_metadata["custom"]
        assert "parent_id" in split_doc.metadata
        assert split_doc.metadata["parent_id"] == original_metadata["id"]
        assert "split_index" in split_doc.metadata
        assert "is_split" in split_doc.metadata
        assert split_doc.metadata["is_split"] is True
        assert "id" in split_doc.metadata
        assert split_doc.metadata["id"] != original_metadata["id"]

def test_content_type_detection(splitter):
    """
    Test document type detection from content
    コンテンツからのドキュメントタイプ検出をテスト
    """
    # HTML detection
    html_doc = Document(
        page_content="<!DOCTYPE html><html><body>Test</body></html>"
    )
    assert splitter._get_document_type(html_doc) == DocumentType.HTML

    # JSON detection
    json_doc = Document(
        page_content='{"key": "value"}'
    )
    assert splitter._get_document_type(json_doc) == DocumentType.JSON

    # Invalid JSON should default to TEXT
    invalid_json_doc = Document(
        page_content='{key: value}'
    )
    assert splitter._get_document_type(invalid_json_doc) == DocumentType.TEXT 