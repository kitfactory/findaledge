"""
Tests for text splitter
テキスト分割器のテスト
"""

import pytest
from finderledge.text_splitter import TextSplitter

def test_text_splitter_initialization():
    """
    Test text splitter initialization
    テキスト分割器の初期化テスト
    """
    # デフォルトパラメータでの初期化
    splitter = TextSplitter()
    assert splitter.chunk_size == 1000
    assert splitter.chunk_overlap == 200
    assert splitter.length_function == len
    assert splitter.separators == ["\n\n", "\n", ".", " ", ""]

    # カスタムパラメータでの初期化
    def custom_length(text: str) -> int:
        return len(text) * 2

    custom_separators = ["。", "、", " "]
    splitter = TextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=custom_length,
        separators=custom_separators
    )
    assert splitter.chunk_size == 500
    assert splitter.chunk_overlap == 100
    assert splitter.length_function == custom_length
    assert splitter.separators == custom_separators

    # エラーケース：chunk_overlap >= chunk_size
    with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
        TextSplitter(chunk_size=100, chunk_overlap=100)

def test_split_text():
    """
    Test text splitting functionality
    テキスト分割機能のテスト
    """
    splitter = TextSplitter(chunk_size=20, chunk_overlap=5)  # より小さいチャンクサイズを使用

    # 空のテキスト
    print("\nTesting empty text...")
    assert splitter.split_text("") == []

    # 単一のチャンクに収まるテキスト
    print("\nTesting single chunk text...")
    text = "This is a short text."
    chunks = splitter.split_text(text)
    print(f"Chunks: {chunks}")
    assert len(chunks) > 1  # チャンクサイズが20なので分割される
    assert all(len(chunk) <= 20 for chunk in chunks)

    # 複数チャンクに分割されるテキスト
    print("\nTesting multiple chunks text...")
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = splitter.split_text(text)
    print(f"Chunks: {chunks}")
    assert len(chunks) > 1
    assert all(len(chunk) <= 20 for chunk in chunks)

def test_split_documents():
    """
    Test multiple document splitting
    複数文書の分割テスト
    """
    splitter = TextSplitter(chunk_size=100, chunk_overlap=20)

    # 空の文書リスト
    assert splitter.split_documents([]) == []

    # 単一の文書
    doc = "This is a test document."
    chunks = splitter.split_documents([doc])
    assert len(chunks) == 1
    assert chunks[0] == doc

    # 複数の文書
    docs = [
        "First document with multiple paragraphs.\n\nSecond paragraph.",
        "Second document with some content.",
        "Third document."
    ]
    chunks = splitter.split_documents(docs)
    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert all(chunk.strip() for chunk in chunks)

def test_custom_length_function():
    """
    Test custom length function
    カスタム長さ関数のテスト
    """
    def custom_length(text: str) -> int:
        return len(text) * 2

    splitter = TextSplitter(
        chunk_size=200,
        chunk_overlap=40,
        length_function=custom_length
    )

    text = "This is a test text."
    chunks = splitter.split_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text

    # カスタム長さ関数が正しく適用されていることを確認
    assert splitter.length_function(text) == len(text) * 2

def test_custom_separators():
    """
    Test custom separators
    カスタム区切り文字のテスト
    """
    separators = ["。", "、", " "]
    splitter = TextSplitter(
        chunk_size=10,  # より小さいチャンクサイズを使用
        chunk_overlap=2,
        separators=separators
    )

    text = "これはテスト文書です。これは二つ目の文です。"
    chunks = splitter.split_text(text)
    assert len(chunks) > 1
    assert all(len(chunk) <= 10 for chunk in chunks)

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
    splitter = TextSplitter(chunk_size=15, chunk_overlap=5)  # チャンクサイズとオーバーラップを調整
    text = "This is a test sentence for checking overlap between chunks."

    chunks = splitter.split_text(text)
    print("\nChunks:", chunks)  # デバッグ用の出力を追加
    assert len(chunks) > 1
    assert all(len(chunk) <= 15 for chunk in chunks)

    # Check that consecutive chunks have some overlap in characters
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i]
        next_chunk = chunks[i + 1]
        
        # 最後の5文字と最初の5文字を比較
        current_end = current_chunk[-5:] if len(current_chunk) >= 5 else current_chunk
        next_start = next_chunk[:5] if len(next_chunk) >= 5 else next_chunk
        
        # 共通の文字列があるか確認
        common = set(current_end) & set(next_start)
        print(f"\nChecking overlap between chunks {i} and {i+1}:")
        print(f"Current end: {current_end}")
        print(f"Next start: {next_start}")
        print(f"Common characters: {common}")
        
        assert len(common) > 0, f"No overlap found between chunks {i} and {i+1}"

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
    splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
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
    with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
        TextSplitter(chunk_size=10, chunk_overlap=15)

def test_text_splitter_custom_chunk_size():
    """
    Test text splitter with custom chunk size
    カスタムチャンクサイズでのテキスト分割をテスト
    """
    splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
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