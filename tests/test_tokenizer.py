"""
Tests for the Tokenizer class
Tokenizerクラスのテスト
"""

import pytest
from finderledge.tokenizer import Tokenizer

def test_tokenizer_initialization():
    """
    Test tokenizer initialization
    トークナイザーの初期化テスト
    """
    tokenizer = Tokenizer()
    assert tokenizer.min_length == 2
    assert tokenizer.max_length == 100
    assert tokenizer.stop_words == set()

def test_tokenizer_with_custom_params():
    """
    Test tokenizer with custom parameters
    カスタムパラメータを使用したトークナイザーのテスト
    """
    tokenizer = Tokenizer(min_length=3, max_length=50)
    assert tokenizer.min_length == 3
    assert tokenizer.max_length == 50
    assert tokenizer.stop_words == set()

def test_tokenizer_add_stop_words():
    """
    Test adding stop words to tokenizer
    トークナイザーへのストップワード追加のテスト
    """
    tokenizer = Tokenizer()
    stop_words = ["the", "a", "an", "and", "or", "but"]
    tokenizer.add_stop_words(stop_words)
    assert tokenizer.stop_words == set(stop_words)

def test_tokenizer_normalize_text():
    """
    Test text normalization
    テキストの正規化テスト
    """
    tokenizer = Tokenizer()
    text = "This is a TEST document! With some special characters: @#$%^&*"
    normalized = tokenizer.normalize_text(text)
    assert normalized == "this is a test document with some special characters"

def test_tokenizer_tokenize():
    """
    Test text tokenization
    テキストのトークン化テスト
    """
    tokenizer = Tokenizer()
    text = "This is a test document with some special characters."
    tokens = tokenizer.tokenize(text)
    assert len(tokens) > 0
    assert all(isinstance(token, str) for token in tokens)
    assert all(tokenizer.min_length <= len(token) <= tokenizer.max_length for token in tokens)

def test_tokenizer_with_stop_words():
    """
    Test tokenization with stop words
    ストップワードを使用したトークン化のテスト
    """
    tokenizer = Tokenizer()
    stop_words = ["is", "a", "with", "some"]
    tokenizer.add_stop_words(stop_words)
    text = "This is a test document with some special characters."
    tokens = tokenizer.tokenize(text)
    assert all(token not in stop_words for token in tokens)

def test_tokenizer_short_words():
    """
    Test tokenization with short words
    短い単語を使用したトークン化のテスト
    """
    tokenizer = Tokenizer(min_length=3)
    text = "a b c d e f g h i j"
    tokens = tokenizer.tokenize(text)
    assert len(tokens) == 0

def test_tokenizer_long_words():
    """
    Test tokenization with long words
    長い単語を使用したトークン化のテスト
    """
    tokenizer = Tokenizer(max_length=5)
    text = "supercalifragilisticexpialidocious"
    tokens = tokenizer.tokenize(text)
    assert len(tokens) == 0

def test_tokenizer_serialization():
    """
    Test tokenizer serialization and deserialization
    トークナイザーのシリアライズとデシリアライズのテスト
    """
    tokenizer = Tokenizer(min_length=3, max_length=50)
    stop_words = ["the", "a", "an", "and", "or", "but"]
    tokenizer.add_stop_words(stop_words)

    # Test serialization
    tokenizer_dict = tokenizer.to_dict()
    assert isinstance(tokenizer_dict, dict)
    assert tokenizer_dict["min_length"] == 3
    assert tokenizer_dict["max_length"] == 50
    assert tokenizer_dict["stop_words"] == list(stop_words)

    # Test deserialization
    new_tokenizer = Tokenizer.from_dict(tokenizer_dict)
    assert new_tokenizer.min_length == tokenizer.min_length
    assert new_tokenizer.max_length == tokenizer.max_length
    assert new_tokenizer.stop_words == tokenizer.stop_words 