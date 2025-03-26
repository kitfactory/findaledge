"""
Tests for the Tokenizer class
Tokenizerクラスのテスト
"""

import pytest
from finderledge import Tokenizer

def test_tokenizer_basic():
    """
    Test basic tokenizer functionality
    基本的なトークナイザー機能をテスト
    """
    tokenizer = Tokenizer()
    text = "This is a test document."
    tokens = tokenizer.tokenize(text)
    
    assert isinstance(tokens, list)
    assert all(isinstance(token, str) for token in tokens)
    assert len(tokens) > 0

def test_tokenizer_min_length():
    """
    Test tokenizer with minimum length filter
    最小長フィルタ付きトークナイザーをテスト
    """
    tokenizer = Tokenizer(min_length=4)
    text = "This is a test document with some short and long words."
    tokens = tokenizer.tokenize(text)
    
    assert all(len(token) >= 4 for token in tokens)

def test_tokenizer_max_length():
    """
    Test tokenizer with maximum length filter
    最大長フィルタ付きトークナイザーをテスト
    """
    tokenizer = Tokenizer(max_length=6)
    text = "This is a test document with some short and long words."
    tokens = tokenizer.tokenize(text)
    
    assert all(len(token) <= 6 for token in tokens)

def test_tokenizer_stop_words():
    """
    Test tokenizer with stop words
    ストップワード付きトークナイザーをテスト
    """
    tokenizer = Tokenizer()
    stop_words = ["the", "a", "an", "and", "or", "but"]
    tokenizer.add_stop_words(stop_words)
    
    text = "The cat and the dog are playing with a ball."
    tokens = tokenizer.tokenize(text)
    
    assert not any(token.lower() in stop_words for token in tokens)

def test_tokenizer_empty_text():
    """
    Test tokenizer with empty text
    空のテキストでのトークナイザーをテスト
    """
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize("")
    
    assert isinstance(tokens, list)
    assert len(tokens) == 0

def test_tokenizer_custom_filters():
    """
    Test tokenizer with custom filters
    カスタムフィルタ付きトークナイザーをテスト
    """
    tokenizer = Tokenizer()
    tokenizer.add_filter(lambda x: x.upper())
    
    text = "This is a test."
    tokens = tokenizer.tokenize(text)
    
    assert all(token.isupper() for token in tokens)

def test_tokenizer_serialization():
    """
    Test tokenizer serialization and deserialization
    トークナイザーのシリアライズとデシリアライズをテスト
    """
    tokenizer = Tokenizer(min_length=3, max_length=50)
    stop_words = ["the", "a", "an", "and", "or", "but"]
    tokenizer.add_stop_words(stop_words)
    
    # Test serialization
    tokenizer_dict = tokenizer.to_dict()
    assert isinstance(tokenizer_dict, dict)
    assert tokenizer_dict["min_length"] == 3
    assert tokenizer_dict["max_length"] == 50
    assert set(tokenizer_dict["stop_words"]) == set(stop_words)
    
    # Test deserialization
    new_tokenizer = Tokenizer.from_dict(tokenizer_dict)
    assert new_tokenizer.min_length == tokenizer.min_length
    assert new_tokenizer.max_length == tokenizer.max_length
    assert set(new_tokenizer.stop_words) == set(tokenizer.stop_words)

def test_tokenizer_invalid_params():
    """
    Test tokenizer with invalid parameters
    不正なパラメータでのトークナイザーをテスト
    """
    with pytest.raises(ValueError):
        Tokenizer(min_length=-1)
    
    with pytest.raises(ValueError):
        Tokenizer(max_length=0)
    
    with pytest.raises(ValueError):
        Tokenizer(min_length=5, max_length=3) 