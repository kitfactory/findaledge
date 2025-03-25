"""
Tests for the BM25 class
BM25クラスのテスト
"""

import pytest
import numpy as np
from finderledge.bm25 import BM25

def test_bm25_initialization():
    """
    Test BM25 initialization
    BM25の初期化テスト
    """
    bm25 = BM25()
    assert bm25.k1 == 1.5
    assert bm25.b == 0.75
    assert bm25.corpus == []
    assert bm25.doc_freqs == {}
    assert bm25.doc_lens == []
    assert bm25.avg_doc_len == 0
    assert bm25.idf == {}
    assert not bm25.initialized

def test_bm25_fit():
    """
    Test BM25 fitting with corpus
    コーパスを使用したBM25のフィッティングテスト
    """
    bm25 = BM25()
    corpus = [
        ["this", "is", "a", "test", "document"],
        ["this", "is", "another", "test", "document"],
        ["this", "is", "a", "different", "document"]
    ]

    bm25.fit(corpus)

    assert bm25.initialized
    assert len(bm25.corpus) == 3
    assert len(bm25.doc_lens) == 3
    assert all(length == 5 for length in bm25.doc_lens)
    assert bm25.avg_doc_len == 5.0
    assert len(bm25.doc_freqs) > 0
    assert len(bm25.idf) > 0

def test_bm25_get_scores():
    """
    Test BM25 scoring
    BM25のスコアリングテスト
    """
    bm25 = BM25()
    corpus = [
        ["this", "is", "a", "test", "document"],
        ["this", "is", "another", "test", "document"],
        ["this", "is", "a", "different", "document"]
    ]

    bm25.fit(corpus)
    query = ["test", "document"]
    scores = bm25.get_scores(query)

    assert len(scores) == 3
    assert all(isinstance(score, float) for score in scores)
    assert all(score >= 0 for score in scores)

def test_bm25_empty_corpus():
    """
    Test BM25 with empty corpus
    空のコーパスを使用したBM25のテスト
    """
    bm25 = BM25()
    corpus = []
    bm25.fit(corpus)

    query = ["test", "document"]
    scores = bm25.get_scores(query)
    assert len(scores) == 0

def test_bm25_unknown_terms():
    """
    Test BM25 with unknown terms in query
    クエリに未知の用語がある場合のBM25のテスト
    """
    bm25 = BM25()
    corpus = [
        ["this", "is", "a", "test", "document"],
        ["this", "is", "another", "test", "document"],
        ["this", "is", "a", "different", "document"]
    ]

    bm25.fit(corpus)
    query = ["unknown", "terms"]
    scores = bm25.get_scores(query)

    assert len(scores) == 3
    assert all(score == 0.0 for score in scores)

def test_bm25_serialization():
    """
    Test BM25 serialization and deserialization
    BM25のシリアライズとデシリアライズのテスト
    """
    bm25 = BM25()
    corpus = [
        ["this", "is", "a", "test", "document"],
        ["this", "is", "another", "test", "document"],
        ["this", "is", "a", "different", "document"]
    ]

    bm25.fit(corpus)

    # Test serialization
    bm25_dict = bm25.to_dict()
    assert isinstance(bm25_dict, dict)
    assert bm25_dict["k1"] == 1.5
    assert bm25_dict["b"] == 0.75
    assert bm25_dict["corpus"] == corpus
    assert bm25_dict["doc_freqs"] == bm25.doc_freqs
    assert bm25_dict["doc_lens"] == bm25.doc_lens
    assert bm25_dict["avg_doc_len"] == bm25.avg_doc_len
    assert bm25_dict["idf"] == bm25.idf
    assert bm25_dict["initialized"] is True

    # Test deserialization
    new_bm25 = BM25.from_dict(bm25_dict)
    assert new_bm25.k1 == bm25.k1
    assert new_bm25.b == bm25.b
    assert new_bm25.corpus == bm25.corpus
    assert new_bm25.doc_freqs == bm25.doc_freqs
    assert new_bm25.doc_lens == bm25.doc_lens
    assert new_bm25.avg_doc_len == bm25.avg_doc_len
    assert new_bm25.idf == bm25.idf
    assert new_bm25.initialized is True 