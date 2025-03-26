"""
Test BM25 ranking algorithm
BM25ランキングアルゴリズムのテスト
"""

import pytest
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
    doc_ids = ["doc1", "doc2", "doc3"]

    bm25.fit(corpus, doc_ids)
    assert bm25.initialized
    assert len(bm25.doc_freqs) > 0
    assert len(bm25.doc_lens) == 3
    assert bm25.avg_doc_len > 0

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
    doc_ids = ["doc1", "doc2", "doc3"]

    bm25.fit(corpus, doc_ids)
    query = ["test", "document"]
    scores = bm25.get_scores(query)
    
    assert len(scores) == 3
    assert all(isinstance(score, float) for score in scores.values())
    assert all(score >= 0 for score in scores.values())

def test_bm25_empty_corpus():
    """
    Test BM25 with empty corpus
    空のコーパスを使用したBM25のテスト
    """
    bm25 = BM25()
    corpus = []
    doc_ids = []

    bm25.fit(corpus, doc_ids)
    assert bm25.initialized
    assert len(bm25.doc_freqs) == 0
    assert len(bm25.doc_lens) == 0
    assert bm25.avg_doc_len == 0

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
    doc_ids = ["doc1", "doc2", "doc3"]

    bm25.fit(corpus, doc_ids)
    query = ["unknown", "term"]
    scores = bm25.get_scores(query)
    
    assert len(scores) == 3
    assert all(score == 0 for score in scores.values())

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
    doc_ids = ["doc1", "doc2", "doc3"]

    bm25.fit(corpus, doc_ids)
    data = bm25.to_dict()
    
    new_bm25 = BM25.from_dict(data)
    assert new_bm25.k1 == bm25.k1
    assert new_bm25.b == bm25.b
    assert new_bm25.doc_freqs == bm25.doc_freqs
    assert new_bm25.doc_lens == bm25.doc_lens
    assert new_bm25.avg_doc_len == bm25.avg_doc_len
    assert new_bm25.corpus == bm25.corpus
    assert new_bm25.initialized == bm25.initialized 