"""
BM25 implementation for document ranking
文書ランキングのためのBM25実装

This module provides a BM25 implementation for ranking documents based on query relevance.
このモジュールは、クエリの関連性に基づいて文書をランク付けするためのBM25実装を提供します。
"""

from typing import List, Dict, Any
import math
import numpy as np

class BM25:
    """
    BM25 implementation for document ranking
    文書ランキングのためのBM25実装
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 with parameters
        パラメータを指定してBM25を初期化

        Args:
            k1 (float): Term frequency saturation parameter / 単語頻度の飽和パラメータ
            b (float): Length normalization parameter / 長さの正規化パラメータ
        """
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_freqs = {}
        self.doc_lens = []
        self.avg_doc_len = 0
        self.idf = {}
        self.initialized = False

    def fit(self, corpus: List[List[str]]) -> None:
        """
        Fit BM25 parameters to the corpus
        コーパスにBM25パラメータを適合させる

        Args:
            corpus (List[List[str]]): List of tokenized documents / トークン化された文書のリスト
        """
        self.corpus = corpus
        self.doc_lens = [len(doc) for doc in corpus]
        self.avg_doc_len = sum(self.doc_lens) / len(corpus) if corpus else 0

        # Calculate document frequencies
        self.doc_freqs = {}
        for doc in corpus:
            for word in set(doc):
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1

        # Calculate IDF
        num_docs = len(corpus)
        self.idf = {
            word: math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1.0)
            for word, freq in self.doc_freqs.items()
        }

        self.initialized = True

    def get_scores(self, query: List[str]) -> List[float]:
        """
        Calculate BM25 scores for a query
        クエリに対するBM25スコアを計算

        Args:
            query (List[str]): Tokenized query / トークン化されたクエリ

        Returns:
            List[float]: List of scores for each document / 各文書のスコアリスト
        """
        if not self.initialized:
            return [0.0] * len(self.corpus)

        scores = np.zeros(len(self.corpus))
        for word in query:
            if word not in self.idf:
                continue
            idf = self.idf[word]
            for doc_idx, doc in enumerate(self.corpus):
                freq = doc.count(word)
                if freq == 0:
                    continue
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * self.doc_lens[doc_idx] / self.avg_doc_len)
                scores[doc_idx] += idf * numerator / denominator

        return scores.tolist()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert BM25 instance to dictionary for serialization
        シリアライズのためにBM25インスタンスを辞書に変換

        Returns:
            Dict[str, Any]: Dictionary representation of BM25 / BM25の辞書表現
        """
        return {
            "k1": self.k1,
            "b": self.b,
            "corpus": self.corpus,
            "doc_freqs": self.doc_freqs,
            "doc_lens": self.doc_lens,
            "avg_doc_len": self.avg_doc_len,
            "idf": self.idf,
            "initialized": self.initialized
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BM25":
        """
        Create BM25 instance from dictionary
        辞書からBM25インスタンスを作成

        Args:
            data (Dict[str, Any]): Dictionary representation of BM25 / BM25の辞書表現

        Returns:
            BM25: New BM25 instance / 新しいBM25インスタンス
        """
        instance = cls(k1=data["k1"], b=data["b"])
        instance.corpus = data["corpus"]
        instance.doc_freqs = data["doc_freqs"]
        instance.doc_lens = data["doc_lens"]
        instance.avg_doc_len = data["avg_doc_len"]
        instance.idf = data["idf"]
        instance.initialized = data["initialized"]
        return instance 