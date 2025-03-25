"""
BM25 ranking algorithm implementation
BM25ランキングアルゴリズムの実装
"""

import os
import numpy as np
import json
from typing import Dict, List, Optional

class BM25:
    """
    BM25 ranking algorithm implementation
    BM25ランキングアルゴリズムの実装
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 with parameters
        BM25をパラメータで初期化

        Args:
            k1 (float): Term frequency saturation parameter
            b (float): Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self.doc_freqs: Dict[str, int] = {}
        self.doc_count = 0
        self.avg_doc_len = 0.0
        self.doc_lens: Dict[str, int] = {}
        self.term_freqs: Dict[str, Dict[str, int]] = {}
        self.doc_ids: List[str] = []

    def fit(self, documents: List[str], doc_ids: Optional[List[str]] = None) -> None:
        """
        Fit the BM25 model to the documents
        BM25モデルを文書に適合させる

        Args:
            documents (List[str]): List of document texts
            doc_ids (Optional[List[str]]): List of document IDs
        """
        if doc_ids is None:
            doc_ids = [str(i) for i in range(len(documents))]

        # 文書IDの順序を保持
        self.doc_ids = list(doc_ids)

        # 文書の長さを計算
        self.doc_lens = {}
        self.term_freqs = {}
        for doc_id, doc in zip(doc_ids, documents):
            terms = doc.split()
            self.doc_lens[doc_id] = len(terms)
            for term in terms:
                if term not in self.term_freqs:
                    self.term_freqs[term] = {}
                if doc_id not in self.term_freqs[term]:
                    self.term_freqs[term][doc_id] = 0
                self.term_freqs[term][doc_id] += 1

        # 文書頻度を計算
        self.doc_count = len(documents)
        self.doc_freqs = {}
        for term in self.term_freqs:
            self.doc_freqs[term] = len(self.term_freqs[term])

        # 平均文書長を計算
        self.avg_doc_len = np.mean(list(self.doc_lens.values()))

    def score(self, query: str, doc_id: str) -> float:
        """
        Calculate BM25 score for a document
        文書のBM25スコアを計算

        Args:
            query (str): Search query
            doc_id (str): Document ID

        Returns:
            float: BM25 score
        """
        score = 0.0
        query_terms = query.split()

        for term in query_terms:
            if term not in self.term_freqs or doc_id not in self.term_freqs[term]:
                continue

            tf = self.term_freqs[term][doc_id]
            df = self.doc_freqs[term]
            doc_len = self.doc_lens[doc_id]

            # 逆文書頻度（IDF）の計算
            idf = np.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)

            # 正規化された文書頻度（TF）の計算
            tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len))

            score += idf * tf_norm

        return score

    def save(self, filepath: str) -> None:
        """
        Save the BM25 model state to a file
        BM25モデルの状態をファイルに保存

        Args:
            filepath (str): Path to save the model state
        """
        # ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        state = {
            "k1": self.k1,
            "b": self.b,
            "doc_freqs": self.doc_freqs,
            "doc_count": self.doc_count,
            "avg_doc_len": self.avg_doc_len,
            "doc_lens": self.doc_lens,
            "term_freqs": self.term_freqs,
            "doc_ids": self.doc_ids
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str) -> None:
        """
        Load the BM25 model state from a file
        BM25モデルの状態をファイルから読み込む

        Args:
            filepath (str): Path to load the model state from
        """
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)
            self.k1 = state["k1"]
            self.b = state["b"]
            self.doc_freqs = state["doc_freqs"]
            self.doc_count = state["doc_count"]
            self.avg_doc_len = state["avg_doc_len"]
            self.doc_lens = state["doc_lens"]
            self.term_freqs = state["term_freqs"]
            self.doc_ids = state["doc_ids"] 