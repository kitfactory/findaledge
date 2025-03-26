"""
BM25 ranking algorithm implementation
BM25ランキングアルゴリズムの実装
"""

from typing import Dict, List, Optional
import json
import math
from collections import defaultdict

class BM25:
    """
    BM25 ranking algorithm implementation
    BM25ランキングアルゴリズムの実装
    """
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        doc_freqs: Optional[Dict[str, int]] = None,
        doc_lens: Optional[List[int]] = None,
        avg_doc_len: Optional[float] = None,
        corpus: Optional[List[List[str]]] = None,
        doc_ids: Optional[List[str]] = None
    ) -> None:
        """
        Initialize BM25
        BM25を初期化

        Args:
            k1 (float): Term frequency saturation parameter / 用語頻度飽和パラメータ
            b (float): Length normalization parameter / 長さ正規化パラメータ
            doc_freqs (Optional[Dict[str, int]]): Document frequencies / 文書頻度
            doc_lens (Optional[List[int]]): Document lengths / 文書の長さ
            avg_doc_len (Optional[float]): Average document length / 平均文書長
            corpus (Optional[List[List[str]]]): Corpus of documents / 文書コーパス
            doc_ids (Optional[List[str]]): List of document IDs / 文書IDのリスト
        """
        self.k1 = k1
        self.b = b
        self.doc_freqs = doc_freqs or defaultdict(int)
        self.doc_lens = doc_lens or []
        self.avg_doc_len = avg_doc_len or 0.0
        self.corpus = corpus or []
        self.doc_ids = doc_ids or []
        self.initialized = False

    def fit(self, documents: List[List[str]], doc_ids: List[str]) -> None:
        """
        Fit BM25 model to documents
        BM25モデルを文書に適合させる

        Args:
            documents (List[List[str]]): List of tokenized documents / トークン化された文書のリスト
            doc_ids (List[str]): List of document IDs / 文書IDのリスト
        """
        if len(documents) != len(doc_ids):
            raise ValueError("Number of documents must match number of document IDs")

        self.corpus = documents
        self.doc_ids = doc_ids
        self.doc_freqs = defaultdict(int)
        self.doc_lens = []
        total_len = 0

        # 文書頻度と文書長を計算
        for doc in documents:
            doc_len = len(doc)
            self.doc_lens.append(doc_len)
            total_len += doc_len
            for term in set(doc):
                self.doc_freqs[term] += 1

        # 平均文書長を計算
        self.avg_doc_len = total_len / len(documents) if documents else 0.0
        self.initialized = True

    def score(self, query: List[str], doc_id: str) -> float:
        """
        Calculate BM25 score for a query and document
        クエリと文書のBM25スコアを計算

        Args:
            query (List[str]): Query terms / クエリ用語
            doc_id (str): Document ID / 文書ID

        Returns:
            float: BM25 score / BM25スコア
        """
        if not self.initialized:
            raise ValueError("BM25 model must be fitted before scoring")

        try:
            doc_idx = self.doc_ids.index(doc_id)
        except ValueError:
            raise ValueError(f"Document ID {doc_id} not found in corpus")

        doc = self.corpus[doc_idx]
        doc_len = self.doc_lens[doc_idx]

        score = 0.0
        for term in query:
            if term not in self.doc_freqs:
                continue

            # 用語頻度を計算
            tf = doc.count(term) / doc_len

            # IDFを計算
            idf = math.log(
                (len(self.corpus) - self.doc_freqs[term] + 0.5) /
                (self.doc_freqs[term] + 0.5) + 1
            )

            # 長さ正規化を適用
            length_norm = (1 - self.b + self.b * doc_len / self.avg_doc_len)

            # スコアに加算
            score += (idf * tf * (self.k1 + 1)) / (tf + self.k1 * length_norm)

        return score

    def get_scores(self, query: List[str]) -> Dict[str, float]:
        """
        Calculate BM25 scores for a query against all documents
        クエリに対する全文書のBM25スコアを計算

        Args:
            query (List[str]): Query terms / クエリ用語

        Returns:
            Dict[str, float]: Dictionary of document IDs and scores / 文書IDとスコアの辞書
        """
        if not self.initialized:
            raise ValueError("BM25 model must be fitted before scoring")

        return {
            doc_id: self.score(query, doc_id)
            for doc_id in self.doc_ids
        }

    def to_dict(self) -> Dict:
        """
        Convert BM25 model to dictionary
        BM25モデルを辞書に変換

        Returns:
            Dict: Dictionary representation of BM25 model / BM25モデルの辞書表現
        """
        return {
            "k1": self.k1,
            "b": self.b,
            "doc_freqs": dict(self.doc_freqs),
            "doc_lens": self.doc_lens,
            "avg_doc_len": self.avg_doc_len,
            "corpus": self.corpus,
            "doc_ids": self.doc_ids,
            "initialized": self.initialized
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "BM25":
        """
        Create BM25 model from dictionary
        辞書からBM25モデルを作成

        Args:
            data (Dict): Dictionary representation of BM25 model / BM25モデルの辞書表現

        Returns:
            BM25: BM25 model instance / BM25モデルインスタンス
        """
        instance = cls(
            k1=data["k1"],
            b=data["b"],
            doc_freqs=data["doc_freqs"],
            doc_lens=data["doc_lens"],
            avg_doc_len=data["avg_doc_len"],
            corpus=data["corpus"],
            doc_ids=data["doc_ids"]
        )
        instance.initialized = data["initialized"]
        return instance

    def save(self, file_path: str) -> None:
        """
        Save BM25 instance to file
        BM25インスタンスをファイルに保存

        Args:
            file_path (str): Path to save file / 保存先ファイルパス
        """
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def load(self, file_path: str) -> None:
        """
        Load BM25 instance from file
        ファイルからBM25インスタンスを読み込む

        Args:
            file_path (str): Path to load file / 読み込むファイルパス
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        instance = self.from_dict(data)
        self.__dict__.update(instance.__dict__) 