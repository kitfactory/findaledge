"""
Embedding model implementation for text embeddings
テキストの埋め込みのための埋め込みモデル実装

This module provides an embedding model implementation for generating text embeddings.
このモジュールは、テキストの埋め込みを生成するための埋め込みモデル実装を提供します。
"""

from typing import List, Dict, Any, Optional
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

class EmbeddingModel:
    """
    Embedding model for generating text embeddings
    テキストの埋め込みを生成するための埋め込みモデル
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding model
        埋め込みモデルを初期化

        Args:
            model_name (str): Name of the model to use / 使用するモデルの名前
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a text
        テキストの埋め込みを生成

        Args:
            text (str): Text to generate embedding for / 埋め込みを生成するテキスト

        Returns:
            np.ndarray: Text embedding / テキストの埋め込み
        """
        # Tokenize text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            embedding = embeddings[0].cpu().numpy()

        return embedding

    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts
        複数のテキストの埋め込みを生成

        Args:
            texts (List[str]): List of texts to generate embeddings for / 埋め込みを生成するテキストのリスト

        Returns:
            List[np.ndarray]: List of text embeddings / テキストの埋め込みのリスト
        """
        return [self.generate_embedding(text) for text in texts]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert embedding model instance to dictionary for serialization
        シリアライズのために埋め込みモデルインスタンスを辞書に変換

        Returns:
            Dict[str, Any]: Dictionary representation of embedding model / 埋め込みモデルの辞書表現
        """
        return {
            "model_name": self.model_name,
            "device": str(self.device)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingModel":
        """
        Create embedding model instance from dictionary
        辞書から埋め込みモデルインスタンスを作成

        Args:
            data (Dict[str, Any]): Dictionary representation of embedding model / 埋め込みモデルの辞書表現

        Returns:
            EmbeddingModel: New embedding model instance / 新しい埋め込みモデルインスタンス
        """
        return cls(model_name=data["model_name"]) 