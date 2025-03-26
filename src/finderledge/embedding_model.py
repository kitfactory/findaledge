"""
Embedding model implementation for text embeddings
テキストの埋め込みのための埋め込みモデル実装

This module provides an embedding model implementation for generating text embeddings.
このモジュールは、テキストの埋め込みを生成するための埋め込みモデル実装を提供します。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import numpy as np
from openai import OpenAI
from chromadb.api.types import EmbeddingFunction, Embeddable

class EmbeddingModel(ABC, EmbeddingFunction):
    """
    Abstract base class for embedding models
    埋め込みモデルの抽象基底クラス
    """

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents
        文書のリストを埋め込む

        Args:
            texts (List[str]): List of texts to embed / 埋め込むテキストのリスト

        Returns:
            List[List[float]]: List of embeddings / 埋め込みのリスト
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query text
        クエリテキストを埋め込む

        Args:
            text (str): Text to embed / 埋め込むテキスト

        Returns:
            List[float]: Embedding / 埋め込み
        """
        pass

    def __call__(self, input: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Call the embedding function
        埋め込み関数を呼び出す

        Args:
            input (Union[str, List[str]]): Input text or list of texts / 入力テキストまたはテキストのリスト

        Returns:
            Union[List[float], List[List[float]]]: Embedding or list of embeddings / 埋め込みまたは埋め込みのリスト
        """
        if isinstance(input, str):
            return self.embed_query(input)
        else:
            return self.embed_documents(input)

class OpenAIEmbeddingModel(EmbeddingModel):
    """
    OpenAI API based embedding model
    OpenAI APIベースの埋め込みモデル
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        """
        Initialize the model
        モデルを初期化

        Args:
            model (str): Name of the OpenAI embedding model / OpenAI埋め込みモデルの名前
        """
        self.client = OpenAI()
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents
        文書のリストを埋め込む

        Args:
            texts (List[str]): List of texts to embed / 埋め込むテキストのリスト

        Returns:
            List[List[float]]: List of embeddings / 埋め込みのリスト
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [data.embedding for data in response.data]

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query text
        クエリテキストを埋め込む

        Args:
            text (str): Text to embed / 埋め込むテキスト

        Returns:
            List[float]: Embedding / 埋め込み
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=[text]
        )
        return response.data[0].embedding

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model to a dictionary
        モデルを辞書に変換

        Returns:
            Dict[str, Any]: Dictionary representation of the model / モデルの辞書表現
        """
        return {
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenAIEmbeddingModel":
        """
        Create a model from a dictionary
        辞書からモデルを作成

        Args:
            data (Dict[str, Any]): Dictionary representation of the model / モデルの辞書表現

        Returns:
            OpenAIEmbeddingModel: Created model / 作成されたモデル
        """
        return cls(model=data["model"]) 