"""
EmbeddingModel - Interface and implementations for text embedding models
EmbeddingModel - テキスト埋め込みモデルのインターフェースと実装

This module provides an interface for text embedding models and implementations
for various embedding services like OpenAI and HuggingFace.
このモジュールは、テキスト埋め込みモデルのインターフェースと、
OpenAIやHuggingFaceなどの各種埋め込みサービスの実装を提供します。
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

class EmbeddingModel(ABC):
    """
    Abstract base class for text embedding models
    テキスト埋め込みモデルの抽象基底クラス

    This class defines the interface that all embedding model implementations
    must follow. It provides methods for generating embeddings from text.
    このクラスは、全ての埋め込みモデル実装が従わなければならない
    インターフェースを定義します。テキストから埋め込みを生成する
    メソッドを提供します。
    """

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents
        文書リストの埋め込みを生成

        Args:
            texts (List[str]): List of text documents / テキスト文書のリスト

        Returns:
            List[List[float]]: List of embedding vectors / 埋め込みベクトルのリスト
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a query text
        クエリテキストの埋め込みを生成

        Args:
            text (str): Query text / クエリテキスト

        Returns:
            List[float]: Embedding vector / 埋め込みベクトル
        """
        pass

class OpenAIEmbeddingModel(EmbeddingModel):
    """
    OpenAI's embedding model implementation
    OpenAIの埋め込みモデルの実装
    """
    def __init__(self, api_key: Optional[str] = None, model_name: str = "text-embedding-ada-002") -> None:
        """
        Initialize OpenAI embedding model
        OpenAI埋め込みモデルを初期化

        Args:
            api_key (Optional[str]): OpenAI API key / OpenAI APIキー
            model_name (str): Model name / モデル名
        """
        super().__init__()
        self.model_name = model_name
        if api_key:
            openai.api_key = api_key
        elif not openai.api_key:
            raise ValueError("OpenAI API key is required")

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text using OpenAI's model
        OpenAIのモデルを使用して単一のテキストを埋め込む

        Args:
            text (str): Text to embed / 埋め込むテキスト

        Returns:
            List[float]: Embedding vector / 埋め込みベクトル
        """
        response = openai.embeddings.create(
            model=self.model_name,
            input=text
        )
        return response.data[0].embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts using OpenAI's model
        OpenAIのモデルを使用して複数のテキストを埋め込む

        Args:
            texts (List[str]): Texts to embed / 埋め込むテキスト

        Returns:
            List[List[float]]: List of embedding vectors / 埋め込みベクトルのリスト
        """
        response = openai.embeddings.create(
            model=self.model_name,
            input=texts
        )
        return [item.embedding for item in response.data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents using OpenAI's API
        OpenAIのAPIを使用して文書リストの埋め込みを生成

        Args:
            texts (List[str]): List of text documents / テキスト文書のリスト

        Returns:
            List[List[float]]: List of embedding vectors / 埋め込みベクトルのリスト
        """
        return self.embed_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a query text using OpenAI's API
        OpenAIのAPIを使用してクエリテキストの埋め込みを生成

        Args:
            text (str): Query text / クエリテキスト

        Returns:
            List[float]: Embedding vector / 埋め込みベクトル
        """
        return self.embed_text(text)

class HuggingFaceEmbeddingModel(EmbeddingModel):
    """
    HuggingFace's embedding model implementation
    HuggingFaceの埋め込みモデルの実装
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """
        Initialize HuggingFace embedding model
        HuggingFace埋め込みモデルを初期化

        Args:
            model_name (str): Model name / モデル名
        """
        super().__init__()
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except ImportError as e:
            raise ImportError("Required packages not found. Please install torch and transformers.") from e

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text using HuggingFace model
        HuggingFaceモデルを使用して単一のテキストを埋め込む

        Args:
            text (str): Text to embed / 埋め込むテキスト

        Returns:
            List[float]: Embedding vector / 埋め込みベクトル
        """
        import torch
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings[0].cpu().numpy().tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts using HuggingFace model
        HuggingFaceモデルを使用して複数のテキストを埋め込む

        Args:
            texts (List[str]): Texts to embed / 埋め込むテキスト

        Returns:
            List[List[float]]: List of embedding vectors / 埋め込みベクトルのリスト
        """
        import torch
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents using HuggingFace's model
        HuggingFaceのモデルを使用して文書リストの埋め込みを生成

        Args:
            texts (List[str]): List of text documents / テキスト文書のリスト

        Returns:
            List[List[float]]: List of embedding vectors / 埋め込みベクトルのリスト
        """
        return self.embed_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a query text using HuggingFace's model
        HuggingFaceのモデルを使用してクエリテキストの埋め込みを生成

        Args:
            text (str): Query text / クエリテキスト

        Returns:
            List[float]: Embedding vector / 埋め込みベクトル
        """
        return self.embed_text(text) 