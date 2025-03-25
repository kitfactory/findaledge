"""
EmbeddingStore - A class for storing and retrieving document embeddings
EmbeddingStore - 文書の埋め込みを保存・取得するためのクラス

This class provides functionality for storing and retrieving document embeddings.
このクラスは、文書の埋め込みを保存・取得する機能を提供します。
"""

import os
import json
from typing import Dict, List, Optional, Union
import numpy as np

class EmbeddingStore:
    """
    A class for storing and retrieving document embeddings
    文書の埋め込みを保存・取得するためのクラス

    Attributes:
        store_dir (str): Directory to store embeddings
        embeddings (Dict[str, np.ndarray]): Dictionary mapping document IDs to embeddings
    """

    def __init__(self, store_dir: str = "embeddings"):
        """
        Initialize the embedding store
        埋め込みストアを初期化

        Args:
            store_dir (str): Directory to store embeddings
        """
        self.store_dir = store_dir
        self.embeddings: Dict[str, np.ndarray] = {}
        os.makedirs(store_dir, exist_ok=True)

    def add_embedding(self, doc_id: str, embedding: np.ndarray) -> None:
        """
        Add an embedding for a document
        文書の埋め込みを追加

        Args:
            doc_id (str): Document ID
            embedding (np.ndarray): Document embedding
        """
        self.embeddings[doc_id] = embedding
        self._save_embedding(doc_id, embedding)

    def get_embedding(self, doc_id: str) -> Optional[np.ndarray]:
        """
        Get the embedding for a document
        文書の埋め込みを取得

        Args:
            doc_id (str): Document ID

        Returns:
            Optional[np.ndarray]: Document embedding if found, None otherwise
        """
        if doc_id in self.embeddings:
            return self.embeddings[doc_id]
        
        # Try to load from disk
        embedding = self._load_embedding(doc_id)
        if embedding is not None:
            self.embeddings[doc_id] = embedding
            return embedding
        
        return None

    def remove_embedding(self, doc_id: str) -> None:
        """
        Remove an embedding for a document
        文書の埋め込みを削除

        Args:
            doc_id (str): Document ID
        """
        if doc_id in self.embeddings:
            del self.embeddings[doc_id]
        self._remove_embedding_file(doc_id)

    def delete_embedding(self, doc_id: str) -> None:
        """
        Delete the embedding for a document
        文書の埋め込みを削除

        Args:
            doc_id (str): Document ID
        """
        if doc_id in self.embeddings:
            del self.embeddings[doc_id]
        self._remove_embedding_file(doc_id)

    def _save_embedding(self, doc_id: str, embedding: np.ndarray) -> None:
        """
        Save an embedding to disk
        埋め込みをディスクに保存

        Args:
            doc_id (str): Document ID
            embedding (np.ndarray): Document embedding
        """
        file_path = os.path.join(self.store_dir, f"{doc_id}.json")
        with open(file_path, "w") as f:
            json.dump({
                "doc_id": doc_id,
                "embedding": embedding.tolist()
            }, f)

    def _load_embedding(self, doc_id: str) -> Optional[np.ndarray]:
        """
        Load an embedding from disk
        埋め込みをディスクから読み込む

        Args:
            doc_id (str): Document ID

        Returns:
            Optional[np.ndarray]: Document embedding if found, None otherwise
        """
        file_path = os.path.join(self.store_dir, f"{doc_id}.json")
        if not os.path.exists(file_path):
            return None

        with open(file_path, "r") as f:
            data = json.load(f)
            return np.array(data["embedding"])

    def _remove_embedding_file(self, doc_id: str) -> None:
        """
        Remove an embedding file from disk
        埋め込みファイルをディスクから削除

        Args:
            doc_id (str): Document ID
        """
        file_path = os.path.join(self.store_dir, f"{doc_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)

    def to_dict(self) -> Dict:
        """
        Convert the store to a dictionary
        ストアを辞書に変換

        Returns:
            Dict: Dictionary representation of the store
        """
        return {
            "store_dir": self.store_dir,
            "embeddings": {
                doc_id: embedding.tolist()
                for doc_id, embedding in self.embeddings.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "EmbeddingStore":
        """
        Create a store from a dictionary
        辞書からストアを作成

        Args:
            data (Dict): Dictionary representation of the store

        Returns:
            EmbeddingStore: New store instance
        """
        store = cls(data["store_dir"])
        store.embeddings = {
            doc_id: np.array(embedding)
            for doc_id, embedding in data["embeddings"].items()
        }
        return store 