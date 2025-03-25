"""
Tokenizer implementation for text processing
テキスト処理のためのトークナイザー実装

This module provides a tokenizer implementation for processing text into tokens.
このモジュールは、テキストをトークンに処理するためのトークナイザー実装を提供します。
"""

from typing import List, Dict, Any
import re
import unicodedata

class Tokenizer:
    """
    Tokenizer for processing text into tokens
    テキストをトークンに処理するためのトークナイザー
    """

    def __init__(self, min_length: int = 2, max_length: int = 100):
        """
        Initialize tokenizer
        トークナイザーを初期化

        Args:
            min_length (int): Minimum token length / トークンの最小長
            max_length (int): Maximum token length / トークンの最大長
        """
        self.min_length = min_length
        self.max_length = max_length
        self.stop_words = set()

    def add_stop_words(self, words: List[str]) -> None:
        """
        Add stop words to filter out
        除外するストップワードを追加

        Args:
            words (List[str]): List of stop words / ストップワードのリスト
        """
        self.stop_words.update(words)

    def normalize_text(self, text: str) -> str:
        """
        Normalize text by removing special characters and converting to lowercase
        特殊文字を削除し、小文字に変換してテキストを正規化

        Args:
            text (str): Text to normalize / 正規化するテキスト

        Returns:
            str: Normalized text / 正規化されたテキスト
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        return text.strip()

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words
        テキストを単語にトークン化

        Args:
            text (str): Text to tokenize / トークン化するテキスト

        Returns:
            List[str]: List of tokens / トークンのリスト
        """
        # Normalize text
        text = self.normalize_text(text)
        
        # Split into words
        tokens = text.split()
        
        # Filter tokens by length and stop words
        tokens = [
            token for token in tokens
            if self.min_length <= len(token) <= self.max_length
            and token not in self.stop_words
        ]
        
        return tokens

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert tokenizer instance to dictionary for serialization
        シリアライズのためにトークナイザーインスタンスを辞書に変換

        Returns:
            Dict[str, Any]: Dictionary representation of tokenizer / トークナイザーの辞書表現
        """
        return {
            "min_length": self.min_length,
            "max_length": self.max_length,
            "stop_words": list(self.stop_words)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tokenizer":
        """
        Create tokenizer instance from dictionary
        辞書からトークナイザーインスタンスを作成

        Args:
            data (Dict[str, Any]): Dictionary representation of tokenizer / トークナイザーの辞書表現

        Returns:
            Tokenizer: New tokenizer instance / 新しいトークナイザーインスタンス
        """
        instance = cls(min_length=data["min_length"], max_length=data["max_length"])
        instance.stop_words = set(data["stop_words"])
        return instance 