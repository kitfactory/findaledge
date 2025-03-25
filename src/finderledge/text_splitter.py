"""
Text splitting utilities for document processing
文書処理のためのテキスト分割ユーティリティ

This module provides functionality for splitting text into chunks
while maintaining context and managing overlap between chunks.
このモジュールは、文脈を維持しながらテキストをチャンクに分割し、
チャンク間のオーバーラップを管理する機能を提供します。
"""

from typing import List, Optional, Callable
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class TextSplitter:
    """
    Text splitter for document processing
    文書処理のためのテキスト分割器
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: Optional[Callable[[str], int]] = None,
        separators: Optional[List[str]] = None
    ) -> None:
        """
        Initialize text splitter
        テキスト分割器を初期化

        Args:
            chunk_size (int): Size of text chunks / テキストチャンクのサイズ
            chunk_overlap (int): Overlap between chunks / チャンク間の重複
            length_function (Optional[Callable[[str], int]]): Function to measure text length / テキストの長さを測る関数
            separators (Optional[List[str]]): List of separators for splitting / 分割に使用する区切り文字のリスト
        """
        if chunk_overlap >= chunk_size:
            chunk_overlap = chunk_size // 2

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function or len,
            separators=separators or ["\n\n", "\n", " ", ""]
        )

    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks
        テキストをチャンクに分割

        Args:
            text (str): Text to split / 分割するテキスト

        Returns:
            List[str]: List of text chunks / テキストチャンクのリスト
        """
        return [doc.page_content for doc in self._splitter.create_documents([text])]

    def split_documents(self, documents: List[str]) -> List[str]:
        """
        Split multiple documents into chunks
        複数の文書をチャンクに分割

        Args:
            documents (List[str]): Documents to split / 分割する文書

        Returns:
            List[str]: List of text chunks / テキストチャンクのリスト
        """
        docs = [Document(page_content=doc) for doc in documents]
        return [doc.page_content for doc in self._splitter.split_documents(docs)] 