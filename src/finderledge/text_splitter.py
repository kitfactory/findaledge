"""
Text splitter for document chunking
文書チャンキングのためのテキスト分割器
"""

from typing import List, Optional, Callable
import re

class TextSplitter:
    """
    Text splitter for document chunking
    文書チャンキングのためのテキスト分割器
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

        Raises:
            ValueError: If chunk_overlap is greater than or equal to chunk_size
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function or len
        self.separators = separators or ["\n\n", "\n", ".", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks
        テキストをチャンクに分割

        Args:
            text (str): Text to split / 分割するテキスト

        Returns:
            List[str]: List of text chunks / テキストチャンクのリスト
        """
        if not text:
            return []

        # テキストが1チャンクに収まる場合は分割しない
        text = text.strip()
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # チャンクの終了位置を計算
            end = min(start + self.chunk_size, text_length)

            # 区切り文字で分割位置を探す
            chunk_end = end
            found_separator = False

            # 区切り文字で分割位置を探す（長い区切り文字から順に）
            separators = sorted(self.separators, key=len, reverse=True)
            for sep in separators:
                if sep:
                    # 区切り文字の位置を探す
                    pos = text.rfind(sep, start, end)
                    if pos != -1 and pos > start:  # 区切り文字が開始位置より後にある場合のみ
                        chunk_end = pos + len(sep)
                        found_separator = True
                        break

            # 区切り文字が見つからない場合は、単語の途中で切らないようにする
            if not found_separator and chunk_end < text_length and chunk_end > start:
                # スペースで区切れる位置を探す
                space_pos = text.rfind(" ", start, chunk_end)
                if space_pos != -1 and space_pos > start:
                    chunk_end = space_pos + 1

            # チャンクを追加（空でない場合のみ）
            chunk = text[start:chunk_end].strip()
            if chunk:
                chunks.append(chunk)

            # 次のチャンクの開始位置を計算
            if chunk_end >= text_length:
                break

            # オーバーラップを考慮して次の開始位置を計算
            next_start = chunk_end - self.chunk_overlap
            if next_start <= start:
                next_start = start + 1

            start = next_start

        return chunks

    def split_documents(self, documents: List[str]) -> List[str]:
        """
        Split multiple documents into chunks
        複数の文書をチャンクに分割

        Args:
            documents (List[str]): List of documents / 文書のリスト

        Returns:
            List[str]: List of text chunks / テキストチャンクのリスト
        """
        chunks = []
        for doc in documents:
            chunks.extend(self.split_text(doc))
        return chunks 