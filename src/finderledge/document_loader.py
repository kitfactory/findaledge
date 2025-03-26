"""
Document loader module for loading documents from various file formats
様々なファイル形式から文書を読み込むためのドキュメントローダーモジュール
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Callable
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
)
from .text_splitter import TextSplitter
from .document import Document

class DocumentLoader:
    """
    Document loader class
    文書ローダークラス
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document loader
        文書ローダーを初期化

        Args:
            chunk_size (int): Size of text chunks / テキストチャンクのサイズ
            chunk_overlap (int): Overlap between chunks / チャンク間のオーバーラップ
        """
        self.text_splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def load_file(self, file_path: Union[str, Path]) -> str:
        """
        Load a file
        ファイルを読み込む

        Args:
            file_path (Union[str, Path]): Path to file / ファイルパス

        Returns:
            str: File content / ファイル内容
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix == ".txt":
            return self.load_text(file_path)
        elif file_path.suffix == ".md":
            return self.load_markdown(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

    def load_directory(self, directory: Union[str, Path], file_filter: Optional[Callable[[Path], bool]] = None) -> List[str]:
        """
        Load all files in a directory
        ディレクトリ内の全ファイルを読み込む

        Args:
            directory (Union[str, Path]): Directory path / ディレクトリパス
            file_filter (Optional[Callable[[Path], bool]]): File filter function / ファイルフィルター関数

        Returns:
            List[str]: List of file contents / ファイル内容のリスト
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        contents = []
        for file_path in directory.glob("**/*"):
            if file_path.is_file() and (file_filter is None or file_filter(file_path)):
                try:
                    content = self.load_file(file_path)
                    contents.append(content)
                except ValueError:
                    continue

        return contents

    def load_text(self, file_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Load a text file
        テキストファイルを読み込む

        Args:
            file_path (Union[str, Path]): Path to text file / テキストファイルパス
            metadata (Optional[Dict[str, Any]]): Document metadata / 文書メタデータ

        Returns:
            str: Text content / テキスト内容
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return content

    def load_markdown(self, file_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Load a Markdown file
        Markdownファイルを読み込む

        Args:
            file_path (Union[str, Path]): Path to Markdown file / Markdownファイルパス
            metadata (Optional[Dict[str, Any]]): Document metadata / 文書メタデータ

        Returns:
            str: Markdown content / Markdown内容
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return content

    def load_document(self, file_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Load a document
        文書を読み込む

        Args:
            file_path (Union[str, Path]): Path to document / 文書パス
            metadata (Optional[Dict[str, Any]]): Document metadata / 文書メタデータ

        Returns:
            str: Document content / 文書内容
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return content

    def load_documents(self, file_paths: List[Union[str, Path]], metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Load multiple documents
        複数の文書を読み込む

        Args:
            file_paths (List[Union[str, Path]]): List of file paths / ファイルパスのリスト
            metadata (Optional[Dict[str, Any]]): Document metadata / 文書メタデータ

        Returns:
            List[str]: List of document contents / 文書内容のリスト
        """
        contents = []
        for file_path in file_paths:
            try:
                content = self.load_document(file_path, metadata)
                contents.append(content)
            except (FileNotFoundError, ValueError):
                continue

        return contents
    
    def load_json(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a JSON document
        JSON文書を読み込む

        Args:
            file_path (Union[str, Path]): Path to the JSON file
                                        JSONファイルへのパス

        Returns:
            Dict[str, Any]: The loaded JSON data
                           読み込まれたJSONデータ

        Raises:
            FileNotFoundError: If the file does not exist
                             ファイルが存在しない場合
            json.JSONDecodeError: If the file is not valid JSON
                                ファイルが有効なJSONでない場合
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def load_markdown(self, file_path: Union[str, Path]) -> str:
        """
        Load a Markdown document
        Markdown文書を読み込む

        Args:
            file_path (Union[str, Path]): Path to the Markdown file
                                        Markdownファイルへのパス

        Returns:
            str: The loaded Markdown text
                 読み込まれたMarkdownテキスト

        Raises:
            FileNotFoundError: If the file does not exist
                             ファイルが存在しない場合
        """
        return self.load_document(file_path) 