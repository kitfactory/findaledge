"""
Document loader module for loading documents from various file formats
様々なファイル形式から文書を読み込むためのドキュメントローダーモジュール
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Union
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
)

class DocumentLoader:
    """
    A class for loading documents from various file formats
    様々なファイル形式から文書を読み込むためのクラス
    """
    
    def __init__(self):
        """
        Initialize the document loader with supported file extensions
        サポートされているファイル拡張子でドキュメントローダーを初期化
        """
        self.loaders = {
            ".txt": TextLoader,
            ".pdf": PyPDFLoader,
            ".md": UnstructuredMarkdownLoader,
        }
    
    def load_document(self, file_path: Union[str, Path]) -> str:
        """
        Load a document from a file
        ファイルから文書を読み込む

        Args:
            file_path (Union[str, Path]): Path to the file
                                        ファイルへのパス

        Returns:
            str: The loaded document text
                 読み込まれた文書のテキスト

        Raises:
            FileNotFoundError: If the file does not exist
                             ファイルが存在しない場合
            ValueError: If the file format is not supported
                       ファイル形式がサポートされていない場合
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = file_path.suffix.lower()
        if ext not in self.loaders:
            raise ValueError(f"Unsupported file format: {ext}")
        
        loader = self.loaders[ext](str(file_path))
        docs = loader.load()
        return "\n".join(doc.page_content for doc in docs)
    
    def load_documents(self, file_paths: List[Union[str, Path]]) -> List[str]:
        """
        Load multiple documents from files
        複数のファイルから文書を読み込む

        Args:
            file_paths (List[Union[str, Path]]): List of file paths
                                               ファイルパスのリスト

        Returns:
            List[str]: List of loaded document texts
                      読み込まれた文書のテキストのリスト
        """
        return [self.load_document(path) for path in file_paths]
    
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