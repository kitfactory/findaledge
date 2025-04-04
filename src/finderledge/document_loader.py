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
    # LangChainのDocumentLoaderを使う場合は以下を追加
    # DirectoryLoader, # 必要に応じて
    # JSONLoader,    # 必要に応じて
    # CSVLoader,     # 必要に応じて
)
# from langchain.schema import Document # LangChainのDocumentをインポート
from langchain.schema import Document as LangchainDocument # エイリアスを使用
import os
import markitdown # markitdown をインポート

from .text_splitter import TextSplitter
# from .document import Document # <-- 削除

# markitdown のクラス名をインポート
from markitdown import MarkItDown

class DocumentLoader:
    """
    Loads documents from various file formats using markitdown, pathlib,
    and direct text reading for code files.
    markitdown, pathlib, およびコードファイル用の直接テキスト読み込みを使用して、
    様々な形式からドキュメントをロードします。

    Provides methods to load single files or recursively load files from directories.
    単一ファイルのロード、またはディレクトリからの再帰的なファイルロードを提供します。
    """

    # markitdownがサポートする可能性のある拡張子 (必要に応じて調整)
    SUPPORTED_EXTENSIONS = {
        ".md", ".markdown",
        ".txt", ".text", # Text files also handled by direct read if needed, but markitdown is fine
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".xls",
        ".csv",
        ".html", ".htm",
        ".epub",
        ".rtf",
        ".odt",
        ".ipynb", # Jupyter Notebook
        ".eml", # Email
        ".xml",
        # ".json", # JSONは構造によるため、markitdownで適切に処理できるか注意 -> CODE_EXTENSIONS で処理
        # 画像 (.jpg, .png) や音声 (.wav, .mp3) はテキスト抽出として扱われる
    }

    # プログラミング言語ファイルの拡張子
    CODE_EXTENSIONS = {
        ".py", ".pyw", # Python
        ".java", ".scala", ".kt", # JVM Languages
        ".js", ".jsx", ".ts", ".tsx", # JavaScript/TypeScript
        ".c", ".h", ".cpp", ".hpp", ".cs", # C/C++/C#
        ".go", # Go
        ".rs", # Rust
        ".php", # PHP
        ".rb", # Ruby
        ".swift", # Swift
        ".pl", # Perl
        ".sh", # Shell script
        ".bat", ".cmd", # Windows Batch
        ".ps1", # PowerShell
        ".sql", # SQL
        ".yaml", ".yml", # YAML
        ".json", # JSONもテキストとして読む
        ".dockerfile", "Dockerfile", # Dockerfile (拡張子なしの場合も)
        ".gitignore", ".gitattributes", # Git files
        # 必要に応じて他の拡張子を追加
    }

    def __init__(self):
        # MarkItDownのインスタンスを作成・保持
        self.md_converter = MarkItDown()

    def _load_single_file(self, file_path: Path) -> Optional[LangchainDocument]:
        """
        Loads a single file, handling code files as text and others via markitdown.
        単一ファイルをロードします。コードファイルはテキストとして、
        その他は markitdown 経由で処理します。

        Args:
            file_path (Path): Path to the file. / ファイルへのパス。

        Returns:
            Optional[LangchainDocument]: Loaded document or None if loading failed or skipped.
                                        ロードされたドキュメント、失敗またはスキップされた場合はNone。
        """
        file_suffix = file_path.suffix.lower()
        # Handle files without extension (like Dockerfile) by checking name
        file_name = file_path.name

        metadata = {"source": str(file_path)}

        # 1. Check for Code Files (including extensionless common names)
        if file_suffix in self.CODE_EXTENSIONS or file_name in self.CODE_EXTENSIONS:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                return LangchainDocument(page_content=content, metadata=metadata)
            except Exception as e:
                print(f"Error reading code file {file_path} as text: {e}")
                return None # Or try other encodings if needed

        # 2. Check for Markitdown Supported Files
        elif file_suffix in self.SUPPORTED_EXTENSIONS:
            try:
                # MarkItDownインスタンスの convert メソッドを使用
                result = self.md_converter.convert(str(file_path))
                markdown_content = result.text_content
                # result.metadata から他のメタデータを取得して追加することも可能
                # metadata.update(result.metadata)
                return LangchainDocument(page_content=markdown_content, metadata=metadata)
            except Exception as e:
                print(f"Error loading file {file_path} with markitdown: {e}")
                return None

        # 3. Unsupported File Type
        else:
            print(f"Skipping unsupported file type: {file_path}")
            return None

    def load_file(self, file_path: Union[str, Path]) -> Optional[LangchainDocument]:
        """
        Load a single file.
        単一ファイルをロードします。

        Args:
            file_path (Union[str, Path]): Path to the file. / ファイルへのパス。

        Returns:
            Optional[LangchainDocument]: Loaded document or None if loading failed.
                                        ロードされたドキュメント、または失敗した場合はNone。
        Raises:
            FileNotFoundError: If the file does not exist or is not a file.
                               ファイルが存在しないか、ファイルでない場合。
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found or is not a file: {path}")
        return self._load_single_file(path)

    def load_from_directory(
        self,
        directory_path: Union[str, Path],
        glob_pattern: str = "**/*", # デフォルトはサブディレクトリを含む全ファイル
        recursive: bool = True,
    ) -> List[LangchainDocument]:
        """
        Load documents from a directory, optionally recursively.
        ディレクトリからドキュメントをロードします（オプションで再帰的に）。

        Args:
            directory_path (Union[str, Path]): Path to the directory. / ディレクトリへのパス。
            glob_pattern (str): Glob pattern to match files within the directory.
                                Supports '**/' for recursive matching if recursive=True.
                                Defaults to "**/*" (all files recursively).
                                ディレクトリ内のファイルに一致するglobパターン。
                                recursive=Trueの場合、再帰マッチングのために'**/'をサポートします。
                                デフォルトは "**/*" （再帰的にすべてのファイル）。
            recursive (bool): Whether to search directories recursively.
                              If False, glob_pattern should not contain '**'.
                              Defaults to True.
                              ディレクトリを再帰的に検索するかどうか。
                              Falseの場合、glob_patternは'**'を含むべきではありません。
                              デフォルトはTrue。

        Returns:
            List[LangchainDocument]: A list of loaded LangChain Document objects.
                                    ロードされたLangChain Documentオブジェクトのリスト。
        Raises:
            FileNotFoundError: If the directory does not exist or is not a directory.
                               ディレクトリが存在しないか、ディレクトリでない場合。
            ValueError: If recursive=False and glob_pattern contains '**'.
                        recursive=Falseでglob_patternが'**'を含む場合。
        """
        dir_path = Path(directory_path)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Directory not found or is not a directory: {dir_path}")

        if not recursive and "**" in glob_pattern:
             raise ValueError("Cannot use '**' in glob_pattern when recursive is False.")

        if recursive and not glob_pattern.startswith("**"):
             # Ensure recursive glob starts correctly if recursive is True
             # and user provided something like "*.md"
             if "/" not in glob_pattern and "\\" not in glob_pattern:
                 glob_pattern = f"**/{glob_pattern}"

        documents: List[LangchainDocument] = []
        file_iterator = dir_path.rglob(glob_pattern) if recursive else dir_path.glob(glob_pattern)

        for file_path in file_iterator:
            if file_path.is_file():
                loaded_doc = self._load_single_file(file_path)
                if loaded_doc:
                    documents.append(loaded_doc)

        return documents

    # --- 古いメソッド (load_json, load_markdown) は削除 ---
    # 必要であれば、load_file や load_from_directory を使って再実装するか、
    # 専用のローダーを別途用意する。
    
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
        return self.load_file(file_path)[0].page_content 