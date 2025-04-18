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
import logging # Import logging

from .text_splitter import TextSplitter
# from .document import Document # <-- 削除

# markitdown のクラス名をインポート
from markitdown import MarkItDown

logger = logging.getLogger(__name__) # Initialize logger

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
        Loads a single file, handling code files first, then markitdown.
        単一のファイルをロードします。まずコードファイルを処理し、次にmarkitdownを使用します。

        Args:
            file_path (Path): The path to the file.
                              ファイルへのパス。

        Returns:
            Optional[LangchainDocument]: The loaded document, or None if loading fails or the file type is unsupported.
                                         ロードされたドキュメント。ロードに失敗した場合やファイルタイプがサポートされていない場合はNone。
        """
        try:
            file_suffix = file_path.suffix.lower()
            file_name = file_path.name # For extensionless files like Dockerfile
            logger.debug(f"Attempting to load file: {file_path} with suffix: '{file_suffix}', name: '{file_name}'")

            # 1. Check for Code Files first (including common extensionless names)
            if file_suffix in self.CODE_EXTENSIONS or file_name in self.CODE_EXTENSIONS:
                try:
                    logger.debug(f"Loading code file as text: {file_path}")
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    logger.debug(f"Successfully loaded code file as text: {file_path}")
                    return LangchainDocument(page_content=content, metadata={"source": str(file_path)})
                except UnicodeDecodeError:
                    # Ensure this block *only* logs and returns None
                    logger.warning(f"UTF-8 decoding failed for code file {file_path}. Skipping.")
                    print(f"[WARN] UTF-8 decoding failed for code file {file_path}. Skipping.")
                    return None # Explicitly return None here
                except Exception as e:
                    logger.error(f"Failed to read code file {file_path} as text: {e}", exc_info=True)
                    print(f"[ERROR] Failed to read code file {file_path} as text: {e}")
                    return None

            # 2. If not a code file, check if supported by Markitdown
            elif file_suffix in self.SUPPORTED_EXTENSIONS:
                try:
                    # Try reading with UTF-8 first to catch explicit decode errors
                    # before passing to markitdown, especially for text-based formats like .txt
                    # Markitdown might handle some errors internally, but this gives us explicit control.
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            f.read() # Attempt to read the whole file
                    except UnicodeDecodeError:
                         logger.warning(f"UTF-8 decoding failed for potential markitdown file {file_path}. Skipping.")
                         print(f"[WARN] UTF-8 decoding failed for file {file_path}. Skipping.")
                         return None # Skip if direct UTF-8 read fails

                    logger.debug(f"Using markitdown loader for {file_path}")
                    # Use convert() method
                    result = self.md_converter.convert(str(file_path))
                    if result is None or not hasattr(result, 'text_content'):
                        logger.warning(f"Markitdown returned None or invalid result for: {file_path}")
                        return None
                    content = result.text_content
                    metadata = result.metadata if hasattr(result, 'metadata') and result.metadata else {}
                    metadata["source"] = str(file_path)
                    logger.debug(f"Successfully loaded with markitdown: {file_path}")
                    return LangchainDocument(page_content=content, metadata=metadata)
                except Exception as e:
                    logger.error(f"Markitdown failed to load {file_path}: {e}", exc_info=True)
                    print(f"[ERROR] Markitdown failed to load {file_path}: {e}")
                    return None

            # 3. If neither code nor markitdown supported
            else:
                logger.warning(f"Skipping unsupported file type (not code and not markitdown): {file_path}")
                print(f"Skipping unsupported file type: {file_path}")
                return None

        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Generic failure loading document {file_path}: {e}", exc_info=True)
            print(f"[ERROR] Failed to load document {file_path}: {e}")
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