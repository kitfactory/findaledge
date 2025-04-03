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

from .text_splitter import TextSplitter
# from .document import Document # <-- 削除


class DocumentLoader:
    """
    Uses LangChain document loaders to load documents from various formats.
    LangChainのドキュメントローダーを使用して、様々な形式からドキュメントをロードします。

    This class acts as a wrapper around various LangChain loaders,
    providing a unified interface for loading files and directories.
    このクラスは、様々なLangChainローダーのラッパーとして機能し、
    ファイルやディレクトリをロードするための統一されたインターフェースを提供します。
    """

    def __init__(self):
        # TextSplitterは不要になる可能性があるため、初期化を削除
        # If splitting is needed after loading, it should be handled separately.
        # 必要であれば、ロード後の分割は別途処理する必要があります。
        pass

    def load_file(self, file_path: Union[str, Path], encoding: str = "utf-8", **loader_kwargs: Any) -> List[LangchainDocument]:
        """
        Load a single file using the appropriate LangChain loader.
        適切なLangChainローダーを使用して単一ファイルをロードします。

        Args:
            file_path (Union[str, Path]): Path to the file. / ファイルへのパス。
            encoding (str): File encoding. Defaults to "utf-8". / ファイルエンコーディング。デフォルトは"utf-8"。
            **loader_kwargs: Additional arguments passed to the specific LangChain loader.
                             特定のLangChainローダーに渡される追加の引数。

        Returns:
            List[LangchainDocument]: A list of LangChain Document objects.
                                    LangChainのDocumentオブジェクトのリスト。

        Raises:
            ValueError: If the file type is not supported.
                        ファイルタイプがサポートされていない場合。
            FileNotFoundError: If the file does not exist.
                               ファイルが存在しない場合。
        """
        file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found or is not a file: {file_path}")

        suffix = file_path.suffix.lower()

        if suffix == ".txt":
            loader = TextLoader(str(file_path), encoding=encoding, **loader_kwargs)
        elif suffix == ".pdf":
            loader = PyPDFLoader(str(file_path), **loader_kwargs)
        elif suffix == ".md":
            loader = UnstructuredMarkdownLoader(str(file_path), mode="elements", **loader_kwargs)
        # elif suffix == ".json":
            # from langchain_community.document_loaders import JSONLoader # 必要に応じてインポート
            # # JSONLoaderには特定の構造が必要な場合があるため、jq_schemaやjson_linesなどを指定
            # # 例: jq_schema='.[]' など
            # loader = JSONLoader(str(file_path), jq_schema='.', **loader_kwargs)
        # 他のファイル形式（CSV, HTMLなど）のサポートを追加可能
        else:
            # デフォルトとしてTextLoaderを試みるか、エラーを発生させる
            try:
                loader = TextLoader(str(file_path), encoding=encoding, **loader_kwargs)
                print(f"Warning: Unsupported file type '{suffix}'. Attempting to load as text.")
            except Exception as e:
                 raise ValueError(f"Unsupported file type: {suffix}. Error: {e}")

        return loader.load()

    # load_directory, load_text, load_markdown, load_document, load_documents, load_json は
    # load_file を使用するように変更するか、LangChainのDirectoryLoaderなどを使用するように書き換える必要があります。
    # 現状のコードは独自実装とLangChainが混在しているため、一旦コメントアウトまたは削除を推奨します。

    # def load_directory(...) -> List[LangchainDocument]: # シグネチャ変更
    #     ...

    # def load_text(...) -> List[LangchainDocument]: # シグネチャ変更
    #     doc = Document(page_content=..., metadata={...}) # LangchainDocumentを使う
    #     ...

    # ... 他のメソッドも同様に修正 ...
    
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