"""
FinderLedge - Document context management library using multiple retrievers and RRF reranking.
FinderLedge - 複数のリトリーバーとRRFリランキングを使用する文書コンテキスト管理ライブラリ。

This module provides a high-level interface for managing document contexts,
leveraging vector stores and keyword search (BM25s) internally,
and reranking results using the Finder class.
このモジュールは、文書コンテキストを管理するための高レベルインターフェースを提供し、
内部的にベクトルストアとキーワード検索（BM25s）を活用し、
Finderクラスを使用して結果をリランキングします。
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

# LangChain document standard
from langchain_core.documents import Document as LangchainDocument
from langchain_core.retrievers import BaseRetriever

# Project components
from .finder import Finder, SearchResult # The new RRF reranker
from .document_store.document_store import BaseDocumentStore
from .document_store.vector_document_store import VectorDocumentStore
from .document_store.bm25s import BM25sStore, BM25sRetriever # Import retriever as well
from .embeddings_factory import EmbeddingModelFactory
from .document_loader import DocumentLoader
from .document_splitter import DocumentSplitter, DocumentType
from .tokenizer import Tokenizer # Needed for BM25s


class FinderLedge:
    """
    Document context management system using multiple retrievers and RRF.
    複数のリトリーバーとRRFを使用する文書コンテキスト管理システム。

    Manages document loading, splitting, indexing into multiple stores (vector and keyword),
    and provides a unified search interface via RRF reranking.
    文書のロード、分割、複数のストア（ベクトルおよびキーワード）へのインデックス作成を管理し、
RRFリランキングを介して統一された検索インターフェースを提供します。
    """

    def __init__(
        self,
        *,
        persist_dir: str = "data",
        embedding_model_name: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        vector_store_subdir: Optional[str] = None,
        bm25_index_subdir: Optional[str] = None,
        bm25_params: Optional[Dict[str, Any]] = None,
        rrf_k: int = 60,
        **kwargs: Any
    ):
        """
        Initialize the FinderLedge system.
        FinderLedgeシステムを初期化します。

        Args:
            persist_dir (str): Base directory for persisting all data.
                               すべてのデータを永続化するベースディレクトリ。
            embedding_model_name (Optional[str]): Name of the embedding model via EmbeddingModelFactory.
                                                 If None, uses FINDERLEDGE_EMBEDDING_MODEL_NAME env var,
                                                 or defaults to "text-embedding-3-small".
                                        EmbeddingModelFactory経由で使用する埋め込みモデルの名前。
                                        Noneの場合、環境変数 FINDERLEDGE_EMBEDDING_MODEL_NAME を使用するか、
                                        デフォルトで "text-embedding-3-small" になります。
            chunk_size (int): Target size of text chunks for splitting, similar to LangChain's TextSplitter.
                              テキスト分割時の目標チャンクサイズ（LangChainのTextSplitterと同様）。
            chunk_overlap (int): Overlap between text chunks, similar to LangChain's TextSplitter.
                                 テキストチャンク間の重複（LangChainのTextSplitterと同様）。
            vector_store_subdir (Optional[str]): Subdirectory for the vector store.
                                                 If None, uses FINDERLEDGE_CHROMA_SUBDIR env var,
                                                 or defaults to "chroma_db".
                                       ベクトルストア用のサブディレクトリ。
                                       Noneの場合、環境変数 FINDERLEDGE_CHROMA_SUBDIR を使用するか、
                                       デフォルトで "chroma_db" になります。
            bm25_index_subdir (Optional[str]): Subdirectory for the BM25s index.
                                               If None, uses FINDERLEDGE_BM25S_SUBDIR env var,
                                               or defaults to "bm25s_index".
                                     BM25sインデックス用のサブディレクトリ。
                                     Noneの場合、環境変数 FINDERLEDGE_BM25S_SUBDIR を使用するか、
                                     デフォルトで "bm25s_index" になります。
            bm25_params (Optional[Dict[str, Any]]): Parameters for BM25s initialization.
                                                    BM25s初期化用のパラメータ。
            rrf_k (int): Ranking constant for RRF calculation in Finder.
                         FinderでのRRF計算のためのランキング定数。
            **kwargs (Any): Additional keyword arguments for future extensions or component configurations.
                            将来の拡張やコンポーネント設定のための追加キーワード引数。
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Determine embedding model name to use
        model_name_to_use = embedding_model_name or os.getenv("FINDERLEDGE_EMBEDDING_MODEL_NAME", "text-embedding-3-small")

        # Determine subdirectory names using environment variables as defaults
        vector_subdir_to_use = vector_store_subdir or os.getenv("FINDERLEDGE_CHROMA_SUBDIR", "chroma_db")
        bm25_subdir_to_use = bm25_index_subdir or os.getenv("FINDERLEDGE_BM25S_SUBDIR", "bm25s_index")

        # Initialize components
        # self.embedding_factory = EmbeddingModelFactory() # No need to instantiate
        # Call the static method directly
        self.embedding_model = EmbeddingModelFactory.create_embeddings(model_name=model_name_to_use)
        self.document_splitter = DocumentSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # model_name=embedding_model_name # Pass model name if splitter needs it
        )
        self.document_loader = DocumentLoader() # Loader might not need chunk params directly

        # Define paths for stores
        vector_store_path = str(self.persist_dir / vector_subdir_to_use)
        bm25_index_path = str(self.persist_dir / bm25_subdir_to_use / "bm25s_index.pkl")

        # Initialize Document Stores
        self.vector_store = VectorDocumentStore(
            persist_directory=vector_store_path,
            embedding_function=self.embedding_model
        )
        self.bm25_store = BM25sStore(
            index_path=bm25_index_path,
            **(bm25_params or {}) # Pass BM25 params
        )

        # List of managed document stores
        self.document_stores: List[BaseDocumentStore] = [self.vector_store, self.bm25_store]

        # Create Retrievers and store them individually
        # Configure k for initial retrieval. This k might be overridden by search method's top_k if retriever supports it.
        self.vector_retriever: BaseRetriever = self.vector_store.as_retriever(search_kwargs={'k': 50, 'filter': None}) # Base filter capability
        self.bm25_retriever: BaseRetriever = self.bm25_store.as_retriever(k=50)

        # Initialize the Finder (RRF Reranker) for hybrid search
        self.rrf_finder = Finder( # Rename finder instance
            retrievers=[self.vector_retriever, self.bm25_retriever],
            rrf_k=rrf_k
        )

        print(f"FinderLedge initialized. Data directory: {self.persist_dir}")
        print(f" Stores: VectorStore in '{vector_subdir_to_use}', BM25sStore in '{bm25_subdir_to_use}'")
        print(f" Embedding model: {model_name_to_use}")

    def _load_documents_from_path(self, path: Path) -> List[LangchainDocument]:
        """Internal helper to load documents from a verified existing path."""
        loaded_docs: List[LangchainDocument] = []
        if path.is_dir():
            print(f"Loading from directory: {path}")
            loaded_docs = self.document_loader.load_from_directory(path)
        elif path.is_file():
            print(f"Loading from file: {path}")
            loaded_doc = self.document_loader.load_file(path)
            if loaded_doc:
                loaded_docs = [loaded_doc]
        else:
            print(f"Warning: Path {path} exists but is neither a file nor a directory recognized for loading. Skipping.")
        return loaded_docs

    def add_document(
        self,
        content_or_path: Union[str, Path],
        doc_type: Optional[DocumentType] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Load, split, and add a document (or documents from a directory) to all managed stores.
        文書（またはディレクトリ内の文書）をロード、分割し、管理対象のすべてのストアに追加します。

        Args:
            content_or_path (Union[str, Path]): Document content as a string, path to a single file, or path to a directory.
                                                  文字列としての文書内容、単一ファイルへのパス、またはディレクトリへのパス。
            doc_type (Optional[DocumentType]): The type of the document(s) if loading from string/path.
                                                文字列/パスからロードする場合の文書タイプ。
                                                If loading a directory, type is inferred from extension.
                                                ディレクトリをロードする場合、タイプは拡張子から推測されます。
            metadata (Optional[Dict[str, Any]]): Optional base metadata to add to all loaded documents/chunks.
                                                   ロードされたすべての文書/チャンクに追加するオプションの基本メタデータ。

        Returns:
            List[str]: List of document IDs added to the stores.
                       ストアに追加された文書IDのリスト。
        """
        print(f"Adding document(s) from: {content_or_path}")
        loaded_docs: List[LangchainDocument] = []
        path_to_load: Optional[Path] = None # Path object to potentially load from

        # --- Determine Input Type and Validate Path --- 
        if isinstance(content_or_path, Path):
            if content_or_path.exists():
                path_to_load = content_or_path
            else:
                 print(f"Warning: Provided Path object does not exist: {content_or_path}. Skipping.")

        elif isinstance(content_or_path, str):
            try:
                potential_path = Path(content_or_path)
                if potential_path.exists():
                    path_to_load = potential_path
                else:
                    # String is not an existing path, treat as raw content
                    print("Input string does not exist as a path, treating as raw content.")
                    if not doc_type:
                         doc_type = DocumentType.TEXT
                    base_meta = {"source": "raw_string"}
                    if metadata:
                         base_meta.update(metadata)
                    loaded_docs = [LangchainDocument(page_content=content_or_path, metadata=base_meta)]
            except OSError:
                 # String is not a valid path, treat as raw content
                 print("Input string is not a valid path, treating as raw content.")
                 if not doc_type:
                      doc_type = DocumentType.TEXT
                 base_meta = {"source": "raw_string"}
                 if metadata:
                      base_meta.update(metadata)
                 loaded_docs = [LangchainDocument(page_content=content_or_path, metadata=base_meta)]
        else:
             print(f"Warning: Unsupported input type for content_or_path: {type(content_or_path)}. Skipping.")

        # --- Load from Path if applicable --- 
        if path_to_load and not loaded_docs:
            loaded_docs = self._load_documents_from_path(path_to_load)

        # --- Process loaded/generated documents --- 
        if not loaded_docs:
             print("No documents were loaded or generated from input.")
             return []

        # Add base metadata if provided (and not already added for raw string)
        if metadata and path_to_load: # Only add if loaded from path, raw content handled above
            for doc in loaded_docs:
                existing_meta = doc.metadata.copy()
                existing_meta.update(metadata)
                doc.metadata = existing_meta

        # Split documents
        all_split_docs: List[LangchainDocument] = []
        for doc in loaded_docs:
             split_docs = self.document_splitter.split_documents([doc])
             all_split_docs.extend(split_docs)

        if not all_split_docs:
             print("No documents generated after splitting.")
             return []

        # Add to all managed stores
        added_ids_combined = set()
        for store in self.document_stores:
            print(f"Adding {len(all_split_docs)} split documents to {store.__class__.__name__}...")
            try:
                added_ids = store.add_documents(all_split_docs)
                if added_ids:
                     added_ids_combined.update(added_ids)
                print(f" Added {len(added_ids)} IDs to {store.__class__.__name__}.")
            except Exception as e:
                print(f"Error adding documents to {store.__class__.__name__}: {e}")

        print(f"Document addition process complete. Added IDs: {list(added_ids_combined)}")
        return list(added_ids_combined)

    def remove_document(self, doc_id: str) -> None:
        """
        Remove a document (and its chunks) from all managed stores.
        管理対象のすべてのストアから文書（およびそのチャンク）を削除します。

        Note: Assumes doc_id corresponds to the original document ID used during addition.
              The stores need logic to find and remove associated chunks/entries.
        注意: doc_idは追加時に使用された元の文書IDに対応すると仮定します。
              ストアは関連するチャンク/エントリを見つけて削除するロジックが必要です。

        Args:
            doc_id (str): ID of the original document to remove.
                          削除する元の文書のID。
        """
        print(f"Removing document with ID: {doc_id} from all stores...")
        for store in self.document_stores:
            try:
                print(f" Deleting from {store.__class__.__name__}...")
                store.delete_document(doc_id)
                print(f" Deletion attempt complete for {store.__class__.__name__}.")
            except NotImplementedError:
                 print(f" Warning: delete_document not implemented for {store.__class__.__name__}. Skipping.")
            except Exception as e:
                print(f"Error deleting document {doc_id} from {store.__class__.__name__}: {e}")
        print(f"Document removal process complete for ID: {doc_id}")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        search_mode: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search for documents using the specified mode (hybrid, vector, keyword).
        指定されたモード（ハイブリッド、ベクトル、キーワード）を使用して文書を検索します。

        Args:
            query (str): The search query. / 検索クエリ。
            top_k (int): The final number of documents to return.
                         返す最終的なドキュメント数。
            filter (Optional[Dict[str, Any]]): Metadata filter for the search.
                                                 検索用のメタデータフィルター。
            search_mode (Optional[str]): The search strategy: "hybrid", "vector", or "keyword".
                                         If None, uses the value from the FINDERLEDGE_DEFAULT_SEARCH_MODE
                                         environment variable, or defaults to "hybrid".
                               検索戦略: "hybrid"、"vector"、または "keyword"。
                               Noneの場合、環境変数 FINDERLEDGE_DEFAULT_SEARCH_MODE の値を使用するか、
                               デフォルトで "hybrid" になります。

        Returns:
            List[SearchResult]: A list of search results, including documents and scores.
                                文書とスコアを含む、検索結果のリスト。

        Raises:
            ValueError: If an invalid search_mode is provided.
        """
        # Determine the search mode to use
        mode_to_use = search_mode or os.getenv("FINDERLEDGE_DEFAULT_SEARCH_MODE", "hybrid")

        print(f"Performing search with mode: '{mode_to_use}', query: '{query}', top_k={top_k}, filter={filter}")

        results: List[SearchResult] = []
        search_kwargs = {"k": top_k, "filter": filter}

        if mode_to_use == "hybrid":
            results = self.rrf_finder.search(query=query, top_k=top_k, filter=filter)
        elif mode_to_use == "vector":
            # Use only the vector retriever
            try:
                # Attempt to pass k and filter directly if retriever supports it via invoke or get_relevant_documents
                # Note: filter support depends heavily on the underlying vector store (e.g., Chroma)
                vector_docs = self.vector_retriever.get_relevant_documents(query, k=top_k, filter=filter)
                # Map to SearchResult, use rank as score proxy
                results = [
                    SearchResult(document=doc, score=1.0/(rank + 1))
                    for rank, doc in enumerate(vector_docs)
                ]
            except TypeError:
                 # Fallback if k/filter cannot be passed directly to get_relevant_documents
                 print(f"Warning: Vector retriever might not support direct k/filter passing. Retrieving with default settings and applying limits/filters post-retrieval might be needed for full support.")
                 vector_docs = self.vector_retriever.get_relevant_documents(query)
                 # Simple post-retrieval limit (filtering would need manual implementation here)
                 results = [
                     SearchResult(document=doc, score=1.0/(rank + 1))
                     for rank, doc in enumerate(vector_docs[:top_k])
                 ]
            except Exception as e:
                print(f"Error during vector search: {e}")
                results = []
        elif mode_to_use == "keyword":
            # Use only the BM25s retriever
            try:
                # BM25sRetriever retrieves docs with bm25_score in metadata
                bm25_docs = self.bm25_retriever.get_relevant_documents(query, k=top_k, filter=filter)
                 # Map to SearchResult, extracting score
                results = [
                    SearchResult(document=doc, score=doc.metadata.get('bm25_score', 0.0))
                    for doc in bm25_docs
                ]
                 # Ensure results are sorted by score (BM25s retriever should return sorted, but double-check)
                results.sort(key=lambda x: x.score, reverse=True)
            except TypeError:
                print(f"Warning: BM25s retriever might not support direct k/filter passing. Retrieving with default settings and applying limits/filters post-retrieval might be needed for full support.")
                bm25_docs = self.bm25_retriever.get_relevant_documents(query)
                results = [
                    SearchResult(document=doc, score=doc.metadata.get('bm25_score', 0.0))
                    for doc in bm25_docs
                ]
                results.sort(key=lambda x: x.score, reverse=True)
                results = results[:top_k] # Apply top_k limit
            except Exception as e:
                print(f"Error during keyword search: {e}")
                results = []
        else:
            raise ValueError(f"Invalid search_mode: '{mode_to_use}'. Must be 'hybrid', 'vector', or 'keyword'.")

        print(f"Search returned {len(results)} results for mode '{mode_to_use}'.")
        return results

    def get_context(
        self,
        query: str,
        top_k: int = 5,
        search_mode: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Get combined context string from the top search results for a query.
        クエリに対する上位の検索結果から結合されたコンテキスト文字列を取得します。

        Args:
            query (str): Query to get context for / コンテキストを取得するクエリ
            top_k (int): Number of top documents to include in the context / コンテキストに含める上位文書の数
            search_mode (Optional[str]): The search mode to use ("hybrid", "vector", "keyword").
                                         If None, uses the value from the FINDERLEDGE_DEFAULT_SEARCH_MODE
                                         environment variable, or defaults to "hybrid".
                               使用する検索モード。Noneの場合、環境変数 FINDERLEDGE_DEFAULT_SEARCH_MODE
                               の値を使用するか、デフォルトで "hybrid" になります。
            filter (Optional[Dict[str, Any]]): Optional filter for the search.
                                                検索用のオプションフィルター。

        Returns:
            str: Combined context string from the page content of the top documents.
                 上位文書のページ内容から結合されたコンテキスト文字列。
        """
        # Determine the search mode to use
        mode_to_use = search_mode or os.getenv("FINDERLEDGE_DEFAULT_SEARCH_MODE", "hybrid")

        print(f"Getting context for query: '{query}', top_k={top_k}, mode='{mode_to_use}', filter={filter}")
        # Use the updated search method, passing the determined mode
        search_results = self.search(query=query, top_k=top_k, filter=filter, search_mode=mode_to_use)

        if not search_results:
            return ""

        # Combine page content from the results
        context = "\n\n---\n\n".join(res.document.page_content for res in search_results)
        return context

    # Remove _persist_state and _load_state as stores handle their own persistence.
    # Remove or comment out get_langchain_retriever as FinderLedge now *uses* retrievers.
    # def get_langchain_retriever(self) -> Any:
    #     ... 