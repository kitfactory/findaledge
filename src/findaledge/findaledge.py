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
from typing import Dict, List, Optional, Union, Any, Literal

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
from .document_store.chroma import ChromaDocumentStore # Moved import to top level

# === Defaults (exported) ===
DEFAULT_EMBEDDING_MODEL_NAME = "text-embedding-3-small"  # Embedding model default\埋め込みモデルのデフォルト
DEFAULT_VECTOR_SUBDIR = "vector_store"  # Vector store subdir default\ベクトルストアのサブディレクトリのデフォルト
DEFAULT_BM25_SUBDIR = "bm25_store"  # BM25 store subdir default\BM25ストアのサブディレクトリのデフォルト
DEFAULT_LOADER_ENCODING = "utf-8"  # Loader encoding default\ローダーのエンコーディングのデフォルト
DEFAULT_SPLITTER_CHUNK_SIZE = 1000  # Splitter chunk size default\分割チャンクサイズのデフォルト
DEFAULT_SPLITTER_CHUNK_OVERLAP = 200  # Splitter chunk overlap default\分割チャンクオーバーラップのデフォルト
DEFAULT_FINDER_K = 10  # Finder top_k default\Finderのtop_kデフォルト
DEFAULT_FINDER_FUSION_METHOD = "rrf"  # Finder fusion method default\Finderの融合手法デフォルト
DEFAULT_SEARCH_MODE = "hybrid"  # Search mode default\検索モードのデフォルト

__all__ = [
    "DEFAULT_EMBEDDING_MODEL_NAME",
    "DEFAULT_VECTOR_SUBDIR",
    "DEFAULT_BM25_SUBDIR",
    "DEFAULT_LOADER_ENCODING",
    "DEFAULT_SPLITTER_CHUNK_SIZE",
    "DEFAULT_SPLITTER_CHUNK_OVERLAP",
    "DEFAULT_FINDER_K",
    "DEFAULT_FINDER_FUSION_METHOD",
    "DEFAULT_SEARCH_MODE",
]

class FindaLedge:
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
        persist_directory: Optional[str] = None,
        vector_subdir_to_use: Optional[str] = None,
        bm25_subdir_to_use: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        embedding_model_kwargs: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[str] = None,
        loader_encoding: Optional[str] = None,
        splitter_chunk_size: int = 1000,
        splitter_chunk_overlap: int = 200,
        bm25_params: Optional[Dict[str, Any]] = None,
        finder_k: int = 10,
        finder_fusion_method: str = 'rrf',
        finder_fusion_kwargs: Optional[Dict[str, Any]] = None,
        # --- 依存性注入用クラス引数 ---
        embedding_factory_cls=EmbeddingModelFactory,
        document_loader_cls=DocumentLoader,
        splitter_cls=DocumentSplitter,
        vector_store_cls=ChromaDocumentStore,
        bm25_store_cls=BM25sStore,
        finder_cls=Finder
    ) -> None:
        """
        Initializes the FindaLedge instance.

        Args:
            persist_directory (Optional[str]): Base directory for storing persistent data (vector store, BM25 index). Defaults to environment variable 'FINDALEDGE_PERSIST_DIR' or '.finderledge_data'.
            vector_subdir_to_use (Optional[str]): Subdirectory within persist_directory for the vector store. Defaults to environment variable 'FINDALEDGE_VECTOR_SUBDIR' or 'vector_store'.
            bm25_subdir_to_use (Optional[str]): Subdirectory within persist_directory for the BM25 index. Defaults to environment variable 'FINDALEDGE_BM25_SUBDIR' or 'bm25_store'.
            embedding_model_name (Optional[str]): Name of the embedding model to use (e.g., 'text-embedding-3-small'). Defaults to environment variable 'FINDALEDGE_EMBEDDING_MODEL_NAME' or 'text-embedding-3-small'.
            embedding_model_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments for the embedding model (e.g., API keys, specific model parameters). Note: API keys should ideally be set via environment variables.
            cache_dir (Optional[str]): Directory for caching embedding models or data. Defaults to environment variable 'FINDALEDGE_CACHE_DIR' or None.
            loader_encoding (Optional[str]): Default encoding to use when loading documents. Defaults to environment variable 'FINDALEDGE_LOADER_ENCODING' or 'utf-8'.
            splitter_chunk_size (int): Chunk size for the document splitter. Defaults to 1000.
            splitter_chunk_overlap (int): Chunk overlap for the document splitter. Defaults to 200.
            bm25_params (Optional[Dict[str, Any]]): Parameters for the BM25sStore (e.g., {'k1': 1.6, 'b': 0.75}).
            finder_k (int): Default number of results to retrieve in searches. Defaults to 10.
            finder_fusion_method (str): Fusion method for hybrid search ('rrf', 'simple'). Defaults to 'rrf'.
            finder_fusion_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments for the fusion method (e.g., {'rank_constant': 60} for rrf).
            embedding_factory_cls (EmbeddingModelFactory): The class to use for creating embedding models.
            document_loader_cls (DocumentLoader): The class to use for loading documents.
            splitter_cls (DocumentSplitter): The class to use for splitting documents.
            vector_store_cls (ChromaDocumentStore): The class to use for the vector store.
            bm25_store_cls (BM25sStore): The class to use for the BM25 store.
            finder_cls (Finder): The class to use for the Finder.
        """
        print("[DEBUG] Initializing FindaLedge...")

        # Resolve configuration parameters (priority: args > env > defaults)
        resolved_persist_dir = persist_directory or os.getenv('FINDALEDGE_PERSIST_DIR', DEFAULT_PERSIST_DIR)
        self.persist_dir = Path(resolved_persist_dir).resolve()
        print(f"[DEBUG] Persist directory: {self.persist_dir}")

        vector_subdir = vector_subdir_to_use or os.getenv('FINDALEDGE_VECTOR_SUBDIR', DEFAULT_VECTOR_SUBDIR)
        bm25_subdir = bm25_subdir_to_use or os.getenv('FINDALEDGE_BM25_SUBDIR', DEFAULT_BM25_SUBDIR)

        self.vector_subdir = vector_subdir
        self.bm25_subdir = bm25_subdir

        # Ensure persist directory exists
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Resolve other parameters using os.getenv
        resolved_embedding_model_name = embedding_model_name or os.getenv('FINDALEDGE_EMBEDDING_MODEL_NAME', DEFAULT_EMBEDDING_MODEL_NAME)
        self.embedding_model_name = resolved_embedding_model_name
        resolved_cache_dir = cache_dir or os.getenv('FINDALEDGE_CACHE_DIR') # Cache dir can be None
        resolved_loader_encoding = loader_encoding or os.getenv('FINDALEDGE_LOADER_ENCODING', DEFAULT_LOADER_ENCODING)
        self.default_search_mode = os.getenv('FINDALEDGE_DEFAULT_SEARCH_MODE', DEFAULT_SEARCH_MODE).lower()
        self.default_k = finder_k
        # Store fusion params for later use in search
        self.default_fusion_method = finder_fusion_method
        resolved_fusion_kwargs = {}
        if self.default_fusion_method == 'rrf':
            resolved_fusion_kwargs = {'rank_constant': 60}
        if finder_fusion_kwargs: # Allow overriding defaults
            resolved_fusion_kwargs.update(finder_fusion_kwargs)
        self.default_fusion_kwargs = resolved_fusion_kwargs

        print(f"[DEBUG] Vector subdir: {vector_subdir}")
        print(f"[DEBUG] BM25 subdir: {bm25_subdir}")
        print(f"[DEBUG] Embedding model: {resolved_embedding_model_name}")
        print(f"[DEBUG] Cache directory: {resolved_cache_dir}")
        print(f"[DEBUG] Loader encoding: {resolved_loader_encoding}")
        print(f"[DEBUG] Default search mode: {self.default_search_mode}")

        # --- Initialize Components ---

        # 1. Embedding Model Factory and Model
        self.embedding_factory = embedding_factory_cls()
        self.embedding_model = self.embedding_factory.create_embeddings(
            model_name=resolved_embedding_model_name,
            cache_dir=resolved_cache_dir,
            model_kwargs=embedding_model_kwargs
        )
        print(f"[DEBUG] Embedding model created: {type(self.embedding_model)}")

        # 2. Document Loader and Splitter
        self.document_loader = document_loader_cls()
        self.splitter = splitter_cls(
            chunk_size=splitter_chunk_size,
            chunk_overlap=splitter_chunk_overlap,
            embedding_model=self.embedding_model
        )
        print("[DEBUG] DocumentLoader and DocumentSplitter initialized.")

        # 3. Document Stores
        vector_store_path = str(self.persist_dir / vector_subdir)
        bm25_store_path = str(self.persist_dir / bm25_subdir)
        bm25_index_file = Path(bm25_store_path) / "bm25s_index.pkl"

        # Ensure store directories exist
        Path(vector_store_path).mkdir(parents=True, exist_ok=True)
        Path(bm25_store_path).mkdir(parents=True, exist_ok=True)

        # Initialize Vector Store (using Chroma by default)
        try:
            self.vector_store = vector_store_cls(
                persist_directory=vector_store_path,
                embedding_function=self.embedding_model
            )
            print(f"[DEBUG] Initialized {getattr(vector_store_cls, '__name__', str(vector_store_cls))} at {vector_store_path}")
        except ImportError:
            print("[ERROR] ChromaDocumentStore not available. Please install `chromadb`.")
            raise ImportError("Failed to import or instantiate ChromaDocumentStore. Ensure `chromadb` is installed.")

        # Initialize BM25 Store
        resolved_bm25_params = {"k1": 1.6, "b": 0.75}
        if bm25_params:
            resolved_bm25_params.update(bm25_params)

        self.bm25_store = bm25_store_cls(
            index_path=str(bm25_index_file),
            **resolved_bm25_params
        )
        print(f"[DEBUG] Initialized {getattr(bm25_store_cls, '__name__', str(bm25_store_cls))} at {bm25_index_file}")

        # 4. Retrievers
        self.vector_retriever: BaseRetriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.default_k}
        )
        self.bm25_retriever: BaseRetriever = self.bm25_store.as_retriever(
            search_kwargs={"k": self.default_k}
        )
        print("[DEBUG] Vector and BM25 retrievers created.")

        # 5. Finder (for hybrid search) - Initialize with k
        self.finder = finder_cls(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            k=self.default_k,
            fusion_method=self.default_fusion_method,
            fusion_kwargs=self.default_fusion_kwargs
        )
        print(f"[DEBUG] Finder initialized.")

        # --- document_stores属性を追加 ---
        self.document_stores = [self.vector_store, self.bm25_store]

        print("[DEBUG] FindaLedge initialization complete.")

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
             split_docs = self.splitter.split_documents([doc])
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
        # Determine the search mode, top_k, and filter based on args, env vars, and defaults
        # 引数、環境変数、デフォルト値に基づいて検索モード、top_k、フィルターを決定
        final_search_mode = search_mode or os.getenv("FINDERLEDGE_DEFAULT_SEARCH_MODE") or self.default_search_mode
        final_top_k = top_k or self.default_k
        final_filter = filter # Use the provided filter directly, default is None

        print(f"Performing search with mode: '{final_search_mode}', query: '{query}', top_k={final_top_k}, filter={final_filter}")

        results: List[SearchResult] = []
        search_kwargs = {"k": final_top_k, "filter": final_filter}

        if final_search_mode == "hybrid":
            # Finder.search には余計なキーワード引数を渡さない
            results = self.finder.search(
                query=query,
                top_k=final_top_k,
                filter=final_filter
            )
        elif final_search_mode == "vector":
            # Use only the vector retriever
            try:
                # Attempt to pass k and filter directly if retriever supports it via invoke or get_relevant_documents
                # Note: filter support depends heavily on the underlying vector store (e.g., Chroma)
                vector_docs = self.vector_retriever.get_relevant_documents(query, k=final_top_k, filter=final_filter)
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
                     for rank, doc in enumerate(vector_docs[:final_top_k])
                 ]
            except Exception as e:
                print(f"Error during vector search: {e}")
                results = []
        elif final_search_mode == "keyword":
            # Use only the BM25s retriever
            try:
                # BM25sRetriever retrieves docs with bm25_score in metadata
                bm25_docs = self.bm25_retriever.get_relevant_documents(query, k=final_top_k, filter=final_filter)
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
                results = results[:final_top_k] # Apply top_k limit
            except Exception as e:
                print(f"Error during keyword search: {e}")
                results = []
        else:
            raise ValueError(f"Invalid search_mode: '{final_search_mode}'. Must be 'hybrid', 'vector', or 'keyword'.")

        print(f"Search returned {len(results)} results for mode '{final_search_mode}'.")
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
            return "No context found for the query."

        context_parts = []
        for result in search_results:
            source = result.document.metadata.get("source", "Unknown Source")
            content = result.document.page_content
            context_parts.append(f"Source: {source}\n{content}")

        return "\n\n---\n\n".join(context_parts)

    # Remove _persist_state and _load_state as stores handle their own persistence.
    # Remove or comment out get_langchain_retriever as FinderLedge now *uses* retrievers.
    # def get_langchain_retriever(self) -> Any:
    #     ... 

    # --- Store Management Methods ---

    def add_documents(
        self,
        documents: List[LangchainDocument],
        ids: Optional[List[str]] = None,
        batch_size: int = 100
    ) -> List[str]:
        """
        Adds pre-loaded and pre-split documents directly to all managed stores.
        事前にロードおよび分割されたドキュメントを、管理対象のすべてのストアに直接追加します。

        Args:
            documents (List[LangchainDocument]): The list of documents to add.
                                                  追加するドキュメントのリスト。
            ids (Optional[List[str]]): Optional list of IDs for the documents.
                                        If provided, must match the length of documents.
                                        ドキュメントのオプションのIDリスト。
                                        指定する場合、ドキュメントの長さと一致する必要があります。
            batch_size (int): The number of documents to add in each batch to the stores.
                              ストアへの各バッチで追加するドキュメント数。

        Returns:
            List[str]: A list of IDs for the added documents (generated if not provided).
                       追加されたドキュメントのIDのリスト（指定されない場合は生成されます）。

        Raises:
            ValueError: If ids are provided and their length doesn't match documents.
                        idsが指定され、その長さがドキュメントと一致しない場合。
        """
        if ids and len(ids) != len(documents):
            raise ValueError("Number of ids must match number of documents")

        added_ids_combined: List[str] = []
        first_store = True # Flag to get IDs from the first store only

        print(f"Adding {len(documents)} documents to stores...")
        for store in self.document_stores:
            store_name = store.__class__.__name__
            print(f"Adding {len(documents)} documents to {store_name}...")
            try:
                # Add in batches
                added_ids_store: List[str] = []
                for i in range(0, len(documents), batch_size):
                    batch_docs = documents[i:i + batch_size]
                    batch_ids = ids[i:i + batch_size] if ids else None
                    print(f" Adding batch {i // batch_size + 1} ({len(batch_docs)} docs) to {store_name}")
                    current_batch_ids = store.add_documents(batch_docs, ids=batch_ids)
                    added_ids_store.extend(current_batch_ids)

                print(f" Added {len(added_ids_store)} IDs to {store_name}.")
                if first_store:
                    added_ids_combined = added_ids_store
                    first_store = False
            except Exception as e:
                print(f"Error adding documents to {store_name}: {e}")
                # Decide if we should continue with other stores or raise
                # raise e # Option 1: Stop processing
                continue # Option 2: Continue with next store

        return added_ids_combined

    def remove_documents(self, ids: List[str]) -> None:
        """
        Remove documents by their IDs from all managed stores.
        管理対象のすべてのストアから、IDによってドキュメントを削除します。

        Args:
            ids (List[str]): List of document IDs to remove.
                             削除するドキュメントIDのリスト。
        """
        print(f"Removing {len(ids)} documents from stores...")
        for store in self.document_stores:
            store_name = store.__class__.__name__
            try:
                print(f" Removing {len(ids)} documents from {store_name}...")
                store.remove_documents(ids)
                print(f" Successfully removed documents from {store_name}.")
            except NotImplementedError:
                print(f" {store_name} does not support document removal. Skipping.")
            except Exception as e:
                print(f"Error removing documents from {store_name}: {e}")
                # Decide whether to continue or raise
                continue

    def add_documents(
        self,
        documents: List[LangchainDocument],
        ids: Optional[List[str]] = None,
        batch_size: int = 100
    ) -> List[str]:
        """
        Adds pre-loaded and pre-split documents directly to all managed stores.
        事前にロードおよび分割されたドキュメントを、管理対象のすべてのストアに直接追加します。

        Args:
            documents (List[LangchainDocument]): The list of documents to add.
                                                  追加するドキュメントのリスト。
            ids (Optional[List[str]]): Optional list of IDs for the documents.
                                        If provided, must match the length of documents.
                                        ドキュメントのオプションのIDリスト。
                                        指定する場合、ドキュメントの長さと一致する必要があります。
            batch_size (int): The number of documents to add in each batch to the stores.
                              ストアへの各バッチで追加するドキュメント数。

        Returns:
            List[str]: A list of IDs for the added documents (generated if not provided).
                       追加されたドキュメントのIDのリスト（指定されない場合は生成されます）。

        Raises:
            ValueError: If ids are provided and their length doesn't match documents.
                        idsが指定され、その長さがドキュメントと一致しない場合。
        """
        if ids and len(ids) != len(documents):
            raise ValueError("Number of ids must match number of documents")

        added_ids_combined: List[str] = []
        first_store = True # Flag to get IDs from the first store only

        print(f"Adding {len(documents)} documents to stores...")
        for store in self.document_stores:
            store_name = store.__class__.__name__
            print(f"Adding {len(documents)} documents to {store_name}...")
            try:
                # Add in batches
                added_ids_store: List[str] = []
                for i in range(0, len(documents), batch_size):
                    batch_docs = documents[i:i + batch_size]
                    batch_ids = ids[i:i + batch_size] if ids else None
                    print(f" Adding batch {i // batch_size + 1} ({len(batch_docs)} docs) to {store_name}")
                    current_batch_ids = store.add_documents(batch_docs, ids=batch_ids)
                    added_ids_store.extend(current_batch_ids)

                print(f" Added {len(added_ids_store)} IDs to {store_name}.")
                if first_store:
                    added_ids_combined = added_ids_store
                    first_store = False
            except Exception as e:
                print(f"Error adding documents to {store_name}: {e}")
                # Decide if we should continue with other stores or raise
                # raise e # Option 1: Stop processing
                continue # Option 2: Continue with next store

        return added_ids_combined

    def remove_documents(self, ids: List[str]) -> None:
        """
        Remove documents by their IDs from all managed stores.
        管理対象のすべてのストアから、IDによってドキュメントを削除します。

        Args:
            ids (List[str]): List of document IDs to remove.
                             削除するドキュメントIDのリスト。
        """
        print(f"Removing {len(ids)} documents from stores...")
        for store in self.document_stores:
            store_name = store.__class__.__name__
            try:
                print(f" Removing {len(ids)} documents from {store_name}...")
                store.remove_documents(ids)
                print(f" Successfully removed documents from {store_name}.")
            except NotImplementedError:
                print(f" {store_name} does not support document removal. Skipping.")
            except Exception as e:
                print(f"Error removing documents from {store_name}: {e}")
                # Decide whether to continue or raise
                continue 