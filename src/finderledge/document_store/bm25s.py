# src/finderledge/document_store/bm25s.py
import os
import pickle
from typing import List, Dict, Any, Optional, Sequence, Tuple
import uuid

import bm25s # 日本語対応版 bm25s-j をインポート
from langchain.schema import Document, BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun # Async版を追加
from langchain_core.retrievers import BaseRetriever
# from ..env import FINDERLEDGE_BM25_INDEX_PATH # 環境変数定義を env.py に追加する必要あり

from .document_store import BaseDocumentStore

class BM25sRetriever(BaseRetriever):
    """
    Custom LangChain retriever using BM25sStore.
    BM25sStoreを使用するカスタムLangChain retriever。
    """
    store: 'BM25sStore'
    k: int = 4 # Default number of documents to retrieve

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to a query."""
        try:
            # Delegate to the store's search method
            results = self.store.search(query, k=self.k)
            # No explicit on_retriever_end in sync run_manager from common examples
            return results
        except Exception as e:
            # No explicit on_retriever_error in sync run_manager from common examples
            # Re-raise the error for proper handling upstream
            raise e

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to a query asynchronously."""
        try:
            # Delegate to the store's search method (synchronous)
            results = self.store.search(query, k=self.k)

            # Notify retrieval end via run_manager
            await run_manager.on_retriever_end(
                results,
            )
            return results
        except Exception as e:
            # Notify retrieval error via run_manager
            await run_manager.on_retriever_error(
                e,
            )
            # Re-raise the error for proper handling upstream
            raise e

class BM25sStore(BaseDocumentStore):
    """
    Document store using the bm25s-j library for BM25 search.
    BM25検索に bm25s-j ライブラリを使用するドキュメントストア。

    This store operates in-memory primarily but can persist the index to disk.
    このストアは主にインメモリで動作しますが、インデックスをディスクに永続化できます。
    """

    def __init__(self, index_path: Optional[str] = None, **kwargs: Any) -> None:
        """
        Initialize BM25sStore.
        BM25sStoreを初期化します。

        Args:
            index_path (Optional[str], optional): Path to load/save the BM25s index.
                If None, uses FINDERLEDGE_BM25_INDEX_PATH env var or defaults to "./bm25s_index.pkl".
                BM25sインデックスをロード/セーブするパス。Noneの場合、環境変数
                FINDERLEDGE_BM25_INDEX_PATH を使うか、デフォルト "./bm25s_index.pkl" になります。
            **kwargs (Any): Additional parameters for bm25s.BM25 (e.g., k1, b).
                bm25s.BM25の追加パラメータ（例: k1, b）。
        """
        # Load environment variables if not already loaded
        # oneenv.load() # Application should load this at startup
        self.index_path = index_path or os.getenv("FINDERLEDGE_BM25_INDEX_PATH", "./bm25s_index.pkl")
        self.bm25_params = kwargs
        self.retriever = bm25s.BM25(**self.bm25_params)
        self._documents: Dict[str, Document] = {} # Store original documents by ID
        self._corpus_map: Dict[int, str] = {} # Map internal bm25s index to our doc ID
        self._doc_id_to_internal: Dict[str, int] = {} # Map our doc ID to internal bm25s index
        self._is_indexed = False

        self._ensure_storage()
        self._load_index()

    def _ensure_storage(self, **kwargs: Any) -> None:
        """Ensure the directory for the index file exists."""
        index_dir = os.path.dirname(self.index_path)
        if index_dir and not os.path.exists(index_dir):
            os.makedirs(index_dir)

    def _load_index(self) -> None:
        """Load index and document map from disk if they exist."""
        index_file = self.index_path
        doc_map_file = self.index_path + ".docs.pkl"

        if os.path.exists(index_file) and os.path.exists(doc_map_file):
            try:
                # bm25s doesn't have a direct load method in 0.1.1, use pickle for now
                # Note: This might change in future bm25s versions.
                # Consider using bm25s.hf.BM25HF if saving/loading from Hub is preferred.
                with open(index_file, 'rb') as f:
                    self.retriever = pickle.load(f) # Load the entire retriever object
                with open(doc_map_file, 'rb') as f:
                    saved_data = pickle.load(f)
                    self._documents = saved_data.get('documents', {})
                    self._corpus_map = saved_data.get('corpus_map', {})
                    self._doc_id_to_internal = saved_data.get('doc_id_to_internal', {})
                self._is_indexed = True
                print(f"BM25s index and document map loaded from {self.index_path}")
            except Exception as e:
                print(f"Warning: Failed to load BM25s index/map from {self.index_path}. Starting fresh. Error: {e}")
                self._reset_store() # Start fresh if loading fails
        else:
            print("No existing BM25s index found. Starting fresh.")
            self._reset_store()

    def _save_index(self) -> None:
        """Save the current index and document map to disk."""
        if not self._is_indexed and not self._documents:
             print("Nothing to save. Index is not built or store is empty.")
             return

        index_file = self.index_path
        doc_map_file = self.index_path + ".docs.pkl"
        try:
            # bm25s doesn't have a direct save method in 0.1.1, use pickle for now
            with open(index_file, 'wb') as f:
                pickle.dump(self.retriever, f) # Save the entire retriever object
            with open(doc_map_file, 'wb') as f:
                 save_data = {
                    'documents': self._documents,
                    'corpus_map': self._corpus_map,
                    'doc_id_to_internal': self._doc_id_to_internal
                }
                 pickle.dump(save_data, f)
            print(f"BM25s index and document map saved to {self.index_path}")
        except Exception as e:
            print(f"Error saving BM25s index/map to {self.index_path}: {e}")

    def _reset_store(self) -> None:
        """Resets the store to an empty state."""
        self.retriever = bm25s.BM25(**self.bm25_params)
        self._documents = {}
        self._corpus_map = {}
        self._doc_id_to_internal = {}
        self._is_indexed = False


    def add_documents(self, documents: Sequence[Document], ids: Optional[Sequence[str]] = None) -> List[str]:
        """
        Add documents and re-index the corpus.
        文書を追加し、コーパスを再インデックスします。

        Note: Currently, this re-indexes the entire corpus every time.
              For large datasets, consider batching additions or a more sophisticated update strategy.
        注意: 現在、これは毎回コーパス全体を再インデックスします。
              大規模なデータセットの場合、追加のバッチ処理やより洗練された更新戦略を検討してください。
        """
        if ids and len(documents) != len(ids):
            raise ValueError("Number of documents and IDs must match.")

        added_ids: List[str] = []
        new_docs_for_index: List[str] = []
        new_corpus_map: Dict[int, str] = {}
        new_doc_id_to_internal: Dict[str, int] = {}

        # Add new documents to the internal dictionary
        for i, doc in enumerate(documents):
            doc_id = ids[i] if ids else str(uuid.uuid4())
            if doc_id in self._documents:
                 print(f"Warning: Document ID {doc_id} already exists. Overwriting.")
                 # Need to handle removal from old index maps if overwriting, complex with current bm25s state
            doc.metadata['id'] = doc_id # Ensure ID is in metadata
            self._documents[doc_id] = doc
            added_ids.append(doc_id)

        # Prepare corpus for re-indexing
        corpus_texts = [d.page_content for d in self._documents.values()]
        doc_ids_list = list(self._documents.keys())

        if not corpus_texts:
            print("No documents to index.")
            self._is_indexed = False
            self._save_index() # Save empty state if needed
            return added_ids

        # Tokenize and index
        try:
            corpus_tokens = bm25s.tokenize(corpus_texts)
            self.retriever = bm25s.BM25(**self.bm25_params) # Re-initialize retriever
            self.retriever.index(corpus_tokens)
            self._is_indexed = True
            print(f"Successfully re-indexed {len(corpus_texts)} documents.")

            # Rebuild maps after successful indexing
            self._corpus_map = {i: doc_id for i, doc_id in enumerate(doc_ids_list)}
            self._doc_id_to_internal = {doc_id: i for i, doc_id in enumerate(doc_ids_list)}

            self._save_index() # Persist the new index and maps
        except Exception as e:
            print(f"Error during BM25s indexing: {e}")
            # Optionally revert _documents state if indexing fails completely
            self._is_indexed = False # Mark as not indexed if error occurs
            # Decide whether to keep added documents in self._documents or remove them

        return added_ids

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID."""
        return self._documents.get(doc_id)

    def update_document(self, doc_id: str, document: Document) -> None:
        """
        Update a document and re-index.
        文書を更新し、再インデックスします。

        Note: Re-indexes the entire corpus. Consider efficiency implications.
        注意: コーパス全体を再インデックスします。効率への影響を考慮してください。
        """
        if doc_id not in self._documents:
            raise ValueError(f"Document with ID {doc_id} not found.")
        document.metadata['id'] = doc_id # Ensure ID is correct
        self.add_documents([document], ids=[doc_id]) # Use add_documents for re-indexing logic

    def delete_document(self, doc_id: str) -> None:
        """
        Delete a document and re-index.
        文書を削除し、再インデックスします。

        Note: Re-indexes the entire corpus. Consider efficiency implications.
        注意: コーパス全体を再インデックスします。効率への影響を考慮してください。
        """
        if doc_id in self._documents:
            del self._documents[doc_id]
            # Re-index the remaining documents
            print(f"Document {doc_id} deleted. Re-indexing remaining documents.")
            remaining_docs = list(self._documents.values())
            if not remaining_docs:
                 self._reset_store()
                 self._save_index()
            else:
                 # This effectively calls add_documents with the remaining docs
                 self.add_documents(remaining_docs, ids=list(self._documents.keys()))
        else:
            print(f"Warning: Document ID {doc_id} not found for deletion.")


    def list_documents(self) -> List[str]:
        """List all document IDs currently in the store."""
        return list(self._documents.keys())

    def search(self, query: str, k: int = 4) -> List[Document]:
        """
        Search for documents relevant to the query using BM25s.
        BM25sを使用してクエリに関連する文書を検索します。

        Args:
            query (str): The search query. / 検索クエリ。
            k (int): Number of documents to retrieve. / 取得する文書数。

        Returns:
            List[Document]: List of relevant documents found.
                           関連する文書のリスト。
        """
        if not self._is_indexed:
            print("Warning: BM25 index is not built. Search results will be empty.")
            return []
        if not self._documents:
            return []

        try:
            query_tokens = bm25s.tokenize(query)
            # retrieve returns doc indices and scores
            results_indices, scores = self.retriever.retrieve(query_tokens, k=k)

            relevant_docs: List[Document] = []
            if results_indices is not None and len(results_indices) > 0:
                 # results_indices is likely shape (1, k) for single query
                 relevant_internal_indices = results_indices[0]
                 for i, internal_idx in enumerate(relevant_internal_indices):
                      doc_id = self._corpus_map.get(int(internal_idx))
                      if doc_id:
                           doc = self.get_document(doc_id)
                           if doc:
                               # Add score to metadata
                               if scores is not None and len(scores) > 0 and i < len(scores[0]):
                                   doc.metadata['bm25_score'] = scores[0][i]
                               relevant_docs.append(doc)
            return relevant_docs
        except Exception as e:
            print(f"Error during BM25s search: {e}")
            return []

    def as_retriever(self, k: int = 4, **kwargs: Any) -> BaseRetriever:
        """
        Get a LangChain retriever interface for this BM25s store.
        このBM25sストアのLangChain retrieverインターフェースを取得します。

        Args:
            k (int): Default number of documents to retrieve. デフォルトで取得する文書数。
            **kwargs (Any): Additional parameters (ignored in this basic implementation).
                           追加パラメータ（この基本実装では無視されます）。

        Returns:
            BaseRetriever: A LangChain retriever interface.
                           LangChain retrieverインターフェース。
        """
        return BM25sRetriever(store=self, k=k) 