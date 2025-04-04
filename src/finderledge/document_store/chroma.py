"""
Vector Document Store Module using ChromaDB
ChromaDBを使用したベクトルドキュメントストアモジュール
"""

from typing import List, Optional, Sequence, Any, Dict
from uuid import uuid4
from langchain.schema import Document, BaseRetriever
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from .vector_document_store import VectorDocumentStore

class ChromaDocumentStore(VectorDocumentStore):
    """
    ChromaDB-based document store implementation
    ChromaDBベースのドキュメントストア実装

    This class provides a vector store implementation using ChromaDB as the backend.
    このクラスはChromaDBをバックエンドとして使用するベクトルストアの実装を提供します。
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        embedding_function: Optional[Embeddings] = None,
        collection_name: str = "documents",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs: Any
    ) -> None:
        """
        Initialize ChromaDB document store
        ChromaDBドキュメントストアを初期化

        Args:
            persist_directory (Optional[str]): Directory to persist ChromaDB data
                ChromaDBデータを永続化するディレクトリ
            embedding_function (Optional[Embeddings]): LangChain embeddings interface
                LangChainのembeddingsインターフェース
            collection_name (str): Name of the ChromaDB collection. Defaults to "documents"
                ChromaDBコレクションの名前。デフォルトは"documents"
            chunk_size (int): Size of text chunks for splitting. Defaults to 1000
                テキスト分割のチャンクサイズ。デフォルトは1000
            chunk_overlap (int): Overlap between chunks. Defaults to 200
                チャンク間のオーバーラップ。デフォルトは200
            **kwargs (Any): Additional arguments passed to parent class
                親クラスに渡される追加の引数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        super().__init__(embedding_function=embedding_function, chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self._ensure_storage()

    def _ensure_storage(self, **kwargs: Any) -> None:
        """
        Initialize ChromaDB storage
        ChromaDBストレージを初期化

        Args:
            **kwargs (Any): Additional arguments for ChromaDB initialization
                ChromaDB初期化のための追加の引数
        """
        self.chroma_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_function,
            collection_name=self.collection_name
        )

    def _add_documents(self, documents: Sequence[Document], ids: Optional[Sequence[str]] = None) -> List[str]:
        """
        Add documents to ChromaDB store
        ChromaDBストアにドキュメントを追加
        (Implementation detail called by VectorDocumentStore.add_documents)
        (VectorDocumentStore.add_documentsによって呼び出される実装詳細)

        Args:
            documents (Sequence[Document]): Documents to add (potentially already split)
                追加するドキュメント（既に分割されている可能性あり）
            ids (Optional[Sequence[str]], optional): Custom IDs for documents. Should be provided by the caller.
                ドキュメントのカスタムID。呼び出し元から提供されるべき。

        Returns:
            List[str]: List of document IDs added/updated.
                追加/更新されたドキュメントIDのリスト。
        """
        if not ids:
             # Assuming VectorDocumentStore ensures documents have IDs in metadata
             # If not, this implementation might need adjustment or error handling
            ids = [doc.metadata["id"] for doc in documents]

        # Call Chroma's add_documents using keyword arguments
        # Chroma handles upsert based on IDs
        self.chroma_store.add_documents(documents=documents, ids=ids)

        # Return the list of IDs that were processed
        return ids

    def _get_document(self, doc_id: str, is_parent: Optional[bool] = None) -> Optional[Document]:
        """
        Get document from ChromaDB store by ID, optionally filtering by parent status.
        IDでChromaDBストアから文書を取得し、オプションで親ステータスでフィルタリングします。

        Args:
            doc_id (str): Document ID.
                文書ID。
            is_parent (Optional[bool]): If True, only retrieve if it's a parent document.
                Trueの場合、親ドキュメントの場合のみ取得します。

        Returns:
            Optional[Document]: Document if found, None otherwise.
                見つかった場合は文書、それ以外はNone。
        """
        chroma_filter: Dict[str, Any] = {"id": doc_id}
        if is_parent is True:
            # Combine filters using $and for ChromaDB compatibility
            chroma_filter = {"$and": [{"id": doc_id}, {"is_parent": True}]}
        # If is_parent is False or None, the initial filter {"id": doc_id} is used

        # Use similarity_search with a filter for reliable retrieval
        # Using get() with complex filters can be less straightforward
        results_ss = self.chroma_store.similarity_search(
            query="*", # Use a dummy query for metadata filtering
            k=1,       # We only expect one document with a unique ID
            filter=chroma_filter
        )
        return results_ss[0] if results_ss else None

    def _get_split_documents(self, parent_id: str) -> List[Document]:
        """
        Get split documents from ChromaDB store by parent ID
        親IDでChromaDBストアから分割文書を取得

        Args:
            parent_id (str): Parent document ID.
                親文書ID。

        Returns:
            List[Document]: List of split documents.
                分割文書のリスト。
        """
        # Combine filters using $and for ChromaDB
        chroma_filter = {"$and": [
            {"parent_id": parent_id},
            {"is_split": True}
        ]}
        
        # Consider using get() with where filter if performance is critical
        # results = self.chroma_store.get(where=chroma_filter, include=["metadatas", "documents"]) 
        # Reconstruct documents from results['metadatas'], results['documents']
        
        # Using similarity_search for simplicity now
        results = self.chroma_store.similarity_search(
            query="*", # Dummy query for metadata filtering
            k=1000,  # Increase k substantially to ensure all splits are retrieved
            filter=chroma_filter
        )
        # Sort by split_index if metadata is present
        try:
            results.sort(key=lambda doc: doc.metadata.get('split_index', 0))
        except Exception:
            pass # Ignore sorting errors if metadata is missing
        return results

    def delete_document(self, doc_id: str) -> None:
        """
        Delete document from ChromaDB store
        ChromaDBストアから文書を削除

        Args:
            doc_id (str): Document ID.
                文書ID。
        """
        # Get the document to check if it's a parent
        doc = self.get_document(doc_id)
        if not doc:
            return

        # If it's a parent, delete all splits
        if doc.metadata.get('is_parent'):
            splits = self.get_split_documents(doc_id)
            split_ids = [split.metadata.get('id') for split in splits]
            self.chroma_store.delete(ids=split_ids)

        # Delete the document itself
        self.chroma_store.delete(ids=[doc_id])

    def list_documents(self) -> List[Document]:
        """
        List all parent documents in the store
        ストア内の全親文書をリスト

        Returns:
            List[Document]: List of parent documents.
                親文書のリスト。
        """
        results = self.chroma_store.similarity_search(
            query="",
            k=1000,  # Assuming no more than 1000 parent documents
            filter={"is_parent": True}
        )
        return results

    def update_document(self, doc_id: str, document: Document) -> None:
        """
        Update document in ChromaDB store
        ChromaDBストアの文書を更新

        Args:
            doc_id (str): Document ID.
                文書ID。
            document (Document): Updated document.
                更新された文書。
        """
        # Delete existing document and its splits
        self.delete_document(doc_id)
        
        # Add the updated document
        self.add_documents([document], [doc_id])

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        """
        Get retriever interface
        リトリーバーインターフェースを取得

        Returns:
            BaseRetriever: Retriever interface for similarity search.
                類似度検索用のリトリーバーインターフェース。
        """
        return self.chroma_store.as_retriever(**kwargs)

# 使用例 / Usage examples:
"""
# 初期化（embedding_functionは必須）
store = ChromaDocumentStore(
    persist_directory="./data/chroma",
    embedding_function=OpenAIEmbeddings(),
    collection_name="my_documents",
    chunk_size=1000,
    chunk_overlap=200
)

# ドキュメントの追加（自動的に分割される）
docs = [
    Document(page_content="長いテキスト文書1", metadata={"source": "test1"}),
    Document(page_content="長いテキスト文書2", metadata={"source": "test2"})
]
parent_ids = store.add_documents(docs)

# 親文書の取得
parent_doc = store.get_parent_document(parent_ids[0])

# 分割文書の取得
split_docs = store.get_split_documents(parent_ids[0])

# retrieverの取得と使用（分割文書から検索される）
retriever = store.as_retriever(search_kwargs={"k": 2})
results = retriever.get_relevant_documents("クエリ")
""" 