"""
FAISS Document Store Module
FAISSドキュメントストアモジュール

This module provides a FAISS-based implementation of the vector document store.
このモジュールはFAISSベースのベクトルドキュメントストアの実装を提供します。
"""

from pathlib import Path
from typing import List, Optional, Sequence
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from .vector_document_store import VectorDocumentStore

class FAISSDocumentStore(VectorDocumentStore):
    """
    FAISS-based document store implementation
    FAISS ベースのドキュメントストア実装
    """

    def __init__(
        self,
        embedding_function,
        persist_directory: Optional[str] = None,
        allow_dangerous_deserialization: bool = False
    ):
        """
        Initialize FAISS document store
        FAISS ドキュメントストアを初期化する

        Args:
            embedding_function: Function to generate embeddings
                              埋め込みを生成する関数
            persist_directory: Directory to persist index, optional
                             インデックスを永続化するディレクトリ（オプション）
            allow_dangerous_deserialization: Allow loading pickled data, optional
                                          シリアライズされたデータの読み込みを許可（オプション）
        """
        super().__init__(embedding_function)
        self.faiss_store = None
        self.persist_directory = persist_directory
        self.allow_dangerous_deserialization = allow_dangerous_deserialization
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        """
        Ensure FAISS storage is initialized
        FAISSストレージが初期化されていることを確認する
        """
        if self.faiss_store is None:
            if self.persist_directory and Path(self.persist_directory).exists():
                try:
                    self.faiss_store = FAISS.load_local(
                        self.persist_directory,
                        self.embedding_function,
                        allow_dangerous_deserialization=self.allow_dangerous_deserialization
                    )
                except Exception:
                    # Create new store with dummy data
                    # ダミーデータで新しいストアを作成
                    dummy_text = "initialization"
                    dummy_embedding = self.embedding_function.embed_query(dummy_text)
                    self.faiss_store = FAISS.from_embeddings(
                        text_embeddings=[(dummy_text, dummy_embedding)],
                        embedding=self.embedding_function,
                        metadatas=[{"id": "dummy"}]
                    )
            else:
                # Create new store with dummy data
                # ダミーデータで新しいストアを作成
                dummy_text = "initialization"
                dummy_embedding = self.embedding_function.embed_query(dummy_text)
                self.faiss_store = FAISS.from_embeddings(
                    text_embeddings=[(dummy_text, dummy_embedding)],
                    embedding=self.embedding_function,
                    metadatas=[{"id": "dummy"}]
                )

    def _add_documents(self, documents: Sequence[Document], ids: Optional[Sequence[str]] = None) -> List[str]:
        """
        Add documents to the store
        ドキュメントをストアに追加する

        Args:
            documents: Documents to add
                     追加するドキュメント
            ids: Optional document IDs
                オプションのドキュメントID

        Returns:
            List of document IDs
            ドキュメントIDのリスト
        """
        if not documents:
            return []

        # Use provided IDs or generate IDs if not present in metadata
        # 提供されたIDを使用するか、メタデータにIDがない場合は生成
        doc_ids = list(ids) if ids else [doc.metadata.get("id") for doc in documents]
        # Ensure all documents have IDs
        for i, doc in enumerate(documents):
            if not doc_ids[i]:
                # Simple UUID generation, consider more robust approach if needed
                import uuid
                doc_ids[i] = str(uuid.uuid4())
                doc.metadata["id"] = doc_ids[i] # Assign generated ID to metadata

        texts = [doc.page_content for doc in documents] # <-- content を page_content に変更
        # Include parent_id in metadata if available
        metadatas = []
        for doc_id, doc in zip(doc_ids, documents):
            meta = doc.metadata.copy() # Start with existing metadata
            meta["id"] = doc_id # Ensure ID is set
            # parent_id might already be in metadata from splitting
            if "parent_id" not in meta:
                 meta["parent_id"] = meta.get("id") # Default parent_id to its own id if not split
            metadatas.append(meta)

        embeddings = self.embedding_function.embed_documents(texts)

        self.faiss_store.add_embeddings(
            text_embeddings=list(zip(texts, embeddings)),
            metadatas=metadatas
        )

        if self.persist_directory:
            self.faiss_store.save_local(self.persist_directory)

        return doc_ids

    def _get_document(self, doc_id: str) -> Optional[Document]:
        """
        Get document by ID
        IDでドキュメントを取得する

        Args:
            doc_id: Document ID
                   ドキュメントID

        Returns:
            Document if found, None otherwise
            ドキュメントが見つかった場合はDocument、見つからない場合はNone
        """
        # FAISS docstore uses internal integer IDs, we search by metadata
        # FAISS docstore は内部整数IDを使用するため、メタデータで検索
        for internal_id, doc in self.faiss_store.docstore._dict.items():
             if doc.metadata.get("id") == doc_id:
                 return doc
        return None

    def _get_split_documents(self, parent_id: str) -> List[Document]:
        """
        Get all split documents for a parent document
        親ドキュメントの全ての分割ドキュメントを取得する

        Args:
            parent_id: Parent document ID
                      親ドキュメントID

        Returns:
            List of split documents
            分割ドキュメントのリスト
        """
        splits = []
        # Iterate through the FAISS docstore to find documents with matching parent_id
        for internal_id, doc in self.faiss_store.docstore._dict.items():
             # Check if metadata exists and contains 'parent_id'
             if isinstance(doc, Document) and doc.metadata and doc.metadata.get("parent_id") == parent_id:
                 # Ensure it's not the parent document itself
                 if doc.metadata.get("id") != parent_id:
                      splits.append(doc)
        return splits

    def delete_document(self, doc_id: str) -> None:
        """
        Delete document by ID (including its splits)
        IDでドキュメント（およびその分割）を削除する

        Args:
            doc_id: Document ID (parent ID)
                   ドキュメントID（親ID）
        """
        ids_to_delete = []
        # Find internal FAISS ids for the parent and its splits
        for internal_id, doc in self.faiss_store.docstore._dict.items():
             if doc.metadata.get("id") == doc_id or doc.metadata.get("parent_id") == doc_id:
                 # FAISS expects the string representation of the internal ID for deletion
                 ids_to_delete.append(self.faiss_store.index_to_docstore_id[internal_id])

        if ids_to_delete:
            self.faiss_store.delete(ids_to_delete)
            if self.persist_directory:
                self.faiss_store.save_local(self.persist_directory)

    def as_retriever(self, **kwargs):
        """
        Get retriever interface
        検索インターフェースを取得する

        Returns:
            Retriever interface
            検索インターフェース
        """
        return self.faiss_store.as_retriever(**kwargs)

    def list_documents(self) -> List[Document]:
        """
        List all parent documents in the store
        ストア内のすべての親ドキュメントをリストアップする

        Returns:
            List[Document]: List of parent documents
                親ドキュメントのリスト
        """
        parent_docs = {}
        for internal_id, doc in self.faiss_store.docstore._dict.items():
             # Consider documents as parents if they don't have a different parent_id
             # or if explicitly marked (adjust logic as needed)
             parent_id = doc.metadata.get("parent_id")
             doc_id = doc.metadata.get("id")
             if doc_id and (parent_id is None or parent_id == doc_id):
                # Use doc_id as key to avoid duplicates if splits are also listed
                parent_docs[doc_id] = doc
        return list(parent_docs.values())

    def update_document(self, id_: str, document: Document) -> None:
        """
        Update a document in the store (deletes old and adds new)
        ストア内のドキュメントを更新する (古いのを削除し、新しいのを追加)

        Args:
            id_ (str): Document ID to update
                更新するドキュメントID
            document (Document): Updated document data
                更新されたドキュメントデータ
        """
        # Ensure the new document has the correct ID in its metadata
        document.metadata["id"] = id_
        self.delete_document(id_)
        self._add_documents([document]) 