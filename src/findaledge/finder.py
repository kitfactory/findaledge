"""
Finder - A class for searching documents using embeddings and BM25
Finder - 埋め込みとBM25を使用して文書を検索するためのクラス

This class provides functionality for searching documents using a combination of embeddings and BM25.
このクラスは、埋め込みとBM25を組み合わせて文書を検索する機能を提供します。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Tuple, Any
import numpy as np
import os
import json
from langchain.schema import Document as LangchainDocument
from .document_store.document_store import BaseDocumentStore
from .tokenizer import Tokenizer
import asyncio # For potential async retriever calls
from langchain_core.documents import Document # Updated import
from langchain_core.retrievers import BaseRetriever # Updated import

@dataclass
class SearchResult:
    """
    A class representing a search result after reranking.
    リランク後の検索結果を表すクラス。

    Attributes:
        document (Document): The matched document / マッチしたドキュメント
        score (float): The reranked score (e.g., RRF score) / リランク後のスコア（例: RRFスコア）
    """
    document: Document
    score: float

class Finder:
    """
    Reranks results from multiple LangChain Retrievers using Reciprocal Rank Fusion (RRF).
    Reciprocal Rank Fusion (RRF) を使用して、複数のLangChain Retrieverからの結果をリランクします。

    Takes a list of retrievers and reranks their combined results based on rank.
    リトリーバーのリストを受け取り、順位に基づいて結合された結果をリランクします。
    """

    def __init__(
        self,
        retrievers: List[BaseRetriever],
        rrf_k: int = 60, # RRF constant, typically 60
    ):
        """
        Initialize the Finder (Reranker).
        Finder (Reranker) を初期化します。

        Args:
            retrievers (List[BaseRetriever]): A list of LangChain BaseRetriever instances.
                LangChain BaseRetrieverインスタンスのリスト。
            rrf_k (int): The ranking constant for RRF calculation. Defaults to 60.
                         RRF計算のためのランキング定数。デフォルトは60。
        """
        if not retrievers:
            raise ValueError("At least one retriever must be provided.")
        self.retrievers = retrievers
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 10,
        # Optional: parameter to control how many results to fetch from each retriever
        k_per_retriever: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Retrieves documents from all underlying retrievers and reranks them using RRF.
        すべての基礎となるリトリーバーからドキュメントを取得し、RRFを使用してリランクします。

        Args:
            query (str): The search query. / 検索クエリ。
            top_k (int): The final number of documents to return after reranking.
                         リランク後に返す最終的なドキュメント数。
            k_per_retriever (Optional[int]): How many documents to fetch from each retriever
                                             before reranking. If None, uses top_k.
                                             リランク前に各リトリーバーから取得するドキュメント数。
                                             Noneの場合、top_kを使用します。
            filter (Optional[Dict[str, Any]]): Metadata filter to apply to the retrievers.
                                                 リトリーバーに適用するメタデータフィルター。

        Returns:
            List[SearchResult]: A list of reranked search results.
                                リランクされた検索結果のリスト。
        """
        if k_per_retriever is None:
            # Fetch more initially to allow for better reranking diversity
            k_per_retriever = max(top_k, 10) * len(self.retrievers)


        # 1. Get results from all retrievers (synchronously for now)
        all_results: List[Tuple[BaseRetriever, List[Document]]] = []
        for retriever in self.retrievers:
            try:
                # Pass k and filter via kwargs if supported
                retriever_kwargs = {'k': k_per_retriever}
                if filter:
                    retriever_kwargs['filter'] = filter

                relevant_docs = retriever.get_relevant_documents(query, **retriever_kwargs)

                # Limit results per retriever if necessary (safety measure)
                all_results.append((retriever, relevant_docs[:k_per_retriever]))
            except TypeError: # Handle retrievers that don't accept 'k' or 'filter'
                 try:
                      # Try without filter first
                      retriever_kwargs_no_filter = {'k': k_per_retriever}
                      relevant_docs = retriever.get_relevant_documents(query, **retriever_kwargs_no_filter)
                      all_results.append((retriever, relevant_docs[:k_per_retriever]))
                 except TypeError: # Try without k and filter
                     try:
                         relevant_docs = retriever.get_relevant_documents(query)
                         all_results.append((retriever, relevant_docs[:k_per_retriever]))
                     except Exception as e:
                         print(f"Error retrieving from {retriever.__class__.__name__} (no kwargs): {e}")
                         all_results.append((retriever, []))
                 except Exception as e:
                      print(f"Error retrieving from {retriever.__class__.__name__} (without filter kwarg): {e}")
                      all_results.append((retriever, []))
            except Exception as e:
                print(f"Error retrieving from {retriever.__class__.__name__}: {e}")
                all_results.append((retriever, [])) # Append empty list on error

        # 2. Calculate RRF scores
        rrf_scores: Dict[str, float] = {} # Store combined RRF score per doc ID
        doc_map: Dict[str, Document] = {} # Store document objects by ID to avoid duplicates

        for retriever, docs in all_results:
            for rank, doc in enumerate(docs):
                # --- Get unique Document ID ---
                # Attempt common metadata keys, provide default if none found
                doc_id = doc.metadata.get("id") or \
                         doc.metadata.get("doc_id") or \
                         str(hash(doc.page_content + doc.metadata.get("source", ""))) # Fallback ID based on content+source hash

                # Store the document object using the derived ID
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc # Store the document object

                # Calculate RRF score for this rank and add to total
                score = 1.0 / (self.rrf_k + rank + 1) # Rank is 0-based
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score

        # 3. Sort documents by combined RRF score
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # 4. Format results
        final_results: List[SearchResult] = []
        for doc_id in sorted_doc_ids[:top_k]:
            document = doc_map.get(doc_id)
            if document: # Should always be found if doc_id is in rrf_scores
                final_results.append(SearchResult(document=document, score=rrf_scores[doc_id]))

        return final_results

    async def asearch(
        self,
        query: str,
        top_k: int = 10,
        k_per_retriever: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Asynchronously retrieves documents from all underlying retrievers and reranks them using RRF.
        非同期ですべての基礎となるリトリーバーからドキュメントを取得し、RRFを使用してリランクします。

        Args:
            query (str): The search query. / 検索クエリ。
            top_k (int): The final number of documents to return after reranking.
                         リランク後に返す最終的なドキュメント数。
            k_per_retriever (Optional[int]): How many documents to fetch from each retriever
                                             before reranking. If None, uses top_k.
                                             リランク前に各リトリーバーから取得するドキュメント数。
                                             Noneの場合、top_kを使用します。
            filter (Optional[Dict[str, Any]]): Metadata filter to apply to the retrievers.
                                                 リトリーバーに適用するメタデータフィルター。

        Returns:
            List[SearchResult]: A list of reranked search results.
                                リランクされた検索結果のリスト。
        """
        if k_per_retriever is None:
             # Fetch more initially to allow for better reranking diversity
             k_per_retriever = max(top_k, 10) * len(self.retrievers)

        # 1. Get results from all retrievers asynchronously
        async def _get_docs(retriever: BaseRetriever) -> List[Document]:
            try:
                # Pass k and filter via kwargs if supported
                retriever_kwargs = {'k': k_per_retriever}
                if filter:
                    retriever_kwargs['filter'] = filter
                return await retriever.aget_relevant_documents(query, **retriever_kwargs)
            except TypeError: # Handle retrievers that don't accept 'k' or 'filter'
                 try:
                     # Try without filter first
                     retriever_kwargs_no_filter = {'k': k_per_retriever}
                     return await retriever.aget_relevant_documents(query, **retriever_kwargs_no_filter)
                 except TypeError: # Try without k and filter
                     try:
                         return await retriever.aget_relevant_documents(query)
                     except Exception as e:
                          print(f"Error retrieving async from {retriever.__class__.__name__} (no kwargs): {e}")
                          return []
                 except Exception as e:
                     print(f"Error retrieving async from {retriever.__class__.__name__} (without filter kwarg): {e}")
                     return []
            except Exception as e:
                print(f"Error retrieving async from {retriever.__class__.__name__}: {e}")
                return []

        tasks = [_get_docs(retriever) for retriever in self.retrievers]
        results_list: List[List[Document]] = await asyncio.gather(*tasks)

        # Combine results with their retriever (though not strictly needed for RRF logic itself)
        all_results_async: List[Tuple[BaseRetriever, List[Document]]] = list(zip(self.retrievers, results_list))
        all_results = [(r, docs[:k_per_retriever]) for r, docs in all_results_async] # Limit results


        # 2. Calculate RRF scores (same logic as sync version)
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for retriever, docs in all_results:
            for rank, doc in enumerate(docs):
                 # --- Get unique Document ID ---
                 doc_id = doc.metadata.get("id") or \
                          doc.metadata.get("doc_id") or \
                          str(hash(doc.page_content + doc.metadata.get("source", ""))) # Fallback ID

                 # Store the document object using the derived ID
                 if doc_id not in doc_map:
                     doc_map[doc_id] = doc

                 # Calculate RRF score for this rank and add to total
                 score = 1.0 / (self.rrf_k + rank + 1)
                 rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score


        # 3. Sort documents by combined RRF score
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # 4. Format results
        final_results: List[SearchResult] = []
        for doc_id in sorted_doc_ids[:top_k]:
            document = doc_map.get(doc_id)
            if document:
                final_results.append(SearchResult(document=document, score=rrf_scores[doc_id]))

        return final_results

    def to_dict(self) -> dict:
        """
        Convert the finder to a dictionary
        finderを辞書に変換

        Returns:
            dict: Dictionary representation of the finder
        """
        return {
            "retrievers": [retriever.to_dict() for retriever in self.retrievers],
            "rrf_k": self.rrf_k
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Finder":
        """
        Create a finder from a dictionary
        辞書からfinderを作成

        Args:
            data (dict): Dictionary representation of the finder

        Returns:
            Finder: New finder instance
        """
        retrievers = [BaseRetriever.from_dict(retriever_data) for retriever_data in data["retrievers"]]
        rrf_k = data["rrf_k"]

        return cls(
            retrievers=retrievers,
            rrf_k=rrf_k
        ) 