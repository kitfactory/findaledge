import pytest
import asyncio
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
# Pydantic field for default values if needed
from pydantic.v1 import Field, BaseModel # Use pydantic.v1 if BaseRetriever still uses it internally, or pydantic if updated

from findaledge.finder import Finder, SearchResult

# --- Mock Retriever Implementations ---

class MockRetriever(BaseRetriever):
    """A simple mock retriever for testing, attempting explicit super().__init__ passing."""
    # Define fields directly
    returned_docs: List[Document] = Field(...) # Mark as required
    supports_filter: bool = True
    supports_k: bool = True
    identifier: str = "mock"

    # Store last received values for verification
    last_filter_received: Optional[Dict[str, Any]] = Field(default=None, exclude=True)
    last_k_received: Optional[int] = Field(default=None, exclude=True)

    # Override __init__ to map 'docs' input to 'returned_docs' field for Pydantic validation
    def __init__(self, *, docs: List[Document], **kwargs: Any):
        """Explicitly pass returned_docs to super init."""
        # Pass 'docs' as 'returned_docs' to the Pydantic validation
        # along with any other BaseRetriever arguments.
        # Ensure kwargs doesn't overwrite returned_docs if accidentally passed.
        kwargs_filtered = {k: v for k, v in kwargs.items() if k != 'returned_docs'}
        super().__init__(returned_docs=docs, **kwargs_filtered)

    # _get_relevant_documents and _aget_relevant_documents methods use self.returned_docs
    # which should be correctly initialized by the __init__ call above.

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        k: Optional[int] = None, # BaseRetriever might not pass these directly in newer LC versions
        filter: Optional[Dict[str, Any]] = None # Arguments might need to be passed via config or specific methods
    ) -> List[Document]:
        """Mock synchronous retrieval."""
        # Note: Accessing k/filter might require different approach if not passed directly
        # For testing Finder, we assume Finder passes them correctly via **kwargs
        # We capture them here if passed via _get_relevant_documents for the mock's internal use/verification
        # but the core logic uses self.supports_k etc defined on the instance.

        actual_k = k # Use argument if passed directly (less common now)
        actual_filter = filter # Use argument if passed directly

        # --- For verification purposes, store what was received --- 
        # Use object.__setattr__ because these fields might be excluded / frozen
        object.__setattr__(self, 'last_k_received', actual_k)
        object.__setattr__(self, 'last_filter_received', actual_filter)
        # -----------------------------------------------------------

        print(f"[{self.identifier}] _get_relevant_documents called with query: '{query}', k: {actual_k}, filter: {actual_filter}")


        # Simulate filter check
        if actual_filter and self.supports_filter:
            print(f"[{self.identifier}] Applying filter: {actual_filter}")
            filtered_docs = [
                doc for doc in self.returned_docs
                if all(item in doc.metadata.items() for item in actual_filter.items())
            ]
        else:
            filtered_docs = self.returned_docs

        # Simulate k limit using self.supports_k and actual_k if provided
        limit = actual_k if actual_k is not None and self.supports_k else len(filtered_docs)
        print(f"[{self.identifier}] Returning {min(limit, len(filtered_docs))} docs (limit={limit})")
        return filtered_docs[:limit]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Mock asynchronous retrieval."""
        print(f"[{self.identifier}] _aget_relevant_documents called with query: '{query}', k: {k}, filter: {filter}")
        await asyncio.sleep(0.01) # Simulate async operation
        # For mock simplicity, call sync version's logic, passing potentially received k/filter
        # Note: A more accurate async mock might handle k/filter differently if BaseRetriever expects them via config.
        # Using sync logic directly might bypass some async specifics, but sufficient for testing Finder's RRF part.

        # --- For verification purposes, store what was received --- 
        object.__setattr__(self, 'last_k_received', k)
        object.__setattr__(self, 'last_filter_received', filter)
        # -----------------------------------------------------------

        # Re-implementing core logic from sync version for clarity
        if filter and self.supports_filter:
            filtered_docs = [
                doc for doc in self.returned_docs
                if all(item in doc.metadata.items() for item in filter.items())
            ]
        else:
            filtered_docs = self.returned_docs

        limit = k if k is not None and self.supports_k else len(filtered_docs)
        return filtered_docs[:limit]

# --- Test Fixtures ---

@pytest.fixture
def sample_docs():
    return [
        Document(page_content="This is document A", metadata={"id": "doc-a", "tag": "tech"}),
        Document(page_content="This is document B", metadata={"doc_id": "doc-b", "tag": "finance"}), # Different ID key
        Document(page_content="This is document C", metadata={"tag": "tech"}), # No standard ID
        Document(page_content="This is document D", metadata={"id": "doc-d", "tag": "general"}),
    ]

@pytest.fixture
def retriever1(sample_docs):
    # Pass 'docs' which will be mapped to 'returned_docs' in __init__
    return MockRetriever(docs=[sample_docs[0], sample_docs[1], sample_docs[2]], identifier="retriever1")

@pytest.fixture
def retriever2(sample_docs):
    # Pass 'docs' which will be mapped to 'returned_docs' in __init__
    return MockRetriever(docs=[sample_docs[1], sample_docs[3], sample_docs[0]], identifier="retriever2")

@pytest.fixture
def retriever_no_filter(sample_docs):
    # Pass 'docs' which will be mapped to 'returned_docs' in __init__
    return MockRetriever(docs=[sample_docs[0], sample_docs[3]], supports_filter=False, identifier="retriever_no_filter")


# --- Test Cases ---

def test_finder_initialization(retriever1):
    """Test Finder initialization."""
    finder = Finder(retrievers=[retriever1])
    assert len(finder.retrievers) == 1
    assert finder.rrf_k == 60

    finder_custom_k = Finder(retrievers=[retriever1], rrf_k=30)
    assert finder_custom_k.rrf_k == 30

def test_finder_initialization_empty_retrievers():
    """Test Finder initialization with no retrievers raises ValueError."""
    with pytest.raises(ValueError):
        Finder(retrievers=[])

def test_finder_search_basic_rrf(retriever1, retriever2, sample_docs):
    """Test basic search and RRF reranking."""
    finder = Finder(retrievers=[retriever1, retriever2], rrf_k=60)
    results = finder.search("test query", top_k=3)

    # Expected RRF scores (k=60):
    # Doc A: R1(rank 0) = 1/61, R2(rank 2) = 1/63. Total = 1/61 + 1/63 = ~0.0322
    # Doc B: R1(rank 1) = 1/62, R2(rank 0) = 1/61. Total = 1/62 + 1/61 = ~0.0325
    # Doc C: R1(rank 2) = 1/63. Total = ~0.0159
    # Doc D: R2(rank 1) = 1/62. Total = ~0.0161
    # Expected order: B > A > D > C (approx)

    assert len(results) == 3
    assert isinstance(results[0], SearchResult)
    assert results[0].document.metadata.get('doc_id') == "doc-b" # B should be first
    assert results[1].document.metadata.get('id') == "doc-a"     # A should be second
    assert results[2].document.metadata.get('id') == "doc-d"     # D should be third

    # Check scores are decreasing
    assert results[0].score > results[1].score > results[2].score

    # Check approximate scores (allow for floating point variations)
    assert results[0].score == pytest.approx(0.0325, rel=1e-3)
    assert results[1].score == pytest.approx(0.03227, rel=1e-3)
    assert results[2].score == pytest.approx(0.01613, rel=1e-3)


def test_finder_search_top_k(retriever1, retriever2):
    """Test the top_k parameter limits results."""
    finder = Finder(retrievers=[retriever1, retriever2])
    results = finder.search("test query", top_k=2)
    assert len(results) == 2
    # Order should still be B > A
    assert results[0].document.metadata.get('doc_id') == "doc-b"
    assert results[1].document.metadata.get('id') == "doc-a"

def test_finder_search_k_per_retriever(mocker, retriever1, retriever2):
    """Test that k_per_retriever is passed down (using mocks)."""
    spy1 = mocker.spy(retriever1, "_get_relevant_documents")
    spy2 = mocker.spy(retriever2, "_get_relevant_documents")

    finder = Finder(retrievers=[retriever1, retriever2])
    # Fetch 1 doc per retriever initially, rerank top 2
    results = finder.search("another query", top_k=2, k_per_retriever=1)

    assert len(results) <= 2 # Can be less if duplicates or fewer available

    # Check mocks were called with k=1
    spy1.assert_called_once()
    spy2.assert_called_once()
    assert spy1.call_args.kwargs.get('k') == 1
    assert spy2.call_args.kwargs.get('k') == 1

def test_finder_search_with_filter(mocker, retriever1, retriever2, retriever_no_filter, sample_docs):
    """Test search with filter passes filter to supported retrievers."""
    spy1 = mocker.spy(retriever1, "_get_relevant_documents")
    spy2 = mocker.spy(retriever2, "_get_relevant_documents")
    spy3 = mocker.spy(retriever_no_filter, "_get_relevant_documents")

    finder = Finder(retrievers=[retriever1, retriever2, retriever_no_filter])
    test_filter = {"tag": "tech"}
    results = finder.search("filter query", top_k=3, filter=test_filter)

    # Check mocks received the filter correctly
    spy1.assert_called_once()
    assert spy1.call_args.kwargs.get('filter') == test_filter

    spy2.assert_called_once()
    assert spy2.call_args.kwargs.get('filter') == test_filter

    spy3.assert_called_once()
    # retriever_no_filter might receive filter in kwargs but its internal mock logic ignores it
    # The Finder's try/except handles TypeError if the retriever *signature* doesn't accept it
    # Our mock *does* accept it in the signature but internally ignores it if supports_filter=False
    assert spy3.call_args.kwargs.get('filter') == test_filter # Filter is passed, mock ignores

    # Check results (mock filter logic is basic)
    # R1 (filter): gets A, C. Returns A (k=10*3 default)
    # R2 (filter): gets None. Returns None.
    # R3 (no filter): gets A, D. Returns A, D.
    # RRF: A from R1(rank 0) + R3(rank 0), C from R1(rank 1), D from R3(rank 1)
    # Scores: A: 1/61+1/61, C: 1/62, D: 1/62. Order: A > C ~ D
    assert len(results) > 0 # Should find at least doc A
    assert results[0].document.metadata.get('id') == "doc-a"
    # Based on the mock filter logic, only doc A and C from retriever1,
    # and A, D from retriever_no_filter should potentially appear.
    # R2 returns nothing with the filter.
    found_tags = {res.document.metadata.get("tag") for res in results}
    assert "finance" not in found_tags # Doc B should be filtered out by R1/R2

@pytest.mark.asyncio
async def test_finder_asearch_basic_rrf(retriever1, retriever2, sample_docs):
    """Test basic async search and RRF reranking."""
    finder = Finder(retrievers=[retriever1, retriever2], rrf_k=60)
    results = await finder.asearch("test query async", top_k=3)

    # Expected order is the same as sync: B > A > D > C
    assert len(results) == 3
    assert isinstance(results[0], SearchResult)
    assert results[0].document.metadata.get('doc_id') == "doc-b"
    assert results[1].document.metadata.get('id') == "doc-a"
    assert results[2].document.metadata.get('id') == "doc-d"
    assert results[0].score > results[1].score > results[2].score

@pytest.mark.asyncio
async def test_finder_asearch_with_filter(mocker, retriever1, retriever2):
    """Test async search with filter."""
    spy1 = mocker.spy(retriever1, "_aget_relevant_documents")
    spy2 = mocker.spy(retriever2, "_aget_relevant_documents")

    finder = Finder(retrievers=[retriever1, retriever2])
    test_filter = {"tag": "finance"}
    results = await finder.asearch("filter query async", top_k=2, filter=test_filter)

    spy1.assert_called_once()
    assert spy1.call_args.kwargs.get('filter') == test_filter
    spy2.assert_called_once()
    assert spy2.call_args.kwargs.get('filter') == test_filter

    # R1 filter: None
    # R2 filter: B
    # RRF: Only B remains
    assert len(results) == 1
    assert results[0].document.metadata.get("doc_id") == "doc-b"

def test_finder_search_doc_id_handling(retriever1, retriever2, sample_docs):
    """Test documents with different/missing IDs are handled."""
    finder = Finder(retrievers=[retriever1, retriever2]) # Includes doc C without standard ID
    results = finder.search("doc id test", top_k=4) # Get all potential results

    assert len(results) == 4 # A, B, C, D

    result_contents = {res.document.page_content for res in results}
    assert sample_docs[0].page_content in result_contents
    assert sample_docs[1].page_content in result_contents
    assert sample_docs[2].page_content in result_contents # Doc C should be present
    assert sample_docs[3].page_content in result_contents

    # Ensure no duplicates based on content despite different retriever rankings
    contents = [r.document.page_content for r in results]
    assert len(contents) == len(set(contents))

def test_finder_search_empty_results():
    """Test search when retrievers return nothing."""
    empty_retriever1 = MockRetriever(docs=[], identifier="empty1")
    empty_retriever2 = MockRetriever(docs=[], identifier="empty2")
    finder = Finder(retrievers=[empty_retriever1, empty_retriever2])
    results = finder.search("empty test")
    assert len(results) == 0

@pytest.mark.asyncio
async def test_finder_asearch_empty_results():
    """Test async search when retrievers return nothing."""
    empty_retriever1 = MockRetriever(docs=[], identifier="empty1")
    empty_retriever2 = MockRetriever(docs=[], identifier="empty2")
    finder = Finder(retrievers=[empty_retriever1, empty_retriever2])
    results = await finder.asearch("empty test async")
    assert len(results) == 0

# Optional: Test retriever error handling if needed
# class FailingRetriever(BaseRetriever):
#     def _get_relevant_documents(self, query: str, *, run_manager, **kwargs) -> List[Document]:
#         raise ValueError("Simulated retrieval error")
#     async def _aget_relevant_documents(self, query: str, *, run_manager, **kwargs) -> List[Document]:
#         raise ValueError("Simulated async retrieval error")

# def test_finder_search_retriever_error(retriever1):
#      failing_retriever = FailingRetriever()
#      finder = Finder(retrievers=[retriever1, failing_retriever])
#      # Finder should still return results from the working retriever
#      results = finder.search("error test")
#      assert len(results) > 0
#      # Check sys print for error message? (Or use logging mock) 