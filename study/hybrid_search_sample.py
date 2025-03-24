"""
Hybrid Search Sample with RRF (Reciprocal Rank Fusion)
RRF（Reciprocal Rank Fusion）を使用したハイブリッド検索サンプル

This sample demonstrates the combination of vector search and BM25 keyword search using the RRF algorithm.
このサンプルはRRFアルゴリズムを使用してベクトル検索とBM25キーワード検索を組み合わせた方法を示します。
"""

import os
import numpy as np
import chromadb
import bm25s
from oneenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env file
# 環境変数を.envファイルから読み込む
load_dotenv()

# RRF constant k - controls the relative impact of lower-ranked documents
# RRF定数k - 下位ランクのドキュメントの相対的な影響を制御
RRF_K = 60

def reciprocal_rank_fusion(results_lists, k=RRF_K):
    """
    Implementation of Reciprocal Rank Fusion algorithm
    Reciprocal Rank Fusionアルゴリズムの実装
    
    Args:
        results_lists: List of lists of document IDs, ordered by rank from each search method
        k: RRF constant (default: 60)
        
    Returns:
        List of document IDs sorted by RRF score
    """
    # Initialize dictionary to store RRF scores for each document
    # 各ドキュメントのRRFスコアを格納する辞書を初期化
    rrf_scores = {}
    
    # Calculate RRF scores
    # RRFスコアを計算
    for results in results_lists:
        for rank, doc_id in enumerate(results):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
            # RRF formula: 1 / (rank + k)
            # RRF式: 1 / (ランク + k)
            rrf_scores[doc_id] += 1.0 / (rank + k)
    
    # Sort documents by RRF score in descending order
    # RRFスコアの降順でドキュメントをソート
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return sorted document IDs
    # ソートされたドキュメントIDを返す
    return [doc_id for doc_id, _ in sorted_docs]

def weighted_score_fusion(doc_scores_map1, doc_scores_map2, weight1=0.5, weight2=0.5):
    """
    Combine scores from two retrieval methods using weighted averaging
    2つの検索方法からのスコアを重み付き平均を使用して組み合わせる
    
    Args:
        doc_scores_map1: Dictionary mapping document IDs to scores from first method
        doc_scores_map2: Dictionary mapping document IDs to scores from second method
        weight1: Weight for first method scores (default: 0.5)
        weight2: Weight for second method scores (default: 0.5)
        
    Returns:
        Dictionary mapping document IDs to combined scores
    """
    # Get all unique document IDs
    # すべてのユニークなドキュメントIDを取得
    all_docs = set(doc_scores_map1.keys()) | set(doc_scores_map2.keys())
    
    # Combine scores
    # スコアを組み合わせる
    combined_scores = {}
    
    for doc_id in all_docs:
        score1 = doc_scores_map1.get(doc_id, 0)
        score2 = doc_scores_map2.get(doc_id, 0)
        combined_scores[doc_id] = weight1 * score1 + weight2 * score2
    
    return combined_scores

def hybrid_search_sample():
    """
    Demonstrate a hybrid search using vector search and BM25 keyword search, combining results with RRF method
    ベクトル検索とBM25キーワード検索を使用したハイブリッド検索を実演し、RRF方式で結果を組み合わせる
    """
    # Initialize environment
    # 環境の初期化
    load_dotenv()
    
    # Check if OpenAI API Key is set
    # OpenAI APIキーが設定されているか確認
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        print("OpenAI APIキーを環境変数に設定してください")
        return
    
    print("Hybrid Search Sample (Vector + BM25)")
    print("ハイブリッド検索サンプル（ベクトル + BM25）")
    
    # Sample documents - same as other samples for consistency
    # サンプルドキュメント - 一貫性のために他のサンプルと同じ
    documents = [
        "東京は日本の首都であり、世界最大の都市の一つです。",
        "京都は日本の古都で、多くの寺院や神社があります。",
        "大阪は日本第二の都市で、美味しい食べ物で有名です。",
        "北海道は日本最北の島で、美しい自然と温泉があります。",
        "沖縄は日本最南の県で、美しいビーチとサンゴ礁があります。"
    ]
    
    # Setup for vector search with ChromaDB
    # ChromaDBを使用したベクトル検索のセットアップ
    print("\nSetting up vector search with ChromaDB...")
    print("ChromaDBを使用したベクトル検索を設定しています...")
    
    # Create embeddings
    # 埋め込みを作成
    embeddings = OpenAIEmbeddings()
    
    # Create ChromaDB client and collection
    # ChromaDBクライアントとコレクションを作成
    chroma_client = chromadb.PersistentClient(path="./chroma_hybrid_db")
    collection_name = "hybrid_search_sample"
    
    # Get or create collection
    # コレクションを取得または作成
    try:
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Sample documents for hybrid search"}
        )
    except Exception as e:
        print(f"Error creating collection: {e}")
        print(f"コレクション作成エラー: {e}")
        return
    
    # Clear the collection to avoid duplicates in multiple runs
    # 複数回実行時の重複を避けるためにコレクションをクリア
    try:
        # Get all document IDs in the collection
        # コレクション内のすべてのドキュメントIDを取得
        all_ids = collection.get()["ids"]
        if all_ids:
            # Delete documents by IDs if there are any
            # IDによってドキュメントを削除（存在する場合）
            collection.delete(ids=all_ids)
    except Exception as e:
        print(f"Warning when clearing collection: {e}")
        print(f"コレクションのクリア時の警告: {e}")
        # Try to recreate the collection
        # コレクションを再作成
        try:
            chroma_client.delete_collection(name=collection_name)
            collection = chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "Sample documents for hybrid search"}
            )
            print("Collection was recreated")
            print("コレクションが再作成されました")
        except Exception as e2:
            print(f"Error recreating collection: {e2}")
            print(f"コレクション再作成エラー: {e2}")
    
    # Generate embeddings and add documents to ChromaDB
    # 埋め込みを生成し、ドキュメントをChromaDBに追加
    embeddings_list = embeddings.embed_documents(documents)
    
    for i, (doc, emb) in enumerate(zip(documents, embeddings_list)):
        collection.add(
            ids=[f"doc_{i}"],
            documents=[doc],
            embeddings=[emb]
        )
    
    # Setup for BM25 search
    # BM25検索のセットアップ
    print("Setting up BM25 search...")
    print("BM25検索を設定しています...")
    
    # Tokenize corpus for BM25
    # BM25用にコーパスをトークン化
    corpus_tokens = bm25s.tokenize(documents, stopwords="japanese")
    
    # Create BM25 retriever and index corpus
    # BM25リトリーバーを作成し、コーパスをインデックス化
    bm25_retriever = bm25s.BM25()
    bm25_retriever.index(corpus_tokens)
    
    # Query examples
    # クエリ例
    queries = [
        "日本の首都はどこですか？",
        "日本で古い寺院を見るならどこがいいですか？",
        "美味しい食べ物で有名な都市はどこですか？",
        "日本で自然を楽しむならどこがいいですか？"
    ]
    
    print("\nPerforming hybrid searches...")
    print("ハイブリッド検索を実行しています...\n")
    
    for query in queries:
        print(f"Query: {query}")
        print(f"クエリ: {query}")
        
        # Vector search with ChromaDB
        # ChromaDBでベクトル検索
        query_embedding = embeddings.embed_query(query)
        vector_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=len(documents)  # Get all documents for demo
        )
        
        # BM25 search
        # BM25検索
        query_tokens = bm25s.tokenize(query, stopwords="japanese")
        bm25_results, bm25_scores = bm25_retriever.retrieve(
            query_tokens, 
            k=len(documents),  # Get all documents for demo
            corpus=documents
        )
        
        # Extract document IDs and scores
        # ドキュメントIDとスコアを抽出
        vector_doc_ids = vector_results["ids"][0]
        vector_doc_scores = {
            doc_id: 1 - dist  # Convert distance to similarity
            for doc_id, dist in zip(
                vector_doc_ids, 
                vector_results["distances"][0] if "distances" in vector_results else [0] * len(vector_doc_ids)
            )
        }
        
        # Convert BM25 results to map of doc_id -> score
        # BM25結果をdoc_id -> scoreのマップに変換
        bm25_doc_ids = []
        bm25_doc_scores = {}
        
        for i, doc_text in enumerate(bm25_results[0]):
            # Find the index of the document in the original documents list
            # 元のドキュメントリストでのドキュメントのインデックスを検索
            doc_idx = documents.index(doc_text) if doc_text in documents else -1
            if doc_idx != -1:
                doc_id = f"doc_{doc_idx}"
                bm25_doc_ids.append(doc_id)
                bm25_doc_scores[doc_id] = bm25_scores[0][i]
        
        # Method 1: Use RRF for rank-based fusion
        # 方法1: ランクベースの融合にRRFを使用
        print("\nMethod 1: RRF (Reciprocal Rank Fusion)")
        print("方法1: RRF（Reciprocal Rank Fusion）")
        
        rrf_results = reciprocal_rank_fusion([vector_doc_ids, bm25_doc_ids])
        
        print("RRF Results / RRF結果:")
        for rank, doc_id in enumerate(rrf_results[:3]):  # Show top 3
            doc_idx = int(doc_id.split("_")[1])
            print(f"  {rank+1}. Document: {documents[doc_idx]}")
            print(f"     Original Vector Rank: {vector_doc_ids.index(doc_id) + 1 if doc_id in vector_doc_ids else 'Not found'}")
            print(f"     Original BM25 Rank: {bm25_doc_ids.index(doc_id) + 1 if doc_id in bm25_doc_ids else 'Not found'}")
        
        # Method 2: Use weighted score fusion
        # 方法2: 重み付きスコア融合を使用
        print("\nMethod 2: Weighted Score Fusion")
        print("方法2: 重み付きスコア融合")
        
        # Normalize BM25 scores to [0, 1] range for fair comparison
        # 公平な比較のためにBM25スコアを[0, 1]範囲に正規化
        if bm25_doc_scores:
            max_bm25_score = max(bm25_doc_scores.values())
            normalized_bm25_scores = {
                doc_id: score / max_bm25_score
                for doc_id, score in bm25_doc_scores.items()
            }
        else:
            normalized_bm25_scores = {}
        
        # Combine scores with equal weights
        # 等しい重みでスコアを結合
        combined_scores = weighted_score_fusion(
            vector_doc_scores, 
            normalized_bm25_scores,
            weight1=0.5,  # Vector search weight
            weight2=0.5   # BM25 search weight
        )
        
        # Sort by combined score
        # 結合スコアでソート
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        print("Weighted Fusion Results / 重み付き融合結果:")
        for rank, (doc_id, score) in enumerate(sorted_results[:3]):  # Show top 3
            doc_idx = int(doc_id.split("_")[1])
            print(f"  {rank+1}. Document: {documents[doc_idx]}")
            print(f"     Combined Score: {score:.4f}")
            print(f"     Vector Score: {vector_doc_scores.get(doc_id, 0):.4f}")
            print(f"     Normalized BM25 Score: {normalized_bm25_scores.get(doc_id, 0):.4f}")
        
        print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    hybrid_search_sample() 