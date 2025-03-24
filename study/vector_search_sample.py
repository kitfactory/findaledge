"""
Vector Search Sample with ChromaDB
ベクトル検索サンプル（ChromaDB使用）

This sample demonstrates the basic usage of ChromaDB for vector search.
このサンプルはChromaDBを使用したベクトル検索の基本的な使い方を示します。
"""

import os
import chromadb
from langchain_openai import OpenAIEmbeddings
from oneenv import load_dotenv

# Load environment variables from .env file
# 環境変数を.envファイルから読み込む
load_dotenv()

def vector_search_sample():
    """
    Demonstrates basic vector search using ChromaDB
    ChromaDBを使用した基本的なベクトル検索のデモンストレーション
    """
    # Check if OpenAI API Key is set
    # OpenAI APIキーが設定されているか確認
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        print("OpenAI APIキーを環境変数に設定してください")
        return

    print("Creating ChromaDB client...")
    print("ChromaDBクライアントを作成しています...")
    
    # Create a persistent ChromaDB client
    # 永続化されたChromaDBクライアントを作成
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Create a collection (or get if it already exists)
    # コレクションを作成（または既存のものを取得）
    collection_name = "sample_documents"
    try:
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Sample documents for vector search"}
        )
    except Exception as e:
        print(f"Error creating collection: {e}")
        print(f"コレクション作成エラー: {e}")
        return

    # Create OpenAI embeddings
    # OpenAIのembeddingsを作成
    embeddings = OpenAIEmbeddings()

    # Sample documents
    # サンプルドキュメント
    documents = [
        "東京は日本の首都であり、世界最大の都市の一つです。",
        "京都は日本の古都で、多くの寺院や神社があります。",
        "大阪は日本第二の都市で、美味しい食べ物で有名です。",
        "北海道は日本最北の島で、美しい自然と温泉があります。",
        "沖縄は日本最南の県で、美しいビーチとサンゴ礁があります。"
    ]
    
    # Add documents to collection
    # ドキュメントをコレクションに追加
    print("Adding documents to collection...")
    print("ドキュメントをコレクションに追加しています...")
    
    # First clear the collection to avoid duplicates in multiple runs
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
                metadata={"description": "Sample documents for vector search"}
            )
            print("Collection was recreated")
            print("コレクションが再作成されました")
        except Exception as e2:
            print(f"Error recreating collection: {e2}")
            print(f"コレクション再作成エラー: {e2}")
    
    # Generate embeddings for documents
    # ドキュメントのembeddingsを生成
    embeddings_list = embeddings.embed_documents(documents)
    
    # Add documents to collection with their embeddings
    # ドキュメントとそのembeddingsをコレクションに追加
    for i, (doc, emb) in enumerate(zip(documents, embeddings_list)):
        collection.add(
            ids=[f"doc_{i}"],
            documents=[doc],
            embeddings=[emb]
        )
    
    # Query examples
    # クエリ例
    queries = [
        "日本の首都はどこですか？",
        "日本で古い寺院を見るならどこがいいですか？",
        "美味しい食べ物で有名な都市はどこですか？",
        "日本で自然を楽しむならどこがいいですか？"
    ]
    
    print("\nPerforming vector searches...")
    print("ベクトル検索を実行しています...\n")
    
    for query in queries:
        print(f"Query: {query}")
        print(f"クエリ: {query}")
        
        # Generate embedding for the query
        # クエリのembeddingを生成
        query_embedding = embeddings.embed_query(query)
        
        # Search the collection
        # コレクションを検索
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=2  # Return top 2 results
        )
        
        # Display results
        # 結果を表示
        print("Results / 結果:")
        for i, (doc_id, document, score) in enumerate(zip(
                results["ids"][0],
                results["documents"][0],
                results["distances"][0] if "distances" in results else [None] * len(results["ids"][0])
            )):
            print(f"  {i+1}. Document: {document}")
            if score is not None:
                print(f"     Similarity: {1 - score:.4f}")  # Convert distance to similarity
            print(f"     ID: {doc_id}")
        print()

if __name__ == "__main__":
    vector_search_sample() 