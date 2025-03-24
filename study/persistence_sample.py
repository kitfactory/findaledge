"""
Index Persistence Sample
インデックス永続化サンプル

This sample demonstrates how to persist and reload search indices for both vector and keyword search.
このサンプルはベクトル検索とキーワード検索の両方のインデックスを永続化し、再読み込みする方法を示します。
"""

import os
import shutil
import json
import time
import numpy as np
import chromadb
from bm25s import BM25
from langchain_openai import OpenAIEmbeddings
from oneenv import load_dotenv

def vector_index_persistence_sample():
    """
    Demonstrates persisting and reloading a vector index with ChromaDB
    ChromaDBを使用したベクトルインデックスの永続化と再読み込みのデモンストレーション
    """
    print("\n=== Vector Index Persistence Sample / ベクトルインデックス永続化サンプル ===")
    
    # Load environment variables
    # 環境変数をロード
    load_dotenv()
    
    # Check if OpenAI API key is set
    # OpenAI APIキーが設定されているか確認
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set. Please set it before running this sample.")
        print("エラー: OPENAI_API_KEY環境変数が設定されていません。このサンプルを実行する前に設定してください。")
        return
    
    # Define persistent directory
    # 永続ディレクトリを定義
    persist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_persist_dir")
    collection_name = "persistence_sample"
    
    # Clean up any existing data (for demonstration purposes)
    # 既存のデータをクリーンアップ（デモンストレーション目的）
    if os.path.exists(persist_dir):
        print(f"Removing existing directory: {persist_dir}")
        print(f"既存のディレクトリを削除します: {persist_dir}")
        shutil.rmtree(persist_dir)
    
    # Initialize embeddings
    # 埋め込みを初期化
    embeddings = OpenAIEmbeddings()
    
    # Sample documents
    # サンプル文書
    documents = [
        "東京は日本の首都で、世界最大の都市圏を形成しています。",
        "京都は千年以上にわたって日本の古都であり、多くの歴史的建造物があります。",
        "大阪は日本第二の都市で、美味しい食べ物と商業で知られています。",
        "北海道は日本最北の島で、美しい自然と冷涼な気候が特徴です。",
        "沖縄は日本最南端の県で、熱帯気候と独自の文化で知られています。"
    ]
    
    document_ids = [f"doc{i}" for i in range(len(documents))]
    
    print("\n1. Creating and persisting a new vector index")
    print("1. 新しいベクトルインデックスを作成して永続化")
    
    # Create a persistent client
    # 永続クライアントを作成
    client = chromadb.PersistentClient(path=persist_dir)
    
    # Create a collection
    # コレクションを作成
    collection = client.create_collection(name=collection_name)
    
    # Generate embeddings and add documents
    # 埋め込みを生成して文書を追加
    doc_embeddings = [embeddings.embed_query(doc) for doc in documents]
    
    collection.add(
        ids=document_ids,
        embeddings=doc_embeddings,
        documents=documents,
        metadatas=[{"source": f"sample_{i}"} for i in range(len(documents))]
    )
    
    # Verify data is in the collection
    # コレクション内のデータを確認
    print(f"Added {len(documents)} documents to the collection")
    print(f"{len(documents)} 件の文書をコレクションに追加しました")
    
    # Perform a search
    # 検索を実行
    query = "日本の歴史的な都市"
    query_embedding = embeddings.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )
    
    print("\nSearch results (original index):")
    print("検索結果（元のインデックス）:")
    for i, (doc_id, doc, score) in enumerate(zip(results["ids"][0], results["documents"][0], results["distances"][0])):
        print(f"{i+1}. ID: {doc_id}, Document: {doc}, Distance: {score}")
    
    # Close the client to ensure data is flushed
    # データが確実に書き込まれるようにクライアントを閉じる
    del collection
    del client
    print("\nVector index has been persisted to disk")
    print("ベクトルインデックスがディスクに永続化されました")
    
    print("\n2. Reloading the persisted vector index")
    print("2. 永続化されたベクトルインデックスを再読み込み")
    
    # Create a new client instance pointing to the same directory
    # 同じディレクトリを指すクライアントの新しいインスタンスを作成
    new_client = chromadb.PersistentClient(path=persist_dir)
    
    # Get the existing collection
    # 既存のコレクションを取得
    new_collection = new_client.get_collection(name=collection_name)
    
    # Verify the collection has the data
    # コレクションにデータがあることを確認
    print(f"Collection loaded with {new_collection.count()} documents")
    print(f"{new_collection.count()} 件の文書を持つコレクションが読み込まれました")
    
    # Perform the same search again
    # 同じ検索を再度実行
    new_results = new_collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )
    
    print("\nSearch results (reloaded index):")
    print("検索結果（再読み込みされたインデックス）:")
    for i, (doc_id, doc, score) in enumerate(zip(new_results["ids"][0], new_results["documents"][0], new_results["distances"][0])):
        print(f"{i+1}. ID: {doc_id}, Document: {doc}, Distance: {score}")
    
    # Clean up (for demonstration purposes)
    # クリーンアップ（デモンストレーション目的）
    del new_collection
    del new_client
    
    print("\nVector index persistence demonstration completed")
    print("ベクトルインデックス永続化のデモンストレーションが完了しました")

def bm25_index_persistence_sample():
    """
    Demonstrates persisting and reloading a BM25 index
    BM25インデックスの永続化と再読み込みのデモンストレーション
    """
    print("\n=== BM25 Index Persistence Sample / BM25インデックス永続化サンプル ===")
    
    # Define persistent directory and filename
    # 永続ディレクトリとファイル名を定義
    persist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bm25_persist_dir")
    corpus_file = os.path.join(persist_dir, "corpus.json")
    
    # Create directory if it doesn't exist
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)
    else:
        # Clean up existing files for demonstration
        # デモンストレーション用に既存のファイルをクリーンアップ
        if os.path.exists(corpus_file):
            os.remove(corpus_file)
            print(f"Removed existing file: {corpus_file}")
            print(f"既存のファイルを削除しました: {corpus_file}")
    
    # Sample corpus with Japanese text
    # 日本語テキストのサンプルコーパス
    corpus = [
        "東京は日本の首都で、世界最大の都市圏を形成しています。",
        "京都は千年以上にわたって日本の古都であり、多くの歴史的建造物があります。",
        "大阪は日本第二の都市で、美味しい食べ物と商業で知られています。",
        "北海道は日本最北の島で、美しい自然と冷涼な気候が特徴です。",
        "沖縄は日本最南端の県で、熱帯気候と独自の文化で知られています。"
    ]
    
    print("\n1. Creating and persisting a new BM25 index")
    print("1. 新しいBM25インデックスを作成して永続化")
    
    # Tokenize the corpus
    # コーパスをトークン化
    tokenized_corpus = [doc.split() for doc in corpus]
    
    # Create the BM25 index
    # BM25インデックスを作成
    bm25 = BM25()
    bm25.index(tokenized_corpus)
    
    # Save the corpus to a file
    # コーパスをファイルに保存
    with open(corpus_file, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    
    print(f"Corpus has been persisted to: {corpus_file}")
    print(f"コーパスが次の場所に永続化されました: {corpus_file}")
    
    # Perform a search using the original index
    # 元のインデックスを使用して検索を実行
    query = "日本 歴史"
    tokenized_query = query.split()
    original_scores = bm25.get_scores(tokenized_query)
    
    print("\nSearch results (original index):")
    print("検索結果（元のインデックス）:")
    for i, (doc, score) in enumerate(zip(corpus, original_scores)):
        if score > 0:
            print(f"{i+1}. Score: {score:.4f}, Document: {doc}")
    
    # Clear memory (for demonstration)
    # メモリをクリア（デモンストレーション用）
    del bm25
    del tokenized_corpus
    
    print("\n2. Reloading the persisted BM25 index")
    print("2. 永続化されたBM25インデックスを再読み込み")
    
    # Load the corpus
    # コーパスを読み込み
    with open(corpus_file, "r", encoding="utf-8") as f:
        loaded_corpus = json.load(f)
    
    # Tokenize the loaded corpus
    # 読み込んだコーパスをトークン化
    loaded_tokenized_corpus = [doc.split() for doc in loaded_corpus]
    
    # Create a new BM25 instance and index the loaded corpus
    # 新しいBM25インスタンスを作成し、読み込んだコーパスをインデックス化
    loaded_bm25 = BM25()
    loaded_bm25.index(loaded_tokenized_corpus)
    
    # Perform the same search using the reloaded index
    # 再読み込みされたインデックスを使用して同じ検索を実行
    loaded_scores = loaded_bm25.get_scores(tokenized_query)
    
    print("\nSearch results (reloaded index):")
    print("検索結果（再読み込みされたインデックス）:")
    for i, (doc, score) in enumerate(zip(loaded_corpus, loaded_scores)):
        if score > 0:
            print(f"{i+1}. Score: {score:.4f}, Document: {doc}")
    
    # Verify that the scores are the same
    # スコアが同じであることを確認
    scores_match = np.allclose(original_scores, loaded_scores)
    print(f"\nDo the scores match? {scores_match}")
    print(f"スコアは一致していますか？ {scores_match}")
    
    print("\nBM25 index persistence demonstration completed")
    print("BM25インデックス永続化のデモンストレーションが完了しました")

def combined_persistence_sample():
    """
    Demonstrates a combined approach for persisting both vector and BM25 indices
    ベクトルインデックスとBM25インデックスの両方を永続化する組み合わせアプローチのデモンストレーション
    """
    print("\n=== Combined Persistence Sample / 組み合わせ永続化サンプル ===")
    
    # Load environment variables
    # 環境変数をロード
    load_dotenv()
    
    # Check if OpenAI API key is set
    # OpenAI APIキーが設定されているか確認
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set. Please set it before running this sample.")
        print("エラー: OPENAI_API_KEY環境変数が設定されていません。このサンプルを実行する前に設定してください。")
        return
    
    # Define root persistent directory
    # ルート永続ディレクトリを定義
    root_persist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hybrid_persist_dir")
    
    # Define subdirectories
    # サブディレクトリを定義
    vector_persist_dir = os.path.join(root_persist_dir, "vector")
    bm25_persist_dir = os.path.join(root_persist_dir, "bm25")
    meta_file = os.path.join(root_persist_dir, "metadata.json")
    
    # Clean up existing directories (for demonstration)
    # 既存のディレクトリをクリーンアップ（デモンストレーション用）
    if os.path.exists(root_persist_dir):
        print(f"Removing existing directory: {root_persist_dir}")
        print(f"既存のディレクトリを削除します: {root_persist_dir}")
        shutil.rmtree(root_persist_dir)
    
    # Create directories
    # ディレクトリを作成
    os.makedirs(vector_persist_dir, exist_ok=True)
    os.makedirs(bm25_persist_dir, exist_ok=True)
    
    # Sample documents
    # サンプル文書
    documents = [
        "東京は日本の首都で、世界最大の都市圏を形成しています。",
        "京都は千年以上にわたって日本の古都であり、多くの歴史的建造物があります。",
        "大阪は日本第二の都市で、美味しい食べ物と商業で知られています。",
        "北海道は日本最北の島で、美しい自然と冷涼な気候が特徴です。",
        "沖縄は日本最南端の県で、熱帯気候と独自の文化で知られています。"
    ]
    
    document_ids = [f"doc{i}" for i in range(len(documents))]
    
    print("\n1. Creating and persisting combined indices")
    print("1. 組み合わせインデックスを作成して永続化")
    
    # Initialize embeddings
    # 埋め込みを初期化
    embeddings = OpenAIEmbeddings()
    
    # Step 1: Create and persist vector index
    # ステップ1: ベクトルインデックスを作成して永続化
    client = chromadb.PersistentClient(path=vector_persist_dir)
    collection = client.create_collection(name="hybrid_sample")
    
    doc_embeddings = [embeddings.embed_query(doc) for doc in documents]
    
    collection.add(
        ids=document_ids,
        embeddings=doc_embeddings,
        documents=documents,
        metadatas=[{"source": f"sample_{i}"} for i in range(len(documents))]
    )
    
    print(f"Vector index created with {len(documents)} documents")
    print(f"ベクトルインデックスが {len(documents)} 件の文書で作成されました")
    
    # Step 2: Create and persist BM25 index
    # ステップ2: BM25インデックスを作成して永続化
    tokenized_corpus = [doc.split() for doc in documents]
    bm25 = BM25()
    bm25.index(tokenized_corpus)
    
    # Save BM25 corpus
    # BM25コーパスを保存
    corpus_file = os.path.join(bm25_persist_dir, "corpus.json")
    
    with open(corpus_file, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    print(f"BM25 index created and corpus persisted")
    print(f"BM25インデックスが作成され、コーパスが永続化されました")
    
    # Step 3: Save metadata
    # ステップ3: メタデータを保存
    metadata = {
        "created_at": time.time(),
        "document_count": len(documents),
        "document_ids": document_ids,
        "embedding_model": "text-embedding-ada-002",  # Adjust based on actual model used
        "vector_collection": "hybrid_sample",
        "has_bm25_index": True
    }
    
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Metadata saved to: {meta_file}")
    print(f"メタデータが次の場所に保存されました: {meta_file}")
    
    # Record search results for comparison
    # 比較のために検索結果を記録
    query = "日本 歴史"
    query_embedding = embeddings.embed_query(query)
    tokenized_query = query.split()
    
    # Vector search
    # ベクトル検索
    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    # BM25 search
    # BM25検索
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Clean up references to ensure data is flushed
    # データが確実に書き込まれるように参照をクリーンアップ
    del collection
    del client
    del bm25
    
    print("\n2. Reloading the persisted combined indices")
    print("2. 永続化された組み合わせインデックスを再読み込み")
    
    # Step 1: Load metadata
    # ステップ1: メタデータを読み込み
    with open(meta_file, "r", encoding="utf-8") as f:
        loaded_metadata = json.load(f)
    
    print(f"Loaded metadata: {loaded_metadata['document_count']} documents, created at {loaded_metadata['created_at']}")
    print(f"読み込まれたメタデータ: {loaded_metadata['document_count']} 件の文書、作成日時 {loaded_metadata['created_at']}")
    
    # Step 2: Load vector index
    # ステップ2: ベクトルインデックスを読み込み
    new_client = chromadb.PersistentClient(path=vector_persist_dir)
    new_collection = new_client.get_collection(name=loaded_metadata["vector_collection"])
    
    # Step 3: Load BM25 index
    # ステップ3: BM25インデックスを読み込み
    with open(corpus_file, "r", encoding="utf-8") as f:
        loaded_corpus = json.load(f)
    
    loaded_tokenized_corpus = [doc.split() for doc in loaded_corpus]
    
    loaded_bm25 = BM25()
    loaded_bm25.index(loaded_tokenized_corpus)
    
    # Perform the same searches to verify
    # 検証のために同じ検索を実行
    new_vector_results = new_collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    new_bm25_scores = loaded_bm25.get_scores(tokenized_query)
    
    # Compare results
    # 結果を比較
    print("\nVector search results match:", 
          vector_results["ids"][0] == new_vector_results["ids"][0])
    print("ベクトル検索結果が一致:", 
          vector_results["ids"][0] == new_vector_results["ids"][0])
    
    print("BM25 search results match:", 
          np.allclose(bm25_scores, new_bm25_scores))
    print("BM25検索結果が一致:", 
          np.allclose(bm25_scores, new_bm25_scores))
    
    print("\nCombined persistence demonstration completed")
    print("組み合わせ永続化のデモンストレーションが完了しました")

def persistence_sample():
    """
    Main function to run all persistence samples
    すべての永続化サンプルを実行するメイン関数
    """
    print("Index Persistence Sample")
    print("インデックス永続化サンプル")
    print("=" * 60)
    
    # Run individual samples
    # 個別サンプルを実行
    vector_index_persistence_sample()
    bm25_index_persistence_sample()
    combined_persistence_sample()
    
    print("\nAll persistence samples completed")
    print("すべての永続化サンプルが完了しました")

if __name__ == "__main__":
    persistence_sample() 