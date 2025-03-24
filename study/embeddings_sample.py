"""
Embeddings Generation Sample with OpenAI
OpenAIを使用した埋め込みベクトル生成サンプル

This sample demonstrates the basic usage of OpenAI API for generating text embeddings.
このサンプルはOpenAI APIを使用したテキスト埋め込みベクトル生成の基本的な使い方を示します。
"""

import os
import numpy as np
from oneenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env file
# 環境変数を.envファイルから読み込む
load_dotenv()

def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors
    2つのベクトル間のコサイン類似度を計算
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        Cosine similarity value between 0 and 1
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def embeddings_sample():
    """
    Demonstrate basic embeddings generation using OpenAI API
    OpenAI APIを使用した基本的な埋め込みベクトル生成のデモンストレーション
    """
    # Check if OpenAI API Key is set
    # OpenAI APIキーが設定されているか確認
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        print("OpenAI APIキーを環境変数に設定してください")
        return
    
    print("OpenAI Embeddings Sample")
    print("OpenAI埋め込みベクトルサンプル")
    
    # Create OpenAI embeddings generator
    # OpenAI埋め込みベクトル生成器を作成
    embeddings = OpenAIEmbeddings()
    
    # Sample texts
    # サンプルテキスト
    texts = [
        "東京は日本の首都であり、世界最大の都市の一つです。",
        "京都は日本の古都で、多くの寺院や神社があります。",
        "大阪は日本第二の都市で、美味しい食べ物で有名です。",
        "北海道は日本最北の島で、美しい自然と温泉があります。",
        "沖縄は日本最南の県で、美しいビーチとサンゴ礁があります。"
    ]
    
    print("\nGenerating embeddings for sample texts...")
    print("サンプルテキストの埋め込みベクトルを生成しています...")
    
    # Generate embeddings for all texts
    # すべてのテキストの埋め込みベクトルを生成
    text_embeddings = embeddings.embed_documents(texts)
    
    # Print some information about the embeddings
    # 埋め込みベクトルに関する情報を表示
    print(f"Embedding model: {embeddings.model}")
    print(f"埋め込みモデル: {embeddings.model}")
    print(f"Embedding dimensions: {len(text_embeddings[0])}")
    print(f"埋め込みの次元数: {len(text_embeddings[0])}")
    
    # Calculate similarity between texts
    # テキスト間の類似度を計算
    print("\nCalculating similarity between texts...")
    print("テキスト間の類似度を計算しています...")
    
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            similarity = cosine_similarity(text_embeddings[i], text_embeddings[j])
            print(f"Similarity between text {i+1} and text {j+1}: {similarity:.4f}")
            print(f"テキスト{i+1}とテキスト{j+1}の類似度: {similarity:.4f}")
    
    # Generate embedding for a query
    # クエリの埋め込みベクトルを生成
    print("\nGenerating embedding for a query...")
    print("クエリの埋め込みベクトルを生成しています...")
    
    query = "日本で美しい自然を見るにはどこがいいですか？"
    query_embedding = embeddings.embed_query(query)
    
    print(f"Query: {query}")
    print(f"クエリ: {query}")
    
    # Find most similar text to query
    # クエリに最も類似したテキストを検索
    similarities = [cosine_similarity(query_embedding, text_embedding) 
                   for text_embedding in text_embeddings]
    
    most_similar_idx = np.argmax(similarities)
    
    print("\nFinding most similar text to query...")
    print("クエリに最も類似したテキストを検索しています...")
    print(f"Most similar text: {texts[most_similar_idx]}")
    print(f"最も類似したテキスト: {texts[most_similar_idx]}")
    print(f"Similarity score: {similarities[most_similar_idx]:.4f}")
    print(f"類似度スコア: {similarities[most_similar_idx]:.4f}")
    
    # Print all similarities in descending order
    # すべての類似度を降順で表示
    print("\nAll similarities in descending order:")
    print("すべての類似度（降順）:")
    
    sorted_indices = np.argsort(similarities)[::-1]
    for i, idx in enumerate(sorted_indices):
        print(f"{i+1}. Text: {texts[idx]}")
        print(f"   Similarity: {similarities[idx]:.4f}")

if __name__ == "__main__":
    embeddings_sample() 