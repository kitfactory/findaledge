"""
BM25 Persistence Sample Simplified
BM25永続化サンプル（簡略化版）

This script demonstrates a simplified way to persist and reload BM25 index.
このスクリプトはBM25インデックスの永続化と再読み込みを示す簡略化したサンプルです。
"""

import os
import json
import numpy as np
from bm25s import BM25, tokenize

def bm25_simple_persistence():
    """
    Simple demonstration of BM25 persistence by reindexing
    再インデックス化によるBM25永続化の簡単なデモンストレーション
    """
    print("BM25 Simple Persistence Sample")
    print("BM25簡易永続化サンプル")
    
    # Define persistent directory and filename
    # 永続ディレクトリとファイル名を定義
    persist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bm25_simple_persist")
    corpus_file = os.path.join(persist_dir, "corpus.json")
    
    # Create directory if it doesn't exist
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)
    elif os.path.exists(corpus_file):
        # Clean up existing file for demonstration
        # デモンストレーション用に既存のファイルをクリーンアップ
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
    
    print("\n1. Creating and using a BM25 index")
    print("1. BM25インデックスを作成して使用")
    
    # Tokenize the corpus
    # コーパスをトークン化
    tokenized_corpus = [doc.split() for doc in corpus]
    
    # Create the BM25 index
    # BM25インデックスを作成
    bm25 = BM25()
    bm25.index(tokenized_corpus)
    
    # Perform a search using the index
    # インデックスを使用して検索を実行
    query = "日本 歴史"
    tokenized_query = query.split()
    original_scores = bm25.get_scores(tokenized_query)
    
    print("\nSearch results (original index):")
    print("検索結果（元のインデックス）:")
    for i, (doc, score) in enumerate(zip(corpus, original_scores)):
        if score > 0:
            print(f"{i+1}. Score: {score:.4f}, Document: {doc}")
    
    # Save the corpus to a file (persistence)
    # コーパスをファイルに保存（永続化）
    with open(corpus_file, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    
    print(f"\nCorpus has been persisted to: {corpus_file}")
    print(f"コーパスが次の場所に永続化されました: {corpus_file}")
    
    # Clear memory (for demonstration)
    # メモリをクリア（デモンストレーション用）
    del bm25
    del tokenized_corpus
    
    print("\n2. Reloading and using the BM25 index")
    print("2. BM25インデックスを再読み込みして使用")
    
    # Load the corpus
    # コーパスを読み込み
    with open(corpus_file, "r", encoding="utf-8") as f:
        loaded_corpus = json.load(f)
    
    # Tokenize the loaded corpus
    # 読み込んだコーパスをトークン化
    loaded_tokenized_corpus = [doc.split() for doc in loaded_corpus]
    
    # Create a new BM25 instance with the loaded corpus
    # 読み込んだコーパスで新しいBM25インスタンスを作成
    loaded_bm25 = BM25()
    loaded_bm25.index(loaded_tokenized_corpus)
    
    # Perform the same search using the recreated index
    # 再作成されたインデックスを使用して同じ検索を実行
    loaded_scores = loaded_bm25.get_scores(tokenized_query)
    
    print("\nSearch results (recreated index):")
    print("検索結果（再作成されたインデックス）:")
    for i, (doc, score) in enumerate(zip(loaded_corpus, loaded_scores)):
        if score > 0:
            print(f"{i+1}. Score: {score:.4f}, Document: {doc}")
    
    # Verify that the scores are the same
    # スコアが同じであることを確認
    scores_match = np.allclose(original_scores, loaded_scores)
    print(f"\nDo the scores match? {scores_match}")
    print(f"スコアは一致していますか？ {scores_match}")
    
    print("\nBM25 simplified persistence demonstration completed")
    print("BM25簡易永続化のデモンストレーションが完了しました")

if __name__ == "__main__":
    bm25_simple_persistence() 