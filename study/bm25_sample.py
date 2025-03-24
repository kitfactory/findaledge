"""
BM25 Keyword Search Sample
BM25キーワード検索サンプル

This sample demonstrates the basic usage of BM25 for keyword search using bm25s library.
このサンプルはbm25sライブラリを使用したBM25キーワード検索の基本的な使い方を示します。
"""

import bm25s

def bm25_search_sample():
    """
    Demonstrate basic keyword search using BM25
    BM25を使用した基本的なキーワード検索のデモンストレーション
    """
    print("BM25 keyword search sample")
    print("BM25キーワード検索サンプル")
    
    # Sample documents - same as the vector search sample for comparison
    # サンプルドキュメント - 比較のためにベクトル検索サンプルと同じもの
    corpus = [
        "東京は日本の首都であり、世界最大の都市の一つです。",
        "京都は日本の古都で、多くの寺院や神社があります。",
        "大阪は日本第二の都市で、美味しい食べ物で有名です。",
        "北海道は日本最北の島で、美しい自然と温泉があります。",
        "沖縄は日本最南の県で、美しいビーチとサンゴ礁があります。"
    ]
    
    print("Tokenizing documents...")
    print("文書をトークン化しています...")
    
    # Tokenize the corpus with Japanese stopwords
    # 日本語ストップワードでコーパスをトークン化
    corpus_tokens = bm25s.tokenize(corpus, stopwords="japanese")
    print(f"Tokenized corpus: {corpus_tokens}")
    print(f"トークン化されたコーパス: {corpus_tokens}")
    
    print("\nCreating BM25 index...")
    print("BM25インデックスを作成しています...")
    
    # Create a BM25 retriever and index the corpus
    # BM25リトリーバーを作成し、コーパスをインデックス化
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    
    # Query examples
    # クエリ例
    queries = [
        "日本の首都はどこですか？",
        "日本で古い寺院を見るならどこがいいですか？",
        "美味しい食べ物で有名な都市はどこですか？",
        "日本で自然を楽しむならどこがいいですか？"
    ]
    
    print("\nPerforming BM25 searches...")
    print("BM25検索を実行しています...\n")
    
    for query in queries:
        print(f"Query: {query}")
        print(f"クエリ: {query}")
        
        # Tokenize the query with Japanese stopwords
        # 日本語ストップワードでクエリをトークン化
        query_tokens = bm25s.tokenize(query, stopwords="japanese")
        print(f"Tokenized query: {query_tokens}")
        print(f"トークン化されたクエリ: {query_tokens}")
        
        # Search the corpus using BM25
        # BM25を使用してコーパスを検索
        results, scores = retriever.retrieve(query_tokens, k=2, corpus=corpus)
        
        # Display results
        # 結果を表示
        print("Results / 結果:")
        for i in range(results.shape[1]):
            doc, score = results[0, i], scores[0, i]
            print(f"  {i+1}. Document: {doc}")
            print(f"     Score: {score:.4f}")
        print()
    
    # Save the BM25 index for later use
    # 後で使用するためにBM25インデックスを保存
    print("Saving BM25 index...")
    print("BM25インデックスを保存しています...")
    
    retriever.save("bm25_sample_index", corpus=corpus)
    
    # Load the BM25 index
    # BM25インデックスを読み込む
    print("\nLoading BM25 index...")
    print("BM25インデックスを読み込んでいます...")
    
    loaded_retriever = bm25s.BM25.load("bm25_sample_index", load_corpus=True)
    
    # Verify loaded index
    # 読み込まれたインデックスを検証
    print("\nVerifying loaded index...")
    print("読み込まれたインデックスを検証しています...")
    
    query = "美しい自然"
    query_tokens = bm25s.tokenize(query, stopwords="japanese")
    
    results, scores = loaded_retriever.retrieve(query_tokens, k=1)
    doc, score = results[0, 0], scores[0, 0]
    
    print(f"Query: {query}")
    print(f"クエリ: {query}")
    print(f"Top document: {doc}")
    print(f"最上位ドキュメント: {doc}")
    print(f"Score: {score:.4f}")
    print(f"スコア: {score:.4f}")

if __name__ == "__main__":
    bm25_search_sample() 