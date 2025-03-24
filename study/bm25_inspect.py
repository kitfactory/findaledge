"""
BM25 Inspection
BM25検査

This script inspects the structure of the BM25 class from bm25s-j library.
このスクリプトはbm25s-jライブラリのBM25クラスの構造を調査します。
"""

from bm25s import BM25, tokenize

def inspect_bm25():
    """
    Inspect the BM25 class structure
    BM25クラスの構造を調査
    """
    # Create a sample corpus
    # サンプルコーパスを作成
    corpus = [
        "これは サンプル 文書 です。",
        "BM25 クラス の 構造 を 調査 します。"
    ]
    
    # Tokenize corpus
    # コーパスをトークン化
    tokenized_corpus = [doc.split() for doc in corpus]
    
    # Create BM25 instance
    # BM25インスタンスを作成
    bm25 = BM25(tokenized_corpus)
    
    # Print attributes
    # 属性を表示
    print("BM25 attributes:")
    print("BM25の属性:")
    for attr in dir(bm25):
        if not attr.startswith('__'):
            try:
                value = getattr(bm25, attr)
                print(f"{attr}: {type(value)}")
                if attr in ['corpus_size', 'avgdl', 'f', 'df', 'idf', 'doc_len']:
                    print(f"  Value: {value}")
            except Exception as e:
                print(f"{attr}: Error - {e}")
    
    # Print methods
    # メソッドを表示
    print("\nBM25 methods:")
    print("BM25のメソッド:")
    for attr in dir(bm25):
        if not attr.startswith('__') and callable(getattr(bm25, attr)):
            print(f"{attr}")
    
    # Test persistence parameters
    # 永続化パラメータをテスト
    print("\nTesting state information for persistence:")
    print("永続化のための状態情報をテスト:")
    
    # Dictionary to store state
    # 状態を格納する辞書
    state = {}
    
    # Try to access attributes typically used for persistence
    # 永続化に通常使用される属性にアクセスを試みる
    for attr in ['corpus', 'corpus_size', 'avg_doc_len', 'avgdl', 'doc_freqs', 
                 'doc_lens', 'idf', 'k1', 'b', 'doc_len', 'doc_freq', 'f', 'df']:
        try:
            value = getattr(bm25, attr)
            state[attr] = type(value).__name__
            print(f"{attr}: {type(value).__name__}")
        except AttributeError:
            print(f"{attr}: Not found")

if __name__ == "__main__":
    inspect_bm25() 