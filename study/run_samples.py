"""
Run Samples CLI Tool
サンプル実行CLIツール

This tool provides a simple command-line interface to run the sample scripts.
このツールはサンプルスクリプトを実行するためのシンプルなコマンドラインインターフェースを提供します。
"""

import os
import sys
import argparse
from oneenv import load_dotenv

# Import sample modules with individual error handling
# サンプルモジュールを個別のエラーハンドリングでインポート
vector_search_sample = None
bm25_search_sample = None
embeddings_sample = None
hybrid_search_sample = None
document_loader_sample = None
openai_chat_sample = None
summarize_document_sample = None
persistence_sample = None

try:
    from vector_search_sample import vector_search_sample
except ImportError:
    print("Warning: Could not import vector_search_sample")
    print("警告: vector_search_sampleをインポートできませんでした")

try:
    from bm25_sample import bm25_search_sample
except ImportError:
    print("Warning: Could not import bm25_search_sample")
    print("警告: bm25_search_sampleをインポートできませんでした")

try:
    from embeddings_sample import embeddings_sample
except ImportError:
    print("Warning: Could not import embeddings_sample")
    print("警告: embeddings_sampleをインポートできませんでした")

try:
    from hybrid_search_sample import hybrid_search_sample
except ImportError:
    print("Warning: Could not import hybrid_search_sample")
    print("警告: hybrid_search_sampleをインポートできませんでした")

try:
    from document_loader_sample import document_loader_sample
except ImportError:
    print("Warning: Could not import document_loader_sample")
    print("警告: document_loader_sampleをインポートできませんでした")

try:
    from openai_sample import openai_chat_sample, summarize_document_sample
except ImportError:
    openai_chat_sample = lambda: print("OpenAI chat sample not available")
    summarize_document_sample = lambda: print("Document summarization sample not available")

try:
    from persistence_sample import persistence_sample
except ImportError:
    print("Warning: Could not import persistence_sample")
    print("警告: persistence_sampleをインポートできませんでした")

def print_header():
    """
    Print a header for the tool
    ツールのヘッダーを表示
    """
    print("\n" + "=" * 60)
    print("  FinderLedge Sample Runner")
    print("  FinderLedge サンプル実行ツール")
    print("=" * 60)

def list_samples():
    """
    List all available samples
    利用可能なすべてのサンプルを一覧表示
    """
    samples = {
        "1": {
            "name": "Vector Search Sample",
            "description": "Demonstrates vector search using ChromaDB",
            "name_ja": "ベクトル検索サンプル",
            "description_ja": "ChromaDBを使用したベクトル検索のデモンストレーション",
            "function": vector_search_sample
        },
        "2": {
            "name": "BM25 Search Sample",
            "description": "Demonstrates keyword search using BM25",
            "name_ja": "BM25検索サンプル",
            "description_ja": "BM25を使用したキーワード検索のデモンストレーション",
            "function": bm25_search_sample
        },
        "3": {
            "name": "Embeddings Sample",
            "description": "Demonstrates creating and using embeddings",
            "name_ja": "埋め込みベクトルサンプル",
            "description_ja": "埋め込みベクトルの作成と使用のデモンストレーション",
            "function": embeddings_sample
        },
        "4": {
            "name": "Hybrid Search Sample",
            "description": "Demonstrates hybrid search combining vector and keyword search",
            "name_ja": "ハイブリッド検索サンプル",
            "description_ja": "ベクトル検索とキーワード検索を組み合わせたハイブリッド検索のデモンストレーション",
            "function": hybrid_search_sample
        },
        "5": {
            "name": "Document Loader Sample",
            "description": "Demonstrates loading documents from various formats",
            "name_ja": "ドキュメントローダーサンプル",
            "description_ja": "様々な形式の文書を読み込むデモンストレーション",
            "function": document_loader_sample
        },
        "6": {
            "name": "OpenAI Chat Sample",
            "description": "Demonstrates chat interaction with OpenAI API",
            "name_ja": "OpenAIチャットサンプル",
            "description_ja": "OpenAI APIを使用したチャットインタラクションのデモンストレーション",
            "function": openai_chat_sample
        },
        "7": {
            "name": "Document Summarization Sample",
            "description": "Demonstrates document summarization using OpenAI API",
            "name_ja": "文書要約サンプル",
            "description_ja": "OpenAI APIを使用した文書要約のデモンストレーション",
            "function": summarize_document_sample
        },
        "8": {
            "name": "Index Persistence Sample",
            "description": "Demonstrates persisting and reloading search indices",
            "name_ja": "インデックス永続化サンプル",
            "description_ja": "検索インデックスの永続化と再読み込みのデモンストレーション",
            "function": persistence_sample
        }
    }
    
    print("\nAvailable Samples / 利用可能なサンプル:")
    print("-" * 60)
    for key, sample in samples.items():
        print(f"{key}. {sample['name']} / {sample['name_ja']}")
        print(f"   {sample['description']}")
        print(f"   {sample['description_ja']}")
        print()
    
    return samples

def run_sample(sample_id, samples):
    """
    Run the selected sample
    選択されたサンプルを実行
    
    Args:
        sample_id (str): The ID of the sample to run
        samples (dict): Dictionary of available samples
    """
    if sample_id in samples:
        sample = samples[sample_id]
        print(f"\nRunning: {sample['name']} / {sample['name_ja']}")
        print("-" * 60)
        
        # Load environment variables
        # 環境変数をロード
        load_dotenv()
        
        # Run the sample function
        # サンプル関数を実行
        try:
            sample["function"]()
        except Exception as e:
            print(f"\nError running sample: {str(e)}")
            print(f"サンプル実行エラー: {str(e)}")
    else:
        print(f"Invalid sample ID: {sample_id}")
        print(f"無効なサンプルID: {sample_id}")

def run_interactive():
    """
    Run the tool in interactive mode
    対話モードでツールを実行
    """
    print_header()
    samples = list_samples()
    
    while True:
        print("\nEnter the number of the sample to run, 'list' to show samples again, or 'exit' to quit:")
        print("実行するサンプルの番号を入力するか、'list'で再度サンプル一覧を表示、'exit'で終了します:")
        choice = input("> ").strip().lower()
        
        if choice == "exit" or choice == "quit":
            print("\nExiting sample runner. Goodbye!")
            print("サンプル実行ツールを終了します。さようなら！")
            break
        elif choice == "list":
            list_samples()
        elif choice in samples:
            run_sample(choice, samples)
        else:
            print("Invalid choice. Please try again.")
            print("無効な選択です。もう一度お試しください。")

def main():
    """
    Main function to run the sample runner tool
    サンプル実行ツールを実行するメイン関数
    """
    parser = argparse.ArgumentParser(description="Run FinderLedge sample scripts")
    parser.add_argument("sample_id", nargs="?", type=str,
                      help="ID of the sample to run (if not provided, runs in interactive mode)")
    
    args = parser.parse_args()
    
    if args.sample_id:
        # Run the specified sample
        # 指定されたサンプルを実行
        print_header()
        samples = list_samples()
        run_sample(args.sample_id, samples)
    else:
        # Run in interactive mode
        # 対話モードで実行
        run_interactive()

if __name__ == "__main__":
    main() 