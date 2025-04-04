# 🇯🇵 FindaLedge: RAGのためのシンプルなアンサンブル検索ライブラリ 🔍

**FindaLedge** は、Retrieval-Augmented Generation (RAG) アプリケーション向けに、**アンサンブル検索システム**の構築と管理を簡素化するために設計されたPythonライブラリです。ベクトル検索とキーワード検索を巧みに組み合わせ、複数のドキュメントストア（Chroma, FAISS, BM25など）のセットアップ、ドキュメントの取り込み、インデックス管理、検索結果のマージ（RRFを使用）といった複雑さを抽象化します。

✨ **強力なRAG検索バックエンドを簡単に構築！** ✨

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/findaledge.svg)](https://badge.fury.io/py/findaledge) <!-- PyPI名が一致すると仮定 -->

---

[🇬🇧 Click here for English README (英語版 README はこちら)](README.md)

---

## 🤔 なぜ FindaLedge？

RAGシステムは、単一の検索方法だけでは苦戦することがよくあります。ベクトル検索は意味的な類似性に優れていますが、特定のキーワードや略語を見逃す可能性があります。キーワード検索は完全一致を見つけますが、意図を理解することはできません。

両方を組み合わせた**アンサンブル検索**は、より高い精度を提供します。しかし、その構築には以下の要素を扱う必要があります：

*   複数のデータベース（Chroma/FAISSのようなベクトルDB、BM25のようなキーワードインデックス）
*   ドキュメントの読み込み、解析、チャンキングのパイプライン
*   埋め込み（Embedding）の生成と管理
*   ストア間での追加/削除の同期
*   複雑な結果マージロジック（Reciprocal Rank Fusion - RRF など）

🤯 **扱うのが大変です！**

**FindaLedge** は、このすべての面倒な作業を**シンプルで統一されたインターフェース**で引き受けます。あなたはLLMアプリケーションに集中でき、FindaLedgeが検索バックエンドを管理します。

## ✨ 主な特徴

*   **🎯 簡単なアンサンブル検索:** ハイブリッド検索（ベクトル + キーワード）のパワーをすぐに利用可能。
*   **🔌 柔軟なコンポーネント:** 環境変数や引数を通じて、ベクトルストア (`Chroma`, `FAISS`)、キーワードストア (`BM25`)、埋め込みモデル (`OpenAI`, `Ollama`, `SentenceTransformers`) を簡単に設定。
*   **📚 手間いらずのドキュメント取り込み:** ファイルやディレクトリ全体（`.md`, `.txt`, `.pdf`, `.docx` など）から、単一のコマンド (`add_document`) でドキュメントを追加。自動解析とチャンキングも含まれます！
*   **🔄 自動インデックス作成と永続化:** インデックス（ベクトルとキーワード）は自動的に作成、更新され、ローカルに保存されます。次回の実行時には即座にロード！
*   **🥇 スマートな結果マージ:** デフォルトで Reciprocal Rank Fusion (RRF) を使用し、異なる検索方法からの結果を賢く組み合わせてランク付け。
*   **🔗 シームレスなLangChain連携:** FindaLedgeをLangChainの `BaseRetriever` として直接使用 (`as_retriever()`) し、既存のチェーン（`RetrievalQA`など）に組み込み可能。
*   **🧹 シンプルな追加/削除:** 新しいドキュメントを簡単に追加したり、既存のドキュメントを削除 (`remove_document`) したりでき、インデックスも自動で更新。

## ⚙️ サポート環境

*   **Python:** 3.10 以上
*   **OS:** Windows, macOS, Linux でテスト済み
*   **依存関係:** `pyproject.toml` を参照。主要なものには `langchain`, `bm25s-j`, `numpy`, `chromadb` (オプション), `faiss-cpu` (オプション), `sentence-transformers` (オプション), `openai` (オプション), `ollama` (オプション) が含まれます。

## 🚀 はじめに

### 1. インストール

```bash
pip install findaledge

# または、FAISSなどのオプション依存関係を使用する場合:
# pip install findaledge[faiss] # 例、pyproject.tomlに基づいて調整
```

選択したベクトルストアと埋め込みモデルに必要な依存関係（例：`pip install chromadb openai sentence-transformers`）があることを確認してください。

### 2. 設定（環境変数）

環境変数を設定します（オプション、デフォルト値が提供されます）：

```bash
export OPENAI_API_KEY="your-openai-key" # OpenAI埋め込みを使用する場合
export FINDALEDGE_PERSIST_DIRECTORY="./my_findaledge_data" # インデックスデータの保存場所
export FINDALEDGE_VECTOR_STORE_PROVIDER="chroma"       # "chroma" または "faiss"
export FINDALEDGE_EMBEDDING_PROVIDER="openai"          # "openai", "ollama", "huggingface"
export FINDALEDGE_EMBEDDING_MODEL_NAME="text-embedding-3-small" # モデル名
# export OLLAMA_BASE_URL="http://localhost:11434" # Ollamaを使用する場合
```

### 3. 基本的な使い方

```python
from findaledge import FindaLedge
from langchain_core.documents import Document

# FindaLedgeを初期化（渡されない場合は環境変数またはデフォルトを使用）
# persist_directoryから既存のデータを自動ロードするか、新しいインデックスを作成します。
ledge = FindaLedge(
    # ここで環境変数を上書きできます、例：
    # persist_directory="./custom_data",
    # vector_store_provider="faiss"
)

# --- ドキュメントの追加 ---

# ファイルパスから追加（拡張子に基づいて自動解析）
print("ファイルを追加中...")
ledge.add_document("path/to/your/report.md")

# ディレクトリから互換性のあるすべてのファイルを追加（再帰的）
print("ディレクトリを追加中...")
ledge.add_document("path/to/your/docs_folder/")

# LangChainのDocumentオブジェクトを直接追加
print("Documentオブジェクトを追加中...")
doc = Document(page_content="これはサンプルドキュメントです。", metadata={"source": "manual", "id": "manual-doc-1"})
ledge.add_document(doc)

print("ドキュメントが追加されました！")

# --- ドキュメントの検索 ---

query = "レポートの主なトピックは何ですか？"
print(f"\n検索中: '{query}'")

# ハイブリッド検索（デフォルト）
results = ledge.search(query, top_k=3)

print("\n検索結果:")
for i, doc in enumerate(results):
    print(f"{i+1}. スコア: {doc.metadata.get('relevance_score', 0):.4f} | ソース: {doc.metadata.get('source', 'N/A')}")
    # print(f"   コンテンツ: {doc.page_content[:150]}...") # コンテンツを表示する場合はコメント解除

# --- LangChain Retriever として使用 ---

print("\nLangChain Retrieverとして使用中...")
retriever = ledge.as_retriever(search_mode="hybrid", top_k=5)

# このretrieverを任意のLangChainコンポーネントで使用します、例：RetrievalQA
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI # 例：LLM

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=retriever,
    return_source_documents=True
)

response = qa_chain.invoke({"query": query})
print("\nLangChain QA 回答:", response["result"])

# --- ドキュメントの削除（例） ---
# 親ドキュメントIDが必要です。多くの場合、ソースパスまたは提供したIDです。
# doc_id_to_remove = "manual-doc-1" # add_documentの結果またはメタデータから取得
# print(f"\nドキュメントを削除中: {doc_id_to_remove}")
# ledge.remove_document(doc_id_to_remove)
# print("ドキュメントが削除されました。")

```

## 📖 ドキュメンテーション

*   [Usage Guide (使い方ガイド)](docs/usage_guide.md)
*   [Architecture (アーキテクチャ設計書)](docs/architecture.md)
*   [Requirements (要件定義書)](docs/requirements.md)
*   [Function Specification (機能仕様書)](docs/function_spec.md)

## 🤝 コントリビューション

コントリビューションを歓迎します！ issue やプルリクエストを気軽にご提出ください。

1.  リポジトリをフォークします。
2.  新しいブランチを作成します (`git checkout -b feature/your-feature`)。
3.  変更を加えます。
4.  テストを実行します (`pytest`)。
5.  変更をコミットします (`git commit -am 'Add some feature'`)。
6.  ブランチにプッシュします (`git push origin feature/your-feature`)。
7.  新しいプルリクエストを作成します。

## 📜 ライセンス

このプロジェクトはMITライセンスの下でライセンスされています - 詳細については [LICENSE](LICENSE) ファイルを参照してください。 