# FinderLedge 使い方ガイド

## 概要
FinderLedgeは、ファイルやディレクトリからドキュメントを読み込み、ベクトル検索とキーワード検索を組み合わせたハイブリッド検索を容易に実現するためのPythonライブラリです。LangChainとの連携もスムーズに行えます。

## インストール
```bash
pip install finderledge
```

(または、プロジェクトの依存関係に追加してください)

## 基本的な使い方

### 1. 初期化
`FinderLedge` を利用する前に、環境変数または `FinderLedge` のコンストラクタ引数で設定を行います。

**環境変数での設定例:**
```bash
export OPENAI_API_KEY="your-api-key" # OpenAI を使う場合
export FINDERLEDGE_PERSIST_DIRECTORY="./finder_data" # データ保存場所 (デフォルト)
export FINDERLEDGE_VECTOR_STORE_PROVIDER="chroma"  # ベクトルストア (chroma or faiss)
export FINDERLEDGE_EMBEDDING_PROVIDER="openai" # 埋め込みモデル (openai, ollama, huggingface)
export FINDERLEDGE_EMBEDDING_MODEL_NAME="text-embedding-3-small" # モデル名
```

**Pythonコードでの初期化例:**
```python
from finderledge import FinderLedge

# 環境変数またはデフォルト値を使用
ledge = FinderLedge()

# または、引数で設定を上書き
ledge = FinderLedge(
    persist_directory="./my_index_data",
    vector_store_provider="faiss",
    embedding_provider="huggingface",
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=500,
    chunk_overlap=50
)
```

*   `persist_directory`: インデックスデータが保存されるディレクトリです。
*   `vector_store_provider`: `chroma` または `faiss` を選択できます。
*   `keyword_store_provider`: 現在は `bm25` のみサポートされています。
*   `embedding_provider`: `openai`, `ollama`, `huggingface` から選択できます。
*   `embedding_model_name`: 選択したプロバイダーに対応するモデル名を指定します。
*   `chunk_size`, `chunk_overlap`: ドキュメント分割時のチャンクサイズとオーバーラップを指定します。

### 2. ドキュメントの追加
ファイルパス、ディレクトリパス、または LangChain の `Document` オブジェクト（単一またはリスト）を指定してドキュメントを追加します。内部で自動的にテキスト抽出、分割、インデックス作成（ベクトルストアとキーワードストアの両方）が行われます。

```python
from langchain_core.documents import Document

# ファイルパスから追加
parent_ids_file = ledge.add_document("path/to/your/document.md")
print(f"追加されたドキュメント (ファイル): {parent_ids_file}")

# ディレクトリパスから追加 (再帰的に .txt ファイルを検索)
parent_ids_dir = ledge.add_document("path/to/your/docs_folder")
print(f"追加されたドキュメント (ディレクトリ): {parent_ids_dir}")

# Document オブジェクトから追加
doc = Document(page_content="これはテストドキュメントです。", metadata={"source": "manual", "category": "test"})
parent_ids_obj = ledge.add_document(doc)
print(f"追加されたドキュメント (オブジェクト): {parent_ids_obj}")

# Document リストから追加
docs = [
    Document(page_content="ドキュメント1の内容", metadata={"id": "doc1"}),
    Document(page_content="ドキュメント2の内容", metadata={"id": "doc2"})
]
parent_ids_list = ledge.add_document(docs)
print(f"追加されたドキュメント (リスト): {parent_ids_list}")
```
*   `add_document` は、追加された元のドキュメントに対応する（親）IDのリストを返します。
*   ファイルやディレクトリを追加する場合、`DocumentLoader` が内部で使用され、可能なファイル形式が自動的に処理されます。
*   `Document` オブジェクトを直接渡す場合、その `metadata` が保持されます。

### 3. ドキュメントの検索
クエリ文字列と検索モードを指定してドキュメントを検索します。

```python
# ハイブリッド検索 (デフォルト)
results_hybrid = ledge.search(
    query="ハイブリッド検索とは？",
    top_k=5
)
print("\n--- ハイブリッド検索結果 ---")
for doc in results_hybrid:
    print(f"Score: {doc.metadata.get('relevance_score', 'N/A'):.4f}, Source: {doc.metadata.get('source', 'N/A')}")
    # print(doc.page_content[:100] + "...") # 内容も表示する場合

# ベクトル検索のみ
results_vector = ledge.search(
    query="意味的に近い文書を探す",
    search_mode="vector",
    top_k=3
)
print("\n--- ベクトル検索結果 ---")
for doc in results_vector:
    print(f"Score: {doc.metadata.get('relevance_score', 'N/A'):.4f}, Source: {doc.metadata.get('source', 'N/A')}")

# キーワード検索のみ
results_keyword = ledge.search(
    query="特定のキーワードを含む文書",
    search_mode="keyword",
    top_k=3
)
print("\n--- キーワード検索結果 ---")
for doc in results_keyword:
    print(f"Score: {doc.metadata.get('relevance_score', 'N/A'):.4f}, Source: {doc.metadata.get('source', 'N/A')}")

# メタデータでフィルタリング (ベクトル検索またはハイブリッド検索時)
results_filtered = ledge.search(
    query="テストカテゴリの文書",
    search_mode="vector",
    top_k=2,
    vector_filter={"category": "test"} # Chroma/FAISS がサポートする形式で指定
)
print("\n--- フィルタリング検索結果 ---")
for doc in results_filtered:
    print(f"Score: {doc.metadata.get('relevance_score', 'N/A'):.4f}, Source: {doc.metadata.get('source', 'N/A')}, Category: {doc.metadata.get('category')}")
```
*   `search_mode`: `hybrid` (デフォルト), `vector`, `keyword` を指定できます。
*   `top_k`: 返す結果の最大数を指定します。
*   `vector_filter`: ベクトルストアでの検索時にメタデータで結果を絞り込むために使用します。指定方法はベクトルストア（Chroma, FAISS）に依存します。
*   検索結果の `Document` オブジェクトの `metadata` には、検索スコア (`relevance_score`) が含まれます。

### 4. ドキュメントの削除
追加時に返された（親）ドキュメントIDを指定して、関連するデータ（分割されたチャンクを含む）をベクトルストアとキーワードストアの両方から削除します。

```python
if parent_ids_file: # add_document の戻り値を使う例
    doc_id_to_remove = parent_ids_file[0]
    try:
        ledge.remove_document(doc_id_to_remove)
        print(f"\nドキュメント {doc_id_to_remove} を削除しました。")
    except Exception as e:
        print(f"ドキュメント削除中にエラー: {e}")
```

### 5. コンテキストの取得
検索結果を結合して、単一のテキストコンテキストとして取得します。RAG (Retrieval-Augmented Generation) などでLLMに渡すコンテキストを生成するのに便利です。

```python
context = ledge.get_context(
    query="主要な機能について教えて",
    search_mode="hybrid",
    top_k=3
)
print("\n--- 取得したコンテキスト ---")
print(context)
```

## LangChain との連携
`FinderLedge` は LangChain の `BaseRetriever` として簡単に利用できます。

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI # 例: OpenAI の LLM を使う場合

# FinderLedge を Retriever として取得
retriever = ledge.as_retriever(
    search_mode="hybrid",
    top_k=5,
    # 必要に応じて k_vector, k_keyword, vector_filter も指定可能
)

# LangChain の QA チェーンを作成
llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # または "map_reduce" など
    retriever=retriever,
    return_source_documents=True
)

# QA チェーンを実行
query = "FinderLedge の初期化方法について教えてください。"
result = qa_chain.invoke({"query": query})

print(f"\n--- LangChain QA 結果 ({query}) ---")
print("回答:", result["result"])
# print("\n参照ドキュメント:")
# for doc in result["source_documents"]:
#     print(f"- {doc.metadata.get('source', 'N/A')}")
```

## 注意事項

*   **APIキー**: `openai` プロバイダーを使用する場合、環境変数 `OPENAI_API_KEY` の設定が必要です。他のプロバイダーも同様に、必要な認証情報の設定が必要になる場合があります。
*   **永続化**: インデックスデータは `persist_directory` で指定された場所に自動的に保存・ロードされます。手動での `persist()` や `load()` の呼び出しは不要です。
*   **依存関係**: 使用するベクトルストア (`chroma`, `faiss`) や埋め込みモデルプロバイダー (`openai`, `ollama-python`, `sentence-transformers`) に応じて、追加のライブラリが必要になる場合があります。適宜インストールしてください。
*   **更新**: ドキュメントの内容を更新したい場合は、一度 `remove_document` で削除してから、新しい内容で `add_document` を実行してください。

## OpenAI Agents SDK との連携 (LangChain経由)

OpenAI Agents SDK と直接連携する機能は現在 `FinderLedge` にはありませんが、LangChain のツールとして `FinderLedge` の Retriever をラップすることで連携が可能です。

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool

# 1. FinderLedge Retriever を取得 (前の例を参照)
retriever = ledge.as_retriever(search_mode="hybrid", top_k=3)

# 2. LangChain Retriever をツール化
retriever_tool = create_retriever_tool(
    retriever,
    "finderledge_search",
    "Searches and returns relevant document excerpts from FinderLedge knowledge base."
)

tools = [retriever_tool]

# 3. OpenAI Functions Agent をセットアップ (LangChain の機能)
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful assistant that uses the FinderLedge knowledge base."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 4. エージェントを実行
response = agent_executor.invoke({"input": "FinderLedgeでドキュメントを追加する方法は？"})

print("\n--- OpenAI Agent (via LangChain Tool) Response ---")
print(response["output"])
```

この例では、LangChain の `create_retriever_tool` を使って `FinderLedge` の Retriever をエージェントが利用可能なツールに変換し、LangChain の Agent 機能と組み合わせています。
 