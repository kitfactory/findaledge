# FinderLedge 使い方ガイド

## 概要
FinderLedgeは、OpenAI Agents SDKと連携して文書コンテキストを管理するためのPythonライブラリです。文書の自動インデックス作成、ハイブリッド検索（ベクトル検索とBM25検索の組み合わせ）、文書の追加・削除、永続化などの機能を提供します。

## インストール
```bash
pip install finderledge
```

## 基本的な使い方

### 1. 初期化
```python
from finderledge import FinderLedge

# FinderLedgeのインスタンスを作成
finder = FinderLedge(
    db_name="my_documents",  # データベース名
    persist_dir="data",      # データの保存ディレクトリ
    chunk_size=1000,         # テキストチャンクのサイズ
    chunk_overlap=200        # チャンク間の重複
)
```

### 2. 文書の追加
```python
# テキストファイルの追加
finder.add_document(
    file_path="path/to/document.txt",
    title="文書タイトル",
    metadata={"author": "著者名", "date": "2024-03-26"}
)

# テキストの直接追加
finder.add_text(
    text="追加するテキスト",
    title="文書タイトル",
    metadata={"source": "手動入力"}
)

# ディレクトリ内の文書を一括追加
finder.add_directory(
    directory_path="path/to/documents",
    file_pattern="*.txt",  # ファイルパターン（オプション）
    recursive=True        # サブディレクトリも含める（オプション）
)
```

### 3. 文書の検索
```python
# ハイブリッド検索（デフォルト）
results = finder.search(
    query="検索クエリ",
    top_k=5,              # 返す結果の数
    search_mode="hybrid"  # 検索モード: "hybrid", "semantic", "keyword"
)

# 検索結果の処理
for result in results:
    print(f"スコア: {result.score}")
    print(f"タイトル: {result.title}")
    print(f"テキスト: {result.text}")
    print(f"メタデータ: {result.metadata}")
```

### 4. 文書の削除
```python
# 文書IDによる削除
finder.remove_document(document_id="doc_id")

# タイトルによる削除
finder.remove_document_by_title(title="文書タイトル")
```

### 5. 文書の取得
```python
# 文書IDによる取得
doc = finder.get_document(document_id="doc_id")

# タイトルによる取得
doc = finder.get_document_by_title(title="文書タイトル")

# 全文書の取得
documents = finder.get_all_documents()
```

### 6. インデックスの永続化
```python
# インデックスの保存
finder.persist()

# インデックスの読み込み（初期化時に自動的に行われます）
finder.load()
```

## 高度な使い方

### 1. カスタム埋め込みモデルの使用
```python
from finderledge.embedding_model import OpenAIEmbeddingModel

# カスタム埋め込みモデルの作成
embedding_model = OpenAIEmbeddingModel(model="text-embedding-3-small")

# FinderLedgeの初期化時に使用
finder = FinderLedge(
    embedding_model=embedding_model,
    db_name="my_documents"
)
```

### 2. 検索モードの切り替え
```python
# セマンティック検索（ベクトル検索のみ）
results = finder.search(
    query="検索クエリ",
    search_mode="semantic"
)

# キーワード検索（BM25検索のみ）
results = finder.search(
    query="検索クエリ",
    search_mode="keyword"
)
```

### 3. メタデータによるフィルタリング
```python
# メタデータでフィルタリングして検索
results = finder.search(
    query="検索クエリ",
    metadata_filter={"author": "著者名", "date": "2024-03-26"}
)
```

### 4. 文書の更新
```python
# 文書の内容を更新
finder.update_document(
    document_id="doc_id",
    text="新しいテキスト",
    metadata={"updated": "2024-03-26"}
)
```

## 注意事項

1. OpenAI APIの使用
   - FinderLedgeはOpenAI APIを使用してテキストの埋め込みを生成します
   - 環境変数`OPENAI_API_KEY`を設定する必要があります
   - APIの使用には料金が発生します

2. データの永続化
   - デフォルトでは`data`ディレクトリにデータが保存されます
   - 重要なデータは定期的にバックアップすることをお勧めします

3. メモリ使用量
   - 大量の文書を扱う場合は、適切なチャンクサイズを設定してください
   - 必要に応じて`chunk_size`と`chunk_overlap`を調整してください

4. 検索の精度
   - ハイブリッド検索は、セマンティック検索とキーワード検索の両方の利点を活かします
   - 用途に応じて適切な検索モードを選択してください

## エラーハンドリング

```python
try:
    # 文書の追加
    finder.add_document(file_path="path/to/document.txt")
except FileNotFoundError:
    print("ファイルが見つかりません")
except ValueError as e:
    print(f"無効な入力: {e}")
except Exception as e:
    print(f"予期せぬエラー: {e}")
```

## パフォーマンス最適化

1. チャンクサイズの調整
   - 小さいチャンク: より細かい検索が可能
   - 大きいチャンク: 処理が高速

2. インデックスの最適化
   - 定期的に`persist()`を呼び出してインデックスを保存
   - 不要な文書は`remove_document()`で削除

3. 検索モードの選択
   - セマンティック検索: 意味的な類似性の検索
   - キーワード検索: 高速な検索
   - ハイブリッド検索: バランスの取れた検索 

## OpenAI Agents SDKとの連携

### 1. 基本的な連携方法
```python
from openai.agents import Agent
from finderledge import FinderLedge

# FinderLedgeのインスタンスを作成
finder = FinderLedge(db_name="my_documents")

# エージェントの作成
agent = Agent(
    name="document_assistant",
    description="文書検索と分析を行うアシスタント",
    tools=[
        finder.search,  # 検索機能をツールとして追加
        finder.add_document,  # 文書追加機能をツールとして追加
        finder.remove_document  # 文書削除機能をツールとして追加
    ]
)

# エージェントとの対話
response = agent.chat(
    "プロジェクトの要件文書を探して、主要な要件をまとめてください。"
)
print(response)
```

### 2. カスタムツールの作成
```python
from openai.agents import Tool
from finderledge import FinderLedge

# FinderLedgeのインスタンスを作成
finder = FinderLedge(db_name="my_documents")

# カスタムツールの作成
def search_documents(query: str, top_k: int = 5) -> str:
    """
    Search for documents and return formatted results
    文書を検索して結果を整形して返す

    Args:
        query (str): Search query / 検索クエリ
        top_k (int): Number of results to return / 返す結果の数

    Returns:
        str: Formatted search results / 整形された検索結果
    """
    results = finder.search(query=query, top_k=top_k)
    formatted_results = []
    for result in results:
        formatted_results.append(
            f"タイトル: {result.title}\n"
            f"スコア: {result.score}\n"
            f"テキスト: {result.text}\n"
            f"メタデータ: {result.metadata}\n"
            "---"
        )
    return "\n".join(formatted_results)

# ツールの登録
search_tool = Tool(
    name="search_documents",
    description="文書を検索して結果を整形して返します",
    function=search_documents
)

# エージェントの作成
agent = Agent(
    name="document_assistant",
    description="文書検索と分析を行うアシスタント",
    tools=[search_tool]
)
```

### 3. コンテキスト管理との連携
```python
from openai.agents import Agent, RunContext
from finderledge import FinderLedge

# FinderLedgeのインスタンスを作成
finder = FinderLedge(db_name="my_documents")

# コンテキストの作成
context = RunContext(
    metadata={
        "project": "プロジェクト名",
        "user": "ユーザー名",
        "timestamp": "2024-03-26"
    }
)

# エージェントの作成
agent = Agent(
    name="document_assistant",
    description="文書検索と分析を行うアシスタント",
    tools=[finder.search],
    context=context
)

# コンテキストを考慮した検索
response = agent.chat(
    "このプロジェクトの要件文書を探して、主要な要件をまとめてください。"
)
```

### 4. エラーハンドリング
```python
from openai.agents import Agent
from finderledge import FinderLedge

# FinderLedgeのインスタンスを作成
finder = FinderLedge(db_name="my_documents")

# エラーハンドリング付きのツール
def safe_search(query: str, top_k: int = 5) -> str:
    """
    Safely search for documents with error handling
    エラーハンドリング付きで文書を安全に検索

    Args:
        query (str): Search query / 検索クエリ
        top_k (int): Number of results to return / 返す結果の数

    Returns:
        str: Formatted search results or error message / 整形された検索結果またはエラーメッセージ
    """
    try:
        results = finder.search(query=query, top_k=top_k)
        if not results:
            return "検索結果が見つかりませんでした。"
        
        formatted_results = []
        for result in results:
            formatted_results.append(
                f"タイトル: {result.title}\n"
                f"スコア: {result.score}\n"
                f"テキスト: {result.text}\n"
                "---"
            )
        return "\n".join(formatted_results)
    except Exception as e:
        return f"検索中にエラーが発生しました: {str(e)}"

# ツールの登録
search_tool = Tool(
    name="safe_search",
    description="エラーハンドリング付きで文書を安全に検索します",
    function=safe_search
)

# エージェントの作成
agent = Agent(
    name="document_assistant",
    description="文書検索と分析を行うアシスタント",
    tools=[search_tool]
)
```

### 5. 非同期処理のサポート
```python
from openai.agents import Agent
from finderledge import FinderLedge
import asyncio

# FinderLedgeのインスタンスを作成
finder = FinderLedge(db_name="my_documents")

# 非同期ツールの作成
async def async_search(query: str, top_k: int = 5) -> str:
    """
    Asynchronously search for documents
    非同期で文書を検索

    Args:
        query (str): Search query / 検索クエリ
        top_k (int): Number of results to return / 返す結果の数

    Returns:
        str: Formatted search results / 整形された検索結果
    """
    # 非同期処理をシミュレート
    await asyncio.sleep(1)
    results = finder.search(query=query, top_k=top_k)
    return "\n".join([f"タイトル: {r.title}\nテキスト: {r.text}" for r in results])

# ツールの登録
search_tool = Tool(
    name="async_search",
    description="非同期で文書を検索します",
    function=async_search,
    is_async=True
)

# エージェントの作成
agent = Agent(
    name="document_assistant",
    description="文書検索と分析を行うアシスタント",
    tools=[search_tool]
)

# 非同期での実行
async def main():
    response = await agent.achat(
        "プロジェクトの要件文書を探して、主要な要件をまとめてください。"
    )
    print(response)

# 非同期処理の実行
asyncio.run(main())
```

### 6. ベストプラクティス

1. ツールの設計
   - 各ツールは単一の責務を持つように設計
   - エラーハンドリングを適切に実装
   - 非同期処理が必要な場合は`is_async=True`を設定

2. コンテキストの活用
   - プロジェクトやユーザー情報をコンテキストに含める
   - 検索結果のフィルタリングにコンテキストを活用

3. パフォーマンスの考慮
   - 大量の文書を扱う場合は非同期処理を検討
   - 検索結果のキャッシュを実装

4. セキュリティ
   - 機密情報はコンテキストに含めない
   - ユーザー認証を適切に実装
 