# FinderLedge

FinderLedgeは、OpenAI Agents SDKと連携する文書コンテキスト管理ライブラリです。エージェントに動的な文書コンテキストを提供し、関連情報の検索と取得を効率化します。

## 特徴

- **文書のインポートと自動インデックス作成**: 様々な形式の文書を読み込み、自動的にインデックスを作成
- **ハイブリッド検索機能**: ベクトル検索（意味的類似性）とキーワード検索（BM25）を組み合わせた高精度な検索
- **ディレクトリ全体のデータベース作成**: フォルダ内の全文書を一括でインデックス化
- **多様な文書形式対応**: テキスト、PDF、Word、Markdownなど様々な形式に対応
- **埋め込みベクトル類似度計算**: OpenAIやその他の埋め込みモデルを使用した意味的検索
- **高性能BM25検索**: キーワードベースの検索アルゴリズムによる関連文書の特定
- **インデックスの永続化とキャッシュ**: 一度作成したインデックスを保存して再利用可能
- **シンプルな検索API**: 直感的に使える検索インターフェース
- **OpenAI Agents SDKとの統合**: エージェントツールやコンテキストプロバイダーとして利用可能
- **SDK非依存の使用も可能**: 単独でも利用可能な設計

## インストール

```bash
pip install finderledge
```

## 基本的な使い方

```python
from finderledge import FinderLedge

# インスタンス作成
ledge = FinderLedge(db_name="my_documents")

# 文書の追加
ledge.add_document("path/to/document.pdf")
ledge.add_directory("path/to/document_folder")

# 文書の検索
results = ledge.find_related("クエリテキスト", mode="hybrid")

# コンテキストの取得（OpenAI Agents SDK向け）
context = ledge.get_context("クエリテキスト")

# 使用後のリソース解放
ledge.close()
```

## OpenAI Agents SDKとの連携

```python
from openai import OpenAI
from finderledge import FinderLedge

# FinderLedgeのインスタンス作成
ledge = FinderLedge(db_name="knowledge_base")
ledge.add_directory("path/to/documents")

# ツールとして登録
@function_tool
def search_docs(query: str) -> str:
    results = ledge.find_related(query)
    return "\n\n".join([r.page_content for r in results])

# エージェント作成とツール登録
client = OpenAI()
assistant = client.beta.assistants.create(
    name="Document Assistant",
    instructions="You help users find information in documents.",
    model="gpt-4-turbo",
    tools=[search_docs.openai_schema],
)
```

## 動作環境

- Python 3.9以上
- Windows/macOS/Linux対応

## ライセンス

MIT

## 開発者向け情報

開発環境のセットアップ:

```bash
git clone https://github.com/yourusername/finderledge.git
cd finderledge
python -m venv .venv
source .venv/bin/activate  # Windowsの場合: .venv\Scripts\activate
pip install -e ".[dev]"
``` 