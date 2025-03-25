# Finderledge 🔍

埋め込みとBM25を使用した、効率的で正確な文書検索ライブラリ。

## 機能 ✨

- チャンク分割をサポートした文書管理
- 埋め込みとBM25を使用したハイブリッド検索
- 効率的な文書埋め込みの保存と取得
- 設定可能なトークン化とテキスト処理
- 使いやすい文書検索API

## インストール 🚀

```bash
pip install finderledge
```

## クイックスタート 🎯

```python
from finderledge import Document, DocumentStore, EmbeddingStore, EmbeddingModel, Tokenizer, BM25, Finder

# コンポーネントの初期化
document_store = DocumentStore("documents")
embedding_store = EmbeddingStore("embeddings")
embedding_model = EmbeddingModel()
tokenizer = Tokenizer()
bm25 = BM25()

# Finderの作成
finder = Finder(
    document_store=document_store,
    embedding_store=embedding_store,
    embedding_model=embedding_model,
    tokenizer=tokenizer,
    bm25=bm25
)

# 文書の追加
doc = Document(
    id="doc1",
    title="サンプル文書",
    content="これはテスト用のサンプル文書です。",
    metadata={"author": "山田太郎"}
)
finder.add_document(doc)

# 文書の検索
results = finder.search("サンプル文書", top_k=5)
for doc, score in results:
    print(f"文書: {doc.title}, スコア: {score}")
```

## ドキュメント 📚

詳細なドキュメントについては、[ドキュメントページ](https://finderledge.readthedocs.io/)をご覧ください。

## 開発 🛠️

1. リポジトリのクローン
2. 仮想環境の作成
3. 開発用依存関係のインストール
4. テストの実行

```bash
git clone https://github.com/yourusername/finderledge.git
cd finderledge
python -m venv .venv
source .venv/bin/activate  # Windowsの場合: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## 貢献 🤝

貢献を歓迎します！プルリクエストをお気軽に送信してください。

## ライセンス 📄

このプロジェクトはMITライセンスの下でライセンスされています - 詳細については[LICENSE](LICENSE)ファイルをご覧ください。

## サポート 💬

ご質問やお困りの点がございましたら、issueを作成するか、support@finderledge.comまでご連絡ください。 