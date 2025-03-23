# Implementation Plan for FinderLedge

## 概要 / Overview
- 本プロジェクトは、OpenAI Agents SDKとの連携を前提に、文書コンテキスト管理ライブラリとして、文書の自動インデックス作成、ハイブリッド検索、文書追加・削除、永続化などの機能を実現する。
- This document outlines a detailed implementation plan in a checklist format.

## ToDo チェックリスト

### 1. プロジェクト構造の整備 / Project Structure Setup
- [x] 必要なディレクトリを作成する：src, examples, tests, docs
- [x] pyproject.toml, README.md, README_ja.md, impl_plan.md を配置

### 2. Git管理の設定 / Git Version Control Setup
- [x] Gitリポジトリの初期化：`git init`
- [x] .gitignoreファイルの作成
  - [x] 仮想環境フォルダ（.venv）を除外
  - [x] Python生成ファイル（__pycache__、*.pyc、*.pyo）を除外
  - [x] ローカル設定ファイル（.env）を除外
  - [x] ビルドファイル（dist/、build/）を除外
  - [x] ChromaDBのデータディレクトリを除外（必要に応じて）
- [x] 初期ファイル構造のコミット：`git add .` と `git commit -m "Initial commit"`
- [x] ブランチ戦略の設定
  - [x] main：唯一のブランチとして使用
- [ ] リモートリポジトリとの連携（オプション）
  - [ ] GitHub/GitLabなどのリポジトリ作成
  - [ ] `git remote add origin <リポジトリURL>`
  - [ ] `git push -u origin main`

### 3. 必要ライブラリのインストール / Installation of Required Libraries
- [ ] 仮想環境が有効になっていることを確認 (.venv)
- [ ] uvコマンドで必要ライブラリをインストール
  - [ ] `uv add bm25s-j` - 日本語対応BM25検索ライブラリ
  - [ ] `uv add langchain` - LLM統合フレームワーク
  - [ ] `uv add langchain-openai` - OpenAIモデル用のLangChainラッパー
  - [ ] `uv add langchain-community` - コミュニティツール（OllamaなどのEmbeddingモデル用）
  - [ ] `uv add chromadb` - ベクトルデータベース
  - [ ] `uv add openai` - OpenAI API クライアント
  - [ ] `uv add ollama` - ローカルモデル用Ollamaクライアント（オプション）
  - [ ] `uv add numpy` - 数値計算ライブラリ（RRF実装などで必要）
  - [ ] `uv add pydantic` - データバリデーション用
- [ ] 開発モードでパッケージ自身をインストール
  - [ ] `uv pip install -e .` - 開発者モードでFinderLedgeをインストール
- [ ] 依存ライブラリが正しくインストールされているか確認
  - [ ] 各ライブラリのバージョン情報を確認

### 4. プロトタイプ作成 / Prototype Development
- [ ] studyディレクトリを作成し、コア機能の最小限サンプルを実装
  - [ ] vector_search_sample.py: ChromaDBを使用したベクトル検索の基本サンプル実装
  - [ ] bm25_sample.py: bm25s-jライブラリを使用したキーワード検索の基本サンプル実装
  - [ ] embeddings_sample.py: OpenAI APIを使用した埋め込みベクトル生成の基本サンプル実装
  - [ ] hybrid_search_sample.py: ベクトル検索とBM25検索の結果を融合するRRFアルゴリズムの実装サンプル
  - [ ] document_loader_sample.py: 様々な形式の文書を読み込み、テキスト抽出するサンプル
  - [ ] persistence_sample.py: インデックスの保存と読み込みの基本サンプル
- [ ] サンプルデータセットを用意し、各プロトタイプの動作を検証
- [ ] 各プロトタイプの実行パフォーマンスを測定し、ベンチマーク結果をまとめる
- [ ] 得られた知見を基に本実装の設計を調整

### 5. 要件定義の確認と設計ドキュメントの作成 / Requirement Analysis and Design Documentation
- [ ] docs/request.md の内容を精査して、機能要件と非機能要件を整理する
- [ ] 要件定義書(requirements.md)、アーキテクチャ設計書(architecture.md)、機能仕様書(function_spec.md)のドラフトをdocsフォルダに作成する
- [ ] PlantUMLを用いた概念的なクラス図、シーケンス図などを作成する

### 6. コアクラスの実装 / Core Module Implementation
- [ ] FinderLedge クラスの実装 (src/finderledge/finderledge.py)
  - [ ] プロパティ: db_name, persist_dir, embedding_model, vector_store, bm25_index, documents を定義
  - [ ] メソッド: __init__, add_document, remove_document, find_related, get_context, persist, close, as_retriever を実装
- [ ] Document クラスの実装 (src/finderledge/document.py)
  - [ ] 文書ID、タイトル、テキスト本文、メタデータ等を保持する
- [ ] EmbeddingModel インターフェースの定義と、OpenAIやHuggingFaceなどの具体実装のスタブ作成
- [ ] VectorStore の統合：Chromaなどを用いたベクトルデータベースの操作実装
- [ ] BM25Index の統合：bm25s-jライブラリを利用したインデックス作成と検索機能の実装
- [ ] Retriever インターフェースまたはそのラッパークラスの作成

### 7. 検索機能の実装 / Search Functionality Implementation
- [ ] ハイブリッド検索機能の実装
  - [ ] Embedding (ベクトル)検索の実装
  - [ ] BM25 キーワード検索の実装
  - [ ] 両結果の統合: Reciprocal Rank Fusion (RRF) や加重平均方式の実装
- [ ] 検索モード (hybrid, semantic, keyword) の切り替え機能の実装

### 8. 文書追加・削除機能 / Document Management
- [ ] add_document メソッドの実装：文書パース、Embedding計算、BM25インデックス更新処理
- [ ] remove_document メソッドの実装：指定文書の削除とインデックスの差分更新

### 9. インデックスとキャッシュの永続化 / Index Persistence
- [ ] persist() メソッドの自動呼び出し実装
- [ ] vector_store と BM25インデックスのディスク保存機能の実装
- [ ] ロード機能の実装：既存インデックスの再利用

### 10. OpenAI Agents SDK との連携 / Integration with OpenAI Agents SDK
- [ ] エージェントのツール関数として利用できるよう、function_tool デコレータを用いた関数の実装例を作成
- [ ] コンテキスト (RunContext) 統合のためのラッパー、型指定などの実装

### 11. テストの作成 / Testing
- [ ] コア機能のユニットテストを tests フォルダに実装
- [ ] 検索API、文書追加・削除機能の統合テストを実装
- [ ] pytest を用いて全テストを実行し、カバレッジを確認

### 12. ドキュメントとコメントの整備 / Documentation and Comments
- [ ] 全クラスおよびメソッドに対し、英語と日本語のバイリンガルコメントを記述 (docstrings)
- [ ] README.md (英語) と README_ja.md (日本語) の内容を整備
- [ ] 設計ドキュメント (requirements.md, architecture.md, function_spec.md) の作成

### 13. 最終確認とデバッグ / Final Verification and Debugging
- [ ] 全テストを .venv 内で実行し、動作確認
- [ ] コードレビューを実施し、単一責任原則およびレイヤーアーキテクチャに沿っているか検証
- [ ] Agents SDK との連携動作確認
- [ ] ドキュメントの最終チェックおよび整備

## スケジュール / Timeline (Tentative)
- 初期セットアップとライブラリインストール: 約2日
- プロトタイプ実装と検証: 約5日
- コアモジュール実装: 約2週間
- 検索機能・永続化機能の統合: 約1～2週間
- Agents SDK 連携とテスト: 約1週間
- ドキュメント整備と最終レビュー: 約1週間 