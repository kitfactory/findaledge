# FinderLedge 機能仕様書

## 1. 文書管理機能

### 1.1 文書インポート（UC-01）
#### ユースケース手順
1. ユーザーが文書ファイルを指定
2. システムがファイル形式を判定
3. システムが文書をパース
4. システムがテキストを抽出
5. システムがチャンクに分割
6. システムがインデックスを更新

#### ユースケースフロー図
```plantuml
@startuml
actor User
participant "UserAPIInterface" as API
participant "DocumentManager" as DM
participant "Document" as Doc
participant "FileSystem" as FS
participant "EmbeddingService" as ES
participant "IndexManager" as IM

User -> API: 文書ファイルを指定
API -> DM: 文書インポート要求
DM -> FS: ファイル読み込み
FS --> DM: ファイル内容
DM -> Doc: 文書パース
Doc -> Doc: テキスト抽出
Doc -> Doc: チャンク分割
Doc -> ES: 埋め込み計算
ES --> Doc: ベクトル
Doc -> IM: インデックス更新
IM --> DM: 更新完了
DM --> API: インポート完了
API --> User: 完了通知
@enduml
```

### 1.2 インデックス自動更新（UC-02）
#### ユースケース手順
1. システムが文書変更を検知
2. システムが変更箇所を特定
3. システムが差分を計算
4. システムがインデックスを更新
5. システムがキャッシュを更新

#### ユースケースフロー図
```plantuml
@startuml
participant "FileSystem" as FS
participant "DocumentManager" as DM
participant "Document" as Doc
participant "EmbeddingService" as ES
participant "IndexManager" as IM
participant "VectorDocumentStore" as VS
participant "BM25Index" as BM25

FS -> DM: 文書変更通知
DM -> Doc: 変更箇所特定
Doc -> Doc: 差分計算
Doc -> ES: 埋め込み更新
ES --> Doc: 新規ベクトル
Doc -> IM: インデックス更新要求
IM -> VS: ベクトル更新
IM -> BM25: キーワード更新
VS --> IM: 更新完了
BM25 --> IM: 更新完了
IM --> DM: 更新完了
@enduml
```

### 1.3 文書削除（UC-03）
#### ユースケース手順
1. ユーザーが文書を指定
2. システムが文書を検証
3. システムがインデックスから削除
4. システムがキャッシュを更新

#### ユースケースフロー図
```plantuml
@startuml
actor User
participant "UserAPIInterface" as API
participant "DocumentManager" as DM
participant "Document" as Doc
participant "IndexManager" as IM
participant "VectorDocumentStore" as VS
participant "BM25Index" as BM25

User -> API: 文書削除要求
API -> DM: 削除要求
DM -> Doc: 文書検証
Doc --> DM: 検証結果
DM -> IM: インデックス削除要求
IM -> VS: ベクトル削除
IM -> BM25: キーワード削除
VS --> IM: 削除完了
BM25 --> IM: 削除完了
IM --> DM: 削除完了
DM --> API: 削除完了
API --> User: 完了通知
@enduml
```

## 2. 検索機能

### 2.1 ハイブリッド検索（UC-04）
#### ユースケース手順
1. ユーザーが検索クエリを入力
2. システムがクエリを前処理
3. システムがベクトル検索を実行
4. システムがキーワード検索を実行
5. システムが結果を統合
6. システムがスコアリングを実行
7. システムが結果を返却

#### ユースケースフロー図
```plantuml
@startuml
actor User
participant "UserAPIInterface" as API
participant "SearchService" as SS
participant "SearchModel" as SM
participant "VectorDocumentStore" as VS
participant "BM25Index" as BM25
participant "EmbeddingService" as ES

User -> API: 検索クエリ入力
API -> SS: 検索要求
SS -> SM: クエリ前処理
SM -> ES: クエリ埋め込み
ES --> SM: クエリベクトル
SM -> VS: ベクトル検索
SM -> BM25: キーワード検索
VS --> SM: ベクトル結果
BM25 --> SM: キーワード結果
SM -> SM: 結果統合
SM -> SM: スコアリング
SM --> SS: 検索結果
SS --> API: 結果返却
API --> User: 結果表示
@enduml
```

### 2.2 検索モード選択（UC-05）
#### ユースケース手順
1. ユーザーが検索モードを選択
2. システムがモードを設定
3. システムが検索を実行
4. システムが結果を返却

#### ユースケースフロー図
```plantuml
@startuml
actor User
participant "UserAPIInterface" as API
participant "SearchService" as SS
participant "SearchModel" as SM
participant "VectorDocumentStore" as VS
participant "BM25Index" as BM25
participant "EmbeddingService" as ES

User -> API: 検索モード選択
API -> SS: モード設定
SS -> SM: モード設定
alt ハイブリッドモード
    SM -> VS: ベクトル検索
    SM -> BM25: キーワード検索
    VS --> SM: ベクトル結果
    BM25 --> SM: キーワード結果
    SM -> SM: 結果統合
else セマンティックモード
    SM -> VS: ベクトル検索
    VS --> SM: ベクトル結果
else キーワードモード
    SM -> BM25: キーワード検索
    BM25 --> SM: キーワード結果
end
SM -> SM: スコアリング
SM --> SS: 検索結果
SS --> API: 結果返却
API --> User: 結果表示
@enduml
```

### 2.3 関連コンテキスト取得（UC-06）
#### ユースケース手順
1. LLMエージェントがコンテキスト要求
2. システムが検索を実行
3. システムが結果を構造化
4. システムがコンテキストを生成
5. システムが結果を返却

#### ユースケースフロー図
```plantuml
@startuml
participant "LLM Agent" as Agent
participant "OpenAIAgentsInterface" as OAI
participant "SearchService" as SS
participant "SearchModel" as SM
participant "VectorDocumentStore" as VS
participant "BM25Index" as BM25
participant "EmbeddingService" as ES

Agent -> OAI: コンテキスト要求
OAI -> SS: 検索要求
SS -> SM: クエリ前処理
SM -> ES: クエリ埋め込み
ES --> SM: クエリベクトル
SM -> VS: ベクトル検索
SM -> BM25: キーワード検索
VS --> SM: ベクトル結果
BM25 --> SM: キーワード結果
SM -> SM: 結果統合
SM -> SM: スコアリング
SM -> SM: コンテキスト生成
SM --> SS: 構造化結果
SS --> OAI: コンテキスト返却
OAI --> Agent: コンテキスト提供
@enduml
```

## 3. インデックス管理機能

### 3.1 インデックス永続化（UC-07）
#### ユースケース手順
1. システムがインデックス更新を検知
2. システムがインデックスをシリアライズ
3. システムがファイルに保存
4. システムが完了を通知

#### ユースケースフロー図
```plantuml
@startuml
participant "IndexManager" as IM
participant "VectorDocumentStore" as VS
participant "BM25Index" as BM25
participant "FileSystem" as FS

IM -> VS: インデックス取得
VS --> IM: ベクトルインデックス
IM -> BM25: インデックス取得
BM25 --> IM: キーワードインデックス
IM -> IM: シリアライズ
IM -> FS: ファイル保存
FS --> IM: 保存完了
@enduml
```

### 3.2 インデックスロード（UC-08）
#### ユースケース手順
1. システムが起動
2. システムがインデックスファイルを検索
3. システムがファイルを読み込み
4. システムがインデックスを復元
5. システムが完了を通知

#### ユースケースフロー図
```plantuml
@startuml
participant "IndexManager" as IM
participant "FileSystem" as FS
participant "VectorDocumentStore" as VS
participant "BM25Index" as BM25

IM -> FS: インデックスファイル検索
FS --> IM: ファイル一覧
IM -> FS: ファイル読み込み
FS --> IM: インデックスデータ
IM -> IM: デシリアライズ
IM -> VS: ベクトルインデックス復元
IM -> BM25: キーワードインデックス復元
VS --> IM: 復元完了
BM25 --> IM: 復元完了
@enduml
```

### 3.3 インデックス更新（UC-09）
#### ユースケース手順
1. システムが変更を検知
2. システムが差分を計算
3. システムがインデックスを更新
4. システムがキャッシュを更新
5. システムが完了を通知

#### ユースケースフロー図
```plantuml
@startuml
participant "IndexManager" as IM
participant "Document" as Doc
participant "VectorDocumentStore" as VS
participant "BM25Index" as BM25
participant "FileSystem" as FS

Doc -> IM: 変更通知
IM -> Doc: 差分計算
Doc --> IM: 差分データ
IM -> VS: ベクトル更新
IM -> BM25: キーワード更新
VS --> IM: 更新完了
BM25 --> IM: 更新完了
IM -> FS: キャッシュ更新
FS --> IM: 更新完了
@enduml
```

## 4. OpenAI Agents SDK連携機能

### 4.1 ツール統合（UC-10）
#### ユースケース手順
1. エージェントがツールを要求
2. システムがツールを提供
3. エージェントがツールを使用
4. システムが結果を返却

#### ユースケースフロー図
```plantuml
@startuml
participant "LLM Agent" as Agent
participant "OpenAIAgentsInterface" as OAI
participant "DocumentManager" as DM
participant "SearchService" as SS
participant "IndexManager" as IM

Agent -> OAI: ツール要求
OAI -> DM: 文書管理ツール提供
OAI -> SS: 検索ツール提供
OAI -> IM: インデックス管理ツール提供
DM --> OAI: ツール定義
SS --> OAI: ツール定義
IM --> OAI: ツール定義
OAI --> Agent: ツール提供
Agent -> OAI: ツール使用
OAI -> DM: 文書操作
OAI -> SS: 検索実行
OAI -> IM: インデックス操作
DM --> OAI: 操作結果
SS --> OAI: 検索結果
IM --> OAI: 操作結果
OAI --> Agent: 実行結果
@enduml
```

### 4.2 コンテキスト統合（UC-11）
#### ユースケース手順
1. エージェントがコンテキストを要求
2. システムが検索を実行
3. システムがコンテキストを生成
4. システムが結果を返却

#### ユースケースフロー図
```plantuml
@startuml
participant "LLM Agent" as Agent
participant "OpenAIAgentsInterface" as OAI
participant "SearchService" as SS
participant "SearchModel" as SM
participant "VectorDocumentStore" as VS
participant "BM25Index" as BM25

Agent -> OAI: コンテキスト要求
OAI -> SS: 検索要求
SS -> SM: クエリ前処理
SM -> VS: ベクトル検索
SM -> BM25: キーワード検索
VS --> SM: ベクトル結果
BM25 --> SM: キーワード結果
SM -> SM: 結果統合
SM -> SM: コンテキスト生成
SM --> SS: コンテキスト
SS --> OAI: コンテキスト返却
OAI --> Agent: コンテキスト提供
@enduml
```

### 4.3 LangChain連携（UC-12）
#### ユースケース手順
1. ユーザーがLangChainを使用
2. システムがRetrieverを提供
3. ユーザーが検索を実行
4. システムが結果を返却

#### ユースケースフロー図
```plantuml
@startuml
actor User
participant "LangChainInterface" as LC
participant "SearchService" as SS
participant "SearchModel" as SM
participant "VectorDocumentStore" as VS
participant "BM25Index" as BM25

User -> LC: LangChain使用
LC -> SS: Retriever提供
SS -> SM: 検索設定
SM -> VS: ベクトル検索
SM -> BM25: キーワード検索
VS --> SM: ベクトル結果
BM25 --> SM: キーワード結果
SM -> SM: 結果統合
SM --> SS: 検索結果
SS --> LC: LangChain形式で返却
LC --> User: 結果表示
@enduml
``` 