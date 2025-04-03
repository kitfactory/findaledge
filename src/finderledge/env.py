from oneenv import oneenv

@oneenv
def common_api_endpoints():
    """Settings for API endpoints and keys.
    APIエンドポイントとキーの設定"""
    return {
        "OPENAI_API_KEY": {
            "description": "API key for accessing the OpenAI API. OpenAI APIにアクセスするためのAPIキー。",
            "required": True, # APIキーは通常必須
            "default": ""
        },
        "OPENAI_BASE_URL": {
            "description": "Base URL for the OpenAI API compatible endpoint (optional). OpenAI API互換エンドポイントのベースURL。(オプション)",
            "required": False,
            "default": None # デフォルトは None または OpenAI の公式エンドポイント
        },
        "OLLAMA_BASE_URL": {
            "description": "Base URL for the Ollama API endpoint (optional). Ollama APIエンドポイントのベースURL。(オプション)",
            "required": False,
            "default": "http://localhost:11434" # OllamaのデフォルトURL
        }
    }

@oneenv
def model_config():
    """Settings for model provider, embedding model, and LLM model.
    モデルプロバイダー、Embeddingモデル、LLMモデルの設定"""
    return {
        "FINDERLEDGE_MODEL_PROVIDER": {
            "description": "The provider for the models (e.g., 'openai', 'ollama'). モデルを提供するプロバイダー (例: 'openai', 'ollama')。",
            "required": False,
            "default": "openai",
            "choices": ["openai", "ollama"] # 選択肢を提示
        },
        "FINDERLEDGE_EMBEDDING_MODEL_NAME": {
            "description": "The name of the embedding model to use. 使用するEmbeddingモデルの名前。",
            "required": False,
            "default": "text-embedding-3-small" # OpenAIの一般的なモデルをデフォルトに
        },
        "FINDERLEDGE_LLM_MODEL_NAME": {
            "description": "The name of the LLM model to use. 使用するLLMモデルの名前。",
            "required": False,
            "default": "gpt-4o" # OpenAIの一般的なモデルをデフォルトに
        }
    }

@oneenv
def vector_store_config():
    """Settings for the vector store provider and specific configurations.
    ベクトルストアプロバイダーと個別設定"""
    return {
        "FINDERLEDGE_VECTOR_STORE_PROVIDER": {
            "description": "The vector store provider to use (e.g., 'chroma', 'faiss', 'pinecone'). 使用するベクトルストアプロバイダー (例: 'chroma', 'faiss', 'pinecone')。",
            "required": False,
            "default": "chroma", # ローカルで使いやすい Chroma をデフォルトに
            "choices": ["chroma", "faiss", "pinecone"]
        },

        # Chroma specific settings
        "FINDERLEDGE_CHROMA_PERSIST_DIRECTORY": {
            "description": "(Chroma only) Path to the directory for persisting Chroma data. (Chroma のみ) Chromaデータを永続化するディレクトリへのパス。",
            "required": False,
            "default": "./chroma_db" # デフォルトの永続化パス
        },

        # FAISS specific settings
        "FINDERLEDGE_FAISS_INDEX_PATH": {
            "description": "(FAISS only) Path to the FAISS index file. (FAISS のみ) FAISSインデックスファイルへのパス。",
            "required": False,
            "default": "./faiss_index.idx" # デフォルトのインデックスファイルパス
        },

        # Pinecone specific settings
        "PINECONE_API_KEY": {
            "description": "(Pinecone only) API key for Pinecone. (Pinecone のみ) Pinecone用APIキー。",
            "required": False,
            "default": ""
        },
        "FINDERLEDGE_PINECONE_ENVIRONMENT": {
            "description": "(Pinecone only) Environment name for Pinecone. (Pinecone のみ) Pineconeの環境名。",
            "required": False,
            "default": ""
        },
        "FINDERLEDGE_PINECONE_INDEX_NAME": {
            "description": "(Pinecone only) Index name in Pinecone. (Pinecone のみ) Pineconeのインデックス名。",
            "required": False,
            "default": ""
        }
    }

