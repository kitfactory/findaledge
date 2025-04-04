"""
Document Loader Sample
文書ローダーサンプル

This sample demonstrates loading documents from various formats (text, PDF) and extracting content.
このサンプルでは、様々な形式の文書（テキスト、PDF、オフィス文書）を読み込み、内容を抽出する方法を示します。
"""

import os
import sys
import json
import base64
import tempfile
from datetime import datetime
from pathlib import Path
from oneenv import load_dotenv
from typing import List, Dict, Optional, Union, Any
import uuid
import shutil
import asyncio

# We'll try to import different document loaders and handle import errors
# 異なる文書ローダーをインポートし、インポートエラーを処理します
try:
    from langchain_community.document_loaders import TextLoader
    from langchain_community.document_loaders import PyPDFLoader
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: langchain_community document loaders not available.")
    print("警告: langchain_community ドキュメントローダーが利用できません。")
    print("Install with: uv add langchain-community")
    print("インストール方法: uv add langchain-community")

# Try to import OpenAI for PDF processing
# PDFの処理のためにOpenAIをインポートしてみます
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai package not available for PDF processing.")
    print("警告: PDFを処理するための openai パッケージが利用できません。")
    print("Install with: uv add openai")
    print("インストール方法: uv add openai")

# Try to import Google's generativeai (Gemini) for PDF processing
# PDFの処理のためにGoogle generativeai（Gemini）をインポートしてみます
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai package not available for PDF processing.")
    print("警告: PDFを処理するための google-generativeai パッケージが利用できません。")
    print("Install with: uv add google-generativeai")
    print("インストール方法: uv add google-generativeai")

# Try to import PIL for image processing
# 画像処理のためにPILをインポートしてみます
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available for image processing.")
    print("警告: 画像処理のためのPILが利用できません。")
    print("Install with: uv add pillow")
    print("インストール方法: uv add pillow")

# Try to import Markitdown for Office document processing
# オフィス文書処理のためにMarkitdownをインポートしてみます
try:
    import markitdown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False
    print("Warning: markitdown package not available for Office document processing.")
    print("警告: オフィス文書を処理するための markitdown パッケージが利用できません。")
    print("Install with: uv add markitdown")
    print("インストール方法: uv add markitdown")

# Add project root to sys.path to allow importing finderledge
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from langchain_core.documents import Document

# Check if the main module can be imported (for using its DocumentLoader)
try:
    from findaledge.document_loader import DocumentLoader as FindaLedgeLoader
    findaledge_loader_available = True
except ImportError:
    findaledge_loader_available = False
    print("Warning: Could not import FindaLedgeLoader. Some features might be limited.")
    print("警告: FindaLedgeLoader をインポートできませんでした。一部の機能が制限される可能性があります。")
    class FindaLedgeLoader: # Dummy class if import fails
        def __init__(self, path: str, **kwargs):
            print(f"[Dummy FindaLedgeLoader] Initialized with path: {path}")
        def load(self) -> List[Document]:
            print("[Dummy FindaLedgeLoader] Load called, returning empty list.")
            return []

# --- Configuration / 設定 ---
# Create a temporary directory for sample files
# サンプルファイル用の一時ディレクトリを作成
SAMPLE_DIR = project_root / "study" / "temp_docs"
SAMPLE_TEXT_FILE = SAMPLE_DIR / "sample.txt"
SAMPLE_MD_FILE = SAMPLE_DIR / "sample.md"
SAMPLE_PDF_FILE = SAMPLE_DIR / "sample.pdf" # Needs a real PDF file
SAMPLE_CSV_FILE = SAMPLE_DIR / "sample.csv"
SAMPLE_JSON_FILE = SAMPLE_DIR / "sample.json"
SAMPLE_DOCX_FILE = SAMPLE_DIR / "sample.docx" # Needs a real DOCX file
SAMPLE_PPTX_FILE = SAMPLE_DIR / "sample.pptx" # Needs a real PPTX file
SAMPLE_XLSX_FILE = SAMPLE_DIR / "sample.xlsx" # Needs a real XLSX file

def create_sample_files():
    """
    Create sample files in the temporary directory.
    一時ディレクトリにサンプルファイルを作成します。
    """
    SAMPLE_DIR.mkdir(exist_ok=True)

    # --- Create Sample Content / サンプルコンテンツの作成 ---
    # Simple text content / シンプルなテキストコンテンツ
    sample_text_content = """FindaLedge サンプルテキストファイル
これはプレーンテキストファイルのサンプルです。
複数の行が含まれています。

FindaLedge ライブラリは、様々な文書形式を読み込み、検索可能なインデックスを作成することができます。
--- English Version ---
FindaLedge Sample Text File
This is a sample plain text file.
It contains multiple lines.

The FindaLedge library can load various document formats and create searchable indices.
"""

    # Markdown content / Markdownコンテンツ
    sample_md_content = """# FindaLedge サンプル Markdown

これは **Markdown** ファイルのサンプルです。

## 特徴
- 箇条書き 1
- 箇条書き 2

`コードブロック` も含めることができます。

## English Section

This is a sample **Markdown** file for FindaLedge.
- Bullet point 1
- Bullet point 2
"""

    # CSV content / CSVコンテンツ
    sample_csv_content = """ID,名前,役割
1,Alice,開発者
2,Bob,デザイナー
3,Charlie,マネージャー
"""

    # JSON content / JSONコンテンツ
    sample_json_content = '''
    [
      {
        "id": 101,
        "item": "ラップトップ",
        "tags": ["電子機器", "仕事"]
      },
      {
        "id": 102,
        "item": "キーボード",
        "tags": ["電子機器", "アクセサリ"]
      }
    ]
    '''

    # --- Write Files / ファイルへの書き込み ---
    try:
        with open(SAMPLE_TEXT_FILE, "w", encoding="utf-8") as f:
            f.write(sample_text_content)
        print(f"Created: {SAMPLE_TEXT_FILE}")

        with open(SAMPLE_MD_FILE, "w", encoding="utf-8") as f:
            f.write(sample_md_content)
        print(f"Created: {SAMPLE_MD_FILE}")

        with open(SAMPLE_CSV_FILE, "w", encoding="utf-8") as f:
            f.write(sample_csv_content)
        print(f"Created: {SAMPLE_CSV_FILE}")

        with open(SAMPLE_JSON_FILE, "w", encoding="utf-8") as f:
            f.write(sample_json_content)
        print(f"Created: {SAMPLE_JSON_FILE}")

        # Create dummy Office files if they don't exist (replace with real files for actual testing)
        # 実際のテストには実際のファイルを使用してください
        if not SAMPLE_PDF_FILE.exists():
            SAMPLE_PDF_FILE.touch() # Create empty file as placeholder
            print(f"Created placeholder: {SAMPLE_PDF_FILE} (Replace with a real PDF)")
        if not SAMPLE_DOCX_FILE.exists():
            SAMPLE_DOCX_FILE.touch()
            print(f"Created placeholder: {SAMPLE_DOCX_FILE} (Replace with a real DOCX)")
        if not SAMPLE_PPTX_FILE.exists():
            SAMPLE_PPTX_FILE.touch()
            print(f"Created placeholder: {SAMPLE_PPTX_FILE} (Replace with a real PPTX)")
        if not SAMPLE_XLSX_FILE.exists():
            SAMPLE_XLSX_FILE.touch()
            print(f"Created placeholder: {SAMPLE_XLSX_FILE} (Replace with a real XLSX)")

    except IOError as e:
        print(f"Error creating sample files: {e}")
        print(f"サンプルファイルの作成中にエラーが発生しました: {e}")

def cleanup_sample_files():
    """
    Remove the temporary directory and sample files.
    一時ディレクトリとサンプルファイルを削除します。
    """
    try:
        if SAMPLE_DIR.exists():
            shutil.rmtree(SAMPLE_DIR)
            print(f"Removed directory: {SAMPLE_DIR}")
    except OSError as e:
        print(f"Error removing sample files: {e}")
        print(f"サンプルファイルの削除中にエラーが発生しました: {e}")

def print_loaded_documents(title: str, docs: List[Document], max_docs: int = 3, content_preview: int = 100):
    """
    Print information about loaded documents.
    読み込まれたドキュメントに関する情報を表示します。
    """
    print(f"\n--- {title} --- ({len(docs)} documents loaded) ---")
    print(f"--- {title} --- ({len(docs)} 件のドキュメントを読み込み) ---")
    if not docs:
        print("No documents loaded.")
        print("ドキュメントは読み込まれませんでした。")
        return

    for i, doc in enumerate(docs[:max_docs]):
        print(f"\nDocument {i+1}: / ドキュメント {i+1}:")
        print(f"  Content Preview (最初の {content_preview} 文字):")
        preview = doc.page_content[:content_preview].replace("\n", " ") + "..."
        print(f"    {preview}")
        print(f"  Metadata / メタデータ:")
        for key, value in doc.metadata.items():
            print(f"    {key}: {value}")
    if len(docs) > max_docs:
        print(f"\n... and {len(docs) - max_docs} more documents. / 他に {len(docs) - max_docs} 件のドキュメントがあります。")

async def load_with_findaledge_loader(path: str):
    """
    Load documents using the FindaLedge internal loader.
    FindaLedge内部ローダーを使用してドキュメントを読み込みます。
    """
    if not findaledge_loader_available:
        print("\nFindaLedgeLoader is not available. Skipping this test.")
        print("FindaLedgeLoaderが利用できません。このテストをスキップします。")
        return []
    try:
        print(f"\nAttempting to load '{path}' with FindaLedgeLoader...")
        print(f"FindaLedgeLoaderで '{path}' の読み込みを試行中...")
        loader = FindaLedgeLoader(path) # Assuming sync initialization
                                       # 同期初期化を想定
        # Assuming loader.load() is potentially async or can be run async
        # loader.load()が非同期である可能性があるか、非同期で実行できると想定
        # If load is purely sync, just call it directly.
        # loadが純粋に同期の場合は、直接呼び出します。
        # docs = loader.load()
        # Example if load needs to be run in thread pool if sync:
        # 同期の場合にloadをスレッドプールで実行する必要がある場合の例:
        # docs = loader.load()
        loop = asyncio.get_running_loop()
        docs = await loop.run_in_executor(None, loader.load)
        print_loaded_documents(f"FindaLedgeLoader ('{path}')", docs)
        return docs
    except Exception as e:
        print(f"Error loading with FindaLedgeLoader: {e}")
        print(f"FindaLedgeLoaderでの読み込み中にエラーが発生しました: {e}")
        return []

async def document_loader_sample():
    """
    Main function to demonstrate document loading.
    ドキュメントの読み込みをデモンストレーションするメイン関数。
    """
    print("Starting Document Loader Sample...")
    print("ドキュメントローダーサンプルを開始します...")
    create_sample_files()

    # --- Load using Langchain Loaders / Langchainローダーを使用した読み込み ---
    try:
        # 1. TextLoader
        print("\n=== Loading Text file with TextLoader ===")
        print("=== TextLoaderでテキストファイルを読み込み ===")
        text_loader = TextLoader(str(SAMPLE_TEXT_FILE), encoding="utf-8")
        text_docs = text_loader.load()
        print_loaded_documents(f"TextLoader ('{SAMPLE_TEXT_FILE.name}')", text_docs)

        # 2. UnstructuredMarkdownLoader
        print("\n=== Loading Markdown file with UnstructuredMarkdownLoader ===")
        print("=== UnstructuredMarkdownLoaderでMarkdownファイルを読み込み ===")
        md_loader = UnstructuredMarkdownLoader(str(SAMPLE_MD_FILE))
        md_docs = md_loader.load()
        print_loaded_documents(f"UnstructuredMarkdownLoader ('{SAMPLE_MD_FILE.name}')", md_docs)

        # 3. PyPDFLoader (Needs a real PDF)
        print("\n=== Loading PDF file with PyPDFLoader ===")
        print("=== PyPDFLoaderでPDFファイルを読み込み ===")
        if SAMPLE_PDF_FILE.exists() and SAMPLE_PDF_FILE.stat().st_size > 0:
            try:
                pdf_loader = PyPDFLoader(str(SAMPLE_PDF_FILE))
                pdf_docs = pdf_loader.load()
                print_loaded_documents(f"PyPDFLoader ('{SAMPLE_PDF_FILE.name}')", pdf_docs)
            except Exception as e:
                print(f"Could not load PDF (ensure PyPDF is installed and file is valid): {e}")
                print(f"PDFを読み込めませんでした (PyPDFがインストールされており、ファイルが有効であることを確認してください): {e}")
        else:
            print(f"Skipping PDF loading. File not found or empty: {SAMPLE_PDF_FILE}")
            print(f"PDFの読み込みをスキップします。ファイルが見つからないか空です: {SAMPLE_PDF_FILE}")

        # 4. CSVLoader
        print("\n=== Loading CSV file with CSVLoader ===")
        print("=== CSVLoaderでCSVファイルを読み込み ===")
        csv_loader = CSVLoader(str(SAMPLE_CSV_FILE), encoding="utf-8")
        csv_docs = csv_loader.load()
        print_loaded_documents(f"CSVLoader ('{SAMPLE_CSV_FILE.name}')", csv_docs)

        # 5. JSONLoader (using jq schema)
        print("\n=== Loading JSON file with JSONLoader ===")
        print("=== JSONLoaderでJSONファイルを読み込み ===")
        # Simple jq schema: load each object in the array as a document, extract 'item' and 'tags'
        # シンプルなjqスキーマ: 配列内の各オブジェクトをドキュメントとして読み込み、「item」と「tags」を抽出
        json_loader = JSONLoader(
            file_path=str(SAMPLE_JSON_FILE),
            jq_schema='.[] | {page_content: (.item + " Tags: " + (.tags | join(", "))), metadata: {id: .id}}',
            text_content=False, # Process based on jq_schema
                               # jq_schemaに基づいて処理
            json_lines=False
        )
        json_docs = json_loader.load()
        print_loaded_documents(f"JSONLoader ('{SAMPLE_JSON_FILE.name}')", json_docs)

        # 6. UnstructuredWordDocumentLoader (Needs python-docx, unstructured)
        print("\n=== Loading DOCX file with UnstructuredWordDocumentLoader ===")
        print("=== UnstructuredWordDocumentLoaderでDOCXファイルを読み込み ===")
        if SAMPLE_DOCX_FILE.exists() and SAMPLE_DOCX_FILE.stat().st_size > 0:
            try:
                docx_loader = UnstructuredWordDocumentLoader(str(SAMPLE_DOCX_FILE))
                docx_docs = docx_loader.load()
                print_loaded_documents(f"UnstructuredWordDocumentLoader ('{SAMPLE_DOCX_FILE.name}')", docx_docs)
            except Exception as e:
                print(f"Could not load DOCX (ensure python-docx and unstructured are installed): {e}")
                print(f"DOCXを読み込めませんでした (python-docxとunstructuredがインストールされていることを確認してください): {e}")
        else:
            print(f"Skipping DOCX loading. File not found or empty: {SAMPLE_DOCX_FILE}")
            print(f"DOCXの読み込みをスキップします。ファイルが見つからないか空です: {SAMPLE_DOCX_FILE}")

        # 7. UnstructuredPowerPointLoader (Needs python-pptx, unstructured)
        print("\n=== Loading PPTX file with UnstructuredPowerPointLoader ===")
        print("=== UnstructuredPowerPointLoaderでPPTXファイルを読み込み ===")
        if SAMPLE_PPTX_FILE.exists() and SAMPLE_PPTX_FILE.stat().st_size > 0:
            try:
                pptx_loader = UnstructuredPowerPointLoader(str(SAMPLE_PPTX_FILE))
                pptx_docs = pptx_loader.load()
                print_loaded_documents(f"UnstructuredPowerPointLoader ('{SAMPLE_PPTX_FILE.name}')", pptx_docs)
            except Exception as e:
                print(f"Could not load PPTX (ensure python-pptx and unstructured are installed): {e}")
                print(f"PPTXを読み込めませんでした (python-pptxとunstructuredがインストールされていることを確認してください): {e}")
        else:
            print(f"Skipping PPTX loading. File not found or empty: {SAMPLE_PPTX_FILE}")
            print(f"PPTXの読み込みをスキップします。ファイルが見つからないか空です: {SAMPLE_PPTX_FILE}")

        # 8. UnstructuredExcelLoader (Needs openpyxl, unstructured)
        print("\n=== Loading XLSX file with UnstructuredExcelLoader ===")
        print("=== UnstructuredExcelLoaderでXLSXファイルを読み込み ===")
        if SAMPLE_XLSX_FILE.exists() and SAMPLE_XLSX_FILE.stat().st_size > 0:
            try:
                xlsx_loader = UnstructuredExcelLoader(str(SAMPLE_XLSX_FILE), mode="elements")
                xlsx_docs = xlsx_loader.load()
                print_loaded_documents(f"UnstructuredExcelLoader ('{SAMPLE_XLSX_FILE.name}')", xlsx_docs)
            except Exception as e:
                print(f"Could not load XLSX (ensure openpyxl and unstructured are installed): {e}")
                print(f"XLSXを読み込めませんでした (openpyxlとunstructuredがインストールされていることを確認してください): {e}")
        else:
            print(f"Skipping XLSX loading. File not found or empty: {SAMPLE_XLSX_FILE}")
            print(f"XLSXの読み込みをスキップします。ファイルが見つからないか空です: {SAMPLE_XLSX_FILE}")

        # 9. DirectoryLoader (loading .txt and .md files)
        print("\n=== Loading .txt and .md files from directory with DirectoryLoader ===")
        print("=== DirectoryLoaderでディレクトリから.txtと.mdファイルを読み込み ===")
        dir_loader = DirectoryLoader(
            str(SAMPLE_DIR),
            glob="**/*[.txt|.md]", # Load .txt and .md files recursively
                                   # .txtと.mdファイルを再帰的に読み込み
            use_multithreading=True,
            show_progress=True,
            loader_cls=TextLoader, # Use TextLoader for these files
                                   # これらのファイルにはTextLoaderを使用
            loader_kwargs={"encoding": "utf-8"}
        )
        dir_docs = dir_loader.load()
        print_loaded_documents(f"DirectoryLoader ('{SAMPLE_DIR.name}')", dir_docs)

    except ImportError as e:
        print(f"\nError: Required libraries for some loaders might be missing: {e}")
        print(f"エラー: 一部のローダーに必要なライブラリが見つからない可能性があります: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during Langchain loading: {e}")
        print(f"Langchainの読み込み中に予期せぬエラーが発生しました: {e}")

    # --- Load using FindaLedge Loader / FindaLedgeローダーを使用した読み込み ---
    # This demonstrates how FindaLedge's internal loader might be used (if available)
    # これはFindaLedgeの内部ローダーがどのように使用されるかを示します（利用可能な場合）
    print("\n=== Loading with FindaLedge Internal Loader ===")
    print("=== FindaLedge内部ローダーを使用した読み込み ===")
    await load_with_findaledge_loader(str(SAMPLE_TEXT_FILE))
    await load_with_findaledge_loader(str(SAMPLE_MD_FILE))
    # Add calls for other file types supported by FindaLedgeLoader
    # FindaLedgeLoaderでサポートされている他のファイルタイプに対する呼び出しを追加
    await load_with_findaledge_loader(str(SAMPLE_DIR))

    # --- Cleanup / クリーンアップ ---
    # cleanup_sample_files() # Uncomment to delete files after running
                           # 実行後にファイルを削除する場合はコメント解除

    print("\nDocument Loader Sample finished.")
    print("ドキュメントローダーサンプルが終了しました。")

# Main execution / メイン実行
if __name__ == "__main__":
    # To run this specific sample directly / この特定のサンプルを直接実行するには:
    # asyncio.run(document_loader_sample())
    # Allow running as part of the main sample runner / メインサンプルランナーの一部として実行を許可
    pass 