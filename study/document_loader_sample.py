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

def load_text_file(file_path: str) -> str:
    """
    Load and return content from a text file
    テキストファイルからコンテンツを読み込んで返す
    
    Args:
        file_path: Path to the text file
                  テキストファイルのパス
    
    Returns:
        String content of the file
        ファイルの文字列内容
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except Exception as e:
        print(f"Error loading text file: {e}")
        print(f"テキストファイルの読み込みエラー: {e}")
        return ""

def load_text_with_langchain(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a text file using LangChain's TextLoader
    LangChainのTextLoaderを使用してテキストファイルを読み込む
    
    Args:
        file_path: Path to the text file
                  テキストファイルのパス
    
    Returns:
        List of document chunks with content and metadata
        コンテンツとメタデータを含むドキュメントチャンクのリスト
    """
    if not LANGCHAIN_AVAILABLE:
        print("LangChain is not available. Cannot use TextLoader.")
        print("LangChainが利用できません。TextLoaderを使用できません。")
        return []
    
    try:
        loader = TextLoader(file_path)
        documents = loader.load()
        
        # Convert LangChain documents to a standardized format
        # LangChainドキュメントを標準形式に変換
        result = []
        for doc in documents:
            result.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        return result
    except Exception as e:
        print(f"Error loading text with LangChain: {e}")
        print(f"LangChainを使用したテキスト読み込みエラー: {e}")
        return []

def load_pdf_with_langchain(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a PDF file using LangChain's PyPDFLoader
    LangChainのPyPDFLoaderを使用してPDFファイルを読み込む
    
    Args:
        file_path: Path to the PDF file
                  PDFファイルのパス
    
    Returns:
        List of document chunks with content and metadata
        コンテンツとメタデータを含むドキュメントチャンクのリスト
    """
    if not LANGCHAIN_AVAILABLE:
        print("LangChain is not available. Cannot use PyPDFLoader.")
        print("LangChainが利用できません。PyPDFLoaderを使用できません。")
        return []
    
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Convert LangChain documents to a standardized format
        # LangChainドキュメントを標準形式に変換
        result = []
        for doc in documents:
            result.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        return result
    except Exception as e:
        print(f"Error loading PDF with LangChain: {e}")
        print(f"LangChainを使用したPDF読み込みエラー: {e}")
        return []

def load_pdf_with_openai(file_path: str, max_pages: int = 5) -> str:
    """
    Extract text from a PDF file using OpenAI's document processing capabilities
    OpenAIのドキュメント処理機能を使用してPDFファイルからテキストを抽出する
    
    Args:
        file_path: Path to the PDF file
                  PDFファイルのパス
        max_pages: Maximum number of pages to process
                  処理する最大ページ数
    
    Returns:
        Extracted text content
        抽出されたテキストコンテンツ
    """
    if not OPENAI_AVAILABLE:
        print("OpenAI package is not available. Cannot process PDF with OpenAI.")
        print("OpenAIパッケージが利用できません。OpenAIでPDFを処理できません。")
        return ""
    
    try:
        # Load environment variables for API key
        # API キーの環境変数を読み込む
        load_dotenv()
        
        # Check if OpenAI API key is set
        # OpenAI APIキーが設定されているか確認
        if not os.environ.get("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY environment variable is not set.")
            print("エラー: OPENAI_API_KEY環境変数が設定されていません。")
            return ""
        
        # Note: Current OpenAI models don't support direct PDF processing
        # 注意: 現在のOpenAIモデルは直接PDFを処理することをサポートしていません
        print("Note: OpenAI's GPT-4o model doesn't directly support PDF format.")
        print("注意: OpenAIのGPT-4oモデルは直接PDF形式をサポートしていません。")
        print("For a complete implementation, PDF should be converted to images first.")
        print("完全な実装では、最初にPDFを画像に変換する必要があります。")
        
        # Placeholder implementation for PDF processing via OpenAI
        # OpenAIを使用したPDF処理のプレースホルダー実装
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"""
This is a demonstration of PDF text extraction capabilities with OpenAI.
In a real implementation, I would:
1. Convert each page of the PDF at {file_path} to images
2. Process each image with OpenAI's vision capabilities
3. Combine the extracted text

For example, using libraries like pdf2image to convert PDF pages to images:
```python
from pdf2image import convert_from_path
images = convert_from_path(pdf_path)
```

Then send each image to OpenAI's vision model and combine the results.
This approach works particularly well for PDFs with complex layouts or those that are image-based.
"""}
                    ]
                }
            ],
            max_tokens=1000
        )
        
        # Extract text from the response
        # レスポンスからテキストを抽出
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error processing PDF with OpenAI: {e}")
        print(f"OpenAIによるPDF処理エラー: {e}")
        return ""

def load_pdf_with_gemini(file_path: str) -> str:
    """
    Extract text from a PDF file using Google's Gemini API
    GoogleのGemini APIを使用してPDFファイルからテキストを抽出する
    
    Args:
        file_path: Path to the PDF file
                  PDFファイルのパス
    
    Returns:
        Extracted text content
        抽出されたテキストコンテンツ
    """
    if not GEMINI_AVAILABLE:
        print("Google Generative AI package is not available. Cannot process PDF with Gemini.")
        print("Google Generative AIパッケージが利用できません。GeminiでPDFを処理できません。")
        return ""
    
    try:
        # Load environment variables for API key
        # API キーの環境変数を読み込む
        load_dotenv()
        
        # Check if Gemini API key is set
        # Gemini APIキーが設定されているか確認
        if not os.environ.get("GOOGLE_API_KEY"):
            print("Error: GOOGLE_API_KEY environment variable is not set.")
            print("エラー: GOOGLE_API_KEY環境変数が設定されていません。")
            return ""
        
        # For the Gemini API, we'd need to convert PDF to images first
        # Gemini APIの場合、最初にPDFを画像に変換する必要があります
        print("Note: For a complete implementation, PDF should be converted to images first.")
        print("注: 完全な実装では、最初にPDFを画像に変換する必要があります。")
        
        # Configure the Gemini API
        # Gemini APIを設定
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        
        # Since we can't easily convert PDF to image here, we'll just query Gemini with a text prompt
        # この例では簡単にPDFを画像に変換できないため、テキストプロンプトでGeminiに問い合わせます
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Create a prompt about PDF extraction
        # PDF抽出に関するプロンプトを作成
        prompt = f"""
        This is a demonstration of PDF text extraction capabilities.
        In a real implementation, I would:
        1. Convert the PDF at {file_path} to images
        2. Process each image with the Gemini Vision API
        3. Combine the extracted text
        
        For now, this is a placeholder response.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error processing with Gemini: {e}")
        print(f"Geminiによる処理エラー: {e}")
        return ""

def load_office_with_markitdown(file_path: str) -> str:
    """
    Load and extract content from Office documents (DOCX, XLSX, PPTX) or PDF using Markitdown
    Markitdownを使用してオフィス文書（DOCX、XLSX、PPTX）またはPDFからコンテンツを読み込んで抽出する
    
    Args:
        file_path: Path to the Office document or PDF
                  オフィス文書またはPDFのパス
    
    Returns:
        Extracted text content
        抽出されたテキストコンテンツ
    """
    if not MARKITDOWN_AVAILABLE:
        print("Markitdown package is not available. Cannot process Office documents.")
        print("Markitdownパッケージが利用できません。オフィス文書を処理できません。")
        return ""
    
    try:
        # Initialize MarkItDown converter
        # MarkItDownコンバーターを初期化
        converter = markitdown.MarkItDown()
        
        # Process different file types
        # 様々なファイルタイプを処理
        print(f"Converting {file_path} using Markitdown...")
        print(f"Markitdownを使用して {file_path} を変換しています...")
        
        # Convert the document to markdown using convert_local method
        # convert_localメソッドを使用して文書をマークダウンに変換
        result = converter.convert_local(file_path)
        
        # Return the markdown content
        # マークダウンコンテンツを返す
        if result and hasattr(result, 'markdown') and result.markdown:
            print(f"Successfully converted {file_path} to markdown")
            print(f"{file_path} をマークダウンに変換しました")
            return result.markdown
        else:
            print(f"Failed to extract content from {file_path}")
            print(f"{file_path} からコンテンツの抽出に失敗しました")
            return ""
            
    except Exception as e:
        print(f"Error processing document with Markitdown: {e}")
        print(f"Markitdownでの文書処理エラー: {e}")
        return ""

def save_extracted_text(text: str, original_path: str, suffix: str = "_extracted") -> str:
    """
    Save extracted text to a file
    抽出されたテキストをファイルに保存する
    
    Args:
        text: Extracted text content
              抽出されたテキストコンテンツ
        original_path: Path to the original document
                      元のドキュメントへのパス
        suffix: Suffix to add to the output filename
               出力ファイル名に追加するサフィックス
    
    Returns:
        Path to the saved file
        保存されたファイルへのパス
    """
    try:
        # Create output filename based on the original file
        # 元のファイルに基づいて出力ファイル名を作成
        path = Path(original_path)
        output_path = path.with_name(f"{path.stem}{suffix}.txt")
        
        # Save the extracted text
        # 抽出されたテキストを保存
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Extracted text saved to: {output_path}")
        print(f"抽出されたテキストが保存されました: {output_path}")
        return str(output_path)
    except Exception as e:
        print(f"Error saving extracted text: {e}")
        print(f"抽出テキストの保存エラー: {e}")
        return ""

def create_sample_text_file() -> str:
    """
    Create a sample text file for demonstration purposes
    デモンストレーション用のサンプルテキストファイルを作成
    
    Returns:
        Path to the created sample file
        作成されたサンプルファイルへのパス
    """
    sample_content = """FinderLedge サンプルテキストファイル
===================================

このファイルは、ドキュメントローダーサンプルのテスト用に作成されたサンプルテキストファイルです。
FinderLedge ライブラリは、様々な文書形式を読み込み、検索可能なインデックスを作成することができます。

主な機能:
- テキストファイルの読み込み
- PDFファイルの読み込み
- ハイブリッド検索 (ベクトル検索 + キーワード検索)
- インデックスの永続化

このサンプルテキストには、いくつかの日本語のキーワードが含まれています：
* 検索エンジン
* 自然言語処理
* 埋め込みベクトル
* データベース
* インデックス作成

FinderLedge Sample Text File
===================================

This is a sample text file created for testing the document loader sample.
The FinderLedge library can load various document formats and create searchable indices.

Main features:
- Loading text files
- Loading PDF files
- Hybrid search (vector search + keyword search)
- Index persistence

This sample text contains some English keywords:
* search engine
* natural language processing
* embedding vectors
* database
* indexing
"""
    
    # Create a file in the temporary directory
    # 一時ディレクトリにファイルを作成
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    file_path = os.path.join(temp_dir, "sample_text.txt")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(sample_content)
    
    print(f"Created sample text file: {file_path}")
    print(f"サンプルテキストファイルを作成しました: {file_path}")
    return file_path

def document_loader_sample():
    """
    Main function demonstrating various document loading approaches
    様々な文書読み込みアプローチを示すメイン関数
    """
    print("Document Loader Sample")
    print("文書ローダーサンプル")
    print("=" * 60)
    
    # Load environment variables
    # 環境変数を読み込む
    load_dotenv()
    
    # 1. Simple text file loading
    # 1. シンプルなテキストファイル読み込み
    print("\n1. Loading text file with basic Python functions")
    print("1. 基本的なPython関数を使用したテキストファイルの読み込み")
    print("-" * 60)
    
    # Create a sample text file
    # サンプルテキストファイルを作成
    sample_text_path = create_sample_text_file()
    
    # Load the text file
    # テキストファイルを読み込む
    text_content = load_text_file(sample_text_path)
    
    # Show a preview of the content
    # コンテンツのプレビューを表示
    print("\nText file content preview:")
    print("テキストファイルのコンテンツプレビュー:")
    print("-" * 60)
    preview_lines = text_content.split('\n')[:5]
    print('\n'.join(preview_lines) + "\n...")
    print(f"Total characters: {len(text_content)}")
    print(f"合計文字数: {len(text_content)}")
    
    # 2. Text file loading with LangChain
    # 2. LangChainを使用したテキストファイル読み込み
    if LANGCHAIN_AVAILABLE:
        print("\n2. Loading text file with LangChain")
        print("2. LangChainを使用したテキストファイルの読み込み")
        print("-" * 60)
        
        langchain_docs = load_text_with_langchain(sample_text_path)
        
        if langchain_docs:
            print(f"Loaded {len(langchain_docs)} document(s) with LangChain")
            print(f"LangChainで {len(langchain_docs)} 個のドキュメントを読み込みました")
            
            # Show metadata
            # メタデータを表示
            print("\nMetadata from LangChain loader:")
            print("LangChainローダーからのメタデータ:")
            print(json.dumps(langchain_docs[0]["metadata"], indent=2, ensure_ascii=False))
    
    # 3. PDF loading with LangChain using sample.pdf
    # 3. sample.pdfを使用したLangChainでのPDF読み込み
    if LANGCHAIN_AVAILABLE:
        print("\n3. PDF loading with LangChain")
        print("3. LangChainを使用したPDF読み込み")
        print("-" * 60)
        
        pdf_path = "sample.pdf"
        if os.path.exists(pdf_path):
            print(f"Loading PDF from: {pdf_path}")
            print(f"PDFの読み込み元: {pdf_path}")
            
            langchain_pdf_docs = load_pdf_with_langchain(pdf_path)
            
            if langchain_pdf_docs:
                print(f"Loaded {len(langchain_pdf_docs)} page(s) from PDF with LangChain")
                print(f"LangChainでPDFから {len(langchain_pdf_docs)} ページを読み込みました")
                
                # Show metadata from first page
                # 最初のページからメタデータを表示
                print("\nMetadata from first page:")
                print("最初のページのメタデータ:")
                print(json.dumps(langchain_pdf_docs[0]["metadata"], indent=2, ensure_ascii=False))
                
                # Show content preview from first page
                # 最初のページからコンテンツプレビューを表示
                print("\nContent preview from first page:")
                print("最初のページのコンテンツプレビュー:")
                content = langchain_pdf_docs[0]["content"]
                preview = content[:200] + "..." if len(content) > 200 else content
                print(preview)
                
                # Save extracted text
                # 抽出されたテキストを保存
                combined_text = "\n\n".join([doc["content"] for doc in langchain_pdf_docs])
                save_path = save_extracted_text(combined_text, pdf_path, "_langchain")
        else:
            print(f"PDF file not found at: {pdf_path}")
            print(f"PDFファイルが見つかりません: {pdf_path}")
    
    # 4. PDF processing with OpenAI
    # 4. OpenAIを使用したPDF処理
    if OPENAI_AVAILABLE:
        print("\n4. PDF processing with OpenAI")
        print("4. OpenAIを使用したPDF処理")
        print("-" * 60)
        
        pdf_path = "sample.pdf"
        if os.path.exists(pdf_path):
            print(f"Processing PDF with OpenAI: {pdf_path}")
            print(f"OpenAIでPDFを処理: {pdf_path}")
            
            openai_extracted_text = load_pdf_with_openai(pdf_path)
            
            if openai_extracted_text:
                # Show content preview
                # コンテンツプレビューを表示
                print("\nContent preview from OpenAI extraction:")
                print("OpenAI抽出からのコンテンツプレビュー:")
                preview = openai_extracted_text[:200] + "..." if len(openai_extracted_text) > 200 else openai_extracted_text
                print(preview)
                
                # Save extracted text
                # 抽出されたテキストを保存
                save_path = save_extracted_text(openai_extracted_text, pdf_path, "_openai")
        else:
            print(f"PDF file not found at: {pdf_path}")
            print(f"PDFファイルが見つかりません: {pdf_path}")
    
    # 5. PDF processing with Google Gemini
    # 5. Google Geminiを使用したPDF処理
    if GEMINI_AVAILABLE:
        print("\n5. PDF processing with Google Gemini")
        print("5. Google Geminiを使用したPDF処理")
        print("-" * 60)
        
        pdf_path = "sample.pdf"
        if os.path.exists(pdf_path):
            print(f"Processing PDF with Gemini: {pdf_path}")
            print(f"GeminiでPDFを処理: {pdf_path}")
            
            gemini_response = load_pdf_with_gemini(pdf_path)
            
            if gemini_response:
                # Show content
                # コンテンツを表示
                print("\nResponse from Gemini:")
                print("Geminiからのレスポンス:")
                print(gemini_response)
                
                # Note about implementation
                # 実装に関する注意
                print("\nNote: In a real implementation, you would need to convert the PDF to images")
                print("注: 実際の実装では、PDFを画像に変換する必要があります")
        else:
            print(f"PDF file not found at: {pdf_path}")
            print(f"PDFファイルが見つかりません: {pdf_path}")
    
    # 6. Office document processing with Markitdown
    # 6. Markitdownを使用したオフィス文書処理
    if MARKITDOWN_AVAILABLE:
        print("\n6. Office document processing with Markitdown")
        print("6. Markitdownを使用したオフィス文書処理")
        print("-" * 60)
        
        # Check for sample office documents
        # サンプルオフィス文書を確認
        sample_files = [
            "sample.docx",  # Word document
            "sample.xlsx",  # Excel spreadsheet
            "sample.pptx",  # PowerPoint presentation
            "sample.pdf"    # PDF (alternative method)
        ]
        
        for sample_file in sample_files:
            if os.path.exists(sample_file):
                print(f"\nProcessing {sample_file} with Markitdown")
                print(f"Markitdownで {sample_file} を処理")
                
                extracted_text = load_office_with_markitdown(sample_file)
                
                if extracted_text:
                    # Show content preview
                    # コンテンツプレビューを表示
                    print(f"\nContent preview from {sample_file}:")
                    print(f"{sample_file} からのコンテンツプレビュー:")
                    preview = extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text
                    print(preview)
                    
                    # Save extracted text
                    # 抽出されたテキストを保存
                    save_path = save_extracted_text(extracted_text, sample_file, "_markitdown")
            else:
                print(f"Sample file not found: {sample_file}")
                print(f"サンプルファイルが見つかりません: {sample_file}")
    
    print("\nDocument loader sample completed")
    print("文書ローダーサンプルが完了しました")

if __name__ == "__main__":
    document_loader_sample() 