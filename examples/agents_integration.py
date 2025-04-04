"""
Example of integrating FindaLedge with OpenAI API
FindaLedgeをOpenAI APIと統合する例

This example demonstrates how to use FindaLedge with OpenAI's function calling.
この例では、FindaLedgeをOpenAIの関数呼び出し機能と統合する方法を示します。
"""

import os
import json
import asyncio
from openai import OpenAI, AsyncOpenAI
from typing import List, Dict, Any
from findaledge import Finder, Document # Assuming FinderLedge exports Finder and Document
from oneenv import load_dotenv

# --- Tool Definition / ツールの定義 ---
class FindaLedgeTool:
    """
    A tool class that wraps FindaLedge functionality for use with OpenAI function calling
    OpenAIの関数呼び出し機能で使用するためのFindaLedge機能をラップするツールクラス
    """
    def __init__(self, finder: Finder):
        """
        Initialize the tool with a FindaLedge instance
        FindaLedgeインスタンスでツールを初期化する

        Args:
            finder (Finder): FindaLedge instance
                         FindaLedgeインスタンス
        """
        self.finder = finder

    def search_documents(self, query: str, top_k: int = 3, search_mode: str = "hybrid") -> str:
        """
        Search documents using FindaLedge
        FindaLedgeを使用して文書を検索する

        Args:
            query (str): Search query / 検索クエリ
            top_k (int): Number of results to return / 返す結果の数
            search_mode (str): Search mode ("hybrid", "vector", "keyword") / 検索モード

        Returns:
            str: Formatted search results / 整形された検索結果
        """
        print(f"[Tool] Searching documents with query: '{query}', mode: {search_mode}, top_k: {top_k}")
        try:
            results: List[Document] = self.finder.search(query, search_mode=search_mode, top_k=top_k)
            if not results:
                return "No relevant documents found."
            
            # Format results for the LLM
            # LLM用に結果をフォーマット
            formatted = []
            for i, doc in enumerate(results):
                source = doc.metadata.get('source', 'N/A')
                score = doc.metadata.get('relevance_score', 0.0)
                content_preview = doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else "")
                formatted.append(f"Result {i+1} (Score: {score:.4f}, Source: {source}):\n{content_preview}")
            return "\n\n".join(formatted)
        except Exception as e:
            print(f"[Tool Error] Error during search: {e}")
            return f"An error occurred during search: {str(e)}"

    def add_document(self, file_path: str) -> str:
        """
        Add a document to FindaLedge
        FindaLedgeに文書を追加する

        Args:
            file_path (str): Path to the document file / ドキュメントファイルへのパス

        Returns:
            str: Confirmation message or error / 確認メッセージまたはエラー
        """
        print(f"[Tool] Adding document: {file_path}")
        try:
            # Assuming FindaLedge instance has an add_document method
            # FindaLedgeインスタンスにadd_documentメソッドがあると仮定
            # This might need adjustments based on the actual FindaLedge API
            # これは実際のFindaLedge APIに基づいて調整が必要になる場合があります
            parent_ids = self.finder.add_document(file_path) # Adjust if finder is not the main interface
            return f"Successfully added document(s) from '{file_path}'. Parent IDs: {parent_ids}"
        except FileNotFoundError:
             return f"Error: File not found at '{file_path}'."
        except Exception as e:
            print(f"[Tool Error] Error adding document: {e}")
            return f"An error occurred while adding the document: {str(e)}"

    def remove_document(self, document_id: str) -> str:
        """
        Remove a document from FindaLedge
        FindaLedgeから文書を削除する

        Args:
            document_id (str): The ID of the document to remove / 削除するドキュメントのID

        Returns:
            str: Confirmation message or error / 確認メッセージまたはエラー
        """
        print(f"[Tool] Removing document: {document_id}")
        try:
            # Assuming FindaLedge instance has a remove_document method
            # FindaLedgeインスタンスにremove_documentメソッドがあると仮定
            self.finder.remove_document(document_id) # Adjust if finder is not the main interface
            return f"Successfully removed document with ID: '{document_id}'."
        except Exception as e:
            print(f"[Tool Error] Error removing document: {e}")
            return f"An error occurred while removing the document: {str(e)}"

    def get_tools_definitions(self) -> List[Dict[str, Any]]:
        """
        Returns the list of tool definitions for OpenAI function calling.
        OpenAI関数呼び出し用のツール定義のリストを返します。
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_documents",
                    "description": "Search for relevant documents within the knowledge base using FindaLedge",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query"},
                            "top_k": {"type": "integer", "description": "Number of results to return", "default": 3},
                            "search_mode": {"type": "string", "description": "Search mode: hybrid, vector, or keyword", "enum": ["hybrid", "vector", "keyword"], "default": "hybrid"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "add_document",
                    "description": "Add a new document to the FindaLedge knowledge base from a file path",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "The full path to the document file to add"}
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "remove_document",
                    "description": "Remove a document from the FindaLedge knowledge base using its ID",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "document_id": {"type": "string", "description": "The unique ID of the document to remove"}
                        },
                        "required": ["document_id"]
                    }
                }
            }
        ]

    def call_tool(self, tool_call) -> Dict[str, Any]:
        """
        Call the appropriate FindaLedge method based on the tool call object.
        ツールコールオブジェクトに基づいて適切なFindaLedgeメソッドを呼び出します。
        """
        function_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        
        result = "Error: Tool not found."
        if function_name == "search_documents":
            result = self.search_documents(**args)
        elif function_name == "add_document":
            result = self.add_document(**args)
        elif function_name == "remove_document":
            result = self.remove_document(**args)
            
        return {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": result,
        }

# --- Main Execution / メイン実行 ---
async def main():
    """
    Main function demonstrating the use of FindaLedge with OpenAI
    FindaLedgeとOpenAIの使用例を示すメイン関数
    """
    load_dotenv()
    # Initialize OpenAI client and FindaLedge
    # OpenAIクライアントとFindaLedgeの初期化
    # Ensure your OPENAI_API_KEY is set in your environment or .env file
    # 環境または.envファイルにOPENAI_API_KEYが設定されていることを確認してください
    try:
        # Use AsyncOpenAI for async operations
        #非同期操作にはAsyncOpenAIを使用
        client = AsyncOpenAI()
        # Make a simple test call to verify API key
        # APIキーを検証するための簡単なテスト呼び出し
        await client.models.list()
        print("OpenAI client initialized successfully.")
        print("OpenAIクライアントは正常に初期化されました。")
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        print(f"OpenAIクライアントの初期化中にエラーが発生しました: {e}")
        print("Please ensure your OPENAI_API_KEY is set correctly.")
        print("OPENAI_API_KEYが正しく設定されていることを確認してください。")
        return

    try:
        # Assuming FindaLedge can be initialized like this
        # FindaLedgeがこのように初期化できると仮定
        # Adjust path as needed
        # 必要に応じてパスを調整
        persist_dir = project_root / "study" / "findaledge_openai_example_data"
        ledge = FindaLedge(persist_directory=str(persist_dir))
        print(f"FindaLedge initialized. Data directory: {persist_dir}")
        print(f"FindaLedgeは初期化されました。データディレクトリ: {persist_dir}")
        
        # Example: Add a document if the store is empty (optional)
        # 例: ストアが空の場合にドキュメントを追加（オプション）
        # You might need a way to check if documents exist in FindaLedge
        # FindaLedgeにドキュメントが存在するか確認する方法が必要になる場合があります
        # if not ledge.has_documents(): # Hypothetical method
        sample_file_path = project_root / "study" / "temp_docs" / "sample.md" # Use one of the created sample files
        if sample_file_path.exists():
             print(f"Adding sample document '{sample_file_path}'...")
             ledge.add_document(str(sample_file_path))
        else:
             print(f"Sample document '{sample_file_path}' not found, skipping initial add.")

    except Exception as e:
        print(f"Error initializing FindaLedge: {e}")
        print(f"FindaLedgeの初期化中にエラーが発生しました: {e}")
        return

    # Create the tool wrapper
    # ツールラッパーを作成
    findaledge_tool = FindaLedgeTool(ledge) # Pass the initialized FindaLedge instance
                                        # 初期化されたFindaLedgeインスタンスを渡す

    messages = [
        {"role": "system", "content": "You are a helpful assistant that can access a knowledge base about FindaLedge."}, 
        {"role": "user", "content": "What is FindaLedge? Search the knowledge base."}
    ]
    tools = findaledge_tool.get_tools_definitions()
    model = "gpt-3.5-turbo" # Or gpt-4

    print(f"\nSending request to {model}...")
    print(f"{model}にリクエストを送信中...")

    try:
        # First API call
        # 最初のAPI呼び出し
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto", 
        )
        response_message = response.choices[0].message
        messages.append(response_message) # Add assistant's response
                                          # アシスタントの応答を追加

        # Check if the model wants to call a tool
        # モデルがツールを呼び出したいかどうかを確認
        tool_calls = response_message.tool_calls
        if tool_calls:
            print("\nModel wants to call tools:", [tc.function.name for tc in tool_calls])
            print("モデルはツールを呼び出したがっています:", [tc.function.name for tc in tool_calls])
            
            # Call the tools and gather results
            # ツールを呼び出し、結果を収集
            tool_responses = []
            for tool_call in tool_calls:
                tool_response = findaledge_tool.call_tool(tool_call)
                tool_responses.append(tool_response)
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_call.function.name,
                    "content": tool_response["content"],
                })
            
            # Second API call with tool responses
            # ツール応答を含む2回目のAPI呼び出し
            print(f"\nSending tool responses back to {model}...")
            print(f"{model}にツール応答を返信中...")
            second_response = await client.chat.completions.create(
                model=model,
                messages=messages,
            )
            final_response_message = second_response.choices[0].message
            print("\nFinal Assistant Response / 最終アシスタント応答:")
            print(final_response_message.content)
        else:
            # Model did not call a tool
            # モデルはツールを呼び出しませんでした
            print("\nAssistant Response (no tool call) / アシスタント応答 (ツール呼び出しなし):")
            print(response_message.content)

    except Exception as e:
        print(f"\nAn error occurred during the OpenAI API call: {e}")
        print(f"OpenAI API呼び出し中にエラーが発生しました: {e}")

if __name__ == "__main__":
    # Run the async main function
    # 非同期メイン関数を実行
    # Ensure sample files are created before running (e.g., by document_loader_sample)
    # 実行前にサンプルファイルが作成されていることを確認（例：document_loader_sampleによって）
    # Consider adding file creation here if run independently
    # 個別に実行する場合は、ここにファイル作成を追加することを検討
    # from study.document_loader_sample import create_sample_files, cleanup_sample_files
    # create_sample_files()
    asyncio.run(main())
    # cleanup_sample_files() # Optional cleanup
                            # オプションのクリーンアップ

</rewritten_file> 