"""
Example of integrating FinderLedge with OpenAI API
FinderLedgeをOpenAI APIと統合する例

This example demonstrates how to use FinderLedge with OpenAI's function calling.
この例では、FinderLedgeをOpenAIの関数呼び出し機能と統合する方法を示します。
"""

import os
import uuid
from typing import List, Optional, Dict, Any, Union
from openai import OpenAI
from finderledge import Finder, Document

class FinderTool:
    """
    A tool class that wraps FinderLedge functionality for use with OpenAI function calling
    OpenAIの関数呼び出し機能で使用するためのFinderLedge機能をラップするツールクラス
    """
    def __init__(self, finder: Finder):
        """
        Initialize the tool with a FinderLedge instance
        FinderLedgeインスタンスでツールを初期化する

        Args:
            finder (Finder): FinderLedge instance
        """
        self.finder = finder

    def search_documents(self, query: str, mode: str = "hybrid", top_k: int = 5) -> Union[List[Dict[str, Any]], Dict[str, str]]:
        """
        Search documents using FinderLedge
        FinderLedgeを使用して文書を検索する

        Args:
            query (str): Search query
            mode (str): Search mode ("hybrid", "semantic", or "keyword")
            top_k (int): Number of results to return

        Returns:
            Union[List[Dict[str, Any]], Dict[str, str]]: List of search results or error message
        """
        try:
            if mode not in ["hybrid", "semantic", "keyword"]:
                return {"error": f"Invalid search mode: {mode}"}

            results = self.finder.search(query, top_k=top_k)
            return [
                {
                    "id": result.document.id,
                    "content": result.document.content,
                    "score": float(result.score)
                }
                for result in results
            ]
        except Exception as e:
            return {"error": str(e)}

    def add_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a document to FinderLedge
        FinderLedgeに文書を追加する

        Args:
            content (str): Document content
            metadata (Optional[Dict[str, Any]]): Document metadata

        Returns:
            str: Document ID
        """
        try:
            doc_id = str(uuid.uuid4())
            doc = Document(id=doc_id, content=content, metadata=metadata or {})
            self.finder.add_document(doc)
            return doc_id
        except Exception as e:
            return str(e)

    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from FinderLedge
        FinderLedgeから文書を削除する

        Args:
            doc_id (str): Document ID to remove

        Returns:
            bool: True if document was removed, False otherwise
        """
        try:
            self.finder.remove_document(doc_id)
            return True
        except Exception as e:
            return False

def get_tool_functions() -> List[Dict[str, Any]]:
    """
    Get the function definitions for OpenAI function calling
    OpenAIの関数呼び出し用の関数定義を取得する

    Returns:
        List[Dict[str, Any]]: List of function definitions
    """
    return [
        {
            "name": "search_documents",
            "description": "Search for documents using FinderLedge",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["hybrid", "semantic", "keyword"],
                        "description": "Search mode"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "add_document",
            "description": "Add a document to FinderLedge",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Document content"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Document metadata"
                    }
                },
                "required": ["content"]
            }
        },
        {
            "name": "remove_document",
            "description": "Remove a document from FinderLedge",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID to remove"
                    }
                },
                "required": ["doc_id"]
            }
        }
    ]

def create_finder_assistant(client: OpenAI, finder: Finder) -> Dict[str, Any]:
    """
    Create an OpenAI Assistant with FinderLedge tools
    FinderLedgeツールを持つOpenAIアシスタントを作成する

    Args:
        client (OpenAI): OpenAI client
        finder (Finder): FinderLedge instance

    Returns:
        Dict[str, Any]: Assistant configuration
    """
    return client.beta.assistants.create(
        name="finder_assistant",
        description="An assistant that can search, add, and remove documents using FinderLedge",
        model="gpt-4-turbo-preview",
        tools=[{"type": "function", "function": func} for func in get_tool_functions()]
    )

def main():
    """
    Main function demonstrating the use of FinderLedge with OpenAI
    FinderLedgeとOpenAIの使用例を示すメイン関数
    """
    try:
        # Initialize OpenAI client and FinderLedge
        # OpenAIクライアントとFinderLedgeの初期化
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        finder = Finder()
        finder_tool = FinderTool(finder)

        # Create assistant
        # アシスタントの作成
        assistant = create_finder_assistant(client, finder)

        # Example: Add a document
        # 例：文書の追加
        content = "Python is a versatile programming language used in data science and web development."
        doc_id = finder_tool.add_document(content)
        print(f"Added document with ID: {doc_id}")

        # Example: Search for documents
        # 例：文書の検索
        results = finder_tool.search_documents("Python programming")
        print("Search results:", results)

        # Example: Remove the document
        # 例：文書の削除
        removed = finder_tool.remove_document(doc_id)
        print(f"Document removed: {removed}")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 