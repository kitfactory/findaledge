"""
OpenAI API Sample
OpenAI APIサンプル

This sample demonstrates interaction with OpenAI API for chat and document summarization.
このサンプルはOpenAI APIを使用したチャットと文書要約のインタラクションを示します。
"""

import os
from oneenv import load_env
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

def openai_chat_sample():
    """
    Demonstrates using OpenAI API for chat interaction
    OpenAI APIを使用したチャットインタラクションのデモンストレーション
    """
    # Load environment variables
    # 環境変数をロード
    load_env()
    
    # Check if OpenAI API key is set
    # OpenAI APIキーが設定されているか確認
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set. Please set it before running this sample.")
        print("エラー: OPENAI_API_KEY環境変数が設定されていません。このサンプルを実行する前に設定してください。")
        return
    
    print("OpenAI Chat Sample")
    print("OpenAI チャットサンプル")
    print("=" * 40)
    
    # Initialize the ChatOpenAI model
    # ChatOpenAIモデルを初期化
    chat_model = ChatOpenAI(
        model="gpt-3.5-turbo",  # or "gpt-4" if you have access
        temperature=0.7
    )
    
    # Example 1: Basic Chat Interaction
    # 例1: 基本的なチャットインタラクション
    print("\nExample 1: Basic Chat Interaction")
    print("例1: 基本的なチャットインタラクション")
    print("-" * 40)
    
    messages = [
        SystemMessage(content="You are a helpful assistant that specializes in Japanese culture. Keep your answers brief."),
        HumanMessage(content="日本の四季について教えてください。")
    ]
    
    response = chat_model.invoke(messages)
    print("User: 日本の四季について教えてください。")
    print("ユーザー: 日本の四季について教えてください。")
    print(f"Assistant: {response.content}")
    print(f"アシスタント: {response.content}")
    
    # Example 2: Using prompt templates
    # 例2: プロンプトテンプレートの使用
    print("\nExample 2: Using Prompt Templates")
    print("例2: プロンプトテンプレートの使用")
    print("-" * 40)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that specializes in {topic}. Keep your answers brief."),
        ("human", "{input}")
    ])
    
    chain = prompt | chat_model | StrOutputParser()
    
    response = chain.invoke({
        "topic": "Japanese cuisine",
        "input": "寿司の歴史について教えてください。"
    })
    
    print("User: 寿司の歴史について教えてください。")
    print("ユーザー: 寿司の歴史について教えてください。")
    print(f"Assistant: {response}")
    print(f"アシスタント: {response}")
    
    # Example 3: Interactive chat session
    # 例3: インタラクティブなチャットセッション
    print("\nExample 3: Interactive Chat (Type 'exit' to end)")
    print("例3: インタラクティブチャット（終了するには 'exit' と入力）")
    print("-" * 40)
    
    chat_history = [
        SystemMessage(content="You are a helpful assistant that specializes in Japanese culture. Keep your answers brief and in Japanese.")
    ]
    
    print("Starting chat session. Type 'exit' to end the conversation.")
    print("チャットセッションを開始します。会話を終了するには 'exit' と入力してください。")
    
    # Interactive chat loop - max 3 turns for this sample
    # インタラクティブチャットループ - このサンプルでは最大3ターン
    for i in range(3):
        user_input = input("\nYou: ")
        print(f"あなた: {user_input}")
        
        if user_input.lower() == "exit":
            print("Ending chat session.")
            print("チャットセッションを終了します。")
            break
        
        chat_history.append(HumanMessage(content=user_input))
        response = chat_model.invoke(chat_history)
        chat_history.append(response)
        
        print(f"Assistant: {response.content}")
        print(f"アシスタント: {response.content}")

def summarize_document_sample():
    """
    Demonstrates document summarization using OpenAI API
    OpenAI APIを使用した文書要約のデモンストレーション
    """
    # Load environment variables
    # 環境変数をロード
    load_env()
    
    # Check if OpenAI API key is set
    # OpenAI APIキーが設定されているか確認
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set. Please set it before running this sample.")
        print("エラー: OPENAI_API_KEY環境変数が設定されていません。このサンプルを実行する前に設定してください。")
        return
    
    print("\nDocument Summarization Sample")
    print("文書要約サンプル")
    print("=" * 40)
    
    # Initialize the ChatOpenAI model
    # ChatOpenAIモデルを初期化
    chat_model = ChatOpenAI(
        model="gpt-3.5-turbo-16k",  # Use a model with larger context window
        temperature=0
    )
    
    # Sample document in Japanese
    # 日本語のサンプル文書
    document = """
日本は、独特の文化と歴史を持つ島国です。四季折々の自然の美しさと、伝統と現代が共存する社会が特徴です。

歴史的には、縄文時代から始まり、飛鳥時代、奈良時代、平安時代、鎌倉時代、室町時代、江戸時代を経て、明治維新により近代化が進みました。
特に江戸時代は260年以上続いた平和な時代で、独自の文化が花開きました。

日本の伝統文化には、茶道、華道、書道などがあります。また、歌舞伎や能といった伝統芸能も世界的に有名です。
和食は2013年にユネスコ無形文化遺産に登録され、その健康的な特性と美的感覚が評価されています。

現代の日本は、技術革新のリーダーとしても知られ、自動車や電子機器などの分野で世界をリードしてきました。
東京、大阪、京都などの大都市は、超高層ビルと歴史的建造物が共存する独特の景観を形成しています。

地理的には、北海道、本州、四国、九州の4つの主要な島と、多くの小さな島々から成り立っています。
富士山は日本のシンボルであり、多くの日本人や外国人観光客に愛されています。

日本の四季は明確で、春は桜、夏は緑と祭り、秋は紅葉、冬は雪景色と、それぞれ特徴的な美しさがあります。
この四季の変化は、日本の文学や芸術にも大きな影響を与えてきました。

日本の社会は、調和と集団意識を重んじる傾向があり、「和」の精神が根付いています。
近年では少子高齢化や働き方改革など、現代社会特有の課題にも直面していますが、
伝統を大切にしながらも革新を続ける姿勢は、日本社会の特徴と言えるでしょう。
    """
    
    # Create a summarization prompt
    # 要約プロンプトを作成
    summarization_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that summarizes documents in Japanese. Keep the summary concise."),
        ("human", "次の文書を200字程度で要約してください。文書: {document}")
    ])
    
    # Create the summarization chain
    # 要約チェーンを作成
    summarization_chain = summarization_prompt | chat_model | StrOutputParser()
    
    # Generate a summary
    # 要約を生成
    summary = summarization_chain.invoke({"document": document})
    
    print("Original Document:")
    print("元の文書:")
    print(document)
    print("\nSummary:")
    print("要約:")
    print(summary)
    
    # Example of extracting key information
    # 重要情報の抽出の例
    print("\nKey Information Extraction:")
    print("重要情報の抽出:")
    print("-" * 40)
    
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that extracts key information from documents in Japanese."),
        ("human", "次の文書から重要なポイントを5つ、箇条書きで抽出してください。文書: {document}")
    ])
    
    extraction_chain = extraction_prompt | chat_model | StrOutputParser()
    key_points = extraction_chain.invoke({"document": document})
    
    print("Key Points:")
    print("重要ポイント:")
    print(key_points)

if __name__ == "__main__":
    # Run the OpenAI chat sample
    # OpenAIチャットサンプルを実行
    openai_chat_sample()
    
    # Run the document summarization sample
    # 文書要約サンプルを実行
    summarize_document_sample() 