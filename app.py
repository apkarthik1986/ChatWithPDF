"""
Chat with PDF Application using Chainlit, Google Gemini API, LlamaParse, and ChromaDB
"""

import os
from typing import List

import chainlit as cl
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from llama_parse import LlamaParse

# Load environment variables
load_dotenv()

# Configuration constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_RETRIEVED_DOCS = 3

# Initialize Google Gemini LLM and Embeddings
def get_llm():
    """Initialize the Google Gemini LLM."""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3,
    )


def get_embeddings():
    """Initialize Google Generative AI Embeddings."""
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


def process_pdf(file_path: str) -> List[Document]:
    """Load and split PDF into chunks using LlamaParse."""
    # Initialize LlamaParse
    parser = LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        result_type="markdown",
    )

    # Parse the PDF
    parsed_documents = parser.load_data(file_path)

    # Convert to LangChain Document format
    documents = [
        Document(page_content=doc.text, metadata={"source": file_path})
        for doc in parsed_documents
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_vectorstore(chunks: List[Document]):
    """Create ChromaDB vector store from document chunks."""
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
    )
    return vectorstore


@cl.on_chat_start
async def on_chat_start():
    """Handle chat start - prompt user to upload a PDF."""
    # Check for API keys
    if not os.getenv("GOOGLE_API_KEY"):
        await cl.Message(
            content="⚠️ Please set your GOOGLE_API_KEY in the .env file to use this application."
        ).send()
        return

    if not os.getenv("LLAMA_CLOUD_API_KEY"):
        await cl.Message(
            content="⚠️ Please set your LLAMA_CLOUD_API_KEY in the .env file to use LlamaParse."
        ).send()
        return

    files = None

    # Wait for the user to upload a PDF file
    while files is None:
        files = await cl.AskFileMessage(
            content="Welcome! Please upload a PDF file to start chatting with it.",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    # Show processing message
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # Process the PDF
    chunks = process_pdf(file.path)

    # Create vector store
    vectorstore = create_vectorstore(chunks)

    # Store vectorstore in user session
    cl.user_session.set("vectorstore", vectorstore)
    cl.user_session.set("llm", get_llm())

    # Update message
    msg.content = f"✅ `{file.name}` processed successfully! You can now ask questions about the document."
    await msg.update()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle user messages and generate responses."""
    vectorstore = cl.user_session.get("vectorstore")
    llm = cl.user_session.get("llm")

    if not vectorstore or not llm:
        await cl.Message(
            content="Please upload a PDF file first."
        ).send()
        return

    # Perform similarity search
    docs = vectorstore.similarity_search(message.content, k=MAX_RETRIEVED_DOCS)

    # Build context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in docs])

    # Create prompt
    prompt = f"""Based on the following context from a PDF document, answer the user's question.
If the answer is not in the context, say "I couldn't find information about that in the document."

Context:
{context}

Question: {message.content}

Answer:"""

    # Generate response
    response = await cl.make_async(llm.invoke)(prompt)

    # Send response
    await cl.Message(content=response.content).send()
