"""
Create_database.py

This script:
1. Loads Markdown (.md) documents from a given directory.
2. Splits the documents into smaller chunks for embeddings.
3. Adds metadata (topic, source, filename) to each document.
4. Generates embeddings using OpenAI.
5. Saves the processed chunks into a persistent Chroma vector database.

Used later for RAG (Retrieval Augmented Generation).
"""

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai
from dotenv import load_dotenv
import os
import shutil

# Load environment variables (e.g., API keys) from .env file
load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

# Paths for Chroma database and data directory
CHROMA_PATH = "chroma"
DATA_PATH = "data/books"


def load_documents():
    """
    Loads all Markdown (.pdf) files from DATA_PATH using DirectoryLoader.
    Adds extra metadata (topic, source, filename) for better context tracking.
    """
    loader = DirectoryLoader(
        DATA_PATH,
        glob="*.pdf",  # only include markdown files
        loader_cls= PyPDFLoader
    )
    documents = loader.load()

    # Add metadata to each document
    for doc in documents:
        doc.metadata["topic"] = "vegetables"  # Example topic tag
        doc.metadata["source"] = doc.metadata.get("source", "unknown")  # Ensure a source always exists
        doc.metadata["filename"] = doc.metadata["source"].split("/")[-1]  # Extract filename from path

    print(f"Loaded {len(documents)} documents with metadata.")
    return documents


def split_text(documents: list[Document]):
    """
    Splits documents into smaller overlapping chunks.
    This makes embeddings more effective for search and retrieval.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,       # max characters per chunk
        chunk_overlap=100,    # overlap for context preservation
        length_function=len,  # length function to count characters
        add_start_index=True, # keep track of original positions
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Show a sample chunk and metadata for debugging
    if len(chunks) > 10:
        document = chunks[10]
        print("Sample chunk content:", document.page_content[:100])
        print("Metadata:", document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    """
    Saves document chunks into a persistent Chroma vector database.
    Deletes old DB before saving a new one.
    """
    # Remove any old database first
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create new Chroma DB and persist
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def query_database():
    # Load existing Chroma DB
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())

    # Create retriever (fetches top 3 results by similarity)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Example: query with filter (only return docs where topic == "vegetables")
    results = retriever.get_relevant_documents(
        "What are the criminal laws?",
        filter={"topic": "criminal law"}
    )

    # Show results
    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print("Content:", doc.page_content[:200])  # preview first 200 chars
        print("Metadata:", doc.metadata)

def generate_data_store():
    """
    Full pipeline:
    1. Load documents
    2. Split them into chunks
    3. Save chunks into Chroma DB
    """
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def main():
    """Runs the data store generation process."""
    generate_data_store()


# Entry point of the script
if __name__ == "__main__":
    main()
