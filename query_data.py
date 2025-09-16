# """
# This script loads Markdown (.md) files from a directory, splits them into chunks,
# adds metadata, generates embeddings using OpenAI, and stores them in a Chroma vector database.
# """

# from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
# import openai
# from dotenv import load_dotenv
# import os
# import shutil

# # Load environment variables from .env (to get OpenAI API key)
# load_dotenv()
# openai.api_key = os.environ['OPENAI_API_KEY']

# # Paths for Chroma DB and data
# CHROMA_PATH = "chroma"
# DATA_PATH = "data/books"


# def load_documents():
#     """
#     Loads all Markdown (.md) documents from the DATA_PATH directory.
#     Adds metadata (topic, source, filename) to each document.
#     """
#     loader = DirectoryLoader(
#         DATA_PATH,
#         glob="*.pdf",  # only load markdown files
#         loader_cls= PyPDFLoader

#     )
#     documents = loader.load()

#     # Add custom metadata for each document
#     for doc in documents:
#         doc.metadata["topic"] = "vegetables"  # custom topic label
#         doc.metadata["source"] = doc.metadata.get("source", "unknown")  # fallback if no source exists
#         doc.metadata["filename"] = doc.metadata["source"].split("/")[-1]  # extract filename from source path

#     print(f"Loaded {len(documents)} documents with metadata.")
#     return documents


# def split_text(documents: list[Document]):
#     """
#     Splits documents into smaller overlapping chunks
#     for better embedding and retrieval performance.
#     """
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=300,   # max characters per chunk
#         chunk_overlap=100,  # overlap to preserve context
#         length_function=len,
#         add_start_index=True,  # keep track of original index
#     )
#     chunks = text_splitter.split_documents(documents)
#     print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

#     # Show sample chunk for debugging
#     if len(chunks) > 10:
#         document = chunks[10]
#         print("Sample chunk content:", document.page_content[:100])
#         print("Metadata:", document.metadata)

#     return chunks


# def save_to_chroma(chunks: list[Document]):
#     """
#     Saves document chunks to a Chroma persistent database.
#     Old DB is deleted before saving new one.
#     """
#     # Remove existing DB if it exists
#     if os.path.exists(CHROMA_PATH):
#         shutil.rmtree(CHROMA_PATH)

#     # Create and persist new vector database
#     db = Chroma.from_documents(
#         chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
#     )
#     db.persist()
#     print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


# def generate_data_store():
#     """
#     Complete pipeline:
#     1. Load documents
#     2. Split them into chunks
#     3. Save chunks into Chroma DB
#     """
#     documents = load_documents()
#     chunks = split_text(documents)
#     save_to_chroma(chunks)


# def main():
#     """Main entry point for script execution."""
#     generate_data_store()


# if __name__ == "__main__":
#     main()



################################################RUNNING VERION BELOW################################################33

# """
# This script takes a user query, searches a Chroma vector database for relevant 
# documents using OpenAI embeddings, and generates an AI-based response 
# using ChatGPT with the retrieved context.
# """

# import argparse
# # from dataclasses import dataclass
# from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings
# from langchain_openai import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate
# from langchain_chroma import Chroma
# from dotenv import load_dotenv

# # Load environment variables (like API keys)
# load_dotenv()

# # Path where Chroma DB is stored
# CHROMA_PATH = "chroma"

# # Template for generating responses
# PROMPT_TEMPLATE = """
# You are a legal assistant specialized in anti-narcotics criminal law. You provide accurate 
# information about drug-related criminal statutes, penalties, procedures, and legal precedents.

# You ONLY answer based on the legal documents and materials provided to you. You do not use 
# your general knowledge about criminal law or make assumptions. You strictly adhere to the 
# specific legal texts, case law, and regulatory materials given.

# If the answer is not found in the provided materials, respond with: "I don't have that 
# information in the provided legal documents."

# When citing legal information, always reference the specific statute, regulation, or case 
# from the provided materials.

# {context}

# ---Answer the question based on the above context: {question}
# """

# def main():
#     """
#     Main function: 
#     1. Reads query from CLI.
#     2. Searches vector DB for relevant context.
#     3. Sends context + query to ChatGPT.
#     4. Prints AI-generated response with sources.
#     """
#     # Parse CLI argument (query text)
#     parser = argparse.ArgumentParser()
#     parser.add_argument("query_text", type=str, help="The query text.")
#     args = parser.parse_args()
#     query_text = args.query_text

#     # Load embedding model and vector DB
#     embedding_function = OpenAIEmbeddings()
#     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

#     # Search DB with similarity and relevance scores
#     results = db.similarity_search_with_relevance_scores(query_text, k=3)

#     # If no good match found
#     if len(results) == 0 or results[0][1] < 0.7:
#         print(f"Unable to find matching results.")
#         return

#     # Collect context text from search results
#     context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

#     # Format prompt with context + question
#     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     prompt = prompt_template.format(context=context_text, question=query_text)
#     print(prompt)

#     # Use ChatGPT model to generate response
#     model = ChatOpenAI(model="gpt-4o-mini")
#     response_text = model.predict(prompt)

#     # Extract sources metadata from documents
#     sources = [doc.metadata.get("source", None) for doc, _score in results]

#     # Print final response with sources
#     formatted_response = f"Response: {response_text}\nSources: {sources}"
#     print(formatted_response)


# if __name__ == "__main__":
#     main()


#################################################STREAMLIT APP BELOW with GPT VERSION################################################33


import streamlit as st
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Paths
CHROMA_PATH = "chroma"

# Prompt template
PROMPT_TEMPLATE = """
You are ANF Academy EduBot.
- If the context contains relevant information, answer strictly based on it.
- If the context does NOT have the answer, then act as a friendly assistant and help the user conversationally.
- Always be polite, supportive, and clear.
- For legal questions, cite the statute/section numbers from the documents (not filenames).
- For general questions, give helpful and friendly answers.

Context:
{context}

Conversation history:
{history}

User's question: {question}
"""

# Load DB once
embedding_function = OpenAIEmbeddings()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# Streamlit app
# st.title("ANF Academy EduBot ")

st.markdown("""
<style>
/* Remove Streamlit container padding */
[data-testid="stAppViewContainer"] {
    padding: 0;
    margin: 0;
    background-color: white;
}

/* Chat container */
.chat {
    margin: 0;
    padding: 0;
    width: 100%;
    font-family: sans-serif;
}

/* Header full width */
.header {
    width: 100%;
    background-color: #072e22;
    color: #fbfbfb;
    text-align: center;
    padding: 20px 0;
    margin: 0;
}
.header img {
    width: 60px;
    height: 60px;
    border-radius: 10px;
    display: block;
    margin: 0 auto 10px auto;
}
.header .title h1 {
    font-size: 30px;
    margin: 0;
    color: #fbfbfb;
}
</style>

<div class="chat">
  <div class="header">
    <img src="https://upload.wikimedia.org/wikipedia/en/7/70/Anti-Narcotics_Force_Logo.png"/>
    <div class="title"><h1>ANF Academy Educational Chatbot</h1></div>
  </div>
</div>
""", unsafe_allow_html=True)

# Session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if query := st.chat_input("Ask a legal question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Build conversation history for the model
    history = ""
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    # Search DB
    results = db.similarity_search_with_relevance_scores(query, k=10)

    # if len(results) == 0 or results[0][1] < 0.7:
    #     response_text = "Unable to find matching results."
    # else:
    #     context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    # Build context if there are results
    if results and results[0][1] >= 0.7:
        context_text = "\n\n".join([doc.page_content for doc, _ in results])
    else:
        context_text = ""  # empty context if no good match

        # Combine history + context into the prompt
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(
            context=context_text,
            history=history,  # optional: last 3 messages
            question=query    # just the user query
        )

        model = ChatOpenAI(model="gpt-4o-mini")
        context_text = model.predict(prompt)

    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": context_text})
    with st.chat_message("assistant"):
        st.markdown(context_text)

# Replay all messages on reload
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])




#################################################STREAMLIT APP BELOW with Gemini VERSION################################################33


# import streamlit as st
# # from langchain_community.vectorstores import Chroma
# from langchain_chroma import Chroma
# # from langchain_openai import OpenAIEmbeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI  # ✅ Gemini
# from langchain.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# import os

# import asyncio
# import nest_asyncio

# try:
#     asyncio.get_running_loop()
# except RuntimeError:
#     asyncio.set_event_loop(asyncio.new_event_loop())

# nest_asyncio.apply()

# # Load env variables
# load_dotenv()

# # Paths
# CHROMA_PATH = "chroma"

# # Prompt template (unchanged)
# PROMPT_TEMPLATE = """
# You are a legal assistant specialized in anti-narcotics criminal law. 
# You provide accurate information strictly from the legal documents and materials provided.

# You MUST:
# - Answer ONLY from the given context.
# - If the answer is not found in the provided materials, respond with: 
#   "I don't have that information in the provided legal documents."
# - When citing, directly quote the statute numbers, section references, 
#   or case names mentioned inside the documents — NOT the file name or source path.

# Context:
# {context}

# ---Answer the question using only the above context: {question}
# """

# # Load DB once
# # embedding_function = OpenAIEmbeddings()
# embedding_function = GoogleGenerativeAIEmbeddings(
#     model="models/embedding-001",
#     google_api_key=os.getenv("OPENAI_API_KEY")  # since you stored Gemini key in OPENAI_API_KEY
# )

# db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# # Streamlit app
# st.title("ANF Academy EduBot ")

# # Session state for chat
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat history
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# # User input
# if query := st.chat_input("Ask a legal question..."):
#     # Add user message
#     st.session_state.messages.append({"role": "user", "content": query})
#     with st.chat_message("user"):
#         st.markdown(query)

#     # Build conversation history for the model
#     conversation = ""
#     for msg in st.session_state.messages:
#         role = "User" if msg["role"] == "user" else "Assistant"
#         conversation += f"{role}: {msg['content']}\n"

#     # Search DB
#     results = db.similarity_search_with_relevance_scores(query, k=3)

#     if len(results) == 0 or results[0][1] < 0.7:
#         response_text = "Unable to find matching results."
#     else:
#         context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

#         # Combine history + context into the prompt
#         prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#         prompt = prompt_template.format(
#             context=context_text,
#             question=f"{conversation}\nUser: {query}"
#         )

#         # ✅ Use Gemini model instead of OpenAI
#         model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
#         response_text = model.predict(prompt)

#     # Add assistant message
#     st.session_state.messages.append({"role": "assistant", "content": response_text})
#     with st.chat_message("assistant"):
#         st.markdown(response_text)

# # Replay all messages on reload
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])



    