# app.py

import os
import streamlit as st
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langgraph import LangGraph
from gemini_llm import GeminiLLM  # Import the custom Gemini LLM

# Load environment variables
load_dotenv()

# Configuration
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_ENDPOINT = os.getenv("GEMINI_ENDPOINT")

# Initialize Qdrant Client
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=True  # Use gRPC for better performance
)

# Define Qdrant collection name
COLLECTION_NAME = "excel_documents"

# Initialize LangChain embeddings
embeddings = OpenAIEmbeddings()  # Continue using OpenAI embeddings or switch to Gemini-compatible embeddings if available

# Initialize LangGraph
lang_graph = LangGraph()

# Function to create collection if not exists
def create_collection():
    existing_collections = [col.name for col in qdrant_client.get_collections().collections]
    if COLLECTION_NAME not in existing_collections:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "size": embeddings.embedding_size,
                "distance": Distance.COSINE
            }
        )
        st.success(f"Collection '{COLLECTION_NAME}' created in Qdrant.")

# Function to process and store Excel data
def process_and_store(file):
    try:
        # Read Excel file
        xls = pd.ExcelFile(file)
        data_frames = {sheet: xls.parse(sheet) for sheet in xls.sheet_names}

        documents = []
        for sheet_name, df in data_frames.items():
            # Convert each row to a string
            for idx, row in df.iterrows():
                text = row.to_json()
                documents.append({
                    "id": f"{sheet_name}_{idx}",
                    "text": text
                })

        # Create embeddings
        texts = [doc["text"] for doc in documents]
        embedding_vectors = embeddings.embed_documents(texts)

        # Prepare points for Qdrant
        points = [
            PointStruct(id=doc["id"], vector=vector, payload={"text": doc["text"]})
            for doc, vector in zip(documents, embedding_vectors)
        ]

        # Upsert points into Qdrant
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
        st.success(f"Successfully uploaded and stored {len(documents)} records from {file.name}.")

    except Exception as e:
        st.error(f"Error processing file {file.name}: {e}")

# Function to initialize LangChain RetrievalQA with Gemini
def initialize_qa():
    # Initialize Qdrant as vector store
    vector_store = Qdrant(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings
    )

    # Initialize Gemini LLM
    gemini_llm = GeminiLLM(
        api_key=GEMINI_API_KEY,
        endpoint=GEMINI_ENDPOINT
    )

    # Initialize RetrievalQA chain with Gemini
    qa = RetrievalQA.from_chain_type(
        llm=gemini_llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        verbose=True
    )

    return qa

# Main Streamlit App
def main():
    st.set_page_config(page_title="Excel RAG with Qdrant and Gemini", layout="wide")
    st.title("📊 Excel Retrieval-Augmented Generation (RAG) App with Gemini")
    st.write("""
        Upload your Excel files, and ask questions about their content.
        The app uses Qdrant Cloud for vector storage and Google Gemini for generating responses.
    """)

    # Sidebar for file upload
    st.sidebar.header("Upload Excel Files")
    uploaded_files = st.sidebar.file_uploader(
        "Upload one or more Excel files", type=["xlsx", "xls"], accept_multiple_files=True
    )

    if st.sidebar.button("Process Files"):
        if uploaded_files:
            create_collection()
            for file in uploaded_files:
                process_and_store(file)
        else:
            st.sidebar.warning("Please upload at least one Excel file.")

    # Query Section
    st.header("Ask a Question")
    user_query = st.text_input("Enter your query here:")

    if st.button("Get Answer"):
        if user_query:
            qa = initialize_qa()
            with st.spinner("Generating answer with Gemini..."):
                try:
                    answer = qa.run(user_query)
                    st.success("**Answer:**")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
        else:
            st.warning("Please enter a query.")

    # Visualization with LangGraph
    st.header("📈 Data Visualization with LangGraph")
    if st.button("Generate Graph"):
        try:
            graph = lang_graph.create_graph(collection=COLLECTION_NAME, client=qdrant_client)
            st.graphviz_chart(graph.source)
            st.success("Graph generated successfully.")
        except Exception as e:
            st.error(f"Error generating graph: {e}")

    # Footer
    st.markdown("---")
    st.write("Developed by [Your Name](https://your-website.com)")

if __name__ == "__main__":
    main()