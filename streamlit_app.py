# app.py

import streamlit as st
from dotenv import load_dotenv
from utils import (
    logger,
    create_qdrant_collection,
    process_excel_file,
    initialize_retrieval_qa,
    create_graph,
    GeminiLLM
)
from qdrant_client import QdrantClient
from langchain.embeddings import OpenAIEmbeddings
import os

# Load environment variables from .env file
load_dotenv()

# Configuration
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_ENDPOINT = os.getenv("GEMINI_ENDPOINT")
COLLECTION_NAME = "excel_documents"

# Validate essential environment variables
if not all([QDRANT_API_KEY, QDRANT_URL, GEMINI_API_KEY, GEMINI_ENDPOINT]):
    logger.critical("One or more essential environment variables are missing.")
    st.error("Configuration error: Please check your environment variables.")
    st.stop()

# Initialize Qdrant Client with error handling
try:
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        prefer_grpc=True  # Use gRPC for better performance
    )
    logger.info("Successfully connected to Qdrant Cloud.")
except Exception as e:
    logger.critical(f"Failed to connect to Qdrant Cloud: {e}")
    st.error("Connection error: Unable to connect to Qdrant Cloud.")
    st.stop()

# Initialize LangChain embeddings
try:
    embeddings = OpenAIEmbeddings()
    logger.info("Initialized OpenAI embeddings successfully.")
except Exception as e:
    logger.error(f"Failed to initialize embeddings: {e}")
    st.error("Embedding error: Unable to initialize embeddings.")
    st.stop()

# Initialize Gemini LLM
gemini_llm = GeminiLLM(
    api_key=GEMINI_API_KEY,
    endpoint=GEMINI_ENDPOINT
)

def main():
    """
    Main function to run the Streamlit application.
    """
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
            try:
                create_qdrant_collection(qdrant_client, COLLECTION_NAME, embeddings.embedding_size)
                for file in uploaded_files:
                    result = process_excel_file(file, embeddings, qdrant_client, COLLECTION_NAME)
                    if result["status"] == "success":
                        st.sidebar.success(result["message"])
                    elif result["status"] == "warning":
                        st.sidebar.warning(result["message"])
                    elif result["status"] == "error":
                        st.sidebar.error(result["message"])
            except Exception as e:
                logger.error(f"Error processing uploaded files: {e}")
                st.sidebar.error(f"Error processing uploaded files: {e}")
        else:
            st.sidebar.warning("Please upload at least one Excel file.")

    # Query Section
    st.header("Ask a Question")
    user_query = st.text_input("Enter your query here:")

    if st.button("Get Answer"):
        if user_query:
            qa = initialize_retrieval_qa(qdrant_client, COLLECTION_NAME, embeddings, gemini_llm)
            if qa:
                with st.spinner("Generating answer with Gemini..."):
                    try:
                        logger.info(f"Received user query: {user_query}")
                        answer = qa.run(user_query)
                        logger.info("Successfully generated answer.")
                        st.success("**Answer:**")
                        st.write(answer)
                    except Exception as e:
                        logger.error(f"Error generating answer: {e}")
                        st.error(f"Error generating answer: {e}")
        else:
            st.warning("Please enter a query.")

    # Visualization with LangGraph
    st.header("📈 Data Visualization with LangGraph")
    if st.button("Generate Graph"):
        try:
            logger.info("Generating data visualization graph.")
            graph = create_graph(qdrant_client, COLLECTION_NAME)
            if graph:
                st.graphviz_chart(graph.source)
                st.success("Graph generated successfully.")
            else:
                st.error("Failed to generate graph.")
        except Exception as e:
            logger.error(f"Error generating graph: {e}")
            st.error(f"Error generating graph: {e}")

    # Footer
    st.markdown("---")
    st.write("Developed by [Your Name](https://your-website.com)")

if __name__ == "__main__":
    main()