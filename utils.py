# utils.py

import os
import logging
import requests
from typing import Optional, List
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance
from dotenv import load_dotenv
from langgraph import LangGraph

# Importing necessary classes from langchain
try:
    # Import Qdrant from langchain.vectorstores (adjust if the module path has changed)
    from langchain.vectorstores.qdrant import Qdrant
    from langchain.embeddings.openai import OpenAIEmbeddings  # Adjust if the module path has changed
    from langchain.chains import RetrievalQA
    from langchain.llms.base import LLM
except ModuleNotFoundError as e:
    raise ImportError("Ensure that `langchain` and its dependencies are installed and compatible.") from e

# Load environment variables from .env file
load_dotenv()

def setup_logging() -> logging.Logger:
    """
    Configures the logging settings for the application.
    Logs are written to both the console and a file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_level = os.getenv("LOGGING_LEVEL", "INFO").upper()
    log_file = os.getenv("LOG_FILE", "app.log")

    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    logger.propagate = False  # Prevent log messages from being duplicated in the root logger

    # Define the format for log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console handler for logging to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(console_handler)

    # File handler for logging to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# Initialize logger
logger = setup_logging()

class GeminiLLM(LLM):
    """
    Custom LLM wrapper for Google Gemini.

    This class integrates Google Gemini's API with LangChain by implementing
    the necessary methods to send prompts and receive generated text.

    Attributes:
        api_key (str): API key for authenticating with Gemini's API.
        endpoint (str): URL endpoint for Gemini's API.
    """
    api_key: str
    endpoint: str

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Sends a prompt to Gemini's API and retrieves the generated text.

        Args:
            prompt (str): The input text prompt for the LLM.
            stop (Optional[List[str]]): List of stop tokens to terminate generation.

        Returns:
            str: The generated response from Gemini.
        """
        logger.debug("Preparing to send prompt to Gemini API.")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.7,
            "stop": stop if stop else [],
        }

        try:
            logger.info("Sending request to Gemini API.")
            response = requests.post(self.endpoint, headers=headers, json=payload)
            response.raise_for_status()
            logger.debug("Received response from Gemini API.")
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
            raise ValueError(f"Gemini API HTTP Error: {http_err}") from http_err
        except Exception as err:
            logger.error(f"Unexpected error occurred: {err}")
            raise ValueError(f"Gemini API Error: {err}") from err

        try:
            data = response.json()
            generated_text = data.get("choices", [{}])[0].get("text", "").strip()
            logger.info("Successfully retrieved generated text from Gemini.")
            return generated_text
        except (KeyError, TypeError, IndexError) as parse_err:
            logger.error(f"Error parsing Gemini API response: {parse_err}")
            raise ValueError(f"Error parsing Gemini API response: {parse_err}") from parse_err

    @property
    def _llm_type(self) -> str:
        """Specifies the type of LLM."""
        return "gemini"

def create_qdrant_collection(qdrant_client: QdrantClient, collection_name: str, embedding_size: int):
    """
    Creates a Qdrant collection if it does not already exist.
    """
    try:
        existing_collections = [col.name for col in qdrant_client.get_collections().collections]
        if collection_name not in existing_collections:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={"size": embedding_size, "distance": Distance.COSINE}
            )
            logger.info(f"Created Qdrant collection '{collection_name}'.")
        else:
            logger.debug(f"Qdrant collection '{collection_name}' already exists.")
    except Exception as e:
        logger.error(f"Error creating Qdrant collection: {e}")
        raise

def process_excel_file(file, embeddings: OpenAIEmbeddings, qdrant_client: QdrantClient, collection_name: str):
    """
    Processes an uploaded Excel file and stores its content in Qdrant.
    """
    logger.info(f"Processing file: {file.name}")
    try:
        xls = pd.ExcelFile(file)
        data_frames = {sheet: xls.parse(sheet) for sheet in xls.sheet_names}
        logger.debug(f"Parsed sheets: {list(data_frames.keys())}")

        documents = []
        for sheet_name, df in data_frames.items():
            for idx, row in df.iterrows():
                text = row.to_json()
                documents.append({"id": f"{sheet_name}_{idx}", "text": text})

        if not documents:
            logger.warning(f"No data found in file: {file.name}")
            return {"status": "warning", "message": f"No data found in file: {file.name}"}

        texts = [doc["text"] for doc in documents]
        embedding_vectors = embeddings.embed_documents(texts)
        points = [
            PointStruct(id=doc["id"], vector=vector, payload={"text": doc["text"]})
            for doc, vector in zip(documents, embedding_vectors)
        ]

        qdrant_client.upsert(collection_name=collection_name, points=points)
        logger.info(f"Successfully uploaded {len(documents)} records from {file.name} to Qdrant.")
        return {"status": "success", "message": f"Successfully uploaded and stored {len(documents)} records from {file.name}."}

    except Exception as e:
        logger.error(f"Error processing file {file.name}: {e}")
        return {"status": "error", "message": f"Error processing file {file.name}: {e}"}

def initialize_retrieval_qa(qdrant_client: QdrantClient, collection_name: str, embeddings: OpenAIEmbeddings, gemini_llm: GeminiLLM) -> Optional[RetrievalQA]:
    """
    Initializes the Retrieval-Augmented Generation (RAG) chain using Gemini as the LLM.
    """
    try:
        vector_store = Qdrant(client=qdrant_client, collection_name=collection_name, embeddings=embeddings)
        qa = RetrievalQA.from_chain_type(
            llm=gemini_llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            verbose=True
        )
        logger.info("Initialized RetrievalQA chain with Gemini LLM.")
        return qa
    except Exception as e:
        logger.error(f"Error initializing RetrievalQA: {e}")
        return None

def create_graph(qdrant_client: QdrantClient, collection_name: str) -> Optional:
    """
    Creates a data visualization graph using LangGraph.
    """
    try:
        lang_graph = LangGraph()
        graph = lang_graph.create_graph(collection=collection_name, client=qdrant_client)
        logger.info("Successfully generated data visualization graph.")
        return graph
    except Exception as e:
        logger.error(f"Error generating graph: {e}")
        return None