import os
import streamlit as st
import json
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import chromadb
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
import pdfplumber
from pdf2image import convert_from_path
import re

# Add at the top of the file with other constants
DEFAULT_COMPANY_NAME = "NCB"  # Default company to search if none specified

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up credentials from Streamlit secrets
logger.info("Setting up credentials from Streamlit secrets")
if not os.path.exists("credentials.json"):
    with open("credentials.json", "w") as f:
        json.dump(json.loads(st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]), f)

# Set the environment variable to point to the credentials file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath("credentials.json")

# Set up embeddings
logger.info("Setting up embeddings")
embeddings = VertexAIEmbeddings(model="text-embedding-004")

# Set up vector store
logger.info("Setting up vector store")

# Initialize persistent ChromaDB storage
DB_PATH = "./chroma_db"

chroma_client = chromadb.PersistentClient(path=DB_PATH)

# Initialize LangChain's Chroma wrapper
vector_store = Chroma(
    client=chroma_client,
    collection_name="company_names",
    embedding_function=embeddings
)

# Load documents from all PDFs in the directory
logger.info("Loading documents from all PDFs in the directory")
pdf_directory = "docs/pdfs"
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

# Load documents
logger.info("Loading documents")
documents = []
for pdf_file in pdf_files:
    logger.info(f"Loading document: {pdf_file}")
    pdf_path = os.path.join(pdf_directory, pdf_file)
    doc_name = pdf_file
    company_name = doc_name.split("-")[0]
    
    # Extract year from filename with validation
    year_match = re.search(r'\d{4}', doc_name)
    year = year_match.group() if year_match else "Unknown"
    if year != "Unknown":
        try:
            year_int = int(year)
            # Validate year is within reasonable range (e.g., between 1900 and 2100)
            if year_int < 1900 or year_int > 2100:
                logger.warning(f"Year {year} outside valid range for document: {doc_name}")
                year = "Unknown"
        except ValueError:
            logger.warning(f"Invalid year format in document: {doc_name}")
            year = "Unknown"
    
    logger.info(f"Company name: {company_name}, Year: {year}")
    
    # Create Document object with metadata
    doc = Document(
        page_content=company_name,
        metadata={"year": year, "document_name": doc_name}
    )
    documents.append(doc)

# Add documents to vector store
logger.info("Adding documents to vector store")
vector_store.add_documents(documents)

# Save vector store
logger.info("Saving vector store")
# ChromaDB automatically persists to disk since we're using PersistentClient
logger.info("Vector store saved to disk at: " + DB_PATH)

def search_vector_store(query):
    # Load the vector store from disk
    loaded_vector_store = Chroma(
        client=chromadb.PersistentClient(path=DB_PATH),
        collection_name="company_names",
        embedding_function=embeddings
    )
    results = loaded_vector_store.similarity_search(
        query, k=1)
    
    return results[0].page_content

# Example query
query = "How did Edufocal perform?"

logger.info(f"🔍 Searching for: {query}")
retrieved_results = search_vector_store(query)

logger.info(f"📄 {retrieved_results}")
# Done
logger.info("Done")
