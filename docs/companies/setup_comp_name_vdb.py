import os
import streamlit as st
import json
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.documents import Document
import chromadb
from langchain_chroma import Chroma
import logging

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

# Delete the collection if it exists
if "company_names" in chroma_client.list_collections():
    chroma_client.delete_collection(name="company_names")

# Initialize LangChain's Chroma wrapper
vector_store = Chroma(
    client=chroma_client,
    collection_name="company_names",
    embedding_function=embeddings
)

# Load json file
logger.info("Loading company names")
json_file = "docs/companies/companies.json"
with open(json_file, "r") as f:
    companies = json.load(f)

documents=[]
for company in companies:
    logger.info(f"Company: {company}")
    
    # Create Document object with metadata
    doc = Document(
        page_content=f"Security Name: {company['security_name']}, Short Name: {company['short_name']}, Ticker Symbol: {company['ticker_symbol']}, ISIN: {company['ISIN']}",
        metadata={
            "company_name": company['security_name'], 
            "short_name": company['short_name'],
            "ticker_symbol": company['ticker_symbol'], 
            "ISIN": company['ISIN']}
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
query = "How did One on One perform?"

logger.info(f"🔍 Searching for: {query}")
retrieved_results = search_vector_store(query)

logger.info(f"📄 {retrieved_results}")
# Done
logger.info("Done")
