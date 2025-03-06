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
    collection_name="company_docs",
    embedding_function=embeddings
)

# Load documents from all PDFs in the directory
logger.info("Loading documents from all PDFs in the directory")
pdf_directory = "docs/pdfs"
image_output_dir = "docs/pdf_images"
os.makedirs(image_output_dir, exist_ok=True)
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

# Split documents
logger.info("Setting up text splitter")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100)

# Load documents
logger.info("Loading documents")
documents = []
for pdf_file in pdf_files:
    logger.info(f"Loading document: {pdf_file}")
    pdf_path = os.path.join(pdf_directory, pdf_file)
    doc_name = re.split("-", pdf_file)[0]
    year = re.search(r'\d{4}', doc_name).group()
    
    # Extract images
    images = convert_from_path(pdf_path, dpi=300)
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, (page, img) in enumerate(zip(pdf.pages, images), start=1):
            text = page.extract_text()
            if text:
                # Save image for the current page
                img_path = os.path.join(image_output_dir, f"{doc_name}_page_{page_number}.jpg")
                img.save(img_path, "JPEG")
                
                # Create Document object with metadata
                doc = Document(
                    page_content=text,
                    metadata={
                        "company_name": doc_name,
                        "year": year,
                        "document_name": doc_name,
                        "page_number": page_number,
                        "image": img_path
                    }
                )
                documents.append(doc)

# Split documents into chunks
logger.info("Splitting documents into chunks")
chunks = text_splitter.split_documents(documents)

# Add documents to vector store
logger.info("Adding documents to vector store")
vector_store.add_documents(chunks)

# Save vector store
logger.info("Saving vector store")
vector_store.save_local("vector_store")

def search_faiss(query, k=3):
    vector_store = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
    results = vector_store.similarity_search(query, k=k)
    
    retrieved_pages = []
    for doc in results:
        retrieved_pages.append({
            "document": doc.metadata["document_name"],
            "page": doc.metadata["page_number"],
            "text": doc.page_content,
            "image": doc.metadata.get("image", None)  # Retrieve stored image path
        })
    
    return retrieved_pages

# Example query
query = "How did NCB perform in 2024 compared to 2023?"

logger.info(f"🔍 Searching for: {query}")
retrieved_results = search_faiss(query)

for res in retrieved_results:
    logger.info(f"📄 {res['document']} - Page {res['page']}")
    logger.info(f"🖼 Image Path: {res['image']}")
    logger.info(f"📝 Text:\n{res['text'][:500]}...\n")
# Done
logger.info("Done")
