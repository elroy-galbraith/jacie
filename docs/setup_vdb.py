import os
import streamlit as st
import json
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
import pdfplumber

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

index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# Load documents from all PDFs in the directory
logger.info("Loading documents from all PDFs in the directory")
pdf_directory = "docs/pdfs"
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

# Split documents
logger.info("Splitting documents")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

documents = []
for pdf_file in pdf_files:
    logger.info(f"Loading document: {pdf_file}")
    with pdfplumber.open(os.path.join(pdf_directory, pdf_file)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                # Create Document object with metadata
                doc = Document(
                    page_content=text,
                    metadata={
                        "document_name": pdf_file,
                        "page_number": page_number
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

# Done
logger.info("Done")
