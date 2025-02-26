import os
import streamlit as st
import json
from langchain_community.document_loaders import PyPDFLoader
from pdf2image import convert_from_path
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# # Define input and output directories
# pdf_folder = "docs/pdfs"  # Folder containing PDFs
# image_output_dir = "docs/pdf_images"
# os.makedirs(image_output_dir, exist_ok=True)

# logger.info(f"📄 Processing {pdf_folder}...")
# logger.info(f"📄 JPEG images of pages will be saved in {image_output_dir}...")

# def process_pdfs(pdf_folder):
#     all_documents = []
#     page_image_map = {}  # Maps (doc_name, page_number) → image_path

#     for pdf_file in os.listdir(pdf_folder):
#         if pdf_file.endswith(".pdf"):
#             pdf_path = os.path.join(pdf_folder, pdf_file)
#             doc_name = os.path.splitext(pdf_file)[0]  # Get doc name without extension

#             logger.info(f"📄 Processing {pdf_file}...")
            
#             # Extract text
#             loader = PyPDFLoader(pdf_path)
#             docs = loader.load()

#             for i, doc in enumerate(docs):
#                 page_number = i + 1
#                 doc.metadata["source"] = doc_name  # Store document name
#                 doc.metadata["page"] = page_number  # Store page number
#                 all_documents.append(doc)

#             # Extract images
#             images = convert_from_path(pdf_path, dpi=300)
#             for i, img in enumerate(images):
#                 page_number = i + 1
#                 img_path = os.path.join(image_output_dir, f"{doc_name}_page_{page_number}.jpg")
#                 img.save(img_path, "JPEG")
#                 page_image_map[(doc_name, page_number)] = img_path  # Map page to image path
    
#     return all_documents, page_image_map

# # Process all PDFs
# documents, page_image_map = process_pdfs(pdf_folder)

# logger.info(f"✅ Processed {len(documents)} pages across multiple PDFs!")
# logger.info("Example document metadata:", documents[0].metadata)

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

# def store_in_faiss(documents, page_image_map):
#     # Ensure documents are not empty before processing
#     non_empty_documents = []
#     for doc in documents:
#         if not doc.page_content.strip():
#             logger.warning(f"Skipping empty document: {doc.metadata['source']} - Page {doc.metadata['page']}")
#             continue
#         non_empty_documents.append(doc)
#         # Proceed with embedding only if document is not empty
#         doc_name = doc.metadata["source"]
#         page_number = doc.metadata["page"]
#         doc.metadata["image"] = page_image_map.get((doc_name, page_number), None)

#     if not non_empty_documents:
#         logger.error("No non-empty documents to process. Exiting.")
#         return

#     # Create FAISS vector store
#     index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
#     vector_store = FAISS(
#         embedding_function=embeddings,
#         index=index,
#         docstore=InMemoryDocstore(),
#         index_to_docstore_id={},
#     )
#     vector_store = FAISS.from_documents(non_empty_documents, embeddings)

#     # Save FAISS index
#     faiss_store_path = "faiss_index"
#     vector_store.save_local(faiss_store_path)

#     logger.info(f"✅ Stored {len(documents)} pages in FAISS with image metadata!")

# # Store documents in FAISS
# logger.info("Storing documents in FAISS")
# store_in_faiss(documents, page_image_map)

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