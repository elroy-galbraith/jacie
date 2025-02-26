import streamlit as st
import json
from langchain_community.vectorstores import FAISS
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_vertexai import ChatVertexAI
from langchain.memory import ConversationBufferMemory
import os

# Check for credentials
try:
    if not os.path.exists("credentials.json"):
        if "GOOGLE_APPLICATION_CREDENTIALS" not in st.secrets:
            st.error("🚫 Missing Google Cloud credentials in Streamlit secrets.")
            st.stop()
        with open("credentials.json", "w") as f:
            json.dump(json.loads(st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]), f)
    
    # Set the environment variable to point to the credentials file
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath("credentials.json")
except Exception as e:
    st.error(f"⚠️ Error setting up credentials: {str(e)}")
    st.stop()
    
    # Initialize embeddings with error handling
try:
    embeddings = VertexAIEmbeddings(model="text-embedding-004")
except Exception as e:
    st.error(f"⚠️ Error initializing embeddings: {str(e)}")
    st.stop()
    
# Load the persisted FAISS vector store with error handling
try:
    if not os.path.exists("vector_store"):
        st.error("🚫 Vector store not found. Please ensure the vector store is properly initialized.")
        st.stop()
    vector_store = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
except Exception as e:
    st.error(f"⚠️ Error loading vector store: {str(e)}")
    st.stop()
    
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

def get_images(query, k=3):
    retrieved_pages = search_faiss(query, k)
    images = [page["image"] for page in retrieved_pages]
    return images

