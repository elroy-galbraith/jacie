import os
import json
import tempfile
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI

# --- Google Credentials ---
def setup_google_credentials():
    try:
        if "GOOGLE_APPLICATION_CREDENTIALS" not in st.secrets:
            st.error("🚫 Missing Google Cloud credentials in Streamlit secrets.")
            st.stop()

        # Create temporary file for credentials
        with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json") as temp_cred:
            json.dump(json.loads(st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]), temp_cred)
            temp_cred_path = temp_cred.name

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_cred_path
    except Exception as e:
        st.error(f"⚠️ Error setting up credentials: {str(e)}")
        st.stop()

# --- Load FAISS Vector Store ---
def load_faiss_vector_store():
    try:
        if not os.path.exists("vector_store"):
            st.error("🚫 Vector store not found. Please ensure it is initialized.")
            st.stop()
        embeddings = VertexAIEmbeddings(model="text-embedding-004")
        vector_store = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
        return vector_store
    except Exception as e:
        st.error(f"⚠️ Error loading vector store: {str(e)}")
        st.stop()

# --- Initialize LLMs ---
def initialize_llms():
    try:
        image_llm = ChatVertexAI(model="gemini-2.0-flash", temperature=0)  # For analyzing images
        summarization_llm = ChatVertexAI(model="gemini-1.5-pro", temperature=0)  # For summarization
        return image_llm, summarization_llm
    except Exception as e:
        st.error(f"⚠️ Error initializing ChatVertexAI: {str(e)}")
        st.stop() 