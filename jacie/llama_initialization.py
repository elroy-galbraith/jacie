import os
import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import JsonOutputParser
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_llms():
    """Initialize Ollama LLMs for image analysis and summarization."""
    try:
        # Initialize the base Llama 2 model for text processing
        text_llm = Ollama(
            model="llama2",
            temperature=0,
            format="json"  # Enable JSON mode for structured outputs
        )
        
        # Initialize Llama 2 with vision capabilities for image analysis
        vision_llm = Ollama(
            model="llama2-vision",  # Make sure you have pulled this model with Ollama
            temperature=0,
            format="json"
        )
        
        return vision_llm, text_llm
    except Exception as e:
        st.error(f"⚠️ Error initializing Ollama LLMs: {str(e)}")
        st.stop()

def initialize_embeddings():
    """Initialize Ollama embeddings."""
    try:
        embeddings = OllamaEmbeddings(
            model="llama2",
            base_url="http://localhost:11434"  # Default Ollama URL
        )
        return embeddings
    except Exception as e:
        st.error(f"⚠️ Error initializing Ollama embeddings: {str(e)}")
        st.stop()

# Initialize LLMs and embeddings
vision_llm, text_llm = initialize_llms()
embeddings = initialize_embeddings()

# Example usage:
# For text processing:
# chain = text_llm | JsonOutputParser()
# result = await chain.ainvoke([message])

# For image processing:
# chain = vision_llm | JsonOutputParser()
# result = await chain.ainvoke([message, image]) 