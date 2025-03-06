import os
import streamlit as st
import json
from PIL import Image
import base64
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
import logging
import pdfplumber
from pdf2image import convert_from_path
from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Set up credentials from Streamlit secrets
logger.info("Setting up credentials from Streamlit secrets")
if not os.path.exists("credentials.json"):
    with open("credentials.json", "w") as f:
        json.dump(json.loads(st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]), f)

# Set the environment variable to point to the credentials file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath("credentials.json")

# Create dataset in LangSmith
client = Client()
dataset_name = "Q&A Eval Set"
# dataset = client.create_dataset(dataset_name=dataset_name)
dataset = client.read_dataset(dataset_name=dataset_name)

# Set up embeddings
logger.info("Setting up embeddings")
embeddings = VertexAIEmbeddings(model="text-embedding-004")

# Load documents from all PDFs in the directory
logger.info("Loading documents from all PDFs in the directory")
pdf_directory = "docs/pdfs"
image_output_dir = "docs/pdf_images"
os.makedirs(image_output_dir, exist_ok=True)
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

QA_GENERATION_PROMPT = """
    You are an AI engineer with a background in financial analysis.
    You are creating a question and answer pair to evaluate the accuracy of a retrieval system.
    Create a question and answer pair for the following document and document name. 
    You'll need to guess the company and reference date from the document name.

    Response Format:

    ```json
        {
            "question": 'the question contextualized to the document',
            "answer": 'the answer to the question',
            "context": 'the context of the question and answer'
        }
    ```

"""
# a fucton to encode an image
def encode_image(image_path):
    """Convert an image to base64 string."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {str(e)}")
        return None

# a  function to use ChatVertexAI to generate a question and answer pair ffrom an eencoded image
def generate_qa_pair(image_path, doc_name):
    # convert image to base64
    image_base64 = encode_image(image_path)
    # use ChatVertexAI to generate a question and answer pair
    chat = ChatVertexAI(model="gemini-1.5-pro")
    prompt = HumanMessage(
        content=[
            {"type": "text", "text": QA_GENERATION_PROMPT},
            {"type": "text", "text": f"Document name: {doc_name}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ]
    )
    chain = chat | JsonOutputParser()
    response = chain.invoke([prompt])
    return response

# Load documents
logger.info("Loading documents")
documents = []
for pdf_file in pdf_files:
    logger.info(f"Loading document: {pdf_file}")
    pdf_path = os.path.join(pdf_directory, pdf_file)
    doc_name = os.path.splitext(pdf_file)[0]
    
    # Extract images
    images = convert_from_path(pdf_path, dpi=300)
    
    with pdfplumber.open(pdf_path) as pdf:
        # get the company and reference date from the pdf file name
    
        for page_number, (page, img) in enumerate(zip(pdf.pages, images), start=1):
            text = page.extract_text()
            if text:
                # Save image for the current page
                img_path = os.path.join(image_output_dir, f"{doc_name}_page_{page_number}.jpg")
                img.save(img_path, "JPEG")
                # generate a question and answer pair
                qa_pair = generate_qa_pair(img_path, doc_name)
                # add the question and answer pair to the documents
                documents.append(qa_pair)

# Create examples from the documents
examples = [(doc["question"], doc["answer"], doc["context"]) for doc in documents]

client.create_examples(
    inputs=[{"question": q} for q, _, _ in examples], 
    outputs=[{"answer": a} for _, a, _ in examples], 
    metadata=[{"context": c} for _, _, c in examples],
    dataset_id=dataset.id,
)