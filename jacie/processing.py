import asyncio
import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from utils import encode_image, async_retry
from initialization import initialize_llms, load_faiss_vector_store

# Initialize LLMs and Vector Store
image_llm, summarization_llm = initialize_llms()
vector_store = load_faiss_vector_store()

# --- FAISS Search Function ---
def search_faiss(query, k=3):
    results = vector_store.similarity_search(query, k=k)
    retrieved_images = [doc.metadata.get("image") for doc in results if doc.metadata.get("image")]
    
    if not retrieved_images:
        st.info("ℹ️ No matching documents found. Try refining your query or uploading new financial documents.")
    return retrieved_images

# --- Async Function to Process a Single Image ---
@async_retry(max_retries=3, initial_delay=1)
async def process_pdf_image(image_path, query):
    """Process an image using Gemini-1.5 Flash with retry logic."""
    img_str = encode_image(image_path)
    if img_str is None:
        return None

    data_url = f"data:image/png;base64,{img_str}"
    message = HumanMessage(
        content=[
            {"type": "text", "text": IMAGE_PROCESSING_PROMPT.format(query=query)},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    )
    chain = image_llm | JsonOutputParser()  # Directly parse as JSON
    result = await chain.ainvoke([message])

    return {"image": image_path, "analysis": result} if result else None

# --- Async Function to Process Multiple Images ---
async def analyze_pdf_images(query):
    """Retrieve and process relevant images with the user query."""
    retrieved_images = search_faiss(query)
    if not retrieved_images:
        st.info("ℹ️ No matching documents found. Try refining your query or uploading new financial documents.")
        return []

    with st.status("⏳ Searching...", expanded=True) as status:
        results = await asyncio.gather(*[process_pdf_image(img, query) for img in retrieved_images])
        status.update(label="✅ Search complete", state="complete", expanded=False)

    return [res for res in results if res is not None]

# --- Async Function to Summarize Responses ---
@async_retry(max_retries=3, initial_delay=1)
async def summarize_responses(query, responses):
    """Summarize extracted financial insights with retry logic."""
    if not responses:
        return {"Final Summary": "No relevant data found.", "Key Takeaways": "", "Caveats or Uncertainties": ""}

    formatted_responses = "\n\n".join(
        [f"- {res['image']}\nSummary: {res['analysis'].get('Summary', 'N/A')}\nKey Figures: {res['analysis'].get('Key Figures', 'N/A')}\nRisks: {res['analysis'].get('Risks or Notes', 'N/A')}" for res in responses]
    )

    message = [HumanMessage(content=SUMMARIZATION_PROMPT.format(query=query, analyses=formatted_responses))]
    chain = summarization_llm | JsonOutputParser()
    
    summary = await chain.ainvoke(message)
    return summary 