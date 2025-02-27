import streamlit as st
import json
import os
import asyncio
import base64
import tempfile
from langchain_community.vectorstores import FAISS
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain_core.messages import HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import JsonOutputParser
from fpdf import FPDF
import functools
import time
import speech_recognition as sr
from gtts import gTTS
import io

# --- Streamlit UI ---
st.set_page_config(
    page_title="Jacie - Financial Assistant",
    page_icon="💼",
    layout="wide"
)
st.title("Jacie - Your Financial Assistant")

# Add a 'How to Use' section in the sidebar
if "first_load" not in st.session_state:
    st.session_state.first_load = True

if st.session_state.first_load:
    st.sidebar.info("💡 How to Use:\n\n"
                    "1️⃣ Enter your **financial query**.\n\n"
                    "2️⃣ The system retrieves **relevant financial documents**.\n\n"
                    "3️⃣ AI **analyzes the pages** and extracts key insights.\n\n"
                    "4️⃣ Get a **structured summary** based on all documents.\n\n"
                    "5️⃣ Download the **PDF report** with all the details.")
    st.session_state.first_load = False

# --- Google Credentials ---
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
try:
    if not os.path.exists("vector_store"):
        st.error("🚫 Vector store not found. Please ensure it is initialized.")
        st.stop()
    embeddings = VertexAIEmbeddings(model="text-embedding-004")
    vector_store = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
except Exception as e:
    st.error(f"⚠️ Error loading vector store: {str(e)}")
    st.stop()

# --- Initialize LLMs ---
try:
    image_llm = ChatVertexAI(model="gemini-2.0-flash", temperature=0)  # For analyzing images
    summarization_llm = ChatVertexAI(model="gemini-1.5-pro", temperature=0)  # For summarization
except Exception as e:
    st.error(f"⚠️ Error initializing ChatVertexAI: {str(e)}")
    st.stop()

# --- Function: Convert Image to Base64 ---
def encode_image(image_path):
    """Convert an image file to a Base64 string."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        st.error(f"⚠️ Error encoding image {image_path}: {str(e)}")
        return None

# --- FAISS Search Function ---
num_images = st.sidebar.slider("Number of images to analyze", 1, 10, 3)  # Default is 3

def search_faiss(query, k=num_images):
    results = vector_store.similarity_search(query, k=k)
    retrieved_images = [doc.metadata.get("image") for doc in results if doc.metadata.get("image")]
    
    if not retrieved_images:
        st.info("ℹ️ No matching documents found. Try refining your query or uploading new financial documents.")
    return retrieved_images

# --- Image Processing Prompt ---
IMAGE_PROCESSING_PROMPT = """
    You are an expert financial analyst. The user has asked a financial question.
    Review the provided financial document (a scanned page of a financial report) 
    and extract relevant information **based on the user's query**.

    **Instructions:**
    - Summarize only information **relevant** to the query.
    - Extract key financial figures (revenues, costs, profits, etc.).
    - Identify risks, inconsistencies, or missing data.
    - The response should be a JSON with properly formatted markdown text.

    **User Query:** {query}

    **Response Format:**
    {{
        "Summary": "[Brief summary relevant to the query]",
        "Key Figures": "[Extracted financial values]",
        "Risks or Notes": "[Any inconsistencies or missing data]"
    }}
"""

# --- Retry Decorator ---
def async_retry(max_retries=3, initial_delay=1):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        st.warning(f"Attempt {attempt + 1} failed, retrying in {delay} seconds... Error: {str(e)}")
                        await asyncio.sleep(delay)
                        delay *= 2  # Exponential backoff
                    else:
                        st.error(f"All {max_retries} attempts failed. Last error: {str(e)}")
            
            raise last_exception
        return wrapper
    return decorator

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

# --- Summarization Prompt ---
SUMMARIZATION_PROMPT = """
    You are an expert financial analyst. Below are multiple financial document analyses related to a user's query.

    **Task:** 
    - Summarize the **most relevant findings** across all analyzed pages.
    - Ensure your response is **concise, structured, and factual**.
    - **Avoid redundancy** if multiple pages contain the same data.
    - The response should be a JSON with properly formatted markdown text.
    **User Query:** {query}

    **Extracted Analyses:**
    {analyses}

    **Response Format:**
    {{
        "Final Summary": "[A structured answer integrating insights from all pages]",
        "Key Takeaways": "[Most important financial figures]",
        "Caveats or Uncertainties": "[Any inconsistencies or missing data]"
    }}
"""

# --- Function to Summarize Responses ---
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

# --- Initialize Conversation Memory ---
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

# --- Function to Enhance Query with LLM ---
async def enhance_query_with_memory(user_query):
    """Enhance the user query using chat history from memory."""
    # Retrieve chat history from memory
    chat_history = st.session_state.memory.load_memory_variables({})
    # Create a message with the user's query and chat history
    message = HumanMessage(content=f"User Query: {user_query}\nChat History: {chat_history}")
    # Use an LLM to enhance the query
    enhanced_query = await summarization_llm.ainvoke([message])
    return enhanced_query.content if enhanced_query and enhanced_query.content else user_query

# Function for Speech-to-Text
def speech_to_text():
    audio_bytes = st.audio_input(
        label="🎤 Click to record"
    )
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        # Read audio bytes into a bytes-like object
        audio_data = audio_bytes.read()
        
        # Save audio bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_data)
            temp_audio.flush()
            
            try:
                import speech_recognition as sr
                recognizer = sr.Recognizer()
                with sr.AudioFile(temp_audio.name) as source:
                    audio_data = recognizer.record(source)
                    text = recognizer.recognize_google(audio_data)
                    return text
            except Exception as e:
                st.error(f"❌ Error processing speech: {e}")
                return None
            finally:
                # Clean up temporary file
                os.unlink(temp_audio.name)
    return None

# Function for Text-to-Speech
def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except Exception as e:
        st.error(f"❌ Error generating speech: {e}")
        return None
        
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)
        if message["role"] == "assistant":
            # Add a play button for assistant messages
            audio_bytes = text_to_speech(message["content"])
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")

# Define an async function to handle the workflow
async def handle_user_query(user_query):
    # Enhance the query using chat history
    enhanced_query = await enhance_query_with_memory(user_query)

    # Process the enhanced query
    st.write("🔍 Searching for relevant document images...")
    pdf_analysis_results = await analyze_pdf_images(enhanced_query)

    # Store pdf_analysis_results in session state
    st.session_state['pdf_analysis_results'] = pdf_analysis_results

    # Display intermediate results in an expander
    with st.expander("📊 Intermediate Analysis Results"):
        for result in pdf_analysis_results:
            col1, col2 = st.columns([2, 3])
            with col1:
                st.image(result["image"], caption="Analyzed Page", use_container_width=True)
            with col2:
                st.markdown(f"**Summary:** {result['analysis']['Summary']}")
                st.markdown(f"**Key Figures:** {result['analysis']['Key Figures']}")
                st.markdown(f"**Risks:** {result['analysis']['Risks or Notes']}")

    # Step 2: Summarize all results
    if pdf_analysis_results:
        st.write("🔍 Generating final summary...")
        final_summary = await summarize_responses(user_query, pdf_analysis_results)

        # Function to escape special characters in text
        def escape_special_chars(text):
            if isinstance(text, str):
                return text.replace("$", r"\$")
            return text  # If not a string, return as is

        # Function to format the summary output properly
        def format_summary_output(summary_dict):
            key_takeaways = summary_dict.get("Key Takeaways", {})

            # Convert dictionary into bullet points with escaped special characters
            if isinstance(key_takeaways, dict):
                key_takeaways_text = "\n".join([
                    f"- **{escape_special_chars(key)}**: `{escape_special_chars(value)}`" for key, value in key_takeaways.items()
                ])
            else:
                key_takeaways_text = escape_special_chars(key_takeaways)

            summary_text = f"""
**📌 Summary:**  
{escape_special_chars(summary_dict.get("Final Summary", "N/A"))}

**📊 Key Takeaways:**  
{key_takeaways_text}

**⚠️ Caveats:**  
{escape_special_chars(summary_dict.get("Caveats or Uncertainties", "N/A"))}
"""
            return summary_text

        # Display Summary Properly
        with st.chat_message("assistant"):
            st.markdown(format_summary_output(final_summary), unsafe_allow_html=True)
            
        # Add the final summary to the chat history
        st.session_state.messages.append({"role": "assistant", "content": f"**Summary:** {final_summary['Final Summary']}\n\n"
                        f"**Key Takeaways:** {final_summary['Key Takeaways']}\n\n"
                        f"**Caveats:** {final_summary['Caveats or Uncertainties']}"})
        # Update conversation memory with the assistant's response
        st.session_state.memory.save_context({"input": user_query}, {"output": final_summary['Final Summary']})

# Chat input and voice input side by side
col1, col2 = st.columns([4, 1])

with col1:
    user_query = st.chat_input("Enter your query or use voice input")

with col2:
    # Record audio directly in the UI
    spoken_text = speech_to_text()
    if spoken_text:
        user_query = spoken_text

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query, unsafe_allow_html=True)
        
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Run the async function using asyncio.run()
    asyncio.run(handle_user_query(user_query))

# --- Function to Generate PDF ---
def generate_pdf(chat_history, analyses):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add chat history to PDF
    pdf.cell(200, 10, txt="Chat History:", ln=True, align='L')
    for message in chat_history:
        role = "User" if message["role"] == "user" else "Assistant"
        pdf.multi_cell(0, 10, txt=f"{role}: {message['content']}")

    # Add a separator
    pdf.cell(200, 10, txt="", ln=True, align='L')

    # Add analyses to PDF
    pdf.cell(200, 10, txt="Intermediate Analyses:", ln=True, align='L')
    for analysis in analyses:
        pdf.multi_cell(0, 10, txt=f"Source: {analysis['image']}")
        pdf.multi_cell(0, 10, txt=f"Summary: {analysis['analysis']['Summary']}")
        pdf.multi_cell(0, 10, txt=f"Key Figures: {analysis['analysis']['Key Figures']}")
        pdf.multi_cell(0, 10, txt=f"Risks: {analysis['analysis']['Risks or Notes']}")
        pdf.cell(200, 10, txt="", ln=True, align='L')

    # Save PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        pdf.output(temp_pdf.name)
        return temp_pdf.name

# --- Add Download Button in Sidebar ---
if 'pdf_analysis_results' in st.session_state:
    pdf_path = generate_pdf(st.session_state.messages, st.session_state['pdf_analysis_results'])
    with open(pdf_path, "rb") as pdf_file:
        st.sidebar.download_button(label="Download Report as PDF", data=pdf_file, file_name="report.pdf", mime="application/pdf")
else:
    st.sidebar.info("No analysis results available to download.")
