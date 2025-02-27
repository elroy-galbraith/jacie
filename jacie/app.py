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
                    "1️⃣ Enter your **financial query**.\n"
                    "2️⃣ The system retrieves **relevant financial documents**.\n"
                    "3️⃣ AI **analyzes the pages** and extracts key insights.\n"
                    "4️⃣ Get a **structured summary** based on all documents.\n")
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

    **User Query:** {query}

    **Response Format:**
    {{
        "Summary": "[Brief summary relevant to the query]",
        "Key Figures": "[Extracted financial values]",
        "Risks or Notes": "[Any inconsistencies or missing data]"
    }}
"""

# --- Async Function to Process a Single Image ---
async def process_pdf_image(image_path, query):
    """Process an image using Gemini-1.5 Flash."""
    img_str = encode_image(image_path)
    if img_str is None:
        return None

    try:
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
    except Exception as e:
        st.error(f"⚠️ Error processing {image_path}: {str(e)}")
        return None

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
async def summarize_responses(query, responses):
    """Summarize extracted financial insights."""
    if not responses:
        return {"Final Summary": "No relevant data found.", "Key Takeaways": "", "Caveats or Uncertainties": ""}

    formatted_responses = "\n\n".join(
        [f"- {res['image']}\nSummary: {res['analysis'].get('Summary', 'N/A')}\nKey Figures: {res['analysis'].get('Key Figures', 'N/A')}\nRisks: {res['analysis'].get('Risks or Notes', 'N/A')}" for res in responses]
    )

    message = [HumanMessage(content=SUMMARIZATION_PROMPT.format(query=query, analyses=formatted_responses))]

    try:
        summary = await summarization_llm.ainvoke(message)
        summary_content = summary.content if summary and summary.content else "{}"
        parser = JsonOutputParser()
        parsed_summary = parser.parse(summary_content)
        return parsed_summary
    except Exception as e:
        st.error(f"⚠️ Error generating summary: {str(e)}")
        return {
            "Final Summary": "Error encountered while summarizing.",
            "Key Takeaways": "No key takeaways available.",
            "Caveats or Uncertainties": "Unable to determine uncertainties."
        }

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

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Define an async function to handle the workflow
async def handle_user_query(user_query):
    # Enhance the query using chat history
    enhanced_query = await enhance_query_with_memory(user_query)

    # Process the enhanced query
    st.write("🔍 Searching for relevant document images...")
    pdf_analysis_results = await analyze_pdf_images(enhanced_query)

    # Display results for each image in the sidebar
    with st.sidebar.status("📄 Processing relevant financial documents...") as status:
        for result in pdf_analysis_results:
            st.image(result["image"], caption="Analyzed Page", use_container_width=True)
        status.update(label="✅ Processing complete", state="complete", expanded=False)

    # Display intermediate results in an expander
    with st.expander("📊 Intermediate Analysis Results"):
        for result in pdf_analysis_results:
            col1, col2 = st.columns([2, 3])
            with col1:
                st.image(result["image"], caption="Analyzed Page", use_column_width=True)
            with col2:
                st.write(f"**Summary:** {result['analysis']['Summary']}")
                st.write(f"**Key Figures:** {result['analysis']['Key Figures']}")
                st.write(f"**Risks:** {result['analysis']['Risks or Notes']}")

    # Step 2: Summarize all results
    if pdf_analysis_results:
        st.write("🔍 Generating final summary...")
        final_summary = await summarize_responses(user_query, pdf_analysis_results)

        # Add assistant's response to chat history
        with st.chat_message("assistant"):
            st.markdown(f"**Summary:** {final_summary['Final Summary']}\n\n"
                        f"**Key Takeaways:** {final_summary['Key Takeaways']}\n\n"
                        f"**Caveats:** {final_summary['Caveats or Uncertainties']}")
            
        # Add the final summary to the chat history
        st.session_state.messages.append({"role": "assistant", "content": f"**Summary:** {final_summary['Final Summary']}\n\n"
                        f"**Key Takeaways:** {final_summary['Key Takeaways']}\n\n"
                        f"**Caveats:** {final_summary['Caveats or Uncertainties']}"})
        # Update conversation memory with the assistant's response
        st.session_state.memory.save_context({"input": user_query}, {"output": final_summary['Final Summary']})

# Call the async function using asyncio.run() at the top level
if user_query := st.chat_input("Enter your query"):
    with st.chat_message("user"):
        st.markdown(user_query)
        
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Run the async function using asyncio.run()
    asyncio.run(handle_user_query(user_query))
