import streamlit as st
import json
import os
import asyncio
import base64
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
with st.sidebar.expander("How to Use"):
    st.markdown("""
    1. **Enter your financial query** in the chat input box.
    2. **View the source documents** in the 'Source Documents' section.
    3. **Check the intermediate analysis** in the 'Intermediate Analysis Results' expander.
    4. **Read the final summary** in the main chat area for concise insights.
    """)

# --- Google Credentials ---
try:
    if not os.path.exists("credentials.json"):
        if "GOOGLE_APPLICATION_CREDENTIALS" not in st.secrets:
            st.error("🚫 Missing Google Cloud credentials in Streamlit secrets.")
            st.stop()
        with open("credentials.json", "w") as f:
            json.dump(json.loads(st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]), f)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath("credentials.json")
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
    image_llm = ChatVertexAI(model="gemini-1.5-flash-001", temperature=0)  # For analyzing images
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
    """Retrieve top-k relevant document images."""
    results = vector_store.similarity_search(query, k=k)
    
    return [doc.metadata.get("image", None) for doc in results if doc.metadata.get("image")]

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
        st.error("🚫 No relevant document images found.")
        return []

    with st.status("⏳ Processing images...", expanded=True) as status:
        results = await asyncio.gather(*[process_pdf_image(img, query) for img in retrieved_images])
        status.update(label="✅ Image processing complete", state="complete", expanded=False)

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
        
        # Ensure content exists before parsing
        summary_content = summary.content if summary and summary.content else "{}"
        
        # Parse the content as JSON
        parser = JsonOutputParser()
        parsed_summary = parser.parse(summary_content)
        
        return parsed_summary if parsed_summary else {
            "Final Summary": "No summary generated.",
            "Key Takeaways": "No key takeaways found.",
            "Caveats or Uncertainties": "No uncertainties noted."
        }
    except Exception as e:
        st.error(f"⚠️ Error generating summary: {str(e)}")
        return {
            "Final Summary": "Error encountered while summarizing.",
            "Key Takeaways": "No key takeaways available.",
            "Caveats or Uncertainties": "Unable to determine uncertainties."
        }

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_query := st.chat_input("Enter your financial query"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_query)

    # Process the user query
    st.write("🔍 Searching for relevant document images...")
    pdf_analysis_results = asyncio.run(analyze_pdf_images(user_query))

    # Display results for each image in the sidebar
    with st.sidebar.expander("Source Documents"):
        for result in pdf_analysis_results:
            st.image(result["image"], caption="Analyzed Page", use_container_width=True)

    # Display intermediate results in an expander
    with st.expander("Intermediate Analysis Results"):
        for result in pdf_analysis_results:
            st.write("### Analysis:")
            st.write(f"**Summary:** {result['analysis']['Summary']}")
            st.write(f"**Key Figures:** {result['analysis']['Key Figures']}")
            st.write(f"**Risks or Notes:** {result['analysis']['Risks or Notes']}")

    # Step 2: Summarize all results
    if pdf_analysis_results:
        st.write("🔍 Generating final summary...")
        final_summary = asyncio.run(summarize_responses(user_query, pdf_analysis_results))

        # Display the final summarized response
        st.write("### Final Answer:")
        st.write(f"**Summary:** {final_summary['Final Summary']}")
        st.write(f"**Key Takeaways:** {final_summary['Key Takeaways']}")
        st.write(f"**Caveats or Uncertainties:** {final_summary['Caveats or Uncertainties']}")

        # Add assistant's response to chat history
        st.session_state.messages.append({"role": "assistant", "content": f"**Summary:** {final_summary['Final Summary']}\n**Key Takeaways:** {final_summary['Key Takeaways']}\n**Caveats or Uncertainties:** {final_summary['Caveats or Uncertainties']}"})
