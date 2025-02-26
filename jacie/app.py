import streamlit as st
import json
from langchain_community.vectorstores import FAISS
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_vertexai import ChatVertexAI
from langchain.memory import ConversationBufferMemory
import os

st.set_page_config(
    page_title="Jacie - Financial Assistant",
    page_icon="💼",
    layout="wide"
)

st.title("Jacie - Your Financial Assistant")

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

# Create a retriever
retriever = vector_store.as_retriever(search_kwargs={'k': 6})

# Initialize ChatVertexAI with error handling
try:
    llm = ChatVertexAI(model="gemini-1.5-pro", temperature=0)
except Exception as e:
    st.error(f"⚠️ Error initializing ChatVertexAI: {str(e)}")
    st.stop()

# Initialize conversation memory
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

memory = st.session_state["memory"]

# Define prompt with memory support
prompt = ChatPromptTemplate.from_template(
    """
    You are an experienced financial analyst. Your task is to answer the user's question based on the available context.
    Remember previous interactions to provide relevant and contextual responses.
    If the user's question is not related to the available context, just say "I'm sorry, I don't have information on that topic."
    If the user's question is not clear, ask for clarification.
    Do not make up information, only answer what is provided in the available context.

    Chat History:
    {chat_history}
    
    Available context:
    {docs}

    Question: {question}
    """
)

answer_chain = (
    {
        "question": lambda x: x["question"],
        "docs": lambda x: retriever.invoke(x["question"]),
        "chat_history":lambda x: memory.load_memory_variables({})["chat_history"]
    }
    | prompt
    | llm
    | StrOutputParser()
)

def get_answer(question: str) -> str:
    """Get the answer to the question while storing memory."""
    response = answer_chain.invoke({"question": question})
    
    # Store the response in memory
    memory.save_context({"question": question}, {"response": response})
    
    return response

# "How to Use" section in the sidebar
with st.sidebar.expander("ℹ️ How to use"):
    st.markdown(
        """
        1. Type your question in the chat input below.
        2. Press Enter to generate an SQL query.
        3. View the generated SQL query in this sidebar.
        4. See the answer displayed in the chat interface.
        """
    )
    
# Sidebar for displaying generated SQL
st.sidebar.title("Source Documents")

# Initialize chat history and visualization data
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    
    # Display chat messages and visualizations from history on app rerun
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat interface
if question := st.chat_input("How can I help you?"):
    st.session_state["messages"].append({"role": "user", "content": question})
    
    with st.chat_message("user"):
        st.markdown(question)

    # Get the answer from LLM
    answer = get_answer(question)
    
    with st.sidebar.expander("Source Documents"):
        for doc in retriever.get_relevant_documents(question):
            st.markdown(f"**{doc.metadata['document_name']}**")
            st.markdown(f"Page {doc.metadata['page_number']}")
            st.markdown(doc.page_content)
            st.markdown("---")
    
    # Display assistant response in chat
    with st.chat_message("assistant"):
        st.markdown(answer)
    
    # Add assistant message to chat history
    st.session_state["messages"].append({"role": "assistant", "content": answer})
