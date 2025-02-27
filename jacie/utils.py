import base64
import tempfile
from fpdf import FPDF
import streamlit as st
import asyncio
import functools

# --- Function: Convert Image to Base64 ---
def encode_image(image_path):
    """Convert an image file to a Base64 string."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        st.error(f"⚠️ Error encoding image {image_path}: {str(e)}")
        return None

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