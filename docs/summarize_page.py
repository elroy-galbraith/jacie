import base64
import requests
import json
import time
import glob
import io
from PIL import Image
import pytesseract

### ====== IMAGE PROCESSING FUNCTIONS ====== ###
def compress_image(image_path, max_size=(800, 800)):
    """Compress and resize image to reduce memory usage."""
    try:
        with Image.open(image_path) as img:
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85, optimize=True)
            buffer.seek(0)
            return buffer.getvalue()
    except Exception as e:
        print(f"Error compressing image: {str(e)}")
        return None

def encode_image(image_path):
    """Convert an image to base64 string after compression."""
    try:
        compressed_image = compress_image(image_path)
        if compressed_image is None:
            return None
        return base64.b64encode(compressed_image).decode('utf-8').strip()
    except Exception as e:
        print(f"Error encoding image {image_path}: {str(e)}")
        return None

def extract_text_from_image(image_path):
    """Extract text from an image using OCR."""
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        return text.strip() if text else "No text extracted"
    except Exception as e:
        print(f"Error extracting text from {image_path}: {str(e)}")
        return ""

### ====== PROMPT TEMPLATE ====== ###
PAGE_SUMMARY_PROMPT = """
You will analyze an image of a PDF page and extract key information.
Provide a structured summary with:

- **Main Topic**: (What is this page about?)
- **Key Findings**: (Important text, numbers, or insights)
- **Tables & Figures**: (Mention their presence and summarize their contents)
- **Contextual Tags**: (Relevant keywords for search retrieval)

Document: {doc_name}
OCR Extracted Text (if available): {ocr_text}
"""

### ====== API REQUEST FUNCTION ====== ###
def generate_summary(image_path, doc_name):
    """Generates a structured summary of a PDF page using Ollama's Llama3 Vision Model."""
    
    print(f"Processing {image_path}...")
    
    # Convert image to base64
    image_base64 = encode_image(image_path)
    if not image_base64:
        return "Failed to encode image"

    # Extract text via OCR (for scanned PDFs)
    ocr_text = extract_text_from_image(image_path)

    # Format the prompt with document name and extracted text
    prompt = PAGE_SUMMARY_PROMPT.format(doc_name=doc_name, ocr_text=ocr_text)

    # API request payload
    payload = {
        "model": "llama3.2-vision",
        "prompt": prompt,
        "images": [image_base64]
    }

    # Send request to Ollama
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True,
            # timeout=300  # Timeout to prevent hanging
        )

        if response.status_code == 200:
            summary = ""
            for line in response.iter_lines():
                if line:
                    try:
                        json_response = json.loads(line.decode('utf-8'))
                        if 'response' in json_response:
                            chunk = json_response['response']
                            print(chunk, end='', flush=True)
                            summary += chunk
                    except json.JSONDecodeError:
                        continue
            print("\nSummary completed")
            return summary

        elif response.status_code == 429:
            print("Rate limit reached. Retrying in 5 seconds...")
            time.sleep(5)
            return generate_summary(image_path, doc_name)  # Retry

        else:
            print(f"Error {response.status_code}: {response.text}")
            return None

    except requests.exceptions.Timeout:
        print("Request timed out. Consider increasing timeout.")
    except requests.exceptions.ConnectionError:
        print("Connection failed. Is the server running?")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
    
    return None

### ====== SUMMARY STORAGE FUNCTION ====== ###
def save_summary(doc_name, page_number, summary):
    """Save the structured summary in a JSON file."""
    output_file = f"{doc_name}_summaries.json"
    entry = {
        "doc_name": doc_name,
        "page_number": page_number,
        "summary": summary
    }

    try:
        # Read existing entries or create new array
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
        except (FileNotFoundError, json.JSONDecodeError):
            data = []
        
        # Add new entry and write back
        data.append(entry)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"✅ Saved summary for {doc_name} - Page {page_number}")
    except Exception as e:
        print(f"Error saving summary: {str(e)}")

### ====== MULTI-PAGE PROCESSING FUNCTION ====== ###
def process_pdf_images(pdf_name, image_folder):
    """Processes all images of a PDF and generates summaries."""
    images = sorted(glob.glob(f"{image_folder}{pdf_name}_page_*.jpg"))
    
    if not images:
        print("No images found for this document.")
        return
    
    for i, image_path in enumerate(images, start=1):
        print(f"\n🚀 Processing page {i}/{len(images)}...")
        summary = generate_summary(image_path, pdf_name)
        
        if summary:
            save_summary(pdf_name, i, summary)
        else:
            print(f"⚠️ Skipping page {i} due to errors.")

### ====== MAIN EXECUTION ====== ###
if __name__ == "__main__":
    pdf_name = "EduFocal-Limited-2023-Financial-Statements"
    image_folder = "/Users/galbraithelroy/Documents/jacie/docs/pdf_images/"
    
    print(f"🔍 Processing all images for {pdf_name}...\n")
    process_pdf_images(pdf_name, image_folder)