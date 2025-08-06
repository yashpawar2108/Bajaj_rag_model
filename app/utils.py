import requests
import pdfplumber

def extract_text_from_pdf(pdf_url: str) -> str:
    response = requests.get(pdf_url, stream=True)
    response.raise_for_status()
    with open("temp.pdf", "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
    text = ""
    with pdfplumber.open("temp.pdf") as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text

def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 100):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
