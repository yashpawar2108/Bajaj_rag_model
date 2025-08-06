import os
import asyncio
import aiohttp
import aiofiles
import pdfplumber
import time
import hashlib
import pickle
import logging
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel

# Langchain components
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Simple retriever without BM25 (avoid compilation issues)
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from functools import lru_cache

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Setup ---
app = FastAPI(title="HackRx RAG API")
router = APIRouter()

# --- Simple In-Memory Cache ---
MEMORY_CACHE = {}
CACHE_MAX_SIZE = 50  # Limit cache size for memory constraints

# --- Bearer Token Authentication ---
EXPECTED_TOKEN = "fde07a66b0421ad3377eb15eef6763dc34f3da77c18417db687fb0b2dc4d08df"

def verify_bearer_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    token = authorization.split(" ")[1]
    if token != EXPECTED_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")

# --- Request Model ---
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

# --- Utility Functions ---
def generate_cache_key(*args) -> str:
    """Generate consistent cache key from arguments"""
    key_string = "|".join(str(arg) for arg in args)
    return hashlib.md5(key_string.encode()).hexdigest()

def clean_memory_cache():
    """Clean memory cache if it gets too large"""
    global MEMORY_CACHE
    if len(MEMORY_CACHE) > CACHE_MAX_SIZE:
        # Remove oldest entries (simple FIFO)
        items_to_remove = len(MEMORY_CACHE) - CACHE_MAX_SIZE + 10
        keys_to_remove = list(MEMORY_CACHE.keys())[:items_to_remove]
        for key in keys_to_remove:
            MEMORY_CACHE.pop(key, None)
        logger.info(f"Cleaned {items_to_remove} items from memory cache")

# --- Simple PDF Processing ---
async def extract_text_from_pdf_async(pdf_url: str) -> str:
    """Async PDF text extraction with simple memory caching"""
    
    cache_key = generate_cache_key(pdf_url)
    
    # Check memory cache
    if cache_key in MEMORY_CACHE:
        logger.info("PDF text retrieved from memory cache")
        return MEMORY_CACHE[cache_key]
    
    temp_filename = f"temp_{int(time.time())}.pdf"
    
    try:
        # Async download with timeout
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(pdf_url) as response:
                response.raise_for_status()
                async with aiofiles.open(temp_filename, "wb") as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)
        
        # Extract text synchronously (simpler approach)
        text = _extract_text_sync(temp_filename)
        
        # Cache in memory
        if text and len(text) > 100:
            clean_memory_cache()
            MEMORY_CACHE[cache_key] = text
            logger.info("PDF text cached in memory")
        
        return text
    
    finally:
        # Cleanup
        try:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        except:
            pass

def _extract_text_sync(filename: str) -> str:
    """Synchronous text extraction helper"""
    text = ""
    try:
        with pdfplumber.open(filename) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    page_text = page_text.strip()
                    page_text = " ".join(page_text.split())
                    text += page_text + "\n\n"
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise
    return text

# --- Simple Model Setup ---
class ModelRouter:
    def __init__(self):
        # Use single model to avoid complexity
        self.model = ChatOpenAI(
            model="llama3-8b-8192",
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1,
            max_tokens=512,
        )
    
    def get_model(self):
        return self.model

model_router = ModelRouter()

# --- Simple Chunking ---
def create_simple_chunks(text: str) -> List[Document]:
    """Simple but effective chunking"""
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""],
    )
    
    chunks = text_splitter.split_text(text)
    documents = []
    
    for i, chunk in enumerate(chunks):
        metadata = {
            "chunk_id": i,
            "chunk_size": len(chunk),
            "total_chunks": len(chunks)
        }
        documents.append(Document(page_content=chunk, metadata=metadata))
    
    return documents

# --- Embedding Setup ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# --- Document Processing Pipeline ---
async def process_document(document_url: str) -> tuple:
    """Simple document processing pipeline"""
    
    # Check if vectorstore is already cached
    cache_key = f"vectorstore_{generate_cache_key(document_url)}"
    
    if cache_key in MEMORY_CACHE:
        logger.info("Vectorstore retrieved from memory cache")
        return MEMORY_CACHE[cache_key]
    
    logger.info("Processing document from scratch...")
    
    # Extract text
    start_time = time.time()
    text = await extract_text_from_pdf_async(document_url)
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
    
    extraction_time = time.time() - start_time
    logger.info(f"Text extracted in {extraction_time:.2f}s ({len(text)} chars)")
    
    # Create chunks
    start_time = time.time()
    documents = create_simple_chunks(text)
    chunking_time = time.time() - start_time
    logger.info(f"Created {len(documents)} chunks in {chunking_time:.2f}s")
    
    # Create vectorstore
    start_time = time.time()
    vectorstore = FAISS.from_documents(documents, embedding=embeddings)
    vectorization_time = time.time() - start_time
    logger.info(f"Vectorstore created in {vectorization_time:.2f}s")
    
    # Cache vectorstore and documents in memory
    result = (vectorstore, documents)
    clean_memory_cache()
    MEMORY_CACHE[cache_key] = result
    
    total_time = extraction_time + chunking_time + vectorization_time
    logger.info(f"Total processing time: {total_time:.2f}s")
    
    return result

# --- Simple Answer Generation ---
async def get_answer_async(vectorstore, question: str, documents: List[Document]) -> str:
    """Simple answer generation"""
    
    # Check answer cache
    cache_key = f"answer_{generate_cache_key(question, len(documents))}"
    if cache_key in MEMORY_CACHE:
        logger.info("Answer retrieved from memory cache")
        return MEMORY_CACHE[cache_key]
    
    # Get model
    llm = model_router.get_model()
    
    # Simple prompt
    template = """Use the provided context to answer the question accurately and concisely.

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Create simple retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
    )

    # Get answer
    try:
        result = qa_chain.invoke({"query": question})
        answer = result["result"] if isinstance(result, dict) else str(result)
        
        # Cache answer
        clean_memory_cache()
        MEMORY_CACHE[cache_key] = answer
        
        return answer
    except Exception as e:
        logger.error(f"Error getting answer: {str(e)}")
        return "I apologize, but I encountered an error while processing your question. Please try again."

# --- Batch Processing ---
async def process_questions_batch(vectorstore, questions: List[str], documents: List[Document]) -> List[str]:
    """Process questions with simple batching"""
    
    answers = []
    
    # Process questions one by one to avoid memory issues
    for question in questions:
        try:
            answer = await get_answer_async(vectorstore, question, documents)
            answers.append(answer)
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            answers.append("Sorry, I couldn't process this question.")
        
        # Small delay to prevent overwhelming the system
        await asyncio.sleep(0.1)
    
    return answers

# --- Cache Management ---
@router.get("/cache/status")
async def get_cache_status():
    """Get cache status"""
    return {
        "memory_cache_size": len(MEMORY_CACHE),
        "memory_cache_max_size": CACHE_MAX_SIZE,
        "cached_items": list(MEMORY_CACHE.keys())[:10]  # Show first 10 keys
    }

@router.delete("/cache/clear")
async def clear_cache():
    """Clear memory cache"""
    global MEMORY_CACHE
    cache_size = len(MEMORY_CACHE)
    MEMORY_CACHE.clear()
    return {"message": f"Cleared {cache_size} items from memory cache"}

# --- Main Endpoint ---
@router.post("/run", dependencies=[Depends(verify_bearer_token)])
async def process_query(request: QueryRequest):
    start_time = time.time()
    
    logger.info("Received new query request")
    logger.info(f"Document: {request.documents}")
    logger.info(f"Questions: {len(request.questions)}")

    try:
        # Process document
        vectorstore, documents = await process_document(request.documents)
        
        # Process questions
        answers = await process_questions_batch(vectorstore, request.questions, documents)
        
        # Log completion
        total_time = time.time() - start_time
        logger.info(f"Query completed in {total_time:.2f}s")
        
        return {"answers": answers}
    
    except aiohttp.ClientError as e:
        logger.error(f"PDF download failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# --- Register Router ---
app.include_router(router, prefix="/api/v1/hackrx")

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    logger.info("RAG API starting up...")
    logger.info("Simple memory caching enabled")
    logger.info("Optimized for Render deployment")

# --- Health Check ---
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "HackRx RAG API",
        "deployment": "render_optimized",
        "cache_type": "memory_only"
    }
