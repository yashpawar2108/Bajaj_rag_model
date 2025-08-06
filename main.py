import os
import asyncio
import aiohttp
import aiofiles
import pdfplumber
import time
import hashlib
import logging
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

# Langchain components - using correct imports for compatibility
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Setup ---
app = FastAPI(title="HackRx RAG API")
router = APIRouter()

# --- Simple In-Memory Cache ---
MEMORY_CACHE = {}
CACHE_MAX_SIZE = 20  # Reduced for memory efficiency

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
    return hashlib.md5(key_string.encode()).hexdigest()[:16]  # Shorter keys

def clean_memory_cache():
    """Clean memory cache if it gets too large"""
    global MEMORY_CACHE
    if len(MEMORY_CACHE) > CACHE_MAX_SIZE:
        # Remove half of the cache (simple cleanup)
        items_to_remove = len(MEMORY_CACHE) // 2
        keys_to_remove = list(MEMORY_CACHE.keys())[:items_to_remove]
        for key in keys_to_remove:
            MEMORY_CACHE.pop(key, None)
        logger.info(f"Cleaned {items_to_remove} items from memory cache")

# --- PDF Processing ---
async def extract_text_from_pdf_async(pdf_url: str) -> str:
    """Async PDF text extraction with memory caching"""
    
    cache_key = f"pdf_{generate_cache_key(pdf_url)}"
    
    # Check memory cache
    if cache_key in MEMORY_CACHE:
        logger.info("PDF text retrieved from memory cache")
        return MEMORY_CACHE[cache_key]
    
    temp_filename = f"temp_{int(time.time() * 1000)}.pdf"
    
    try:
        # Download PDF
        timeout = aiohttp.ClientTimeout(total=120)  # Increased timeout
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(pdf_url, headers={'User-Agent': 'Mozilla/5.0'}) as response:
                response.raise_for_status()
                
                # Write to temp file
                async with aiofiles.open(temp_filename, "wb") as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)
        
        # Extract text
        text = await asyncio.get_event_loop().run_in_executor(
            None, _extract_text_sync, temp_filename
        )
        
        # Cache in memory if successful
        if text and len(text.strip()) > 100:
            clean_memory_cache()
            MEMORY_CACHE[cache_key] = text
            logger.info(f"PDF text cached in memory ({len(text)} chars)")
        
        return text
    
    except Exception as e:
        logger.error(f"PDF processing error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")
    
    finally:
        # Cleanup temp file
        try:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file: {e}")

def _extract_text_sync(filename: str) -> str:
    """Synchronous text extraction"""
    text_parts = []
    
    try:
        with pdfplumber.open(filename) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        # Clean up text
                        page_text = page_text.strip()
                        page_text = " ".join(page_text.split())  # Normalize whitespace
                        text_parts.append(page_text)
                        
                        # Log progress for large PDFs
                        if (page_num + 1) % 10 == 0:
                            logger.info(f"Processed {page_num + 1} pages")
                            
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                    continue
                    
    except Exception as e:
        logger.error(f"Error opening PDF: {str(e)}")
        raise
    
    final_text = "\n\n".join(text_parts)
    
    if not final_text.strip():
        raise ValueError("No readable text found in PDF")
    
    logger.info(f"Successfully extracted {len(final_text)} characters from PDF")
    return final_text

# --- Text Chunking ---
def create_document_chunks(text: str) -> List[Document]:
    """Create document chunks with overlap"""
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        is_separator_regex=False
    )
    
    try:
        chunks = text_splitter.split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks")
        
        documents = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():  # Only add non-empty chunks
                doc = Document(
                    page_content=chunk.strip(),
                    metadata={
                        "chunk_id": i,
                        "chunk_size": len(chunk),
                        "total_chunks": len(chunks)
                    }
                )
                documents.append(doc)
        
        logger.info(f"Created {len(documents)} valid document chunks")
        return documents
        
    except Exception as e:
        logger.error(f"Error creating chunks: {str(e)}")
        raise

# --- Embedding Setup ---
def get_embeddings():
    """Get Google embeddings with error handling"""
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to initialize embeddings")

# --- LLM Setup ---
def get_chat_model():
    """Get ChatOpenAI model with error handling"""
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        return ChatOpenAI(
            model="llama3-8b-8192",
            openai_api_base="https://api.groq.com/openai/v1",
            openai_api_key=groq_api_key,
            temperature=0.2,
            max_tokens=1024,
            request_timeout=60
        )
    except Exception as e:
        logger.error(f"Failed to initialize chat model: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to initialize language model")

# --- Document Processing Pipeline ---
async def process_document_pipeline(document_url: str) -> tuple:
    """Complete document processing pipeline"""
    
    # Check if already processed and cached
    cache_key = f"processed_{generate_cache_key(document_url)}"
    
    if cache_key in MEMORY_CACHE:
        logger.info("Processed document retrieved from cache")
        return MEMORY_CACHE[cache_key]
    
    logger.info("Starting document processing pipeline...")
    start_time = time.time()
    
    try:
        # Step 1: Extract text from PDF
        logger.info("Step 1: Extracting text from PDF...")
        text = await extract_text_from_pdf_async(document_url)
        
        if len(text.strip()) < 100:
            raise HTTPException(status_code=400, detail="PDF contains insufficient text content")
        
        # Step 2: Create document chunks
        logger.info("Step 2: Creating document chunks...")
        documents = create_document_chunks(text)
        
        if len(documents) == 0:
            raise HTTPException(status_code=400, detail="No valid document chunks created")
        
        # Step 3: Create embeddings and vectorstore
        logger.info("Step 3: Creating vectorstore...")
        embeddings = get_embeddings()
        
        # Create vectorstore with error handling
        try:
            vectorstore = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: FAISS.from_documents(documents, embeddings)
            )
        except Exception as e:
            logger.error(f"Failed to create vectorstore: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create document index")
        
        # Cache the results
        result = (vectorstore, documents)
        clean_memory_cache()
        MEMORY_CACHE[cache_key] = result
        
        processing_time = time.time() - start_time
        logger.info(f"Document processing completed in {processing_time:.2f}s")
        logger.info(f"Created vectorstore with {len(documents)} chunks")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Document processing failed")

# --- Question Answering ---
async def answer_question(vectorstore, question: str, documents: List[Document]) -> str:
    """Answer a single question using the vectorstore"""
    
    # Check answer cache
    cache_key = f"answer_{generate_cache_key(question, len(documents))}"
    if cache_key in MEMORY_CACHE:
        logger.info("Answer retrieved from cache")
        return MEMORY_CACHE[cache_key]
    
    try:
        # Get language model
        llm = get_chat_model()
        
        # Create prompt template
        template = """Use the following context to answer the question. Be accurate and concise.

Context:
{context}

Question: {question}

Answer:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=False
        )
        
        # Get answer
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: qa_chain.run(question)
        )
        
        # Clean and validate answer
        if isinstance(result, dict) and "result" in result:
            answer = result["result"]
        else:
            answer = str(result)
        
        answer = answer.strip()
        
        if not answer or len(answer) < 10:
            answer = "I couldn't find enough information to answer this question based on the provided document."
        
        # Cache the answer
        clean_memory_cache()
        MEMORY_CACHE[cache_key] = answer
        
        return answer
        
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        return "I encountered an error while processing your question. Please try rephrasing it."

# --- Batch Question Processing ---
async def process_questions(vectorstore, questions: List[str], documents: List[Document]) -> List[str]:
    """Process multiple questions sequentially"""
    
    answers = []
    
    for i, question in enumerate(questions, 1):
        logger.info(f"Processing question {i}/{len(questions)}")
        
        try:
            answer = await answer_question(vectorstore, question, documents)
            answers.append(answer)
            
            # Small delay to prevent overwhelming the system
            if i < len(questions):
                await asyncio.sleep(0.5)
                
        except Exception as e:
            logger.error(f"Failed to process question {i}: {str(e)}")
            answers.append("Sorry, I couldn't process this question due to an error.")
    
    return answers

# --- API Endpoints ---
@router.post("/run", dependencies=[Depends(verify_bearer_token)])
async def process_query(request: QueryRequest):
    """Main endpoint to process document and answer questions"""
    
    start_time = time.time()
    
    logger.info("=" * 50)
    logger.info("NEW REQUEST RECEIVED")
    logger.info(f"Document URL: {request.documents}")
    logger.info(f"Number of questions: {len(request.questions)}")
    logger.info("=" * 50)
    
    try:
        # Validate input
        if not request.documents.strip():
            raise HTTPException(status_code=400, detail="Document URL is required")
        
        if not request.questions or len(request.questions) == 0:
            raise HTTPException(status_code=400, detail="At least one question is required")
        
        if len(request.questions) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 questions allowed")
        
        # Process document
        logger.info("Processing document...")
        vectorstore, documents = await process_document_pipeline(request.documents)
        
        # Process questions
        logger.info("Processing questions...")
        answers = await process_questions(vectorstore, request.questions, documents)
        
        # Log completion
        total_time = time.time() - start_time
        logger.info(f"Request completed successfully in {total_time:.2f} seconds")
        
        return {
            "answers": answers,
            "processing_time": round(total_time, 2),
            "document_chunks": len(documents)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# --- Utility Endpoints ---
@router.get("/cache/status")
async def cache_status():
    """Get current cache status"""
    return {
        "cache_size": len(MEMORY_CACHE),
        "max_cache_size": CACHE_MAX_SIZE,
        "cache_keys": list(MEMORY_CACHE.keys())
    }

@router.delete("/cache/clear")
async def clear_cache():
    """Clear all cached data"""
    global MEMORY_CACHE
    cache_size = len(MEMORY_CACHE)
    MEMORY_CACHE.clear()
    logger.info(f"Cache cleared - removed {cache_size} items")
    return {"message": f"Cache cleared successfully. Removed {cache_size} items."}

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "HackRx RAG API",
        "version": "render-optimized",
        "cache_size": len(MEMORY_CACHE),
        "environment_vars": {
            "GOOGLE_API_KEY": "✓" if os.getenv("GOOGLE_API_KEY") else "✗",
            "GROQ_API_KEY": "✓" if os.getenv("GROQ_API_KEY") else "✗"
        }
    }

# --- Register Router ---
app.include_router(router, prefix="/api/v1/hackrx")

# --- Application Events ---
@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info("🚀 HackRx RAG API Starting Up...")
    logger.info("📋 Configuration:")
    logger.info(f"   - Memory cache enabled (max {CACHE_MAX_SIZE} items)")
    logger.info(f"   - Google API Key: {'✓' if os.getenv('GOOGLE_API_KEY') else '✗'}")
    logger.info(f"   - Groq API Key: {'✓' if os.getenv('GROQ_API_KEY') else '✗'}")
    logger.info("✅ API Ready for requests!")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info("🛑 HackRx RAG API Shutting Down...")
    MEMORY_CACHE.clear()
    logger.info("✅ Cleanup completed")

# --- Root Endpoint ---
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "HackRx RAG API",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }
