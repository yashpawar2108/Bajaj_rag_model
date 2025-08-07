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
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel

# Langchain components
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# ✅ Use direct imports for retrievers to avoid 'pwd' import on Windows
from langchain.retrievers.bm25 import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever



# Optional imports for optimizations (with fallbacks)
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from functools import lru_cache

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Setup ---
app = FastAPI(title="HackRx RAG API")
router = APIRouter()

# Thread pool for CPU-bound operations
executor = ThreadPoolExecutor(max_workers=6)

# --- Cache Setup ---
CACHE_ENABLED = False
redis_client = None

if REDIS_AVAILABLE:
    try:
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
        redis_client.ping()
        CACHE_ENABLED = True
        logger.info("Redis cache connected successfully")
    except:
        CACHE_ENABLED = False
        logger.warning("Redis not available, caching disabled")

# --- File-based Cache Directory ---
CACHE_DIR = "document_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Bearer Token Authentication ---
EXPECTED_TOKEN = "7ae90faf72ce42e929314a6192e64395286019a5982e85589efeab8312d6061f"

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

def generate_document_cache_name(url: str) -> str:
    """Generate a clean cache name for a document URL"""
    # Create a hash of the URL for consistent naming
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    
    # Try to extract a meaningful name from the URL
    try:
        # Extract filename from URL (remove query parameters)
        filename = url.split('/')[-1].split('?')[0]
        # Remove file extension and clean up
        clean_name = filename.replace('.pdf', '').replace('%20', '_').replace(' ', '_')
        # Keep only alphanumeric and underscores
        clean_name = ''.join(c for c in clean_name if c.isalnum() or c == '_')
        if clean_name and len(clean_name) > 3:
            return f"{clean_name}_{url_hash}"
    except:
        pass
    
    return f"document_{url_hash}"

# --- Document Cache Management ---
def get_document_cache_paths(url: str) -> tuple:
    """Get file paths for cached document data"""
    cache_name = generate_document_cache_name(url)
    vectorstore_path = os.path.join(CACHE_DIR, f"vectorstore_{cache_name}")
    documents_path = os.path.join(CACHE_DIR, f"documents_{cache_name}.pkl")
    metadata_path = os.path.join(CACHE_DIR, f"metadata_{cache_name}.pkl")
    return vectorstore_path, documents_path, metadata_path

def is_document_cached(url: str) -> bool:
    """Check if document is already cached"""
    vectorstore_path, documents_path, metadata_path = get_document_cache_paths(url)
    return (os.path.exists(vectorstore_path) and 
            os.path.exists(documents_path) and 
            os.path.exists(metadata_path))

def save_document_cache(url: str, vectorstore, documents: List[Document], extracted_text: str):
    """Save processed document to cache"""
    try:
        vectorstore_path, documents_path, metadata_path = get_document_cache_paths(url)
        cache_name = generate_document_cache_name(url)
        
        # Save vectorstore
        vectorstore.save_local(vectorstore_path)
        
        # Save documents
        with open(documents_path, 'wb') as f:
            pickle.dump(documents, f)
        
        # Save metadata
        metadata = {
            'url': url,
            'cache_name': cache_name,
            'timestamp': time.time(),
            'num_documents': len(documents),
            'text_length': len(extracted_text),
            'text_preview': extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"📦 Document cached as: {cache_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to save document cache: {str(e)}")
        return False

def load_document_cache(url: str) -> tuple:
    """Load processed document from cache"""
    try:
        vectorstore_path, documents_path, metadata_path = get_document_cache_paths(url)
        
        # Load vectorstore
        vectorstore = FAISS.load_local(
            vectorstore_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Load documents
        with open(documents_path, 'rb') as f:
            documents = pickle.load(f)
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        logger.info(f"✅ Loaded cached document: {metadata['cache_name']}")
        logger.info(f"   - Cached: {time.ctime(metadata['timestamp'])}")
        logger.info(f"   - Chunks: {metadata['num_documents']}")
        logger.info(f"   - Text length: {metadata['text_length']} chars")
        
        return vectorstore, documents, metadata
    except Exception as e:
        logger.error(f"Failed to load document cache: {str(e)}")
        return None, None, None

# --- Sentence Transformer (Optional) ---
if ML_AVAILABLE:
    @lru_cache(maxsize=1)
    def get_sentence_transformer():
        try:
            return SentenceTransformer('all-MiniLM-L6-v2')
        except:
            return None
    
    def calculate_semantic_similarity(text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        try:
            model = get_sentence_transformer()
            if model is None:
                return 0.5
            embeddings = model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except:
            return 0.5
else:
    def calculate_semantic_similarity(text1: str, text2: str) -> float:
        return 0.5

# --- Model Router ---
class ModelRouter:
    def __init__(self):
        self.fast_model = ChatOpenAI(
            model="llama3-8b-8192",
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1,
            max_tokens=256,
        )
        self.powerful_model = ChatOpenAI(
            model="llama3-70b-8192",
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1,
            max_tokens=512,
        )
    
    def route_model(self, question: str, context_length: int = 0):
        """Route to appropriate model based on question complexity"""
        question_words = len(question.split())
        question_complexity = (
            question_words > 15 or
            any(word in question.lower() for word in ['analyze', 'compare', 'explain', 'describe', 'evaluate', 'summarize']) or
            question.count('?') > 1 or
            context_length > 2000
        )
        
        return self.powerful_model if question_complexity else self.fast_model

model_router = ModelRouter()

# --- Enhanced PDF Processing ---
async def extract_text_from_pdf_async(pdf_url: str) -> str:
    """Async PDF text extraction with caching"""
    
    # Check Redis cache first (for text only)
    if CACHE_ENABLED and redis_client:
        cache_key = f"pdf_text:{generate_cache_key(pdf_url)}"
        try:
            cached_text = redis_client.get(cache_key)
            if cached_text:
                logger.info("PDF text retrieved from Redis cache")
                return pickle.loads(cached_text)
        except Exception as e:
            logger.warning(f"Redis cache retrieval failed: {str(e)}")
    
    temp_filename = f"temp_{asyncio.current_task().get_name()}_{int(time.time())}.pdf"
    
    try:
        # Async download with timeout
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(pdf_url) as response:
                response.raise_for_status()
                async with aiofiles.open(temp_filename, "wb") as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)
        
        # Extract text in thread pool
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(executor, _extract_text_sync, temp_filename)
        
        # Cache the extracted text in Redis
        if CACHE_ENABLED and redis_client and text:
            cache_key = f"pdf_text:{generate_cache_key(pdf_url)}"
            try:
                redis_client.setex(cache_key, 3600, pickle.dumps(text))
            except Exception as e:
                logger.warning(f"Redis cache storage failed: {str(e)}")
        
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

# --- Query Enhancement ---
async def enhance_query(question: str) -> str:
    """Enhance query with synonyms and related terms"""
    try:
        enhancement_prompt = f"""Rewrite this question to include relevant synonyms and related terms that would help find better information in a document. Keep it concise but comprehensive.

Original: {question}
Enhanced:"""
        
        llm = model_router.fast_model
        loop = asyncio.get_event_loop()
        enhanced = await loop.run_in_executor(
            executor,
            lambda: llm.invoke(enhancement_prompt).content
        )
        return enhanced.strip()
    except:
        return question

# --- Semantic Chunking ---
def create_semantic_chunks(text: str) -> List[Document]:
    """Create semantically coherent chunks with fallback"""
    if not ML_AVAILABLE:
        return create_advanced_chunks(text)
    
    try:
        # Split into sentences
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            if char in '.!?' and len(current_sentence.strip()) > 10:
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        if len(sentences) < 3:
            return create_advanced_chunks(text)
        
        # Group sentences into chunks
        model = get_sentence_transformer()
        if model is None:
            return create_advanced_chunks(text)
            
        sentence_embeddings = model.encode(sentences)
        
        chunks = []
        current_chunk = []
        current_chunk_size = 0
        max_chunk_size = 800
        min_chunk_size = 200
        
        for i, sentence in enumerate(sentences):
            sentence_len = len(sentence)
            
            if current_chunk_size + sentence_len > max_chunk_size and len(current_chunk) > 0:
                should_split = True
                if i < len(sentences) - 1 and current_chunk_size < max_chunk_size * 1.2:
                    try:
                        current_embedding = np.mean([sentence_embeddings[j] for j in range(len(current_chunk))], axis=0)
                        next_embedding = sentence_embeddings[i]
                        similarity = cosine_similarity([current_embedding], [next_embedding])[0][0]
                        
                        if similarity > 0.7:
                            should_split = False
                    except:
                        should_split = True
                
                if should_split and current_chunk_size >= min_chunk_size:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(Document(
                        page_content=chunk_text,
                        metadata={"chunk_size": len(chunk_text), "chunk_type": "semantic"}
                    ))
                    current_chunk = [sentence]
                    current_chunk_size = sentence_len
                else:
                    current_chunk.append(sentence)
                    current_chunk_size += sentence_len
            else:
                current_chunk.append(sentence)
                current_chunk_size += sentence_len
        
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(Document(
                page_content=chunk_text,
                metadata={"chunk_size": len(chunk_text), "chunk_type": "semantic"}
            ))
        
        return chunks if chunks else create_advanced_chunks(text)
    
    except Exception as e:
        logger.warning(f"Semantic chunking failed, using recursive: {str(e)}")
        return create_advanced_chunks(text)

def create_advanced_chunks(text: str) -> List[Document]:
    """Advanced chunking with recursive text splitter and metadata"""
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        separators=[
            "\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", "",
        ],
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_text(text)
    documents = []
    
    for i, chunk in enumerate(chunks):
        metadata = {
            "chunk_id": i,
            "chunk_size": len(chunk),
            "total_chunks": len(chunks),
            "chunk_type": "recursive"
        }
        documents.append(Document(page_content=chunk, metadata=metadata))
    
    return documents

# --- Enhanced Embedding Setup ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# --- Document Processing Pipeline ---
async def get_or_create_document_pipeline(document_url: str) -> tuple:
    """Main pipeline to get or create processed documents with caching"""
    
    # Check if document is already cached
    if is_document_cached(document_url):
        logger.info("🔍 Document found in cache, loading...")
        vectorstore, documents, metadata = load_document_cache(document_url)
        
        if vectorstore is not None and documents is not None:
            return vectorstore, documents, metadata
        else:
            logger.warning("⚠️ Cache corrupted, rebuilding...")
    
    # Document not cached or cache corrupted, process it
    logger.info("🚀 Processing document from scratch...")
    
    # Step 1: Extract text
    start_time = time.time()
    text = await extract_text_from_pdf_async(document_url)
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
    
    extraction_time = time.time() - start_time
    logger.info(f"📄 Text extracted in {extraction_time:.2f}s ({len(text)} chars)")
    
    # Step 2: Create chunks
    start_time = time.time()
    documents = create_semantic_chunks(text)
    chunking_time = time.time() - start_time
    logger.info(f"✂️ Created {len(documents)} chunks in {chunking_time:.2f}s")
    
    # Step 3: Create vectorstore
    start_time = time.time()
    loop = asyncio.get_event_loop()
    vectorstore = await loop.run_in_executor(
        executor, 
        lambda: FAISS.from_documents(documents, embedding=embeddings)
    )
    vectorization_time = time.time() - start_time
    logger.info(f"🔢 Vectorstore created in {vectorization_time:.2f}s")
    
    # Step 4: Save to cache
    cache_success = save_document_cache(document_url, vectorstore, documents, text)
    
    if cache_success:
        logger.info("💾 Document successfully cached for future use")
    else:
        logger.warning("⚠️ Failed to cache document")
    
    # Create metadata for consistency with cached version
    metadata = {
        'url': document_url,
        'cache_name': generate_document_cache_name(document_url),
        'timestamp': time.time(),
        'num_documents': len(documents),
        'text_length': len(text),
        'processing_time': extraction_time + chunking_time + vectorization_time
    }
    
    return vectorstore, documents, metadata

# --- Hybrid Retrieval ---
def create_hybrid_retriever(documents: List[Document], vectorstore, k: int = 6):
    """Create hybrid retriever combining BM25 and semantic search"""
    try:
        # BM25 for keyword matching
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = max(2, k // 3)
        
        # Semantic search
        faiss_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k - bm25_retriever.k, "fetch_k": k * 2}
        )
        
        # Combine both
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.3, 0.7]
        )
        
        logger.info("🔄 Hybrid retriever created successfully")
        return ensemble_retriever
    
    except Exception as e:
        logger.warning(f"Hybrid retriever creation failed, using semantic only: {str(e)}")
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k, "fetch_k": k * 2}
        )

# --- Answer Quality Scoring ---
def score_answer_quality(question: str, answer: str, context: str) -> float:
    """Score answer quality based on relevance and grounding"""
    try:
        if len(answer.strip()) < 20:
            return 0.3
        
        generic_phrases = ["i don't know", "cannot find", "not mentioned", "unclear"]
        if any(phrase in answer.lower() for phrase in generic_phrases):
            return 0.4
        
        relevance_score = calculate_semantic_similarity(question, answer)
        grounding_score = calculate_semantic_similarity(answer, context)
        
        final_score = (relevance_score * 0.6) + (grounding_score * 0.4)
        
        if any(char.isdigit() for char in answer):
            final_score += 0.1
        
        return min(final_score, 1.0)
    
    except:
        return 0.5

# --- Enhanced LLM Query ---
async def get_answer_async(vectorstore, question: str, documents: List[Document]) -> str:
    """Enhanced answer generation with quality control"""
    
    # Cache check (Redis for quick Q&A caching)
    if CACHE_ENABLED and redis_client:
        cache_key = f"answer:{generate_cache_key(question, len(documents))}"
        try:
            cached_answer = redis_client.get(cache_key)
            if cached_answer:
                logger.info("💨 Answer retrieved from Redis cache")
                return pickle.loads(cached_answer)
        except:
            pass
    
    # Enhance query
    enhanced_question = await enhance_query(question)
    
    # Select model
    context_length = sum(len(doc.page_content) for doc in documents)
    llm = model_router.route_model(question, context_length)
    
    # Enhanced prompt
    template = """You are an expert document analyst. Use the provided context to answer questions accurately and comprehensively.

Instructions:
1. Synthesize information from the context - don't copy-paste directly
2. Use clear, accessible language that non-experts can understand  
3. Provide complete, relevant information in a well-formed paragraph
4. Start directly with your answer - no introductory phrases
5. If answering yes/no questions, start with the answer then explain
6. Base your response only on the provided context
7. Be specific and include relevant details when available

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Create retriever
    retriever = create_hybrid_retriever(documents, vectorstore, k=6)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    # Run inference
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor,
        lambda: qa_chain.invoke({"query": enhanced_question})
    )
    
    answer = result["result"]
    source_docs = result.get("source_documents", [])
    
    # Quality control
    if source_docs and ML_AVAILABLE:
        context = "\n".join([doc.page_content for doc in source_docs])
        quality_score = score_answer_quality(question, answer, context)
        
        if quality_score < 0.5 and enhanced_question != question:
            logger.info("🔄 Low quality answer, retrying with original question")
            result = await loop.run_in_executor(
                executor,
                lambda: qa_chain.invoke({"query": question})
            )
            answer = result["result"]
    
    # Cache answer in Redis (short-term)
    if CACHE_ENABLED and redis_client and len(answer.strip()) > 10:
        cache_key = f"answer:{generate_cache_key(question, len(documents))}"
        try:
            redis_client.setex(cache_key, 900, pickle.dumps(answer))  # 15 minutes
        except:
            pass
    
    return answer

# --- Batch Processing ---
async def batch_process_questions(vectorstore, questions: List[str], documents: List[Document]) -> List[str]:
    """Process questions with smart batching"""
    
    # Categorize questions
    simple_questions = []
    complex_questions = []
    
    for q in questions:
        if len(q.split()) < 10 and not any(word in q.lower() for word in ['analyze', 'compare', 'explain']):
            simple_questions.append(q)
        else:
            complex_questions.append(q)
    
    all_results = []
    question_order = {q: i for i, q in enumerate(questions)}
    
    # Process simple questions in batches of 4
    simple_batch_size = 20
    for i in range(0, len(simple_questions), simple_batch_size):
        batch = simple_questions[i:i + simple_batch_size]
        batch_tasks = [get_answer_async(vectorstore, q, documents) for q in batch]
        batch_results = await asyncio.gather(*batch_tasks)
        for q, result in zip(batch, batch_results):
            all_results.append((question_order[q], result))
    
    # Process complex questions in batches of 2
    complex_batch_size = 10
    for i in range(0, len(complex_questions), complex_batch_size):
        batch = complex_questions[i:i + complex_batch_size]
        batch_tasks = [get_answer_async(vectorstore, q, documents) for q in batch]
        batch_results = await asyncio.gather(*batch_tasks)
        for q, result in zip(batch, batch_results):
            all_results.append((question_order[q], result))
    
    # Sort by original order
    all_results.sort(key=lambda x: x[0])
    return [result for _, result in all_results]

# --- Cache Management Endpoints ---
@router.get("/cache/status")
async def get_cache_status():
    """Get cache status and statistics"""
    try:
        cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.pkl') and 'metadata_' in f]
        cached_documents = []
        
        for metadata_file in cache_files:
            try:
                with open(os.path.join(CACHE_DIR, metadata_file), 'rb') as f:
                    metadata = pickle.load(f)
                    cached_documents.append({
                        'cache_name': metadata.get('cache_name', 'unknown'),
                        'url': metadata.get('url', 'unknown')[:100] + '...',
                        'timestamp': time.ctime(metadata.get('timestamp', 0)),
                        'num_chunks': metadata.get('num_documents', 0),
                        'text_length': metadata.get('text_length', 0)
                    })
            except:
                continue
        
        return {
            "total_cached_documents": len(cached_documents),
            "redis_enabled": CACHE_ENABLED,
            "ml_features_enabled": ML_AVAILABLE,
            "cache_directory": CACHE_DIR,
            "cached_documents": cached_documents
        }
    except Exception as e:
        return {"error": str(e)}

@router.delete("/cache/clear")
async def clear_cache():
    """Clear all cached documents (use with caution!)"""
    try:
        import shutil
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
            os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Clear Redis cache if available
        if CACHE_ENABLED and redis_client:
            redis_client.flushdb()
        
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        return {"error": str(e)}

# --- Optimized Main Endpoint ---
@router.post("/run", dependencies=[Depends(verify_bearer_token)])
async def process_query(request: QueryRequest):
    start_time = time.time()
    
    print("Received a new query request:")
    print(f"Documents link: {request.documents}")
    print(f"Questions asked: {request.questions}")
    print("---------------------------------------")

    try:
        # Main processing pipeline with caching
        vectorstore, documents, metadata = await get_or_create_document_pipeline(request.documents)
        
        # Process questions with batching
        if len(request.questions) == 1:
            # Single question - process directly
            answers = [await get_answer_async(vectorstore, request.questions[0], documents)]
        else:
            # Multiple questions - use batching
            answers = await batch_process_questions(vectorstore, request.questions, documents)
        
        # Log performance
        total_time = time.time() - start_time
        cache_status = "cached" if metadata.get('timestamp', 0) < start_time - 1 else "processed"
        
        logger.info(f"🎯 Query completed in {total_time:.2f}s ({cache_status})")
        logger.info(f"   - Document: {metadata.get('cache_name', 'unknown')}")
        logger.info(f"   - Questions: {len(request.questions)}")
        logger.info(f"   - Chunks: {len(documents)}")
        
        return {"answers": answers}
    
    except aiohttp.ClientError as e:
        logger.error(f"PDF download failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during processing")

# --- Register API Router ---
app.include_router(router, prefix="/api/v1/hackrx")

# --- Startup/Shutdown Events ---
@app.on_event("startup")
async def startup_event():
    print("RAG API starting up...")
    print(f"Thread pool initialized with {executor._max_workers} workers")
    print(f"Redis cache: {'enabled' if CACHE_ENABLED else 'disabled'}")
    print(f"ML features: {'enabled' if ML_AVAILABLE else 'disabled'}")
    print(f"Document cache directory: {CACHE_DIR}")
    
    # Show cached documents on startup
    try:
        cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.pkl') and 'metadata_' in f]
        if cache_files:
            print(f"📚 Found {len(cache_files)} cached documents:")
            for metadata_file in cache_files[:5]:  # Show first 5
                try:
                    with open(os.path.join(CACHE_DIR, metadata_file), 'rb') as f:
                        metadata = pickle.load(f)
                        print(f"   - {metadata.get('cache_name', 'unknown')} ({metadata.get('num_documents', 0)} chunks)")
                except:
                    continue
            if len(cache_files) > 5:
                print(f"   ... and {len(cache_files) - 5} more")
        else:
            print("📚 No cached documents found")
    except Exception as e:
        print(f"⚠️ Error checking cache: {str(e)}")

@app.on_event("shutdown") 
async def shutdown_event():
    print("RAG API shutting down...")
    executor.shutdown(wait=True)
    if CACHE_ENABLED and redis_client:
        redis_client.close()

# --- Health Check ---
@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "HackRx RAG API",
        "redis_cache": "enabled" if CACHE_ENABLED else "disabled",
        "ml_features": "enabled" if ML_AVAILABLE else "disabled",
        "cache_directory": CACHE_DIR,
        "optimizations": "all_applied_with_document_caching"
    }
