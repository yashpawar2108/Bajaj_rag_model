from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from app.utils import extract_text_from_pdf, split_text_into_chunks
from app.embeddings_store import create_vectorstore
from app.llm_query import get_answer

router = APIRouter()

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
    questions: list[str]

# --- Endpoint ---
@router.post("/run", dependencies=[Depends(verify_bearer_token)])
async def process_query(request: QueryRequest):
    # Log the incoming request details
    print("Received a new query request:")
    print(f"Documents link: {request.documents}")
    print(f"Questions asked: {request.questions}")
    print("---------------------------------------")

    text = extract_text_from_pdf(request.documents)
    chunks = split_text_into_chunks(text, chunk_size=1000, overlap=100)
    vectorstore = create_vectorstore(chunks)
    answers = [get_answer(vectorstore, q) for q in request.questions]
    return {"answers": answers}
