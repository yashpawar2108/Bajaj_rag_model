from fastapi import APIRouter
from pydantic import BaseModel
from app.utils import extract_text_from_pdf, split_text_into_chunks
from app.embeddings_store import create_vectorstore
from app.llm_query import get_answer

router = APIRouter()

class QueryRequest(BaseModel):
    documents: str
    questions: list[str]

@router.post("/run")
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
