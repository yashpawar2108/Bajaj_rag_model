from fastapi import FastAPI
from app.api import router as api_router

app = FastAPI(title="HackRx RAG API")

app.include_router(api_router, prefix="/hackrx")