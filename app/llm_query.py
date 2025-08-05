from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os

def get_answer(vectorstore, question: str):
    llm = ChatOpenAI(
        model="llama3-70b-8192",
        base_url="https://api.groq.com/openai/v1",  # Groq-compatible base URL
        api_key=os.getenv("GROQ_API_KEY")  # Set this in your environment
    )
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    result = qa.run(question)
    return result
