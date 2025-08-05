from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import os

def get_answer(vectorstore, question: str):
    llm = ChatOpenAI(
        model="llama3-70b-8192",
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    # Define a custom prompt template
    template = """
    Instructions for your Answer:
    1.Synthesize, Don't Just Extract: Do not simply copy-paste sentences or phrases from the context. You must process the relevant information and formulate a response in your own words to create a new, coherent, and logical answer.
    2.Be Clear and Accessible: Translate any technical jargon, specialized terminology, or complex concepts from the document's domain into plain, understandable language. The goal is to make the information accessible to a non-expert.
    3.Provide Complete and Relevant Information: Your answer must be a single, well-formed paragraph. It should directly address the user's question and include all essential details, conditions, or key data points required for a comprehensive understanding.
    4.Use Natural Language: Ensure the answer is grammatically correct and flows naturally as a human-written response would. Start the answer directly, without any introductory phrases like "According to the context..." or "The document states...".
    5.Maintain a Helpful Tone: If the question can be logically answered with a "Yes" or "No," you may start with that, but you must always follow it with the complete explanation derived from the context.
    The context from the policy document is as follows:
    {context}

    Question: {question}
    Answer:
    """
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    # Configure the RetrievalQA chain with the new prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )
    
    result = qa_chain.invoke({"query": question})
    return result["result"]
