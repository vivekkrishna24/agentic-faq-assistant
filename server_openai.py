import os
import logging
import time
from datetime import datetime
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnablePassthrough
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'api_openai_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

API_KEY = os.getenv("X_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class RAGService:
    def __init__(
        self,
        collection_name: str = "rag_faq",
        persist_directory: str = "rag_faq",
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.1,
        embedding_model: str = "text-embedding-3-small",
        k_retrieval: int = 2
    ):
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=OPENAI_API_KEY
        )
        
        # Initialize OpenAI LLM
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=OPENAI_API_KEY
        )
        
        self.parser = StrOutputParser()
        
        # Initialize Chroma vector store
        self.chroma_client = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        
        self.retriever = self.chroma_client.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k_retrieval}
        )
        
        # Set up the QA chain
        self.qa_chain = self._setup_qa_chain()
    
    def _setup_qa_chain(self):
        """Set up the QA chain with a custom prompt."""
        template = """You are a helpful AI assistant that answers questions based on the provided context.
        Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Question: {question}
        
        Helpful Answer:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        return (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | self.parser
        )
    
    def query(self, question: str) -> dict:
        """Query the RAG system with a question."""
        start_time = time.time()
        
        try:
            # Get the answer from the QA chain
            answer = self.qa_chain.invoke(question)
            
            # Get relevant documents
            docs = self.retriever.get_relevant_documents(question)
            
            end_time = time.time()
            
            return {
                "answer": answer,
                "sources": [doc.metadata.get('source', 'Unknown') for doc in docs],
                "metrics": {
                    "retrieval_time": end_time - start_time,
                    "total_tokens": len(question) + len(answer)  # Simple token estimation
                }
            }
            
        except Exception as e:
            logger.error(f"Error in RAG query: {str(e)}", exc_info=True)
            raise

# Initialize FastAPI app
app = FastAPI(title="RAG FAQ API with OpenAI")

# Initialize RAG service
rag_service = RAGService()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    metrics: dict

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest, req: Request):
    """
    Endpoint to query the RAG system with OpenAI.
    
    Args:
        request (QueryRequest): The query request containing the question
        req (Request): FastAPI request object for logging
        
    Returns:
        QueryResponse: The response containing the answer and metrics
    """
    logger.info(f"Received query: {request.question}")
    
    try:
        start_time = time.time()
        
        # Process the query
        result = rag_service.query(request.question)
        
        # Log the response time
        response_time = time.time() - start_time
        logger.info(f"Query processed in {response_time:.2f} seconds")
        
        return QueryResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            metrics={
                "response_time": response_time,
                **result.get("metrics", {})
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Simple health check - verify we can access the vector store
        rag_service.chroma_client._collection.count()
        return {
            "status": "healthy",
            "model": rag_service.llm.model_name,
            "embedding_model": rag_service.embeddings.model
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"status": "unhealthy", "error": str(e)}
        ) 

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
