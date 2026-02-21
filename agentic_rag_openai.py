import os
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'agent_api_openai_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class AgenticRAGService:
    def __init__(
        self,
        collection_name: str = "rag_faq",
        persist_directory: str = "rag_faq",
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.1,
        embedding_model: str = "text-embedding-3-small",
        k_retrieval: int = 4,
    ) -> None:
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

        # Chroma vector store and retriever
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k_retrieval}
        )

        # Set up the tools
        self.tools = self._setup_tools()
        
        # Initialize the agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True
        )

    def _setup_tools(self) -> List[Tool]:
        """Set up the tools for the agent."""
        def get_relevant_documents(query: str) -> List[Dict[str, Any]]:
            """Retrieve relevant documents for a query."""
            docs = self.retriever.get_relevant_documents(query)
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in docs
            ]

        tools = [
            Tool(
                name="search_knowledge_base",
                func=get_relevant_documents,
                description="Useful for searching the knowledge base for relevant information."
            )
        ]
        
        return tools

    def query(self, question: str) -> Dict[str, Any]:
        """Query the agent with a question."""
        start_time = time.time()
        
        try:
            # Get the response from the agent
            response = self.agent.invoke({"input": question})
            
            end_time = time.time()
            
            return {
                "answer": response.get("output", "No answer generated."),
                "sources": [],
                "intermediate_steps": [],
                "metrics": {
                    "response_time": end_time - start_time,
                }
            }
            
        except Exception as e:
            logger.error(f"Error in agent query: {str(e)}", exc_info=True)
            raise

# Initialize FastAPI app
app = FastAPI(title="Agentic RAG API with OpenAI")

# Initialize RAG service
rag_service = AgenticRAGService()

class AgentQueryRequest(BaseModel):
    question: str

class AgentQueryResponse(BaseModel):
    answer: str
    sources: List[str]
    intermediate_steps: List[Dict[str, Any]]
    metrics: Dict[str, Any]

@app.post("/agent/query", response_model=AgentQueryResponse)
async def agent_query_endpoint(request: AgentQueryRequest):
    """Endpoint to query the agentic RAG system."""
    logger.info(f"Received agent query: {request.question}")
    
    try:
        result = rag_service.query(request.question)
        return AgentQueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Error processing agent query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Simple health check
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
    uvicorn.run(app, host="0.0.0.0", port=8001)
