"""
FastAPI application for Domain-specific Q&A Agent
Educational example showing how to wrap a LangChain agent in a REST API
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

from qa_agent import DomainQAAgent

# Load environment variables from .env file
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_environment() -> Dict[str, Any]:
    """Validate and load environment variables with defaults"""
    # Check required API keys
    google_api_key = os.getenv("GOOGLE_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")

    if not google_api_key or google_api_key == "your_google_api_key_here":
        raise ValueError("GOOGLE_API_KEY environment variable is required")

    if not tavily_api_key or tavily_api_key == "your_tavily_api_key_here":
        raise ValueError("TAVILY_API_KEY environment variable is required")

    # Helper functions for safe environment variable parsing
    def get_int_env(key: str, default: int) -> int:
        try:
            return int(os.getenv(key, default))
        except ValueError:
            logger.warning(f"Invalid {key}, using default: {default}")
            return default

    def get_float_env(key: str, default: float) -> float:
        try:
            return float(os.getenv(key, default))
        except ValueError:
            logger.warning(f"Invalid {key}, using default: {default}")
            return default

    # Validate search depth parameter
    search_depth = os.getenv("SEARCH_DEPTH", "basic")
    if search_depth not in ["basic", "advanced"]:
        logger.warning(f"Invalid SEARCH_DEPTH '{search_depth}', using default: basic")
        search_depth = "basic"

    # Build configuration dictionary
    config = {
        "google_api_key": google_api_key,
        "tavily_api_key": tavily_api_key,
        "max_results": get_int_env("MAX_RESULTS", 10),
        "search_depth": search_depth,
        "max_content_size": get_int_env("MAX_CONTENT_SIZE", 10000),
        "llm_temperature": get_float_env("LLM_TEMPERATURE", 0.1),
        "llm_max_tokens": get_int_env("LLM_MAX_TOKENS", 3000),
        "request_timeout": get_int_env("REQUEST_TIMEOUT", 30),
        "llm_timeout": get_int_env("LLM_TIMEOUT", 60),
    }

    logger.info("Environment validation completed")
    return config


# Validate environment on startup
try:
    ENV_CONFIG = validate_environment()
except Exception as e:
    logger.error(f"Environment validation failed: {str(e)}")
    raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager - handles startup and shutdown"""
    global qa_agent
    try:
        # Startup: Initialize the Q&A agent
        logger.info("Initializing Q&A Agent...")
        qa_agent = DomainQAAgent(config=ENV_CONFIG)
        logger.info("Q&A Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Q&A Agent: {str(e)}")
        raise

    yield  # Application runs here

    # Shutdown: Cleanup if needed
    logger.info("Shutting down Q&A Agent...")


# Create FastAPI application with lifespan management
app = FastAPI(
    title="Domain Q&A Agent API",
    description="A Q&A agent that searches specific domains using Tavily and Langchain",
    version="1.0.0",
    lifespan=lifespan,
)

# Global variable to store the agent instance
qa_agent = None


# Pydantic models for request/response validation
class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str
    reset_memory: bool = False


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str
    status: str = "success"


class DomainsResponse(BaseModel):
    """Response model for domains endpoint"""
    domains: List[Dict[str, str]]
    count: int


# API Endpoints
@app.get("/", summary="Health Check")
async def root():
    """Basic health check endpoint"""
    return {
        "message": "Domain Q&A Agent API is running",
        "status": "healthy",
        "version": "1.0.0",
    }


@app.post("/chat", response_model=ChatResponse, summary="Chat with Q&A Agent")
async def chat_with_agent(request: ChatRequest):
    """Main chat endpoint - processes user questions"""
    if qa_agent is None:
        raise HTTPException(status_code=500, detail="Q&A Agent not initialized")

    logger.info(f"Received chat request: {request.message}")

    # Reset memory if requested
    if request.reset_memory:
        qa_agent.reset_memory()
        logger.info("Memory reset requested")

    # Process the question through the agent
    response = await qa_agent.achat(request.message)
    logger.info("Successfully processed chat request")

    return ChatResponse(response=response, status="success")


@app.post("/reset", summary="Reset Conversation Memory")
async def reset_memory():
    """Reset the conversation memory of the agent"""
    if qa_agent is None:
        raise HTTPException(status_code=500, detail="Q&A Agent not initialized")

    qa_agent.reset_memory()
    logger.info("Memory reset via endpoint")

    return {"message": "Conversation memory has been reset", "status": "success"}


@app.get("/health", summary="Detailed Health Check")
async def health_check():
    """Detailed health check with system status"""
    try:
        agent_status = "initialized" if qa_agent is not None else "not_initialized"

        return {
            "status": "healthy",
            "agent_status": agent_status,
            "version": "1.0.0",
            "config": {
                "max_results": ENV_CONFIG["max_results"],
                "search_depth": ENV_CONFIG["search_depth"],
                "llm_temperature": ENV_CONFIG["llm_temperature"],
                "llm_max_tokens": ENV_CONFIG["llm_max_tokens"],
            },
        }

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e), "version": "1.0.0"}


# Run the application
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
