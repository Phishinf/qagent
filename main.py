"""
FastAPI application for Domain-specific Q&A Agent
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

from qa_agent import DomainQAAgent

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_int_env(key: str, default: int) -> int:
    """Parse integer from environment variable with fallback"""
    try:
        return int(os.getenv(key, default))
    except ValueError:
        logger.warning(f"Invalid {key}, using default: {default}")
        return default


def get_float_env(key: str, default: float) -> float:
    """Parse float from environment variable with fallback"""
    try:
        return float(os.getenv(key, default))
    except ValueError:
        logger.warning(f"Invalid {key}, using default: {default}")
        return default


def validate_api_keys():
    """Validate required API keys are present"""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")

    if not google_api_key or google_api_key == "your_google_api_key_here":
        raise ValueError("GOOGLE_API_KEY environment variable is required")

    if not tavily_api_key or tavily_api_key == "your_tavily_api_key_here":
        raise ValueError("TAVILY_API_KEY environment variable is required")

    return google_api_key, tavily_api_key


def build_config() -> Dict[str, Any]:
    """Build configuration from environment variables"""
    google_api_key, tavily_api_key = validate_api_keys()

    search_depth = os.getenv("SEARCH_DEPTH", "basic")
    if search_depth not in ["basic", "advanced"]:
        logger.warning(f"Invalid SEARCH_DEPTH '{search_depth}', using default: basic")
        search_depth = "basic"

    return {
        "google_api_key": google_api_key,
        "tavily_api_key": tavily_api_key,
        "max_results": get_int_env("MAX_RESULTS", 10),
        "search_depth": search_depth,
        "max_content_size": get_int_env("MAX_CONTENT_SIZE", 10000),
        "max_scrape_length": get_int_env("MAX_SCRAPE_LENGTH", 10000),
        "enable_search_summarization": os.getenv(
            "ENABLE_SEARCH_SUMMARIZATION", "false"
        ).lower()
        == "true",
        "llm_temperature": get_float_env("LLM_TEMPERATURE", 0.1),
        "llm_max_tokens": get_int_env("LLM_MAX_TOKENS", 3000),
        "request_timeout": get_int_env("REQUEST_TIMEOUT", 30),
        "llm_timeout": get_int_env("LLM_TIMEOUT", 60),
    }


def log_config(config: Dict[str, Any]):
    """Pretty print configuration (excluding API keys)"""
    safe_config = {k: v for k, v in config.items() if not k.endswith("_api_key")}
    logger.info("Configuration loaded:")
    for key, value in safe_config.items():
        logger.info(f"  {key}: {value}")


def create_config() -> Dict[str, Any]:
    """Create and validate complete configuration"""
    try:
        config = build_config()
        log_config(config)
        logger.info("Environment validation completed")
        return config
    except Exception as e:
        logger.error(f"Environment validation failed: {str(e)}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup Q&A agent"""
    try:
        logger.info("Initializing Q&A Agent...")
        config = create_config()
        app.state.qa_agent = DomainQAAgent(config=config)
        logger.info("Q&A Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Q&A Agent: {str(e)}")
        raise

    yield

    logger.info("Shutting down Q&A Agent...")


app = FastAPI(
    title="Domain Q&A Agent API",
    description="A Q&A agent that searches specific domains using Tavily and Langchain",
    version="1.0.0",
    lifespan=lifespan,
)


def get_qa_agent() -> DomainQAAgent:
    """Get the initialized QA agent instance"""
    if not hasattr(app.state, "qa_agent") or app.state.qa_agent is None:
        raise HTTPException(status_code=500, detail="Q&A Agent not initialized")
    return app.state.qa_agent


class ChatRequest(BaseModel):
    message: str
    reset_memory: bool = False


class ChatResponse(BaseModel):
    response: str
    status: str = "success"


@app.get("/health")
async def health_check(qa_agent: DomainQAAgent = Depends(get_qa_agent)):
    """Health check with agent status"""
    return {
        "message": "Domain Q&A Agent API is running",
        "status": "healthy",
        "version": "1.0.0",
        "agent_status": "initialized",
    }


@app.post("/chat", response_model=ChatResponse, summary="Chat with Q&A Agent")
async def chat(request: ChatRequest, qa_agent: DomainQAAgent = Depends(get_qa_agent)):
    """Process user questions through the Q&A agent"""
    logger.info(f"Received chat request: {request.message}")

    if request.reset_memory:
        qa_agent.reset_memory()
        logger.info("Memory reset requested")

    response = await qa_agent.achat(request.message)
    logger.info("Successfully processed chat request")

    return ChatResponse(response=response)


@app.post("/reset", summary="Reset conversation memory")
async def reset_memory(qa_agent: DomainQAAgent = Depends(get_qa_agent)):
    """Reset conversation memory"""
    qa_agent.reset_memory()
    logger.info("Memory reset via endpoint")
    return {"message": "Conversation memory has been reset", "status": "success"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
