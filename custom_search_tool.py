"""
Custom Tavily search tool for domain-specific web search
"""

import logging
from typing import List, Optional, Type, Any
from pydantic import BaseModel, Field, ConfigDict
from langchain.tools import BaseTool
from tavily import TavilyClient
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TavilySearchInput(BaseModel):
    """Input schema for the Tavily search tool"""

    input: str = Field(
        description="The enriched search query string. This should be a well-crafted search query "
        "that incorporates relevant keywords from the user's question to maximize search "
        "effectiveness. Include technical terms, framework names, and specific concepts "
        "mentioned in the user's query."
    )
    sites: List[str] = Field(
        description="List of specific website domains to search within (e.g., 'docs.langchain.com', "
        "'fastapi.tiangolo.com'). These should be selected based on the user's query context and "
        "the available documentation sites. Choose sites that are most likely to contain relevant "
        "documentation for the user's question. Examples: ['docs.python.org'], "
        "['fastapi.tiangolo.com', 'docs.python.org']"
    )
    max_results: Optional[int] = Field(
        default=None,
        description="Maximum number of search results to return. If not specified, uses the "
        "configured default value from environment variables.",
    )
    depth: Optional[str] = Field(
        default=None,
        description="Search depth level - 'basic' for quick results or 'advanced' for more "
        "comprehensive search. If not specified, uses the configured default value.",
    )


class TavilyDomainSearchTool(BaseTool):
    """Custom tool for searching specific domains using Tavily"""

    name: str = "tavily_domain_search"
    description: str = """
    Search for information within specific documentation websites using Tavily web search.

    This tool is designed to find relevant documentation and technical information by:
    1. Taking an enriched search query that you should craft from the user's question
    2. Searching only within the specified website domains (e.g., docs.langchain.com, fastapi.tiangolo.com)
    3. Returning formatted results with titles, URLs, and content snippets

    Key Usage Guidelines:
    - INPUT: Create a detailed, keyword-rich search query from the user's question. Include specific
      technical terms, framework names, and concepts to get the best results.
    - SITES: Analyze the user's question to determine which documentation websites are most relevant.
      Select appropriate website domains from the available options based on the technologies mentioned.
      For example, if the user asks about LangChain, select 'docs.langchain.com'.
    - Always prefer official documentation websites for technical queries.
    - Use multiple sites if the question spans multiple technologies or frameworks.

    The tool will search only within the specified website domains and return comprehensive results
    from the selected documentation sources.
    """
    args_schema: Type[BaseModel] = TavilySearchInput

    # Define fields properly for Pydantic v2
    tavily_client: Any = Field(default=None, exclude=True)
    api_key: str = Field(exclude=True)
    default_max_results: int = Field(default=10, exclude=True)
    default_depth: str = Field(default="basic", exclude=True)
    max_content_size: int = Field(default=10000, exclude=True)

    # Allow arbitrary types for the TavilyClient
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        api_key: str,
        max_results: int = 10,
        depth: str = "basic",
        max_content_size: int = 10000,
    ):
        super().__init__(
            api_key=api_key,
            default_max_results=max_results,
            default_depth=depth,
            max_content_size=max_content_size,
        )

        # Initialize Tavily client
        if not api_key:
            raise ValueError("TAVILY_API_KEY is required either as parameter or environment variable")

        # Set the tavily_client using object.__setattr__ to bypass Pydantic
        # validation
        object.__setattr__(self, "tavily_client", TavilyClient(api_key=api_key))
        logger.info("Tavily Domain Search Tool initialized")

    def _run(
        self, input: str, sites: List[str], max_results: int = None, depth: str = None
    ) -> str:
        """Execute the search with the given parameters"""
        try:
            # Use provided values or fall back to defaults
            final_max_results = max_results or self.default_max_results
            final_depth = depth or self.default_depth

            logger.info(
                f"🔍 Starting Tavily search with query: '{input}' on sites: {sites}"
            )
            logger.info(
                f"📊 Search parameters: max_results={final_max_results}, depth={final_depth}"
            )

            # Perform the search using Tavily
            search_results = self.tavily_client.search(
                query=input,
                max_results=final_max_results,
                search_depth=final_depth,
                include_domains=sites,
            )

            logger.info(
                f"📥 Raw search results received: {len(search_results.get('results', []))} results"
            )

            # Format the results
            if not search_results.get("results"):
                result = "No results found for the given query and sites. Please try a different search query or check if the domains are accessible."
                logger.warning("⚠️ No search results returned")
                return result

            formatted_results = []
            for i, result in enumerate(
                search_results["results"][:final_max_results], 1
            ):
                title = result.get("title", "No title")
                url = result.get("url", "No URL")
                content = result.get("content", "No content available")

                logger.info(f"📄 Processing result {i}: {title[:50]}...")

                # Truncate content based on max_content_size
                if len(content) > self.max_content_size:
                    content = content[: self.max_content_size] + "..."

                formatted_result = f"""
                    Result {i}:
                    Title: {title}
                    URL: {url}
                    Content: {content}
                    ---
                    """
                formatted_results.append(formatted_result)

            final_result = "\n".join(formatted_results)
            logger.info(
                f"✅ Successfully processed {
                    len(
                        search_results['results'])} results, returning {
                    len(final_result)} characters"
            )

            return final_result

        except Exception as e:
            error_msg = f"❌ Error performing Tavily search: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Exception details: {type(e).__name__}: {e}")
            return error_msg

    async def _arun(
        self, input: str, sites: List[str], max_results: int = None, depth: str = None
    ) -> str:
        """Async version of the search"""
        # For now, we'll use the sync version since TavilyClient doesn't have async methods
        # In a production environment, you might want to use aiohttp or similar
        # for async requests
        return self._run(input, sites, max_results, depth)
