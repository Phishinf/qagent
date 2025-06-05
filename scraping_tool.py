"""
Web scraping tool using Chromium for dynamic content extraction
"""

import logging
import asyncio
from typing import List, Type

from pydantic import BaseModel, Field, ConfigDict
from langchain.tools import BaseTool
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_default_tags() -> List[str]:
    """Get default HTML tags for web scraping"""
    return ["p", "li", "div", "a", "span", "h1", "h2", "h3", "h4", "h5", "h6"]


class WebScrapingInput(BaseModel):
    url: str = Field(description="URL to scrape")
    tags_to_extract: List[str] = Field(
        default_factory=get_default_tags, description="HTML tags to extract"
    )


class WebScrapingTool(BaseTool):
    """Scrape websites when search results are insufficient"""

    name: str = "scrape_website"
    description: str = """Scrape complete website content using Chromium.

    Use only when search results don't provide adequate information.
    
    Features:
    - Headless Chromium for JavaScript rendering
    - Extracts content from multiple HTML tags
    - Handles modern dynamic websites
    
    Warning: Slower than search. Use sparingly.
    """
    args_schema: Type[BaseModel] = WebScrapingInput

    max_content_length: int = Field(default=10000, exclude=True)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, max_content_length: int = 10000):
        super().__init__(max_content_length=max_content_length)

    def _run(self, url: str, tags_to_extract: List[str] = None) -> str:
        """Scrape website content"""
        try:
            if tags_to_extract is None:
                tags_to_extract = get_default_tags()

            loader = AsyncChromiumLoader([url])
            html_docs = loader.load()

            if not html_docs:
                return f"Failed to load content from {url}"

            bs_transformer = BeautifulSoupTransformer()
            docs_transformed = bs_transformer.transform_documents(
                html_docs, tags_to_extract=tags_to_extract
            )

            if not docs_transformed:
                return f"No content extracted from {url}"

            content = docs_transformed[0].page_content

            if len(content) > self.max_content_length:
                content = (
                    content[: self.max_content_length] + "\n\n... (content truncated)"
                )

            return f"""
**Website Scraped:** {url}
**Content Extracted:**

{content}

**Note:** Complete website content for comprehensive analysis.
"""

        except Exception as e:
            return f"Web scraping error for {url}: {str(e)}"

    async def _arun(self, url: str, tags_to_extract: List[str] = None) -> str:
        """Async version of scraping"""
        try:
            if tags_to_extract is None:
                tags_to_extract = get_default_tags()

            loader = AsyncChromiumLoader([url])
            html_docs = await asyncio.to_thread(loader.load)

            if not html_docs:
                return f"Failed to load content from {url}"

            bs_transformer = BeautifulSoupTransformer()
            docs_transformed = await asyncio.to_thread(
                bs_transformer.transform_documents,
                html_docs,
                tags_to_extract=tags_to_extract,
            )

            if not docs_transformed:
                return f"No content extracted from {url}"

            content = docs_transformed[0].page_content

            if len(content) > self.max_content_length:
                content = (
                    content[: self.max_content_length] + "\n\n... (content truncated)"
                )

            return f"""
**Website Scraped:** {url}
**Content Extracted:**

{content}

**Note:** Complete website content for comprehensive analysis.
"""

        except Exception as e:
            return f"Web scraping error for {url}: {str(e)}" 