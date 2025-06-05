"""
Q&A Agent with domain-specific web search capabilities
"""

import logging
import pandas as pd
from typing import List, Dict, Any, Optional

from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory

from search_tool import TavilyDomainSearchTool
from scraping_tool import WebScrapingTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_sites_data(csv_file_path: str) -> pd.DataFrame:
    """Load and validate sites data from CSV"""
    df = pd.read_csv(csv_file_path)
    required_columns = ["site", "domain", "description"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df


def create_llm(config: Dict[str, Any]) -> ChatGoogleGenerativeAI:
    """Initialize Gemini model with configuration"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=config["google_api_key"],
        temperature=config["llm_temperature"],
        max_tokens=config["llm_max_tokens"],
        timeout=config["llm_timeout"],
    )


def create_search_tool(config: Dict[str, Any]) -> TavilyDomainSearchTool:
    """Create configured search tool"""
    return TavilyDomainSearchTool(
        api_key=config["tavily_api_key"],
        max_results=config["max_results"],
        depth=config["search_depth"],
        max_content_size=config["max_content_size"],
        enable_summarization=config.get("enable_search_summarization", False),
        google_api_key=(
            config.get("google_api_key")
            if config.get("enable_search_summarization", False)
            else None
        ),
    )


def create_scraping_tool(config: Dict[str, Any]) -> WebScrapingTool:
    """Create configured web scraping tool"""
    return WebScrapingTool(max_content_length=config["max_scrape_length"])


def build_knowledge_sources_text(sites_df: pd.DataFrame) -> tuple[str, List[str]]:
    """Build formatted knowledge sources text and domain list"""
    domain_groups = {}
    domains = []

    for _, row in sites_df.iterrows():
        domain = row["domain"]
        if domain not in domain_groups:
            domains.append(domain)
            domain_groups[domain] = []
        domain_groups[domain].append(
            {"site": row["site"], "description": row["description"]}
        )

    knowledge_sources_md = ""
    for domain, sources in domain_groups.items():
        knowledge_sources_md += f"\n## {domain}\n\n"
        for source in sources:
            knowledge_sources_md += f"- {source['site']}: {source['description']}\n"
        knowledge_sources_md += "\n"

    return knowledge_sources_md, domains


def create_system_prompt(knowledge_sources_md: str, domains: List[str]) -> str:
    """Create the system prompt with knowledge sources"""
    return f"""You are a specialized Q&A agent that searches specific documentation websites.

AVAILABLE KNOWLEDGE SOURCES split by category/domain/topic having the website and description for each category:
{knowledge_sources_md}

INSTRUCTIONS:
1. ALWAYS start with the search_documentation tool for ANY question
2. Analyze the user's question to determine relevant domains/topics/categories
3. Select appropriate sites based on technologies/topics mentioned
4. If search results don't provide sufficient information to answer the question completely, then use scrape_website tool on the most relevant URL from search results
5. You must only answer questions about available knowledge sources: {domains}
6. If question is outside available knowledge sources, do not answer the question and suggest which topics you can answer

TOOL USAGE STRATEGY:
- First: Use search_documentation to find relevant information quickly
- Second: If search results are incomplete or unclear, use scrape_website on the most promising URL from search results
- Always prefer search over scraping for efficiency

RULES:
- Be helpful and comprehensive
- Cite sources when possible
- Only use scraping when absolutely necessary (when search results are insufficient)
- When scraping, choose the most relevant URL from previous search results

You have access to the following tools:

{{tools}}

Use JSON format for actions:
```
{{{{
  "action": "$TOOL_NAME",
  "action_input": "$INPUT"
}}}}
```

Valid actions: "Final Answer" or {{tool_names}}

Format:
Question: input question
Thought: analyze and plan
Action: ```$JSON_BLOB```
Observation: action result
... (repeat as needed)
Thought: ready to respond
Action: 
```
{{{{
  "action": "Final Answer",
  "action_input": "response"
}}}}
```
"""


class DomainQAAgent:
    """Q&A Agent that searches specific domains based on user queries"""

    def __init__(
        self,
        csv_file_path: str = "sites_data.csv",
        config: Optional[Dict[str, Any]] = None,
    ):
        if config is None:
            raise ValueError("Configuration is required")

        self.config = config
        self.sites_df = load_sites_data(csv_file_path)
        self.llm = create_llm(config)
        self.search_tool = create_search_tool(config)
        self.scraping_tool = create_scraping_tool(config)
        self.chat_history: List[BaseMessage] = []
        self.agent_executor = self._create_agent()

        logger.info(f"Agent initialized with {len(self.sites_df)} sites")

    def _create_agent(self) -> AgentExecutor:
        """Create structured chat agent with tools and prompt"""
        knowledge_sources_md, domains = build_knowledge_sources_text(self.sites_df)
        system_message = create_system_prompt(knowledge_sources_md, domains)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{input}\n\n{agent_scratchpad}"),
            ]
        )

        agent = create_structured_chat_agent(
            llm=self.llm, tools=[self.search_tool, self.scraping_tool], prompt=prompt
        )

        return AgentExecutor(
            agent=agent,
            tools=[self.search_tool, self.scraping_tool],
            verbose=True,
            max_iterations=5,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
        )

    async def achat(self, user_input: str) -> str:
        """Process user input asynchronously"""
        try:
            logger.info(f"Processing: {user_input}")

            agent_input = {"input": user_input, "chat_history": self.chat_history}
            response = await self.agent_executor.ainvoke(agent_input)
            answer = response.get("output", "I couldn't process your request.")

            self.chat_history.extend(
                [HumanMessage(content=user_input), AIMessage(content=answer)]
            )

            return answer

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def reset_memory(self):
        """Reset conversation memory"""
        self.chat_history.clear()
        logger.info("Memory reset")
