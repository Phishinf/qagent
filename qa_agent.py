"""
Q&A Agent with domain-specific web search capabilities
Educational example for building structured chat agents with LangChain
"""

import logging
import pandas as pd
from typing import List, Dict, Any, Optional

from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import BaseMessage, HumanMessage, AIMessage

from custom_search_tool import TavilyDomainSearchTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        
        # Load domain data from CSV file
        self.sites_df = self._load_sites_data(csv_file_path)
        
        # Initialize the LLM (Google Gemini)
        self.llm = self._initialize_llm()
        
        # Create custom search tool with domain restrictions
        self.search_tool = TavilyDomainSearchTool(
            api_key=config["tavily_api_key"],
            max_results=config["max_results"],
            depth=config["search_depth"],
            max_content_size=config["max_content_size"],
        )
        
        # Store conversation history for context
        self.chat_history: List[BaseMessage] = []
        
        # Create the structured chat agent
        self.agent_executor = self._create_agent()

        logger.info(f"Agent initialized with {len(self.sites_df)} sites")

    def _load_sites_data(self, csv_file_path: str) -> pd.DataFrame:
        """Load sites data from CSV"""
        df = pd.read_csv(csv_file_path)
        required_columns = ["site", "domain", "description"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        return df

    def _initialize_llm(self) -> ChatGoogleGenerativeAI:
        """Initialize Gemini model with configuration"""
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=self.config["google_api_key"],
            temperature=self.config["llm_temperature"],
            max_tokens=self.config["llm_max_tokens"],
            timeout=self.config["llm_timeout"],
        )

    def _create_agent(self) -> AgentExecutor:
        """Create structured chat agent with tools and prompt"""
        # Create the prompt template
        prompt = self._create_structured_chat_prompt()
        
        # Create the agent with LLM, tools, and prompt
        agent = create_structured_chat_agent(
            llm=self.llm, tools=[self.search_tool], prompt=prompt
        )

        # Wrap agent in executor for execution control
        return AgentExecutor(
            agent=agent,
            tools=[self.search_tool],
            verbose=True,
            max_iterations=5,
            return_intermediate_steps=True,
            handle_parsing_errors="Check your output format!",
        )

    def _create_structured_chat_prompt(self) -> ChatPromptTemplate:
        """Create the structured chat prompt template"""
        # Group sources by domain for organized display
        domain_groups = {}
        domains = []
        for _, row in self.sites_df.iterrows():
            domain = row['domain']
            if domain not in domain_groups:
                domains.append(domain)
                domain_groups[domain] = []
            
            domain_groups[domain].append({
                'site': row['site'],
                'description': row['description']
            })

        # Create markdown formatted knowledge sources
        knowledge_sources_md = ""
        for domain, sources in domain_groups.items():
            knowledge_sources_md += f"\n## {domain}\n\n"
            for source in sources:
                knowledge_sources_md += f"- {source['site']}: {source['description']}\n"
            knowledge_sources_md += "\n"

        # System message template
        system_message = f"""You are a specialized Q&A agent that searches specific documentation websites.

AVAILABLE KNOWLEDGE SOURCES split by category/domain/topic having the website and description for each category:
{knowledge_sources_md}


INSTRUCTIONS:
1. You MUST use the search tool for ANY question
2. Analyze the user's question to determine relevant domains/topics/categories
3. Select appropriate sites based on technologies/topics mentioned
4. You must only answer questions about available knowledge sources: {domains}
5. If question is outside available knowledge sources, do not answer the question and suggest which topics you can answer

RULES:
- Be helpful and comprehensive
- Cite sources when possible

You have access to: {{tools}}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{{tool_names}}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{input}}
Thought:{{agent_scratchpad}}"""

        # Human message template with placeholders
        human_message = """{input}

{agent_scratchpad}"""

        # Combine into chat prompt template
        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                MessagesPlaceholder("chat_history", optional=True),  # For conversation context
                ("human", human_message),
            ]
        )

    async def achat(self, user_input: str) -> str:
        """Process user input asynchronously"""
        try:
            logger.info(f"Processing: {user_input}")

            # Prepare input with conversation history
            agent_input = {"input": user_input, "chat_history": self.chat_history}

            # Execute agent with async call
            response = await self.agent_executor.ainvoke(agent_input)
            answer = response.get("output", "I couldn't process your request.")

            # Update conversation history for context
            self.chat_history.extend(
                [HumanMessage(content=user_input), AIMessage(content=answer)]
            )

            return answer

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def chat(self, user_input: str) -> str:
        """Process user input synchronously"""
        try:
            logger.info(f"Processing: {user_input}")

            # Prepare input with conversation history
            agent_input = {"input": user_input, "chat_history": self.chat_history}

            # Execute agent synchronously
            response = self.agent_executor.invoke(agent_input)
            answer = response.get("output", "I couldn't process your request.")

            # Update conversation history for context
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
