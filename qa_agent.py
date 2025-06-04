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
            early_stopping_method="generate",
            return_intermediate_steps=True,
            handle_parsing_errors="Check your output format!",
        )

    def _create_structured_chat_prompt(self) -> ChatPromptTemplate:
        """Create structured chat prompt with domain restrictions"""
        # Get available domains for the prompt
        domains_info = self.get_available_domains()
        domains_text = "\n".join(
            [
                f"- {domain['domain']}: {domain['description']}"
                for domain in domains_info
            ]
        )

        # System message defines agent behavior and rules
        system_message = f"""You are a Q&A assistant that ONLY provides information from specific documentation sources.

AVAILABLE DOMAINS:
{domains_text}

RULES:
1. ALWAYS use the search tool for ANY question
2. ONLY answer questions related to the domains above
3. If question is outside available domains, refuse and list available domains
4. Search the most relevant domain for the user's query

You have access to: {{tools}}

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

If question is outside available domains, respond:
"I can only provide information about: {domains_text}

Your question is outside my available domains. Please ask about the above topics."

Always use search tool first, then provide comprehensive Final Answer."""

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

    def get_available_domains(self) -> List[Dict[str, str]]:
        """Get available domains and descriptions"""
        return self.sites_df[["domain", "description"]].to_dict("records")

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
