"""
Client example for the Domain Q&A Agent API
This script demonstrates how to interact with the agent through HTTP requests
"""

import requests
import json
import time
import asyncio
import aiohttp
from typing import Dict, List, Optional


class QAAgentClient:
    """Synchronous client for Q&A Agent API"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        # Set reasonable timeouts
        self.session.timeout = 30

    def check_health(self) -> Dict:
        """Check if the API server is running and get configuration info"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"❌ Health check failed: {e}")
            return {}

    def get_domains(self) -> List[str]:
        """Get list of available domains"""
        try:
            response = self.session.get(f"{self.base_url}/domains")
            response.raise_for_status()
            return response.json()["domains"]
        except requests.RequestException as e:
            print(f"❌ Failed to get domains: {e}")
            return []

    def chat(self, message: str) -> Optional[str]:
        """Send a message to the agent and get response"""
        try:
            payload = {"message": message}
            response = self.session.post(
                f"{self.base_url}/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

            result = response.json()
            return result.get("response", "No response received")

        except requests.RequestException as e:
            print(f"❌ Chat request failed: {e}")
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_detail = e.response.json()
                    print(f"   Error details: {error_detail}")
                except BaseException:
                    print(
                        f"   HTTP {
                            e.response.status_code}: {
                            e.response.text}"
                    )
            return None

    def reset_memory(self) -> bool:
        """Reset the agent's conversation memory"""
        try:
            response = self.session.post(f"{self.base_url}/reset")
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            print(f"❌ Failed to reset memory: {e}")
            return False


class AsyncQAAgentClient:
    """Asynchronous client for Q&A Agent API"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    async def check_health(self) -> Dict:
        """Check if the API server is running and get configuration info"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health") as response:
                    response.raise_for_status()
                    return await response.json()
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return {}

    async def get_domains(self) -> List[str]:
        """Get list of available domains"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/domains") as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result["domains"]
        except Exception as e:
            print(f"❌ Failed to get domains: {e}")
            return []

    async def chat(self, message: str) -> Optional[str]:
        """Send a message to the agent and get response"""
        try:
            payload = {"message": message}
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result.get("response", "No response received")

        except Exception as e:
            print(f"❌ Chat request failed: {e}")
            return None

    async def reset_memory(self) -> bool:
        """Reset the agent's conversation memory"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/reset") as response:
                    response.raise_for_status()
                    return True
        except Exception as e:
            print(f"❌ Failed to reset memory: {e}")
            return False


def demo_sync_client():
    """Demonstrate synchronous client usage"""
    print("🔄 Synchronous Client Demo")
    print("=" * 40)

    client = QAAgentClient()

    # Check server health
    print("1. Checking server health...")
    health = client.check_health()
    if health:
        print("✅ Server is healthy!")
        print(
            f"   Configuration: {
                json.dumps(
                    health.get(
                        'config',
                        {}),
                    indent=2)}"
        )
    else:
        print(
            "❌ Server is not responding. Make sure it's running with: python main.py"
        )
        return

    # Get available domains
    print("\n2. Getting available domains...")
    domains = client.get_domains()
    if domains:
        print(f"✅ Found {len(domains)} domains: {', '.join(domains)}")
    else:
        print("❌ No domains available")
        return

    # Test questions
    test_questions = [
        "How do I create a FastAPI application with async endpoints?",
        "What are the best practices for training neural networks with PyTorch?",
        "How do I handle data cleaning with pandas?",
        "What's the difference between classification and regression in machine learning?",
    ]

    print("\n3. Testing chat functionality...")
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Question {i}: {question}")
        print("🤔 Thinking...")

        start_time = time.time()
        response = client.chat(question)
        end_time = time.time()

        if response:
            print(f"✅ Response received in {end_time - start_time:.2f}s")
            print(f"🤖 Agent: {response[:200]}{'...' if len(response) > 200 else ''}")
        else:
            print("❌ No response received")

        # Small delay between questions
        time.sleep(1)

    # Reset memory
    print("\n4. Resetting conversation memory...")
    if client.reset_memory():
        print("✅ Memory reset successfully")
    else:
        print("❌ Failed to reset memory")


async def demo_async_client():
    """Demonstrate asynchronous client usage"""
    print("\n🚀 Asynchronous Client Demo")
    print("=" * 40)

    client = AsyncQAAgentClient()

    # Check server health
    print("1. Checking server health...")
    health = await client.check_health()
    if health:
        print("✅ Server is healthy!")
        print(
            f"   Configuration: {
                json.dumps(
                    health.get(
                        'config',
                        {}),
                    indent=2)}"
        )
    else:
        print(
            "❌ Server is not responding. Make sure it's running with: python main.py"
        )
        return

    # Get available domains
    print("\n2. Getting available domains...")
    domains = await client.get_domains()
    if domains:
        print(f"✅ Found {len(domains)} domains: {', '.join(domains)}")
    else:
        print("❌ No domains available")
        return

    # Test concurrent questions
    concurrent_questions = [
        "How do I use FastAPI middleware?",
        "What are PyTorch tensors?",
        "How do I merge DataFrames in pandas?",
    ]

    print("\n3. Testing concurrent chat requests...")
    start_time = time.time()

    # Send all questions concurrently
    tasks = []
    for question in concurrent_questions:
        print(f"📝 Queuing: {question}")
        task = asyncio.create_task(client.chat(question))
        tasks.append((question, task))

    # Wait for all responses
    print("🤔 Processing all questions concurrently...")
    for question, task in tasks:
        response = await task
        if response:
            print(
                f"✅ '{question[:50]}...' -> {response[:100]}{'...' if len(response) > 100 else ''}"
            )
        else:
            print(f"❌ '{question[:50]}...' -> No response")

    end_time = time.time()
    print(
        f"🏁 All {
            len(concurrent_questions)} questions processed in {
            end_time -
            start_time:.2f}s"
    )

    # Reset memory
    print("\n4. Resetting conversation memory...")
    if await client.reset_memory():
        print("✅ Memory reset successfully")
    else:
        print("❌ Failed to reset memory")


def interactive_mode():
    """Interactive chat mode"""
    print("\n💬 Interactive Chat Mode")
    print("=" * 40)
    print(
        "Type 'quit' to exit, 'reset' to clear memory, 'domains' to see available domains"
    )

    client = QAAgentClient()

    # Check if server is running
    if not client.check_health():
        print("❌ Server is not running. Please start it with: python main.py")
        return

    while True:
        try:
            user_input = input("\n🙋 You: ").strip()

            if user_input.lower() == "quit":
                print("👋 Goodbye!")
                break
            elif user_input.lower() == "reset":
                if client.reset_memory():
                    print("✅ Memory reset")
                else:
                    print("❌ Failed to reset memory")
                continue
            elif user_input.lower() == "domains":
                domains = client.get_domains()
                print(f"📚 Available domains: {', '.join(domains)}")
                continue
            elif not user_input:
                print("Please enter a question or 'quit' to exit")
                continue

            print("🤔 Agent is thinking...")
            response = client.chat(user_input)

            if response:
                print(f"🤖 Agent: {response}")
            else:
                print("❌ Sorry, I couldn't process your request")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


async def main():
    """Main demo function"""
    print("🤖 Domain Q&A Agent - Client Demo")
    print("=" * 50)

    # Run sync demo
    demo_sync_client()

    # Run async demo
    await demo_async_client()

    # Ask user if they want interactive mode
    print("\n" + "=" * 50)
    try:
        choice = (
            input("Would you like to try interactive mode? (y/n): ").strip().lower()
        )
        if choice in ["y", "yes"]:
            interactive_mode()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    # Install aiohttp if not available
    try:
        import aiohttp
    except ImportError:
        print("⚠️  aiohttp not found. Installing...")
        import subprocess

        subprocess.check_call(["pip", "install", "aiohttp"])
        import aiohttp

    # Run the demo
    asyncio.run(main())
