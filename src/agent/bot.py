import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.agents import create_agent
from src.agent.tools import query_clinical_guidelines, query_user_health_data
import structlog

logger = structlog.get_logger()
load_dotenv()

def initialize_agent():
    logger.info("Initializing agent")
    
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

    if not deepseek_api_key:
        raise Exception("Missing API Key")

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com",
        max_completion_tokens=1024
    )

    tools = [query_clinical_guidelines, query_user_health_data]

    system_prompt = SystemMessage(content=(
        "You are VitalAgent, an expert health data copilot. "
        "You have two tools at your disposal: "
        "1. query_user_health_data: Use this to write and execute SQL to analyze the user's biometric trends. "
        "2. query_clinical_guidelines: Use this to search medical documents for health advice. "
        "If a user asks a complex question, use BOTH tools: "
        "first query the database to find their metrics, then query the guidelines to see if it is normal. "
        "Always synthesize the final answer clearly and concisely."
    ))

    agent_executer = create_agent(llm, tools, system_prompt=system_prompt)
    return agent_executer


def chat_loop():
    agent = initialize_agent()
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
                
            if not user_input.strip():
                continue

            print("\nVitalAgent: ", end="", flush=True)
            
            result = agent.invoke({"messages": [HumanMessage(content=user_input)]})
            
            print(result["messages"][-1].content)
            print("\n" + "-"*50)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error("agent_crash", error=str(e))
            print(f"\n[Error: {e}]")

if __name__ == "__main__":
    chat_loop()
