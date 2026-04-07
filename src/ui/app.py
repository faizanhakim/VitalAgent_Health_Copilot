import streamlit as st
from langchain_core.messages import HumanMessage
import sys
import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.agent.bot import initialize_agent

st.set_page_config(
    page_title="VitalAgent Copilot",
    page_icon="🧬",
    layout="centered"
)

st.title("🧬 VitalAgent Health Copilot")
st.markdown("""
Welcome to your personal health intelligence layer. 
Ask me to analyze your wearable data or look up clinical guidelines.
""")

@st.cache_resource
def get_agent():
    return initialize_agent()

agent_executor = get_agent()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am VitalAgent. I have access to your normalized wearable data and clinical guidelines. How can I help you today?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("E.g., What was my average resting heart rate last week?"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.status("🧠 Agent is thinking... (Executing Tools)", expanded=True) as status:
            st.write("Analyzing query, writing SQL, and searching Vector DB...")
            
            try:
                invoke_payload = {
                    "input": prompt,
                    "messages": [HumanMessage(content=prompt)]
                }
                response = agent_executor.invoke(invoke_payload)
                
                if "output" in response:
                    final_answer = response["output"]
                elif "messages" in response:
                    final_answer = response["messages"][-1].content
                else:
                    final_answer = str(response) 
                
                status.update(label="✅ Answer generated!", state="complete", expanded=False)
                
            except Exception as e:
                status.update(label="❌ Error occurred", state="error")
                st.error(f"Agent failed: {str(e)}")
                st.stop()
        
        # Display the final synthesized answer
        st.write(final_answer)
        
        # Save to history
        st.session_state.messages.append({"role": "assistant", "content": final_answer})