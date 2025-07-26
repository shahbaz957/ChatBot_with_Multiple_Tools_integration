import streamlit as st
from dotenv import load_dotenv
import os
import json

# Langchain tools and wrappers
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults

# Langchain core and model setup
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AnyMessage
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# Load .env keys
load_dotenv()
os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

# Tool setup
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=5000)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

api_wrapper_wikipedia = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=500)
wikipedia = WikipediaQueryRun(api_wrapper=api_wrapper_wikipedia)

tavily = TavilySearchResults()
tools = [arxiv, wikipedia, tavily]

# Model binding
model = ChatGroq(model="qwen/qwen3-32b")
llm_with_tool = model.bind_tools(tools)


# Define LangGraph state
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# Node: tool-calling LLM
def tool_calling_llm(state: State):
    return {"messages": [llm_with_tool.invoke(state['messages'])]}


# Build LangGraph
builder = StateGraph(State)
builder.add_node('tool_calling_llm', tool_calling_llm)
builder.add_node('tools', ToolNode(tools))
builder.add_edge(START, 'tool_calling_llm')
builder.add_conditional_edges('tool_calling_llm', tools_condition)
builder.add_edge('tools', END)
graph = builder.compile()


# ------------ Streamlit UI Starts Below ------------ #

st.set_page_config(page_title="AI Agent Chat", layout="centered")
st.title("🧠 AI Agent Chatbot")

# Initialize history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



# Formatting function for tool result (list of dicts)
def format_tool_result(response_list):
    formatted = ""
    for i, item in enumerate(response_list, 1):
        formatted += f"""
### 🔍 Result {i}
**[{item['title']}]({item['url']})**

{item['content'][:3000]}...

**Relevance Score:** `{round(item['score'], 2)}`
---
"""
    return formatted


# User input
user_input = st.chat_input("Ask about AI research, news, science, etc...")
if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    result = graph.invoke({"messages": st.session_state.chat_history})
    st.session_state.chat_history = result["messages"]


# Display messages
for msg in st.session_state.chat_history:
    if msg.type == "human":
        with st.chat_message("🧑‍💬 User"):
            st.markdown(f"**You said:**\n\n{msg.content}")

    elif msg.type == "ai":
        with st.chat_message("🤖 Agent"):
            st.markdown(f"**Agent Response:**\n\n{msg.content}")

    elif msg.type == "tool":
        with st.chat_message("🛠️ Tool"):
            try:
                parsed = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                formatted = format_tool_result(parsed)
                st.markdown(formatted, unsafe_allow_html=True)
            except Exception:
                st.markdown(f"**Tool Result:**\n\n{msg.content}")
