"""
================================================================================
Simple LangGraph Chatbot Tutorial (Configured for Google Gemini & LangSmith)
================================================================================

This tutorial script is adapted from Tech With Tim's LangGraph YouTube tutorial:
"LangGraph Tutorial - How to Build Advanced AI Agent Systems"

Core concepts covered here:
1. Environment & API Setup (Google Gemini & LangSmith Tracing)
2. Defining Graph State with `TypedDict` and the `add_messages` Reducer
3. Creating a Graph Node (LLM Chatbot)
4. Connecting Nodes with Edges (`START`, custom nodes, `END`)
5. Compiling and Visualizing the Graph (saving `output.png`)
6. Running an Interactive Streaming/Invoking Chat Loop
"""

import os
import sys
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

# ------------------------------------------------------------------------------
# STEP 1: Load Environment Variables & Configure LangSmith Tracing
# ------------------------------------------------------------------------------
# `load_dotenv()` reads key-value pairs from your `.env` file and puts them into
# `os.environ`. This keeps your API keys secret and out of source control.
load_dotenv()

# Gemini Authentication:
# `langchain-google-genai` natively checks both `GOOGLE_API_KEY` and `GEMINI_API_KEY`.
# We ensure both are synchronized if only one is set in the `.env`.
if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

# LangSmith Tracing Configuration:
# LangSmith is an observability platform that logs every graph step, LLM call,
# latency metric, token count, and metadata.
# - LANGCHAIN_TRACING_V2: Enables tracing in LangChain and LangGraph.
# - LANGCHAIN_ENDPOINT: The API endpoint URL. Note that if your account is in Europe,
#   this must be "https://eu.api.smith.langchain.com". For the US, "https://api.smith.langchain.com".
# - LANGCHAIN_PROJECT: Groups traces under a specific project dashboard in LangSmith.
# - LANGCHAIN_API_KEY / LANGSMITH_API_KEY: Authentication token starting with 'lsv2_...'.
os.environ["LANGCHAIN_TRACING_V2"] = "true"
if not os.environ.get("LANGCHAIN_ENDPOINT"):
    os.environ["LANGCHAIN_ENDPOINT"] = "https://eu.api.smith.langchain.com"
if not os.environ.get("LANGCHAIN_PROJECT"):
    os.environ["LANGCHAIN_PROJECT"] = "LangGraph-Tutorial"
if not os.environ.get("LANGCHAIN_API_KEY") and os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGSMITH_API_KEY"]

# ------------------------------------------------------------------------------
# STEP 2: Import LangGraph and Gemini Chat Model
# ------------------------------------------------------------------------------
# - StateGraph: The core state-machine builder class in LangGraph.
# - START: A special virtual node representing the entry point of the graph.
# - END: A special virtual node representing the terminal exit point of the graph.
# - add_messages: A reducer function that appends incoming messages to conversation history
#   instead of replacing the existing list.
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# Read model name from `.env` (defaults to "gemini-3.5-flash-lite" or "gemini-2.5-flash")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

print(f"🔧 Initializing Google Gemini model: '{MODEL_NAME}'")
print(f"📊 LangSmith Tracing: Active (Project: {os.environ.get('LANGCHAIN_PROJECT')})")

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0.7,
)

# ------------------------------------------------------------------------------
# STEP 3: Define the Graph State Schema
# ------------------------------------------------------------------------------
# In LangGraph, every graph has a shared "State" that acts as its memory.
# Nodes take this State as input and return an updated dictionary.
#
# What does `Annotated[list, add_messages]` mean?
# - By default, when a node returns a dictionary like `{"messages": [new_msg]}`,
#   LangGraph would OVERWRITE the `messages` key with `[new_msg]`.
# - By attaching the `add_messages` reducer via `Annotated`, we instruct LangGraph:
#   "Whenever any node returns `messages`, DO NOT overwrite; APPEND the new messages
#    to the existing list and manage message IDs/replacements automatically."
class State(TypedDict):
    messages: Annotated[list, add_messages]


# ------------------------------------------------------------------------------
# STEP 4: Define the Chatbot Node Function
# ------------------------------------------------------------------------------
# A node in LangGraph is simply a Python function that:
# 1. Accepts the current `State` dictionary as its argument.
# 2. Performs computation (e.g., calling an LLM, querying a database).
# 3. Returns a dictionary specifying updates to the state keys.
def chatbot(state: State) -> dict:
    """
    Core chatbot node:
    Takes the accumulated conversation messages from state and invokes the Gemini model.
    """
    # Invoke the Gemini LLM with the entire conversation history
    response = llm.invoke(state["messages"])
    
    # Return an update to the 'messages' list containing the LLM's response.
    # The `add_messages` reducer handles appending this to the existing state.
    return {"messages": [response]}


# ------------------------------------------------------------------------------
# STEP 5: Build and Compile the Graph
# ------------------------------------------------------------------------------
# 1. Instantiate the StateGraph with our State class schema.
graph_builder = StateGraph(State)

# 2. Add our node(s). The first argument is the unique node name (string identifier),
#    and the second argument is the function that implements the node.
graph_builder.add_node("chatbot", chatbot)

# 3. Add edges:
#    - From START to "chatbot": When the graph starts, execute the "chatbot" node first.
#    - From "chatbot" to END: When the "chatbot" node finishes, transition to END.
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# 4. Compile the graph into a runnable `CompiledStateGraph`.
#    The compiled graph provides standard Runnable methods: `.invoke()`, `.stream()`, etc.
graph = graph_builder.compile()


# ------------------------------------------------------------------------------
# STEP 6: Optional Graph Visualization (Mermaid PNG)
# ------------------------------------------------------------------------------
def save_graph_diagram(output_path: str = "output.png"):
    """
    Generates a Mermaid diagram image of the graph and saves it as a PNG file.
    This lets you visually inspect the workflow architecture.
    """
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open(output_path, "wb") as f:
            f.write(png_bytes)
        print(f"🖼️  Saved visual graph diagram to: {output_path}")
    except Exception as e:
        print(f"ℹ️  Could not generate visual diagram ({e}). Skipping diagram export.")


# ------------------------------------------------------------------------------
# Helper: Extract Clean Text from LLM Response
# ------------------------------------------------------------------------------
def get_clean_text(content) -> str:
    """
    Gemini responses in newer SDKs can sometimes return content as either a plain string
    or a list of structured content blocks (e.g., [{'type': 'text', 'text': '...'}]).
    This helper guarantees clean string output for printing.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return str(content)


# ------------------------------------------------------------------------------
# STEP 7: Interactive Chat Loop
# ------------------------------------------------------------------------------
def run_chatbot():
    """
    Runs an interactive terminal chat loop with our compiled LangGraph chatbot.
    All turns are automatically tracked in LangSmith!
    """
    save_graph_diagram("output.png")
    
    print("\n" + "=" * 60)
    print("🤖 LangGraph Simple Chatbot Ready!")
    print("Type your message and press Enter. Type 'exit' or 'q' to quit.")
    print("=" * 60 + "\n")

    # Initial state with empty conversation history
    state: State = {"messages": []}

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("Assistant: Goodbye! Have a great day.")
            break

        # Append the new user message to our conversation input state
        new_message = {"role": "user", "content": user_input}
        state["messages"] = state.get("messages", []) + [new_message]

        # Invoke the compiled graph:
        # LangGraph executes START -> chatbot -> END, and returns the updated state.
        # Everything executed here is traced in LangSmith in real time.
        state = graph.invoke(state)

        # Retrieve the latest message from state, which is the assistant's response
        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            content = getattr(last_message, "content", last_message)
            clean_reply = get_clean_text(content)
            print(f"Assistant: {clean_reply}\n")


if __name__ == "__main__":
    run_chatbot()
