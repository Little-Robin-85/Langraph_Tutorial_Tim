"""
================================================================================
Advanced LangGraph Agent System Tutorial (Google Gemini + LangSmith Tracing)
================================================================================

This tutorial script is adapted from Tech With Tim's LangGraph YouTube tutorial:
"LangGraph Tutorial - How to Build Advanced AI Agent Systems"

Core concepts covered here:
1. Observability: Full LangSmith tracing configuration (EU/US endpoint support)
2. State Management: TypedDict state schemas with `add_messages` reducer
3. Structured Output: Using Pydantic models with Gemini (`with_structured_output`)
4. Dynamic Conditional Routing: Directing user intent to specialized sub-agents
5. Specialized Agent Personas:
   - 'therapist' agent (empathetic, emotional support, exploratory questions)
   - 'logical' agent (analytical, concise, fact-driven)
6. Graph Compilation & Visualization: Saving graph topology to `output1.png`
7. Interactive Execution Loop: Full multi-turn conversation support
"""

import os
import sys
from typing import Annotated, Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ------------------------------------------------------------------------------
# STEP 1: Load Environment Variables & Configure LangSmith Tracing
# ------------------------------------------------------------------------------
# `load_dotenv()` loads the .env file containing your GEMINI_API_KEY, GEMINI_MODEL,
# and LANGSMITH_API_KEY / LANGCHAIN_API_KEY.
load_dotenv()

# Synchronize Gemini API keys for seamless integration with google-genai
if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

# LangSmith Tracing Configuration:
# - LANGCHAIN_TRACING_V2: Enables tracing of all chains, agents, and graphs.
# - LANGCHAIN_ENDPOINT: Explicitly set to the EU endpoint ('https://eu.api.smith.langchain.com')
#   matching your LangSmith account region.
# - LANGCHAIN_PROJECT: Groups traces in your LangSmith dashboard under this project.
# - LANGCHAIN_API_KEY: Authenticates your trace ingestion requests.
os.environ["LANGCHAIN_TRACING_V2"] = "true"
if not os.environ.get("LANGCHAIN_ENDPOINT"):
    os.environ["LANGCHAIN_ENDPOINT"] = "https://eu.api.smith.langchain.com"
if not os.environ.get("LANGCHAIN_PROJECT"):
    os.environ["LANGCHAIN_PROJECT"] = "LangGraph-Tutorial"
if not os.environ.get("LANGCHAIN_API_KEY") and os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGSMITH_API_KEY"]

# ------------------------------------------------------------------------------
# STEP 2: Import LangGraph & Initialize the Gemini Model
# ------------------------------------------------------------------------------
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Read model name from `.env`
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

print(f"🔧 Initializing Google Gemini model: '{MODEL_NAME}'")
print(f"📊 LangSmith Tracing: Active (Project: {os.environ.get('LANGCHAIN_PROJECT')})")
print(f"🌐 LangSmith Endpoint: {os.environ.get('LANGCHAIN_ENDPOINT')}")

# Initialize our primary LLM
llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
)

# ------------------------------------------------------------------------------
# STEP 3: Define Structured Output Schema (Pydantic)
# ------------------------------------------------------------------------------
# We want our classifier node to strictly categorize incoming user queries.
# Rather than parsing unstructured text, we use Pydantic with `with_structured_output`.
#
# Pydantic validates that the LLM response is strictly either "emotional" or "logical".
class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description="Classify if the message requires an emotional (therapist) or logical response."
    )


# ------------------------------------------------------------------------------
# STEP 4: Define the Graph State Schema
# ------------------------------------------------------------------------------
# The state is shared across all nodes throughout the graph's execution.
#
# - `messages`: Uses `Annotated[list, add_messages]` so new messages are appended
#   rather than overwriting prior conversation history.
# - `message_type`: Stores the classifier's classification ("emotional" or "logical")
# - `next`: Stores the routing destination determined by the router node.
class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None
    next: str | None


# ------------------------------------------------------------------------------
# STEP 5: Define Graph Node Functions
# ------------------------------------------------------------------------------

# Helper function to extract clean plain text from Gemini's message content
def get_clean_text(content) -> str:
    """
    Extracts text content reliably from string or structured content blocks.
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


# NODE 1: Message Classifier
def classify_message(state: State) -> dict:
    """
    Node 1: Analyzes the user's latest message and classifies it into
    either 'emotional' or 'logical' using structured output.
    """
    last_message = state["messages"][-1]
    last_message_text = get_clean_text(getattr(last_message, "content", last_message))

    # Bind the Pydantic schema to Gemini to guarantee structured JSON output
    classifier_llm = llm.with_structured_output(MessageClassifier)

    system_instruction = (
        "Classify the user message as either:\n"
        "- 'emotional': if it asks for emotional support, therapy, deals with feelings, or personal problems.\n"
        "- 'logical': if it asks for facts, information, logical analysis, code, or practical solutions."
    )

    result = classifier_llm.invoke([
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": last_message_text}
    ])

    print(f"\n🔍 [Classifier Node] Classified intent as: '{result.message_type.upper()}'")
    return {"message_type": result.message_type}


# NODE 2: Router Node
def router(state: State) -> dict:
    """
    Node 2: Decides the next agent to route to based on `state['message_type']`.
    """
    message_type = state.get("message_type", "logical")
    if message_type == "emotional":
        next_destination = "therapist"
    else:
        next_destination = "logical"

    print(f"🔀 [Router Node] Routing workflow to: '{next_destination}' agent")
    return {"next": next_destination}


# NODE 3: Therapist Persona Agent
def therapist_agent(state: State) -> dict:
    """
    Node 3: Specialized persona that acts as a compassionate, empathetic therapist.
    Focuses on validating feelings and asking reflective questions.
    """
    last_message = state["messages"][-1]
    last_message_text = get_clean_text(getattr(last_message, "content", last_message))

    system_prompt = (
        "You are a compassionate therapist. Focus on the emotional aspects of the user's message. "
        "Show deep empathy, validate their feelings, and help them process their emotions. "
        "Ask thoughtful questions to help them explore their feelings more deeply. "
        "Avoid giving cold or purely logical solutions unless explicitly asked."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": last_message_text}
    ]

    reply = llm.invoke(messages)
    clean_reply_text = get_clean_text(reply.content)

    return {"messages": [{"role": "assistant", "content": clean_reply_text}]}


# NODE 4: Logical Persona Agent
def logical_agent(state: State) -> dict:
    """
    Node 4: Specialized persona that acts as a purely logical, objective assistant.
    Focuses on facts, structured reasoning, and clear evidence-based answers.
    """
    last_message = state["messages"][-1]
    last_message_text = get_clean_text(getattr(last_message, "content", last_message))

    system_prompt = (
        "You are a purely logical assistant. Focus strictly on facts, data, and information. "
        "Provide clear, concise answers based on logic and evidence. "
        "Do not address emotions or provide emotional support. "
        "Be direct, structured, and straightforward in your responses."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": last_message_text}
    ]

    reply = llm.invoke(messages)
    clean_reply_text = get_clean_text(reply.content)

    return {"messages": [{"role": "assistant", "content": clean_reply_text}]}


# ------------------------------------------------------------------------------
# STEP 6: Assemble the StateGraph
# ------------------------------------------------------------------------------
graph_builder = StateGraph(State)

# Add nodes to graph
graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("therapist", therapist_agent)
graph_builder.add_node("logical", logical_agent)

# Add fixed edges
# 1. Flow starts at virtual START node and heads directly into "classifier"
graph_builder.add_edge(START, "classifier")

# 2. Once classification is done, transition into "router"
graph_builder.add_edge("classifier", "router")

# 3. Add conditional edge from "router" to either "therapist" or "logical"
#    The condition function looks at state["next"], and maps:
#    "therapist" -> therapist node
#    "logical"   -> logical node
graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next"),
    {
        "therapist": "therapist",
        "logical": "logical",
    }
)

# 4. Once either agent generates its response, finish graph execution at END
graph_builder.add_edge("therapist", END)
graph_builder.add_edge("logical", END)

# Compile the graph into a runnable instance
graph = graph_builder.compile()


# ------------------------------------------------------------------------------
# STEP 7: Export Architecture Diagram (output1.png)
# ------------------------------------------------------------------------------
def save_graph_diagram(output_path: str = "output1.png"):
    """
    Visualizes the multi-agent graph with Mermaid and saves it as a PNG image.
    """
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open(output_path, "wb") as f:
            f.write(png_bytes)
        print(f"🖼️  Saved multi-agent graph architecture diagram to: {output_path}")
    except Exception as e:
        print(f"ℹ️  Could not generate visual diagram ({e}). Skipping diagram export.")


# ------------------------------------------------------------------------------
# STEP 8: Interactive Terminal Chat Loop
# ------------------------------------------------------------------------------
def run_chatbot():
    """
    Interactive conversation loop allowing you to test the classifier,
    router, and persona agents in real time with LangSmith tracing.
    """
    save_graph_diagram("output1.png")

    print("\n" + "=" * 65)
    print("🧠 LangGraph Multi-Agent System (Emotional vs Logical)")
    print("Try emotional messages (e.g., 'I am feeling overwhelmed today')")
    print("or logical questions (e.g., 'Explain binary search complexity')")
    print("Type 'exit', 'quit', or 'q' to end.")
    print("=" * 65 + "\n")

    state: State = {
        "messages": [],
        "message_type": None,
        "next": None,
    }

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        # Append new user message to state
        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]

        # Execute graph (traced in LangSmith)
        state = graph.invoke(state)

        # Print the assistant's response from the targeted agent
        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            content = getattr(last_message, "content", last_message)
            clean_reply = get_clean_text(content)
            agent_label = state.get("message_type", "assistant").capitalize()
            print(f"\n[{agent_label} Assistant]: {clean_reply}\n")


if __name__ == "__main__":
    run_chatbot()
