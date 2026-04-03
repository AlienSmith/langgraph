import os
import asyncio
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

# 1. State Definition


class AudioState(TypedDict):
    audio_file: str
    transcript: str
    translation: str
    status: str


# 2. LLM Setup (SiliconFlow)
llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",
    openai_api_key=os.getenv("SILICONFLOW_API_KEY"),
    openai_api_base="https://api.siliconflow.cn/v1",
)

# --- GLOBAL TOOL MAP ---
# We define this globally so the nodes can see it easily
tool_map = {}

# 3. Node Definitions (Top-Level)


async def extraction_node(state: AudioState):
    print("--- Extracting Text from Audio ---")
    result = await tool_map["listen_to_audio"].ainvoke({"file_path": state["audio_file"]})

    # Robust extraction: handle strings, dicts, or lists
    if isinstance(result, str):
        text = result
    elif isinstance(result, dict) and "content" in result:
        text = result["content"]
    elif isinstance(result, list) and len(result) > 0:
        # If it's a list of dicts, get the first one's 'text' or 'content'
        first = result[0]
        text = first.get("text", first.get("content", str(first))
                         ) if isinstance(first, dict) else str(first)
    else:
        text = str(result)
    print(f"{text}")
    return {"transcript": text}


async def translation_node(state: AudioState):
    print("--- Translating with SiliconFlow ---")
    system_prompt = (
        "You are a professional translator. "
        "Task: Translate the input text into natural English. "
        "Rules: Output ONLY the translated English text and the joke. "
        "Do NOT include Chinese characters, explanations, or parentheses. "
        "Humor: End with a single funny English sentence about GPUs or AI."
    )

    user_prompt = f"Translate the following to English: {state['transcript']}"
    response = await llm.ainvoke([
        ("system", system_prompt),
        ("user", user_prompt)
    ])
    print(f"llm response {response.content}")
    return {"translation": response.content}


async def reading_node(state: AudioState):
    print("--- Generating Output Speech ---")
    result = await tool_map["speak_text"].ainvoke({"text": state["translation"]})
    return {"status": str(result)}

# 4. The Orchestrator


async def main():
    global tool_map

    # Initialize MCP Client
    client = MultiServerMCPClient({
        "audio_forge": {
            "command": "python",
            "args": ["/workspace/tools.py"],
            "transport": "stdio",
        }
    })

    print("--- Connecting to MCP Servers ---")
    tools = await client.get_tools()
    tool_map = {t.name: t for t in tools}

    # Build the Graph
    workflow = StateGraph(AudioState)

    workflow.add_node("extract", extraction_node)
    workflow.add_node("translate", translation_node)
    workflow.add_node("read", reading_node)

    workflow.add_edge(START, "extract")
    workflow.add_edge("extract", "translate")
    workflow.add_edge("translate", "read")
    workflow.add_edge("read", END)

    app = workflow.compile()

    # Run the Workflow
    initial_input = {
        "audio_file": "/workspace/audio_in/test_voice.mp3",
        "transcript": "",
        "translation": "",
        "status": ""
    }

    final_state = await app.ainvoke(initial_input)
    print(f"\nFinal Result: {final_state['status']}")

if __name__ == "__main__":
    asyncio.run(main())
