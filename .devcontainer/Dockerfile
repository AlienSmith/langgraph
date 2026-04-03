FROM mcr.microsoft.com/devcontainers/python:3.11-bookworm

# 1. Tells pip it's okay to install globally as root
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# 2. Ensures Python doesn't buffer logs (so you see LangGraph output instantly)
ENV PYTHONUNBUFFERED=1

# 3. One-shot install for all your orchestration tools
RUN pip install --no-cache-dir \
    langgraph \
    langchain-openai \
    mcp \
    langchain-mcp-adapters \
    python-dotenv \
    httpx

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch \
    numpy \
    matplotlib