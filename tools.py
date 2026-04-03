import httpx
import logging
import os
from mcp.server.fastmcp import FastMCP

# Set up logging to stderr so it doesn't interfere with the MCP stdout pipe
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

mcp = FastMCP("AudioForge")

# These work because of your --net=host config
WHISPER_URL = "http://localhost:9000/v1/audio/transcriptions"
KOKORO_URL = "http://localhost:8880/v1/audio/speech"


@mcp.tool()
async def listen_to_audio(file_path: str) -> str:
    """Transcribes an audio file into text using Whisper."""
    if not os.path.exists(file_path):
        return f"Error: File {file_path} not found."

    async with httpx.AsyncClient(timeout=60.0) as client:
        with open(file_path, "rb") as f:
            files = {"file": (file_path, f, "audio/mpeg")}
            resp = await client.post(WHISPER_URL, files=files, data={"model": "whisper-1"})

        if resp.status_code != 200:
            return f"Error: Whisper service failed (Status {resp.status_code})"

        text = resp.json().get("text", "Error: No transcription found.")
        return text  # Returning just the text is cleaner for the LLM to process


@mcp.tool()
async def speak_text(text: str, voice: str = "af_heart") -> str:
    """Converts text to an mp3 file and saves it to the workspace."""
    output_path = "/workspace/output.mp3"  # Save to your shared mount

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(KOKORO_URL, json={"input": text, "voice": voice})

        if resp.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(resp.content)
            return f"Success: Audio saved to {output_path}"
        return f"Error: Kokoro failed (Status {resp.status_code})"

if __name__ == "__main__":
    # Standard STDIO run for LangGraph integration
    mcp.run()
