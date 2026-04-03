import asyncio
import httpx  # Cleaner than 'requests' for async
from typing import List, Dict

# The Worker Registry
WORKERS = [
    {"name": "Whisper", "url": "http://localhost:9000/health"},
    {"name": "Kokoro",  "url": "http://localhost:8880/health"}
]

async def check_worker(worker: Dict[str, str]) -> bool:
    """Attempt to ping a worker until it's ready."""
    async with httpx.AsyncClient() as client:
        for attempt in range(15):
            try:
                response = await client.get(worker["url"], timeout=2.0)
                if response.status_code == 200:
                    print(f"✅ {worker['name']} is online and ready.")
                    return True
            except (httpx.ConnectError, httpx.TimeoutException):
                print(f"[!] {worker['name']} warming up... (Attempt {attempt+1}/15)")
                await asyncio.sleep(2)
        return False

async def run_bootstrap():
    print("🚀 [Bootstrap] Starting GPU Worker verification on 2070 Super...")
    
    # We run these in parallel to save time
    results = await asyncio.gather(*(check_worker(w) for w in WORKERS))
    
    if all(results):
        print("🟢 All systems nominal. Starting OpenClaw Agent.")
    else:
        print("⚠️ Warning: Some workers failed to report healthy. Check 'docker logs'.")

if __name__ == "__main__":
    asyncio.run(run_bootstrap())