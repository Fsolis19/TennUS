# run_backend.py
import asyncio
import sys
import uvicorn
from pathlib import Path

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent

    uvicorn.run(
        "app.backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,  
        log_level="info"
    )
