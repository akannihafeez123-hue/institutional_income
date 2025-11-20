#!/usr/bin/env python3
"""
Run both bot and monitor together
"""

import asyncio
import subprocess
import sys
import os


async def run_process(command, name):
    """Run a process and capture output"""
    print(f"🚀 Starting {name}...")
    
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    async def read_stream(stream, prefix):
        while True:
            line = await stream.readline()
            if not line:
                break
            print(f"[{prefix}] {line.decode().rstrip()}")
    
    await asyncio.gather(
        read_stream(process.stdout, name),
        read_stream(process.stderr, f"{name}-ERR")
    )
    
    return await process.wait()


async def main():
    """Run both processes"""
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "🔥 LEGENDARY EMPIRE - STARTING ALL SYSTEMS 🔥" + " " * 15 + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    try:
        # Run both in parallel
        await asyncio.gather(
            run_process([sys.executable, "main.py"], "BOT"),
            run_process([sys.executable, "monitor.py"], "MONITOR")
        )
    except KeyboardInterrupt:
        print("\n\n✨ Shutting down gracefully... ✨\n")


if __name__ == "__main__":
    asyncio.run(main())
