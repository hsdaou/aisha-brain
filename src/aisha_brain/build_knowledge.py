#!/usr/bin/env python3
"""Convenience wrapper — runs the knowledge builder from the package module."""
import asyncio
from aisha_brain.build_knowledge import main

if __name__ == "__main__":
    asyncio.run(main())
