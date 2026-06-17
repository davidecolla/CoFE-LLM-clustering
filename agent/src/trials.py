# © Copyright European Union - 2026

"""
Development/testing scripts for MCP client integration and sandbox testing.
This file contains experimental code used during development.
"""

import os
import asyncio
import requests
from openai import OpenAI
from dotenv import load_dotenv
from contextlib import AsyncExitStack
from client import MCPClient
from pydantic import BaseModel, Field
from typing import Annotated
from enum import Enum

# Load environment variables from .env file
load_dotenv()

LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_BASE_URL = os.environ["LLM_BASE_URL"]


def preprocess_proposals():
    """Preprocess raw proposals into the simplified format used by the pipeline."""
    import json
    data = json.load(open("../data/proposals.json"))
    new_data = []
    for el in data:
        new_data.append({
            "text": el['title'],
            "id": el['guid'],
            "language": el['language']
        })
    json.dump(new_data, open("../data/proposals_tiny.json", "w"))


async def aloha_client():
    """Test ALOHA MCP server connectivity."""
    from fastmcp import Client
    from fastmcp.client import StreamableHttpTransport
    from fastmcp.client.auth import BearerAuth

    aloha_server = os.environ.get("ALOHA_SERVER_URL", "")
    aloha_token = os.environ.get("ALOHA_TOKEN", "")

    if not aloha_server or not aloha_token:
        print("Error: Set ALOHA_SERVER_URL and ALOHA_TOKEN in .env file")
        return

    transport = StreamableHttpTransport(
        url=aloha_server,
        auth=BearerAuth(token=aloha_token)
    )

    client = Client(transport=transport)
    async with client:
        print(f"Connected: {client.is_connected()}")
        tools = await client.list_tools()
        print(tools)
        print(f"Connected: {client.is_connected()}")


async def custom_client():
    """Test custom MCP client."""
    aloha_server = os.environ.get("ALOHA_SERVER_URL", "")
    aloha_token = os.environ.get("ALOHA_TOKEN", "")

    if not aloha_server or not aloha_token:
        print("Error: Set ALOHA_SERVER_URL and ALOHA_TOKEN in .env file")
        return

    mcp_client = MCPClient("streamable-http", aloha_server, aloha_token)
    await mcp_client.connect()
    tools = mcp_client.log_tools()
    print(tools)


if __name__ == "__main__":
    asyncio.run(aloha_client())
