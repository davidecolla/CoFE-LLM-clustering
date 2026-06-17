# © Copyright European Union - 2026

import os
import logging
import asyncio
from typing import List, Dict, Any, Literal
from contextlib import AsyncExitStack
from datetime import timedelta
import asyncio
import logging
from dataclasses import dataclass
from typing import Literal
from pydantic import BaseModel

from mcp import ClientSession, McpError
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import Content

logger = logging.getLogger()

READ_TIMEOUT_SECONDS = timedelta(seconds=int(os.getenv("READ_TIMEOUT_SECONDS", 10)))



logger = logging.getLogger()


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    
class ParameterizedSingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        # Use the class and the arguments as a key to identify the instance
        key = (cls, args, frozenset(kwargs.items()))
        if key not in cls._instances:
            # Create a new instance if it doesn't exist
            instance = super().__call__(*args, **kwargs)
            cls._instances[key] = instance
        return cls._instances[key]
    
class McpConnection(BaseModel):
    name: str
    url: str
    clientType: Literal["sse", "streamable-http"]


@dataclass
class McpSession:
    name: str
    client: "ClientSession"
    task: asyncio.Task
    stop_event: asyncio.Event

    async def close(self) -> None:
        logger.info(f"Sending stop event to {self.name}")
        self.stop_event.set()
        logger.info(f"Waiting for task {self.name} to finish")
        await self.task
        logger.info(f"Task {self.name} finished")


class IMCPCLient:
    session: McpSession

    async def connect(self):
        raise NotImplementedError()
    
    async def cleanup(self):
        if hasattr(self, "session"):
            await self.session.close()
    
    async def get_tools_schema(self, attempt: int = 1) -> List[Dict[str, Any]]:        
        if not hasattr(self, "session"):
            await self.connect()
        session = self.session
        try:            
            response = await session.client.list_tools()        
        except Exception as e:
            logger.error(f"Could not fetch tools: {e}")
            if attempt >= 2:
                raise
            await self.reconnect()
            return await self.get_tools_schema(attempt+1)        
        available_tools = [{ 
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]        
        return available_tools
    
    async def log_tools(self):
        response = await self.session.client.list_tools()
        tools = response.tools
        print(f"\nConnected to server with tools: {[tool.name for tool in tools]}")
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]|None = None) -> List[Content]:
        try:
            result = await self.session.client.call_tool(name, arguments)
        except AttributeError:
            await self.connect()
            return await self.call_tool(name, arguments)        
        return result.content
    
    async def reconnect(self):        
        try:
            await self.cleanup()        
        except Exception as e:
            logger.error(f"Error during cleanup of client session: {e}")
        try:
            await self.connect()
        except (McpError, ConnectionError) as e:
            logger.error(f"Could not reconnect to MCP server: {e}")
        except ExceptionGroup as e:
            logger.error(f"{e}")
        except Exception as e:
            logger.error(f"Unhandled exception while trying to reconnect MCP server: {e}")

class MCPClient(IMCPCLient, metaclass=ParameterizedSingletonMeta):    
    connection_name: str
    transport_type: Literal['sse', 'streamable-http']
    server_url: str
    api_key: str    
    
    def __init__(self, transport_type: Literal['sse', 'streamable-http'], 
                 server_url: str, api_key: str|None = None):
        self.connection_name = "ALOHA"
        self.transport_type = transport_type
        self.server_url = server_url
        self.api_key = api_key or "empty"    

    async def connect(self):
        ready_event = asyncio.Event()
        transport_type = self.transport_type
        server_url = self.server_url
        api_key = self.api_key
        headers = {"Authorization": f"Bearer {api_key}"}
        
        async def mcp_session_runner() -> None:            
            # cleanup existing collection
            try:
                await self.cleanup()
            except Exception as e:
                logger.exception(e)
                pass
            
            # Create a new MCP connection
            try:
                exit_stack = AsyncExitStack()
                mcp_connection = McpConnection(
                    url=server_url, name=self.connection_name,
                    clientType=transport_type
                )

                if transport_type == 'sse':
                    read_stream, write_stream = await exit_stack.enter_async_context(
                        sse_client(
                            url=mcp_connection.url,
                            headers=headers
                        )
                    )
                else:                                
                    read_stream, write_stream, _ = await exit_stack.enter_async_context(
                        streamablehttp_client(
                            url=mcp_connection.url,
                            headers=headers
                        )
                    )
                
                mcp_client: ClientSession = await exit_stack.enter_async_context(
                    ClientSession(
                        read_stream, write_stream, read_timeout_seconds=READ_TIMEOUT_SECONDS
                    )
                )                

                # Initialize the session                
                await mcp_client.initialize()
                
            except (BaseException, Exception) as e:
                logger.exception(
                    "Something went wrong during MCP connection",
                    exc_info=e,
                )
                raise ConnectionError(
                    f"Failed to connect to MCP server at {server_url}: {e}"
                ) from e

            finally:
                logger.info("Sending MCP connection ready event")
                ready_event.set()

            try:
                stop_event = asyncio.Event()        

                # Store the session
                current_task = asyncio.current_task()
                assert current_task is not None, "Current task should not be None"
                self.session = McpSession(
                    name=mcp_connection.name,
                    client=mcp_client,
                    task=current_task,
                    stop_event=stop_event,
                )

                await self.log_tools()
                              
                # Wait for the stop event
                await stop_event.wait()
            except asyncio.CancelledError:
                logger.info(f"MCP session {self.connection_name} cancelled")
                raise

            except Exception as e:
                logger.exception(
                    "Something went wrong during MCP connection",
                    exc_info=e,
                )
                raise

            finally:
                logger.info(f"Closing MCP session {self.connection_name}")
                try:
                    await exit_stack.aclose()

                except Exception as e:
                    logger.exception("Error during exit stack close", exc_info=e)
                    pass

                logger.info(f"MCP session {self.connection_name} closed")
        
        # Run the session runner in a separate task
        asyncio.create_task(mcp_session_runner())
        logger.info(f"Waiting for MCP connection {self.connection_name} to be ready")
        await ready_event.wait()
        logger.info(f"MCP connection {self.connection_name} is ready")
