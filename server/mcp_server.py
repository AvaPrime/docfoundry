# DocFoundry MCP Server - Full JSON-RPC 2.0 implementation
# Implements Model Context Protocol for serving documentation resources

import sys, json, pathlib, sqlite3, asyncio, logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "docfoundry.db"
DOCS_DIR = BASE_DIR / "docs"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class JSONRPCRequest:
    jsonrpc: str
    id: Union[str, int]
    method: str
    params: Optional[Dict[str, Any]] = None

@dataclass
class JSONRPCResponse:
    jsonrpc: str
    id: Union[str, int]
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

class MCPServer:
    def __init__(self):
        self.capabilities = {
            "resources": {
                "subscribe": True,
                "listChanged": True
            },
            "tools": {
                "listChanged": True
            }
        }
        self.server_info = {
            "name": "docfoundry-mcp-server",
            "version": "1.0.0"
        }
        self.session_initialized = False
    
    def db_connection(self):
        """Get database connection with row factory"""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    
    async def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialize request"""
        client_info = params.get("clientInfo", {})
        logger.info(f"Initializing MCP session with client: {client_info.get('name', 'unknown')}")
        
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": self.capabilities,
            "serverInfo": self.server_info
        }
    
    async def handle_initialized(self, params: Dict[str, Any]) -> None:
        """Handle MCP initialized notification"""
        self.session_initialized = True
        logger.info("MCP session initialized successfully")
    
    async def handle_resources_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List available documentation resources"""
        cursor = params.get("cursor")
        limit = min(params.get("limit", 100), 200)  # Cap at 200
        
        conn = self.db_connection()
        try:
            if cursor:
                # Implement cursor-based pagination
                query = "SELECT path, title, source_url FROM documents WHERE path > ? ORDER BY path LIMIT ?"
                rows = conn.execute(query, (cursor, limit)).fetchall()
            else:
                query = "SELECT path, title, source_url FROM documents ORDER BY path LIMIT ?"
                rows = conn.execute(query, (limit,)).fetchall()
            
            resources = []
            for row in rows:
                resources.append({
                    "uri": f"docfoundry://{row['path']}",
                    "name": row['title'] or row['path'],
                    "description": f"Documentation from {row['source_url'] or 'local'}",
                    "mimeType": "text/markdown"
                })
            
            result = {"resources": resources}
            if len(resources) == limit:
                result["nextCursor"] = resources[-1]["uri"].replace("docfoundry://", "")
            
            return result
        finally:
            conn.close()
    
    async def handle_resources_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read a specific documentation resource"""
        uri = params.get("uri", "")
        if not uri.startswith("docfoundry://"):
            raise ValueError(f"Invalid URI scheme: {uri}")
        
        path = uri.replace("docfoundry://", "")
        file_path = BASE_DIR / path
        
        # Security check: ensure path is within allowed directories
        try:
            file_path = file_path.resolve()
            if not (str(file_path).startswith(str(DOCS_DIR.resolve())) or 
                   str(file_path).startswith(str((BASE_DIR / "docs").resolve()))):
                raise ValueError("Access denied: path outside allowed directories")
        except Exception:
            raise ValueError(f"Invalid path: {path}")
        
        if not file_path.exists() or not file_path.is_file():
            raise ValueError(f"Resource not found: {uri}")
        
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": "text/markdown",
                    "text": content
                }]
            }
        except Exception as e:
            raise ValueError(f"Failed to read resource: {e}")
    
    async def handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List available MCP tools"""
        return {
            "tools": [
                {
                    "name": "search_docs",
                    "description": "Search through documentation using full-text search",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 20
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "get_document_info",
                    "description": "Get metadata about a specific document",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Document path"
                            }
                        },
                        "required": ["path"]
                    }
                }
            ]
        }
    
    async def handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute MCP tool calls"""
        name = params.get("name")
        arguments = params.get("arguments", {})
        
        if name == "search_docs":
            return await self._tool_search_docs(arguments)
        elif name == "get_document_info":
            return await self._tool_get_document_info(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    async def _tool_search_docs(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Search documentation tool implementation"""
        query = args.get("query", "")
        limit = min(args.get("limit", 5), 20)
        
        if not query.strip():
            return {"content": [{"type": "text", "text": "Error: Empty search query"}]}
        
        conn = self.db_connection()
        try:
            # Try FTS5 search first
            try:
                rows = conn.execute("""
                    SELECT d.path, d.title, d.source_url, c.heading, c.anchor,
                           snippet(chunks_fts, 0, '<b>', '</b>', '…', 8) AS snippet
                    FROM chunks_fts
                    JOIN chunks c ON c.id = chunks_fts.rowid
                    JOIN documents d ON d.id = c.document_id
                    WHERE chunks_fts MATCH ?
                    LIMIT ?
                """, (query, limit)).fetchall()
            except Exception:
                # Fallback to LIKE search
                rows = conn.execute("""
                    SELECT d.path, d.title, d.source_url, c.heading, c.anchor, 
                           substr(c.text, 1, 240) as snippet
                    FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    WHERE c.text LIKE ?
                    LIMIT ?
                """, (f"%{query}%", limit)).fetchall()
            
            if not rows:
                return {"content": [{"type": "text", "text": f"No results found for: {query}"}]}
            
            results = []
            for row in rows:
                result_text = f"**{row['title']}**\n"
                if row['heading']:
                    result_text += f"Section: {row['heading']}\n"
                result_text += f"Path: {row['path']}\n"
                if row['source_url']:
                    result_text += f"Source: {row['source_url']}\n"
                result_text += f"\n{row['snippet']}\n"
                results.append(result_text)
            
            return {
                "content": [{
                    "type": "text",
                    "text": f"Found {len(results)} results for '{query}':\n\n" + "\n---\n".join(results)
                }]
            }
        finally:
            conn.close()
    
    async def _tool_get_document_info(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get document info tool implementation"""
        path = args.get("path", "")
        if not path:
            return {"content": [{"type": "text", "text": "Error: No path provided"}]}
        
        conn = self.db_connection()
        try:
            row = conn.execute(
                "SELECT path, title, source_url, captured_at, hash FROM documents WHERE path = ?",
                (path,)
            ).fetchone()
            
            if not row:
                return {"content": [{"type": "text", "text": f"Document not found: {path}"}]}
            
            # Get chunk count
            chunk_count = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE document_id = (SELECT id FROM documents WHERE path = ?)",
                (path,)
            ).fetchone()[0]
            
            info_text = f"**Document Information**\n\n"
            info_text += f"Path: {row['path']}\n"
            info_text += f"Title: {row['title']}\n"
            if row['source_url']:
                info_text += f"Source URL: {row['source_url']}\n"
            if row['captured_at']:
                info_text += f"Captured: {row['captured_at']}\n"
            info_text += f"Chunks: {chunk_count}\n"
            info_text += f"Hash: {row['hash'][:16]}...\n"
            
            return {"content": [{"type": "text", "text": info_text}]}
        finally:
            conn.close()
    
    async def handle_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main request handler following JSON-RPC 2.0 spec"""
        try:
            # Validate JSON-RPC structure
            if request_data.get("jsonrpc") != "2.0":
                raise ValueError("Invalid JSON-RPC version")
            
            method = request_data.get("method")
            params = request_data.get("params", {})
            request_id = request_data.get("id")
            
            if not method:
                raise ValueError("Missing method")
            
            # Handle different MCP methods
            if method == "initialize":
                result = await self.handle_initialize(params)
            elif method == "initialized":
                await self.handle_initialized(params)
                return None  # Notification, no response
            elif method == "resources/list":
                result = await self.handle_resources_list(params)
            elif method == "resources/read":
                result = await self.handle_resources_read(params)
            elif method == "tools/list":
                result = await self.handle_tools_list(params)
            elif method == "tools/call":
                result = await self.handle_tools_call(params)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request_data.get("id"),
                "error": {
                    "code": -32603,  # Internal error
                    "message": str(e)
                }
            }

async def main():
    """Main entry point for MCP server"""
    server = MCPServer()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--stdio":
        # JSON-RPC over stdio mode
        logger.info("Starting MCP server in stdio mode")
        
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                
                request_data = json.loads(line.strip())
                response = await server.handle_request(request_data)
                
                if response:  # Don't send response for notifications
                    print(json.dumps(response))
                    sys.stdout.flush()
                    
            except json.JSONDecodeError as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,  # Parse error
                        "message": f"Parse error: {e}"
                    }
                }
                print(json.dumps(error_response))
                sys.stdout.flush()
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                break
    else:
        # Legacy mode for testing
        print("DocFoundry MCP Server")
        print("Usage: python server/mcp_server.py --stdio")
        print("\nAvailable resources:")
        result = await server.handle_resources_list({})
        for resource in result["resources"][:5]:
            print(f"  - {resource['name']} ({resource['uri']})")

if __name__ == "__main__":
    asyncio.run(main())
