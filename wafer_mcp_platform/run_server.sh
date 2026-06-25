#!/usr/bin/env bash
# Start the MCP server. Default: stdio (for Claude Desktop / the dashboard).
# Pass --http for a streamable-HTTP server on :8000.
cd "$(dirname "$0")"
exec python -m mcp_server.server "$@"
