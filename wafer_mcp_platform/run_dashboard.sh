#!/usr/bin/env bash
# Launch the Streamlit dashboard (which itself launches the MCP server on stdio).
cd "$(dirname "$0")"
exec streamlit run dashboard/streamlit_app.py "$@"
