"""
MCP server wrapping the retrieval backbone (src/rag/retrieval.retrieve).

Build this ANY TIME AFTER Phase 1 (the MVP) is stable and retrieve()
already works and is tested — this file should be a thin wrapper adding
zero new retrieval logic. If you find yourself writing new logic here,
it belongs in rag/retrieval.py instead.

Once this exists, Claude (or any other MCP-compatible agent) can query
your filing index directly as a tool, instead of you hand-wiring every
agent's calls to retrieve() individually.
"""
# TODO: use the `mcp` Python SDK (see requirements.txt) to expose a
# single tool, e.g.:
#
#   from mcp.server.fastmcp import FastMCP
#   from src.rag.retrieval import retrieve
#
#   mcp_app = FastMCP("equity-research-retrieval")
#
#   @mcp_app.tool()
#   def retrieve_filing_text(query: str, ticker: str = None,
#                             filing_type: str = None, k: int = 5) -> list[dict]:
#       """Search SEC filing text (10-K/10-Q/8-K) by semantic query,
#       optionally filtered by ticker and filing type."""
#       return retrieve(query, ticker=ticker, filing_type=filing_type, k=k)
#
#   if __name__ == "__main__":
#       mcp_app.run()
raise NotImplementedError
