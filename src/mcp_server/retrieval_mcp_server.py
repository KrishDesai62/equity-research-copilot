from mcp.server.fastmcp import FastMCP
from src.rag.retrieval import retrieve

# Initialize the FastMCP server wrapper
mcp_app = FastMCP("equity-research-retrieval")

@mcp_app.tool()
def retrieve_filing_text(
    query: str, 
    ticker: str = None, 
    filing_type: str = None, 
    k: int = 5
) -> list[dict]:
    """
    Search SEC filing text (10-K/10-Q) by semantic vector similarity 
    and sparse keyword matching, optionally filtered by ticker and filing type.
    """
    # Delegates directly to your core retrieval pipeline with zero logic duplication
    return retrieve(query, ticker=ticker, filing_type=filing_type, k=k)

if __name__ == "__main__":
    # Runs the server over standard I/O (stdio) for local host communication
    mcp_app.run()