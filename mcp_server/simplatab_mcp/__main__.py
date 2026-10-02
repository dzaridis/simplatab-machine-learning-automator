"""Starts the Simplatab MCP server.

    python -m simplatab_mcp                                  # stdio (an agent launches the server)
    python -m simplatab_mcp --transport http --port 8000     # Streamable HTTP on http://host:8000/mcp

Environment: SIMPLATAB_WORKSPACE (experiments, uploads; default /workspace or ~/.simplatab-mcp),
SIMPLATAB_DATA_DIR (data folder, default /data), SIMPLATAB_WORKER_PYTHON (Python of the Simplatab
pipelines, default this one), SIMPLATAB_MAX_PARALLEL (runs at the same time, default 1).
"""
import argparse
import logging
import os
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="simplatab_mcp", description="Simplatab MCP server")
    parser.add_argument("--transport", choices=["stdio", "http", "sse"], default=os.environ.get("SIMPLATAB_MCP_TRANSPORT", "stdio"))
    parser.add_argument("--host", default=os.environ.get("SIMPLATAB_MCP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SIMPLATAB_MCP_PORT", "8000")))
    parser.add_argument("--workspace", help="Folder of the experiments and uploads.")
    parser.add_argument("--data-dir", help="Folder data paths are relative to (default /data).")
    args = parser.parse_args(argv)
    if args.workspace:
        os.environ["SIMPLATAB_WORKSPACE"] = os.path.abspath(args.workspace)
    if args.data_dir:
        os.environ["SIMPLATAB_DATA_DIR"] = os.path.abspath(args.data_dir)
    # stdout carries the protocol in stdio mode: logs go to stderr
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from .server import create_server
    server = create_server()
    if args.transport == "stdio":
        server.run("stdio")
    elif args.transport == "http":
        server.run("streamable-http", host=args.host, port=args.port)
    else:
        server.run("sse", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
