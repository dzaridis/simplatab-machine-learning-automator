"""Simplatab MCP server: the Simplatab automators as tools for AI agents.

The package has two halves:
- the server (server.py, jobs.py, __main__.py): the MCP protocol and the experiment queue. It
  needs the ``mcp`` SDK (Python >= 3.10) and imports nothing from Simplatab;
- the worker (worker.py and the modules it imports): data contracts, data checks, configuration,
  runs and results. It runs in the Python environment of Simplatab (the same pipelines as the web
  application) as a subprocess of the server, so that a long training never blocks the server,
  can use the GPU, and can be cancelled.
"""
__version__ = "1.0.0"
