"""Where things are: the Simplatab sources (Helpers/, Examples/), the workspace of the server and
the folders agents may read data from. Python 3.9 compatible (imported by the worker)."""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def simplatab_root():
    """The folder holding Helpers/ (and Examples/): SIMPLATAB_ROOT, the image's /app, or the
    repository this package sits in (mcp_server/simplatab_mcp -> repository root)."""
    candidates = [os.environ.get("SIMPLATAB_ROOT"), HERE.parent, HERE.parent.parent]
    for candidate in candidates:
        if candidate and (Path(candidate) / "Helpers" / "pipelines_main.py").exists():
            return Path(candidate).resolve()
    raise RuntimeError("The Simplatab sources (Helpers/) were not found: set SIMPLATAB_ROOT.")


def workspace():
    """Experiments, uploads and examples: SIMPLATAB_WORKSPACE, else /workspace (Docker), else
    ~/.simplatab-mcp."""
    configured = os.environ.get("SIMPLATAB_WORKSPACE")
    if configured:
        path = Path(configured)
    elif Path("/workspace").is_dir() and os.access("/workspace", os.W_OK):
        path = Path("/workspace")
    else:
        path = Path.home() / ".simplatab-mcp"
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def experiments_dir():
    path = workspace() / "experiments"
    path.mkdir(parents=True, exist_ok=True)
    return path


def uploads_dir():
    path = workspace() / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_roots():
    """Folders data paths may be given relative to, in order: SIMPLATAB_DATA_DIR (default /data,
    the mounted data volume of the Docker image), the uploads and the workspace."""
    roots = [Path(os.environ.get("SIMPLATAB_DATA_DIR", "/data")), uploads_dir(), workspace()]
    return [r for r in roots if r.is_dir()]


def _allowed(path):
    """Data must lie in a data root (the data folder, the uploads, the workspace), unless
    SIMPLATAB_ALLOW_ANY_PATH=1: a shared server does not hand its other files to agents."""
    if os.environ.get("SIMPLATAB_ALLOW_ANY_PATH", "0") == "1":
        return True
    return any(root.resolve() == path or root.resolve() in path.parents for root in data_roots())


def resolve_data_path(value):
    """An existing file or folder from an absolute path, or a path relative to a data root."""
    path = Path(os.path.expanduser(str(value)))
    if path.is_absolute():
        if not path.exists():
            raise FileNotFoundError(f"{value} does not exist (inside the server: mount it, or upload it with upload_file).")
        path = path.resolve()
        if not _allowed(path):
            raise PermissionError(f"{value} is outside the data folders of the server ({', '.join(str(r) for r in data_roots())}): "
                                  "put it in the data folder, upload it, or start the server with SIMPLATAB_ALLOW_ANY_PATH=1.")
        return path
    for root in data_roots():
        candidate = (root / path).resolve()
        if candidate.exists() and _allowed(candidate):
            return candidate
    raise FileNotFoundError(f"{value} was not found in {', '.join(str(r) for r in data_roots())}.")
