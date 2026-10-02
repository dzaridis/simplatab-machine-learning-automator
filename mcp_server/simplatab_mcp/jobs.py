"""Experiments of the MCP server: their folders, the worker subprocesses and the run queue.

An experiment lives in <workspace>/experiments/<id>/: experiment.json (what the agent asked and
the state), input/ (the data), summary.json (the data check), Materials/ (the outputs of the
pipeline), run.log, status.json (live progress, written by the worker) and results.json.
States: preparing -> ready (or invalid) -> queued -> running -> completed | failed | cancelled.
Runs are queued and started in order, at most SIMPLATAB_MAX_PARALLEL at a time (default 1: the
pipelines use the whole GPU/CPU).
"""
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

from . import paths

PACKAGE_PARENT = Path(__file__).resolve().parent.parent
ACTIVE = ("queued", "running")


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:  # a finished child not yet reaped is a zombie
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split()[2] != "Z"
    except OSError:
        return True


def _write(path, data):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1, default=str)
    os.replace(tmp, path)


def _read(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


class WorkerError(RuntimeError):
    pass


class Experiments:
    def __init__(self, worker_python=None, max_parallel=None):
        self.python = worker_python or os.environ.get("SIMPLATAB_WORKER_PYTHON") or sys.executable
        self.max_parallel = max(1, int(max_parallel or os.environ.get("SIMPLATAB_MAX_PARALLEL", "1")))
        self.root = paths.experiments_dir()
        self._lock = threading.RLock()
        self._procs = {}
        self._cache = {}
        self._recover()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._schedule, daemon=True, name="simplatab-scheduler")
        self._thread.start()

    # ---- worker ---------------------------------------------------------------------------
    def _env(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(PACKAGE_PARENT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        env["SIMPLATAB_ROOT"] = str(paths.simplatab_root())
        env["SIMPLATAB_WORKSPACE"] = str(paths.workspace())
        env.setdefault("MPLBACKEND", "Agg")
        env.setdefault("PYTHONUNBUFFERED", "1")
        return env

    def _command(self, command, output, **options):
        args = [self.python, "-m", "simplatab_mcp.worker", command, "--output", str(output)]
        for key, value in options.items():
            if value is not None:
                args += [f"--{key}", value if isinstance(value, str) else json.dumps(value)]
        return args

    def worker(self, command, timeout=600, **options):
        """Runs a short worker command and returns its JSON result (raises WorkerError)."""
        with tempfile.TemporaryDirectory(prefix="simplatab-mcp-") as tmp:
            output = Path(tmp) / "result.json"
            try:
                run = subprocess.run(self._command(command, output, **options), env=self._env(), cwd=tmp,
                                     capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                raise WorkerError(f"The {command} step took longer than {timeout} s.")
            result = _read(output)
            if result is None:
                raise WorkerError(f"The {command} step failed: {(run.stderr or run.stdout)[-2000:]}")
        if "error" in result:
            raise WorkerError(result["error"])
        return result

    def cached(self, key, command, **options):
        if key not in self._cache:
            self._cache[key] = self.worker(command, **options)
        return self._cache[key]

    # ---- experiments ----------------------------------------------------------------------
    def folder(self, experiment_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", experiment_id or ""):
            raise KeyError(f"Unknown experiment {experiment_id!r}.")
        folder = self.root / experiment_id
        if not (folder / "experiment.json").exists():
            raise KeyError(f"Unknown experiment {experiment_id!r}: see list_experiments.")
        return folder

    def meta(self, experiment_id):
        return _read(self.folder(experiment_id) / "experiment.json", {})

    def _save(self, experiment_id, **changes):
        with self._lock:
            path = self.root / experiment_id / "experiment.json"
            meta = _read(path, {})
            meta.update(changes)
            _write(path, meta)
            return meta

    def create(self, automator, train, test, name=None):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        experiment_id = f"{stamp}-{automator.split('-')[0]}-{uuid.uuid4().hex[:6]}"
        folder = self.root / experiment_id
        folder.mkdir(parents=True)
        _write(folder / "experiment.json", {"id": experiment_id, "name": name or experiment_id, "automator": automator,
                                            "train": train, "test": test, "state": "preparing", "created": _now()})
        try:
            prepared = self.worker("prepare", timeout=3600, experiment=str(folder))
        except WorkerError as e:
            self._save(experiment_id, state="invalid", errors=[str(e)])
            return {"experiment_id": experiment_id, "state": "invalid", "errors": [str(e)], "warnings": [],
                    "summary": {}, "default_config": None}
        state = "invalid" if prepared["errors"] else "ready"
        self._save(experiment_id, state=state, errors=prepared["errors"], warnings=prepared["warnings"],
                   sources=prepared["sources"], default_config=prepared["default_config"])
        return {"experiment_id": experiment_id, "state": state, "errors": prepared["errors"],
                "warnings": prepared["warnings"], "summary": prepared["summary"], "default_config": prepared["default_config"]}

    def start(self, experiment_id, config=None, dry_run=False):
        meta = self.meta(experiment_id)
        if meta["state"] in ACTIVE:
            raise ValueError(f"Experiment {experiment_id} is {meta['state']}: cancel it first.")
        if meta["state"] in ("invalid", "preparing"):
            raise ValueError(f"Experiment {experiment_id} has no valid data ({'; '.join(meta.get('errors', [])) or meta['state']}).")
        folder = self.folder(experiment_id)
        configured = self.worker("configure", timeout=600, experiment=str(folder), config=config or {})
        if dry_run:
            return {"experiment_id": experiment_id, "state": meta["state"], "dry_run": True, **configured}
        for name in ("Materials", "status.json", "results.json", "run.log", "run_result.json"):
            path = folder / name
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists():
                path.unlink()
        meta = self._save(experiment_id, state="queued", config=config or {}, params=configured["params"],
                          model_names=configured["model_names"], queued=_now(), started=None, finished=None,
                          message="", pid=None)
        return {"experiment_id": experiment_id, "state": "queued", "models": configured["model_names"],
                "queue_position": self._position(experiment_id), "params": configured["params"]}

    def cancel(self, experiment_id):
        meta = self.meta(experiment_id)
        if meta["state"] not in ACTIVE:
            return {"experiment_id": experiment_id, "state": meta["state"], "message": "Not running."}
        with self._lock:
            pid = meta.get("pid")
            if meta["state"] == "running" and _alive(pid):
                try:
                    os.killpg(pid, signal.SIGTERM)
                except OSError:
                    pass
                for _ in range(50):
                    if not _alive(pid):
                        break
                    time.sleep(0.2)
                if _alive(pid):
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except OSError:
                        pass
            proc = self._procs.pop(experiment_id, None)
            if proc is not None:
                proc.poll()
            self._save(experiment_id, state="cancelled", finished=_now(), message="Cancelled.")
        return {"experiment_id": experiment_id, "state": "cancelled"}

    def delete(self, experiment_id):
        meta = self.meta(experiment_id)
        if meta["state"] in ACTIVE:
            raise ValueError(f"Experiment {experiment_id} is {meta['state']}: cancel it first.")
        shutil.rmtree(self.folder(experiment_id))
        return {"experiment_id": experiment_id, "deleted": True}

    def _position(self, experiment_id):
        queued = sorted((m for m in self._all() if m["state"] == "queued"), key=lambda m: m.get("queued") or "")
        ids = [m["id"] for m in queued]
        return ids.index(experiment_id) + 1 if experiment_id in ids else None

    def _all(self):
        out = []
        for path in sorted(self.root.glob("*/experiment.json")):
            meta = _read(path)
            if meta:
                out.append(meta)
        return out

    def list(self):
        rows = []
        for meta in self._all():
            status = _read(self.root / meta["id"] / "status.json", {})
            rows.append({k: meta.get(k) for k in ("id", "name", "automator", "state", "created", "started", "finished")}
                        | {"progress": status.get("progress"), "message": meta.get("message") or status.get("message")})
        return rows

    def status(self, experiment_id, log_lines=20):
        meta = self.meta(experiment_id)
        folder = self.folder(experiment_id)
        status = _read(folder / "status.json", {})
        out = {k: meta.get(k) for k in ("id", "name", "automator", "state", "created", "queued", "started", "finished",
                                        "errors", "warnings", "model_names", "config")}
        out["message"] = meta.get("message") or status.get("message", "")
        if meta["state"] == "queued":
            out["queue_position"] = self._position(experiment_id)
        for key in ("phase", "phase_name", "progress", "elapsed_seconds", "models"):
            if key in status:
                out[key] = status[key]
        out["log_tail"] = self.log(experiment_id, log_lines)["lines"] if (folder / "run.log").exists() else []
        out["results_available"] = (folder / "results.json").exists()
        return out

    def log(self, experiment_id, tail=200):
        path = self.folder(experiment_id) / "run.log"
        if not path.exists():
            return {"lines": [], "path": str(path)}
        with open(path, errors="replace") as f:
            lines = [line.rstrip("\n") for line in f if line.strip() and set(line.strip()) - {"-"}]
        return {"lines": lines[-max(1, int(tail)):], "total_lines": len(lines), "path": str(path)}

    def results(self, experiment_id):
        folder = self.folder(experiment_id)
        meta = self.meta(experiment_id)
        if meta["state"] in ACTIVE:
            raise ValueError(f"Experiment {experiment_id} is {meta['state']}: its results are not ready.")
        data = _read(folder / "results.json")
        if data is None:
            data = self.worker("results", experiment=str(folder))
        data["state"], data["message"] = meta["state"], meta.get("message", "")
        data["experiment_id"] = experiment_id
        return data

    def file_path(self, experiment_id, relative):
        """A file of the experiment's Materials folder (or run.log), refusing paths outside it."""
        folder = self.folder(experiment_id)
        if relative in ("run.log", "status.json", "summary.json", "experiment.json"):
            return folder / relative
        base = (folder / "Materials").resolve()
        target = (base / relative).resolve()
        if base not in target.parents and target != base:
            raise ValueError("The path must be inside the Materials folder of the experiment.")
        if not target.is_file():
            raise FileNotFoundError(f"{relative} is not a file of the experiment (see get_results: files).")
        return target

    # ---- scheduling -----------------------------------------------------------------------
    def _recover(self):
        """Runs that were going when the server stopped: still alive (another server process) or lost."""
        for meta in self._all():
            if meta["state"] == "running" and not _alive(meta.get("pid")):
                status = _read(self.root / meta["id"] / "status.json", {})
                final = status.get("state") if status.get("state") in ("completed", "failed") else "failed"
                self._save(meta["id"], state=final, finished=meta.get("finished") or _now(),
                           message=status.get("message") or "Interrupted: the server stopped during the run.")

    def _launch(self, meta):
        folder = self.root / meta["id"]
        log = open(folder / "run.log", "a")
        proc = subprocess.Popen(self._command("run", folder / "run_result.json", experiment=str(folder)), cwd=str(folder),
                                env=self._env(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        log.close()
        self._procs[meta["id"]] = proc
        self._save(meta["id"], state="running", started=_now(), pid=proc.pid)

    def _finished(self, experiment_id):
        folder = self.root / experiment_id
        status = _read(folder / "status.json", {})
        result = _read(folder / "run_result.json", {})
        state = status.get("state") if status.get("state") in ("completed", "failed") else "failed"
        message = status.get("message") or result.get("message") or result.get("error")
        if not message:
            tail = self.log(experiment_id, 5)["lines"]
            message = "The run stopped unexpectedly: " + (" | ".join(tail) or "no output")
        self._save(experiment_id, state=state, finished=_now(), message=message)

    def tick(self):
        with self._lock:
            running = 0
            for meta in self._all():
                if meta["state"] != "running":
                    continue
                proc = self._procs.get(meta["id"])
                done = proc.poll() is not None if proc is not None else not _alive(meta.get("pid"))
                if done:
                    self._procs.pop(meta["id"], None)
                    self._finished(meta["id"])
                else:
                    running += 1
            queued = sorted((m for m in self._all() if m["state"] == "queued"), key=lambda m: m.get("queued") or "")
            for meta in queued[:max(0, self.max_parallel - running)]:
                try:
                    self._launch(meta)
                except Exception as e:
                    self._save(meta["id"], state="failed", finished=_now(), message=f"The run could not start: {e}")

    def _schedule(self):
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                pass
            self._stop.wait(1.0)

    def close(self):
        self._stop.set()
