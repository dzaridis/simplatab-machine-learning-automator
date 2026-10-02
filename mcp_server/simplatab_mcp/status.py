"""Live status of a run, written to <experiment>/status.json from the lines the pipelines print
("<model> is starting", "Training on K-Fold cross validation", ...), as web/jobs.py does for the
web interface. Python 3.9 compatible."""
import json
import os
import re
import threading
import time
from collections import deque

PHASES = {
    "tabular": ["bias", "data", "kfold", "test", "done"],
    "time-series-forecasting": ["data", "kfold", "test", "done"],
}
DEFAULT_PHASES = ["prep", "kfold", "test", "done"]
PHASE_NAMES = {"bias": "bias assessment", "data": "preparing data", "prep": "preparing images",
               "kfold": "validation", "test": "external test", "done": "report"}

_STARTING = re.compile(r"^(.+?) is starting$")
_COMPLETED = re.compile(r"^(.+?) is completed successfully$")
_SKIPPED = re.compile(r"^(.+?) failed and was skipped: (.*)$")


def write_json(path, data):
    """Atomic write (readers never see a partial file)."""
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1, default=str)
    os.replace(tmp, path)


class Status:
    def __init__(self, path, automator, models, log_lines=60):
        self.path = path
        self.automator = automator
        self.phases = PHASES.get(automator, DEFAULT_PHASES)
        self.phase = self.phases[0] if automator != "tabular" else "data"
        self.models = {name: {"validation": "pending", "test": "pending", "note": ""} for name in models}
        self.log = deque(maxlen=log_lines)
        self.state, self.message = "running", ""
        self.started = time.time()
        self.finished = None
        self._lock = threading.Lock()
        self._last_write = 0.0
        self.write(force=True)

    def line(self, text):
        text = text.strip()
        if not text or set(text) <= {"-"}:
            return
        with self._lock:
            self.log.append(text[:500])
            self._parse(text)
        self.write()

    def _parse(self, text):
        if "Bias Detection Started" in text:
            self.phase = "bias"
        elif text.startswith(("Loading Data", "Preparing data")):
            self.phase = "data"
        elif text.startswith("Preparing images"):
            self.phase = "prep"
        elif text.startswith(("Training on K-Fold cross validation", "Training on hold-out validation")) and "completed" not in text:
            self.phase = "kfold"
        elif text.startswith("Evaluating algorithms on"):
            self.phase = "test"
        elif self.phase in ("kfold", "test"):
            column = "validation" if self.phase == "kfold" else "test"
            for pattern, status in ((_STARTING, "running"), (_COMPLETED, "done")):
                match = pattern.match(text)
                if match and match.group(1) in self.models:
                    self.models[match.group(1)][column] = status
                    return
            match = _SKIPPED.match(text)
            if match and match.group(1) in self.models:
                model = self.models[match.group(1)]
                model["validation"] = model["test"] = "skipped"
                model["note"] = match.group(2)

    def finish(self, state, message):
        with self._lock:
            self.state, self.message = state, message
            self.finished = time.time()
            if state == "completed":
                self.phase = "done"
        self.write(force=True)

    def snapshot(self):
        units = 2 * len(self.models) or 1
        finished = sum(s in ("done", "skipped") for m in self.models.values() for s in (m["validation"], m["test"]))
        progress = 100 if self.state == "completed" else min(99, int(100 * finished / units))
        return {"state": self.state, "message": self.message, "phase": self.phase,
                "phase_name": PHASE_NAMES.get(self.phase, self.phase), "phases": self.phases, "progress": progress,
                "elapsed_seconds": int((self.finished or time.time()) - self.started),
                "models": [dict(name=n, **s) for n, s in self.models.items()], "log_tail": list(self.log)[-20:],
                "pid": os.getpid(), "updated": time.time()}

    def write(self, force=False):
        now = time.time()
        if not force and now - self._last_write < 1.0:
            return
        with self._lock:
            data = self.snapshot()
        self._last_write = now
        write_json(self.path, data)


class Tee:
    """Duplicates a stream (the run log) and hands every complete line to a callback."""

    def __init__(self, stream, on_line):
        self._stream, self._on_line, self._buffer = stream, on_line, ""

    def write(self, text):
        self._stream.write(text)
        self._buffer += text
        *lines, self._buffer = self._buffer.split("\n")
        for line in lines:
            self._on_line(line)
        return len(text)

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)
