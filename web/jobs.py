"""Runs the machine learning pipeline in the background and reports its progress.

The pipeline prints its progress ("<model> is starting", "<model> is completed
successfully", ...). While a job runs, the standard output is duplicated into the job,
which turns these lines into a live status for the web interface.
"""
import re
import sys
import threading
import time
from collections import deque

PHASES = [
    ("bias", "Bias assessment"),
    ("data", "Loading data"),
    ("kfold", "K-fold training"),
    ("test", "External test"),
    ("done", "Report"),
]

IMAGE_PHASES = [
    ("prep", "Preparing images"),
    ("kfold", "K-fold training"),
    ("test", "External test"),
    ("done", "Report"),
]

DETECTION_PHASES = [
    ("prep", "Preparing images"),
    ("kfold", "Validation"),
    ("test", "External test"),
    ("done", "Report"),
]

SEGMENTATION_PHASES = [
    ("prep", "Preparing images"),
    ("kfold", "Validation"),
    ("test", "External test"),
    ("done", "Report"),
]

FORECAST_PHASES = [
    ("data", "Preparing data"),
    ("kfold", "Rolling-origin validation"),
    ("test", "External test"),
    ("done", "Report"),
]

_STARTING = re.compile(r"^(.+?) is starting$")
_COMPLETED = re.compile(r"^(.+?) is completed successfully$")
_SKIPPED = re.compile(r"^(.+?) failed and was skipped: (.*)$")


class _Tee:
    """Writes to the original stream and hands every complete line to a callback."""

    def __init__(self, stream, on_line):
        self._stream = stream
        self._on_line = on_line
        self._buffer = ""

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


class PipelineJob:
    """A single pipeline run at a time, whatever the automator (the tabular pipeline keeps
    module-level state, and both write their outputs to the Materials folder)."""

    LOG_LINES = 400

    def __init__(self):
        self._lock = threading.Lock()
        self._reset(models=[])
        self.state = "idle"

    def _reset(self, models, automator="tabular", phases=PHASES, initial_phase="data"):
        self.automator = automator
        self.phases = list(phases)
        self.state = "running"
        self.message = ""
        self.started_at = time.time()
        self.finished_at = None
        self.phase = initial_phase
        self.models = {name: {"kfold": "pending", "test": "pending", "note": ""} for name in models}
        self.log = deque(maxlen=self.LOG_LINES)

    @property
    def running(self):
        return self.state == "running"

    def start(self, target, models, automator="tabular", phases=PHASES, initial_phase="data"):
        """Run ``target()`` in a background thread. Returns False if a job is already running."""
        with self._lock:
            if self.running:
                return False
            self._reset(models, automator, phases, initial_phase)
        threading.Thread(target=self._run, args=(target,), daemon=True).start()
        return True

    def _run(self, target):
        original = sys.stdout
        sys.stdout = _Tee(original, self._on_line)
        try:
            result = target()
            if isinstance(result, str) and result.startswith("Error"):
                self.state, self.message = "error", result
            else:
                self.state, self.message = "done", result or "Pipeline completed successfully"
                self.phase = "done"
        except Exception as e:  # run_pipeline catches its errors; this is a safety net
            self.state, self.message = "error", f"Error: {e}"
        finally:
            sys.stdout = original
            self.finished_at = time.time()

    def _on_line(self, line):
        text = line.strip()
        if not text or set(text) <= {"-"}:
            return
        with self._lock:
            self.log.append(text)
            self._parse(text)

    def _parse(self, text):
        if "Bias Detection Started" in text:
            self.phase = "bias"
        elif text.startswith("Loading Data"):
            self.phase = "data"
        elif text.startswith("Preparing images"):
            self.phase = "prep"
        elif text.startswith(("Training on K-Fold cross validation", "Training on hold-out validation")) and "completed" not in text:
            self.phase = "kfold"
        elif text.startswith("Evaluating algorithms on"):
            self.phase = "test"
        elif self.phase in ("kfold", "test"):
            for pattern, status in ((_STARTING, "running"), (_COMPLETED, "done")):
                match = pattern.match(text)
                if match and match.group(1) in self.models:
                    self.models[match.group(1)][self.phase] = status
                    return
            match = _SKIPPED.match(text)
            if match and match.group(1) in self.models:
                model = self.models[match.group(1)]
                model["kfold"] = model["test"] = "skipped"
                model["note"] = match.group(2)

    def snapshot(self, log_lines=150):
        with self._lock:
            units = 2 * len(self.models) or 1
            finished = sum(status in ("done", "skipped")
                           for model in self.models.values() for status in (model["kfold"], model["test"]))
            progress = 100 if self.state == "done" else min(99, int(100 * finished / units))
            end = self.finished_at or time.time()
            return {
                "state": self.state,
                "automator": self.automator,
                "phases": self.phases,
                "message": self.message,
                "phase": self.phase,
                "progress": progress if self.state != "idle" else 0,
                "elapsed": int(end - self.started_at) if self.state != "idle" else 0,
                "models": [dict(name=name, **status) for name, status in self.models.items()],
                "log": list(self.log)[-log_lines:],
            }
