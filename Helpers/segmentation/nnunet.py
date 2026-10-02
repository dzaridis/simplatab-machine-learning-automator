"""The official nnU-Net v2 inside the automator.

The cached cases (already in the nnU-Net naming) form a raw dataset; nnU-Net extracts its
fingerprint, plans its configuration (2d or 3d_fullres) and preprocesses the data once. Each
network training uses the automator's folds (written to splits_final.json) with a shortened
schedule (nnU-Net's default is 1000 epochs of 250 iterations), or fold "all" for the final model.
Predictions keep the probabilities, so that nnU-Net gets the same metrics and uncertainty maps as
the other networks. Each step runs in a subprocess (nnunet_runner.py).
"""
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile

import numpy as np

RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nnunet_runner.py")
DATASET_ID = 501
DATASET = f"Dataset{DATASET_ID:03d}_Simplatab"
MODE_NAMES = {"ct": "CT", "rgb": "rgb_to_0_1", "zscore": "zscore"}
LOG_PATTERN = re.compile(r"(Epoch \d+|train_loss|val_loss|Pseudo dice|Epoch time|Error|error|Traceback)")


class NNUNet:
    def __init__(self, workdir, configuration, log=print):
        self.workdir, self.configuration, self.log = workdir, configuration, log
        self.raw = os.path.join(workdir, "raw")
        self.preprocessed = os.path.join(workdir, "preprocessed")
        self.results = os.path.join(workdir, "results")

    # ---- environment and subprocess ---------------------------------------------------------
    def _env(self):
        import torch
        env = dict(os.environ, nnUNet_raw=self.raw, nnUNet_preprocessed=self.preprocessed, nnUNet_results=self.results)
        # data augmentation workers (also the planner's thread count: at least 1)
        env.setdefault("nnUNet_n_proc_DA", str(min(4, max(1, (os.cpu_count() or 2) // 2)) if torch.cuda.is_available() else 1))
        env.setdefault("nnUNet_compile", "f")
        env.pop("PYTHONPATH", None)
        return env

    @staticmethod
    def device():
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _run(self, command, args, label=""):
        process = subprocess.Popen([sys.executable, RUNNER, command, json.dumps(args)], env=self._env(), cwd=self.workdir,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        tail = []
        for line in process.stdout:
            line = line.rstrip()
            tail = (tail + [line])[-30:]
            if LOG_PATTERN.search(line):
                self.log(f"{label} nnU-Net: {line.split(': ', 1)[-1] if ': ' in line and line[:4].isdigit() else line}".strip())
        if process.wait() != 0:
            raise RuntimeError("nnU-Net failed: " + " | ".join(l for l in tail[-6:] if l))

    # ---- dataset ----------------------------------------------------------------------------
    def write_dataset(self, cache_folder, metas, modes, classes):
        """The raw dataset from the cached training cases (copied: the cache stays untouched)."""
        folder = os.path.join(self.raw, DATASET)
        shutil.rmtree(folder, ignore_errors=True)
        for sub in ("imagesTr", "labelsTr"):
            os.makedirs(os.path.join(folder, sub))
        for meta in metas:
            for c in range(len(modes)):
                name = f"{meta['name']}_{c:04d}.nii.gz"
                shutil.copy(os.path.join(cache_folder, name), os.path.join(folder, "imagesTr", name))
            shutil.copy(os.path.join(cache_folder, f"{meta['name']}.nii.gz"),
                        os.path.join(folder, "labelsTr", f"{meta['name']}.nii.gz"))
        labels, used = {}, set()
        for i, name in enumerate(classes):
            key = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name)).strip("_") or f"class_{i}"
            while key in used:
                key += f"_{i}"
            used.add(key)
            labels["background" if i == 0 else key] = i
        with open(os.path.join(folder, "dataset.json"), "w") as f:
            json.dump({"channel_names": {str(c): MODE_NAMES[m] for c, m in enumerate(modes)}, "labels": labels,
                       "numTraining": len(metas), "file_ending": ".nii.gz"}, f, indent=2)

    def plan(self):
        self.log(f"nnU-Net: planning and preprocessing the {self.configuration} configuration")
        self._run("plan", {"dataset": DATASET_ID, "configuration": self.configuration,
                           "processes": max(1, min(4, os.cpu_count() or 1))})

    def set_splits(self, splits):
        """splits: [(train case names, validation case names)] for folds 0..n-1."""
        with open(os.path.join(self.preprocessed, DATASET, "splits_final.json"), "w") as f:
            json.dump([{"train": list(t), "val": list(v)} for t, v in splits], f, indent=2)

    @property
    def model_folder(self):
        return os.path.join(self.results, DATASET, f"nnUNetTrainer__nnUNetPlans__{self.configuration}")

    # ---- training and prediction ------------------------------------------------------------
    def train(self, fold, settings, label=""):
        self._run("train", {"dataset": DATASET_ID, "configuration": self.configuration, "fold": fold,
                            "epochs": int(settings["nnunet_epochs"]), "iterations": int(settings["nnunet_iterations"]),
                            "val_iterations": max(1, int(settings["nnunet_iterations"]) // 5), "device": self.device()},
                  label=label)

    def predict(self, fold, cache_folder, metas, channels, tta=True, label=""):
        """{case name: probabilities (K + 1, z, y, x)} on the cached grid of the cases."""
        out = os.path.join(self.workdir, "predictions", str(fold))
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(out)
        inputs = [[os.path.join(cache_folder, f"{m['name']}_{c:04d}.nii.gz") for c in range(channels)] for m in metas]
        outputs = [os.path.join(out, m["name"]) for m in metas]
        self._run("predict", {"model": self.model_folder, "fold": fold, "inputs": inputs, "outputs": outputs,
                              "mirroring": bool(tta), "device": self.device()}, label=label)
        probabilities = {}
        for meta in metas:
            with np.load(os.path.join(out, f"{meta['name']}.npz")) as data:
                p = data["probabilities"].astype(np.float32)
            probabilities[meta["name"]] = p.reshape((p.shape[0],) + tuple(meta["shape"]))
        return probabilities

    def export(self, fold, path):
        """A zip of the trained model (plans, dataset description, the fold's final checkpoint),
        usable with nnUNetv2_predict / nnUNetPredictor. The checkpoint keeps what prediction needs:
        the optimizer and gradient-scaler states (two thirds of its size) are left out."""
        import io
        import torch
        folder = self.model_folder
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
            for name in ("dataset.json", "plans.json", "dataset_fingerprint.json"):
                if os.path.exists(os.path.join(folder, name)):
                    archive.write(os.path.join(folder, name), name)
            # Trained in this run: the checkpoint holds training metadata, refused by weights_only
            checkpoint = torch.load(os.path.join(folder, f"fold_{fold}", "checkpoint_final.pth"), map_location="cpu",
                                    weights_only=False)
            for key in ("optimizer_state", "grad_scaler_state"):
                checkpoint[key] = None
            buffer = io.BytesIO()
            torch.save(checkpoint, buffer)
            archive.writestr(f"fold_{fold}/checkpoint_final.pth", buffer.getvalue())
