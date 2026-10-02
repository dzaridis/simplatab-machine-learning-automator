"""Runs the steps of the official nnU-Net v2 in a separate process (its paths are environment
variables read at import, and it starts its own worker processes). Usage:
    python nnunet_runner.py plan    '{"dataset": 501, "configuration": "2d", "processes": 2}'
    python nnunet_runner.py train   '{"dataset": 501, "configuration": "2d", "fold": 0, "epochs": 50,
                                      "iterations": 250, "val_iterations": 50, "device": "cuda"}'
    python nnunet_runner.py predict '{"model": "...", "fold": 0, "inputs": [[...]], "outputs": [...],
                                      "mirroring": true, "device": "cuda"}'
The nnUNet_raw, nnUNet_preprocessed and nnUNet_results environment variables must be set.
This file imports nothing from Simplatab.
"""
import json
import sys


def _compatibility():
    """nnU-Net 2.4.2 (the last version supporting numpy 1.23) with PyTorch >= 2.6:
    - its poly learning-rate scheduler passes the ``verbose`` argument PyTorch removed;
    - torch.load defaults to weights_only=True, which refuses nnU-Net checkpoints (they hold
      training metadata). Only checkpoints trained in this run are loaded."""
    import functools
    import torch
    from torch.optim.lr_scheduler import _LRScheduler
    from nnunetv2.training.lr_scheduler import polylr

    def __init__(self, optimizer, initial_lr, max_steps, exponent=0.9, current_step=None):
        self.optimizer, self.initial_lr, self.max_steps, self.exponent, self.ctr = optimizer, initial_lr, max_steps, exponent, 0
        _LRScheduler.__init__(self, optimizer, current_step if current_step is not None else -1)
    polylr.PolyLRScheduler.__init__ = __init__
    torch.load = functools.partial(torch.load, weights_only=False)


def plan(args):
    from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments, preprocess
    processes = int(args.get("processes", 2))
    extract_fingerprints([args["dataset"]], num_processes=processes, check_dataset_integrity=True)
    plan_experiments([args["dataset"]])
    preprocess([args["dataset"]], configurations=(args["configuration"],), num_processes=(processes,))


def _trainer_class(epochs, iterations, val_iterations):
    """nnUNetTrainer with a shorter schedule. The class keeps the name nnUNetTrainer, so that
    nnU-Net's predictor finds it (it only rebuilds the architecture from it)."""
    import torch
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    # nnU-Net records the constructor arguments by the names of this signature: keep its own
    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True, device=torch.device("cuda")):
        nnUNetTrainer.__init__(self, plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = epochs
        self.num_iterations_per_epoch = iterations
        self.num_val_iterations_per_epoch = val_iterations
        self.save_every = max(1, epochs)
    return type("nnUNetTrainer", (nnUNetTrainer,), {"__init__": __init__})


def train(args):
    import os
    import torch
    from batchgenerators.utilities.file_and_folder_operations import load_json
    from nnunetv2.paths import nnUNet_preprocessed
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
    name = maybe_convert_to_dataset_name(args["dataset"])
    folder = os.path.join(nnUNet_preprocessed, name)
    plans = load_json(os.path.join(folder, "nnUNetPlans.json"))
    dataset_json = load_json(os.path.join(folder, "dataset.json"))
    trainer = _trainer_class(int(args["epochs"]), int(args["iterations"]), int(args["val_iterations"]))(
        plans=plans, configuration=args["configuration"], fold=args["fold"], dataset_json=dataset_json,
        unpack_dataset=True, device=torch.device(args.get("device", "cuda")))
    trainer.disable_checkpointing = False
    trainer.run_training()


def predict(args):
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    device = torch.device(args.get("device", "cuda"))
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=bool(args.get("mirroring", True)),
                                perform_everything_on_device=device.type == "cuda", device=device, verbose=False,
                                allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(args["model"], use_folds=(args["fold"],),
                                                   checkpoint_name="checkpoint_final.pth")
    predictor.predict_from_files(args["inputs"], args["outputs"], save_probabilities=True, overwrite=True,
                                 num_processes_preprocessing=1, num_processes_segmentation_export=1)


if __name__ == "__main__":
    command, arguments = sys.argv[1], json.loads(sys.argv[2])
    _compatibility()
    {"plan": plan, "train": train, "predict": predict}[command](arguments)
