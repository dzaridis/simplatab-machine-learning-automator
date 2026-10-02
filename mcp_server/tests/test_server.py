"""Tests of the MCP server through an MCP client (in process): the tools, uploads, data errors,
configuration errors, a full tabular experiment, its results and files, cancellation and deletion.

Needs the MCP SDK (Python >= 3.10); the worker runs in SIMPLATAB_WORKER_PYTHON (the Python of the
Simplatab pipelines; default: this one):
    cd mcp_server && SIMPLATAB_WORKER_PYTHON=/path/to/simplatab/python python -m unittest tests.test_server
"""
import asyncio
import base64
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("SIMPLATAB_PRETRAINED", "0")

try:
    from mcp import Client
except ImportError:  # the Python environment of the pipelines (3.9) has no MCP SDK
    Client = None


def _data(result):
    if result.is_error:
        raise AssertionError(result.content[0].text)
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(result.content[0].text)


@unittest.skipIf(Client is None, "the MCP SDK (Python >= 3.10) is not installed")
class TestServer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self._saved = {k: os.environ.get(k) for k in ("SIMPLATAB_WORKSPACE", "SIMPLATAB_DATA_DIR")}
        os.environ["SIMPLATAB_WORKSPACE"] = str(self.dir / "ws")
        os.environ["SIMPLATAB_DATA_DIR"] = str(self.dir / "data")
        from simplatab_mcp.jobs import Experiments
        from simplatab_mcp.server import create_server
        self.experiments = Experiments(worker_python=os.environ.get("SIMPLATAB_WORKER_PYTHON"))
        self.server = create_server(self.experiments)

    def tearDown(self):
        self.experiments.close()
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()

    def session(self, steps):
        async def run():
            async with Client(self.server) as client:
                return await steps(client)
        return asyncio.run(run())

    async def wait(self, client, experiment_id, timeout=900):
        start = time.time()
        while time.time() - start < timeout:
            status = _data(await client.call_tool("get_experiment", {"experiment_id": experiment_id}))
            if status["state"] not in ("queued", "running"):
                return status
            await asyncio.sleep(2)
        raise AssertionError("the experiment did not finish")

    def test_discovery(self):
        async def steps(client):
            names = {t.name for t in (await client.list_tools()).tools}
            self.assertTrue({"list_automators", "get_data_contract", "create_experiment", "start_experiment", "run_experiment",
                             "get_experiment", "get_results", "read_result_file", "cancel_experiment", "upload_file"} <= names)
            automators = _data(await client.call_tool("list_automators", {}))["automators"]
            self.assertEqual(len(automators), 5)
            contract = _data(await client.call_tool("get_data_contract", {"automator": "image-segmentation", "dim": 3}))
            self.assertEqual(list(contract["models_by_dim"]), ["3d"])
            self.assertIn("nnunet_3d", [m["key"] for m in contract["models_by_dim"]["3d"]])
            bad = await client.call_tool("get_data_contract", {"automator": "regression"})
            self.assertTrue(bad.is_error and "Unknown automator" in bad.content[0].text)
            resource = await client.read_resource("simplatab://contracts/tabular")
            self.assertIn("Target", resource.contents[0].text)
            info = _data(await client.call_tool("get_server_info", {}))
            self.assertIn("gpu", info)
        self.session(steps)

    def test_upload_errors_and_dry_run(self):
        async def steps(client):
            csv = "ID,x,Target\n" + "".join(f"P{i},{i % 7},{i % 2}\n" for i in range(40))
            half = len(csv) // 2
            first = _data(await client.call_tool("upload_file", {"filename": "Train.csv",
                                                                  "content_base64": base64.b64encode(csv[:half].encode()).decode()}))
            second = _data(await client.call_tool("upload_file", {"filename": "Train.csv", "append": True,
                                                                   "content_base64": base64.b64encode(csv[half:].encode()).decode()}))
            self.assertEqual(second["size"], len(csv))
            self.assertEqual(first["path"], second["path"])
            await client.call_tool("upload_file", {"filename": "Test.csv", "content_base64": base64.b64encode(csv.encode()).decode()})
            contained = _data(await client.call_tool("upload_file", {"filename": "../x.csv", "content_base64": ""}))
            self.assertEqual(Path(contained["path"]), self.dir / "ws" / "uploads" / "x.csv")  # folders are dropped
            unsafe = await client.call_tool("upload_file", {"filename": "x?.csv", "content_base64": ""})
            self.assertTrue(unsafe.is_error)

            missing = await client.call_tool("create_experiment", {"automator": "tabular", "train": "nope.csv", "test": "Test.csv"})
            self.assertEqual(_data(missing)["state"], "invalid")
            created = _data(await client.call_tool("create_experiment", {"automator": "tabular", "train": "Train.csv",
                                                                         "test": "Test.csv", "name": "uploaded"}))
            self.assertEqual(created["state"], "ready", created)
            self.assertEqual(created["summary"]["class_counts"], {"0": 20, "1": 20})
            bad = await client.call_tool("start_experiment", {"experiment_id": created["experiment_id"], "config": {"k_folds": 99}})
            self.assertTrue(bad.is_error and "between 2 and 20" in bad.content[0].text)
            dry = _data(await client.call_tool("start_experiment", {"experiment_id": created["experiment_id"],
                                                                    "config": {"models": ["xgboost"]}, "dry_run": True}))
            self.assertEqual(dry["model_names"], ["XGBoost"])
            self.assertEqual(_data(await client.call_tool("get_experiment", {"experiment_id": created["experiment_id"]}))["state"], "ready")
            unknown = await client.call_tool("get_experiment", {"experiment_id": "../../etc"})
            self.assertTrue(unknown.is_error)
        self.session(steps)

    def test_full_tabular_experiment(self):
        async def steps(client):
            example = _data(await client.call_tool("get_example_data", {"automator": "tabular"}))
            config = {"models": ["logistic_regression", "decision_trees"], "k_folds": 3, "hyperparameter_search": "none"}
            started = _data(await client.call_tool("run_experiment", {"automator": "tabular", "train": example["train"],
                                                                      "test": example["test"], "config": config}))
            self.assertEqual(started["state"], "queued")
            status = await self.wait(client, started["experiment_id"])
            self.assertEqual(status["state"], "completed", status)
            self.assertEqual(status["progress"], 100)
            self.assertEqual({m["test"] for m in status["models"]}, {"done"})
            results = _data(await client.call_tool("get_results", {"experiment_id": started["experiment_id"]}))
            self.assertIn(results["best_model"], ("Logistic Regression", "Decision Trees"))
            self.assertEqual(len(results["validation_metrics"]), 2)
            self.assertEqual(results["validation_file"], "3_fold_results.xlsx")
            self.assertEqual(len(results["splits"]["folds"]), 3)
            text = await client.call_tool("read_result_file", {"experiment_id": started["experiment_id"], "path": "Splits/splits.csv"})
            self.assertTrue(text.content[0].text.startswith("fold,set,id,class,row"))
            figure = next(f["path"] for f in results["files"] if f["path"].endswith(".png"))
            image = await client.call_tool("read_result_file", {"experiment_id": started["experiment_id"], "path": figure})
            self.assertEqual(image.content[0].type, "image")
            escape = await client.call_tool("read_result_file", {"experiment_id": started["experiment_id"], "path": "../experiment.json"})
            self.assertTrue(escape.is_error)
            log = _data(await client.call_tool("get_log", {"experiment_id": started["experiment_id"], "tail": 5}))
            self.assertIn("Pipeline completed successfully", " ".join(log["lines"]))
            listed = _data(await client.call_tool("list_experiments", {}))["experiments"]
            self.assertEqual([e["state"] for e in listed], ["completed"])
            # Started again with another configuration: the previous results are replaced
            again = _data(await client.call_tool("start_experiment", {"experiment_id": started["experiment_id"],
                                                                      "config": dict(config, models=["logistic_regression"])}))
            self.assertEqual(again["models"], ["Logistic Regression"])
            cancelled = _data(await client.call_tool("cancel_experiment", {"experiment_id": started["experiment_id"]}))
            self.assertEqual(cancelled["state"], "cancelled")
            await asyncio.sleep(2)
            self.assertEqual(_data(await client.call_tool("get_experiment", {"experiment_id": started["experiment_id"]}))["state"],
                             "cancelled")
            deleted = _data(await client.call_tool("delete_experiment", {"experiment_id": started["experiment_id"]}))
            self.assertTrue(deleted["deleted"])
            self.assertEqual(_data(await client.call_tool("list_experiments", {}))["experiments"], [])
        self.session(steps)


if __name__ == "__main__":
    unittest.main()
