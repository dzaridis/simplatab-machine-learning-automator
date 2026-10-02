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
            self.assertEqual(len(automators), 7)
            self.assertIn("clustering", [a["id"] for a in automators])
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

    def test_inspect_and_auto(self):
        async def steps(client):
            example = _data(await client.call_tool("get_example_data", {"automator": "tabular"}))
            found = _data(await client.call_tool("inspect_data", {"train": example["train"]}))
            self.assertEqual(found["suggested_automator"], "tabular")
            # A wrong configuration: the experiment is created (data fine) and the error names the field
            run = _data(await client.call_tool("run_experiment", {"automator": "auto", "train": example["train"],
                                                                  "test": example["test"], "config": {"models": ["gbm"]}}))
            self.assertEqual((run["automator"], run["state"]), ("tabular", "ready"))
            self.assertIn("Unknown model", run["config_error"])
            started = _data(await client.call_tool("start_experiment", {
                "experiment_id": run["experiment_id"],
                "config": {"models": ["logistic_regression"], "k_folds": 2, "hyperparameter_search": "none"}}))
            status = await self.wait(client, started["experiment_id"])
            self.assertEqual(status["state"], "completed", status)
            nothing = _data(await client.call_tool("create_experiment", {"automator": "auto", "train": "nothing.csv",
                                                                         "test": "nothing.csv"}))
            self.assertEqual(nothing["state"], "invalid")
            self.assertIn("nothing.csv was not found", nothing["errors"][0])
        self.session(steps)


@unittest.skipIf(Client is None, "the MCP SDK (Python >= 3.10) is not installed")
class TestSharedWorkspace(unittest.TestCase):
    def test_two_servers_start_a_queued_run_once(self):
        import threading
        from simplatab_mcp.jobs import Experiments
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["SIMPLATAB_WORKSPACE"] = tmp
            self.addCleanup(os.environ.pop, "SIMPLATAB_WORKSPACE", None)
            servers = [Experiments(worker_python=sys.executable) for _ in range(2)]
            for server in servers:  # stop the schedulers: the test drives the ticks
                server.close()
                server._thread.join()
            folder = Path(tmp) / "experiments" / "e1"
            folder.mkdir(parents=True)
            (folder / "experiment.json").write_text(json.dumps({"id": "e1", "automator": "tabular", "state": "queued",
                                                                 "queued": "1"}))
            launched = []

            def launch(server):
                def fake(meta):
                    time.sleep(0.3)  # the other server ticks meanwhile
                    launched.append(meta["id"])
                    server._save(meta["id"], state="running", pid=os.getpid())
                return fake
            for server in servers:
                server._launch = launch(server)
            threads = [threading.Thread(target=server.tick) for server in servers for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(launched, ["e1"])


EXAMPLES = [("tabular", "2d"), ("time-series-forecasting", "2d"), ("image-classification", "2d"),
            ("image-classification", "3d"), ("object-detection", "2d"), ("object-detection", "3d"),
            ("image-segmentation", "2d"), ("image-segmentation", "3d"), ("clustering", "labeled"),
            ("survival-analysis", "2d")]


@unittest.skipIf(Client is None, "the MCP SDK (Python >= 3.10) is not installed")
@unittest.skipUnless(os.environ.get("SIMPLATAB_MCP_ALL") == "1", "SIMPLATAB_MCP_ALL=1 runs every automator (about 15-20 min on a CPU)")
class TestEveryAutomator(TestServer):
    """What an agent does with each automator: example data, automator chosen from the data, a run with the
    quick configuration, and the results."""

    def test_every_automator_end_to_end(self):
        async def steps(client):
            report = []
            for automator, variant in EXAMPLES:
                example = _data(await client.call_tool("get_example_data", {"automator": automator, "variant": variant}))
                run = _data(await client.call_tool("run_experiment", {"automator": "auto", "train": example["train"],
                                                                      "test": example["test"], "config": example["quick_config"]}))
                self.assertEqual(run.get("automator"), automator, run)
                self.assertEqual(run["state"], "queued", run)
                status = await self.wait(client, run["experiment_id"], timeout=3600)
                self.assertEqual(status["state"], "completed", (automator, variant, status.get("message"), status.get("log_tail")))
                results = _data(await client.call_tool("get_results", {"experiment_id": run["experiment_id"]}))
                self.assertTrue(results["test_metrics"], automator)
                self.assertTrue(results["validation_metrics"], automator)
                self.assertTrue(results["best_model"], automator)
                self.assertTrue(results["models"], automator)
                self.assertTrue(results["splits"] and results["splits"]["folds"], automator)
                splits = await client.call_tool("read_result_file", {"experiment_id": run["experiment_id"],
                                                                     "path": "Splits/splits.csv"})
                self.assertTrue(splits.content[0].text.startswith("fold,set,id"))
                report.append((automator, variant, results["best_model"], status.get("elapsed_seconds")))
            print("\n".join(f"{a} {v}: best {b} in {t} s" for a, v, b, t in report))
        self.session(steps)

    # the other tests of TestServer run in their own class
    test_discovery = test_upload_errors_and_dry_run = test_full_tabular_experiment = test_inspect_and_auto = None


if __name__ == "__main__":
    unittest.main()
