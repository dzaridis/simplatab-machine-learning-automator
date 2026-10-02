"""Smoke test of the MCP server image, as it runs on a computer: a background container (HTTP, as
compose.yaml starts it) with a data folder and a workspace mounted from the host, used by an agent
over HTTP and over stdio (docker exec), with host paths.

    docker build -f mcp_server/Dockerfile -t simplatab-mcp .
    python mcp_server/tests/docker_smoke.py simplatab-mcp      # Python >= 3.10 with mcp_server/requirements.txt
"""
import asyncio
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from mcp import Client, StdioServerParameters

AUTOMATORS = ["tabular", "image-classification", "object-detection", "image-segmentation", "time-series-forecasting", "clustering"]
EXAMPLES = [("tabular", "2d"), ("time-series-forecasting", "2d"), ("image-classification", "2d"), ("image-classification", "3d"),
            ("object-detection", "2d"), ("object-detection", "3d"), ("image-segmentation", "2d"), ("image-segmentation", "3d"),
            ("clustering", "labeled"), ("clustering", "unlabeled")]


def data(result):
    if result.is_error:
        raise AssertionError(result.content[0].text)
    return result.structured_content if result.structured_content is not None else json.loads(result.content[0].text)


async def wait(client, experiment_id, timeout=1800):
    start = time.time()
    while time.time() - start < timeout:
        status = data(await client.call_tool("get_experiment", {"experiment_id": experiment_id}))
        if status["state"] not in ("queued", "running"):
            return status
        await asyncio.sleep(5)
    raise AssertionError(f"{experiment_id} did not finish")


async def over_http(url, host_data, host_workspace):
    async with Client(url) as client:
        tools = {t.name for t in (await client.list_tools()).tools}
        assert {"inspect_data", "run_experiment", "get_results"} <= tools, tools
        info = data(await client.call_tool("get_server_info", {}))
        print("server:", info["device"], "python", info["python"], "torch", info["torch"])
        for automator in AUTOMATORS:
            contract = data(await client.call_tool("get_data_contract", {"automator": automator}))
            assert contract["config"], automator
        for automator, variant in EXAMPLES:  # the automator is recognised from every example dataset
            example = data(await client.call_tool("get_example_data", {"automator": automator, "variant": variant}))
            found = data(await client.call_tool("inspect_data", {"train": example["train"]}))
            assert found["suggested_automator"] == automator, (automator, variant, found)
        print("contracts and automator detection: ok")

        # The agent gives host paths of the mounted data folder: they are translated
        example = data(await client.call_tool("get_example_data", {"automator": "tabular"}))
        for name in ("Train.csv", "Test.csv"):
            shutil.copy(host_workspace / "examples" / "tabular-2d" / name, host_data / name)
        run = data(await client.call_tool("run_experiment", {
            "automator": "auto", "train": str(host_data / "Train.csv"), "test": str(host_data / "Test.csv"),
            "config": dict(example["quick_config"], k_folds=2)}))
        assert run["state"] == "queued", run
        status = await wait(client, run["experiment_id"])
        assert status["state"] == "completed", status
        results = data(await client.call_tool("get_results", {"experiment_id": run["experiment_id"]}))
        model = Path(results["materials_host"]) / results["models"][0]
        assert model.exists(), model  # the trained model is on the host
        print("tabular over HTTP with host paths: best", results["best_model"], "->", model)
        return run["experiment_id"]


async def over_stdio(container, previous):
    server = StdioServerParameters(command="docker", args=["exec", "-i", container, "/opt/mcp/bin/python", "-m", "simplatab_mcp"])
    async with Client(server) as client:
        listed = data(await client.call_tool("list_experiments", {}))["experiments"]
        assert previous in [e["id"] for e in listed], listed  # one workspace for every session
        example = data(await client.call_tool("get_example_data", {"automator": "image-classification"}))
        run = data(await client.call_tool("run_experiment", {"automator": "auto", "train": example["train"],
                                                             "test": example["test"], "config": example["quick_config"]}))
        status = await wait(client, run["experiment_id"])
        assert status["state"] == "completed", status
        results = data(await client.call_tool("get_results", {"experiment_id": run["experiment_id"]}))
        print("image classification over stdio (docker exec): best", results["best_model"],
              {k: v for k, v in results["test_metrics"][0].items() if k in ("AUC", "Balanced Accuracy")})


def main(image):
    tmp = Path(tempfile.mkdtemp(prefix="simplatab-mcp-smoke-"))
    host_data, host_workspace = tmp / "data", tmp / "workspace"
    host_data.mkdir()
    host_workspace.mkdir()
    container = f"simplatab-mcp-smoke-{os.getpid()}"
    subprocess.run(["docker", "run", "-d", "--rm", "--init", "--name", container, "-p", "127.0.0.1:18765:8000",
                    "--shm-size=2g", "-v", f"{host_data}:/data", "-v", f"{host_workspace}:/workspace",
                    "-e", f"SIMPLATAB_HOST_DATA_DIR={host_data}", "-e", f"SIMPLATAB_HOST_WORKSPACE_DIR={host_workspace}",
                    # extra options, e.g. a worker Python mounted from the host when testing a reduced image
                    *shlex.split(os.environ.get("SIMPLATAB_SMOKE_DOCKER_ARGS", "")),
                    image, "--transport", "http", "--host", "0.0.0.0", "--port", "8000"], check=True)
    try:
        for _ in range(60):
            if subprocess.run(["docker", "logs", container], capture_output=True, text=True).stderr.count("Uvicorn running"):
                break
            time.sleep(1)
        previous = asyncio.run(over_http("http://127.0.0.1:18765/mcp", host_data, host_workspace))
        asyncio.run(over_stdio(container, previous))
        print("docker smoke test: ok")
    except BaseException:
        print(subprocess.run(["docker", "logs", "--tail", "60", container], capture_output=True, text=True).stderr)
        raise
    finally:
        subprocess.run(["docker", "stop", container], capture_output=True)
        subprocess.run(["docker", "run", "--rm", "-v", f"{tmp}:/t", "--entrypoint", "rm", image, "-rf", "/t/data", "/t/workspace"],
                       capture_output=True)  # files written by the container's root user
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "simplatab-mcp")
