import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from .common import (ROOT, atomic_json, bind_json, check_sources, checkpoint_fingerprint,
                     code_fingerprint, digest, error_info, read_json, recipe, sources)


def runtime_contract(devices):
    import importlib.metadata
    import platform
    import torch
    expected = {"torch": "2.8.0", "transformers": "4.57.1", "tokenizers": "0.22.1",
                "flash-attn": "2.8.3"}
    actual = {name: importlib.metadata.version(name) for name in expected}
    if any(actual[n].split("+")[0] != version for n, version in expected.items()):
        raise RuntimeError("Use the pinned inference package versions from scripts/setup.sh.")
    if platform.system() != "Linux" or sys.version_info[:2] != (3, 11) or torch.version.cuda != "12.8":
        raise RuntimeError("Real collection requires the documented Linux/Python 3.11/CUDA 12.8 environment.")
    return {"versions": actual, "python": platform.python_version(), "cuda": torch.version.cuda,
            "gpus": [{"device": d, "name": torch.cuda.get_device_name(d),
                      "capability": list(torch.cuda.get_device_capability(d))} for d in devices]}


def establish_contract(work, checkpoint, devices):
    check_sources()
    cfg = read_json(ROOT / "configs" / "collection.json")
    sampling = recipe()
    if cfg["seed"] != sampling["seed"]:
        raise ValueError("Collector and pinned RL seed differ.")
    if cfg["actor_observation"] != "text_only" or cfg["action_protocol"] != "native_tool_call_or_answer":
        raise ValueError("Only the matched text/native protocol is implemented.")
    if not os.environ.get("SERPER_API_KEY", "").strip():
        raise ValueError("Set SERPER_API_KEY in the environment; never put it in a command argument.")
    runtime = runtime_contract(devices)
    # The URL is deliberately not archived (it may contain provider credentials).
    # Its hash still prevents a backend change on resume.
    backend = os.environ.get("SERPER_BASE_URL", "https://google.serper.dev").rstrip("/")
    from genevolve.system_prompt import SYSTEM_PROMPT, FINAL_STEP_MESSAGE
    from genevolve_repro.actor_text import ACTOR_TEXT_PROTOCOL
    value = {
        "schema": 1, "sources": sources(), "code": code_fingerprint(), "config": cfg,
        "sampling": sampling, "checkpoint_sha256": checkpoint_fingerprint(checkpoint),
        "data": read_json(work / "data_contract.json"), "plan_sha256": digest(read_json(work / "plan.json")),
        "system_prompt_sha256": digest(SYSTEM_PROMPT), "final_step_prompt_sha256": digest(FINAL_STEP_MESSAGE),
        "serper_endpoint_sha256": digest(backend), "devices": devices,
        "runtime": runtime,
        "actor": "existing_qwen3_vl_8b_sft", "images_in_actor_input": False,
        "actor_text_protocol": ACTOR_TEXT_PROTOCOL,
    }
    bind_json(work / "contract.json", value)
    return value


def worker(work, checkpoint, device, rank, world, phase, retry_errors):
    import torch
    from transformers import AutoTokenizer
    from .observe import ToolObserver, collect_one

    check_sources()
    from genevolve_repro.distributed import load_model
    from genevolve_repro.policy import make_agent
    cfg = read_json(ROOT / "configs" / "collection.json")
    contract = read_json(work / "contract.json")
    if contract["code"] != code_fingerprint():
        raise RuntimeError("Collector code changed during a run.")
    plan = read_json(work / "plan.json")
    tasks = plan[:cfg["pilot"]] if phase == "pilot" else plan[cfg["pilot"]:]
    tasks = [t for t in tasks if t["index"] % world == rank]
    pending = []
    for task in tasks:
        dest = work / "episodes" / f"{task['id']}.json"
        if dest.exists():
            old = read_json(dest)
            if old["task"] != task or old["contract_sha256"] != digest(contract):
                raise RuntimeError("Existing episode identity differs from contract.")
            failed = bool(old.get("infrastructure_error") or any(e.get("ok") is False for e in old["tool_events"]))
            if not failed or not retry_errors:
                continue
            archive = work / "failed_attempts" / task["id"] / f"{time.time_ns()}.json"
            archive.parent.mkdir(parents=True, exist_ok=True)
            dest.rename(archive)
        pending.append(task)
    if not pending:
        return
    if not torch.cuda.is_available() or not device.startswith("cuda:"):
        raise RuntimeError("Real collection requires a CUDA device; CPU is only for offline tests.")
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Use a BF16-capable GPU; precision is not silently changed.")
    model = load_model(checkpoint, torch.bfloat16, attention=cfg["attention"]).to(device).eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    observer = ToolObserver(work / "images")
    agent = make_agent(model, tokenizer, torch.device(device), contract["sampling"],
                       text_search_tool=observer.text, image_search_tool=observer.images)
    try:
        for task in pending:
            if shutil.disk_usage(work).free < cfg["min_free_gib"] * 1024 ** 3:
                raise RuntimeError("Low free disk space; existing episodes are safe. Resume after expanding storage.")
            random.seed(task["seed"])
            torch.manual_seed(task["seed"])
            row = collect_one(agent, observer, tokenizer, task, work, digest(contract))
            atomic_json(work / "episodes" / f"{task['id']}.json", row)
            print(f"{phase}: {task['index'] + 1}/{cfg['total']} decisions={len(row['decisions'])} "
                  f"program_valid={row['program_valid']}", flush=True)
            if row["infrastructure_error"] or any(e.get("ok") is False for e in row["tool_events"]):
                raise RuntimeError("Infrastructure request failed; stopping this worker with evidence saved.")
    finally:
        observer.close()
        agent._client.close()


def phase(work, checkpoint, devices, name, retry_errors):
    from .audit import require_pilot
    if name == "full":
        require_pilot(work)
    children, streams = [], []
    try:
        for rank, device in enumerate(devices):
            path = work / "logs" / f"{name}-rank{rank}-{time.time_ns()}.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open("w", encoding="utf-8")
            streams.append(stream)
            args = [sys.executable, "-u", str(ROOT / "collect.py"), "_worker", "--work", str(work),
                    "--checkpoint", str(checkpoint), "--devices", device, "--rank", str(rank),
                    "--world", str(len(devices)), "--phase", name]
            if retry_errors:
                args.append("--retry-errors")
            children.append(subprocess.Popen(args, stdout=stream, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in children):
            failed = [p.returncode for p in children if p.poll() not in (None, 0)]
            if failed:
                raise RuntimeError("A collection worker failed; no next phase was started. Inspect private logs.")
            time.sleep(2)
        if any(p.returncode != 0 for p in children):
            raise RuntimeError("Collection worker failed; partial evidence retained.")
    finally:
        # Only children created by this invocation, never unrelated GPU jobs.
        for p in children:
            if p.poll() is None:
                p.terminate()
        for p in children:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        for stream in streams:
            stream.close()
