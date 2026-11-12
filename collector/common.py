import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
METHOD = ROOT / ".deps" / "method"
UPSTREAM = METHOD / ".deps" / "genevolve"
SPLITS = ("GenEvolve-Data-RL/train.jsonl", "GenEvolve-Data-RL/eval.jsonl",
          "GenEvolve-Bench/test.jsonl")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        try:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def git(folder, *args):
    return subprocess.check_output(["git", "-C", str(folder), *args], text=True).strip()


def sources():
    return read_json(ROOT / "sources.json")


def check_sources():
    for name, folder in (("method", METHOD), ("agent", UPSTREAM)):
        pin = sources()[name]
        if git(folder, "rev-parse", "HEAD") != pin["revision"]:
            raise RuntimeError(f"Unexpected {name} revision; run collect.py setup.")
        if git(folder, "status", "--porcelain", "--untracked-files=no"):
            raise RuntimeError(f"Modified pinned {name} source; refusing protocol drift.")
    if str(METHOD) not in sys.path:
        sys.path.insert(0, str(METHOD))
    from genevolve_repro.common import load_upstream
    load_upstream()


def recipe():
    cfg = read_json(METHOD / "configs" / "evolve.json")
    fields = ("seed", "temperature", "top_p", "max_rounds", "max_initial_tokens",
              "max_response_tokens", "max_turn_tokens", "context_tokens")
    return {key: cfg[key] for key in fields}


def code_fingerprint():
    files = [ROOT / "collect.py", ROOT / "sources.json", ROOT / "configs" / "collection.json",
             *sorted((ROOT / "collector").glob("*.py"))]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in files}


def checkpoint_fingerprint(folder):
    folder = Path(folder).resolve()
    cfg = read_json(folder / "config.json")
    text = cfg.get("text_config", {})
    if (cfg.get("model_type") != "qwen3_vl" or text.get("hidden_size") != 4096
            or text.get("num_hidden_layers") != 36):
        raise ValueError("Expected a full Qwen3-VL-8B SFT export (not 2B or LoRA-only).")
    if cfg.get("quantization_config"):
        raise ValueError("Quantized checkpoints change the sampling contract; use BF16 export.")
    if not (folder / "tokenizer_config.json").is_file() or not (folder / "tokenizer.json").is_file():
        raise ValueError("The SFT export must include its tokenizer and chat template.")
    index = folder / "model.safetensors.index.json"
    names = (set(read_json(index)["weight_map"].values()) if index.is_file()
             else {"model.safetensors"})
    if not names:
        raise ValueError("Empty checkpoint index.")
    for name in names:
        p = (folder / name).resolve()
        if not p.is_relative_to(folder) or not p.is_file() or p.stat().st_size == 0:
            raise ValueError("Missing or unsafe checkpoint shard.")
    files = {*(folder / n for n in names), *folder.glob("*.json"), *folder.glob("*.jinja"),
             *folder.glob("*.txt"), *folder.glob("*.model")}
    return {str(p.relative_to(folder)).replace("\\", "/"): sha256(p) for p in sorted(files)}


def bind_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f"Existing {path.name} belongs to another run; use a new work directory.")
    else:
        atomic_json(path, value)


def error_info(error):
    # Do not print or export provider exception bodies / headers / credentials.
    response = getattr(error, "response", None)
    return {"type": type(error).__name__, "http_status": getattr(response, "status_code", None)}


def diagnostic(error):
    import traceback
    message = str(error)
    for key, value in os.environ.items():
        if len(value) >= 6 and any(word in key.upper() for word in ("KEY", "TOKEN", "PASSWORD", "SECRET")):
            message = message.replace(value, "[redacted]")
    message = re.sub(r"(?:oc_sk_|sk-|github_pat_|gh[pousr]_)[A-Za-z0-9_-]{10,}", "[redacted]", message)
    message = re.sub(r"https?://[^\s]+", "[endpoint omitted]", message)
    return {**error_info(error), "message": message[:2000], "stack": [
        {"module": Path(frame.filename).name, "function": frame.name, "line": frame.lineno}
        for frame in traceback.extract_tb(error.__traceback__)
    ]}
