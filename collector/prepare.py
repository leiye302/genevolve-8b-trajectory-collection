from collections import defaultdict
from pathlib import Path
import random
import subprocess

from .common import (METHOD, ROOT, SPLITS, UPSTREAM, atomic_json, bind_json, check_sources,
                     git, jsonl, read_json, sha256, sources)


def setup():
    for name, target in (("method", METHOD), ("agent", UPSTREAM)):
        pin = sources()[name]
        if not (target / ".git").is_dir():
            if target.exists() and any(target.iterdir()):
                raise RuntimeError(f"Nonempty dependency directory: {target.name}")
            subprocess.run(["git", "clone", "--quiet", pin["repo"], str(target)], check=True)
            subprocess.run(["git", "-C", str(target), "checkout", "--quiet", "--detach",
                            pin["revision"]], check=True)
        if git(target, "remote", "get-url", "origin") != pin["repo"]:
            raise RuntimeError("Unexpected dependency remote.")
    check_sources()
    print("Pinned public runtime ready.", flush=True)


def normalized_prompt(row):
    text = row.get("prompt") or row.get("question")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Missing prompt in official dataset.")
    return text


def build_plan(train, heldout, total, seed):
    # Only request text is used; no reference answer, image, or future evidence.
    held = {" ".join(normalized_prompt(row).lower().split()) for row in heldout}
    seen_ids, seen_prompts = set(), set()
    groups = defaultdict(list)
    excluded = 0
    for row in train:
        key = str(row["id"])
        prompt = normalized_prompt(row)
        norm = " ".join(prompt.lower().split())
        if key in seen_ids:
            raise ValueError("Duplicate source ID in RL train split.")
        seen_ids.add(key)
        if norm in held or norm in seen_prompts:
            excluded += 1
            continue
        seen_prompts.add(norm)
        meta = row.get("meta") or {}
        category = str(row.get("category") or meta.get("category") or row.get("eval_type") or "unspecified")
        groups[category].append({
            "source_id": key, "prompt": prompt, "category": category,
            "prompt_type": str(meta.get("prompt_type") or row.get("eval_type") or "unspecified"),
            "difficulty": str(meta.get("difficulty") or row.get("difficulty") or "unspecified"),
        })
    rng = random.Random(seed)
    # Proportional stratification across fixed shuffled category queues.
    for values in groups.values():
        rng.shuffle(values)
    sizes = {k: len(v) for k, v in groups.items()}
    taken = dict.fromkeys(groups, 0)
    ordered = []
    for _ in range(total):
        available = [k for k, v in groups.items() if taken[k] < len(v)]
        if not available:
            raise ValueError("Not enough distinct training prompts for requested count.")
        key = min(available, key=lambda k: ((taken[k] + 1) / sizes[k], k))
        ordered.append(groups[key][taken[key]])
        taken[key] += 1
    return [{**row, "index": i, "id": f"trajectory_{i:05d}", "seed": seed + i}
            for i, row in enumerate(ordered)], excluded


def prepare(work):
    from huggingface_hub import snapshot_download

    check_sources()
    cfg = read_json(ROOT / "configs" / "collection.json")
    pin = sources()["data"]
    snapshot_download(pin["repo"], revision=pin["revision"], repo_type="dataset",
                      local_dir=work / "data", allow_patterns=list(SPLITS))
    train, val, bench = [jsonl(work / "data" / rel) for rel in SPLITS]
    if [len(train), len(val), len(bench)] != [2575, 600, 594]:
        raise ValueError("Official pinned split sizes do not match 2575/600/594.")
    plan, excluded = build_plan(train, val + bench, cfg["total"], cfg["seed"])
    bind_json(work / "plan.json", plan)
    bind_json(work / "data_contract.json", {
        "data": pin, "sha256": {rel: sha256(work / "data" / rel) for rel in SPLITS},
        "excluded_overlap_or_duplicate_prompts": excluded,
        "selection": "proportional_category_strata_without_replacement",
        "total": len(plan), "pilot": cfg["pilot"], "seed": cfg["seed"],
    })
    print(f"Prepared {len(plan)} unique training tasks; no benchmark tasks or reference images downloaded.")
