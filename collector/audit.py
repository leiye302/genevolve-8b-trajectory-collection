from collections import Counter
from pathlib import Path

from .common import ROOT, atomic_json, digest, read_json, sha256


def summarize(records, expected, gate):
    steps = [d for r in records for d in r["decisions"]]
    nonfinal = [d for d in steps if not d["forced_final_step"]]
    events = [e for r in records for e in r["tool_events"]]
    calls = [e for e in events if e["kind"] == "image_call"]
    assets = [a for r in records for a in r["assets"]]
    ratio = lambda n, d: n / d if d else 0.0
    fractions = {
        "image_search_episode_fraction": ratio(sum(any(e["kind"] == "image_call"
                    for e in r["tool_events"]) for r in records), expected),
        "image_call_download_fraction": ratio(sum(bool(c["refs"]) for c in calls), len(calls)),
        "decodable_download_fraction": ratio(sum(a["decodable"] for a in assets), len(assets)),
        "valid_program_fraction": ratio(sum(r["program_valid"] for r in records), expected),
        "protocol_fraction": ratio(sum(d["protocol_ok"] for d in steps), len(steps)),
        "nonfinal_reasoning_fraction": ratio(sum(bool(d["reasoning"]) for d in nonfinal), len(nonfinal)),
    }
    errors = []
    if len(records) != expected:
        errors.append(f"Only {len(records)}/{expected} trajectories are complete.")
    if len({r["task"]["id"] for r in records}) != len(records):
        errors.append("Duplicate trajectory IDs.")
    if any(r.get("infrastructure_error") for r in records):
        errors.append("Actor/infrastructure failure; inspect private episode records.")
    if any(e.get("ok") is False for e in events):
        errors.append("At least one search API request exhausted native retries.")
    if any(not d["text_only"] or not d["prefix_matches"] for d in steps):
        errors.append("Actor-visible context/token mismatch.")
    for key, minimum in gate.items():
        if fractions[key.removeprefix("min_")] < minimum:
            errors.append(f"{key} below {minimum}.")
    return {
        "passed": not errors, "errors": errors, "count": len(records), "expected": expected,
        "decisions": len(steps), "forced_final_steps": len(steps) - len(nonfinal),
        "route_candidates": sum(d["route_candidate"] for d in steps),
        "image_calls": len(calls), "downloaded_assets": len(assets), "fractions": fractions,
        "thresholds": gate,
        "terminations": dict(Counter((r.get("result") or {}).get("termination", "infrastructure_error")
                                      for r in records)),
        "categories": dict(Counter(r["task"]["category"] for r in records)),
        "prompt_types": dict(Counter(r["task"].get("prompt_type", "unspecified") for r in records)),
        "difficulties": dict(Counter(r["task"].get("difficulty", "unspecified") for r in records)),
        "scope": "engineering gate only; not semantic Good/Bad or drawing quality",
    }


def audit(work, pilot=True):
    config = read_json(ROOT / "configs" / "collection.json")
    plan = read_json(work / "plan.json")
    contract_hash = digest(read_json(work / "contract.json"))
    expected = config["pilot"] if pilot else config["total"]
    records, hashes = [], {}
    for task in plan[:expected]:
        path = work / "episodes" / f"{task['id']}.json"
        if not path.exists():
            continue
        row = read_json(path)
        if row["task"] != task or row["contract_sha256"] != contract_hash:
            raise RuntimeError("Episode provenance differs from immutable plan/contract.")
        for asset in row["assets"]:
            if not asset.get("relative_path"):
                continue
            file = (work / asset["relative_path"]).resolve()
            if not file.is_relative_to(work.resolve()) or not file.is_file() or sha256(file) != asset["sha256"]:
                raise RuntimeError("An archived reference image is missing or changed.")
        records.append(row)
        hashes[path.name] = sha256(path)
    report = summarize(records, expected, config["pilot_gate"])
    report.update(contract_sha256=contract_hash, episode_sha256=hashes)
    atomic_json(work / ("pilot_audit.json" if pilot else "full_audit.json"), report)
    return report


def require_pilot(work):
    # Recompute rather than trusting a user-edited passed flag or stale receipt.
    report = audit(work, pilot=True)
    if not report["passed"]:
        raise RuntimeError("Pilot gate failed; full collection has not started. See pilot_audit.json.")
