import json
import os
import tempfile

from .common import atomic_json, read_json, sha256


def export(work):
    """Write review inputs separately from actions, tool outcomes and assets."""
    plan = read_json(work / "plan.json")
    destination = work / "exports"
    destination.mkdir(parents=True, exist_ok=True)
    names = ("trajectories.jsonl", "decisions.jsonl", "reasoning_review.jsonl", "image_manifest.jsonl")
    streams, temps, counts = {}, {}, dict.fromkeys(names, 0)
    try:
        for name in names:
            f = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination, delete=False)
            streams[name], temps[name] = f, f.name
        def put(name, row):
            streams[name].write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[name] += 1
        for task in plan:
            path = work / "episodes" / f"{task['id']}.json"
            if not path.is_file():
                continue
            row = read_json(path)
            put("trajectories.jsonl", row)
            for step in row["decisions"]:
                key = f"{task['id']}:turn{step['turn']:02d}"
                put("decisions.jsonl", {"id": key, "source_id": task["source_id"], **step})
                if step["route_candidate"]:
                    put("reasoning_review.jsonl", {
                        "id": key, "messages": step["messages"], "reasoning": step["reasoning"],
                        "visible_input_sha256": step["visible_input_sha256"],
                        "review_rule": "Use only these visible text messages and current reasoning. "
                                       "Do not inspect image pixels, current draft action, future observations, "
                                       "final program, or task outcome to assign a reasoning label.",
                    })
            for asset in row["assets"]:
                put("image_manifest.jsonl", {"trajectory_id": task["id"], **asset})
        for f in streams.values():
            f.flush()
            os.fsync(f.fileno())
            f.close()
        for name in names:
            os.replace(temps[name], destination / name)
        atomic_json(destination / "manifest.json", {
            "counts": counts, "sha256": {n: sha256(destination / n) for n in names},
            "note": "Raw private collection, not labeled or accepted Route-SFT data. "
                    "Image files remain under ../images; preserve the whole work directory.",
        })
    finally:
        from pathlib import Path
        for f in streams.values():
            if not f.closed:
                f.close()
        for path in temps.values():
            Path(path).unlink(missing_ok=True)
    return counts
