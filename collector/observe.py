import copy
from pathlib import Path
import re
import time

from PIL import Image

from .common import digest, error_info, sha256


THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
SUFFIX = re.compile(r"<(tool_call|answer)>.*?</\1>", re.DOTALL)


def image_info(path, work):
    p = Path(path)
    info = {"original_path": str(p), "relative_path": None, "decodable": False}
    if p.is_file() and p.resolve().is_relative_to(work.resolve()):
        info.update(relative_path=p.resolve().relative_to(work.resolve()).as_posix(),
                    bytes=p.stat().st_size, sha256=sha256(p))
        try:
            with Image.open(p) as im:
                im.load()
                info.update(decodable=True, width=im.width, height=im.height, format=im.format)
        except Exception as exc:
            info["error"] = error_info(exc)
    return info


class ToolObserver:
    def __init__(self, image_dir):
        from genevolve.tools import ImageSearchTool, WebTextSearchTool

        self.events = []
        observer = self

        class ObserveHTTP:
            def _post(self, payload):
                start = time.monotonic()
                try:
                    result = super()._post(payload)
                except Exception as exc:
                    observer.events.append({"kind": self.kind, "ok": False,
                                            "error": error_info(exc)})
                    raise
                observer.events.append({"kind": self.kind, "ok": True,
                                        "seconds": time.monotonic() - start,
                                        "returned": len(result.get(self.field) or [])})
                return result

        class Text(ObserveHTTP, WebTextSearchTool):
            kind, field = "search_api", "organic"

        class Images(ObserveHTTP, ImageSearchTool):
            kind, field = "image_api", "images"

            def call(self, *args, **kwargs):
                refs = super().call(*args, **kwargs)
                observer.events.append({"kind": "image_call", "refs": copy.deepcopy(refs)})
                return refs

        self.text = Text()
        self.images = Images(download_dir=str(image_dir))

    def reset(self, image_dir):
        self.events.clear()
        self.images.download_dir = Path(image_dir)
        self.images.download_dir.mkdir(parents=True, exist_ok=True)

    def close(self):
        self.images._executor.shutdown(wait=True)


def decision(turn, tokenizer, index):
    from genevolve.agent import _parse_answer, _parse_tool_call
    from genevolve_repro.actor_text import decode_actor_response
    from genevolve_repro.policy import tokenize_history
    from genevolve.system_prompt import FINAL_STEP_MESSAGE

    messages = copy.deepcopy(turn.messages)
    text_only = all(isinstance(m.get("content"), str) for m in messages)
    raw = tokenizer.decode(turn.tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    response = decode_actor_response(tokenizer, turn.tokens)
    think = THINK.search(response)
    blocks = list(SUFFIX.finditer(response))
    final = bool(messages and messages[-1].get("content") == FINAL_STEP_MESSAGE)
    reasoning = think.group(1).strip() if think else ""
    try:
        tool, answer = _parse_tool_call(response), _parse_answer(response)
        parse_ok = bool((tool and tool[0] in {"search", "image_search", "query_knowledge"})
                        or isinstance(answer, dict))
    except (TypeError, ValueError, AttributeError):
        parse_ok = False
    protocol = bool(len(blocks) == 1 and parse_ok and
                    (final or (len(THINK.findall(response)) == 1 and reasoning and think.end() <= blocks[0].start())))
    return {
        "turn": index, "messages": messages, "prefix_ids": turn.prefix,
        "response_ids": turn.tokens, "raw_response": raw, "actor_response": response,
        "action_token_positions": turn.action, "reasoning": reasoning,
        "action": blocks[0].group() if len(blocks) == 1 else None,
        "forced_final_step": final, "route_candidate": bool(reasoning and not final and protocol),
        "text_only": text_only, "prefix_matches": tokenize_history(tokenizer, messages) == turn.prefix,
        "protocol_ok": protocol, "visible_input_sha256": digest(messages),
        "contains_rethink": "<RETHINK>" in raw,
    }


def collect_one(agent, observer, tokenizer, task, work, contract_hash):
    from genevolve_repro.policy import valid_program

    observer.reset(work / "images" / task["id"])
    start = time.monotonic()
    outcome, failure = None, None
    try:
        outcome = agent.collect(task["prompt"])
    except Exception as exc:
        failure = error_info(exc)
    turns = outcome.turns if outcome else getattr(agent, "turns", [])
    steps = [decision(turn, tokenizer, i) for i, turn in enumerate(turns)]
    events = copy.deepcopy(observer.events)
    paths = {r["local_path"] for e in events if e["kind"] == "image_call"
             for r in e["refs"] if r.get("local_path")}
    result = outcome.result.to_dict() if outcome else None
    if result:
        paths.update(r["local_path"] for r in result["reference_images"] if r.get("local_path"))
    assets = [image_info(p, work) for p in sorted(paths)]
    return {
        "schema": 1, "contract_sha256": contract_hash, "task": task, "decisions": steps,
        "result": result, "program_valid": bool(outcome and valid_program(outcome.result)),
        "infrastructure_error": failure, "tool_events": events, "assets": assets,
        "seconds": time.monotonic() - start,
    }
