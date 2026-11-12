import copy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import torch

from collector.audit import audit, require_pilot, summarize
from collector.common import (ROOT, atomic_json, bind_json, check_sources, checkpoint_fingerprint,
                              digest, jsonl, read_json, recipe)
from collector.export import export
from collector.observe import ToolObserver, collect_one, decision, image_info
from collector.prepare import build_plan


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    all_special_ids = [0, 2, 999]

    def get_vocab(self):
        return {"<RETHINK>": 999}

    def apply_chat_template(self, messages, **_):
        return self("\n".join(m["role"] + ": " + m["content"] for m in messages) + "\nassistant:")["input_ids"]

    def __call__(self, text, **_):
        ids, offsets, pos = [], [], 0
        while pos < len(text):
            marker = text.startswith("<RETHINK>", pos)
            end = pos + (len("<RETHINK>") if marker else 1)
            ids.append(999 if marker else ord(text[pos]))
            offsets.append((pos, end))
            pos = end
        return {"input_ids": ids, "offset_mapping": offsets}

    def decode(self, ids, skip_special_tokens=False, **_):
        parts = []
        for i in ids:
            if skip_special_tokens and i in self.all_special_ids:
                continue
            parts.append("<RETHINK>" if i == 999 else chr(i))
        return "".join(parts)


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate(self, input_ids, **kwargs):
        self.calls.append({"input_ids": input_ids.tolist(), **kwargs})
        tokens = CharacterTokenizer()(next(self.responses))["input_ids"]
        return torch.cat((input_ids, torch.tensor([tokens])), dim=1)


def response_json(value):
    return SimpleNamespace(raise_for_status=lambda: None, json=lambda: value)


class CollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        check_sources()
        torch.set_num_threads(1)

    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.work = Path(t.name)
        self.cfg = read_json(ROOT / "configs" / "collection.json")
        self.tokenizer = CharacterTokenizer()
        self.task = {"index": 0, "id": "trajectory_00000", "seed": 42, "source_id": "train_1",
                     "prompt": "Draw a gold bird.", "category": "animals"}

    def collect(self, responses=None, corrupt=False, tool_failure=False, config=None):
        from genevolve_repro.policy import make_agent
        if responses is None:
            responses = [
                '<think>I need a bird reference.</think><tool_call>{"name":"image_search","arguments":{"query":"bird"}}</tool_call>',
                '<think>The search returned a bird title. Use this reference.</think><answer>{"gen_prompt":"Depict a gold bird using the first reference image.","reference_images":[{"img_id":"IMG_001"}]}</answer>',
            ]
        model = ScriptedModel(responses)
        observer = ToolObserver(self.work / "images")
        self.addCleanup(observer.close)
        # The test tokenizer uses one token per character, unlike real Qwen BPE.
        # Only this fixture increases the initial cap to fit the unchanged system prompt.
        fixture_cfg = {**(config or recipe()), "max_initial_tokens": 20000}
        agent = make_agent(model, self.tokenizer, "cpu", fixture_cfg,
                           text_search_tool=observer.text, image_search_tool=observer.images)
        self.addCleanup(agent._client.close)
        content = io.BytesIO()
        with Image.new("RGB", (8, 8), "gold") as im:
            im.save(content, format="PNG")
        image_bytes = b"not an image" if corrupt else content.getvalue()
        request = Mock(return_value=response_json({"images": [{
            "title": "A bird", "imageUrl": "https://example.org/bird.png",
        }]}))
        if tool_failure:
            request.side_effect = RuntimeError("offline transport failure")
        with patch.dict(os.environ, {"SERPER_API_KEY": "offline-fixture"}), \
                patch("requests.post", request), \
                patch("requests.get", return_value=SimpleNamespace(raise_for_status=lambda: None,
                      iter_content=lambda **_: [image_bytes])), \
                patch("genevolve.tools.web_search.time.sleep"):
            row = collect_one(agent, observer, self.tokenizer, self.task, self.work, "fixture-contract")
        return row, model

    def test_real_native_tool_loop_and_actual_download(self):
        from genevolve.system_prompt import SYSTEM_PROMPT
        row, model = self.collect()
        self.assertTrue(row["program_valid"])
        self.assertEqual(len(row["decisions"]), 2)
        first, second = row["decisions"]
        self.assertEqual(first["messages"][0]["content"], SYSTEM_PROMPT)
        self.assertTrue(all(d["text_only"] and d["prefix_matches"] for d in row["decisions"]))
        self.assertIn("IMG_001: title: A bird", second["messages"][-1]["content"])
        self.assertNotIn("image_url", [m.get("type") for m in second["messages"]])
        self.assertTrue(row["assets"][0]["decodable"])
        self.assertEqual(row["assets"][0]["width"], 8)
        self.assertEqual(model.calls[0]["temperature"], 0.7)
        self.assertEqual(model.calls[0]["top_p"], 0.95)
        self.assertEqual(model.calls[0]["top_k"], 0)
        self.assertNotIn("pixel_values", model.calls[0])
        self.assertNotIn("<action>", first["actor_response"])

    def test_corrupt_download_is_not_an_image_success(self):
        row, _ = self.collect(corrupt=True)
        self.assertFalse(row["assets"][0]["decodable"])
        report = summarize([row], 1, self.cfg["pilot_gate"])
        self.assertFalse(report["passed"])
        self.assertEqual(report["fractions"]["decodable_download_fraction"], 0)

    def test_http_failures_not_silently_counted_as_valid_search(self):
        row, _ = self.collect(tool_failure=True)
        self.assertTrue(any(e.get("ok") is False for e in row["tool_events"]))
        self.assertFalse(summarize([row], 1, self.cfg["pilot_gate"])["passed"])

    def test_final_override_has_no_reasoning_and_is_not_route_candidate(self):
        from genevolve.system_prompt import FINAL_STEP_MESSAGE
        row, _ = self.collect(responses=[
            '<think>Need a bird.</think><tool_call>{"name":"image_search","arguments":{"query":"bird"}}</tool_call>',
            '<answer>{"gen_prompt":"Draw a bird using the first reference image.","reference_images":[{"img_id":"IMG_001"}]}</answer>',
        ], config={**recipe(), "max_rounds": 2})
        step = row["decisions"][-1]
        self.assertTrue(step["forced_final_step"])
        self.assertEqual(step["reasoning"], "")
        self.assertTrue(step["protocol_ok"])
        self.assertFalse(step["route_candidate"])
        self.assertEqual(step["messages"][-1]["content"], FINAL_STEP_MESSAGE)

    def test_exact_visible_history_matches_training_path(self):
        from genevolve_repro.policy import make_agent
        row, _ = self.collect()
        responses = [d["actor_response"] for d in row["decisions"]]
        refs = row["tool_events"][-1]["refs"]
        agent = make_agent(ScriptedModel(responses), CharacterTokenizer(), "cpu",
                           {**recipe(), "max_initial_tokens": 20000},
                           image_search_tool=SimpleNamespace(call=lambda **_: refs))
        self.addCleanup(agent._client.close)
        trajectory = agent.collect(self.task["prompt"])
        for original, captured in zip(trajectory.turns, row["decisions"]):
            self.assertEqual(original.messages, captured["messages"])
            self.assertEqual(original.prefix, captured["prefix_ids"])
            self.assertEqual(original.tokens, captured["response_ids"])

    def test_marker_supported_without_inserting_or_registering_tokens(self):
        from genevolve_repro.actor_text import decode_actor_response
        self.assertEqual(decode_actor_response(self.tokenizer, [ord("a"), 999, 2]), "a<RETHINK>")
        self.assertEqual(decode_actor_response(self.tokenizer, [ord("a"), 2]), "a")
        self.assertEqual(self.tokenizer.get_vocab(), {"<RETHINK>": 999})

    def test_special_marker_matches_training_and_survives_next_turn(self):
        from genevolve_repro.policy import make_agent
        responses = [
            '<think>Need a bird reference.</think><RETHINK><tool_call>{"name":"image_search","arguments":{"query":"bird"}}</tool_call>\x02',
            '<think>Use the reference.</think><answer>{"gen_prompt":"Draw a gold bird.","reference_images":[{"img_id":"IMG_001"}]}</answer>\x02',
        ]
        row, _ = self.collect(responses=responses)
        self.assertTrue(row["program_valid"])
        first, second = row["decisions"]
        self.assertIn("</think><RETHINK><tool_call>", first["actor_response"])
        self.assertNotIn("\x02", first["actor_response"])
        self.assertEqual(first["response_ids"], self.tokenizer(responses[0])["input_ids"])
        self.assertEqual(first["response_ids"][-1], 2)
        self.assertIn(999, first["response_ids"])
        self.assertEqual(second["messages"][2]["content"], first["actor_response"])
        self.assertIn(999, second["prefix_ids"])
        self.assertTrue(first["contains_rethink"])
        refs = row["tool_events"][-1]["refs"]
        agent = make_agent(ScriptedModel(responses), self.tokenizer, "cpu",
                           {**recipe(), "max_initial_tokens": 20000},
                           image_search_tool=SimpleNamespace(call=lambda **_: refs))
        self.addCleanup(agent._client.close)
        trajectory = agent.collect(self.task["prompt"])
        for original, captured in zip(trajectory.turns, row["decisions"]):
            self.assertEqual(original.messages, captured["messages"])
            self.assertEqual(original.prefix, captured["prefix_ids"])
            self.assertEqual(original.tokens, captured["response_ids"])

    def test_model_error_preserves_collected_prefix_evidence(self):
        row, _ = self.collect(responses=['<think>Use a reference.</think><tool_call>{"name":"image_search","arguments":{"query":"bird"}}</tool_call>'])
        self.assertIsNotNone(row["infrastructure_error"])
        self.assertEqual(len(row["decisions"]), 1)
        self.assertTrue(row["assets"][0]["decodable"])

    def test_successful_fixture_passes_gate(self):
        row, _ = self.collect()
        self.assertTrue(summarize([row], 1, self.cfg["pilot_gate"])["passed"])

    def test_gate_rejects_empty_incomplete_duplicate_and_mismatched_prefix(self):
        row, _ = self.collect()
        self.assertFalse(summarize([], 50, self.cfg["pilot_gate"])["passed"])
        self.assertFalse(summarize([row], 50, self.cfg["pilot_gate"])["passed"])
        self.assertFalse(summarize([row, row], 2, self.cfg["pilot_gate"])["passed"])
        row["decisions"][0]["prefix_matches"] = False
        self.assertFalse(summarize([row], 1, self.cfg["pilot_gate"])["passed"])

    def test_no_extra_image_injected_into_reasoning_review(self):
        row, _ = self.collect()
        atomic_json(self.work / "plan.json", [self.task])
        atomic_json(self.work / "episodes" / f"{self.task['id']}.json", row)
        counts = export(self.work)
        review = jsonl(self.work / "exports" / "reasoning_review.jsonl")
        self.assertEqual(counts["reasoning_review.jsonl"], 2)
        self.assertEqual(review[0]["messages"], row["decisions"][0]["messages"])
        self.assertNotIn("reference_images", review[0])
        self.assertNotIn("action", review[0])
        self.assertNotIn("result", review[0])
        self.assertNotIn("IMG_001: title", json.dumps(review[0]))
        self.assertNotIn("Depict a gold bird", json.dumps(review[0]))

    def test_source_plan_is_train_only_distinct_and_deterministic(self):
        train = [{"id": i, "prompt": f"Drawing {i}", "category": "a" if i % 2 else "b"} for i in range(10)]
        train.append({"id": 10, "prompt": "Drawing 0", "category": "b"})
        plan, excluded = build_plan(train, [{"question": " Drawing   2 "}], 8, 42)
        self.assertEqual(excluded, 2)
        self.assertEqual(plan, build_plan(train, [{"question": " Drawing   2 "}], 8, 42)[0])
        self.assertEqual(len(set(r["prompt"] for r in plan)), 8)
        self.assertNotIn("Drawing 2", [r["prompt"] for r in plan])
        self.assertEqual(set(r["category"] for r in plan[:2]), {"a", "b"})
        self.assertEqual([r["seed"] for r in plan], list(range(42, 50)))

    def test_bind_contract_rejects_resume_drift(self):
        path = self.work / "contract.json"
        bind_json(path, {"temperature": 0.7})
        bind_json(path, {"temperature": 0.7})
        with self.assertRaises(RuntimeError):
            bind_json(path, {"temperature": 0.6})

    def test_checkpoint_fingerprint_rejects_wrong_backbone_missing_shards_and_changes(self):
        ckpt = self.work / "checkpoint"
        atomic_json(ckpt / "config.json", {"model_type": "qwen3_vl", "text_config": {"hidden_size": 2048}})
        with self.assertRaises(ValueError):
            checkpoint_fingerprint(ckpt)
        atomic_json(ckpt / "config.json", {"model_type": "qwen3_vl", "text_config": {"hidden_size": 4096, "num_hidden_layers": 36}})
        atomic_json(ckpt / "tokenizer.json", {})
        atomic_json(ckpt / "tokenizer_config.json", {})
        with self.assertRaises(ValueError):
            checkpoint_fingerprint(ckpt)
        (ckpt / "model.safetensors").write_bytes(b"test fixture, not real weights")
        before = checkpoint_fingerprint(ckpt)
        (ckpt / "model.safetensors").write_bytes(b"changed test fixture")
        self.assertNotEqual(before, checkpoint_fingerprint(ckpt))

    def test_real_pilot_receipt_rechecks_images_not_just_passed_flag(self):
        row, _ = self.collect()
        contract = {"fixture": 1}
        row["contract_sha256"] = digest(contract)
        atomic_json(self.work / "contract.json", contract)
        atomic_json(self.work / "plan.json", [self.task])
        atomic_json(self.work / "episodes" / f"{self.task['id']}.json", row)
        atomic_json(self.work / "pilot_audit.json", {"passed": True})
        with self.assertRaises(RuntimeError):
            require_pilot(self.work)  # one is not fifty
        Path(row["assets"][0]["original_path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "missing or changed"):
            audit(self.work)

    def test_no_full_launch_on_failed_pilot(self):
        from collector.run import phase
        with patch("collector.audit.require_pilot", side_effect=RuntimeError("fixture failed")), \
                patch("collector.run.subprocess.Popen") as spawn:
            with self.assertRaises(RuntimeError):
                phase(self.work, self.work / "checkpoint", ["cuda:0"], "full", False)
        spawn.assert_not_called()

    def test_sampler_recipe_remains_rl_not_benchmark_defaults(self):
        sampling = recipe()
        self.assertEqual((sampling["temperature"], sampling["top_p"]), (0.7, 0.95))
        self.assertEqual(sampling["max_rounds"], 11)
        self.assertEqual(sampling["max_response_tokens"], 30000)
        self.assertEqual(sampling["context_tokens"], 65536)

    def test_nested_official_metadata_is_used_for_diversity(self):
        train = [{"id": i, "prompt": str(i), "meta": {"category": str(i % 2),
                 "prompt_type": "Knowledge-Anchored", "difficulty": "hard"}} for i in range(10)]
        plan, _ = build_plan(train, [], 4, 42)
        self.assertEqual(set(r["category"] for r in plan[:2]), {"0", "1"})
        self.assertTrue(all(r["prompt_type"] == "Knowledge-Anchored" for r in plan))

    def test_protocol_diagnostic_rejects_empty_reasoning_or_invalid_json(self):
        from genevolve_repro.policy import Turn
        messages = [{"role": "system", "content": "Fixture"}, {"role": "user", "content": "Draw a bird"}]
        for response in ('<think></think><answer>{}</answer>',
                         '<think>Need a bird.</think><tool_call>not json</tool_call>',
                         '<answer>{}</answer><think>Too late</think>'):
            turn = Turn(messages, self.tokenizer.apply_chat_template(messages), list(map(ord, response)), [])
            self.assertFalse(decision(turn, self.tokenizer, 0)["protocol_ok"])

    def test_diagnostics_redact_secret_and_endpoint(self):
        from collector.common import diagnostic
        with patch.dict(os.environ, {"SERPER_API_KEY": "private-test-fixture"}):
            info = diagnostic(RuntimeError("private-test-fixture https://example.org/?key=private-test-fixture"))
        self.assertNotIn("private-test-fixture", json.dumps(info))
        self.assertNotIn("example.org", json.dumps(info))

    def test_coordinator_runs_pilot_before_full(self):
        from collector.cli import main
        argv = ["collect.py", "run", "--work", str(self.work), "--checkpoint", str(self.work / "model")]
        report = {"passed": True, "count": 50, "downloaded_assets": 100, "decisions": 120}
        events = []
        with patch("sys.argv", argv), patch("collector.run.establish_contract"), \
                patch("collector.run.phase", side_effect=lambda *a: events.append(a[3])), \
                patch("collector.audit.audit", return_value=report), \
                patch("collector.audit.require_pilot", side_effect=lambda _: events.append("gate")), \
                patch("collector.export.export", return_value={}):
            main()
        self.assertEqual(events, ["pilot", "gate", "full"])

    def test_coordinator_failure_exports_partial_evidence_without_expansion(self):
        from collector.cli import main
        argv = ["collect.py", "run", "--work", str(self.work), "--checkpoint", str(self.work / "model")]
        with patch("sys.argv", argv), patch("collector.run.establish_contract"), \
                patch("collector.run.phase", side_effect=RuntimeError("worker failed")) as phase, \
                patch("collector.audit.audit", return_value={"passed": False}) as report, \
                patch("collector.export.export", return_value={}) as exported:
            with self.assertRaises(RuntimeError):
                main()
        self.assertEqual(phase.call_count, 1)
        self.assertEqual(phase.call_args.args[3], "pilot")
        report.assert_called_once_with(self.work, pilot=True)
        exported.assert_called_once_with(self.work)


if __name__ == "__main__":
    unittest.main()
