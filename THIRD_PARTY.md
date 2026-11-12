# Third-party sources

- [MeiGen-AI/GenEvolve](https://github.com/MeiGen-AI/GenEvolve), pinned in `sources.json`: agent, system prompt, tools, knowledge skills. Downloaded, not vendored; consult the upstream repository for licensing and notices.
- [leiye302/GenEvolve-2B-Reproduction](https://github.com/leiye302/GenEvolve-2B-Reproduction), pinned in `sources.json`: matched model loading and generation adapter. Downloaded, not re-licensed here.
- [MeiGen-AI/GenEvolve-Data-Bench](https://huggingface.co/datasets/MeiGen-AI/GenEvolve-Data-Bench): only pinned train/validation/test JSONL metadata is downloaded at runtime; never committed here.
- [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct): reference architecture for the caller's existing SFT checkpoint. No weights bundled or downloaded automatically.
- Serper and image-source websites: live search/download services supplied by the operator, subject to their terms. Downloaded images are private run artifacts, not repository assets.

No broad license grant in this repository overrides the licenses or terms of its upstream dependencies, checkpoint, dataset, or images.
