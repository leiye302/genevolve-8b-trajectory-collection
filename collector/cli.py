import argparse
from pathlib import Path
import sys

from .common import ROOT, atomic_json, diagnostic, read_json


def main():
    parser = argparse.ArgumentParser(description="Matched GenEvolve 8B SFT trajectory collection; no training.")
    parser.add_argument("command", choices=("setup", "prepare", "pilot", "run", "audit", "export", "_worker"))
    parser.add_argument("--work", type=Path, default=Path("runs/collection"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/qwen3-vl-8b-sft"))
    parser.add_argument("--devices", default="cuda:0", help="One independent BF16 Actor replica per listed GPU.")
    parser.add_argument("--retry-errors", action="store_true", help="Archive and retry infrastructure failures, not bad policy outputs.")
    parser.add_argument("--full", action="store_true", help="audit all 1000 instead of the first 50")
    parser.add_argument("--rank", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--world", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--phase", choices=("pilot", "full"), default="pilot", help=argparse.SUPPRESS)
    args = parser.parse_args()
    work = args.work.resolve()
    checkpoint = args.checkpoint.resolve()
    work.mkdir(parents=True, exist_ok=True)
    from filelock import FileLock

    if args.command == "setup":
        from .prepare import setup
        with FileLock(str(ROOT / ".setup.lock"), timeout=0):
            setup()
        return
    if args.command == "_worker":
        from .run import worker
        from .audit import require_pilot
        if args.phase == "full":
            require_pilot(work)
        with FileLock(str(work / f".worker-{args.rank}.lock"), timeout=0):
            worker(work, checkpoint, args.devices, args.rank, args.world, args.phase, args.retry_errors)
        return
    with FileLock(str(work / ".collection.lock"), timeout=0):
        if args.command == "prepare":
            from .prepare import prepare
            prepare(work)
        elif args.command == "audit":
            from .audit import audit
            report = audit(work, pilot=not args.full)
            print({k: report[k] for k in ("passed", "count", "decisions", "fractions", "errors")})
            if not report["passed"]:
                raise RuntimeError("Collection audit failed; see saved audit report.")
        elif args.command == "export":
            from .export import export
            print(export(work))
        else:
            from .audit import audit, require_pilot
            from .export import export
            from .run import establish_contract, phase
            devices = args.devices.split(",")
            if len(set(devices)) != len(devices) or any(not d.startswith("cuda:") or not d[5:].isdigit() for d in devices):
                raise ValueError("Use unique devices such as cuda:0 or cuda:0,cuda:1.")
            establish_contract(work, checkpoint, devices)
            current_phase = "pilot"
            try:
                phase(work, checkpoint, devices, "pilot", args.retry_errors)
                report = audit(work, pilot=True)
                print(f"Pilot: passed={report['passed']} count={report['count']} "
                      f"downloaded_assets={report['downloaded_assets']}", flush=True)
                require_pilot(work)
                if args.command == "run":
                    current_phase = "full"
                    phase(work, checkpoint, devices, "full", args.retry_errors)
                    report = audit(work, pilot=False)
                    print(f"Full: passed={report['passed']} count={report['count']} decisions={report['decisions']}", flush=True)
                    if not report["passed"]:
                        raise RuntimeError("Full collection audit failed; raw evidence retained, not silently filtered.")
            except Exception:
                # Even an interrupted/incomplete phase gets a factual partial report.
                audit(work, pilot=current_phase == "pilot")
                raise
            finally:
                print(export(work), flush=True)


def entry():
    try:
        main()
    except Exception as exc:
        # Avoid leaking HTTP credentials through traceback strings into shared consoles.
        try:
            args = sys.argv
            folder = Path(args[args.index("--work") + 1]) if "--work" in args else Path("runs/collection")
            rank = args[args.index("--rank") + 1] if "--rank" in args else "coordinator"
            atomic_json(folder / "logs" / f"last_error_{rank}.json", diagnostic(exc))
        except Exception:
            pass
        print(f"Stopped ({type(exc).__name__}). Inspect private work/logs and audit files; "
              "completed trajectories are preserved.", file=sys.stderr)
        return 1
    return 0
