"""Run the dura evals on a registered model.

    python -m dura.run <model> <output_dir> [--model-kwargs key=value ...] [--tasks ...]

Models register with `@register_model` in `dura.models`, including from other repos as
namespace package plugins. See the README.
"""

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

from dura.models.registry import create_model, list_models
from dura.tasks import TASKS

logger = logging.getLogger(__name__)


def git_sha() -> str:
    """Short sha of the dura checkout, "-dirty" if it has changes, "unknown" before any commit."""
    kwargs = {"cwd": Path(__file__).parent, "capture_output": True, "text": True}
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=False, **kwargs)
    if sha.returncode != 0:
        return "unknown"
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "-uno"], check=False, **kwargs
    ).stdout.strip()
    return f"{sha.stdout.strip()}-dirty" if dirty else sha.stdout.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=list_models())
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--model-kwargs",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help="string kwargs for the model constructor, e.g. ckpt_path=/path/to/ckpt.pt",
    )
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_kwargs = dict(kv.split("=", 1) for kv in args.model_kwargs)
    transform, model = create_model(args.model, **model_kwargs)
    model = model.to(args.device)

    for task in args.tasks:
        output_path = args.output_dir / f"{task}.json"
        if output_path.exists():
            logger.info(f"skipping {task}, {output_path} exists")
            continue
        logger.info(f"running {task}")
        start = time.perf_counter()
        probe = TASKS[task]()
        result, _ = probe(
            model,
            transform,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            amp=not args.no_amp,
        )
        result = {
            "task": task,
            "model": args.model,
            "model_kwargs": model_kwargs,
            "dura_git_sha": git_sha(),
            "total_seconds": time.perf_counter() - start,
            **result,
        }
        with output_path.open("w") as f:
            json.dump(result, f)
        logger.info(f"{task} done in {result['total_seconds']:.0f}s")
