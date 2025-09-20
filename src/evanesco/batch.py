"""Simple batch runner with multiprocessing concurrency.

Processes a directory or glob of inputs and writes redacted PDFs to an output
directory, preserving base filenames.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import List, Tuple
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from .core import RunConfig, process_path


def _one(args: Tuple[str, str, RunConfig]) -> Tuple[str, str, str]:
    inp, out_dir, cfg = args
    p = Path(inp)
    out_path = str(Path(out_dir) / f"{p.stem}.redacted.pdf")
    res = process_path(inp, out_path, cfg)
    return inp, out_path, res.get("out", out_path)


def run_batch(inputs: List[str], output_dir: str, cfg: RunConfig, workers: int = 2) -> List[Tuple[str, str]]:
    """Process multiple inputs concurrently.

    Returns a list of (input, output) pairs.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results: List[Tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = [ex.submit(_one, (i, output_dir, cfg)) for i in inputs]
        for f in as_completed(futs):
            try:
                inp, _, out = f.result()
                results.append((inp, out))
            except Exception:
                pass
    return results

