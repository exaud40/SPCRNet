# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 21:59:06 2026

@author: Exaud
"""

from __future__ import annotations
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Job:
    name: str
    run_dir: str
    args: list[str] = field(default_factory=list)          
    iterations: int = 400
    module: str = "uavsc.rl.wgae_ppo"
    cmd: list[str] | None = None                            

    def command(self) -> list[str]:
        if self.cmd is not None:
            return list(self.cmd)
        module = os.environ.get("UAVSC_TRAINER_MODULE", self.module)   
        return [sys.executable, "-m", module, *self.args,
                "--iterations", str(self.iterations), "--log_dir", self.run_dir]


def is_complete(run_dir: str | Path, iterations: int) -> bool:
    """A run is complete when progress.csv has >= iterations rows and checkpoint.pt exists."""
    run_dir = Path(run_dir)
    log, ckpt = run_dir / "progress.csv", run_dir / "checkpoint.pt"
    if not (log.exists() and ckpt.exists()):
        return False
    with open(log, newline="", encoding="utf-8") as fh:
        rows = sum(1 for _ in csv.DictReader(fh))
    return rows >= iterations


def tail(path: str | Path, n: int = 20) -> str:
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "(no output file)"
    return "\n".join(lines[-n:]) if lines else "(empty output)"


def launch(jobs: list[Job], max_parallel: int = 4, poll_s: float = 2.0, verbose: bool = True,
           fail_fast: int = 3) -> dict[str, int]:
    pending = [j for j in jobs if not is_complete(j.run_dir, j.iterations)]
    codes = {j.name: 0 for j in jobs if j not in pending}
    if verbose and codes:
        print(f"[launcher] skipping {len(codes)} complete run(s)")
    running: list[tuple[Job, subprocess.Popen, object, float]] = []
    n_ok = n_fail = 0
    while pending or running:
        while pending and len(running) < max_parallel:
            job = pending.pop(0)
            Path(job.run_dir).mkdir(parents=True, exist_ok=True)
            out = open(Path(job.run_dir) / "stdout.txt", "w", encoding="utf-8")
            proc = subprocess.Popen(job.command(), stdout=out, stderr=subprocess.STDOUT)
            running.append((job, proc, out, time.time()))
            if verbose:
                print(f"[launcher] started {job.name}  ({len(running)} running, {len(pending)} queued)", flush=True)
        for item in list(running):
            job, proc, out, t0 = item
            if proc.poll() is not None:
                out.close()
                running.remove(item)
                codes[job.name] = proc.returncode
                n_fail += proc.returncode != 0
                n_ok += proc.returncode == 0
                if verbose:
                    status = "done" if proc.returncode == 0 else f"FAILED (code {proc.returncode})"
                    print(f"[launcher] {job.name} {status} after {(time.time() - t0) / 60:.1f} min", flush=True)
                if proc.returncode != 0 and n_fail == 1:
                    print(f"[launcher] ---- last lines of {Path(job.run_dir) / 'stdout.txt'} ----\n"
                          f"{tail(Path(job.run_dir) / 'stdout.txt')}\n[launcher] ----", flush=True)
        if fail_fast and n_ok == 0 and n_fail >= fail_fast:
            for job, proc, out, _ in running:
                proc.kill()
                proc.wait()
                out.close()
            raise RuntimeError(f"the first {n_fail} jobs all failed, stopping (setup problem?). "
                               f"See the output printed above and <run_dir>/stdout.txt")
        if running:
            time.sleep(poll_s)
    failed = [n for n, c in codes.items() if c != 0]
    if failed:
        raise RuntimeError(f"jobs failed: {failed}; see <run_dir>/stdout.txt")
    return codes
