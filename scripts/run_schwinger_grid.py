#!/usr/bin/env python3
"""
Run the dynamical-Schwinger HMC binary across a (β, m) grid.

Each invocation of ``LatticeQFT --model schwinger-hmc`` produces one CSV
row of (plaquette, condensate, acceptance, ΔH). This script wraps that
single-point execution in two outer loops (β, m), parses the CSV output
(stdout), aggregates everything into one tidy dataframe-ready file at
``data/schwinger_grid.csv``, and prints progress to stderr.

Resumable: if the output CSV already exists, points already present are
skipped. Delete the file (or pass ``--restart``) to force a full re-run.

Default grid is small enough to finish in ~2 minutes on Apple silicon at
L=8. Bigger grids and longer trajectories are knobs.
"""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_one(
    binary: Path,
    beta: float,
    mass: float,
    L: int,
    n_traj: int,
    therm: int,
    hmc_dt: float,
    hmc_n_steps: int,
    n_sources: int,
    seed: int,
) -> dict | None:
    """Execute one (β, m) configuration; return parsed CSV row or None on failure."""
    cmd = [
        str(binary),
        "--model", "schwinger-hmc",
        "--L", str(L),
        "--mass", str(mass),
        "--seed", str(seed),
        "--beta-min", str(beta),
        "--beta-max", str(beta),
        "--beta-steps", "1",
        "--therm", str(therm),
        "--measure-sweeps", str(n_traj),
        "--sample-every", "2",
        "--hmc-dt", str(hmc_dt),
        "--hmc-n-steps", str(hmc_n_steps),
        "--n-sources", str(n_sources),
    ]
    print(f"[run] β={beta:.3f} m={mass:.3f} : "
          f"{' '.join(shlex.quote(s) for s in cmd)}",
          file=sys.stderr)

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  FAILED in {elapsed:.1f}s: {result.stderr.strip()}", file=sys.stderr)
        return None

    lines = [ln for ln in result.stdout.strip().split("\n") if ln]
    if len(lines) < 2:
        print(f"  no CSV body: stdout={result.stdout!r}", file=sys.stderr)
        return None

    header = lines[0].split(",")
    row = lines[1].split(",")
    if len(row) != len(header):
        print(f"  malformed CSV: header={header} row={row}", file=sys.stderr)
        return None

    out: dict = {"mass": mass}
    for k, v in zip(header, row):
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    out["wall_seconds"] = round(elapsed, 2)
    print(f"  ok  in {elapsed:.1f}s : plaq={out.get('plaq'):.4f}  "
          f"cond={out.get('condensate'):.4f}  acc={out.get('acceptance'):.3f}",
          file=sys.stderr)
    return out


def load_existing(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def already_have(rows: list[dict], beta: float, mass: float, tol: float = 1e-6) -> bool:
    for r in rows:
        try:
            if (abs(float(r["beta"]) - beta) < tol
                and abs(float(r["mass"]) - mass) < tol):
                return True
        except (KeyError, ValueError):
            continue
    return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", default=str(REPO_ROOT / "cmake-build-debug" / "LatticeQFT"))
    p.add_argument("--out",    default=str(REPO_ROOT / "data" / "schwinger_grid.csv"))
    p.add_argument("--L",            type=int,   default=8)
    p.add_argument("--n-traj",       type=int,   default=100)
    p.add_argument("--therm",        type=int,   default=30)
    p.add_argument("--hmc-dt",       type=float, default=0.04)
    p.add_argument("--hmc-n-steps",  type=int,   default=20)
    p.add_argument("--n-sources",    type=int,   default=6)
    p.add_argument("--betas",  nargs="+", type=float, default=[1.0, 2.0, 4.0])
    p.add_argument("--masses", nargs="+", type=float, default=[0.10, 0.20, 0.30, 0.50])
    p.add_argument("--restart", action="store_true",
                   help="ignore any existing CSV and re-run the entire grid")
    args = p.parse_args()

    binary = Path(args.binary)
    if not binary.exists():
        print(f"binary not found: {binary}", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = [] if args.restart else load_existing(out_path)

    rows: list[dict] = list(existing)
    seed = 12345

    total_grid = len(args.betas) * len(args.masses)
    print(f"grid: {len(args.betas)} betas × {len(args.masses)} masses = {total_grid} points",
          file=sys.stderr)
    print(f"existing: {len(existing)} rows in {out_path}", file=sys.stderr)

    for beta in args.betas:
        for mass in args.masses:
            if not args.restart and already_have(existing, beta, mass):
                print(f"[skip] β={beta:.3f} m={mass:.3f} (already in CSV)", file=sys.stderr)
                seed += 1
                continue
            r = run_one(
                binary, beta, mass, args.L,
                args.n_traj, args.therm, args.hmc_dt, args.hmc_n_steps,
                args.n_sources, seed,
            )
            seed += 1
            if r is not None:
                rows.append(r)
                # Re-write the file incrementally so a Ctrl-C doesn't lose work.
                _write(out_path, rows)

    print(f"wrote {len(rows)} rows total to {out_path}", file=sys.stderr)
    return 0


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    # Union of keys preserves all columns even if early/late rows differ.
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    sys.exit(main())
