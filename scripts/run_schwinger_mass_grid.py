#!/usr/bin/env python3
"""Drive `LatticeQFT --model schwinger-hmc --corr-out` over a (beta, mass) grid and collect the
time-sliced pseudoscalar correlators into one CSV for scripts/analyze_schwinger_mass.py.

Usage: run_schwinger_mass_grid.py --bin build/LatticeQFT --out data/schwinger_mass_grid.csv
       [--L 16] [--betas 2,3,4,6] [--masses -0.4,-0.3,-0.2,-0.1,0,0.1] [--traj 400] [--every 4]
Points already in the output file are skipped, so the grid can be extended incrementally."""
import argparse, csv, os, subprocess, sys
p = argparse.ArgumentParser()
p.add_argument('--bin', required=True); p.add_argument('--out', required=True)
p.add_argument('--L', type=int, default=16)
p.add_argument('--betas', default='2,3,4,6'); p.add_argument('--masses', default='-0.4,-0.3,-0.2,-0.1,0,0.1')
p.add_argument('--traj', type=int, default=400); p.add_argument('--every', type=int, default=4)
p.add_argument('--therm', type=int, default=150); p.add_argument('--seed', type=int, default=1234567)
a = p.parse_args()
done = set()
if os.path.exists(a.out):
    for r in csv.DictReader(open(a.out)): done.add((round(float(r['beta']), 5), round(float(r['mass']), 5)))
for beta in [float(x) for x in a.betas.split(',')]:
    for m in [float(x) for x in a.masses.split(',')]:
        if (round(beta, 5), round(m, 5)) in done: continue
        cmd = [a.bin, '--model', 'schwinger-hmc', '--L', str(a.L), '--mass', str(m), '--beta-min', str(beta), '--beta-max', str(beta),
               '--beta-steps', '1', '--therm', str(a.therm), '--measure-sweeps', str(a.traj), '--sample-every', str(a.every),
               '--seed', str(a.seed), '--corr-out', a.out]
        print('[grid] beta', beta, 'mass', m, flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0: print(r.stderr, file=sys.stderr); sys.exit(1)
        last = r.stdout.strip().splitlines()[-1]
        print('       ', last, flush=True)
        # Quality side file: the HMC's acceptance and <dH> for this point, so the analysis can
        # drop points past the critical mass where the Dirac operator goes near-singular and
        # the trajectories blow up.
        q = os.path.splitext(a.out)[0] + '_quality.csv'
        fresh = not os.path.exists(q)
        with open(q, 'a') as f:
            if fresh: f.write('beta,mass,plaq,condensate,acceptance,avg_dH\n')
            c = last.split(',')
            f.write(f"{beta},{m},{c[1]},{c[3]},{c[5]},{c[6]}\n")
