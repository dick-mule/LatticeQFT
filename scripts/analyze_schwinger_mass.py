#!/usr/bin/env python3
"""Schwinger-model mass gap from the pseudoscalar correlators written by
`LatticeQFT --model schwinger-hmc --corr-out` (via scripts/run_schwinger_mass_grid.py).

Per (beta, mass): effective masses M_pi (connected) and M_eta (singlet) from the periodic
cosh form, averaged over a plateau window. Per beta: the chiral point m_c from a linear fit of
M_pi^2 against the bare mass, and M_eta interpolated to m_c. Across beta: M_eta / e against the
lattice spacing e a = 1 / sqrt(beta), extrapolated linearly to a -> 0 and compared with the
exact Schwinger boson, M / e = 1 / sqrt(pi) = 0.5642.

Usage: analyze_schwinger_mass.py data/schwinger_mass_grid.csv [--tmin 3] [--figdir docs/figures]"""
import argparse, csv, math, os, sys
from collections import defaultdict
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('csv'); p.add_argument('--tmin', type=int, default=3); p.add_argument('--figdir', default='docs/figures')
a = p.parse_args()

rows = list(csv.DictReader(open(a.csv)))
pts = defaultdict(dict)   # (beta, mass) -> {t: (conn, err, disc, err, eta, err)}
for r in rows:
    key = (float(r['beta']), float(r['mass'])); pts[key][int(r['t'])] = tuple(float(r[k]) for k in ('conn', 'conn_err', 'disc', 'disc_err', 'eta', 'eta_err'))
L = int(rows[0]['L'])

def meff(Ct, Ct1, t):
    if not (Ct > 0 and Ct1 > 0): return float('nan')
    tgt = Ct / Ct1; u, v = t - L / 2, t + 1 - L / 2
    f = lambda M: math.cosh(M * u) / math.cosh(M * v)
    lo, hi = 1e-6, 10.0
    if (f(lo) - tgt) * (f(hi) - tgt) > 0: return float('nan')
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if (f(mid) - tgt) * (f(lo) - tgt) <= 0: hi = mid
        else: lo = mid
    return 0.5 * (lo + hi)

def plateau(C, E, t0, t1):
    """Error-weighted mean of the effective mass over t in [t0, t1]; the error of each M_eff is
    propagated from the two correlator errors by finite differences."""
    vals, ws = [], []
    for t in range(t0, t1):
        M = meff(C[t], C[t + 1], t)
        if math.isnan(M): continue
        dM = 0.5 * (abs(meff(C[t] + E[t], C[t + 1], t) - meff(C[t] - E[t], C[t + 1], t)) +
                    abs(meff(C[t], C[t + 1] + E[t + 1], t) - meff(C[t], C[t + 1] - E[t + 1], t)))
        if math.isnan(dM) or dM <= 0: continue
        vals.append(M); ws.append(1.0 / dM ** 2)
    if not vals: return float('nan'), float('nan')
    w = np.array(ws); v = np.array(vals)
    return float((w * v).sum() / w.sum()), float(1.0 / math.sqrt(w.sum()))

results = {}
print(f"{'beta':>5} {'mass':>6} {'M_pi':>8} {'+-':>6} {'M_eta':>8} {'+-':>6}")
for (beta, m), tab in sorted(pts.items()):
    ts = sorted(tab)
    conn = [tab[t][0] for t in ts]; econ = [tab[t][1] for t in ts]
    eta = [tab[t][4] for t in ts]; eeta = [tab[t][5] for t in ts]
    t1 = L // 2 - 1
    Mpi, dMpi = plateau(conn, econ, a.tmin, t1)
    Meta, dMeta = plateau(eta, eeta, a.tmin, t1)
    results[(beta, m)] = (Mpi, dMpi, Meta, dMeta)
    print(f"{beta:5.2f} {m:6.2f} {Mpi:8.4f} {dMpi:6.4f} {Meta:8.4f} {dMeta:6.4f}")

# Per beta: chiral point from M_pi^2 vs m, then M_eta at m_c.
print("\nper beta: chiral point and the Schwinger boson")
print(f"{'beta':>5} {'e a':>6} {'m_c':>7} {'M_eta(m_c) a':>12} {'M_eta/e':>8} {'exact':>6}")
summary = []
for beta in sorted({k[0] for k in results}):
    ms = sorted(k[1] for k in results if k[0] == beta)
    mm = np.array([m for m in ms if not math.isnan(results[(beta, m)][0])])
    if len(mm) < 2: continue
    pi2 = np.array([results[(beta, m)][0] ** 2 for m in mm]); eta_ = np.array([results[(beta, m)][2] for m in mm])
    s, c = np.polyfit(mm, pi2, 1); m_c = -c / s if s != 0 else float('nan')
    Meta_c = float(np.interp(m_c, mm, eta_)) if mm.min() <= m_c <= mm.max() else float(np.polyval(np.polyfit(mm, eta_, 1), m_c))
    ea = 1.0 / math.sqrt(beta)
    summary.append((ea, Meta_c / ea))
    print(f"{beta:5.2f} {ea:6.3f} {m_c:7.3f} {Meta_c:12.4f} {Meta_c / ea:8.4f} {1 / math.sqrt(math.pi):6.4f}")

if len(summary) >= 2:
    ea = np.array([s[0] for s in summary]); ratio = np.array([s[1] for s in summary])
    s, c = np.polyfit(ea, ratio, 1)
    print(f"\ncontinuum extrapolation (linear in e a): M_eta / e -> {c:.4f}   exact 1/sqrt(pi) = {1 / math.sqrt(math.pi):.4f}   ({100 * (c * math.sqrt(math.pi) - 1):+.1f}%)")
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        os.makedirs(a.figdir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.plot(ea, ratio, 'o', label='lattice, chiral point')
        xx = np.linspace(0, ea.max() * 1.05, 20); ax.plot(xx, s * xx + c, '-', label=f'linear: {c:.3f} at a=0')
        ax.axhline(1 / math.sqrt(math.pi), color='k', ls='--', label='exact e/sqrt(pi)')
        ax.set_xlabel('e a = 1/sqrt(beta)'); ax.set_ylabel('M_eta / e'); ax.set_xlim(0, None); ax.legend(); ax.set_title('Schwinger boson mass vs lattice spacing')
        fig.tight_layout(); fig.savefig(os.path.join(a.figdir, 'schwinger_mass_gap_extrapolation.png'), dpi=130)
        fig2, ax2 = plt.subplots(figsize=(5.5, 4))
        for beta in sorted({k[0] for k in results}):
            ms = sorted(k[1] for k in results if k[0] == beta)
            ax2.errorbar(ms, [results[(beta, m)][0] for m in ms], [results[(beta, m)][1] for m in ms], fmt='o-', label=f'pi, beta={beta:g}')
            ax2.errorbar(ms, [results[(beta, m)][2] for m in ms], [results[(beta, m)][3] for m in ms], fmt='s--', label=f'eta, beta={beta:g}')
        ax2.set_xlabel('bare Wilson mass m a'); ax2.set_ylabel('M a'); ax2.legend(fontsize=7); ax2.set_title('pion and singlet masses vs bare mass')
        fig2.tight_layout(); fig2.savefig(os.path.join(a.figdir, 'schwinger_masses_vs_bare_mass.png'), dpi=130)
        print('figures written to', a.figdir)
    except ImportError:
        print('(matplotlib not available: no figures)')
