#!/usr/bin/env python3
"""Schwinger-model mass gap from the pseudoscalar correlators written by
`LatticeQFT --model schwinger-hmc --corr-out` (via scripts/run_schwinger_mass_grid.py).

Per (beta, mass): effective masses M_pi (connected) and M_eta (singlet) averaged over a plateau
window. The pion uses the periodic cosh form C(t)/C(t+1). The singlet correlator carries a
t-independent constant on top of its cosh: in a fixed topological sector <psi-bar gamma5 psi> is
nonzero (index theorem), so the disconnected piece contains (1/V) <Q^2>-type terms that do not
decay. The singlet mass is therefore taken from the difference correlator D(t) = C(t) - C(t+1),
which cancels any constant, via D(t)/D(t+1).

Per beta: the chiral point m_c from a linear fit of M_pi^2 against the bare mass on the branch
that is continuous with the heavy-quark side (Wilson fermions: past m_c the pion mass rises
again, those points are dropped), and M_eta at m_c from a weighted linear fit on that branch.
Across beta: M_eta / e against e a = 1 / sqrt(beta), extrapolated linearly to a -> 0 and compared
with the exact Schwinger boson, M / e = 1 / sqrt(pi) = 0.5642.

Usage: analyze_schwinger_mass.py data/schwinger_mass_grid.csv [--tmin 3] [--figdir docs/figures]"""
import argparse, csv, math, os, sys
from collections import defaultdict
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('csv'); p.add_argument('--tmin', type=int, default=3); p.add_argument('--figdir', default='docs/figures')
p.add_argument('--exclude', default='', help='comma list of beta:mass points to drop, e.g. 2:-0.4,2:-0.3')
p.add_argument('--max-dh', type=float, default=0.5, help='drop points whose HMC <dH> exceeds this (from the _quality.csv side file)')
a = p.parse_args()

rows = list(csv.DictReader(open(a.csv)))
bad = set()
for item in [x for x in a.exclude.split(',') if x]:
    b, m = item.split(':'); bad.add((round(float(b), 5), round(float(m), 5)))
qfile = os.path.splitext(a.csv)[0] + '_quality.csv'
if os.path.exists(qfile):
    for r in csv.DictReader(open(qfile)):
        if abs(float(r['avg_dH'])) > a.max_dh or float(r['acceptance']) < 0.6:
            bad.add((round(float(r['beta']), 5), round(float(r['mass']), 5)))
if bad: print('dropping (unstable HMC / excluded):', sorted(bad))
rows = [r for r in rows if (round(float(r['beta']), 5), round(float(r['mass']), 5)) not in bad]
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

def meff_sub(Ct, Ct1, Ct2, t):
    """Effective mass from the difference correlator D(t) = C(t) - C(t+1), which removes any
    t-independent constant: D(t)/D(t+1) = [cosh(Mu) - cosh(Mv)] / [cosh(Mv) - cosh(Mw)]."""
    D0, D1 = Ct - Ct1, Ct1 - Ct2
    if not (D0 > 0 and D1 > 0): return float('nan')
    tgt = D0 / D1; u, v, w = t - L / 2, t + 1 - L / 2, t + 2 - L / 2
    def f(M):
        den = math.cosh(M * v) - math.cosh(M * w)
        return (math.cosh(M * u) - math.cosh(M * v)) / den if den != 0 else float('inf')
    lo, hi = 1e-4, 10.0
    if (f(lo) - tgt) * (f(hi) - tgt) > 0: return float('nan')
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if (f(mid) - tgt) * (f(lo) - tgt) <= 0: hi = mid
        else: lo = mid
    return 0.5 * (lo + hi)

def plateau(C, E, t0, t1, subtracted=False):
    """Error-weighted mean of the effective mass over t in [t0, t1); the error of each M_eff is
    propagated from the correlator errors by finite differences."""
    vals, ws = [], []
    for t in range(t0, t1):
        if subtracted:
            if t + 2 >= len(C): break
            g = lambda d0, d1, d2: meff_sub(C[t] + d0, C[t + 1] + d1, C[t + 2] + d2, t)
            M = g(0, 0, 0)
            if math.isnan(M): continue
            parts = [abs(g(E[t], 0, 0) - g(-E[t], 0, 0)), abs(g(0, E[t + 1], 0) - g(0, -E[t + 1], 0)), abs(g(0, 0, E[t + 2]) - g(0, 0, -E[t + 2]))]
            if any(math.isnan(x) for x in parts): continue
            dM = 0.5 * math.sqrt(sum(x * x for x in parts))
        else:
            M = meff(C[t], C[t + 1], t)
            if math.isnan(M): continue
            dM = 0.5 * (abs(meff(C[t] + E[t], C[t + 1], t) - meff(C[t] - E[t], C[t + 1], t)) +
                        abs(meff(C[t], C[t + 1] + E[t + 1], t) - meff(C[t], C[t + 1] - E[t + 1], t)))
        if math.isnan(dM) or dM <= 0: continue
        vals.append(M); ws.append(1.0 / dM ** 2)
    if not vals: return float('nan'), float('nan')
    w = np.array(ws); v = np.array(vals)
    return float((w * v).sum() / w.sum()), float(1.0 / math.sqrt(w.sum()))

def coshfit(C, E, t0, with_const):
    """Weighted least-squares fit of C(t) = A cosh(M (t - L/2)) [+ B] over t in [t0, L - t0].
    For fixed M the model is linear in (A, B), so the search is one-dimensional in M (golden
    section on chi^2); the error on M is from delta chi^2 = 1 (parabolic)."""
    ts = np.arange(t0, L - t0 + 1); y = np.array([C[t] for t in ts]); w = 1.0 / np.array([E[t] for t in ts]) ** 2
    if not np.all(np.isfinite(w)) or len(ts) < 4: return float('nan'), float('nan'), float('nan')
    def chi2(M):
        cols = [np.cosh(M * (ts - L / 2))] + ([np.ones_like(ts, dtype=float)] if with_const else [])
        A = np.vstack(cols).T
        try: th = np.linalg.solve(A.T @ (w[:, None] * A), A.T @ (w * y))
        except np.linalg.LinAlgError: return float('inf'), None
        r = A @ th - y
        return float((w * r * r).sum()), th
    # coarse scan then golden section
    grid = np.linspace(0.02, 2.5, 125); vals = [chi2(M)[0] for M in grid]
    k = int(np.argmin(vals)); lo, hi = grid[max(0, k - 1)], grid[min(len(grid) - 1, k + 1)]
    g = (math.sqrt(5) - 1) / 2
    for _ in range(60):
        m1, m2 = hi - g * (hi - lo), lo + g * (hi - lo)
        if chi2(m1)[0] < chi2(m2)[0]: hi = m2
        else: lo = m1
    M = 0.5 * (lo + hi); c0, th = chi2(M)
    if th is None or th[0] <= 0: return float('nan'), float('nan'), float('nan')
    h = 1e-3; curv = (chi2(M + h)[0] - 2 * c0 + chi2(M - h)[0]) / h ** 2
    dM = math.sqrt(2.0 / curv) if curv > 0 else float('nan')
    dof = len(ts) - (3 if with_const else 2)
    return M, dM, c0 / max(1, dof)

def wfit(x, y, dy):
    """Weighted linear fit y = s x + c; returns s, c, cov."""
    w = 1.0 / np.asarray(dy) ** 2
    A = np.vstack([x, np.ones_like(x)]).T
    cov = np.linalg.inv(A.T @ (w[:, None] * A)); sc = cov @ (A.T @ (w * y))
    return sc[0], sc[1], cov

results = {}
print(f"{'beta':>5} {'mass':>6} {'M_pi fit':>8} {'+-':>6} {'M_pi plat':>8} {'+-':>6} {'M_eta+B':>8} {'+-':>6} {'chi2/d':>6} {'M_eta B=0':>8} {'+-':>6} {'chi2/d':>6} {'M_eta diff':>8} {'+-':>6}")
print("(M_pi fit: cosh fit over t in [tmin, L-tmin]; plat: plateau of C(t)/C(t+1); M_eta+B: A cosh + constant; B=0: pure cosh; diff: from C(t)-C(t+1))")
for (beta, m), tab in sorted(pts.items()):
    ts = sorted(tab)
    conn = [tab[t][0] for t in ts]; econ = [tab[t][1] for t in ts]
    eta = [tab[t][4] for t in ts]; eeta = [tab[t][5] for t in ts]
    t1 = L // 2 - 1
    Mpi, dMpi = plateau(conn, econ, a.tmin, t1)
    Mpi_f, dMpi_f, q_pi = coshfit(conn, econ, a.tmin, False)
    Meta, dMeta, q_eta = coshfit(eta, eeta, a.tmin, True)
    Meta0, dMeta0, q_eta0 = coshfit(eta, eeta, a.tmin, False)
    Meta_d, dMeta_d = plateau(eta, eeta, a.tmin, t1, subtracted=True)
    results[(beta, m)] = (Mpi_f, dMpi_f, Meta, dMeta, Meta0, dMeta0)
    print(f"{beta:5.2f} {m:6.2f} {Mpi_f:8.4f} {dMpi_f:6.4f} {Mpi:8.4f} {dMpi:6.4f} {Meta:8.4f} {dMeta:6.4f} {q_eta:6.2f} {Meta0:8.4f} {dMeta0:6.4f} {q_eta0:6.2f} {Meta_d:8.4f} {dMeta_d:6.4f}")

# Per beta: chiral point from M_pi^2 vs m, then M_eta at m_c.
print("\nper beta: chiral point and the Schwinger boson")
print(f"{'beta':>5} {'e a':>6} {'m_c':>14} {'M_eta(m_c) a':>16} {'M_eta/e':>16} {'exact':>6}")
summary = []
for beta in sorted({k[0] for k in results}):
    ms = sorted((k[1] for k in results if k[0] == beta and not math.isnan(results[k][0])), reverse=True)
    # chiral branch: walk down from the heaviest mass while M_pi keeps falling
    branch = []
    for m in ms:
        if branch and results[(beta, m)][0] >= results[(beta, branch[-1])][0]: break
        branch.append(m)
    if len(branch) < 3: continue
    mm = np.array(branch)
    Mpi = np.array([results[(beta, m)][0] for m in mm]); dMpi = np.array([results[(beta, m)][1] for m in mm])
    s, c, cov = wfit(mm, Mpi ** 2, 2 * Mpi * dMpi)
    m_c = -c / s
    dm_c = math.sqrt(cov[1, 1] / s ** 2 + cov[0, 0] * c ** 2 / s ** 4 - 2 * cov[0, 1] * c / s ** 3) if cov[0, 0] > 0 else float('nan')
    ok = np.array([not math.isnan(results[(beta, m)][2]) for m in mm])
    if ok.sum() < 2: continue
    Meta = np.array([results[(beta, m)][2] for m in mm])[ok]; dMeta = np.array([results[(beta, m)][3] for m in mm])[ok]
    se, ce, cove = wfit(mm[ok], Meta, dMeta)
    Meta_c = se * m_c + ce
    dMeta_c = math.sqrt(max(0.0, m_c ** 2 * cove[0, 0] + cove[1, 1] + 2 * m_c * cove[0, 1] + (se * dm_c) ** 2))
    ea = 1.0 / math.sqrt(beta)
    summary.append((ea, Meta_c / ea, dMeta_c / ea))
    print(f"{beta:5.2f} {ea:6.3f} {m_c:7.3f}({dm_c:.3f}) {Meta_c:9.4f}({dMeta_c:.4f}) {Meta_c / ea:8.4f}({dMeta_c / ea:.4f}) {1 / math.sqrt(math.pi):6.4f}   branch m = {[float(x) for x in mm]}")

print("\nsame, with the singlet fitted to a pure cosh (no constant):")
for beta in sorted({k[0] for k in results}):
    ms = sorted((k[1] for k in results if k[0] == beta and not math.isnan(results[k][0])), reverse=True)
    branch = []
    for m in ms:
        if branch and results[(beta, m)][0] >= results[(beta, branch[-1])][0]: break
        branch.append(m)
    if len(branch) < 3: continue
    mm = np.array(branch); Mpi = np.array([results[(beta, m)][0] for m in mm]); dMpi = np.array([results[(beta, m)][1] for m in mm])
    s_, c_, cov = wfit(mm, Mpi ** 2, 2 * Mpi * dMpi); m_c = -c_ / s_
    ok = np.array([not math.isnan(results[(beta, m)][4]) for m in mm])
    if ok.sum() < 2: continue
    Me = np.array([results[(beta, m)][4] for m in mm])[ok]; dMe = np.array([results[(beta, m)][5] for m in mm])[ok]
    se, ce, cove = wfit(mm[ok], Me, dMe); Mc = se * m_c + ce; dMc = math.sqrt(max(0.0, m_c ** 2 * cove[0, 0] + cove[1, 1] + 2 * m_c * cove[0, 1]))
    ea = 1.0 / math.sqrt(beta)
    print(f"{beta:5.2f} {ea:6.3f} {m_c:7.3f} {Mc:9.4f}({dMc:.4f}) {Mc / ea:8.4f}({dMc / ea:.4f}) {1 / math.sqrt(math.pi):6.4f}")

if len(summary) >= 2:
    ea = np.array([s[0] for s in summary]); ratio = np.array([s[1] for s in summary]); dratio = np.array([s[2] for s in summary])
    s, c, cov = wfit(ea, ratio, dratio); dc = math.sqrt(cov[1, 1])
    chi2 = float(((((s * ea + c) - ratio) / dratio) ** 2).sum()) if len(ea) > 2 else 0.0
    print(f"\ncontinuum extrapolation (weighted linear in e a, {len(ea)} points): M_eta / e -> {c:.4f} +- {dc:.4f}   "
          f"exact 1/sqrt(pi) = {1 / math.sqrt(math.pi):.4f}   ({100 * (c * math.sqrt(math.pi) - 1):+.1f}%, {abs(c - 1 / math.sqrt(math.pi)) / dc:.1f} sigma); chi2/dof = {chi2 / max(1, len(ea) - 2):.2f}")
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        os.makedirs(a.figdir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.errorbar(ea, ratio, dratio, fmt='o', label='lattice, chiral point')
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
