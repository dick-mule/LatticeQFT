#!/usr/bin/env python3
"""4D SU(2) (or, with --group su3, SU(3)) finite-temperature deconfinement from
`LatticeQFT --model su2|su3 --dim 4 --nt N`.

Reads one or more CSVs (one per N_t) with the Polyakov-loop columns, prints <|P|> and the
susceptibility chi_P against beta with the susceptibility peak as the estimate of beta_c, and
compares with the known values beta_c(N_t = 2) ~ 1.88, beta_c(N_t = 4) ~ 2.30 (Fingberg, Heller
& Karsch 1993). Also uses the symmetric-lattice Creutz ratio chi(2,2) = sigma a^2 against the
two-loop asymptotic scaling form, a Lambda_L = (beta_0 g^2)^(-beta_1/(2 beta_0^2)) exp(-1/(2 beta_0 g^2))
with g^2 = 4/beta, fitting the single prefactor (sqrt(sigma)/Lambda_L)^2.

Usage: analyze_su2_deconfinement.py su2_nt2.csv su2_nt4.csv [--creutz su2_sym.csv] [--figdir docs/figures]"""
import argparse, csv, math, os
import numpy as np
p = argparse.ArgumentParser()
p.add_argument('csvs', nargs='+'); p.add_argument('--creutz', default=''); p.add_argument('--figdir', default='docs/figures')
p.add_argument('--group', default='su2', choices=['su2', 'su3'])
a = p.parse_args()
N = 2 if a.group == 'su2' else 3
KNOWN = {2: 1.88, 4: 2.30, 6: 2.43} if N == 2 else {4: 5.69, 6: 5.89}
# two-loop SU(N) lattice scale: a Lambda_L = (b0 g^2)^(-b1/(2 b0^2)) exp(-1/(2 b0 g^2)), g^2 = 2N/beta
B0 = 11.0 * N / (48.0 * math.pi ** 2); B1 = 34.0 * N * N / (3.0 * (16.0 * math.pi ** 2) ** 2)
aLambda = lambda beta: (B0 * 2.0 * N / beta) ** (-B1 / (2 * B0 ** 2)) * np.exp(-beta / (4.0 * N * B0))
TAG = f'su{N}'

series = []
for f in a.csvs:
    rows = list(csv.DictReader(open(f)))
    if not rows or 'polyakov_abs' not in rows[0]: print('no Polyakov columns in', f); continue
    nt = int(rows[0]['nt']); b = np.array([float(r['beta']) for r in rows]); P = np.array([float(r['polyakov_abs']) for r in rows])
    dP = np.array([float(r['polyakov_abs_err']) for r in rows]); chi = np.array([float(r['polyakov_chi']) for r in rows])
    ns = int(rows[0]['ns']) if 'ns' in rows[0] else 0
    k = int(np.argmax(chi)); tag = f"N_t = {nt}" + (f", N_s = {ns}" if ns else "")
    print(f"{tag}: chi_P peaks at beta = {b[k]:.3f} (chi = {chi[k]:.2f}; known beta_c ~ {KNOWN.get(nt, float('nan')):.2f}); "
          f"two-loop T_c/Lambda_L = 1/(N_t a Lambda_L) = {1.0 / (nt * aLambda(b[k])):.1f}; <|P|> from {P[0]:.3f} at beta {b[0]:.2f} to {P[-1]:.3f} at beta {b[-1]:.2f}")
    for bb, pp, dd, cc in zip(b, P, dP, chi): print(f"   beta {bb:5.3f}  <|P|> {pp:.4f} +- {dd:.4f}   chi {cc:.3f}")
    series.append((nt, ns, b, P, dP, chi))
scal = None
if a.creutz and os.path.exists(a.creutz):
    rows = list(csv.DictReader(open(a.creutz)))
    b = np.array([float(r['beta']) for r in rows]); sig = np.array([float(r['creutz22']) for r in rows])
    ok = sig > 0
    aL = aLambda
    pref = np.exp(np.mean(np.log(sig[ok]) - 2 * np.log(aL(b[ok]))))
    slope_meas = np.polyfit(b[ok], np.log(sig[ok]), 1)[0]; slope_2loop = -2 * (1 / (4.0 * N * B0) - (B1 / (2 * B0 ** 2)) / np.mean(b[ok]))
    print(f"\nasymptotic scaling of chi(2,2) = sigma a^2 over beta {b.min():.2f}..{b.max():.2f}: measured d ln(sigma a^2)/d beta = {slope_meas:.2f} "
          f"vs two-loop {slope_2loop:.2f}; fitted (sqrt(sigma)/Lambda_L)^2 = {pref:.0f} -> sqrt(sigma)/Lambda_L ~ {math.sqrt(pref):.0f}; "
          f"residual scatter of ln(sigma a^2) about the two-loop curve: {np.std(np.log(sig[ok]) - np.log(pref * aL(b[ok]) ** 2)):.2f}")
    scal = (b, sig, pref, aL)
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    os.makedirs(a.figdir, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for nt, ns, b, P, dP, chi in series:
        lab = f'N_t = {nt}' + (f', N_s = {ns}' if ns else '')
        ax[0].errorbar(b, P, dP, fmt='o-', ms=3, label=lab); ax[1].plot(b, chi, 's-', ms=3, label=lab)
        if nt in KNOWN: ax[0].axvline(KNOWN[nt], color='gray', ls=':'); ax[1].axvline(KNOWN[nt], color='gray', ls=':')
    ax[0].set_xlabel('beta'); ax[0].set_ylabel('<|P|>'); ax[0].set_title('Polyakov loop (dotted: known beta_c)'); ax[0].legend()
    ax[1].set_xlabel('beta'); ax[1].set_ylabel('chi_P'); ax[1].set_title('susceptibility'); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(a.figdir, f'{TAG}_deconfinement.png'), dpi=130)
    if scal:
        b, sig, pref, aL = scal
        fig2, ax2 = plt.subplots(figsize=(5.5, 4)); xx = np.linspace(b.min(), b.max(), 50)
        ax2.semilogy(b, sig, 'o', label='chi(2,2) = sigma a^2'); ax2.semilogy(xx, pref * aL(xx) ** 2, '-', label='two-loop scaling, fitted prefactor')
        ax2.set_xlabel('beta'); ax2.set_ylabel('sigma a^2'); ax2.legend(); ax2.set_title(f'4D SU({N}) asymptotic scaling')
        fig2.tight_layout(); fig2.savefig(os.path.join(a.figdir, f'{TAG}_asymptotic_scaling.png'), dpi=130)
    print('figures written to', a.figdir)
except ImportError:
    print('(matplotlib not available: no figures)')
