#!/usr/bin/env python3
"""
Analyze the dynamical-Schwinger HMC grid:

  - Compute the continuum Schwinger condensate symbolically via SymPy
        ⟨ψ̄ψ⟩ / g = -e^γ / (2 π^{3/2})    ≈ -0.15993...
  - Plot the measured (bare-lattice) condensate ⟨ψ̄ψ⟩_lat vs m at each β.
  - Plot the same data sliced the other way: vs β at each m.
  - Print a tidy summary table.

The lattice and continuum quantities are *not* equal: ⟨ψ̄ψ⟩_lat has a
positive, m-dependent additive renormalization from the Wilson term, plus
a multiplicative renormalization that depends on β and the conventions
used to identify the lattice spacing. The point of these plots is the
shape and trends — monotonicity in m, β-dependence — not pointwise
agreement with the continuum value.

See ``docs/math/schwinger_continuum.md`` for the analytic background and
a discussion of what would be required for a true a→0 extrapolation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")  # headless-safe; PNG output only
    import matplotlib.pyplot as plt
    import sympy as sp
except ImportError as e:
    print(f"missing Python dependency: {e}", file=sys.stderr)
    print("install with: pip install numpy pandas matplotlib sympy", file=sys.stderr)
    sys.exit(2)

REPO_ROOT = Path(__file__).resolve().parent.parent


def schwinger_continuum_condensate():
    """Symbolic and numerical value of  ⟨ψ̄ψ⟩ / g  in the massless Schwinger model.

    Result (Schwinger 1962, refined by Smilga; see docs/math/schwinger_continuum.md):

        ⟨ψ̄ψ⟩ / g = - e^γ / (2 π^{3/2})
    """
    expr = -sp.exp(sp.EulerGamma) / (2 * sp.pi ** sp.Rational(3, 2))
    return expr, float(expr)


def plot_condensate_vs_mass(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for beta, sub in df.groupby("beta"):
        sub = sub.sort_values("mass")
        ax.errorbar(
            sub["mass"], sub["condensate"], yerr=sub["condensate_se"],
            marker="o", capsize=3, linewidth=1.2,
            label=fr"$\beta = {beta:.2f}$",
        )
    ax.set_xlabel("bare lattice fermion mass $m$")
    ax.set_ylabel(r"$\langle \bar\psi \psi \rangle_{\rm lat}$  (lattice units)")
    ax.set_title("Dynamical-Schwinger HMC: condensate vs mass")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  saved {out_path}", file=sys.stderr)


def plot_condensate_vs_beta(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for mass, sub in df.groupby("mass"):
        sub = sub.sort_values("beta")
        ax.errorbar(
            sub["beta"], sub["condensate"], yerr=sub["condensate_se"],
            marker="s", capsize=3, linewidth=1.2,
            label=fr"$m = {mass:.2f}$",
        )
    ax.set_xlabel(r"inverse coupling $\beta = 1 / g^2 a^2$")
    ax.set_ylabel(r"$\langle \bar\psi \psi \rangle_{\rm lat}$  (lattice units)")
    ax.set_title("Dynamical-Schwinger HMC: condensate vs gauge coupling")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  saved {out_path}", file=sys.stderr)


def plot_plaquette_vs_beta(df: pd.DataFrame, out_path: Path) -> None:
    """Sanity-check plot: the gauge sector should still reproduce the Bessel
    ratio I_1(β)/I_0(β) (modulo small fermion-determinant corrections from
    the dynamical sea quarks)."""
    from scipy.special import i0, i1  # type: ignore[import-untyped]
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for mass, sub in df.groupby("mass"):
        sub = sub.sort_values("beta")
        ax.errorbar(
            sub["beta"], sub["plaq"], yerr=sub["plaq_err"],
            marker="o", capsize=3, linewidth=1.2,
            label=fr"$m = {mass:.2f}$ (HMC)",
        )
    beta_smooth = np.linspace(df["beta"].min() * 0.9, df["beta"].max() * 1.05, 200)
    ax.plot(beta_smooth, i1(beta_smooth) / i0(beta_smooth),
            "k--", linewidth=1.2, label=r"$I_1(\beta) / I_0(\beta)$ (pure-gauge 2D U(1))")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\langle \cos\theta_{\rm plaq} \rangle$")
    ax.set_title("Plaquette: dynamical HMC vs pure-gauge Bessel ratio")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  saved {out_path}", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv",     default=str(REPO_ROOT / "data" / "schwinger_grid.csv"))
    p.add_argument("--out-dir", default=str(REPO_ROOT / "docs" / "figures"))
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"csv not found: {csv_path}", file=sys.stderr)
        print("run scripts/run_schwinger_grid.py first.", file=sys.stderr)
        return 1
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    # Force numeric columns.
    for c in ["beta", "mass", "plaq", "plaq_err",
              "condensate", "condensate_se", "acceptance", "avg_dH"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["beta", "mass", "condensate"]).reset_index(drop=True)

    expr, value = schwinger_continuum_condensate()
    print()
    print("Continuum Schwinger condensate (massless limit):")
    print(f"    ⟨ψ̄ψ⟩ / g  =  {sp.pretty(expr)}")
    print(f"             ≈  {value:.7f}")
    print()
    print(f"loaded {len(df)} measurement rows from {csv_path}")
    print()

    plot_condensate_vs_mass(df, out_dir / "schwinger_condensate_vs_mass.png")
    plot_condensate_vs_beta(df, out_dir / "schwinger_condensate_vs_beta.png")
    try:
        plot_plaquette_vs_beta(df, out_dir / "schwinger_plaquette_vs_beta.png")
    except ImportError:
        print("  scipy not available — skipping Bessel-ratio plot", file=sys.stderr)

    # Tidy summary table.
    summary_cols = [c for c in
                    ["beta", "mass", "plaq", "plaq_err",
                     "condensate", "condensate_se", "acceptance", "avg_dH"]
                    if c in df.columns]
    summary = df[summary_cols].sort_values(["beta", "mass"]).reset_index(drop=True)
    print("Summary table:")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
