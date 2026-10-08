# 2D φ⁴ Scalar Field Theory — Phase 1 Reference

The first genuinely-quantum-field-theoretic model in the phase plan: a single
real scalar field φ(x) ∈ ℝ on a Dim-dimensional periodic lattice, with the
standard λφ⁴ self-interaction.

## Continuum action

```
S[φ] = ∫ dᴰx [ ½ (∂φ)² + ½ m² φ² + (λ/4!) φ⁴ ]
```

For `m² > 0` the theory is in its symmetric (paramagnetic) phase; the
two-point function decays exponentially with a mass m, and the field
fluctuates around zero. For `m² < m_c²(λ) < 0` the Z₂ symmetry
φ → -φ breaks spontaneously and the field develops a vacuum expectation
value `⟨φ⟩ ≠ 0`. The critical line `m_c²(λ)` is the analog of the Ising
β = β_c transition.

## Lattice discretization

With lattice spacing a = 1 and forward differences `∂_μ φ(x) = φ(x + ê_μ) − φ(x)`:

```
S[φ] = Σ_x [ ½ Σ_μ (φ_{x+ê_μ} − φ_x)²
           + ½ m² φ_x²
           + (λ/24) φ_x⁴ ]
```

Each forward link `(x, x + ê_μ)` appears once in the sum; we never walk
backward links to avoid double-counting.

## Local Metropolis update

Propose `φ_x → φ_x' = φ_x + δ` with `δ ~ N(0, σ²)`. The terms in S that
change are the D forward kinetic links from x, the D backward kinetic
links into x, and the on-site mass + quartic. Expanding:

```
ΔS = -δ · S_nn(x) + D · (φ'² − φ²)
                  + ½ m² (φ'² − φ²)
                  + (λ/24) (φ'⁴ − φ⁴)

where  S_nn(x) = Σ_μ (φ_{x+ê_μ} + φ_{x-ê_μ})
       D       = Dim
```

Accept with probability `min(1, exp(-ΔS))`. The step size σ is tuned during
thermalization by `mc::autoTuneStepSize` to a target acceptance of 0.5.

## Closed-form free-field benchmark (λ = 0)

For `λ = 0` the action is purely Gaussian and exactly solvable on the lattice.
In momentum space with `p_μ ∈ {2π k_μ / L_μ : k_μ = 0, …, L_μ − 1}`:

```
S = ½ Σ_p [ 4 Σ_μ sin²(p_μ / 2) + m² ] |φ̃_p|²

⟨|φ̃_p|²⟩  = 1 / (4 Σ_μ sin²(p_μ / 2) + m²)

⟨φ_x²⟩    = (1/V) Σ_p ⟨|φ̃_p|²⟩
          = (1/V) Σ_p  1 / (4 Σ_μ sin²(p_μ / 2) + m²)
```

This is what `phi4::freeFieldPhi2(lattice, m²)` computes. The Phase-1
CI test asserts that the MC measurement matches this closed form within
8 % on `L = 8`, `m² = 1`, λ = 0 after thermalization + tuning.

## Phase-1 validation summary

| Test                                | Bound                          | What it certifies                                                  |
|-------------------------------------|--------------------------------|--------------------------------------------------------------------|
| Cold start φ ≡ 0                    | S = 0 exactly                  | `totalAction` accounting                                           |
| δS(site) vs S_after − S_before      | < 1e-9 absolute                | `delta_action` derivation matches `totalAction` literally          |
| Free-field ⟨φ²⟩ vs analytic         | < 8 % relative                 | MC engine reproduces Gaussian propagator on a finite lattice       |
| `autoTuneStepSize` lands in target  | acc ∈ [0.40, 0.60]             | Adaptive step-size tuner works at arbitrary starting σ             |

## When HMC starts paying off

For Phase 1 single-site Metropolis is sufficient: the action is local with
nearest-neighbor coupling, and the acceptance tuner keeps σ near 50 %.
Critical slowing down only really bites near `m_c²(λ)` on large lattices;
the Phase-1 CI runs are far enough from criticality to be MC-comfortable.

HMC becomes the right tool when we add fermions (Phase 3, Schwinger model):
the fermion determinant induces non-local effective couplings, single-site
moves no longer have a local ΔS, and importance sampling has to act on the
full configuration through molecular-dynamics trajectories.

## References

- I. Montvay and G. Münster, *Quantum Fields on a Lattice*, Cambridge
  Monographs on Mathematical Physics, 1994 (chapter on φ⁴).
- C. Gattringer and C. B. Lang, *Quantum Chromodynamics on the Lattice*,
  Lecture Notes in Physics 788, Springer, 2010 (chapter 1: scalar fields
  as warm-up).
- W. Loinaz and R. S. Willey, "Monte Carlo simulation calculation of the
  critical coupling constant for two-dimensional continuum φ⁴ theory",
  *Phys. Rev. D* 58, 076003 (1998) — quantitative `m_c²(λ)` numbers.
