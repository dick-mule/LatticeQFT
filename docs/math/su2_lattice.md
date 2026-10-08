# SU(2) Lattice Yang-Mills — Wilson Loops, Creutz Ratios, String Tension

This note covers the SU(2) infrastructure (Phase 3a) and the physics
observables it supports (Phase 3b). The data in the tables comes from the
`--model su2` headless driver.

## Group structure

SU(2) elements are parameterized by unit quaternions
`q = (s, v) ∈ S³`, with `s² + |v|² = 1`:

```
U(q) = s · I + i · v · σ ∈ SU(2)
```

For the matrix product `(s₁, v₁) · (s₂, v₂)` we use the identity
`σ_i σ_j = δ_{ij} I + i ε_{ijk} σ_k`, which gives

```
(s₁ s₂ − v₁·v₂ ,  s₁ v₂ + s₂ v₁ − v₁ × v₂)
```

This formula is implemented in `su2::multiply` and is correct *both* for
SU(2) elements and for arbitrary linear combinations of them (such as the
gauge "staple") — the only difference is that for general linear
combinations `s² + |v|² ≠ 1`.

Useful identities:
- `tr U = 2 s`
- `Re tr(U V) = 2(s_U s_V − v_U · v_V)`
- `U† = (s, −v)` — for unit `U` this is also `U⁻¹`

## Wilson plaquette action

```
S[U] = β Σ_□ (1 − ½ tr U_□)
     = β Σ_□ (1 − [U_□].s)
```

The plaquette `U_□(x; μ, ν) = U_μ(x) · U_ν(x+ê_μ) · U_μ†(x+ê_ν) · U_ν†(x)`
is computed by three `su2::multiply` calls; we only need `.s` for the
action since `½ tr U_□ = [U_□].s`.

## Local update — Metropolis

Propose `U → ΔU · U` with the Lie-algebra Gaussian
`ΔU = exp(i θ · σ / 2)`, where `θ ∼ N(0, σ² · I)`. The symmetry
`θ → −θ` corresponds exactly to `ΔU → ΔU†`, so the proposal is symmetric
without any modification — no Hastings ratio correction needed.

The action change collapses to

```
ΔS = − (β / 2) · [Re tr(U_new · A_μ(x)) − Re tr(U_old · A_μ(x))]
```

with `A_μ(x)` the staple (sum of 2(Dim − 1) SU(2)-valued matrices
defined in `staple(field, site, mu)`).

## Local update — Kennedy–Pendleton heat-bath

The single-link conditional at fixed staple is

```
P(U | A) ∝ exp((β/2) Re tr(U · A)) = exp(β k w_0(U · V))
```

with `k = |A|`, `V = A / k ∈ SU(2)`, and `w_0` the scalar (quaternion)
part of `W = U · V`. The conditional on `W` is

```
P(W) dW ∝ exp(β k w_0) · √(1 − w_0²) · dΩ_{n̂}
```

— a 1D distribution on `w_0` times a uniform on the level-set `S²` for
the vector part. Kennedy–Pendleton (Phys. Lett. B 156, 393, 1985)
rejection-sample `w_0` via the parameterization `w_0 = 1 − 2λ²`:

```
loop {
    X₁, X₂, X₃, X₄ ∼ Uniform(0, 1)
    λ² = -[ ln X₁ + cos²(2π X₂) · ln X₃ ] / (2 β k)
    if X₄² ≤ 1 − λ² : break
}
```

The acceptance is ≥ 0.95 for all `β k`. After accepting `λ²`, draw `n̂`
uniformly on S² for the vector part, form `W`, and return
`U_new = W · V†`.

## Wilson loops and the area law

The Wilson loop is the trace of the path-ordered product of links around
a closed loop. For a rectangle of sides `R` and `T` in directions `(μ, ν)`
starting at site `x`:

```
W(x; μ, ν, R, T) = ½ tr [ Π U_μ Π U_ν Π U_μ†  Π U_ν† ]
```

Gauge invariance is automatic from the cyclic trace: under
`U_μ(x) → Ω(x) U_μ(x) Ω†(x + ê_μ)`, the gauge rotations at every vertex
cancel in pairs along the loop, leaving `½ tr W` unchanged. The CI test
`SU2WilsonLoop.GaugeInvariantUnderRandomTransformation` verifies this to
1e-9 on a random SU(2) gauge transformation.

In the confining phase the Wilson loop obeys an **area law**:

```
⟨W(R, T)⟩ ~ exp(−σ · R T − μ · (R + T) + const)
```

The `σ` term is the *string tension*: the energy per unit length of the
chromoelectric flux tube connecting a static quark–antiquark pair, and
the order parameter of confinement. The perimeter `μ` piece is the
self-energy of the static charges and is UV-divergent in the continuum.

## Creutz ratios

The standard way to subtract off the perimeter piece and isolate the
string tension is the **Creutz ratio**:

```
χ(R, T) = − log[ ⟨W(R, T)⟩ · ⟨W(R−1, T−1)⟩ / (⟨W(R−1, T)⟩ · ⟨W(R, T−1)⟩) ]
```

Under the area-law Ansatz `log W = −σ R T − μ(R + T) + const`, the
linear-in-(R, T) and constant terms cancel in the ratio:

```
χ(R, T) = −[(RT − (R−1)T − R(T−1) + (R−1)(T−1)] · (−σ)
        − [(R+T) − ((R−1)+T) − (R+(T−1)) + ((R−1)+(T−1))] · (−μ)
        = σ        (the (R,T) factor) + 0      (the perimeter factor)
        = σ
```

at large `(R, T)`. Along the diagonal `T = R` the convergence is
fastest; at finite `R` there are corrections from higher-loop terms in
the static potential.

## Phase-3b validation data

### 3D SU(2), heat-bath, L = 8, ~400 trajectories per β

| β | ⟨½ tr U_□⟩ | W(2,1) | W(2,2) | χ(2,2) |
|---|------------|--------|--------|--------|
| 1.0 | 0.242 | 0.059 | 0.003 | **1.513** |
| 2.0 | 0.455 | 0.208 | 0.044 | **0.768** |
| 3.0 | 0.623 | 0.398 | 0.168 | **0.416** |
| 4.0 | 0.727 | 0.544 | 0.320 | **0.237** |
| 5.0 | 0.786 | 0.637 | 0.438 | **0.161** |
| 6.0 | 0.825 | 0.698 | 0.525 | **0.120** |

- χ(2,2) > 0 at every β — confinement at all couplings, as expected for
  3D SU(2) (no bulk deconfinement transition in 3D).
- χ(2,2) decreases monotonically with β — the lattice string tension
  shrinks in lattice units toward the continuum, consistent with the
  3D Yang-Mills scale identification.
- W(2,1) ≈ W(1,2) to within statistical error every row — μ ↔ ν
  symmetry of the Wilson loop is an unforced cross-check; the
  path-traversal code computes them separately.

### 4D SU(2), heat-bath, L = 6, ~300 trajectories per β

| β | ⟨½ tr U_□⟩ | W(2,1) | W(2,2) | χ(2,2) |
|---|------------|--------|--------|--------|
| 1.5 | 0.362 | 0.133 | 0.017 | **1.007** |
| 2.0 | 0.501 | 0.258 | 0.072 | **0.609** |
| 2.5 | 0.652 | 0.457 | 0.259 | **0.212** |
| 3.0 | 0.724 | 0.559 | 0.381 | **0.129** |
| 3.5 | 0.768 | 0.624 | 0.460 | **0.100** |

- Same pattern: χ(2,2) positive everywhere, dropping with β.
- The steepest drop is around β ≈ 2.0–2.5 where 4D SU(2) has its
  "crossover" between strong and weak coupling on small lattices.

The Creutz ratio at fixed (R, T) is a finite-`L` estimate of σ; a full
scaling analysis would track `χ(R, R)` as a function of `R` and `β`,
identify the asymptotic-freedom scale where physical-units σ becomes
β-independent, and extract the lattice spacing `a(β)` from the
identification `σ_phys ≈ (440 MeV)²` (in 4D) or the corresponding
analytic continuum result in 3D.

## References

- K. G. Wilson, *Phys. Rev. D* 10, 2445 (1974). The original Wilson loop.
- M. Creutz, *Phys. Rev. D* 21, 2308 (1980). Numerical estimation of the
  SU(2) string tension via Wilson loop ratios.
- A. D. Kennedy and B. J. Pendleton, *Phys. Lett. B* 156, 393 (1985).
  The improved SU(2) heat-bath algorithm.
- C. Gattringer and C. B. Lang, *Quantum Chromodynamics on the Lattice*,
  ch. 3 (gauge actions), ch. 4 (string tension and Wilson loops).
