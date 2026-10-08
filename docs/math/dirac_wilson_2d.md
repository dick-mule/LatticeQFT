# 2D Wilson-Dirac Operator and Conjugate-Gradient Solver — Phase 3 Reference

The first fermion phase. This note covers the discretization, the explicit
γ-matrix conventions, the projector trick that makes the operator cheap,
the γ⁵-Hermiticity identity, and the CG solver that inverts `D†D`.

## γ-matrix convention (2D Euclidean)

```
γ⁰ = σ_x = [[0, 1],  [1, 0]]
γ¹ = σ_y = [[0, -i], [i, 0]]
γ⁵ = γ⁰γ¹ / (−i) = σ_z = [[1, 0], [0, −1]]
```

All three are Hermitian. `γ⁵` anticommutes with both `γ⁰` and `γ¹` and has
real eigenvalues ±1 (the chirality projectors are `P_± = ½(1 ± γ⁵)`).

## Wilson-Dirac operator

A two-component Dirac spinor `ψ(x) ∈ ℂ²` lives at every site. With a U(1)
gauge background `U_μ(x) = exp(i θ_μ(x))` and bare mass `m`:

```
(D ψ)(x) = (m + 2) ψ(x)
         − ½ Σ_μ [ (1 − γ_μ) U_μ(x)        ψ(x + ê_μ)
                 + (1 + γ_μ) U_μ†(x − ê_μ) ψ(x − ê_μ) ].
```

The `+ 2` (in general `+ Dim · r` with Wilson parameter `r = 1`) is the
**Wilson term**, added to eliminate the doublers from the naïve fermion
action. It breaks chiral symmetry explicitly — the standard Wilson trade-off.

## Projector trick

The `(1 ± γ_μ)` matrices have rank 1 in 2D. Each one collapses a 2-spinor to
a single complex scalar plus a fixed write-out pattern:

| μ | projector | scalar combination       | write-out at (component 0, 1) |
|---|-----------|--------------------------|--------------------------------|
| 0 | (1 − γ⁰)  | ψ_0 − ψ_1                | (+1, −1)                       |
| 0 | (1 + γ⁰)  | ψ_0 + ψ_1                | (+1, +1)                       |
| 1 | (1 − γ¹)  | ψ_0 + i ψ_1              | (+1, −i)                       |
| 1 | (1 + γ¹)  | ψ_0 − i ψ_1              | (+1, +i)                       |

So each Wilson-Dirac hopping term is one complex multiply (the link factor),
one scalar combination, and two write-outs — not a full 2×2 spinor mat-vec.
The same trick collapses 4×4 to 2-vector in 4D.

## γ⁵-Hermiticity

The Wilson-Dirac operator satisfies

```
γ⁵ D γ⁵ = D†.
```

Equivalently, `D†` differs from `D` only by the substitution `γ_μ → −γ_μ`,
which exchanges the `(1 − γ_μ)` and `(1 + γ_μ)` projectors at every hop.
We exploit this directly in the implementation: `applyDagger(...)` is the
same code path as `apply(...)` with a single `dagger` flag that selects the
swapped projectors. This guarantees `D†` is bit-for-bit consistent with `D`
and saves writing a second operator.

The CI test `WilsonDirac2D.IsGamma5Hermitian` verifies the operator-level
identity `⟨x, D y⟩ = ⟨D† x, y⟩` against the explicit `applyDagger`.

## D†D and the conjugate-gradient solver

To invert `D` we instead invert the Hermitian positive-definite operator

```
A = D† D
```

via conjugate gradient. Each CG iteration applies `A` once (which is two
calls — one to `apply`, one to `applyDagger`). The Hestenes-Stiefel
iteration is implemented exactly as in the textbook:

```
r ← b − A x        p ← r        ρ ← ⟨r, r⟩
loop k:
  v ← A p
  α ← ρ / Re⟨p, v⟩
  x ← x + α p
  r ← r − α v
  ρ_new ← ⟨r, r⟩
  if √(ρ_new / ⟨b, b⟩) < tol: converged
  β ← ρ_new / ρ
  p ← r + β p
  ρ ← ρ_new
```

The solver template is parameterized on the `apply_A` callable, so the same
code drives `D†D`, a gauge Laplacian, or any other Hermitian PSD operator —
the deflation wrapper we add at Phase 5 will sit on top of the same
interface without touching the inner loop.

### Stopping criterion

Relative residual: `‖r_k‖ / ‖b‖ < tol`. The test suite uses
`tol = 1e-9`, on hot U(1) backgrounds at `m = 0.5` on `L = 8`. CG
converges in well under 100 iterations there; near-massless or near-critical
backgrounds will need many more (and eventually the deflation acceleration).

### Safety check

If `Re⟨p, A p⟩ ≤ 0` the iteration cannot continue (the operator is no
longer numerically positive). The solver returns `converged = false`
rather than producing nonsense. `ConjugateGradient.ReportsNonConvergeOnSingularOperator`
exercises this branch with a projector operator that has a null space.

## Phase-3 Round-1 validation summary

| Test                                            | Bound           | What it certifies                                                  |
|-------------------------------------------------|-----------------|--------------------------------------------------------------------|
| D(α x + β y) = α Dx + β Dy                      | < 1e-12         | Linearity of `apply`                                               |
| ⟨x, D y⟩ = ⟨D† x, y⟩                            | < 1e-10         | `applyDagger` matches inner-product Hermitian conjugate            |
| (D δ_x)(x) = (m + 2) · spinor at x for U ≡ 1    | < 1e-12         | Diagonal term in the free-field limit                              |
| CG on `A = m · I`                               | ≤ 1 iteration   | Scalar HPD is exact in one step                                    |
| CG on D†[U] D[U]                                | rel < 1e-8      | Full fermion-solver round-trip on a hot U(1) background            |
| CG bail on a singular operator                  | not converged   | Solver doesn't silently lie when A is rank-deficient               |

## Next rounds

- **Round 2:** quenched chiral condensate ⟨ψ̄ψ⟩(m, β) by stochastic sources.
  Compares to the known Schwinger continuum value `⟨ψ̄ψ⟩/g = −e^γ/(2π^{3/2})`
  (in the massless limit), and tracks the explicit-symmetry-breaking
  Wilson artifact at finite a.
- **Round 3:** pseudofermions + HMC. The Schwinger model with dynamical
  fermions — the first place we use CG inside molecular dynamics
  (fermion force per leapfrog step).
- **Phase 5:** deflation acceleration around CG. Project out the lowest-
  eigenvalue modes of `D†D` once and reuse the deflation subspace across
  many right-hand sides. The interface here (CG as a template on apply_A)
  is designed to slot a deflated `apply_A_with_deflation` in without
  rewriting the inner loop.

## References

- K. G. Wilson, "Quarks and strings on a lattice", in *New Phenomena in
  Subnuclear Physics* (Plenum, 1977) — the Wilson term and doubler
  cancellation.
- I. Montvay and G. Münster, *Quantum Fields on a Lattice*, ch. 4.
- C. Gattringer and C. B. Lang, *Quantum Chromodynamics on the Lattice*,
  ch. 5 (Wilson-Dirac), ch. 8 (fermion algorithms).
- M. R. Hestenes and E. Stiefel, "Methods of conjugate gradients for solving
  linear systems", *J. Res. NBS* 49, 409 (1952) — the original CG paper.
