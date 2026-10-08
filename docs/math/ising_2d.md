# 2D Ising Model — Phase 0 Reference

The 2D Ising model is not a quantum field theory, but it is the smallest
non-trivial Monte Carlo problem with the same algorithmic structure as the
lattice gauge theories of later phases (local action, nearest-neighbor
coupling, checkerboard-parallelizable updates). Phase 0 uses it as the test
bench for the lattice + parity + RNG + Metropolis infrastructure.

## Setup

A spin σ_x ∈ {−1, +1} lives at every site x of an L × L periodic square
lattice. The action (Euclidean, in units where the coupling J = 1) is

```
S[σ] = -β Σ_{<x,y>} σ_x σ_y
```

where ⟨x, y⟩ runs over each unordered pair of nearest neighbors once. On an
L × L torus the lattice has 2 L² such pairs (L² sites × 2 forward directions
per site).

## Metropolis update

Propose flipping a single spin σ_x → −σ_x. Only neighbors of x contribute
to the action change:

```
ΔS = 2 β σ_x Σ_{μ=0..1} (σ_{x+ê_μ} + σ_{x-ê_μ})
```

Accept with probability `min(1, exp(-ΔS))`. By construction this satisfies
detailed balance with respect to the Boltzmann distribution `exp(-S[σ]) / Z`.

## Checkerboard parallelism

Because S only couples nearest neighbors, the conditional distribution of any
single spin given all others depends only on its four nearest neighbors. On
the bipartite (even/odd parity) decomposition this means: all even-parity
spins can be updated in parallel given the odd-parity spins, and vice versa.

We exploit this in Phase 0 even on the CPU — we always sweep even sites,
then odd sites. The same structure carries forward to every later phase and
is what the GPU port will need anyway (one compute dispatch per parity).

## Onsager critical point

The 2D Ising model on the square lattice was solved exactly by Onsager
(1944). The critical inverse temperature is

```
β_c = ½ ln(1 + √2) ≈ 0.4406867935...
```

For β < β_c the system is in the disordered (paramagnetic) phase with
⟨|σ|⟩ → 0 in the thermodynamic limit; for β > β_c it is in the ordered
(ferromagnetic) phase with ⟨|σ|⟩ → m₀ > 0. The order parameter has the
exact closed form (Yang 1952)

```
⟨|σ|⟩ = (1 - 1/sinh⁴(2β))^{1/8}  for β > β_c,    0 for β ≤ β_c.
```

## Phase-0 validation

The CI test suite asserts:

- Cold start ⇒ total action = −β · 2L² (every pair aligned).
- Single-flip ΔS predicted by the model equals the literal action
  difference computed before and after `apply()`. Held to machine precision.
- On an L = 24 lattice, β = 0.30 (disordered) gives ⟨|m|⟩ < 0.20.
- β = 0.55 (ordered) gives ⟨|m|⟩ > 0.80.
- ⟨|m|⟩ is monotone non-decreasing across the transition (within MC noise).

These are deliberately loose bounds; the point is the framework, not finite-
size scaling. A separate notebook driver (not in CI) will be used for the
quantitative comparison to the Yang closed form.

## References

- L. Onsager, "Crystal Statistics. I. A Two-Dimensional Model with an
  Order-Disorder Transition", *Phys. Rev.* 65, 117 (1944).
- C. N. Yang, "The Spontaneous Magnetization of a Two-Dimensional Ising
  Model", *Phys. Rev.* 85, 808 (1952).
- M. Creutz, *Quarks, Gluons and Lattices*, ch. 5.
