# LatticeQFT

A lattice-gauge-theory simulator written in modern C++ with a Vulkan compute
back-end (planned). The project is organized as a progression of increasingly
non-trivial models — from the 2D Ising warm-up through compact U(1), the
Schwinger model, and SU(N) pure gauge theory — each implementing the standard
numerical techniques of the field (Metropolis / heat-bath / Hybrid Monte Carlo
importance sampling, link-variable gauge actions, conjugate-gradient Dirac
inversion with deflation acceleration).

The goal is a rigorous, numerically validated simulator with real-time
visualization of confinement physics (Wilson loops, flux tubes, Polyakov-loop
correlators).

## Design goals

- **CPU reference + GPU port.** Every algorithm is implemented and tested on
  the CPU first (in double precision, with quantitative tests against known
  analytic / numerical benchmarks). The Vulkan compute port then targets the
  validated reference, with bit-for-bit-structure equivalence checks.
- **Dimension-generic infrastructure.** Lattice topology, field containers,
  and Monte Carlo engines are templated on the spatial dimension. Phase work
  starts in 2D and lifts to 3D and 4D without algorithm rewrites.
- **Model-agnostic Monte Carlo.** A small uniform `Model` concept
  (`local_action_density`, `propose`, `delta_action`, `apply`) lets the
  Metropolis (and later HMC) engine drive every model.
- **Checkerboard parallelism from day one.** Even the 2D Ising sweeps are
  organized by site parity — the same structure used by every later gauge
  theory on the GPU.

## Current phase

**Phase 4 / 5 (rounds 3j–3n) — infrastructure beyond the SU(2) flux tube.** The rounds
after 3i were never written up; this entry catches the README up with the tree.

- **SU(3) pure gauge** (`src/math/su3.hpp`, `src/models/su3.hpp`): 3×3 complex link
  variables, Wilson action `S = β Σ_□ (1 − ⅓ Re tr U_□)`, unitarity re-projection against
  round-off drift, and a single-link Metropolis whose proposal is a near-identity SU(2)
  rotation embedded in one of the three Cabibbo–Marinari sub-groups (symmetric, so plain
  Metropolis acceptance holds). Twelve tests: group algebra, `U U† = I` after projection,
  `ΔS` against a direct recompute, cold and random limits. Not yet on the CLI; the SU(3)
  heat-bath and 4D plaquette benchmarks are the next round.
- **Link smearing** (`src/smearing/ape.hpp`, `stout.hpp`): APE (Albanese–Petronzio,
  re-projected) and stout (Morningstar–Peardon, analytic) smearing for SU(2), with a
  direction mask so temporal links can be left alone for transfer-matrix observables.
  Tests: SU(2) preserved, cold configuration is a fixed point, thermalized action drops,
  `ρ = 0` is the identity.
- **Lanczos + deflated CG** (`src/solvers/lanczos.hpp`, `deflation.hpp`): symmetric Lanczos
  with full re-orthogonalisation and a tridiagonal eigensolver give Ritz pairs of `D†D`;
  deflated CG starts from the projected solution `x₀ = V (V†AV)⁻¹ V†b`, whose residual is
  exactly orthogonal to the basis, and converges on the spectrum above the deflated modes.
  Tests: diagonal operators, Ritz pairs of the Wilson–Dirac operator are eigenpairs, deflated
  CG matches vanilla CG and uses fewer iterations.
- **Lüscher–Weisz multilevel Wilson loops** (`src/monte_carlo/multilevel_wilson.hpp`): the
  temporal extent is cut into slabs whose interiors are sub-sampled with the boundary spatial
  links frozen; slab tensors in the doubled colour-index space compose by 4×4 products. Tests:
  the slab algebra reproduces the loop exactly at `N_sub = 1`, the mean agrees with standard
  sampling, the per-measurement variance is smaller. A bias was found and fixed in this round:
  the constrained sweep updated spatial links outside its slab, which moved the closing line
  an earlier slab's tensor had been conditioned on (27% low on `⟨W(2,2)⟩` at `β = 5`).
- **Polyakov-loop correlator scene** (tenth visualizer scene): `P(x,y) = ½ tr ∏ₜ U_ẑ`, the
  translation-averaged correlator `C(Δ) = ⟨P(0)P(Δ)⟩ − ⟨P⟩²`, and the static potential
  `V(R) = −(1/L_t) log C(R)` on a 16³ lattice — the area-law-free route to the string tension.
- **GPU pseudofermion HMC** (`gpu_wilson_dirac_2d.cpp`, `gpu_cg_2d.cpp`,
  `gpu_schwinger_force.cpp`): the Wilson–Dirac operator, the CG solve and the gauge +
  fermion force run as compute shaders; the Schwinger HMC scene toggles GPU/CPU and the GPU
  path self-validates against the CPU force (relative L² error reported). The "CPU only"
  remark under round 3i below is superseded.
- **CI**: a CPU-only GitHub Actions job builds the core and runs the full test suite on every
  push (the Vulkan visualizer needs a GPU and is built locally).

**Round 4a — 4D SU(2) deconfinement.** `--nt N` makes the `--model su2 --dim 4` driver run
an `L³ × N` lattice and append the Polyakov loop `⟨|P̄|⟩`, its susceptibility
`χ_P = N_s³(⟨P̄²⟩ − ⟨|P̄|⟩²)` and the extents to the β-scan (`src/observables/polyakov.hpp`;
tests: cold loop, cyclicity, the Z₂ centre flip negates every loop and keeps every plaquette,
hot loop small). Heat-bath scans at `N_s = 8, 12` put the susceptibility peak at `β = 1.875`
for `N_t = 2` and `β = 2.300` for `N_t = 4` (published: 1.88 and 2.30), with the peak growing
with volume as the 3D-Ising class requires. The companion asymptotic-scaling check shows that
the `2×2` Creutz ratio on `8⁴` runs 2.5× slower than two-loop over `β = 2.0–2.6`, i.e. the
small-loop string tension is not yet in the scaling window. Figures in `docs/figures/`,
regeneration commands and the note in
[`docs/math/su2_deconfinement.md`](docs/math/su2_deconfinement.md);
`scripts/analyze_su2_deconfinement.py` reproduces the tables.

### Validation map

| Analytic / numerical benchmark | Where it is checked |
|---|---|
| Onsager critical point `β_c = ½ ln(1+√2)` | `tests/test_ising.cpp` |
| Free Klein–Gordon propagator (`λ = 0`) | `tests/test_phi4.cpp` |
| 2D U(1) Bessel ratio `⟨cos θ_□⟩ = I₁(β)/I₀(β)`, gauge invariance | `tests/test_u1.cpp` |
| Wilson–Dirac γ⁵-Hermiticity, CG residual, force vs finite difference, `⟨ΔH⟩`, trace identity | `tests/test_dirac_2d.cpp`, `test_cg.cpp`, `test_hmc.cpp`, `test_schwinger.cpp` |
| SU(2): `U U† = I`, exact `ΔS`, gauge invariance under random `Ω(x)`, strong/weak plaquette, `W(2,1) = W(1,2)` | `tests/test_su2.cpp` |
| SU(3) algebra and limits | `tests/test_su3.cpp` |
| Smearing fixed points and masks | `tests/test_ape.cpp`, `test_stout.cpp` |
| Ritz pairs, deflated vs vanilla CG | `tests/test_deflation.cpp` |
| Multilevel exact algebra, mean, variance | `tests/test_multilevel.cpp` |
| Dense propagator `D D⁻¹ = 1`, γ⁵-Hermiticity, free-field two-fermion threshold | `tests/test_mesons.cpp` |
| Polyakov loop: cold, cyclic, Z₂ centre flip, hot | `tests/test_polyakov.cpp` |
| 4D SU(2) `β_c(N_t = 2) ≈ 1.88`, `β_c(N_t = 4) ≈ 2.30` | `docs/math/su2_deconfinement.md` (scan commands + figures) |

**Phase 4 (round 3b) — Wilson loops, Creutz ratios, β-sweep driver.**
`averageWilsonLoop(R, T)` for arbitrary rectangle, `creutzRatio` extracting
the string-tension estimator `σ ≈ −log[W(R,T)·W(R−1,T−1)/(W(R−1,T)·W(R,T−1))]`
that subtracts off the perimeter divergence, and a new `--model su2`
driver with `--dim {2, 3, 4}` dispatch and `--use-heatbath` toggle. 3D
heat-bath β-sweep produces clean confinement signatures (χ(2,2) > 0 at
every β, monotone decreasing with β); 4D shows the same pattern with the
steepest drop near β ≈ 2.0–2.5 where 4D SU(2) has its strong→weak coupling
crossover on small lattices. W(2,1) ≈ W(1,2) at every β confirms μ↔ν
symmetry — an unforced consistency check, since the path-traversal code
computes them independently. See
[`docs/math/su2_lattice.md`](docs/math/su2_lattice.md) for the full
derivation and validation tables.

**Phase 4 (round 3a) — SU(2) pure-gauge infrastructure.** First non-Abelian
model: SU(2) link variables stored as unit quaternions
`(s, v) ∈ S³` ↔ `U = s·I + i v·σ`. Wilson action
`S = β Σ_□ (1 − ½ tr U_□)`. The staple is now a sum of 2(Dim − 1) SU(2)-
valued matrices stored in the same 4-vector representation; `su2::multiply`
implements the general 2×2 product `(s₁s₂ − v₁·v₂, s₁v₂ + s₂v₁ − v₁×v₂)`
that works for SU(2) elements *and* their sums. Two update algorithms ship:
- **Metropolis** — proposal `U → ΔU · U` with `ΔU = exp(i θ·σ/2)`,
  `θ ∼ N(0, σ²·I)`. Symmetric proposal automatically.
- **Heat-bath** — Kennedy–Pendleton 1985 rejection sampler for the
  conditional `P(w₀ | A) ∝ exp(β|A| w₀) √(1 − w₀²)`. Acceptance ≥ 0.95
  across all β·|A|; ≪ 2 inner iterations on average.

Color partition: 2·Dim sublattices (same `(parity, μ)` as U(1)) — the
generic `MetropolisSweep` consumes it unchanged.

Validated by: identity neutral, `U U† = I`, multiplication preserves
unit norm, near-identity proposals are close to identity for small σ,
cold-start `S = 0` exactly, `ΔS` matches `totalAction` difference at
1e-9, **gauge invariance** under random `Ω(x) ∈ SU(2)`
(`U_μ(x) → Ω(x) U_μ(x) Ω†(x + ê_μ)`), heat-bath plaquette in correct
strong-/weak-coupling regimes (β = 0.4 → ≈ 0.1; β = 10 → > 0.85), and
the same exact-arithmetic tests pass at **Dim = 3** (6 colors) and
**Dim = 4** (8 colors) with zero algorithm changes.

**Phase 4 (round 3h + 3i) — Full menu coverage: every CLI model gets a
scene + GPU speedup where feasible.** Five additional scenes round out
the menu:

- **2D compact U(1) gauge** — heatmap of `1 − cos θ_□`, GPU Metropolis
  (existing `GpuU1Sweeper` now takes a `Dim` parameter; same shader does
  both 2D and 3D).
- **2D SU(2) Yang-Mills** — heatmap of `1 − ½ tr U_□`, GPU Metropolis
  (existing `GpuSU2Sweeper` likewise Dim-parameterized).
- **2D Schwinger (quenched)** — U(1) gauge field on the GPU, CPU CG-based
  stochastic-condensate measurement every N frames. Heatmap shows the
  gauge action density; readout shows running `⟨ψ̄ψ⟩(β, m)`.
- **2D Schwinger (dynamical HMC)** — full pseudofermion HMC on a 10²
  lattice via the existing `SchwingerProvider` + `GaugeHMC` engines.
  Live readouts of acceptance, ⟨ΔH⟩, condensate. (Originally CPU only;
  the GPU fermion force landed in a later round, see the top of this section.)

Menu now ships **nine scenes** covering every CLI model. All gauge and
spin/scalar scenes have a GPU/CPU toggle.

**Phase 4 (round 3g) — 2D rendering infrastructure + 2D Ising and 2D φ⁴
scenes with GPU sweepers.** New `Heatmap2DRenderer` (R32_SFLOAT 2D image
+ `shaders/heatmap_2d.frag` with viridis / diverging-blue-yellow colormaps
selectable per scene). `Scene` interface gains `isVolumetric()` and
`heatmapStyle()`; the existing volumetric scenes return defaults and are
untouched. Two new scenes:

- **2D Ising (Onsager)** — 64² spins, GPU `ising_metropolis.comp`
  (~50 sweeps/frame), diverging colormap so the two ordered phases pop.
- **2D φ⁴ scalar field** — 64² real scalar, GPU `phi4_metropolis.comp`
  with the same site-parity coloring and ΔS = −δ·S_nn + D·Δφ² + ½m²·Δφ² + (λ/24)·Δφ⁴.

VulkanApp now lazily builds the right renderer (volume or heatmap) on
scene activation, tearing down the other to free GPU memory. All five
menu scenes share the same GPU/CPU toggle pattern: a checkbox in the
panel switches algorithms at runtime; CPU paths are preserved as
fallbacks. Schwinger and 2D U(1)/SU(2) scenes are next round.

**Phase 4 (round 3f) — GPU compute extended to all 3D gauge scenes.**
Both `U1Scene3D` and `SU2Scene3D` now ship with the same GPU/CPU toggle
the flux-tube scene introduced. The U(1) scene gets its own
`GpuU1Sweeper` driving `shaders/u1_metropolis.comp` (single-float storage,
`vec2` staple, ΔS formula matching `U1Model::delta_action`); the SU(2)
scene reuses the existing `GpuSU2Sweeper`. All three scenes default to
GPU when a `VulkanContext` is available and fall back transparently to
CPU heat-bath (U(1) Best-Fisher von Mises, SU(2) Kennedy-Pendleton)
when not. Toggle is live; flipping it doesn't change the equilibrium
distribution, just the algorithm.

**Phase 4 (round 3e) — GPU SU(2) Metropolis compute pipeline.** Flux-tube
convergence on the CPU was bottlenecked by the Kennedy-Pendleton heat-bath
sweep (~90 % of frame time). Ported the sweep to the GPU:

- New compute shader `shaders/su2_metropolis.comp` running parallel link
  Metropolis with per-site xoshiro128++ RNG state. One thread per link,
  dispatched once per (parity, μ) color (6 dispatches per sweep in 3D).
  Memory barriers between dispatches; same staple math as `src/math/su2.hpp`.
- New `GpuSU2Sweeper` class wrapping the storage buffers (gauge field as
  `vec4[V·Dim]`, RNG state as `uvec4[V]`, ~256 KB total at L = 16),
  descriptor set, compute pipeline, command buffer, and fence.
- New `VulkanContext` struct passed to scene factories so any future scene
  can spin up its own GPU compute.
- `FluxTubeScene` toggles between GPU Metropolis (default) and CPU
  heat-bath via the ImGui panel. With GPU: ~30 sweeps/frame default at
  60 FPS = 1 800 sweeps/sec, vs ~180 sweeps/sec on CPU heat-bath — a
  **10× wall-clock speedup of the MC**, plus the per-sweep Metropolis is
  enough cheaper than heat-bath that the per-second link-update rate is
  more like **20–50× higher**.

Push-constant block (36 B): `Lx, Ly, Lz, Dim, color_parity, color_mu`
(ints), `beta, step_size` (floats), `sweep_count` (int for RNG
de-correlation). Buffers are host-visible/coherent so the CPU readback
for BSS measurement is just a memcpy. CPU heat-bath path is preserved as
a fallback when no Vulkan context is supplied to the scene (e.g.
unit tests or future headless modes).

**Phase 4 (round 3d) — Honest SU(2) flux tube between static color charges.**
A third scene `FluxTubeScene` joins the menu, implementing the classic
Bali–Schilling–Schlichter chromoelectric-flux observable in 3D SU(2):

- Static QQ̄ pair separated by R lattice units, with a rectangular Wilson
  loop running R steps in `+x̂` and T steps in `+ẑ` ("time")
- Local plaquette action density `ρ(x) = Σ_{μ<ν}(1 − ½ tr U_□)` measured
  everywhere
- **Connected correlator** `f(x; R, T) = ⟨W · ρ(x)⟩ / ⟨W⟩ − ⟨ρ⟩` —
  vacuum-subtraction makes `f → 0` far from the QQ̄ pair and `f > 0` in
  the flux tube
- **Translation-averaged** over the y-axis (perpendicular to the loop
  plane), folding L Wilson-loop measurements into each MC configuration
  for a factor-`L` variance reduction essentially for free
- Live `f(x, Δy, z)` rendered through the same `VolumeRenderer` —
  amplification slider in the panel lets the user tune visibility
  against the noise floor; the connected signal localizes into a
  bright tube structure between the charges over O(30 s) of MC time

ImGui controls in the scene: β, R (1..L/2), T (1..L/2), sweeps/frame,
heat-bath toggle, pause, reset hot/cold, reset statistics,
amplification, `N_meas` counter, live `⟨W⟩` and `⟨ρ⟩` readouts.
Parameter changes (β, R, T) automatically reset the accumulators
because the running averages would otherwise mix incompatible
configurations.

**Phase 4 (round 4d) — Scene-selector menu + 3D SU(2) volumetric.** The
visualizer opens on a menu page (modeled after the BlackHolePathTracer
scene selector) that lists the available scenes with descriptions and a
"Launch" button. Activating a scene tears down/rebuilds the volumetric
renderer for the scene's lattice size, hands camera and transfer-function
control off to render-controls panels, and offers a "← Back to menu"
button in the scene's controls panel. Two scenes ship:

- **3D compact U(1) gauge** — Wilson plaquette on 24³, per-site action
  density `Σ_{μ<ν}(1 − cos θ_□)`, heat-bath / Metropolis toggle.
- **3D SU(2) Yang-Mills** — non-Abelian counterpart on 20³, per-site
  action density `Σ_{μ<ν}(1 − ½ tr U_□)`, Kennedy–Pendleton heat-bath /
  Metropolis toggle.

Both scenes share the same `Scene` abstract interface
(`latticeSize` / `step` / `fillVolume` / `buildControlsUI` returns
true to request "back to menu"), so adding a 4D U(1) slice viewer, a 2D
Ising heatmap, or the flux-tube static-charge scene later is just a new
`SceneDescriptor` registered at startup. The shared `VolumeRenderer` is
gauge-group agnostic by design — it takes any `float[L³]` scalar field.

**Phase 4 (round 4c) — Volumetric ray-marcher for 3D U(1).** Vulkan 1.2 +
GLFW + ImGui visualizer with a 3D U(1) Monte Carlo running on the CPU and
a volumetric ray-marched render of the per-site action density
`ρ(x) = Σ_{μ<ν=0..2}(1 − cos θ_□(x; μ, ν))` each frame. Architecture:
- `LatticeController` owns the 3D U(1) gauge field and runs link Metropolis
  or Best–Fisher heat-bath sweeps (toggle in the panel).
- `VolumeRenderer` owns a device-local R32_SFLOAT 3D `vk::Image`,
  host-visible staging buffer, sampler, and a fullscreen-triangle pipeline
  whose fragment shader ray-marches the [−1, +1]³ bounding cube,
  samples the volume with linear filtering, applies a thresholded-linear
  opacity transfer function, and composites a viridis-colored alpha
  front-to-back. Push constants carry `inv(P · V)`, camera world position,
  and three transfer-function knobs (92 B; well under the 128 B guarantee).
- `Camera` is an orbit camera (distance, polar, azimuth) with mouse drag
  and scroll wheel. Camera input is chained through ImGui's GLFW
  callbacks so dragging in the panel doesn't rotate the view.
- Swapchain rebuild on window resize.

macOS / MoltenVK portability extensions (`VK_KHR_portability_enumeration`,
`VK_KHR_portability_subset`) are handled in the instance / device
creation. Build is optional: `-DBUILD_VISUALIZER=OFF` keeps the headless
CLI build working when Vulkan SDK / GLFW / GLM aren't installed.

**Phase 4 (rounds 4a / 4b) — Vulkan scaffold and 2D heatmap (subsumed by 4c).**

**Phase 3 (analysis round) — Schwinger continuum comparison.**
[`scripts/run_schwinger_grid.py`](scripts/run_schwinger_grid.py) drives the
`schwinger-hmc` binary over a `(β, m)` grid and aggregates a tidy CSV in
`data/`. [`scripts/analyze_schwinger.py`](scripts/analyze_schwinger.py)
computes the continuum Schwinger condensate
`⟨ψ̄ψ⟩/g = -e^γ/(2π^{3/2})` symbolically with SymPy, produces three
PNG plots in `docs/figures/`, and prints a summary table. The
[methods note `schwinger_continuum.md`](docs/math/schwinger_continuum.md)
covers the continuum derivation, the lattice formulation, the additive
and multiplicative Wilson renormalizations that obstruct pointwise
comparison at finite `a`, and the qualitative tests the dynamical HMC
passes (monotonicity in `m`, plaquette tracking the pure-gauge Bessel
ratio, ⟨ΔH⟩ ≈ 0). A full continuum extrapolation is sketched but not
performed.

**Phase 3 — 2D Schwinger model with dynamical fermions via HMC.** Full
pseudofermion HMC for the 2D Schwinger model. Wilson-Dirac operator with
the (1 ± γ_μ) projector trick; γ⁵-Hermiticity exploited so apply / applyDagger
share one code path. Templated conjugate-gradient solver inverts `D†D` to
relative residual < 1e-9 on hot U(1) backgrounds. Quenched chiral condensate
estimated via Z₂-noise stochastic sources. Pseudofermion field sampled by
`χ ∼ N(0, ½)`, `φ = D† χ` gives the correct `exp(-φ†(D†D)⁻¹ φ)` marginal.
Leapfrog + Metropolis HMC trajectory; fermion force computed by one CG solve
per leapfrog step, with the same (1 ± γ_μ) projector collapse for
`∂D/∂θ_μ(x)`. Validated by force vs finite-difference matches (5e-4 abs on
fermion sector), ΔH = O(dt²) leapfrog scaling, pseudofermion-refresh trace
identity `⟨φ†(D†D)⁻¹ φ⟩ = 2V`, and acceptance > 50 % at (β, m) = (2, 0.5).

**Phase 2 — Compact U(1) lattice gauge theory.** Link variables
θ_μ(x) ∈ ℝ, Wilson plaquette action `S = β Σ_□ (1 − cos θ_□)`. Update units
are colored by (site parity, μ) — 2·Dim sublattices — so the CPU sweep
matches the parallel-update structure the GPU port will need.

Two update algorithms ship in parallel:

- **Link Metropolis** via the staple-sum derivation
  `ΔS = β · Re[(e^{iθ_old} − e^{iθ_new}) · A_μ]`.
- **Heat-bath** sampling the exact von Mises conditional
  `P(θ | A) ∝ exp(β |A| cos(θ + φ_A))` via Best–Fisher 1979. 100 %
  acceptance, no step-size tuning, dramatic reduction in autocorrelation
  at strong coupling. Enabled with `--use-heatbath`.

Validated against the 2D exact-factorization Bessel-ratio benchmark
`⟨cos θ_□⟩ = I_1(β) / I_0(β)`, the gauge-invariance identity
`S[θ + dα] = S[θ]`, and the textbook 4D U(1) deconfinement transition at
β_c ≈ 1.01 (visible in the `--model u1 --dim 4` driver).

**Phase 1 — 2D φ⁴ scalar field theory.** Continuous real scalar with
`S = ½ (∂φ)² + ½ m² φ² + (λ/4!) φ⁴`. Single-site Gaussian Metropolis with
adaptive step-size tuning. Validated against the closed-form Klein-Gordon
propagator in the free (λ = 0) limit and the textbook Z₂ broken-phase
transition for λ = 1.

**Phase 0 — 2D Ising warm-up.** Lattice + parity + RNG + Metropolis
substrate validated against the Onsager critical point
β_c = ½ ln(1 + √2) ≈ 0.4407.

## Roadmap

```
Phase 0  2D Ising                        — Metropolis, parity, RNG, observables
Phase 1  2D φ⁴ scalar field              — continuous DOFs, action gradients
Phase 2  2D / 3D compact U(1) gauge      — link variables, Wilson action
Phase 3  2D Schwinger (U(1) + fermions)  — Dirac op, pseudofermions, CG
Phase 4  3D / 4D SU(2) pure gauge        — non-Abelian links, flux tubes
Phase 5  4D SU(3) with Wilson fermions   — CG + deflation, hadron spectrum
```

Vulkan visualization comes online at phase 2 once the CPU reference for at
least one gauge theory is validated.

## Directory structure

```
LatticeQFT/
├── docs/
│   └── math/                # Derivations, validation notes, action definitions
├── src/
│   ├── lattice/             # Lattice topology, site indexing, parity
│   ├── fields/              # SiteField, LinkField, SpinorField<Dim, Nc>
│   ├── rng/                 # RNG abstraction (mt19937 now, counter-based for GPU)
│   ├── math/                # Operators on fields (Wilson-Dirac, …)
│   ├── solvers/             # Conjugate-gradient (deflation later)
│   ├── models/              # Per-model action functionals (Ising, φ⁴, U(1), …)
│   ├── monte_carlo/         # Metropolis, step-size autotuner, HMC, U(1) heat-bath
│   ├── rendering/vulkan/    # Vulkan visualizer scaffold (optional target)
│   ├── observables/         # Magnetization, plaquette, Wilson loop, …
│   ├── rendering/           # Vulkan layer (added in phase 2)
│   ├── utils/               # JSON / glm helpers
│   └── main.cpp
├── tests/                   # GoogleTest suite
├── CMakeLists.txt
└── README.md
```

## Building

```bash
git clone <repo-url>
cd LatticeQFT
mkdir -p build && cd build
cmake .. -G Ninja
ninja
ctest                       # runs the test suite
./LatticeQFT --help         # --model {ising,phi4,u1,su2,schwinger-quenched,schwinger-hmc}
./LatticeQFTVis             # Vulkan visualizer (live 3D U(1) volumetric, drag-to-rotate, scroll-to-zoom)
python3 scripts/run_schwinger_grid.py  # generate (β, m) data → data/schwinger_grid.csv
python3 scripts/analyze_schwinger.py   # plots + SymPy continuum value → docs/figures/
```

Dependencies fetched by CMake `FetchContent`: nlohmann/json, GoogleTest,
magic_enum. The visualizer additionally fetches Dear ImGui and requires the
following installed system-wide (e.g. via vcpkg or Homebrew): Vulkan SDK
(with `glslc` for shader compilation), GLFW, GLM. The visualizer target
turns off automatically (`BUILD_VISUALIZER=OFF`) if any of those are
missing — the headless CLI build keeps working.

## References

- M. Creutz, *Quarks, Gluons and Lattices*, Cambridge University Press, 1983.
- H. Rothe, *Lattice Gauge Theories: An Introduction*, 4th ed., World Scientific, 2012.
- C. Gattringer and C. B. Lang, *Quantum Chromodynamics on the Lattice*,
  Lecture Notes in Physics 788, Springer, 2010.
- L. Onsager, "Crystal Statistics. I. A Two-Dimensional Model with an
  Order-Disorder Transition", *Phys. Rev.* 65, 117 (1944).
