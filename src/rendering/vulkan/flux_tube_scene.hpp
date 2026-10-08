#pragma once

/**
 * @file flux_tube_scene.hpp
 * @brief 3D SU(2) flux tube between two static color charges.
 *
 * The classic Bali–Schilling–Schlichter (BSS) observable: a rectangular
 * Wilson loop W in the (x̂, ẑ) plane represents a static quark–antiquark
 * pair separated by R lattice units, propagated for "time" T. The
 * spatial field strength between them is probed by the local plaquette
 * action density
 *
 *     ρ(x) = Σ_{μ<ν=0..2} (1 − ½ tr U_□(x; μ, ν))
 *
 * and the *connected* correlator
 *
 *     f(x; R, T) = ⟨W · ρ(x)⟩ / ⟨W⟩  −  ⟨ρ⟩
 *
 * subtracts off the vacuum background so that f → 0 far from the QQ̄ pair
 * and f > 0 in the chromoelectric flux tube. In a confining phase this
 * field is squeezed into a tube of finite transverse width — visible
 * directly on screen.
 *
 * Translation-average over the y-axis (perpendicular to the loop plane)
 * by storing the cumulants indexed by the *relative* y-coordinate
 * Δy = y − y₀ mod L. That gives an L-fold reduction in stochastic noise
 * essentially for free, important because W(R, T) is small in the
 * confining phase (area law) and the signal/noise on the raw correlator
 * is poor without translation averaging.
 *
 * Renders f(x, Δy, z), clamped to non-negative values and multiplied by a
 * user-controllable amplification, into the L³ volume buffer.
 */

#include "../../fields/link_field.hpp"
#include "../../lattice/lattice.hpp"
#include "../../math/su2.hpp"
#include "../../models/su2.hpp"
#include "../../monte_carlo/metropolis.hpp"
#include "../../monte_carlo/su2_heatbath.hpp"
#include "../../rng/rng.hpp"
#include "scene.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace lqft::vis
{

class GpuSU2Sweeper; // forward declaration

class FluxTubeScene : public Scene
{
public:
    explicit FluxTubeScene(int L = 16, std::uint64_t seed = 24601u,
                           const VulkanContext* gpu_ctx = nullptr);
    ~FluxTubeScene() override;

    int  latticeSize() const override { return m_L; }
    void step() override;
    void fillVolume(float* out) const override;
    bool buildControlsUI() override;

private:
    // ---- internals ----
    void resetStats();
    void resetHot(std::uint64_t seed);
    void resetCold();
    void measureCurrentConfig();
    void computeRhoField() const;
    void prepareSmearedField() const;
    double computeWilsonLoop(int y_0,
                             const LinkField<su2::Element, 3>& U) const;

    // ---- state ----
    int                                 m_L;
    long long                           m_V;
    Lattice<3>                          m_lattice;
    LinkField<su2::Element, 3>          m_field;
    Rng                                 m_rng;

    // β tuned for the **3D** SU(2) theory we actually run here. Bali-
    // Schilling-Schlichter is famously a 4D measurement at β ≈ 2.4 — in 3D
    // that same β sits in deep strong coupling (a²σ ≈ 0.5), suppressing
    // ⟨W(R, T)⟩ below the noise floor for any (R, T) you'd want to look at.
    // β = 6 in 3D SU(2) gives a²σ ≈ 0.25 — confining (3D SU(2) confines
    // at every β) but with a measurable signal at R = 4, T = 2.
    double                              m_beta             = 6.0;
    su2_model::SU2Model<3>              m_model;
    mc::MetropolisSweep<su2_model::SU2Model<3>> m_sweeper;

    // R, T defaults sized so ⟨W(R,T)⟩ is clearly positive against the
    // statistical noise floor at β = 2.4 with our sample rate. The Wilson
    // loop falls exponentially in area, so R=6 T=3 sinks below noise within
    // a few thousand measurements; R=4 T=2 stays measurable.
    int                                 m_R                = 4;
    int                                 m_T                = 2;

    // APE smearing applied to spatial links only (x̂, ŷ) prior to Wilson-loop
    // measurement. The MC chain itself runs on the unsmeared field; the
    // smeared copy is rebuilt each measurement step. Default iters = 0 means
    // no smearing.
    // BSS smearing is a Goldilocks knob — the connected correlator
    //   f(x) = ⟨W·ρ⟩/⟨W⟩ − ⟨ρ⟩
    // requires ⟨W⟩ to be both (a) measurably above the noise floor and
    // (b) measurably BELOW 1, otherwise the loop doesn't perturb the field
    // and f(x) collapses to noise around zero. At β = 6 in 3D SU(2):
    //   0 iters:   ⟨W(4,2)⟩ ≈ 0.14, marginal signal, narrow tube
    //   4 iters:   ⟨W⟩ ≈ 0.20–0.25, tube emerges clearly
    //   10 iters:  ⟨W⟩ → 0.4–0.5, loop is too identity-like, f(x) flattens
    // The Polyakov scene next door doesn't have this knob to misset because
    // it measures a different observable that's smearing-monotone.
    int                                 m_smear_iters      = 4;
    float                               m_smear_alpha      = 0.5f;
    int                                 m_measure_period   = 1;     // measure every K frames
    int                                 m_frames_since_measure = 0;
    mutable LinkField<su2::Element, 3>  m_smeared_field;
    mutable LinkField<su2::Element, 3>  m_smear_scratch;

    int                                 m_sweeps_per_frame = 3;
    bool                                m_paused           = false;
    bool                                m_use_heatbath     = true;

    // GPU sweeper. Optional — when present it owns the gauge field on the
    // GPU and we read back into `m_field` only for measurement.
    std::unique_ptr<GpuSU2Sweeper>      m_gpu;
    bool                                m_use_gpu          = false;
    int                                 m_gpu_sweeps_per_frame = 30;

    // BSS accumulators
    long long                           m_n_measurements   = 0;
    double                              m_sum_W_loops      = 0.0; // Σ over configs of Σ_{y_0} W(y_0)
    double                              m_sum_rho_mean     = 0.0; // Σ over configs of mean ρ
    std::vector<double>                 m_sum_W_rho;              // [V], Σ_{y_0} W · ρ at relative coords

    // Workspaces (mutable because computeRhoField runs in fillVolume's path too)
    mutable std::vector<double>         m_rho_field;              // [V]

    // Display
    float                               m_amplification    = 250.0f;
};

} // namespace lqft::vis
