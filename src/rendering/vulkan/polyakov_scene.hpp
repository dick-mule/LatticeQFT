#pragma once

/**
 * @file polyakov_scene.hpp
 * @brief 3D SU(2) Polyakov-loop correlator as an alternative to the
 *        Bali-Schilling-Schlichter Wilson-loop measurement.
 *
 * Trades an extended Wilson loop W(R, T) (which suffers area-law noise
 * amplification ~ exp(σRT) / √N) for a single-link-wrapping observable:
 *
 *     P(x, y) = (1/N_c) tr [ ∏_{t=0}^{L_t − 1} U_z(x, y, t) ]
 *
 * P(x, y) is gauge-invariant (the loop is closed around the z-direction
 * by lattice periodicity) and lives on the 2D spatial slice. The correlator
 *
 *     C(R) = ⟨P(x) · P(x + R·ê_x)⟩ − ⟨P⟩²
 *
 * gives the static QQ̄ potential via  V(R) = −(1/L_t) log C(R).  Because
 * the observable has NO temporal extent of W, the noise floor stays
 * O(1/√N) at every R — completely sidesteps the area-law suppression.
 *
 * Visualization shows the **translation-averaged** correlator
 *
 *     C(Δx, Δy) = (1 / V_2D) Σ_{x_0, y_0} ⟨P(x_0, y_0) · P(x_0+Δx, y_0+Δy)⟩
 *                  − ⟨P⟩²
 *
 * as a 2D heatmap centered at the origin: in the confining phase a peak at
 * (0,0) decaying exponentially with separation — directly the static
 * potential's exponential-of-exponential shape.
 *
 * Side panel prints V(R) along R = (1..L/2, 0) so the user can read the
 * string tension off the slope log C vs R.
 *
 * Smearing toggle (APE / stout) and amplification follow FluxTubeScene's
 * pattern.
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

class GpuSU2Sweeper;

class PolyakovScene : public Scene
{
public:
    enum class SmearingMethod { APE, Stout };

    explicit PolyakovScene(int L = 16, std::uint64_t seed = 11235u,
                           const VulkanContext* gpu_ctx = nullptr);
    ~PolyakovScene() override;

    int  latticeSize() const override { return m_L; }
    // P(x, y) is intrinsically a 2D observable, but the correlator
    // C(Δx, Δy) is *independent of ẑ-offset* on a periodic lattice (the
    // Polyakov loop is the same regardless of where the ẑ-wrap starts).
    // So the natural volumetric rendering is to extrude the 2D
    // correlator along ẑ — a uniform bright tube at the QQ̄ separation,
    // decaying radially. Side-by-side with the BSS flux-tube scene, that
    // makes the two observables directly comparable.
    bool isVolumetric() const override { return true; }
    void step() override;
    void fillVolume(float* out) const override;
    bool buildControlsUI() override;

private:
    void resetStats();
    void resetHot(std::uint64_t seed);
    void resetCold();
    void measurePolyakov();
    void prepareSmearedField() const;
    double polyakovAt(int x, int y, const LinkField<su2::Element, 3>& U) const;

    int                                          m_L;
    Lattice<3>                                   m_lattice;
    LinkField<su2::Element, 3>                   m_field;
    Rng                                          m_rng;

    double                                       m_beta             = 6.0;
    su2_model::SU2Model<3>                       m_model;
    mc::MetropolisSweep<su2_model::SU2Model<3>>  m_sweeper;

    int                                          m_sweeps_per_frame = 3;
    bool                                         m_paused           = false;
    bool                                         m_use_heatbath     = true;

    std::unique_ptr<GpuSU2Sweeper>               m_gpu;
    bool                                         m_use_gpu                = false;
    int                                          m_gpu_sweeps_per_frame   = 30;

    // Smearing applied to the spatial (x̂, ŷ) links only; the temporal ẑ
    // links are what make up the Polyakov loop and must be left alone.
    // Default to a meaningful 10 iters at α=0.6 so the bright peak in
    // C(Δ) is spread over a few voxels — a single-voxel peak gets
    // filtered out by the volume renderer's opacity threshold and the
    // user sees nothing.
    int                                          m_smear_iters    = 10;
    float                                        m_smear_alpha    = 0.6f;
    float                                        m_smear_rho      = 0.10f;
    SmearingMethod                               m_smear_method   = SmearingMethod::APE;
    mutable LinkField<su2::Element, 3>           m_smeared_field;
    mutable LinkField<su2::Element, 3>           m_smear_scratch;

    int                                          m_measure_period       = 1;
    int                                          m_frames_since_measure = 0;

    // Accumulators
    long long                                    m_n_measurements = 0;
    double                                       m_sum_p_global   = 0.0;
    std::vector<double>                          m_sum_C;     // L²: C(Δx, Δy)
    mutable std::vector<double>                  m_p_field;   // L² scratch

    float                                        m_amplification = 250.0f;
};

} // namespace lqft::vis
