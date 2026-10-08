#pragma once

/**
 * @file phi4_2d_scene.hpp
 * @brief 2D φ⁴ scalar theory as a heatmap scene.
 *
 *   S[φ] = Σ_x [ ½ Σ_μ (φ_{x+μ} − φ_x)² + ½ m² φ_x² + (λ/24) φ_x⁴ ]
 *
 * Diverging colormap so the broken-Z₂ phase (∣⟨φ⟩∣ > 0) reads cleanly:
 * negative ⟨φ⟩ in blue, positive in yellow. Sliders for m² (Z₂ symmetry
 * spontaneously breaks below m_c² ≈ −0.7 at λ = 1) and λ (default 1).
 */

#include "../../fields/site_field.hpp"
#include "../../lattice/lattice.hpp"
#include "../../models/phi4.hpp"
#include "../../monte_carlo/metropolis.hpp"
#include "../../monte_carlo/tuning.hpp"
#include "../../observables/observables.hpp"
#include "../../rng/rng.hpp"
#include "scene.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace lqft::vis
{

class GpuPhi4Sweeper;

class Phi4_2DScene : public Scene
{
public:
    explicit Phi4_2DScene(int L = 64, std::uint64_t seed = 5678u,
                          const VulkanContext* gpu_ctx = nullptr);
    ~Phi4_2DScene() override;

    int  latticeSize() const override { return m_L; }
    bool isVolumetric() const override { return false; }
    HeatmapStyle heatmapStyle() const override
    {
        // φ ∈ ℝ, we expect ∣φ∣ ≲ 2 in the broken phase; clamp via gain.
        return HeatmapStyle{ /*gain*/0.7f, /*bias*/0.0f, /*gamma*/1.0f, /*signed*/1 };
    }
    void step() override;
    void fillVolume(float* out) const override;
    bool buildControlsUI() override;

private:
    void singleStep();
    void resetHot(std::uint64_t seed);
    void resetCold();
    void syncFromGpu() const;

    int                                                m_L;
    Lattice<2>                                         m_lattice;
    SiteField<double, 2>                               m_field;
    Rng                                                m_rng;
    // m² = −1.0 at λ = 1 puts the lattice safely into the Z₂-broken phase
    // on 64² (critical line is around m² ≈ −0.7 at this λ). On launch the
    // user sees the field commit to ±φ_min within seconds: domains form,
    // then coarsen via kink/anti-kink annihilation. Drag m² up toward 0 to
    // walk back into the symmetric phase.
    double                                             m_m_sq      = -1.0;
    double                                             m_lambda    = 1.0;
    double                                             m_step_size = 1.0;
    phi4::Phi4Model<2>                                 m_model;
    mc::MetropolisSweep<phi4::Phi4Model<2>>            m_sweeper;

    int  m_sweeps_per_frame = 3;
    bool m_paused           = false;

    std::unique_ptr<GpuPhi4Sweeper>                    m_gpu;
    bool                                               m_use_gpu              = false;
    int                                                m_gpu_sweeps_per_frame = 40;

    mutable std::vector<float>                         m_upload_buf;  // L*L
};

} // namespace lqft::vis
