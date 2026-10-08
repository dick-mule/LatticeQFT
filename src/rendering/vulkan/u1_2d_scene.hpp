#pragma once

/**
 * @file u1_2d_scene.hpp
 * @brief 2D compact U(1) as a heatmap scene.
 *
 * Wilson plaquette action  S = β Σ_□ (1 − cos θ_□)  on a 64² lattice.
 * Heatmap shows the per-site action density (1 − cos θ_□) — bright where
 * the gauge field is "active" (strong coupling), dim where it's smooth.
 * Reuses the existing `GpuU1Sweeper` (now Dim-parameterized) for GPU MC.
 *
 * Exact 2D benchmark: ⟨cos θ_□⟩ = I_1(β) / I_0(β); strong/weak coupling
 * regimes are visually distinct.
 */

#include "../../fields/link_field.hpp"
#include "../../lattice/lattice.hpp"
#include "../../models/u1.hpp"
#include "../../monte_carlo/metropolis.hpp"
#include "../../monte_carlo/tuning.hpp"
#include "../../monte_carlo/u1_heatbath.hpp"
#include "../../observables/observables.hpp"
#include "../../rng/rng.hpp"
#include "scene.hpp"

#include <cstdint>
#include <memory>

namespace lqft::vis
{

class GpuU1Sweeper;

class U1_2DScene : public Scene
{
public:
    explicit U1_2DScene(int L = 64, std::uint64_t seed = 9876u,
                        const VulkanContext* gpu_ctx = nullptr);
    ~U1_2DScene() override;

    int  latticeSize() const override { return m_L; }
    bool isVolumetric() const override { return false; }
    HeatmapStyle heatmapStyle() const override
    {
        // ρ = 1 − cos θ_□ ∈ [0, 2]; viridis-positive colormap.
        return HeatmapStyle{ /*gain*/1.0f, /*bias*/0.0f, /*gamma*/0.7f, /*signed*/0 };
    }
    void step() override;
    void fillVolume(float* out) const override;
    bool buildControlsUI() override;

private:
    void singleStep();
    void resetHot(std::uint64_t seed);
    void resetCold();
    void syncFromGpu() const;

    int                                              m_L;
    Lattice<2>                                       m_lattice;
    LinkField<double, 2>                             m_field;
    Rng                                              m_rng;
    double                                           m_beta = 1.5;
    u1::U1Model<2>                                   m_model;
    mc::MetropolisSweep<u1::U1Model<2>>              m_sweeper;

    int  m_sweeps_per_frame = 2;
    bool m_paused           = false;
    bool m_use_heatbath     = true;

    std::unique_ptr<GpuU1Sweeper>                    m_gpu;
    bool                                             m_use_gpu              = false;
    int                                              m_gpu_sweeps_per_frame = 30;
};

} // namespace lqft::vis
