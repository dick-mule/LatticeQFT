#pragma once

/**
 * @file ising_2d_scene.hpp
 * @brief 2D Ising model as a heatmap scene.
 *
 * Spins σ ∈ {−1, +1} on a 64² periodic lattice. CPU Metropolis sweep
 * (Phase-0 substrate) with a GPU compute fallback path. Heatmap uses the
 * diverging blue/yellow colormap so the two ordered phases are immediately
 * distinguishable from the disordered phase.
 *
 * Onsager critical inverse temperature β_c = ½ ln(1 + √2) ≈ 0.4407 is
 * marked on the β slider.
 */

#include "../../fields/site_field.hpp"
#include "../../lattice/lattice.hpp"
#include "../../models/ising.hpp"
#include "../../monte_carlo/metropolis.hpp"
#include "../../observables/observables.hpp"
#include "../../rng/rng.hpp"
#include "scene.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace lqft::vis
{

class GpuIsingSweeper;

class Ising2DScene : public Scene
{
public:
    explicit Ising2DScene(int L = 64, std::uint64_t seed = 1234u,
                          const VulkanContext* gpu_ctx = nullptr);
    ~Ising2DScene() override;

    int  latticeSize() const override { return m_L; }
    bool isVolumetric() const override { return false; }
    HeatmapStyle heatmapStyle() const override
    {
        return HeatmapStyle{ /*gain*/1.0f, /*bias*/0.0f, /*gamma*/1.0f, /*signed*/1 };
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
    SiteField<std::int8_t, 2>                        m_field;
    Rng                                              m_rng;
    double                                           m_beta = 0.44068679350977147;
    ising::IsingModel<2>                             m_model;
    mc::MetropolisSweep<ising::IsingModel<2>>        m_sweeper;

    int  m_sweeps_per_frame = 5;
    bool m_paused           = false;

    std::unique_ptr<GpuIsingSweeper>                 m_gpu;
    bool                                             m_use_gpu              = false;
    int                                              m_gpu_sweeps_per_frame = 50;
};

} // namespace lqft::vis
