#pragma once

/**
 * @file schwinger_quenched_scene.hpp
 * @brief Quenched 2D Schwinger model as a heatmap scene.
 *
 * Two-loop structure:
 *   - **Gauge sector**: U(1) Metropolis sweep on the GPU (reuses
 *     `GpuU1Sweeper<2>`) or CPU heat-bath as fallback.
 *   - **Fermion measurement**: every `N_meas_frames` frames, run the
 *     stochastic-condensate estimator on the current gauge configuration
 *     (CPU CG solve against the Wilson-Dirac operator). Running average
 *     over MC time displayed in the panel.
 *
 * Heatmap shows the gauge action density 1 − cos θ_□. The fermion
 * contribution is read out as ⟨ψ̄ψ⟩(β, m) in the side panel.
 */

#include "../../fields/link_field.hpp"
#include "../../lattice/lattice.hpp"
#include "../../math/dirac_2d.hpp"
#include "../../models/u1.hpp"
#include "../../monte_carlo/metropolis.hpp"
#include "../../monte_carlo/tuning.hpp"
#include "../../monte_carlo/u1_heatbath.hpp"
#include "../../observables/condensate.hpp"
#include "../../observables/observables.hpp"
#include "../../rng/rng.hpp"
#include "scene.hpp"

#include <cstdint>
#include <memory>

namespace lqft::vis
{

class GpuU1Sweeper;

class SchwingerQuenchedScene : public Scene
{
public:
    explicit SchwingerQuenchedScene(int L = 32, std::uint64_t seed = 31415u,
                                    const VulkanContext* gpu_ctx = nullptr);
    ~SchwingerQuenchedScene() override;

    int  latticeSize() const override { return m_L; }
    bool isVolumetric() const override { return false; }
    HeatmapStyle heatmapStyle() const override
    {
        return HeatmapStyle{ /*gain*/1.0f, /*bias*/0.0f, /*gamma*/0.7f, /*signed*/0 };
    }
    void step() override;
    void fillVolume(float* out) const override;
    bool buildControlsUI() override;

private:
    void resetHot(std::uint64_t seed);
    void resetCold();
    void syncFromGpu() const;
    void measureCondensate();
    void resetStats();

    int                                            m_L;
    Lattice<2>                                     m_lattice;
    LinkField<double, 2>                           m_field;
    Rng                                            m_rng;
    double                                         m_beta = 2.0;
    double                                         m_mass = 0.4;
    u1::U1Model<2>                                 m_model;
    mc::MetropolisSweep<u1::U1Model<2>>            m_sweeper;
    dirac::WilsonDirac2D                           m_dirac;

    int  m_sweeps_per_frame      = 2;
    int  m_meas_every_frames     = 4;
    int  m_n_stochastic_sources  = 3;
    bool m_paused                = false;
    bool m_use_heatbath          = true;

    // Running condensate
    long long m_n_measurements   = 0;
    double    m_sum_condensate   = 0.0;
    // CG iteration count from the most recent stochastic condensate
    // measurement. Surfaced in the panel so users watching the chiral
    // limit see κ(D†D) ~ 1/m² → iter count ~ √κ ~ 1/m balloon when
    // they slide the fermion mass toward zero.
    int       m_last_cg_iters    = 0;
    double    m_avg_cg_iters     = 0.0;

    int       m_frame_counter    = 0;

    std::unique_ptr<GpuU1Sweeper>                  m_gpu;
    bool                                           m_use_gpu              = false;
    int                                            m_gpu_sweeps_per_frame = 30;
};

} // namespace lqft::vis
