#pragma once

/**
 * @file schwinger_hmc_scene.hpp
 * @brief Dynamical 2D Schwinger model via pseudofermion HMC as a heatmap scene.
 *
 * Wraps the existing `schwinger::SchwingerProvider` (pseudofermion action +
 * fermion force from a CG solve per leapfrog step) and the generic
 * `hmc::GaugeHMC` engine. Per frame: a small batch of HMC trajectories,
 * each refreshing the pseudofermion and momentum and accepting/rejecting
 * on ΔH; then a stochastic condensate measurement.
 *
 * CPU-only — HMC with fermion forces on the GPU is a separate project
 * (large amount of compute-shader code for the per-leapfrog-step CG).
 */

#include "../../fields/link_field.hpp"
#include "../../lattice/lattice.hpp"
#include "../../math/dirac_2d.hpp"
#include "../../models/u1.hpp"
#include "../../monte_carlo/hmc.hpp"
#include "../../monte_carlo/schwinger_hmc.hpp"
#include "../../observables/condensate.hpp"
#include "../../observables/observables.hpp"
#include "../../rng/rng.hpp"
#include "scene.hpp"

#include <cstdint>
#include <memory>

namespace lqft::vis
{

class GpuWilsonDirac2D;  // forward declaration
class GpuCG2D;           // forward declaration
class GpuSchwingerForce; // forward declaration

class SchwingerHMCScene : public Scene
{
public:
    explicit SchwingerHMCScene(int L = 10, std::uint64_t seed = 27182u,
                               const VulkanContext* gpu_ctx = nullptr);
    ~SchwingerHMCScene() override;

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
    void resetStats();
    void rebuildHmc();

    int                                                m_L;
    Lattice<2>                                         m_lattice;
    LinkField<double, 2>                               m_field;
    Rng                                                m_rng;
    double                                             m_beta = 2.0;
    double                                             m_mass = 0.4;
    u1::U1Model<2>                                     m_gauge_model;
    dirac::WilsonDirac2D                               m_dirac;
    schwinger::SchwingerProvider                       m_provider;
    std::unique_ptr<hmc::GaugeHMC<schwinger::SchwingerProvider>> m_hmc;

    float m_hmc_dt              = 0.04f;
    int   m_hmc_n_steps         = 20;
    int   m_trajectories_per_frame = 1;
    bool  m_paused              = false;
    bool  m_use_gpu_hmc         = false; // switchable inside the scene

    // Stats
    long long m_n_trajectories  = 0;
    long long m_n_accepted      = 0;
    double    m_sum_dH          = 0.0;
    long long m_n_measurements  = 0;
    double    m_sum_condensate  = 0.0;
    int       m_meas_every_traj = 4;
    int       m_n_sources       = 3;

    // GPU Wilson-Dirac validation. Constructed if a Vulkan context is passed
    // in; on construction we run the GPU operator against the CPU operator
    // on a deterministic random configuration and record the max element-wise
    // error. The number is shown in the controls panel so the user has a
    // visible correctness signal independent of the HMC chain.
    std::unique_ptr<GpuWilsonDirac2D> m_gpu_dirac;
    double                            m_gpu_validation_err = -1.0;

    // GPU CG validation. Built alongside the GPU Wilson-Dirac when a Vulkan
    // context is present; runs one D†D solve on GPU and compares the result
    // to a CPU CG solve on the same configuration.
    std::unique_ptr<GpuCG2D> m_gpu_cg;
    double                   m_gpu_cg_rel_err = -1.0;

    // GPU Schwinger force validation. Stress-tests the full chain: CG +
    // Wilson-Dirac apply + gauge force + fermion force, all on GPU, vs the
    // CPU SchwingerProvider on the same configuration.
    std::unique_ptr<GpuSchwingerForce> m_gpu_force;
    double                              m_gpu_force_rel_err = -1.0;
    double                              m_gpu_traj_dH_err   = -1.0;
};

} // namespace lqft::vis
