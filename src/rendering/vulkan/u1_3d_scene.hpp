#pragma once

/**
 * @file u1_3d_scene.hpp
 * @brief 3D compact U(1) lattice gauge theory as a volumetric scene.
 *
 * Per-frame: a few Monte Carlo sweeps (Metropolis or Best–Fisher heat-bath,
 * toggleable from the ImGui panel) of the Wilson plaquette action
 *
 *   S[θ] = β Σ_□ (1 − cos θ_□)
 *
 * and `fillVolume(out)` writes the action density
 *
 *   ρ(x) = Σ_{μ<ν=0..2} (1 − cos θ_□(x; μ, ν))   ∈ [0, 6]
 *
 * to the volumetric texture. The user sees the gauge field thermalize
 * in real time and can drag β to walk the strong↔weak coupling transition.
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

class U1Scene3D : public Scene
{
public:
    explicit U1Scene3D(int L = 24, std::uint64_t seed = 12345u,
                       const VulkanContext* gpu_ctx = nullptr);
    ~U1Scene3D() override;

    int  latticeSize() const override { return m_L; }
    void step() override;
    void fillVolume(float* out) const override;
    bool buildControlsUI() override;

private:
    void singleStep();
    void resetHot(std::uint64_t seed);
    void resetCold();

    int                            m_L;
    Lattice<3>                     m_lattice;
    LinkField<double, 3>           m_field;
    Rng                            m_rng;

    double                         m_beta             = 1.7;
    u1::U1Model<3>                 m_model;
    mc::MetropolisSweep<u1::U1Model<3>> m_sweeper;

    int                            m_sweeps_per_frame = 1;
    bool                           m_paused           = false;
    bool                           m_use_heatbath     = true;

    std::unique_ptr<GpuU1Sweeper>  m_gpu;
    bool                           m_use_gpu              = false;
    int                            m_gpu_sweeps_per_frame = 30;
};

} // namespace lqft::vis
