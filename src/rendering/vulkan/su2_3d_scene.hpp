#pragma once

/**
 * @file su2_3d_scene.hpp
 * @brief 3D SU(2) pure-gauge Yang-Mills as a volumetric scene.
 *
 * Non-Abelian counterpart to `U1Scene3D`. Same MC engine substrate; the
 * only changes from U(1) are the link DOF type (`su2::Element` 4-vector
 * vs `double` angle), the staple algebra, and the heat-bath sampler
 * (Kennedy–Pendleton 1985). The volume diagnostic is the per-site action
 * density
 *
 *   ρ(x) = Σ_{μ<ν=0..2} (1 − ½ tr U_□(x; μ, ν))   ∈ [0, 3]
 *
 * which goes through the same R32_SFLOAT texture and ray-march fragment
 * shader the U(1) scene uses — `VolumeRenderer` is gauge-group agnostic
 * by design.
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

namespace lqft::vis
{

class GpuSU2Sweeper;

class SU2Scene3D : public Scene
{
public:
    explicit SU2Scene3D(int L = 20, std::uint64_t seed = 67890u,
                       const VulkanContext* gpu_ctx = nullptr);
    ~SU2Scene3D() override;

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
    LinkField<su2::Element, 3>     m_field;
    Rng                            m_rng;

    double                         m_beta             = 2.0;
    su2_model::SU2Model<3>         m_model;
    mc::MetropolisSweep<su2_model::SU2Model<3>> m_sweeper;

    int                            m_sweeps_per_frame = 1;
    bool                           m_paused           = false;
    bool                           m_use_heatbath     = true;

    std::unique_ptr<GpuSU2Sweeper> m_gpu;
    bool                           m_use_gpu              = false;
    int                            m_gpu_sweeps_per_frame = 30;
};

} // namespace lqft::vis
