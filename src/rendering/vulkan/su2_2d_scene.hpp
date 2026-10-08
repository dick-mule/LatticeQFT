#pragma once

/**
 * @file su2_2d_scene.hpp
 * @brief 2D SU(2) Yang-Mills as a heatmap scene.
 *
 * Wilson plaquette action  S = β Σ_□ (1 − ½ tr U_□)  on a 64² lattice.
 * Heatmap shows the per-site (1 − ½ tr U_□) action density. Reuses the
 * existing `GpuSU2Sweeper` instantiated with Dim = 2.
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

class SU2_2DScene : public Scene
{
public:
    explicit SU2_2DScene(int L = 48, std::uint64_t seed = 13579u,
                         const VulkanContext* gpu_ctx = nullptr);
    ~SU2_2DScene() override;

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
    void singleStep();
    void resetHot(std::uint64_t seed);
    void resetCold();
    void syncFromGpu() const;

    int                                                m_L;
    Lattice<2>                                         m_lattice;
    LinkField<su2::Element, 2>                         m_field;
    Rng                                                m_rng;
    double                                             m_beta = 2.5;
    su2_model::SU2Model<2>                             m_model;
    mc::MetropolisSweep<su2_model::SU2Model<2>>        m_sweeper;

    int  m_sweeps_per_frame = 1;
    bool m_paused           = false;
    bool m_use_heatbath     = true;

    std::unique_ptr<GpuSU2Sweeper>                     m_gpu;
    bool                                               m_use_gpu              = false;
    int                                                m_gpu_sweeps_per_frame = 30;
};

} // namespace lqft::vis
