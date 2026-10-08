#pragma once

/**
 * @file gpu_wilson_dirac_2d.hpp
 * @brief GPU implementation of the 2D Wilson-Dirac operator for U(1).
 *
 * One compute pass over the lattice volume per apply: the shader
 * `shaders/wilson_dirac_2d.comp` reads a gauge field θ_μ(x) and an input
 * spinor ψ(x), writes D ψ (or D† ψ depending on a push-constant flag) to
 * an output spinor buffer.
 *
 * The class owns three storage buffers:
 *
 *   - gauge: V · 2 floats (link angles)
 *   - psi_in, psi_out: V · 2 vec2 (complex spinor components)
 *
 * Multiple input/output spinor slots are not provided yet — host-side CG
 * will need them and is a follow-up. For now the apply uploads psi_in,
 * dispatches, and downloads psi_out per call (slow but correct, which is
 * what we want for the first cut + validation).
 *
 * `selfValidate()` runs both the CPU `dirac::WilsonDirac2D::apply` and the
 * GPU apply on a deterministic random gauge+spinor pair and returns the
 * maximum element-wise error — call it once after construction to confirm
 * the SPIR-V kernel matches the CPU implementation bit-equivalent (up to
 * float ↔ double rounding).
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

#include "../../fields/link_field.hpp"
#include "../../fields/spinor_field.hpp"
#include "../../lattice/lattice.hpp"
#include "../../math/dirac_2d.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace lqft::vis
{

class GpuWilsonDirac2D
{
public:
    GpuWilsonDirac2D(vk::PhysicalDevice  phys,
                     vk::Device          device,
                     vk::Queue           queue,
                     std::uint32_t       queue_family,
                     vk::CommandPool     cmd_pool,
                     int                 L,
                     const std::string&  shader_dir);
    ~GpuWilsonDirac2D();

    GpuWilsonDirac2D(const GpuWilsonDirac2D&)            = delete;
    GpuWilsonDirac2D& operator=(const GpuWilsonDirac2D&) = delete;

    int latticeSize() const { return m_L; }

    /// Upload gauge field θ_μ(x) ∈ ℝ.
    void uploadGauge(const LinkField<double, 2>& U);

    /// Upload / download spinor ψ(x) ∈ ℂ². Spinors are packed as vec2 per
    /// component (real, imag).
    void uploadInputSpinor (const SpinorField<2, 2>& psi);
    void downloadOutputSpinor(SpinorField<2, 2>& psi) const;

    /// Apply D (dagger = false) or D† (dagger = true) to the uploaded input
    /// spinor; result lands in the output buffer (download separately).
    void apply(double mass, bool dagger);

    /// All-in-one: upload everything, dispatch, download. Equivalent to
    ///   uploadGauge(U); uploadInputSpinor(in); apply(mass, dagger); downloadOutputSpinor(out);
    void applyFull(const LinkField<double, 2>&  U,
                   const SpinorField<2, 2>&     in,
                   double                       mass,
                   bool                         dagger,
                   SpinorField<2, 2>&           out);

    /// Run CPU and GPU side by side on a deterministic random configuration
    /// and return the maximum element-wise error in the output spinor. A
    /// value below ~1e-5 means the GPU shader matches the CPU operator (the
    /// floor is set by float vs double rounding inside the apply).
    double selfValidate(double mass = 0.4, std::uint64_t seed = 7u);

private:
    void createBuffers();
    void createDescriptors();
    void createPipeline(const std::string& shader_dir);
    void allocateCommandBuffer();

    static std::vector<char> readFile(const std::string& path);
    std::uint32_t findMemoryType(std::uint32_t type_bits,
                                 vk::MemoryPropertyFlags props) const;

    struct PushConstants
    {
        int   Lx;
        int   Ly;
        float mass;
        int   dagger;
    };

    vk::PhysicalDevice m_phys;
    vk::Device         m_device;
    vk::Queue          m_queue;
    std::uint32_t      m_queue_family;
    vk::CommandPool    m_cmd_pool;
    int                m_L;
    int                m_V;

    vk::Buffer       m_gauge_buf;
    vk::DeviceMemory m_gauge_mem;
    void*            m_gauge_mapped = nullptr;
    vk::DeviceSize   m_gauge_bytes  = 0;

    vk::Buffer       m_psi_in_buf;
    vk::DeviceMemory m_psi_in_mem;
    void*            m_psi_in_mapped = nullptr;
    vk::DeviceSize   m_psi_bytes     = 0;

    vk::Buffer       m_psi_out_buf;
    vk::DeviceMemory m_psi_out_mem;
    void*            m_psi_out_mapped = nullptr;

    vk::DescriptorSetLayout m_desc_layout;
    vk::DescriptorPool      m_desc_pool;
    vk::DescriptorSet       m_desc_set;

    vk::PipelineLayout m_pipe_layout;
    vk::Pipeline       m_pipeline;
    vk::ShaderModule   m_shader;

    vk::CommandBuffer  m_cmd;
    vk::Fence          m_fence;
};

} // namespace lqft::vis
