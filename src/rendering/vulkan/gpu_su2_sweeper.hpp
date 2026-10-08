#pragma once

/**
 * @file gpu_su2_sweeper.hpp
 * @brief GPU SU(2) Metropolis sweeper.
 *
 * Owns:
 *   - A host-visible storage buffer for the SU(2) gauge field
 *     (`vec4[V · Dim]`).
 *   - A host-visible storage buffer for per-site xoshiro128++ state
 *     (`uvec4[V]`).
 *   - A compute pipeline running `shaders/su2_metropolis.comp.spv` with
 *     a descriptor set binding the two buffers and a push-constant block
 *     describing the lattice + color + β + step.
 *
 * Workflow:
 *   - `uploadField(...)` writes a CPU `LinkField<su2::Element, 3>` into
 *     the GPU buffer.
 *   - `sweep(n_sweeps, β, step_size)` records n_sweeps × (2·Dim) compute
 *     dispatches into a single command buffer, submits, and waits on a
 *     fence. Each dispatch handles one (parity, μ) color so the parallel
 *     updates within a color are independent.
 *   - `downloadField(...)` reads the GPU buffer back into the CPU field
 *     for measurement. The buffer is host-coherent so this is just a
 *     `memcpy`.
 *
 * Buffers are host-visible (256 KB at L = 16); for a production push to
 * larger lattices we'd switch to device-local + staging.
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

#include "../../fields/link_field.hpp"
#include "../../math/su2.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace lqft::vis
{

class GpuSU2Sweeper
{
public:
    GpuSU2Sweeper(vk::PhysicalDevice phys,
                  vk::Device         device,
                  vk::Queue          queue,
                  std::uint32_t      queue_family,
                  vk::CommandPool    cmd_pool,
                  int                L,
                  int                Dim,
                  const std::string& shader_dir,
                  std::uint64_t      rng_seed = 0xdeadbeefcafebabeULL);
    ~GpuSU2Sweeper();

    GpuSU2Sweeper(const GpuSU2Sweeper&)            = delete;
    GpuSU2Sweeper& operator=(const GpuSU2Sweeper&) = delete;

    int latticeSize() const { return m_L; }
    int dimension() const { return m_Dim; }

    template<int Dim>
    void uploadField(const LinkField<su2::Element, Dim>& field)
    {
        auto* dst = static_cast<float*>(m_field_mapped);
        for (int s = 0; s < m_V; ++s)
            for (int mu = 0; mu < m_Dim; ++mu)
            {
                const auto& U = field(s, mu);
                const int idx = (s * m_Dim + mu) * 4;
                dst[idx + 0] = static_cast<float>(U.s);
                dst[idx + 1] = static_cast<float>(U.v[0]);
                dst[idx + 2] = static_cast<float>(U.v[1]);
                dst[idx + 3] = static_cast<float>(U.v[2]);
            }
    }
    template<int Dim>
    void downloadField(LinkField<su2::Element, Dim>& field) const
    {
        const auto* src = static_cast<const float*>(m_field_mapped);
        for (int s = 0; s < m_V; ++s)
            for (int mu = 0; mu < m_Dim; ++mu)
            {
                const int idx = (s * m_Dim + mu) * 4;
                su2::Element U;
                U.s    = src[idx + 0];
                U.v[0] = src[idx + 1];
                U.v[1] = src[idx + 2];
                U.v[2] = src[idx + 3];
                field(s, mu) = U;
            }
    }

    /// Re-seed every site's RNG stream from `seed`.
    void reseedRng(std::uint64_t seed);

    /// Run `n_sweeps` × (2·Dim) Metropolis dispatches at fixed β.
    void sweep(int n_sweeps, double beta, double step_size);

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
        int   Lx, Ly, Lz, Dim;
        int   color_parity;
        int   color_mu;
        float beta;
        float step_size;
        int   sweep_count;
    };
    static_assert(sizeof(PushConstants) <= 128,
                  "push constants exceed Vulkan minimum guarantee");

    // ----- non-owning refs -----
    vk::PhysicalDevice m_phys;
    vk::Device         m_device;
    vk::Queue          m_queue;
    std::uint32_t      m_queue_family;
    vk::CommandPool    m_cmd_pool;

    int                m_L;
    int                m_V;
    int                m_Dim;

    // ----- owned -----
    vk::Buffer         m_field_buf;
    vk::DeviceMemory   m_field_mem;
    void*              m_field_mapped = nullptr;
    vk::DeviceSize     m_field_bytes  = 0;

    vk::Buffer         m_rng_buf;
    vk::DeviceMemory   m_rng_mem;
    void*              m_rng_mapped   = nullptr;
    vk::DeviceSize     m_rng_bytes    = 0;

    vk::DescriptorSetLayout m_desc_layout;
    vk::DescriptorPool      m_desc_pool;
    vk::DescriptorSet       m_desc_set;

    vk::PipelineLayout m_pipe_layout;
    vk::Pipeline       m_pipeline;
    vk::ShaderModule   m_shader;

    vk::CommandBuffer  m_cmd;
    vk::Fence          m_fence;

    int                m_sweep_count = 0;
};

} // namespace lqft::vis
