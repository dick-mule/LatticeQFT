#pragma once

/**
 * @file gpu_u1_sweeper.hpp
 * @brief GPU U(1) Metropolis sweeper.
 *
 * Mirror of `GpuSU2Sweeper` with:
 *   - storage buffer of single floats (`theta[V·Dim]`) instead of `vec4`s,
 *   - same per-site xoshiro128++ RNG state buffer,
 *   - the `shaders/u1_metropolis.comp` pipeline,
 *   - identical dispatch loop (one pass per (parity, μ) color, fenced wait).
 *
 * Host-visible buffers; ~196 KB total at L = 24.
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

#include "../../fields/link_field.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace lqft::vis
{

class GpuU1Sweeper
{
public:
    GpuU1Sweeper(vk::PhysicalDevice phys,
                 vk::Device         device,
                 vk::Queue          queue,
                 std::uint32_t      queue_family,
                 vk::CommandPool    cmd_pool,
                 int                L,
                 int                Dim,
                 const std::string& shader_dir,
                 std::uint64_t      rng_seed = 0xc0ffee123456789ULL);
    ~GpuU1Sweeper();

    GpuU1Sweeper(const GpuU1Sweeper&)            = delete;
    GpuU1Sweeper& operator=(const GpuU1Sweeper&) = delete;

    int latticeSize() const { return m_L; }
    int dimension() const { return m_Dim; }

    template<int Dim>
    void uploadField(const LinkField<double, Dim>& field)
    {
        auto* dst = static_cast<float*>(m_field_mapped);
        for (int s = 0; s < m_V; ++s)
            for (int mu = 0; mu < m_Dim; ++mu)
                dst[s * m_Dim + mu] = static_cast<float>(field(s, mu));
    }
    template<int Dim>
    void downloadField(LinkField<double, Dim>& field) const
    {
        const auto* src = static_cast<const float*>(m_field_mapped);
        for (int s = 0; s < m_V; ++s)
            for (int mu = 0; mu < m_Dim; ++mu)
                field(s, mu) = static_cast<double>(src[s * m_Dim + mu]);
    }

    void reseedRng    (std::uint64_t seed);
    void sweep        (int n_sweeps, double beta, double step_size);

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

    vk::PhysicalDevice m_phys;
    vk::Device         m_device;
    vk::Queue          m_queue;
    std::uint32_t      m_queue_family;
    vk::CommandPool    m_cmd_pool;
    int                m_L;
    int                m_V;
    int                m_Dim;

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
