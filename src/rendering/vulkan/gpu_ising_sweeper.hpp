#pragma once

/**
 * @file gpu_ising_sweeper.hpp
 * @brief GPU 2D Ising Metropolis sweeper.
 *
 * Storage: `int32[V]` (spins ±1) + `uvec4[V]` (xoshiro128++ per site).
 * One dispatch per parity (2 dispatches per sweep). Reuses the same
 * push-constant style as the gauge sweepers.
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace lqft::vis
{

class GpuIsingSweeper
{
public:
    GpuIsingSweeper(vk::PhysicalDevice phys,
                    vk::Device         device,
                    vk::Queue          queue,
                    std::uint32_t      queue_family,
                    vk::CommandPool    cmd_pool,
                    int                L,
                    const std::string& shader_dir,
                    std::uint64_t      rng_seed = 0xabcdef0123456789ULL);
    ~GpuIsingSweeper();

    GpuIsingSweeper(const GpuIsingSweeper&)            = delete;
    GpuIsingSweeper& operator=(const GpuIsingSweeper&) = delete;

    int latticeSize() const { return m_L; }

    void uploadSpins  (const std::int8_t* spins);          // size L*L
    void downloadSpins(std::int8_t* spins) const;          // size L*L
    void reseedRng    (std::uint64_t seed);
    void sweep        (int n_sweeps, double beta);

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
        int   Lx, Ly;
        int   color_parity;
        float beta;
        int   sweep_count;
    };

    vk::PhysicalDevice m_phys;
    vk::Device         m_device;
    vk::Queue          m_queue;
    std::uint32_t      m_queue_family;
    vk::CommandPool    m_cmd_pool;
    int                m_L;
    int                m_V;

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
