#pragma once

/**
 * @file heatmap_2d_renderer.hpp
 * @brief 2D scalar-field heatmap renderer (counterpart to `VolumeRenderer`).
 *
 * Owns:
 *   - device-local `vk::Image` (R32_SFLOAT, 2D, extent L × L),
 *   - host-visible staging buffer for CPU → GPU updates each frame,
 *   - sampler (LINEAR, clamp-to-edge),
 *   - descriptor set / pipeline (`shaders/fullscreen.vert` + `heatmap_2d.frag`).
 *
 * Push constants pick between a viridis colormap (positive-only, for action
 * densities) and a diverging colormap (for ±1 Ising spins / signed φ⁴
 * scalar values).
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace lqft::vis
{

class Heatmap2DRenderer
{
public:
    struct PushConstants
    {
        float gain        = 1.0f;
        float bias        = 0.0f;
        float gamma       = 1.0f;
        int   signed_map  = 0;
    };

    Heatmap2DRenderer(vk::PhysicalDevice phys,
                      vk::Device         device,
                      vk::Queue          queue,
                      std::uint32_t      queue_family,
                      vk::CommandPool    cmd_pool,
                      vk::RenderPass     render_pass,
                      std::uint32_t      L,
                      const std::string& shader_dir);
    ~Heatmap2DRenderer();

    Heatmap2DRenderer(const Heatmap2DRenderer&)            = delete;
    Heatmap2DRenderer& operator=(const Heatmap2DRenderer&) = delete;

    /// Upload L*L floats to the texture (must be outside an active render pass).
    void upload(vk::CommandBuffer cmd, const float* data);

    /// Bind pipeline + descriptor, push constants, draw fullscreen triangle.
    void recordDraw(vk::CommandBuffer cmd,
                    vk::Extent2D viewport,
                    const PushConstants& push);

private:
    void createImageAndMemory();
    void createSampler();
    void createStagingBuffer();
    void createDescriptorResources();
    void createPipeline();

    vk::ShaderModule loadShader(const std::string& path) const;
    static std::vector<char> readFile(const std::string& path);
    std::uint32_t findMemoryType(std::uint32_t type_bits,
                                 vk::MemoryPropertyFlags props) const;

    vk::PhysicalDevice m_phys;
    vk::Device         m_device;
    vk::Queue          m_queue;
    std::uint32_t      m_queue_family;
    vk::CommandPool    m_cmd_pool;
    vk::RenderPass     m_render_pass;
    std::uint32_t      m_L;
    std::string        m_shader_dir;

    vk::Image          m_image;
    vk::DeviceMemory   m_image_memory;
    vk::ImageView      m_image_view;
    vk::Sampler        m_sampler;

    vk::Buffer         m_staging_buffer;
    vk::DeviceMemory   m_staging_memory;
    void*              m_staging_mapped = nullptr;
    vk::DeviceSize     m_staging_size   = 0;

    vk::DescriptorSetLayout m_descriptor_layout;
    vk::DescriptorPool      m_descriptor_pool;
    vk::DescriptorSet       m_descriptor_set;

    vk::PipelineLayout m_pipeline_layout;
    vk::Pipeline       m_pipeline;
    vk::ShaderModule   m_vert_module;
    vk::ShaderModule   m_frag_module;

    bool m_first_upload = true;
};

} // namespace lqft::vis
