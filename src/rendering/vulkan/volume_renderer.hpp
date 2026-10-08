#pragma once

/**
 * @file volume_renderer.hpp
 * @brief 3D scalar-field volumetric ray-marcher.
 *
 * Owns:
 *   - device-local `vk::Image` (R32_SFLOAT, 3D, extent L × L × L)
 *   - host-visible staging buffer for CPU → GPU uploads each frame
 *   - sampler (linear, clamp-to-edge) — linear filter avoids the gridded
 *     "lattice-cell" look from nearest filtering; ray-march samples often
 *     fall between texels and we want smooth opacity
 *   - descriptor set / layout / pool binding the sampler to fragment slot 0
 *   - graphics pipeline (fullscreen triangle → ray-march fragment shader)
 *
 * The PushConstants layout matches std430 in the shader:
 *
 *     mat4 inv_view_proj          // 64 B
 *     vec3 cam_pos    (+4 pad)    // 16 B
 *     float density_scale         //  4 B
 *     float opacity_threshold     //  4 B
 *     float gamma                 //  4 B
 *                                 // 92 B used; 128 B guaranteed.
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

#include <glm/glm.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace lqft::vis
{

class VolumeRenderer
{
public:
    struct PushConstants
    {
        glm::mat4 inv_view_proj;
        alignas(16) glm::vec3 cam_pos;
        float     density_scale;
        float     opacity_threshold;
        float     gamma;
    };
    static_assert(sizeof(PushConstants) <= 128,
                  "Push constants exceed Vulkan minimum guarantee");

    VolumeRenderer(vk::PhysicalDevice phys,
                   vk::Device         device,
                   vk::Queue          queue,
                   std::uint32_t      queue_family,
                   vk::CommandPool    cmd_pool,
                   vk::RenderPass     render_pass,
                   std::uint32_t      L,
                   const std::string& shader_dir);
    ~VolumeRenderer();

    VolumeRenderer(const VolumeRenderer&)            = delete;
    VolumeRenderer& operator=(const VolumeRenderer&) = delete;

    /// Copy L*L*L floats to the texture (must be outside an active render pass).
    void upload(vk::CommandBuffer cmd, const float* data);

    /// Set dynamic viewport/scissor, push the constants, draw the fullscreen
    /// triangle. Must be inside an active render pass.
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
