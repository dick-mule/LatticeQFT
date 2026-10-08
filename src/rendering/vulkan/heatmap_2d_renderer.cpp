#include "heatmap_2d_renderer.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace lqft::vis
{

Heatmap2DRenderer::Heatmap2DRenderer(vk::PhysicalDevice phys,
                                     vk::Device         device,
                                     vk::Queue          queue,
                                     std::uint32_t      queue_family,
                                     vk::CommandPool    cmd_pool,
                                     vk::RenderPass     render_pass,
                                     std::uint32_t      L,
                                     const std::string& shader_dir)
    : m_phys(phys), m_device(device), m_queue(queue)
    , m_queue_family(queue_family), m_cmd_pool(cmd_pool)
    , m_render_pass(render_pass), m_L(L), m_shader_dir(shader_dir)
{
    createImageAndMemory();
    createSampler();
    createStagingBuffer();
    createDescriptorResources();
    createPipeline();
}

Heatmap2DRenderer::~Heatmap2DRenderer()
{
    if (!m_device) return;

    if (m_pipeline)          m_device.destroyPipeline(m_pipeline);
    if (m_pipeline_layout)   m_device.destroyPipelineLayout(m_pipeline_layout);
    if (m_vert_module)       m_device.destroyShaderModule(m_vert_module);
    if (m_frag_module)       m_device.destroyShaderModule(m_frag_module);

    if (m_descriptor_pool)   m_device.destroyDescriptorPool(m_descriptor_pool);
    if (m_descriptor_layout) m_device.destroyDescriptorSetLayout(m_descriptor_layout);

    if (m_staging_mapped)    m_device.unmapMemory(m_staging_memory);
    if (m_staging_buffer)    m_device.destroyBuffer(m_staging_buffer);
    if (m_staging_memory)    m_device.freeMemory(m_staging_memory);

    if (m_sampler)           m_device.destroySampler(m_sampler);
    if (m_image_view)        m_device.destroyImageView(m_image_view);
    if (m_image)             m_device.destroyImage(m_image);
    if (m_image_memory)      m_device.freeMemory(m_image_memory);
}

// -----------------------------------------------------------------------------
// Image + sampler + staging
// -----------------------------------------------------------------------------

void Heatmap2DRenderer::createImageAndMemory()
{
    vk::ImageCreateInfo ici{};
    ici.imageType     = vk::ImageType::e2D;
    ici.format        = vk::Format::eR32Sfloat;
    ici.extent        = vk::Extent3D{ m_L, m_L, 1 };
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.samples       = vk::SampleCountFlagBits::e1;
    ici.tiling        = vk::ImageTiling::eOptimal;
    ici.usage         = vk::ImageUsageFlagBits::eTransferDst
                      | vk::ImageUsageFlagBits::eSampled;
    ici.sharingMode   = vk::SharingMode::eExclusive;
    ici.initialLayout = vk::ImageLayout::eUndefined;
    m_image = m_device.createImage(ici);

    auto req = m_device.getImageMemoryRequirements(m_image);
    vk::MemoryAllocateInfo mai{};
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = findMemoryType(req.memoryTypeBits,
                                         vk::MemoryPropertyFlagBits::eDeviceLocal);
    m_image_memory = m_device.allocateMemory(mai);
    m_device.bindImageMemory(m_image, m_image_memory, 0);

    vk::ImageViewCreateInfo vci{};
    vci.image    = m_image;
    vci.viewType = vk::ImageViewType::e2D;
    vci.format   = vk::Format::eR32Sfloat;
    vci.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
    vci.subresourceRange.baseMipLevel   = 0;
    vci.subresourceRange.levelCount     = 1;
    vci.subresourceRange.baseArrayLayer = 0;
    vci.subresourceRange.layerCount     = 1;
    m_image_view = m_device.createImageView(vci);
}

void Heatmap2DRenderer::createSampler()
{
    vk::SamplerCreateInfo sci{};
    sci.magFilter    = vk::Filter::eLinear;
    sci.minFilter    = vk::Filter::eLinear;
    sci.addressModeU = vk::SamplerAddressMode::eClampToEdge;
    sci.addressModeV = vk::SamplerAddressMode::eClampToEdge;
    sci.addressModeW = vk::SamplerAddressMode::eClampToEdge;
    sci.borderColor  = vk::BorderColor::eIntOpaqueBlack;
    sci.compareOp    = vk::CompareOp::eAlways;
    sci.mipmapMode   = vk::SamplerMipmapMode::eNearest;
    m_sampler = m_device.createSampler(sci);
}

void Heatmap2DRenderer::createStagingBuffer()
{
    m_staging_size = static_cast<vk::DeviceSize>(m_L) * m_L * sizeof(float);

    vk::BufferCreateInfo bci{};
    bci.size        = m_staging_size;
    bci.usage       = vk::BufferUsageFlagBits::eTransferSrc;
    bci.sharingMode = vk::SharingMode::eExclusive;
    m_staging_buffer = m_device.createBuffer(bci);

    auto req = m_device.getBufferMemoryRequirements(m_staging_buffer);
    vk::MemoryAllocateInfo mai{};
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = findMemoryType(req.memoryTypeBits,
        vk::MemoryPropertyFlagBits::eHostVisible
      | vk::MemoryPropertyFlagBits::eHostCoherent);
    m_staging_memory = m_device.allocateMemory(mai);
    m_device.bindBufferMemory(m_staging_buffer, m_staging_memory, 0);
    m_staging_mapped = m_device.mapMemory(m_staging_memory, 0, m_staging_size);
}

void Heatmap2DRenderer::createDescriptorResources()
{
    vk::DescriptorSetLayoutBinding binding{};
    binding.binding         = 0;
    binding.descriptorType  = vk::DescriptorType::eCombinedImageSampler;
    binding.descriptorCount = 1;
    binding.stageFlags      = vk::ShaderStageFlagBits::eFragment;

    vk::DescriptorSetLayoutCreateInfo lci{};
    lci.bindingCount = 1;
    lci.pBindings    = &binding;
    m_descriptor_layout = m_device.createDescriptorSetLayout(lci);

    vk::DescriptorPoolSize pool_size{};
    pool_size.type            = vk::DescriptorType::eCombinedImageSampler;
    pool_size.descriptorCount = 1;
    vk::DescriptorPoolCreateInfo pci{};
    pci.maxSets       = 1;
    pci.poolSizeCount = 1;
    pci.pPoolSizes    = &pool_size;
    m_descriptor_pool = m_device.createDescriptorPool(pci);

    vk::DescriptorSetAllocateInfo dsai{};
    dsai.descriptorPool     = m_descriptor_pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &m_descriptor_layout;
    m_descriptor_set = m_device.allocateDescriptorSets(dsai).front();

    vk::DescriptorImageInfo dii{};
    dii.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    dii.imageView   = m_image_view;
    dii.sampler     = m_sampler;

    vk::WriteDescriptorSet w{};
    w.dstSet          = m_descriptor_set;
    w.dstBinding      = 0;
    w.descriptorCount = 1;
    w.descriptorType  = vk::DescriptorType::eCombinedImageSampler;
    w.pImageInfo      = &dii;
    m_device.updateDescriptorSets(1, &w, 0, nullptr);
}

// -----------------------------------------------------------------------------
// Pipeline
// -----------------------------------------------------------------------------

void Heatmap2DRenderer::createPipeline()
{
    m_vert_module = loadShader(m_shader_dir + "/fullscreen.vert.spv");
    m_frag_module = loadShader(m_shader_dir + "/heatmap_2d.frag.spv");

    vk::PipelineShaderStageCreateInfo vert{};
    vert.stage  = vk::ShaderStageFlagBits::eVertex;
    vert.module = m_vert_module;
    vert.pName  = "main";
    vk::PipelineShaderStageCreateInfo frag{};
    frag.stage  = vk::ShaderStageFlagBits::eFragment;
    frag.module = m_frag_module;
    frag.pName  = "main";
    vk::PipelineShaderStageCreateInfo stages[] = { vert, frag };

    vk::PipelineVertexInputStateCreateInfo vi{};
    vk::PipelineInputAssemblyStateCreateInfo ia{};
    ia.topology = vk::PrimitiveTopology::eTriangleList;

    vk::PipelineViewportStateCreateInfo vp{};
    vp.viewportCount = 1;
    vp.scissorCount  = 1;

    vk::PipelineRasterizationStateCreateInfo rs{};
    rs.polygonMode = vk::PolygonMode::eFill;
    rs.cullMode    = vk::CullModeFlagBits::eNone;
    rs.frontFace   = vk::FrontFace::eCounterClockwise;
    rs.lineWidth   = 1.0f;

    vk::PipelineMultisampleStateCreateInfo ms{};
    ms.rasterizationSamples = vk::SampleCountFlagBits::e1;

    vk::PipelineColorBlendAttachmentState cb_att{};
    cb_att.colorWriteMask = vk::ColorComponentFlagBits::eR
                          | vk::ColorComponentFlagBits::eG
                          | vk::ColorComponentFlagBits::eB
                          | vk::ColorComponentFlagBits::eA;
    cb_att.blendEnable    = VK_FALSE;
    vk::PipelineColorBlendStateCreateInfo cb{};
    cb.attachmentCount = 1;
    cb.pAttachments    = &cb_att;

    vk::DynamicState dyn_states[] = { vk::DynamicState::eViewport,
                                      vk::DynamicState::eScissor };
    vk::PipelineDynamicStateCreateInfo dyn{};
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates    = dyn_states;

    vk::PushConstantRange pcr{};
    pcr.stageFlags = vk::ShaderStageFlagBits::eFragment;
    pcr.offset     = 0;
    pcr.size       = sizeof(PushConstants);

    vk::PipelineLayoutCreateInfo pl{};
    pl.setLayoutCount         = 1;
    pl.pSetLayouts            = &m_descriptor_layout;
    pl.pushConstantRangeCount = 1;
    pl.pPushConstantRanges    = &pcr;
    m_pipeline_layout = m_device.createPipelineLayout(pl);

    vk::GraphicsPipelineCreateInfo gpi{};
    gpi.stageCount          = 2;
    gpi.pStages             = stages;
    gpi.pVertexInputState   = &vi;
    gpi.pInputAssemblyState = &ia;
    gpi.pViewportState      = &vp;
    gpi.pRasterizationState = &rs;
    gpi.pMultisampleState   = &ms;
    gpi.pColorBlendState    = &cb;
    gpi.pDynamicState       = &dyn;
    gpi.layout              = m_pipeline_layout;
    gpi.renderPass          = m_render_pass;
    gpi.subpass             = 0;
    auto pr = m_device.createGraphicsPipeline(nullptr, gpi);
    if (pr.result != vk::Result::eSuccess)
        throw std::runtime_error("Heatmap2DRenderer::createGraphicsPipeline failed");
    m_pipeline = pr.value;
}

// -----------------------------------------------------------------------------
// Per-frame upload / draw
// -----------------------------------------------------------------------------

void Heatmap2DRenderer::upload(vk::CommandBuffer cmd, const float* data)
{
    std::memcpy(m_staging_mapped, data, m_staging_size);

    vk::ImageMemoryBarrier b0{};
    b0.srcAccessMask = m_first_upload
        ? vk::AccessFlags{}
        : vk::AccessFlagBits::eShaderRead;
    b0.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
    b0.oldLayout     = m_first_upload
        ? vk::ImageLayout::eUndefined
        : vk::ImageLayout::eShaderReadOnlyOptimal;
    b0.newLayout     = vk::ImageLayout::eTransferDstOptimal;
    b0.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b0.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b0.image         = m_image;
    b0.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
    b0.subresourceRange.baseMipLevel   = 0;
    b0.subresourceRange.levelCount     = 1;
    b0.subresourceRange.baseArrayLayer = 0;
    b0.subresourceRange.layerCount     = 1;

    cmd.pipelineBarrier(
        m_first_upload ? vk::PipelineStageFlagBits::eTopOfPipe
                       : vk::PipelineStageFlagBits::eFragmentShader,
        vk::PipelineStageFlagBits::eTransfer,
        {}, {}, {}, b0);

    vk::BufferImageCopy region{};
    region.bufferOffset      = 0;
    region.bufferRowLength   = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    region.imageSubresource.mipLevel       = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount     = 1;
    region.imageOffset       = vk::Offset3D{ 0, 0, 0 };
    region.imageExtent       = vk::Extent3D{ m_L, m_L, 1 };
    cmd.copyBufferToImage(m_staging_buffer, m_image,
        vk::ImageLayout::eTransferDstOptimal, region);

    vk::ImageMemoryBarrier b1 = b0;
    b1.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    b1.dstAccessMask = vk::AccessFlagBits::eShaderRead;
    b1.oldLayout     = vk::ImageLayout::eTransferDstOptimal;
    b1.newLayout     = vk::ImageLayout::eShaderReadOnlyOptimal;
    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eFragmentShader,
        {}, {}, {}, b1);

    m_first_upload = false;
}

void Heatmap2DRenderer::recordDraw(vk::CommandBuffer cmd,
                                   vk::Extent2D viewport,
                                   const PushConstants& push)
{
    vk::Viewport vp{};
    vp.x        = 0.0f;
    vp.y        = 0.0f;
    vp.width    = static_cast<float>(viewport.width);
    vp.height   = static_cast<float>(viewport.height);
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;
    cmd.setViewport(0, 1, &vp);
    vk::Rect2D scissor{ { 0, 0 }, viewport };
    cmd.setScissor(0, 1, &scissor);

    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                           m_pipeline_layout, 0, 1, &m_descriptor_set,
                           0, nullptr);
    cmd.pushConstants(m_pipeline_layout,
                      vk::ShaderStageFlagBits::eFragment,
                      0, sizeof(PushConstants), &push);
    cmd.draw(3, 1, 0, 0);
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

std::vector<char> Heatmap2DRenderer::readFile(const std::string& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) throw std::runtime_error("failed to open " + path);
    const std::size_t sz = static_cast<std::size_t>(f.tellg());
    std::vector<char> buf(sz);
    f.seekg(0);
    f.read(buf.data(), static_cast<std::streamsize>(sz));
    return buf;
}

vk::ShaderModule Heatmap2DRenderer::loadShader(const std::string& path) const
{
    auto bytes = readFile(path);
    vk::ShaderModuleCreateInfo ci{};
    ci.codeSize = bytes.size();
    ci.pCode    = reinterpret_cast<const std::uint32_t*>(bytes.data());
    return m_device.createShaderModule(ci);
}

std::uint32_t Heatmap2DRenderer::findMemoryType(std::uint32_t type_bits,
                                                vk::MemoryPropertyFlags props) const
{
    auto mem = m_phys.getMemoryProperties();
    for (std::uint32_t i = 0; i < mem.memoryTypeCount; ++i)
    {
        if ((type_bits & (1u << i))
            && (mem.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    throw std::runtime_error("no suitable memory type");
}

} // namespace lqft::vis
