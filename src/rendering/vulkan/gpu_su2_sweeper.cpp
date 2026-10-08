#include "gpu_su2_sweeper.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace lqft::vis
{

namespace
{

inline std::uint64_t splitmix64(std::uint64_t& state)
{
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

} // anonymous namespace

// -----------------------------------------------------------------------------
// Construction
// -----------------------------------------------------------------------------

GpuSU2Sweeper::GpuSU2Sweeper(vk::PhysicalDevice phys,
                             vk::Device         device,
                             vk::Queue          queue,
                             std::uint32_t      queue_family,
                             vk::CommandPool    cmd_pool,
                             int                L,
                             int                Dim,
                             const std::string& shader_dir,
                             std::uint64_t      rng_seed)
    : m_phys(phys), m_device(device), m_queue(queue)
    , m_queue_family(queue_family), m_cmd_pool(cmd_pool)
    , m_L(L)
    , m_V((Dim == 2) ? L * L : L * L * L)
    , m_Dim(Dim)
{
    createBuffers();
    createDescriptors();
    createPipeline(shader_dir);
    allocateCommandBuffer();
    reseedRng(rng_seed);
}

GpuSU2Sweeper::~GpuSU2Sweeper()
{
    if (!m_device) return;

    if (m_fence)        m_device.destroyFence(m_fence);
    if (m_cmd)
        m_device.freeCommandBuffers(m_cmd_pool, m_cmd);

    if (m_pipeline)     m_device.destroyPipeline(m_pipeline);
    if (m_pipe_layout)  m_device.destroyPipelineLayout(m_pipe_layout);
    if (m_shader)       m_device.destroyShaderModule(m_shader);

    if (m_desc_pool)    m_device.destroyDescriptorPool(m_desc_pool);
    if (m_desc_layout)  m_device.destroyDescriptorSetLayout(m_desc_layout);

    if (m_rng_mapped)   m_device.unmapMemory(m_rng_mem);
    if (m_rng_buf)      m_device.destroyBuffer(m_rng_buf);
    if (m_rng_mem)      m_device.freeMemory(m_rng_mem);

    if (m_field_mapped) m_device.unmapMemory(m_field_mem);
    if (m_field_buf)    m_device.destroyBuffer(m_field_buf);
    if (m_field_mem)    m_device.freeMemory(m_field_mem);
}

// -----------------------------------------------------------------------------
// Resource creation
// -----------------------------------------------------------------------------

void GpuSU2Sweeper::createBuffers()
{
    // Gauge field: V · Dim links × 4 floats (vec4) per link.
    m_field_bytes = static_cast<vk::DeviceSize>(m_V) * m_Dim * sizeof(float) * 4;

    vk::BufferCreateInfo bci_field{};
    bci_field.size        = m_field_bytes;
    bci_field.usage       = vk::BufferUsageFlagBits::eStorageBuffer;
    bci_field.sharingMode = vk::SharingMode::eExclusive;
    m_field_buf = m_device.createBuffer(bci_field);

    auto req = m_device.getBufferMemoryRequirements(m_field_buf);
    vk::MemoryAllocateInfo mai{};
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = findMemoryType(
        req.memoryTypeBits,
        vk::MemoryPropertyFlagBits::eHostVisible
      | vk::MemoryPropertyFlagBits::eHostCoherent);
    m_field_mem = m_device.allocateMemory(mai);
    m_device.bindBufferMemory(m_field_buf, m_field_mem, 0);
    m_field_mapped = m_device.mapMemory(m_field_mem, 0, m_field_bytes);

    // RNG state: V uvec4 per site.
    m_rng_bytes = static_cast<vk::DeviceSize>(m_V) * sizeof(std::uint32_t) * 4;

    vk::BufferCreateInfo bci_rng{};
    bci_rng.size        = m_rng_bytes;
    bci_rng.usage       = vk::BufferUsageFlagBits::eStorageBuffer;
    bci_rng.sharingMode = vk::SharingMode::eExclusive;
    m_rng_buf = m_device.createBuffer(bci_rng);

    auto req_rng = m_device.getBufferMemoryRequirements(m_rng_buf);
    vk::MemoryAllocateInfo mai_rng{};
    mai_rng.allocationSize  = req_rng.size;
    mai_rng.memoryTypeIndex = findMemoryType(
        req_rng.memoryTypeBits,
        vk::MemoryPropertyFlagBits::eHostVisible
      | vk::MemoryPropertyFlagBits::eHostCoherent);
    m_rng_mem = m_device.allocateMemory(mai_rng);
    m_device.bindBufferMemory(m_rng_buf, m_rng_mem, 0);
    m_rng_mapped = m_device.mapMemory(m_rng_mem, 0, m_rng_bytes);
}

void GpuSU2Sweeper::createDescriptors()
{
    std::array<vk::DescriptorSetLayoutBinding, 2> bindings{};
    bindings[0].binding         = 0;
    bindings[0].descriptorType  = vk::DescriptorType::eStorageBuffer;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags      = vk::ShaderStageFlagBits::eCompute;
    bindings[1].binding         = 1;
    bindings[1].descriptorType  = vk::DescriptorType::eStorageBuffer;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags      = vk::ShaderStageFlagBits::eCompute;

    vk::DescriptorSetLayoutCreateInfo lci{};
    lci.bindingCount = static_cast<std::uint32_t>(bindings.size());
    lci.pBindings    = bindings.data();
    m_desc_layout = m_device.createDescriptorSetLayout(lci);

    vk::DescriptorPoolSize pool_size{};
    pool_size.type            = vk::DescriptorType::eStorageBuffer;
    pool_size.descriptorCount = 2;
    vk::DescriptorPoolCreateInfo pci{};
    pci.maxSets       = 1;
    pci.poolSizeCount = 1;
    pci.pPoolSizes    = &pool_size;
    m_desc_pool = m_device.createDescriptorPool(pci);

    vk::DescriptorSetAllocateInfo dsai{};
    dsai.descriptorPool     = m_desc_pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &m_desc_layout;
    m_desc_set = m_device.allocateDescriptorSets(dsai).front();

    std::array<vk::DescriptorBufferInfo, 2> buf_info{};
    buf_info[0].buffer = m_field_buf;
    buf_info[0].offset = 0;
    buf_info[0].range  = m_field_bytes;
    buf_info[1].buffer = m_rng_buf;
    buf_info[1].offset = 0;
    buf_info[1].range  = m_rng_bytes;

    std::array<vk::WriteDescriptorSet, 2> writes{};
    for (std::size_t i = 0; i < 2; ++i)
    {
        writes[i].dstSet          = m_desc_set;
        writes[i].dstBinding      = static_cast<std::uint32_t>(i);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = vk::DescriptorType::eStorageBuffer;
        writes[i].pBufferInfo     = &buf_info[i];
    }
    m_device.updateDescriptorSets(2, writes.data(), 0, nullptr);
}

void GpuSU2Sweeper::createPipeline(const std::string& shader_dir)
{
    auto bytes = readFile(shader_dir + "/su2_metropolis.comp.spv");
    vk::ShaderModuleCreateInfo smci{};
    smci.codeSize = bytes.size();
    smci.pCode    = reinterpret_cast<const std::uint32_t*>(bytes.data());
    m_shader = m_device.createShaderModule(smci);

    vk::PipelineShaderStageCreateInfo stage{};
    stage.stage  = vk::ShaderStageFlagBits::eCompute;
    stage.module = m_shader;
    stage.pName  = "main";

    vk::PushConstantRange pcr{};
    pcr.stageFlags = vk::ShaderStageFlagBits::eCompute;
    pcr.offset     = 0;
    pcr.size       = sizeof(PushConstants);

    vk::PipelineLayoutCreateInfo pli{};
    pli.setLayoutCount         = 1;
    pli.pSetLayouts            = &m_desc_layout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges    = &pcr;
    m_pipe_layout = m_device.createPipelineLayout(pli);

    vk::ComputePipelineCreateInfo cpci{};
    cpci.stage  = stage;
    cpci.layout = m_pipe_layout;
    auto r = m_device.createComputePipeline(nullptr, cpci);
    if (r.result != vk::Result::eSuccess)
        throw std::runtime_error("createComputePipeline failed");
    m_pipeline = r.value;
}

void GpuSU2Sweeper::allocateCommandBuffer()
{
    vk::CommandBufferAllocateInfo cai{};
    cai.commandPool        = m_cmd_pool;
    cai.level              = vk::CommandBufferLevel::ePrimary;
    cai.commandBufferCount = 1;
    m_cmd = m_device.allocateCommandBuffers(cai).front();

    m_fence = m_device.createFence({}); // unsignaled
}

// -----------------------------------------------------------------------------
// Upload / download / reseed
// -----------------------------------------------------------------------------

void GpuSU2Sweeper::reseedRng(std::uint64_t seed)
{
    auto* dst = static_cast<std::uint32_t*>(m_rng_mapped);
    for (int s = 0; s < m_V; ++s)
    {
        std::uint64_t st = seed ^ (static_cast<std::uint64_t>(s) * 0x9E3779B97F4A7C15ULL + 1);
        for (int k = 0; k < 4; ++k)
        {
            std::uint64_t v = splitmix64(st);
            // Guard against all-zero state — xoshiro128++ is degenerate there.
            dst[s * 4 + k] = static_cast<std::uint32_t>(v ? v : (k + 1));
        }
    }
    m_sweep_count = 0;
}

// -----------------------------------------------------------------------------
// Sweep — record n_sweeps × 2·Dim dispatches and submit.
// -----------------------------------------------------------------------------

void GpuSU2Sweeper::sweep(int n_sweeps, double beta, double step_size)
{
    if (n_sweeps <= 0) return;

    (void)m_device.resetFences(1, &m_fence);

    m_cmd.reset();
    vk::CommandBufferBeginInfo bi{};
    bi.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    m_cmd.begin(bi);

    m_cmd.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
    m_cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                             m_pipe_layout, 0, 1, &m_desc_set, 0, nullptr);

    constexpr std::uint32_t kWorkgroup = 64;
    const std::uint32_t groups = (static_cast<std::uint32_t>(m_V) + kWorkgroup - 1) / kWorkgroup;

    PushConstants pc{};
    pc.Lx        = m_L;
    pc.Ly        = m_L;
    pc.Lz        = (m_Dim == 2) ? 1 : m_L;
    pc.Dim       = m_Dim;
    pc.beta      = static_cast<float>(beta);
    pc.step_size = static_cast<float>(step_size);

    for (int s = 0; s < n_sweeps; ++s)
    {
        for (int parity = 0; parity < 2; ++parity)
            for (int mu = 0; mu < m_Dim; ++mu)
            {
                pc.color_parity = parity;
                pc.color_mu     = mu;
                pc.sweep_count  = m_sweep_count++;
                m_cmd.pushConstants(m_pipe_layout,
                                    vk::ShaderStageFlagBits::eCompute,
                                    0, sizeof(PushConstants), &pc);
                m_cmd.dispatch(groups, 1, 1);

                // Barrier between dispatches: each parity-color reads
                // neighbors' links written by the opposite parity in the
                // previous dispatch.
                vk::MemoryBarrier mb{};
                mb.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
                mb.dstAccessMask = vk::AccessFlagBits::eShaderRead
                                 | vk::AccessFlagBits::eShaderWrite;
                m_cmd.pipelineBarrier(
                    vk::PipelineStageFlagBits::eComputeShader,
                    vk::PipelineStageFlagBits::eComputeShader,
                    {}, 1, &mb, 0, nullptr, 0, nullptr);
            }
    }

    m_cmd.end();

    vk::SubmitInfo si{};
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &m_cmd;
    (void)m_queue.submit(1, &si, m_fence);

    // Block until the GPU is done — host-coherent buffers mean a successful
    // wait is enough for the CPU to read the result without further barriers.
    if (m_device.waitForFences(1, &m_fence, VK_TRUE, UINT64_MAX) != vk::Result::eSuccess)
        throw std::runtime_error("GpuSU2Sweeper::sweep fence wait failed");
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

std::vector<char> GpuSU2Sweeper::readFile(const std::string& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) throw std::runtime_error("failed to open " + path);
    const std::size_t sz = static_cast<std::size_t>(f.tellg());
    std::vector<char> buf(sz);
    f.seekg(0);
    f.read(buf.data(), static_cast<std::streamsize>(sz));
    return buf;
}

std::uint32_t GpuSU2Sweeper::findMemoryType(std::uint32_t type_bits,
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
