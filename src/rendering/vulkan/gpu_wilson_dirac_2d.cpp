#include "gpu_wilson_dirac_2d.hpp"

#include "../../rng/rng.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace lqft::vis
{

GpuWilsonDirac2D::GpuWilsonDirac2D(vk::PhysicalDevice  phys,
                                   vk::Device          device,
                                   vk::Queue           queue,
                                   std::uint32_t       queue_family,
                                   vk::CommandPool     cmd_pool,
                                   int                 L,
                                   const std::string&  shader_dir)
    : m_phys(phys), m_device(device), m_queue(queue)
    , m_queue_family(queue_family), m_cmd_pool(cmd_pool)
    , m_L(L), m_V(L * L)
{
    createBuffers();
    createDescriptors();
    createPipeline(shader_dir);
    allocateCommandBuffer();
}

GpuWilsonDirac2D::~GpuWilsonDirac2D()
{
    if (!m_device) return;
    if (m_fence)         m_device.destroyFence(m_fence);
    if (m_cmd)           m_device.freeCommandBuffers(m_cmd_pool, m_cmd);
    if (m_pipeline)      m_device.destroyPipeline(m_pipeline);
    if (m_pipe_layout)   m_device.destroyPipelineLayout(m_pipe_layout);
    if (m_shader)        m_device.destroyShaderModule(m_shader);
    if (m_desc_pool)     m_device.destroyDescriptorPool(m_desc_pool);
    if (m_desc_layout)   m_device.destroyDescriptorSetLayout(m_desc_layout);

    if (m_psi_out_mapped) m_device.unmapMemory(m_psi_out_mem);
    if (m_psi_out_buf)    m_device.destroyBuffer(m_psi_out_buf);
    if (m_psi_out_mem)    m_device.freeMemory(m_psi_out_mem);

    if (m_psi_in_mapped) m_device.unmapMemory(m_psi_in_mem);
    if (m_psi_in_buf)    m_device.destroyBuffer(m_psi_in_buf);
    if (m_psi_in_mem)    m_device.freeMemory(m_psi_in_mem);

    if (m_gauge_mapped)  m_device.unmapMemory(m_gauge_mem);
    if (m_gauge_buf)     m_device.destroyBuffer(m_gauge_buf);
    if (m_gauge_mem)     m_device.freeMemory(m_gauge_mem);
}

void GpuWilsonDirac2D::createBuffers()
{
    m_gauge_bytes = static_cast<vk::DeviceSize>(m_V) * 2 * sizeof(float);
    m_psi_bytes   = static_cast<vk::DeviceSize>(m_V) * 2 * 2 * sizeof(float);
    //                                                    ^   ^
    //                                                    |   complex = 2 floats
    //                                                    Nc = 2 components

    auto alloc = [&](vk::DeviceSize bytes,
                     vk::Buffer&  buf,
                     vk::DeviceMemory& mem,
                     void*& mapped)
    {
        vk::BufferCreateInfo bci{};
        bci.size        = bytes;
        bci.usage       = vk::BufferUsageFlagBits::eStorageBuffer;
        bci.sharingMode = vk::SharingMode::eExclusive;
        buf = m_device.createBuffer(bci);

        auto req = m_device.getBufferMemoryRequirements(buf);
        vk::MemoryAllocateInfo mai{};
        mai.allocationSize  = req.size;
        mai.memoryTypeIndex = findMemoryType(
            req.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible
          | vk::MemoryPropertyFlagBits::eHostCoherent);
        mem = m_device.allocateMemory(mai);
        m_device.bindBufferMemory(buf, mem, 0);
        mapped = m_device.mapMemory(mem, 0, bytes);
    };

    alloc(m_gauge_bytes, m_gauge_buf,   m_gauge_mem,   m_gauge_mapped);
    alloc(m_psi_bytes,   m_psi_in_buf,  m_psi_in_mem,  m_psi_in_mapped);
    alloc(m_psi_bytes,   m_psi_out_buf, m_psi_out_mem, m_psi_out_mapped);
}

void GpuWilsonDirac2D::createDescriptors()
{
    std::array<vk::DescriptorSetLayoutBinding, 3> bindings{};
    for (std::size_t i = 0; i < 3; ++i)
    {
        bindings[i].binding         = static_cast<std::uint32_t>(i);
        bindings[i].descriptorType  = vk::DescriptorType::eStorageBuffer;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = vk::ShaderStageFlagBits::eCompute;
    }
    vk::DescriptorSetLayoutCreateInfo lci{};
    lci.bindingCount = 3;
    lci.pBindings    = bindings.data();
    m_desc_layout = m_device.createDescriptorSetLayout(lci);

    vk::DescriptorPoolSize ps{};
    ps.type            = vk::DescriptorType::eStorageBuffer;
    ps.descriptorCount = 3;
    vk::DescriptorPoolCreateInfo pci{};
    pci.maxSets       = 1;
    pci.poolSizeCount = 1;
    pci.pPoolSizes    = &ps;
    m_desc_pool = m_device.createDescriptorPool(pci);

    vk::DescriptorSetAllocateInfo dsai{};
    dsai.descriptorPool     = m_desc_pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &m_desc_layout;
    m_desc_set = m_device.allocateDescriptorSets(dsai).front();

    std::array<vk::DescriptorBufferInfo, 3> info{};
    info[0].buffer = m_gauge_buf;   info[0].range = m_gauge_bytes;
    info[1].buffer = m_psi_in_buf;  info[1].range = m_psi_bytes;
    info[2].buffer = m_psi_out_buf; info[2].range = m_psi_bytes;

    std::array<vk::WriteDescriptorSet, 3> writes{};
    for (std::size_t i = 0; i < 3; ++i)
    {
        writes[i].dstSet          = m_desc_set;
        writes[i].dstBinding      = static_cast<std::uint32_t>(i);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = vk::DescriptorType::eStorageBuffer;
        writes[i].pBufferInfo     = &info[i];
    }
    m_device.updateDescriptorSets(3, writes.data(), 0, nullptr);
}

void GpuWilsonDirac2D::createPipeline(const std::string& shader_dir)
{
    auto bytes = readFile(shader_dir + "/wilson_dirac_2d.comp.spv");
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

void GpuWilsonDirac2D::allocateCommandBuffer()
{
    vk::CommandBufferAllocateInfo cai{};
    cai.commandPool        = m_cmd_pool;
    cai.level              = vk::CommandBufferLevel::ePrimary;
    cai.commandBufferCount = 1;
    m_cmd   = m_device.allocateCommandBuffers(cai).front();
    m_fence = m_device.createFence({});
}

void GpuWilsonDirac2D::uploadGauge(const LinkField<double, 2>& U)
{
    auto* dst = static_cast<float*>(m_gauge_mapped);
    for (int s = 0; s < m_V; ++s)
        for (int mu = 0; mu < 2; ++mu)
            dst[s * 2 + mu] = static_cast<float>(U(s, mu));
}

void GpuWilsonDirac2D::uploadInputSpinor(const SpinorField<2, 2>& psi)
{
    auto* dst = static_cast<float*>(m_psi_in_mapped);
    for (int s = 0; s < m_V; ++s)
        for (int c = 0; c < 2; ++c)
        {
            const auto z = psi(s, c);
            dst[(s * 2 + c) * 2 + 0] = static_cast<float>(z.real());
            dst[(s * 2 + c) * 2 + 1] = static_cast<float>(z.imag());
        }
}

void GpuWilsonDirac2D::downloadOutputSpinor(SpinorField<2, 2>& psi) const
{
    const auto* src = static_cast<const float*>(m_psi_out_mapped);
    for (int s = 0; s < m_V; ++s)
        for (int c = 0; c < 2; ++c)
        {
            const double re = src[(s * 2 + c) * 2 + 0];
            const double im = src[(s * 2 + c) * 2 + 1];
            psi(s, c) = std::complex<double>(re, im);
        }
}

void GpuWilsonDirac2D::apply(double mass, bool dagger)
{
    (void)m_device.resetFences(1, &m_fence);
    m_cmd.reset();

    vk::CommandBufferBeginInfo bi{};
    bi.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    m_cmd.begin(bi);

    m_cmd.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
    m_cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                             m_pipe_layout, 0, 1, &m_desc_set, 0, nullptr);

    PushConstants pc{};
    pc.Lx     = m_L;
    pc.Ly     = m_L;
    pc.mass   = static_cast<float>(mass);
    pc.dagger = dagger ? 1 : 0;
    m_cmd.pushConstants(m_pipe_layout, vk::ShaderStageFlagBits::eCompute,
                        0, sizeof(PushConstants), &pc);

    constexpr std::uint32_t kWG = 64;
    const std::uint32_t groups = (static_cast<std::uint32_t>(m_V) + kWG - 1) / kWG;
    m_cmd.dispatch(groups, 1, 1);

    m_cmd.end();

    vk::SubmitInfo si{};
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &m_cmd;
    (void)m_queue.submit(1, &si, m_fence);

    if (m_device.waitForFences(1, &m_fence, VK_TRUE, UINT64_MAX) != vk::Result::eSuccess)
        throw std::runtime_error("GpuWilsonDirac2D::apply fence wait failed");
}

void GpuWilsonDirac2D::applyFull(const LinkField<double, 2>& U,
                                 const SpinorField<2, 2>&    in,
                                 double                       mass,
                                 bool                         dagger,
                                 SpinorField<2, 2>&           out)
{
    uploadGauge(U);
    uploadInputSpinor(in);
    apply(mass, dagger);
    downloadOutputSpinor(out);
}

double GpuWilsonDirac2D::selfValidate(double mass, std::uint64_t seed)
{
    Lattice<2> lattice = Lattice<2>::cube(m_L);
    LinkField<double, 2> U(lattice);
    SpinorField<2, 2>    psi(lattice);
    SpinorField<2, 2>    y_cpu(lattice);
    SpinorField<2, 2>    y_gpu(lattice);

    Rng rng(seed);
    for (int s = 0; s < m_V; ++s)
    {
        for (int mu = 0; mu < 2; ++mu)
            U(s, mu) = rng.uniform() * 2.0 * 3.14159265358979 - 3.14159265358979;
        for (int c = 0; c < 2; ++c)
            psi(s, c) = std::complex<double>(rng.normal(), rng.normal());
    }

    dirac::WilsonDirac2D D(lattice, mass);
    D.apply(U, psi, y_cpu);

    applyFull(U, psi, mass, /*dagger*/false, y_gpu);

    double max_err = 0.0;
    for (int s = 0; s < m_V; ++s)
        for (int c = 0; c < 2; ++c)
        {
            const auto d = y_cpu(s, c) - y_gpu(s, c);
            max_err = std::max(max_err, std::abs(d));
        }
    return max_err;
}

std::vector<char> GpuWilsonDirac2D::readFile(const std::string& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) throw std::runtime_error("failed to open " + path);
    const std::size_t sz = static_cast<std::size_t>(f.tellg());
    std::vector<char> buf(sz);
    f.seekg(0);
    f.read(buf.data(), static_cast<std::streamsize>(sz));
    return buf;
}

std::uint32_t GpuWilsonDirac2D::findMemoryType(std::uint32_t type_bits,
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
