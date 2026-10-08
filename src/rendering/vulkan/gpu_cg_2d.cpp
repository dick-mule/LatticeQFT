#include "gpu_cg_2d.hpp"

#include "../../math/dirac_2d.hpp"
#include "../../rng/rng.hpp"
#include "../../solvers/conjugate_gradient.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace lqft::vis
{

namespace
{
constexpr std::uint32_t kWG = 64;

struct AxpbyPushConstants
{
    int   n_components;
    float a;
    float b;
};

struct InnerPushConstants
{
    int n_components;
};

struct WdPushConstants
{
    int   Lx;
    int   Ly;
    float mass;
    int   dagger;
};
} // namespace

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

GpuCG2D::GpuCG2D(vk::PhysicalDevice phys,
                 vk::Device         device,
                 vk::Queue          queue,
                 std::uint32_t      queue_family,
                 vk::CommandPool    cmd_pool,
                 int                L,
                 const std::string& shader_dir)
    : m_phys(phys), m_device(device), m_queue(queue)
    , m_queue_family(queue_family), m_cmd_pool(cmd_pool)
    , m_L(L), m_V(L * L)
    , m_n_components(L * L * 2)
    , m_n_workgroups(
        (static_cast<std::uint32_t>(m_n_components) + kWG - 1) / kWG)
{
    createBuffers();
    createPipelinesAndDescriptors(shader_dir);
    allocateCommandBuffer();
}

GpuCG2D::~GpuCG2D()
{
    if (!m_device) return;
    if (m_fence)  m_device.destroyFence(m_fence);
    if (m_cmd)    m_device.freeCommandBuffers(m_cmd_pool, m_cmd);

    if (m_wd_pipeline)    m_device.destroyPipeline(m_wd_pipeline);
    if (m_axpby_pipeline) m_device.destroyPipeline(m_axpby_pipeline);
    if (m_inner_pipeline) m_device.destroyPipeline(m_inner_pipeline);

    if (m_wd_pipe_layout)    m_device.destroyPipelineLayout(m_wd_pipe_layout);
    if (m_axpby_pipe_layout) m_device.destroyPipelineLayout(m_axpby_pipe_layout);
    if (m_inner_pipe_layout) m_device.destroyPipelineLayout(m_inner_pipe_layout);

    if (m_wd_shader)    m_device.destroyShaderModule(m_wd_shader);
    if (m_axpby_shader) m_device.destroyShaderModule(m_axpby_shader);
    if (m_inner_shader) m_device.destroyShaderModule(m_inner_shader);

    if (m_desc_pool)     m_device.destroyDescriptorPool(m_desc_pool);
    if (m_wd_layout)     m_device.destroyDescriptorSetLayout(m_wd_layout);
    if (m_axpby_layout)  m_device.destroyDescriptorSetLayout(m_axpby_layout);
    if (m_inner_layout)  m_device.destroyDescriptorSetLayout(m_inner_layout);

    auto freeBuf = [&](Buffer& b)
    {
        if (b.mapped) m_device.unmapMemory(b.mem);
        if (b.buf)    m_device.destroyBuffer(b.buf);
        if (b.mem)    m_device.freeMemory(b.mem);
    };
    freeBuf(m_partials);
    freeBuf(m_tmp);
    freeBuf(m_Ap);
    freeBuf(m_p);
    freeBuf(m_r);
    freeBuf(m_x);
    freeBuf(m_b);
    freeBuf(m_gauge);
}

void GpuCG2D::createBuffers()
{
    const vk::DeviceSize gauge_bytes
        = static_cast<vk::DeviceSize>(m_V) * 2 * sizeof(float);
    const vk::DeviceSize spinor_bytes
        = static_cast<vk::DeviceSize>(m_n_components) * 2 * sizeof(float);
    //  ^ vec2 (complex) per component
    const vk::DeviceSize partials_bytes
        = static_cast<vk::DeviceSize>(m_n_workgroups) * sizeof(float);

    auto alloc = [&](vk::DeviceSize bytes, Buffer& b)
    {
        b.bytes = bytes;
        vk::BufferCreateInfo bci{};
        bci.size        = bytes;
        bci.usage       = vk::BufferUsageFlagBits::eStorageBuffer;
        bci.sharingMode = vk::SharingMode::eExclusive;
        b.buf = m_device.createBuffer(bci);

        auto req = m_device.getBufferMemoryRequirements(b.buf);
        vk::MemoryAllocateInfo mai{};
        mai.allocationSize  = req.size;
        mai.memoryTypeIndex = findMemoryType(
            req.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible
          | vk::MemoryPropertyFlagBits::eHostCoherent);
        b.mem = m_device.allocateMemory(mai);
        m_device.bindBufferMemory(b.buf, b.mem, 0);
        b.mapped = m_device.mapMemory(b.mem, 0, bytes);
    };

    alloc(gauge_bytes,    m_gauge);
    alloc(spinor_bytes,   m_b);
    alloc(spinor_bytes,   m_x);
    alloc(spinor_bytes,   m_r);
    alloc(spinor_bytes,   m_p);
    alloc(spinor_bytes,   m_Ap);
    alloc(spinor_bytes,   m_tmp);
    alloc(partials_bytes, m_partials);
}

// Helper: build a descriptor set layout with `n` storage-buffer bindings.
static vk::DescriptorSetLayout makeStorageLayout(vk::Device device, int n)
{
    std::array<vk::DescriptorSetLayoutBinding, 4> bindings{};
    for (int i = 0; i < n; ++i)
    {
        bindings[i].binding         = static_cast<std::uint32_t>(i);
        bindings[i].descriptorType  = vk::DescriptorType::eStorageBuffer;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = vk::ShaderStageFlagBits::eCompute;
    }
    vk::DescriptorSetLayoutCreateInfo lci{};
    lci.bindingCount = static_cast<std::uint32_t>(n);
    lci.pBindings    = bindings.data();
    return device.createDescriptorSetLayout(lci);
}

void GpuCG2D::createPipelinesAndDescriptors(const std::string& shader_dir)
{
    // ----- descriptor set layouts -----
    m_wd_layout    = makeStorageLayout(m_device, 3);
    m_axpby_layout = makeStorageLayout(m_device, 2);
    m_inner_layout = makeStorageLayout(m_device, 3);

    // ----- descriptor pool sized for 7 sets total -----
    vk::DescriptorPoolSize ps{};
    ps.type            = vk::DescriptorType::eStorageBuffer;
    ps.descriptorCount = 2 * 3 + 3 * 2 + 2 * 3; // 6 + 6 + 6 = 18 bindings
    vk::DescriptorPoolCreateInfo pci{};
    pci.maxSets       = 7;
    pci.poolSizeCount = 1;
    pci.pPoolSizes    = &ps;
    m_desc_pool = m_device.createDescriptorPool(pci);

    // Allocate one set per (pipeline, buffer-binding) pair.
    auto allocSet = [&](vk::DescriptorSetLayout layout)
    {
        vk::DescriptorSetAllocateInfo dsai{};
        dsai.descriptorPool     = m_desc_pool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts        = &layout;
        return m_device.allocateDescriptorSets(dsai).front();
    };
    m_set_wd_Dp     = allocSet(m_wd_layout);
    m_set_wd_DtAp   = allocSet(m_wd_layout);
    m_set_axpby_x   = allocSet(m_axpby_layout);
    m_set_axpby_r   = allocSet(m_axpby_layout);
    m_set_axpby_p   = allocSet(m_axpby_layout);
    m_set_inner_pap = allocSet(m_inner_layout);
    m_set_inner_rr  = allocSet(m_inner_layout);

    // ----- bind buffers to descriptor sets -----
    auto writeStorage =
        [&](vk::DescriptorSet set, std::uint32_t binding, const Buffer& buf)
    {
        vk::DescriptorBufferInfo dbi{};
        dbi.buffer = buf.buf;
        dbi.offset = 0;
        dbi.range  = buf.bytes;
        vk::WriteDescriptorSet w{};
        w.dstSet          = set;
        w.dstBinding      = binding;
        w.descriptorCount = 1;
        w.descriptorType  = vk::DescriptorType::eStorageBuffer;
        w.pBufferInfo     = &dbi;
        m_device.updateDescriptorSets(1, &w, 0, nullptr);
    };

    // Wilson-Dirac sets: (gauge, in, out)
    writeStorage(m_set_wd_Dp,   0, m_gauge);
    writeStorage(m_set_wd_Dp,   1, m_p);
    writeStorage(m_set_wd_Dp,   2, m_tmp);
    writeStorage(m_set_wd_DtAp, 0, m_gauge);
    writeStorage(m_set_wd_DtAp, 1, m_tmp);
    writeStorage(m_set_wd_DtAp, 2, m_Ap);

    // axpby sets: (x, y)  →  y = a x + b y
    writeStorage(m_set_axpby_x, 0, m_p);   // x_in = p
    writeStorage(m_set_axpby_x, 1, m_x);   // y    = x_solution
    writeStorage(m_set_axpby_r, 0, m_Ap);  // x_in = Ap
    writeStorage(m_set_axpby_r, 1, m_r);   // y    = r
    writeStorage(m_set_axpby_p, 0, m_r);   // x_in = r
    writeStorage(m_set_axpby_p, 1, m_p);   // y    = p

    // inner sets: (a, b, partials)
    writeStorage(m_set_inner_pap, 0, m_p);
    writeStorage(m_set_inner_pap, 1, m_Ap);
    writeStorage(m_set_inner_pap, 2, m_partials);
    writeStorage(m_set_inner_rr,  0, m_r);
    writeStorage(m_set_inner_rr,  1, m_r);
    writeStorage(m_set_inner_rr,  2, m_partials);

    // ----- shader modules -----
    auto loadModule = [&](const std::string& file)
    {
        auto bytes = readFile(shader_dir + "/" + file);
        vk::ShaderModuleCreateInfo smci{};
        smci.codeSize = bytes.size();
        smci.pCode    = reinterpret_cast<const std::uint32_t*>(bytes.data());
        return m_device.createShaderModule(smci);
    };
    m_wd_shader    = loadModule("wilson_dirac_2d.comp.spv");
    m_axpby_shader = loadModule("spinor_axpby_real.comp.spv");
    m_inner_shader = loadModule("spinor_inner_real.comp.spv");

    auto makePipeline =
        [&](vk::ShaderModule shader,
            vk::DescriptorSetLayout layout,
            std::uint32_t pc_size,
            vk::PipelineLayout& out_pl,
            vk::Pipeline& out_pipe)
    {
        vk::PipelineShaderStageCreateInfo stage{};
        stage.stage  = vk::ShaderStageFlagBits::eCompute;
        stage.module = shader;
        stage.pName  = "main";

        vk::PushConstantRange pcr{};
        pcr.stageFlags = vk::ShaderStageFlagBits::eCompute;
        pcr.offset     = 0;
        pcr.size       = pc_size;

        vk::PipelineLayoutCreateInfo pli{};
        pli.setLayoutCount         = 1;
        pli.pSetLayouts            = &layout;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges    = &pcr;
        out_pl = m_device.createPipelineLayout(pli);

        vk::ComputePipelineCreateInfo cpci{};
        cpci.stage  = stage;
        cpci.layout = out_pl;
        auto r = m_device.createComputePipeline(nullptr, cpci);
        if (r.result != vk::Result::eSuccess)
            throw std::runtime_error("createComputePipeline failed");
        out_pipe = r.value;
    };

    makePipeline(m_wd_shader,    m_wd_layout,
                 sizeof(WdPushConstants),    m_wd_pipe_layout,    m_wd_pipeline);
    makePipeline(m_axpby_shader, m_axpby_layout,
                 sizeof(AxpbyPushConstants), m_axpby_pipe_layout, m_axpby_pipeline);
    makePipeline(m_inner_shader, m_inner_layout,
                 sizeof(InnerPushConstants), m_inner_pipe_layout, m_inner_pipeline);
}

void GpuCG2D::allocateCommandBuffer()
{
    vk::CommandBufferAllocateInfo cai{};
    cai.commandPool        = m_cmd_pool;
    cai.level              = vk::CommandBufferLevel::ePrimary;
    cai.commandBufferCount = 1;
    m_cmd   = m_device.allocateCommandBuffers(cai).front();
    m_fence = m_device.createFence({});
}

// ---------------------------------------------------------------------------
// IO
// ---------------------------------------------------------------------------

void GpuCG2D::uploadGauge(const LinkField<double, 2>& U)
{
    auto* dst = static_cast<float*>(m_gauge.mapped);
    for (int s = 0; s < m_V; ++s)
        for (int mu = 0; mu < 2; ++mu)
            dst[s * 2 + mu] = static_cast<float>(U(s, mu));
}

void GpuCG2D::uploadSpinor(vk::DeviceMemory /*mem*/, void* mapped,
                           const SpinorField<2, 2>& s)
{
    auto* dst = static_cast<float*>(mapped);
    for (int i = 0; i < m_V; ++i)
        for (int c = 0; c < 2; ++c)
        {
            const auto z = s(i, c);
            dst[(i * 2 + c) * 2 + 0] = static_cast<float>(z.real());
            dst[(i * 2 + c) * 2 + 1] = static_cast<float>(z.imag());
        }
}

void GpuCG2D::downloadSpinor(void* mapped, SpinorField<2, 2>& s) const
{
    const auto* src = static_cast<const float*>(mapped);
    for (int i = 0; i < m_V; ++i)
        for (int c = 0; c < 2; ++c)
        {
            const double re = src[(i * 2 + c) * 2 + 0];
            const double im = src[(i * 2 + c) * 2 + 1];
            s(i, c) = std::complex<double>(re, im);
        }
}

// ---------------------------------------------------------------------------
// Dispatch helpers
// ---------------------------------------------------------------------------

double GpuCG2D::sumPartials() const
{
    const auto*  src = static_cast<const float*>(m_partials.mapped);
    double sum = 0.0;
    for (std::uint32_t i = 0; i < m_n_workgroups; ++i)
        sum += static_cast<double>(src[i]);
    return sum;
}

// Record + submit one command buffer; wait on the fence.
namespace
{
inline void recordBarrier(vk::CommandBuffer cmd)
{
    vk::MemoryBarrier mb{};
    mb.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
    mb.dstAccessMask = vk::AccessFlagBits::eShaderRead
                     | vk::AccessFlagBits::eShaderWrite;
    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        {}, 1, &mb, 0, nullptr, 0, nullptr);
}
} // namespace

double GpuCG2D::dispatchPhaseAp(float mass)
{
    (void)m_device.resetFences(1, &m_fence);
    m_cmd.reset();
    vk::CommandBufferBeginInfo bi{};
    bi.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    m_cmd.begin(bi);

    const std::uint32_t spinor_groups
        = (static_cast<std::uint32_t>(m_V) + kWG - 1) / kWG;

    // ---- D · p → tmp ----
    m_cmd.bindPipeline(vk::PipelineBindPoint::eCompute, m_wd_pipeline);
    m_cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                             m_wd_pipe_layout, 0, 1, &m_set_wd_Dp, 0, nullptr);
    {
        WdPushConstants pc{};
        pc.Lx = m_L; pc.Ly = m_L; pc.mass = mass; pc.dagger = 0;
        m_cmd.pushConstants(m_wd_pipe_layout, vk::ShaderStageFlagBits::eCompute,
                            0, sizeof(pc), &pc);
        m_cmd.dispatch(spinor_groups, 1, 1);
    }
    recordBarrier(m_cmd);

    // ---- D† · tmp → Ap ----
    m_cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                             m_wd_pipe_layout, 0, 1, &m_set_wd_DtAp, 0, nullptr);
    {
        WdPushConstants pc{};
        pc.Lx = m_L; pc.Ly = m_L; pc.mass = mass; pc.dagger = 1;
        m_cmd.pushConstants(m_wd_pipe_layout, vk::ShaderStageFlagBits::eCompute,
                            0, sizeof(pc), &pc);
        m_cmd.dispatch(spinor_groups, 1, 1);
    }
    recordBarrier(m_cmd);

    // ---- inner ⟨p, Ap⟩ → partials ----
    m_cmd.bindPipeline(vk::PipelineBindPoint::eCompute, m_inner_pipeline);
    m_cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                             m_inner_pipe_layout, 0, 1, &m_set_inner_pap, 0, nullptr);
    {
        InnerPushConstants pc{};
        pc.n_components = m_n_components;
        m_cmd.pushConstants(m_inner_pipe_layout, vk::ShaderStageFlagBits::eCompute,
                            0, sizeof(pc), &pc);
        m_cmd.dispatch(m_n_workgroups, 1, 1);
    }
    m_cmd.end();

    vk::SubmitInfo si{};
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &m_cmd;
    (void)m_queue.submit(1, &si, m_fence);
    if (m_device.waitForFences(1, &m_fence, VK_TRUE, UINT64_MAX) != vk::Result::eSuccess)
        throw std::runtime_error("GpuCG2D phase Ap fence wait failed");

    return sumPartials();
}

double GpuCG2D::dispatchPhaseUpdate(float alpha)
{
    (void)m_device.resetFences(1, &m_fence);
    m_cmd.reset();
    vk::CommandBufferBeginInfo bi{};
    bi.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    m_cmd.begin(bi);

    m_cmd.bindPipeline(vk::PipelineBindPoint::eCompute, m_axpby_pipeline);

    // x  += α · p          (y = x_solution, x_in = p, a = α, b = 1)
    m_cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                             m_axpby_pipe_layout, 0, 1, &m_set_axpby_x, 0, nullptr);
    {
        AxpbyPushConstants pc{};
        pc.n_components = m_n_components;
        pc.a = alpha;
        pc.b = 1.0f;
        m_cmd.pushConstants(m_axpby_pipe_layout, vk::ShaderStageFlagBits::eCompute,
                            0, sizeof(pc), &pc);
        m_cmd.dispatch(m_n_workgroups, 1, 1);
    }
    recordBarrier(m_cmd);

    // r  −= α · Ap         (y = r, x_in = Ap, a = −α, b = 1)
    m_cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                             m_axpby_pipe_layout, 0, 1, &m_set_axpby_r, 0, nullptr);
    {
        AxpbyPushConstants pc{};
        pc.n_components = m_n_components;
        pc.a = -alpha;
        pc.b = 1.0f;
        m_cmd.pushConstants(m_axpby_pipe_layout, vk::ShaderStageFlagBits::eCompute,
                            0, sizeof(pc), &pc);
        m_cmd.dispatch(m_n_workgroups, 1, 1);
    }
    recordBarrier(m_cmd);

    // ⟨r, r⟩ → partials
    m_cmd.bindPipeline(vk::PipelineBindPoint::eCompute, m_inner_pipeline);
    m_cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                             m_inner_pipe_layout, 0, 1, &m_set_inner_rr, 0, nullptr);
    {
        InnerPushConstants pc{};
        pc.n_components = m_n_components;
        m_cmd.pushConstants(m_inner_pipe_layout, vk::ShaderStageFlagBits::eCompute,
                            0, sizeof(pc), &pc);
        m_cmd.dispatch(m_n_workgroups, 1, 1);
    }
    m_cmd.end();

    vk::SubmitInfo si{};
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &m_cmd;
    (void)m_queue.submit(1, &si, m_fence);
    if (m_device.waitForFences(1, &m_fence, VK_TRUE, UINT64_MAX) != vk::Result::eSuccess)
        throw std::runtime_error("GpuCG2D phase Update fence wait failed");

    return sumPartials();
}

void GpuCG2D::dispatchPhaseRecurse(float beta)
{
    (void)m_device.resetFences(1, &m_fence);
    m_cmd.reset();
    vk::CommandBufferBeginInfo bi{};
    bi.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    m_cmd.begin(bi);

    m_cmd.bindPipeline(vk::PipelineBindPoint::eCompute, m_axpby_pipeline);

    // p  = r + β p   (y = p, x_in = r, a = 1, b = β)
    m_cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                             m_axpby_pipe_layout, 0, 1, &m_set_axpby_p, 0, nullptr);
    {
        AxpbyPushConstants pc{};
        pc.n_components = m_n_components;
        pc.a = 1.0f;
        pc.b = beta;
        m_cmd.pushConstants(m_axpby_pipe_layout, vk::ShaderStageFlagBits::eCompute,
                            0, sizeof(pc), &pc);
        m_cmd.dispatch(m_n_workgroups, 1, 1);
    }
    m_cmd.end();

    vk::SubmitInfo si{};
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &m_cmd;
    (void)m_queue.submit(1, &si, m_fence);
    if (m_device.waitForFences(1, &m_fence, VK_TRUE, UINT64_MAX) != vk::Result::eSuccess)
        throw std::runtime_error("GpuCG2D phase Recurse fence wait failed");
}

// ---------------------------------------------------------------------------
// Top-level solve
// ---------------------------------------------------------------------------

GpuCG2D::Result GpuCG2D::solve(const LinkField<double, 2>& U,
                               const SpinorField<2, 2>&    b,
                               double                       mass,
                               double                       tol,
                               int                          max_iters,
                               SpinorField<2, 2>&           x_out)
{
    Result res{};

    uploadGauge(U);
    uploadSpinor(m_b.mem, m_b.mapped, b);

    // x = 0, r = b, p = r — all done by direct host writes since buffers are
    // host-coherent. No GPU dispatch needed for initialization.
    std::memset(m_x.mapped, 0, static_cast<std::size_t>(m_x.bytes));
    std::memcpy(m_r.mapped, m_b.mapped, static_cast<std::size_t>(m_b.bytes));
    std::memcpy(m_p.mapped, m_r.mapped, static_cast<std::size_t>(m_r.bytes));

    // Initial r² = ⟨b, b⟩ — sum on host.
    double r_norm2 = 0.0;
    {
        const auto* src = static_cast<const float*>(m_b.mapped);
        for (int i = 0; i < m_n_components * 2; ++i)
            r_norm2 += static_cast<double>(src[i]) * static_cast<double>(src[i]);
    }
    const double tol2 = tol * tol * r_norm2;
    if (r_norm2 < 1e-30)
    {
        res.converged  = true;
        res.final_res2 = r_norm2;
        downloadSpinor(m_x.mapped, x_out);
        return res;
    }

    const float mass_f = static_cast<float>(mass);

    for (int k = 0; k < max_iters; ++k)
    {
        // ---- phase 1: Ap = D†D p ;  ⟨p, Ap⟩ ----
        const double pAp = dispatchPhaseAp(mass_f);
        if (pAp <= 0.0)
        {
            res.iterations = k;
            res.final_res2 = r_norm2;
            res.converged  = false;
            break;
        }
        const double alpha = r_norm2 / pAp;

        // ---- phase 2: x += αp ; r −= αAp ; ⟨r, r⟩ ----
        const double r_norm2_new
            = dispatchPhaseUpdate(static_cast<float>(alpha));

        if (r_norm2_new < tol2)
        {
            res.iterations = k + 1;
            res.final_res2 = r_norm2_new;
            res.converged  = true;
            break;
        }

        // ---- phase 3: p = r + β p ----
        const double beta = r_norm2_new / r_norm2;
        dispatchPhaseRecurse(static_cast<float>(beta));

        r_norm2 = r_norm2_new;
        res.iterations = k + 1;
    }
    if (!res.converged)
        res.final_res2 = r_norm2;

    downloadSpinor(m_x.mapped, x_out);
    return res;
}

// ---------------------------------------------------------------------------
// Self-validation
// ---------------------------------------------------------------------------

double GpuCG2D::selfValidateCG(double mass, double tol, int max_iters,
                               std::uint64_t seed)
{
    Lattice<2> lattice = Lattice<2>::cube(m_L);
    LinkField<double, 2> U(lattice);
    SpinorField<2, 2>    b(lattice);
    SpinorField<2, 2>    x_cpu(lattice);
    SpinorField<2, 2>    x_gpu(lattice);
    SpinorField<2, 2>    r_tmp(lattice), p_tmp(lattice), Ap_tmp(lattice), tmp_tmp(lattice);

    Rng rng(seed);
    for (int s = 0; s < m_V; ++s)
    {
        for (int mu = 0; mu < 2; ++mu)
            U(s, mu) = rng.uniform() * 2.0 * 3.14159265358979 - 3.14159265358979;
        for (int c = 0; c < 2; ++c)
            b(s, c) = std::complex<double>(rng.normal(), rng.normal());
    }

    // ---- CPU CG on D†D ----
    dirac::WilsonDirac2D D(lattice, mass);
    auto apply_DagD = [&](const SpinorField<2, 2>& in, SpinorField<2, 2>& out)
    {
        D.applyDagD(U, in, out, tmp_tmp);
    };
    x_cpu.zero();
    auto cpu_res = solvers::conjugateGradient<2, 2>(
        apply_DagD, b, x_cpu, tol, max_iters, r_tmp, p_tmp, Ap_tmp);

    // ---- GPU CG on D†D ----
    x_gpu.zero();
    Result gpu_res = solve(U, b, mass, tol, max_iters, x_gpu);
    (void)cpu_res; (void)gpu_res;

    // Relative L² error: ‖x_cpu − x_gpu‖ / ‖x_cpu‖.
    double num = 0.0, den = 0.0;
    for (int s = 0; s < m_V; ++s)
        for (int c = 0; c < 2; ++c)
        {
            const auto d = x_cpu(s, c) - x_gpu(s, c);
            num += std::norm(d);
            den += std::norm(x_cpu(s, c));
        }
    return (den > 0) ? std::sqrt(num / den) : std::sqrt(num);
}

// ---------------------------------------------------------------------------
// File / memory helpers
// ---------------------------------------------------------------------------

std::vector<char> GpuCG2D::readFile(const std::string& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) throw std::runtime_error("failed to open " + path);
    const std::size_t sz = static_cast<std::size_t>(f.tellg());
    std::vector<char> buf(sz);
    f.seekg(0);
    f.read(buf.data(), static_cast<std::streamsize>(sz));
    return buf;
}

std::uint32_t GpuCG2D::findMemoryType(std::uint32_t type_bits,
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
