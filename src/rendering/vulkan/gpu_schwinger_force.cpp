#include "gpu_schwinger_force.hpp"

#include "gpu_cg_2d.hpp"

#include "../../math/dirac_2d.hpp"
#include "../../models/u1.hpp"
#include "../../monte_carlo/schwinger_hmc.hpp"
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

namespace
{
constexpr std::uint32_t kWG = 64;

vk::DescriptorSetLayout makeStorageLayout(vk::Device device, int n)
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
} // namespace

GpuSchwingerForce::GpuSchwingerForce(vk::PhysicalDevice phys,
                                     vk::Device         device,
                                     vk::Queue          queue,
                                     std::uint32_t      queue_family,
                                     vk::CommandPool    cmd_pool,
                                     int                L,
                                     const std::string& shader_dir)
    : m_phys(phys), m_device(device), m_queue(queue)
    , m_queue_family(queue_family), m_cmd_pool(cmd_pool)
    , m_L(L), m_V(L * L)
{
    m_cg = std::make_unique<GpuCG2D>(phys, device, queue, queue_family,
                                     cmd_pool, L, shader_dir);
    createBuffers();
    createPipelinesAndDescriptors(shader_dir);
    allocateCommandBuffer();
}

GpuSchwingerForce::~GpuSchwingerForce()
{
    if (!m_device) return;
    if (m_fence) m_device.destroyFence(m_fence);
    if (m_cmd)   m_device.freeCommandBuffers(m_cmd_pool, m_cmd);

    if (m_wd_pipe)     m_device.destroyPipeline(m_wd_pipe);
    if (m_gforce_pipe) m_device.destroyPipeline(m_gforce_pipe);
    if (m_fforce_pipe) m_device.destroyPipeline(m_fforce_pipe);

    if (m_wd_pl)     m_device.destroyPipelineLayout(m_wd_pl);
    if (m_gforce_pl) m_device.destroyPipelineLayout(m_gforce_pl);
    if (m_fforce_pl) m_device.destroyPipelineLayout(m_fforce_pl);

    if (m_wd_mod)     m_device.destroyShaderModule(m_wd_mod);
    if (m_gforce_mod) m_device.destroyShaderModule(m_gforce_mod);
    if (m_fforce_mod) m_device.destroyShaderModule(m_fforce_mod);

    if (m_desc_pool)     m_device.destroyDescriptorPool(m_desc_pool);
    if (m_wd_layout)     m_device.destroyDescriptorSetLayout(m_wd_layout);
    if (m_gforce_layout) m_device.destroyDescriptorSetLayout(m_gforce_layout);
    if (m_fforce_layout) m_device.destroyDescriptorSetLayout(m_fforce_layout);

    auto freeBuf = [&](Buffer& b)
    {
        if (b.mapped) m_device.unmapMemory(b.mem);
        if (b.buf)    m_device.destroyBuffer(b.buf);
        if (b.mem)    m_device.freeMemory(b.mem);
    };
    freeBuf(m_force);
    freeBuf(m_eta);
    freeBuf(m_phi);

    m_cg.reset();
}

void GpuSchwingerForce::createBuffers()
{
    const vk::DeviceSize spinor_bytes
        = static_cast<vk::DeviceSize>(m_V) * 2 * 2 * sizeof(float);
    const vk::DeviceSize force_bytes
        = static_cast<vk::DeviceSize>(m_V) * 2 * sizeof(float);

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

    alloc(spinor_bytes, m_phi);
    alloc(spinor_bytes, m_eta);
    alloc(force_bytes,  m_force);
}

void GpuSchwingerForce::createPipelinesAndDescriptors(const std::string& shader_dir)
{
    // WD apply: (gauge, psi_in=cg.x, eta_out)   — 3 storage buffers
    // gauge_force: (gauge, force)                — 2 storage buffers
    // fermion_force: (gauge, psi=cg.x, eta, force) — 4 storage buffers
    m_wd_layout     = makeStorageLayout(m_device, 3);
    m_gforce_layout = makeStorageLayout(m_device, 2);
    m_fforce_layout = makeStorageLayout(m_device, 4);

    vk::DescriptorPoolSize ps{};
    ps.type            = vk::DescriptorType::eStorageBuffer;
    ps.descriptorCount = 3 + 2 + 4;  // bindings across all sets
    vk::DescriptorPoolCreateInfo pci{};
    pci.maxSets       = 3;
    pci.poolSizeCount = 1;
    pci.pPoolSizes    = &ps;
    m_desc_pool = m_device.createDescriptorPool(pci);

    auto allocSet = [&](vk::DescriptorSetLayout layout)
    {
        vk::DescriptorSetAllocateInfo dsai{};
        dsai.descriptorPool     = m_desc_pool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts        = &layout;
        return m_device.allocateDescriptorSets(dsai).front();
    };
    m_set_wd     = allocSet(m_wd_layout);
    m_set_gforce = allocSet(m_gforce_layout);
    m_set_fforce = allocSet(m_fforce_layout);

    auto writeStorage =
        [&](vk::DescriptorSet set, std::uint32_t binding,
            vk::Buffer buf, vk::DeviceSize bytes)
    {
        vk::DescriptorBufferInfo dbi{};
        dbi.buffer = buf;
        dbi.offset = 0;
        dbi.range  = bytes;
        vk::WriteDescriptorSet w{};
        w.dstSet          = set;
        w.dstBinding      = binding;
        w.descriptorCount = 1;
        w.descriptorType  = vk::DescriptorType::eStorageBuffer;
        w.pBufferInfo     = &dbi;
        m_device.updateDescriptorSets(1, &w, 0, nullptr);
    };

    // WD apply: η = D · ψ — gauge from CG, psi from CG solution, eta own.
    writeStorage(m_set_wd, 0, m_cg->gaugeBuffer(),    m_cg->gaugeBytes());
    writeStorage(m_set_wd, 1, m_cg->solutionBuffer(), m_cg->spinorBytes());
    writeStorage(m_set_wd, 2, m_eta.buf,              m_eta.bytes);

    // Gauge force: (gauge, force)
    writeStorage(m_set_gforce, 0, m_cg->gaugeBuffer(), m_cg->gaugeBytes());
    writeStorage(m_set_gforce, 1, m_force.buf,         m_force.bytes);

    // Fermion force: (gauge, psi=cg.x, eta, force)
    writeStorage(m_set_fforce, 0, m_cg->gaugeBuffer(),    m_cg->gaugeBytes());
    writeStorage(m_set_fforce, 1, m_cg->solutionBuffer(), m_cg->spinorBytes());
    writeStorage(m_set_fforce, 2, m_eta.buf,              m_eta.bytes);
    writeStorage(m_set_fforce, 3, m_force.buf,            m_force.bytes);

    // ---- shader modules ----
    auto loadModule = [&](const std::string& file)
    {
        auto bytes = readFile(shader_dir + "/" + file);
        vk::ShaderModuleCreateInfo smci{};
        smci.codeSize = bytes.size();
        smci.pCode    = reinterpret_cast<const std::uint32_t*>(bytes.data());
        return m_device.createShaderModule(smci);
    };
    m_wd_mod     = loadModule("wilson_dirac_2d.comp.spv");
    m_gforce_mod = loadModule("gauge_force_u1.comp.spv");
    m_fforce_mod = loadModule("fermion_force_u1.comp.spv");

    auto makePipeline =
        [&](vk::ShaderModule shader, vk::DescriptorSetLayout layout,
            std::uint32_t pc_size,
            vk::PipelineLayout& out_pl, vk::Pipeline& out_pipe)
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

    makePipeline(m_wd_mod,     m_wd_layout,
                 sizeof(WdPushConstants),
                 m_wd_pl, m_wd_pipe);
    makePipeline(m_gforce_mod, m_gforce_layout,
                 sizeof(GaugeForcePushConstants),
                 m_gforce_pl, m_gforce_pipe);
    makePipeline(m_fforce_mod, m_fforce_layout,
                 sizeof(FermionForcePushConstants),
                 m_fforce_pl, m_fforce_pipe);
}

void GpuSchwingerForce::allocateCommandBuffer()
{
    vk::CommandBufferAllocateInfo cai{};
    cai.commandPool        = m_cmd_pool;
    cai.level              = vk::CommandBufferLevel::ePrimary;
    cai.commandBufferCount = 1;
    m_cmd   = m_device.allocateCommandBuffers(cai).front();
    m_fence = m_device.createFence({});
}

void GpuSchwingerForce::uploadPhi(const SpinorField<2, 2>& phi)
{
    auto* dst = static_cast<float*>(m_phi.mapped);
    for (int i = 0; i < m_V; ++i)
        for (int c = 0; c < 2; ++c)
        {
            const auto z = phi(i, c);
            dst[(i * 2 + c) * 2 + 0] = static_cast<float>(z.real());
            dst[(i * 2 + c) * 2 + 1] = static_cast<float>(z.imag());
        }
}

void GpuSchwingerForce::downloadForce(LinkField<double, 2>& force_out) const
{
    const auto* src = static_cast<const float*>(m_force.mapped);
    for (int s = 0; s < m_V; ++s)
        for (int mu = 0; mu < 2; ++mu)
            force_out(s, mu) = static_cast<double>(src[s * 2 + mu]);
}

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

void GpuSchwingerForce::dispatchEta(float mass)
{
    (void)m_device.resetFences(1, &m_fence);
    m_cmd.reset();
    vk::CommandBufferBeginInfo bi{};
    bi.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    m_cmd.begin(bi);

    m_cmd.bindPipeline(vk::PipelineBindPoint::eCompute, m_wd_pipe);
    m_cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                             m_wd_pl, 0, 1, &m_set_wd, 0, nullptr);
    WdPushConstants pc{};
    pc.Lx = m_L; pc.Ly = m_L; pc.mass = mass; pc.dagger = 0;
    m_cmd.pushConstants(m_wd_pl, vk::ShaderStageFlagBits::eCompute,
                        0, sizeof(pc), &pc);
    const std::uint32_t groups
        = (static_cast<std::uint32_t>(m_V) + kWG - 1) / kWG;
    m_cmd.dispatch(groups, 1, 1);
    m_cmd.end();

    vk::SubmitInfo si{};
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &m_cmd;
    (void)m_queue.submit(1, &si, m_fence);
    if (m_device.waitForFences(1, &m_fence, VK_TRUE, UINT64_MAX) != vk::Result::eSuccess)
        throw std::runtime_error("GpuSchwingerForce::dispatchEta fence wait failed");
}

void GpuSchwingerForce::dispatchForces(float beta)
{
    (void)m_device.resetFences(1, &m_fence);
    m_cmd.reset();
    vk::CommandBufferBeginInfo bi{};
    bi.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    m_cmd.begin(bi);

    const std::uint32_t groups
        = (static_cast<std::uint32_t>(m_V) + kWG - 1) / kWG;

    // ---- gauge force: overwrite force buffer ----
    m_cmd.bindPipeline(vk::PipelineBindPoint::eCompute, m_gforce_pipe);
    m_cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                             m_gforce_pl, 0, 1, &m_set_gforce, 0, nullptr);
    {
        GaugeForcePushConstants pc{};
        pc.Lx = m_L; pc.Ly = m_L; pc.beta = beta;
        m_cmd.pushConstants(m_gforce_pl, vk::ShaderStageFlagBits::eCompute,
                            0, sizeof(pc), &pc);
        m_cmd.dispatch(groups, 1, 1);
    }
    recordBarrier(m_cmd);

    // ---- fermion force: accumulate into force buffer ----
    m_cmd.bindPipeline(vk::PipelineBindPoint::eCompute, m_fforce_pipe);
    m_cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                             m_fforce_pl, 0, 1, &m_set_fforce, 0, nullptr);
    {
        FermionForcePushConstants pc{};
        pc.Lx = m_L; pc.Ly = m_L;
        m_cmd.pushConstants(m_fforce_pl, vk::ShaderStageFlagBits::eCompute,
                            0, sizeof(pc), &pc);
        m_cmd.dispatch(groups, 1, 1);
    }
    m_cmd.end();

    vk::SubmitInfo si{};
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &m_cmd;
    (void)m_queue.submit(1, &si, m_fence);
    if (m_device.waitForFences(1, &m_fence, VK_TRUE, UINT64_MAX) != vk::Result::eSuccess)
        throw std::runtime_error("GpuSchwingerForce::dispatchForces fence wait failed");
}

void GpuSchwingerForce::computeForce(const LinkField<double, 2>&    U,
                                     const SpinorField<2, 2>&       phi,
                                     double                          mass,
                                     double                          beta,
                                     double                          cg_tol,
                                     int                             cg_max_iters,
                                     LinkField<double, 2>&          force_out)
{
    // 1. Solve ψ = (D†D)⁻¹ φ on GPU using the inner CG. This uploads U and
    //    φ; ψ lives in m_cg's solution buffer.
    Lattice<2> lattice = Lattice<2>::cube(m_L);
    SpinorField<2, 2> psi_tmp(lattice);  // discard
    (void)m_cg->solve(U, phi, mass, cg_tol, cg_max_iters, psi_tmp);

    // 2. η = D ψ on GPU.
    dispatchEta(static_cast<float>(mass));

    // 3. F = F_gauge + F_pf written into m_force.
    dispatchForces(static_cast<float>(beta));

    // 4. Download.
    downloadForce(force_out);
}

GpuSchwingerForce::Result GpuSchwingerForce::trajectory(
    LinkField<double, 2>& theta_io,
    double beta, double mass, double dt, int n_steps,
    double cg_tol, int cg_max_iters, Rng& rng)
{
    Lattice<2> lattice = Lattice<2>::cube(m_L);
    const int  V_links = m_V * 2;

    // ---- sample π, χ, build φ = D†χ ----
    LinkField<double, 2> P(lattice);
    SpinorField<2, 2>    chi(lattice), phi(lattice);
    for (int s = 0; s < m_V; ++s)
    {
        for (int mu = 0; mu < 2; ++mu) P(s, mu) = rng.normal();
        const double sigma = 1.0 / std::sqrt(2.0);
        for (int c = 0; c < 2; ++c)
            chi(s, c) = std::complex<double>(sigma * rng.normal(),
                                             sigma * rng.normal());
    }
    dirac::WilsonDirac2D D(lattice, mass);
    D.applyDagger(theta_io, chi, phi);

    u1::U1Model<2> gauge_model(lattice, beta, /*step*/1.0);

    auto kinetic = [&](const LinkField<double, 2>& Pi)
    {
        double s = 0.0;
        for (int i = 0; i < V_links; ++i) s += Pi.data()[i] * Pi.data()[i];
        return 0.5 * s;
    };
    auto pf_action = [&](const LinkField<double, 2>& th)
    {
        SpinorField<2, 2> psi(lattice);
        (void)m_cg->solve(th, phi, mass, cg_tol, cg_max_iters, psi);
        return inner_product(phi, psi).real();
    };
    auto hamiltonian = [&](const LinkField<double, 2>& th,
                           const LinkField<double, 2>& Pi)
    {
        return kinetic(Pi) + gauge_model.totalAction(th) + pf_action(th);
    };
    auto force = [&](const LinkField<double, 2>& th,
                     LinkField<double, 2>& F)
    {
        computeForce(th, phi, mass, beta, cg_tol, cg_max_iters, F);
    };

    LinkField<double, 2> theta_new = theta_io;   // working copy
    LinkField<double, 2> P_new     = P;
    LinkField<double, 2> F_buf(lattice);

    const double H_init = hamiltonian(theta_new, P_new);

    // Initial half-kick: π += (dt/2)·F
    force(theta_new, F_buf);
    for (int i = 0; i < V_links; ++i)
        P_new.data()[i] += 0.5 * dt * F_buf.data()[i];

    // N − 1 drift + full-kick pairs
    for (int step = 0; step < n_steps - 1; ++step)
    {
        for (int i = 0; i < V_links; ++i)
            theta_new.data()[i] += dt * P_new.data()[i];
        force(theta_new, F_buf);
        for (int i = 0; i < V_links; ++i)
            P_new.data()[i] += dt * F_buf.data()[i];
    }

    // Final drift + half-kick
    for (int i = 0; i < V_links; ++i)
        theta_new.data()[i] += dt * P_new.data()[i];
    force(theta_new, F_buf);
    for (int i = 0; i < V_links; ++i)
        P_new.data()[i] += 0.5 * dt * F_buf.data()[i];

    const double H_final = hamiltonian(theta_new, P_new);
    const double dH = H_final - H_init;

    Result r{};
    r.dH = dH;
    const bool accept = (dH <= 0.0) || (rng.uniform() < std::exp(-dH));
    r.accepted = accept;
    if (accept) theta_io = theta_new;
    return r;
}

double GpuSchwingerForce::selfValidateTrajectory(double beta,
                                                 double mass,
                                                 double dt,
                                                 int    n_steps,
                                                 double cg_tol,
                                                 int    cg_max_iters,
                                                 std::uint64_t seed)
{
    Lattice<2> lattice = Lattice<2>::cube(m_L);
    const int   V_links = m_V * 2;
    const int   N_spin  = m_V * 2;     // V sites × Nc components

    // ---- shared deterministic initial state ----
    LinkField<double, 2> theta0(lattice);
    LinkField<double, 2> pi0(lattice);
    SpinorField<2, 2>    chi(lattice);

    Rng rng(seed);
    for (int s = 0; s < m_V; ++s)
    {
        for (int mu = 0; mu < 2; ++mu)
        {
            theta0(s, mu) = rng.uniform() * 2.0 * 3.14159265358979 - 3.14159265358979;
            pi0(s, mu)    = rng.normal();
        }
        for (int c = 0; c < 2; ++c)
        {
            const double sigma = 1.0 / std::sqrt(2.0);
            chi(s, c) = std::complex<double>(sigma * rng.normal(),
                                             sigma * rng.normal());
        }
    }

    // Pseudofermion is the same on both sides: φ = D† χ (CPU evaluation —
    // double precision so both backends see identical φ).
    dirac::WilsonDirac2D D(lattice, mass);
    SpinorField<2, 2> phi(lattice);
    D.applyDagger(theta0, chi, phi);

    u1::U1Model<2> gauge_model(lattice, beta, /*step*/1.0);

    auto kinetic = [&](const LinkField<double, 2>& P)
    {
        double s = 0.0;
        for (int i = 0; i < V_links; ++i) s += P.data()[i] * P.data()[i];
        return 0.5 * s;
    };

    auto cpu_pseudofermion_action = [&](const LinkField<double, 2>& th)
    {
        // ψ = (D†D)⁻¹ φ via CPU CG; then S_pf = Re ⟨φ, ψ⟩.
        SpinorField<2, 2> psi(lattice), tmp(lattice);
        SpinorField<2, 2> r(lattice), p(lattice), Ap(lattice);
        auto apply_DagD = [&](const SpinorField<2, 2>& in, SpinorField<2, 2>& out)
        {
            D.applyDagD(th, in, out, tmp);
        };
        psi.zero();
        (void)solvers::conjugateGradient<2, 2>(
            apply_DagD, phi, psi, cg_tol, cg_max_iters, r, p, Ap);
        return inner_product(phi, psi).real();
    };

    auto cpu_H = [&](const LinkField<double, 2>& th, const LinkField<double, 2>& P)
    {
        return kinetic(P) + gauge_model.totalAction(th)
             + cpu_pseudofermion_action(th);
    };

    // ---- CPU trajectory (manual leapfrog matching hmc::GaugeHMC) ----
    auto cpu_force = [&](const LinkField<double, 2>& th,
                         LinkField<double, 2>& F)
    {
        for (int i = 0; i < V_links; ++i) F.data()[i] = 0.0;
        schwinger::SchwingerProvider prov(gauge_model, D, lattice,
                                          cg_tol, cg_max_iters);
        auto& phi_ref = const_cast<SpinorField<2, 2>&>(prov.phi());
        phi_ref = phi;
        prov.computeAllGaugeForces(th, F);
    };

    LinkField<double, 2> theta_cpu = theta0;
    LinkField<double, 2> P_cpu     = pi0;
    LinkField<double, 2> F_cpu(lattice);

    const double H_init_cpu = cpu_H(theta_cpu, P_cpu);
    cpu_force(theta_cpu, F_cpu);
    for (int i = 0; i < V_links; ++i)
        P_cpu.data()[i] += 0.5 * dt * F_cpu.data()[i];
    for (int step = 0; step < n_steps - 1; ++step)
    {
        for (int i = 0; i < V_links; ++i)
            theta_cpu.data()[i] += dt * P_cpu.data()[i];
        cpu_force(theta_cpu, F_cpu);
        for (int i = 0; i < V_links; ++i)
            P_cpu.data()[i] += dt * F_cpu.data()[i];
    }
    for (int i = 0; i < V_links; ++i)
        theta_cpu.data()[i] += dt * P_cpu.data()[i];
    cpu_force(theta_cpu, F_cpu);
    for (int i = 0; i < V_links; ++i)
        P_cpu.data()[i] += 0.5 * dt * F_cpu.data()[i];
    const double H_final_cpu = cpu_H(theta_cpu, P_cpu);
    const double dH_cpu = H_final_cpu - H_init_cpu;

    // ---- GPU trajectory (same host-side orchestration; force on GPU) ----
    auto gpu_force = [&](const LinkField<double, 2>& th,
                         LinkField<double, 2>& F)
    {
        computeForce(th, phi, mass, beta, cg_tol, cg_max_iters, F);
    };
    auto gpu_pseudofermion_action = [&](const LinkField<double, 2>& th)
    {
        // Reuse the inner GpuCG2D to solve ψ = (D†D)⁻¹ φ on GPU, then take
        // the inner product on host (cheap — V·Nc complex multiplies).
        SpinorField<2, 2> psi(lattice);
        (void)m_cg->solve(th, phi, mass, cg_tol, cg_max_iters, psi);
        return inner_product(phi, psi).real();
    };
    auto gpu_H = [&](const LinkField<double, 2>& th, const LinkField<double, 2>& P)
    {
        return kinetic(P) + gauge_model.totalAction(th)
             + gpu_pseudofermion_action(th);
    };

    LinkField<double, 2> theta_gpu = theta0;
    LinkField<double, 2> P_gpu     = pi0;
    LinkField<double, 2> F_gpu(lattice);

    const double H_init_gpu = gpu_H(theta_gpu, P_gpu);
    gpu_force(theta_gpu, F_gpu);
    for (int i = 0; i < V_links; ++i)
        P_gpu.data()[i] += 0.5 * dt * F_gpu.data()[i];
    for (int step = 0; step < n_steps - 1; ++step)
    {
        for (int i = 0; i < V_links; ++i)
            theta_gpu.data()[i] += dt * P_gpu.data()[i];
        gpu_force(theta_gpu, F_gpu);
        for (int i = 0; i < V_links; ++i)
            P_gpu.data()[i] += dt * F_gpu.data()[i];
    }
    for (int i = 0; i < V_links; ++i)
        theta_gpu.data()[i] += dt * P_gpu.data()[i];
    gpu_force(theta_gpu, F_gpu);
    for (int i = 0; i < V_links; ++i)
        P_gpu.data()[i] += 0.5 * dt * F_gpu.data()[i];
    const double H_final_gpu = gpu_H(theta_gpu, P_gpu);
    const double dH_gpu = H_final_gpu - H_init_gpu;

    (void)N_spin;
    return std::abs(dH_cpu - dH_gpu);
}

double GpuSchwingerForce::selfValidate(double beta, double mass,
                                       double cg_tol, int cg_max_iters,
                                       std::uint64_t seed)
{
    Lattice<2> lattice = Lattice<2>::cube(m_L);
    LinkField<double, 2> U(lattice);
    SpinorField<2, 2>    phi(lattice);
    LinkField<double, 2> F_gpu(lattice), F_cpu(lattice, 0.0);

    Rng rng(seed);
    for (int s = 0; s < m_V; ++s)
    {
        for (int mu = 0; mu < 2; ++mu)
            U(s, mu) = rng.uniform() * 2.0 * 3.14159265358979 - 3.14159265358979;
        for (int c = 0; c < 2; ++c)
            phi(s, c) = std::complex<double>(rng.normal(), rng.normal());
    }

    // CPU reference: use the existing SchwingerProvider, set its φ, compute
    // forces. We poke φ via refreshPseudofermion-like upload (it owns m_phi
    // internally) — simpler is to just call computeAllGaugeForces after
    // staging φ. Since SchwingerProvider takes φ as a side effect of
    // refreshPseudofermion (which samples χ then sets φ = D†χ), we use a
    // small local copy of the same code path.
    u1::U1Model<2> gauge_model(lattice, beta, /*step*/1.0);
    dirac::WilsonDirac2D D(lattice, mass);
    // Build a SchwingerProvider and force its φ to match ours.
    schwinger::SchwingerProvider prov(gauge_model, D, lattice,
                                      cg_tol, cg_max_iters);
    // Trick: refreshPseudofermion uses internal RNG; instead we run it with
    // our seed so χ matches ours, but we need φ = our phi. Simpler: do not
    // use refreshPseudofermion — instead overwrite the provider's φ buffer
    // through a const_cast on phi() (since it returns const&).
    auto& phi_ref = const_cast<SpinorField<2, 2>&>(prov.phi());
    phi_ref = phi;

    prov.computeAllGaugeForces(U, F_cpu);
    computeForce(U, phi, mass, beta, cg_tol, cg_max_iters, F_gpu);

    double num = 0.0, den = 0.0;
    for (int s = 0; s < m_V; ++s)
        for (int mu = 0; mu < 2; ++mu)
        {
            const double d = F_cpu(s, mu) - F_gpu(s, mu);
            num += d * d;
            den += F_cpu(s, mu) * F_cpu(s, mu);
        }
    return (den > 0) ? std::sqrt(num / den) : std::sqrt(num);
}

std::vector<char> GpuSchwingerForce::readFile(const std::string& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) throw std::runtime_error("failed to open " + path);
    const std::size_t sz = static_cast<std::size_t>(f.tellg());
    std::vector<char> buf(sz);
    f.seekg(0);
    f.read(buf.data(), static_cast<std::streamsize>(sz));
    return buf;
}

std::uint32_t GpuSchwingerForce::findMemoryType(std::uint32_t type_bits,
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
