#include "schwinger_hmc_scene.hpp"

#include "gpu_cg_2d.hpp"
#include "gpu_schwinger_force.hpp"
#include "gpu_wilson_dirac_2d.hpp"

#include <imgui.h>

#include <cmath>
#include <stdexcept>

namespace lqft::vis
{

SchwingerHMCScene::SchwingerHMCScene(int L, std::uint64_t seed,
                                     const VulkanContext* ctx)
    : m_L(L)
    , m_lattice(Lattice<2>::cube(L))
    , m_field(m_lattice)
    , m_rng(seed)
    , m_gauge_model(m_lattice, m_beta, /*step*/1.0)
    , m_dirac(m_lattice, m_mass)
    , m_provider(m_gauge_model, m_dirac, m_lattice, /*cg_tol*/1e-9, /*cg_max*/2000)
{
    resetHot(seed);
    rebuildHmc();

    // Optional GPU-side Wilson-Dirac validation. If we were handed a Vulkan
    // context, build a GpuWilsonDirac2D and run it head-to-head against the
    // CPU operator on a deterministic random gauge+spinor pair. The max
    // element-wise error becomes a visible correctness check in the UI.
    if (ctx && ctx->device)
    {
        try
        {
            m_gpu_dirac = std::make_unique<GpuWilsonDirac2D>(
                ctx->phys, ctx->device, ctx->queue,
                ctx->queue_family, ctx->cmd_pool,
                L, ctx->shader_dir);
            m_gpu_validation_err
                = m_gpu_dirac->selfValidate(/*mass*/m_mass, /*seed*/seed ^ 0xC0FFEEull);
        }
        catch (const std::exception&)
        {
            m_gpu_dirac.reset();
            m_gpu_validation_err = -1.0;
        }

        try
        {
            m_gpu_cg = std::make_unique<GpuCG2D>(
                ctx->phys, ctx->device, ctx->queue,
                ctx->queue_family, ctx->cmd_pool,
                L, ctx->shader_dir);
            m_gpu_cg_rel_err = m_gpu_cg->selfValidateCG(
                /*mass*/m_mass, /*tol*/1e-6, /*max_iters*/4000,
                /*seed*/seed ^ 0xCAFEBABEull);
        }
        catch (const std::exception&)
        {
            m_gpu_cg.reset();
            m_gpu_cg_rel_err = -1.0;
        }

        try
        {
            m_gpu_force = std::make_unique<GpuSchwingerForce>(
                ctx->phys, ctx->device, ctx->queue,
                ctx->queue_family, ctx->cmd_pool,
                L, ctx->shader_dir);
            m_gpu_force_rel_err = m_gpu_force->selfValidate(
                /*beta*/m_beta, /*mass*/m_mass,
                /*cg_tol*/1e-6, /*cg_max_iters*/4000,
                /*seed*/seed ^ 0xDEADBEEFull);
            m_gpu_traj_dH_err = m_gpu_force->selfValidateTrajectory(
                /*beta*/m_beta, /*mass*/m_mass,
                /*dt*/m_hmc_dt, /*n_steps*/m_hmc_n_steps,
                /*cg_tol*/1e-6, /*cg_max_iters*/4000,
                /*seed*/seed ^ 0xFEEDFACEull);
        }
        catch (const std::exception&)
        {
            m_gpu_force.reset();
            m_gpu_force_rel_err = -1.0;
            m_gpu_traj_dH_err   = -1.0;
        }
    }
}

SchwingerHMCScene::~SchwingerHMCScene() = default;

void SchwingerHMCScene::rebuildHmc()
{
    m_hmc = std::make_unique<hmc::GaugeHMC<schwinger::SchwingerProvider>>(
        m_provider, m_hmc_dt, m_hmc_n_steps);
}

void SchwingerHMCScene::resetStats()
{
    m_n_trajectories = 0;
    m_n_accepted     = 0;
    m_sum_dH         = 0.0;
    m_n_measurements = 0;
    m_sum_condensate = 0.0;
}

void SchwingerHMCScene::resetHot(std::uint64_t seed)
{
    m_rng = Rng(seed);
    u1::U1Model<2>::hot(m_field, m_rng);
    resetStats();
}

void SchwingerHMCScene::resetCold()
{
    u1::U1Model<2>::cold(m_field);
    resetStats();
}

void SchwingerHMCScene::step()
{
    if (m_paused || !m_hmc) return;

    for (int t = 0; t < m_trajectories_per_frame; ++t)
    {
        bool accepted; double dH;
        if (m_use_gpu_hmc && m_gpu_force)
        {
            const auto r = m_gpu_force->trajectory(
                m_field, m_beta, m_mass,
                static_cast<double>(m_hmc_dt), m_hmc_n_steps,
                /*cg_tol*/1e-7, /*cg_max*/4000, m_rng);
            accepted = r.accepted;
            dH       = r.dH;
        }
        else
        {
            m_provider.refreshPseudofermion(m_field, m_rng);
            const auto r = m_hmc->trajectory(m_field, m_rng);
            accepted = r.accepted;
            dH       = r.dH;
        }
        (void)accepted; (void)dH; // silence -Wunused-but-set if the
                                  // counters below get refactored away
        ++m_n_trajectories;
        if (accepted) ++m_n_accepted;
        m_sum_dH += dH;

        if ((m_n_trajectories % m_meas_every_traj) == 0)
        {
            const auto c = obs::stochasticCondensate(
                m_dirac, m_field, m_lattice, m_rng,
                m_n_sources, /*cg_tol*/1e-8, /*cg_max*/2000);
            m_sum_condensate += c.mean;
            ++m_n_measurements;
        }
    }
}

void SchwingerHMCScene::fillVolume(float* out) const
{
    const int L = m_L;
    for (int y = 0; y < L; ++y)
        for (int x = 0; x < L; ++x)
        {
            const int site = m_lattice.siteIndex({ x, y });
            const int s_mu = m_lattice.forward(site, 0);
            const int s_nu = m_lattice.forward(site, 1);
            const double theta_plaq = m_field(site, 0) + m_field(s_mu, 1)
                                    - m_field(s_nu, 0) - m_field(site, 1);
            out[y * L + x] = static_cast<float>(1.0 - std::cos(theta_plaq));
        }
}

bool SchwingerHMCScene::buildControlsUI()
{
    ImGui::Text("Lattice L = %d   V = %d", m_L, m_L * m_L);
    ImGui::TextDisabled("Dynamical 2D Schwinger — pseudofermion HMC (CPU only)");
    if (m_gpu_dirac)
    {
        const ImVec4 c_pass(0.55f, 0.85f, 0.55f, 1.0f);
        const ImVec4 c_fail(0.85f, 0.45f, 0.45f, 1.0f);
        const bool   ok = m_gpu_validation_err >= 0.0 && m_gpu_validation_err < 1e-4;
        ImGui::TextColored(ok ? c_pass : c_fail,
            "GPU Wilson-Dirac self-validation: max |D_gpu − D_cpu| = %.2e",
            m_gpu_validation_err);
    }
    else
    {
        ImGui::TextDisabled("(No GPU Wilson-Dirac available in this context.)");
    }
    if (m_gpu_cg)
    {
        const ImVec4 c_pass(0.55f, 0.85f, 0.55f, 1.0f);
        const ImVec4 c_fail(0.85f, 0.45f, 0.45f, 1.0f);
        const bool   ok = m_gpu_cg_rel_err >= 0.0 && m_gpu_cg_rel_err < 1e-3;
        ImGui::TextColored(ok ? c_pass : c_fail,
            "GPU CG self-validation:          ‖x_gpu − x_cpu‖/‖x_cpu‖ = %.2e",
            m_gpu_cg_rel_err);
    }
    if (m_gpu_force)
    {
        const ImVec4 c_pass(0.55f, 0.85f, 0.55f, 1.0f);
        const ImVec4 c_fail(0.85f, 0.45f, 0.45f, 1.0f);
        {
            const bool ok = m_gpu_force_rel_err >= 0.0 && m_gpu_force_rel_err < 1e-3;
            ImGui::TextColored(ok ? c_pass : c_fail,
                "GPU Schwinger force self-val.:   ‖F_gpu − F_cpu‖/‖F_cpu‖ = %.2e",
                m_gpu_force_rel_err);
        }
        {
            const bool ok = m_gpu_traj_dH_err >= 0.0 && m_gpu_traj_dH_err < 1e-2;
            ImGui::TextColored(ok ? c_pass : c_fail,
                "GPU HMC trajectory self-val.:    |ΔH_gpu − ΔH_cpu| = %.2e",
                m_gpu_traj_dH_err);
        }
    }
    ImGui::Separator();

    if (ImGui::CollapsingHeader("Monte Carlo", ImGuiTreeNodeFlags_DefaultOpen))
    {
        float beta = static_cast<float>(m_beta);
        if (ImGui::SliderFloat("beta", &beta, 0.5f, 6.0f, "%.3f"))
        {
            m_beta = beta;
            m_gauge_model.setBeta(beta);
            resetStats();
        }
        float mass = static_cast<float>(m_mass);
        if (ImGui::SliderFloat("fermion mass", &mass, 0.05f, 1.0f, "%.3f"))
        {
            m_mass = mass;
            m_dirac.setMass(mass);
            resetStats();
        }
        if (ImGui::SliderFloat("HMC dt", &m_hmc_dt, 0.005f, 0.2f, "%.4f"))
        {
            if (m_hmc) m_hmc->setDt(m_hmc_dt);
            resetStats();
        }
        if (ImGui::SliderInt("HMC steps", &m_hmc_n_steps, 4, 60))
        {
            if (m_hmc) m_hmc->setNSteps(m_hmc_n_steps);
            resetStats();
        }
        if (m_gpu_force)
        {
            bool gpu = m_use_gpu_hmc;
            if (ImGui::Checkbox("GPU HMC backend (vs CPU)", &gpu))
            {
                m_use_gpu_hmc = gpu;
                resetStats();
            }
        }
        else
        {
            ImGui::TextDisabled("(GPU HMC backend unavailable.)");
        }
        ImGui::SliderInt("trajectories/frame", &m_trajectories_per_frame, 1, 6);
        ImGui::SliderInt("measure every K traj", &m_meas_every_traj, 1, 30);
        ImGui::SliderInt("stochastic sources", &m_n_sources, 1, 12);
        ImGui::Checkbox("paused", &m_paused);
        ImGui::SameLine();
        if (ImGui::Button("reset hot"))
            resetHot(static_cast<std::uint64_t>(ImGui::GetTime() * 1e6));
        ImGui::SameLine();
        if (ImGui::Button("reset cold")) resetCold();
        ImGui::SameLine();
        if (ImGui::Button("reset stats")) resetStats();
    }

    ImGui::Separator();
    ImGui::Text("⟨cos θ_□⟩  = %.5f", u1::averagePlaquette(m_lattice, m_field));
    if (m_n_trajectories > 0)
    {
        const double acc = static_cast<double>(m_n_accepted) / m_n_trajectories;
        const double dH  = m_sum_dH / m_n_trajectories;
        ImGui::Text("HMC acceptance = %.3f", acc);
        ImGui::Text("⟨ΔH⟩          = %+.5f", dH);
        ImGui::Text("trajectories   = %lld", m_n_trajectories);
    }
    if (m_n_measurements > 0)
        ImGui::Text("⟨ψ̄ψ⟩ (running) = %.5f   [N = %lld]",
                    m_sum_condensate / m_n_measurements,
                    m_n_measurements);
    else
        ImGui::TextDisabled("⟨ψ̄ψ⟩: waiting for first measurement...");
    ImGui::Text("avg %.2f ms/frame (%.1f FPS)",
                1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

    ImGui::Separator();
    return ImGui::Button("← Back to menu");
}

} // namespace lqft::vis
