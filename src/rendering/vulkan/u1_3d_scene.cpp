#include "u1_3d_scene.hpp"

#include "gpu_u1_sweeper.hpp"

#include <imgui.h>

#include <cmath>
#include <stdexcept>

namespace lqft::vis
{

U1Scene3D::U1Scene3D(int L, std::uint64_t seed, const VulkanContext* ctx)
    : m_L(L)
    , m_lattice(Lattice<3>::cube(L))
    , m_field(m_lattice)
    , m_rng(seed)
    , m_model(m_lattice, m_beta, /*step*/1.0)
    , m_sweeper(m_model)
{
    resetHot(seed);
    mc::autoTuneStepSize(m_model, m_field, m_rng,
                         /*batch*/40, /*max*/10, /*target*/0.5, /*tol*/0.05);
    m_sweeper.resetCounters();

    if (ctx && ctx->device)
    {
        try
        {
            m_gpu = std::make_unique<GpuU1Sweeper>(
                ctx->phys, ctx->device, ctx->queue,
                ctx->queue_family, ctx->cmd_pool,
                L, /*Dim*/3, ctx->shader_dir, seed);
            m_gpu->uploadField(m_field);
            m_use_gpu = true;
        }
        catch (const std::exception&) { m_gpu.reset(); m_use_gpu = false; }
    }
}

U1Scene3D::~U1Scene3D() = default;

void U1Scene3D::step()
{
    if (m_paused) return;
    if (m_use_gpu && m_gpu)
    {
        m_gpu->sweep(m_gpu_sweeps_per_frame, m_beta, /*step_size*/1.0);
        m_gpu->downloadField(m_field);
    }
    else if (m_use_heatbath)
    {
        u1::heatBathSweepN(m_model, m_field, m_rng, m_sweeps_per_frame);
    }
    else
    {
        m_sweeper.sweepN(m_field, m_rng, m_sweeps_per_frame);
    }
}

void U1Scene3D::singleStep()
{
    if (m_use_heatbath) u1::heatBathSweep(m_model, m_field, m_rng);
    else                m_sweeper.sweep(m_field, m_rng);
}

void U1Scene3D::resetHot(std::uint64_t seed)
{
    m_rng = Rng(seed);
    u1::U1Model<3>::hot(m_field, m_rng);
    m_sweeper.resetCounters();
    if (m_gpu) { m_gpu->reseedRng(seed); m_gpu->uploadField(m_field); }
}

void U1Scene3D::resetCold()
{
    u1::U1Model<3>::cold(m_field);
    m_sweeper.resetCounters();
    if (m_gpu) m_gpu->uploadField(m_field);
}

void U1Scene3D::fillVolume(float* out) const
{
    const int L = m_L;
    for (int z = 0; z < L; ++z)
        for (int y = 0; y < L; ++y)
            for (int x = 0; x < L; ++x)
            {
                const int site = m_lattice.siteIndex({ x, y, z });
                double rho = 0.0;
                for (int mu = 0; mu < 3; ++mu)
                    for (int nu = mu + 1; nu < 3; ++nu)
                    {
                        const int s_mu = m_lattice.forward(site, mu);
                        const int s_nu = m_lattice.forward(site, nu);
                        const double theta_plaq =
                            m_field(site, mu) + m_field(s_mu, nu)
                          - m_field(s_nu, mu) - m_field(site, nu);
                        rho += 1.0 - std::cos(theta_plaq);
                    }
                out[(static_cast<std::size_t>(z) * L + y) * L + x]
                    = static_cast<float>(rho);
            }
}

bool U1Scene3D::buildControlsUI()
{
    ImGui::Text("Lattice L = %d  (V = %d)", m_L, m_L * m_L * m_L);
    ImGui::TextUnformatted("3D compact U(1) Wilson plaquette action");
    ImGui::Separator();

    if (ImGui::CollapsingHeader("Monte Carlo", ImGuiTreeNodeFlags_DefaultOpen))
    {
        float beta = static_cast<float>(m_beta);
        if (ImGui::SliderFloat("beta", &beta, 0.1f, 6.0f, "%.3f"))
        {
            m_beta = beta;
            m_model.setBeta(beta);
            if (!m_use_heatbath && !m_use_gpu)
                mc::autoTuneStepSize(m_model, m_field, m_rng,
                                     /*batch*/30, /*max*/5,
                                     /*target*/0.5, /*tol*/0.08);
        }

        if (m_gpu)
        {
            bool use_gpu = m_use_gpu;
            if (ImGui::Checkbox("GPU Metropolis (vs CPU)", &use_gpu))
                m_use_gpu = use_gpu;
            if (m_use_gpu)
                ImGui::SliderInt("GPU sweeps/frame", &m_gpu_sweeps_per_frame, 1, 200);
        }
        else
        {
            ImGui::TextDisabled("(GPU compute unavailable; CPU only)");
        }
        if (!m_use_gpu)
        {
            ImGui::SliderInt("sweeps/frame", &m_sweeps_per_frame, 1, 20);
            ImGui::Checkbox("heat-bath (vs Metropolis)", &m_use_heatbath);
        }

        ImGui::Checkbox("paused", &m_paused);
        ImGui::SameLine();
        if (ImGui::Button("step")) singleStep();
        ImGui::SameLine();
        if (ImGui::Button("reset hot"))
            resetHot(static_cast<std::uint64_t>(ImGui::GetTime() * 1e6));
        ImGui::SameLine();
        if (ImGui::Button("reset cold")) resetCold();
    }

    ImGui::Separator();
    ImGui::Text("avg plaquette = %.5f",
                u1::averagePlaquette(m_lattice, m_field));
    if (!m_use_heatbath)
        ImGui::Text("acceptance    = %.3f", m_sweeper.cumulativeAcceptance());
    ImGui::Text("avg %.2f ms/frame (%.1f FPS)",
                1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

    ImGui::Separator();
    return ImGui::Button("← Back to menu");
}

} // namespace lqft::vis
