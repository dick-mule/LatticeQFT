#include "u1_2d_scene.hpp"

#include "gpu_u1_sweeper.hpp"

#include <imgui.h>

#include <cmath>
#include <stdexcept>

namespace lqft::vis
{

U1_2DScene::U1_2DScene(int L, std::uint64_t seed, const VulkanContext* ctx)
    : m_L(L)
    , m_lattice(Lattice<2>::cube(L))
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
                L, /*Dim*/2, ctx->shader_dir, seed);
            m_gpu->uploadField<2>(m_field);
            m_use_gpu = true;
        }
        catch (const std::exception&) { m_gpu.reset(); m_use_gpu = false; }
    }
}

U1_2DScene::~U1_2DScene() = default;

void U1_2DScene::step()
{
    if (m_paused) return;
    if (m_use_gpu && m_gpu)
        m_gpu->sweep(m_gpu_sweeps_per_frame, m_beta, /*step_size*/1.0);
    else if (m_use_heatbath)
        u1::heatBathSweepN(m_model, m_field, m_rng, m_sweeps_per_frame);
    else
        m_sweeper.sweepN(m_field, m_rng, m_sweeps_per_frame);
}

void U1_2DScene::singleStep()
{
    if (m_use_gpu && m_gpu) m_gpu->sweep(1, m_beta, 1.0);
    else if (m_use_heatbath) u1::heatBathSweep(m_model, m_field, m_rng);
    else                     m_sweeper.sweep(m_field, m_rng);
}

void U1_2DScene::resetHot(std::uint64_t seed)
{
    m_rng = Rng(seed);
    u1::U1Model<2>::hot(m_field, m_rng);
    m_sweeper.resetCounters();
    if (m_gpu) { m_gpu->reseedRng(seed); m_gpu->uploadField<2>(m_field); }
}

void U1_2DScene::resetCold()
{
    u1::U1Model<2>::cold(m_field);
    m_sweeper.resetCounters();
    if (m_gpu) m_gpu->uploadField<2>(m_field);
}

void U1_2DScene::syncFromGpu() const
{
    if (m_use_gpu && m_gpu)
        m_gpu->downloadField<2>(const_cast<LinkField<double, 2>&>(m_field));
}

void U1_2DScene::fillVolume(float* out) const
{
    syncFromGpu();
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

bool U1_2DScene::buildControlsUI()
{
    ImGui::Text("Lattice L = %d   V = %d", m_L, m_L * m_L);
    ImGui::TextDisabled("2D compact U(1) — Bessel-ratio ⟨cos θ_□⟩ = I_1(β)/I_0(β)");
    ImGui::Separator();

    if (ImGui::CollapsingHeader("Monte Carlo", ImGuiTreeNodeFlags_DefaultOpen))
    {
        float beta = static_cast<float>(m_beta);
        if (ImGui::SliderFloat("beta", &beta, 0.1f, 6.0f, "%.3f"))
        {
            m_beta = beta;
            m_model.setBeta(beta);
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
    syncFromGpu();
    ImGui::Text("⟨cos θ_□⟩ = %.5f", u1::averagePlaquette(m_lattice, m_field));
    ImGui::Text("avg %.2f ms/frame (%.1f FPS)",
                1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

    ImGui::Separator();
    return ImGui::Button("← Back to menu");
}

} // namespace lqft::vis
