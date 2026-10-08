#include "su2_2d_scene.hpp"

#include "gpu_su2_sweeper.hpp"

#include <imgui.h>

#include <stdexcept>

namespace lqft::vis
{

SU2_2DScene::SU2_2DScene(int L, std::uint64_t seed, const VulkanContext* ctx)
    : m_L(L)
    , m_lattice(Lattice<2>::cube(L))
    , m_field(m_lattice)
    , m_rng(seed)
    , m_model(m_lattice, m_beta, /*step*/0.4)
    , m_sweeper(m_model)
{
    resetHot(seed);
    if (ctx && ctx->device)
    {
        try
        {
            m_gpu = std::make_unique<GpuSU2Sweeper>(
                ctx->phys, ctx->device, ctx->queue,
                ctx->queue_family, ctx->cmd_pool,
                L, /*Dim*/2, ctx->shader_dir, seed);
            m_gpu->uploadField<2>(m_field);
            m_use_gpu = true;
        }
        catch (const std::exception&) { m_gpu.reset(); m_use_gpu = false; }
    }
}

SU2_2DScene::~SU2_2DScene() = default;

void SU2_2DScene::step()
{
    if (m_paused) return;
    if (m_use_gpu && m_gpu)
        m_gpu->sweep(m_gpu_sweeps_per_frame, m_beta, /*step_size*/0.4);
    else if (m_use_heatbath)
        su2_model::heatBathSweepN(m_model, m_field, m_rng, m_sweeps_per_frame);
    else
        m_sweeper.sweepN(m_field, m_rng, m_sweeps_per_frame);
}

void SU2_2DScene::singleStep()
{
    if (m_use_gpu && m_gpu) m_gpu->sweep(1, m_beta, 0.4);
    else if (m_use_heatbath) su2_model::heatBathSweep(m_model, m_field, m_rng);
    else                     m_sweeper.sweep(m_field, m_rng);
}

void SU2_2DScene::resetHot(std::uint64_t seed)
{
    m_rng = Rng(seed);
    su2_model::SU2Model<2>::hot(m_field, m_rng);
    m_sweeper.resetCounters();
    if (m_gpu) { m_gpu->reseedRng(seed); m_gpu->uploadField<2>(m_field); }
}

void SU2_2DScene::resetCold()
{
    su2_model::SU2Model<2>::cold(m_field);
    m_sweeper.resetCounters();
    if (m_gpu) m_gpu->uploadField<2>(m_field);
}

void SU2_2DScene::syncFromGpu() const
{
    if (m_use_gpu && m_gpu)
        m_gpu->downloadField<2>(const_cast<LinkField<su2::Element, 2>&>(m_field));
}

void SU2_2DScene::fillVolume(float* out) const
{
    syncFromGpu();
    const int L = m_L;
    for (int y = 0; y < L; ++y)
        for (int x = 0; x < L; ++x)
        {
            const int site = m_lattice.siteIndex({ x, y });
            const int s_mu = m_lattice.forward(site, 0);
            const int s_nu = m_lattice.forward(site, 1);
            su2::Element P = su2::multiply(m_field(site, 0), m_field(s_mu, 1));
            P = su2::multiply(P, su2::dagger(m_field(s_nu, 0)));
            P = su2::multiply(P, su2::dagger(m_field(site, 1)));
            out[y * L + x] = static_cast<float>(1.0 - P.s);
        }
}

bool SU2_2DScene::buildControlsUI()
{
    ImGui::Text("Lattice L = %d   V = %d", m_L, m_L * m_L);
    ImGui::TextDisabled("2D SU(2) Yang-Mills — exactly factorizable in 2D");
    ImGui::Separator();

    if (ImGui::CollapsingHeader("Monte Carlo", ImGuiTreeNodeFlags_DefaultOpen))
    {
        float beta = static_cast<float>(m_beta);
        if (ImGui::SliderFloat("beta", &beta, 0.4f, 10.0f, "%.3f"))
        {
            m_beta = beta;
            m_model.setBeta(beta);
        }
        if (m_gpu)
        {
            bool use_gpu = m_use_gpu;
            if (ImGui::Checkbox("GPU Metropolis (vs CPU heat-bath)", &use_gpu))
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
            ImGui::SliderInt("sweeps/frame", &m_sweeps_per_frame, 1, 10);
            ImGui::Checkbox("heat-bath (Kennedy-Pendleton)", &m_use_heatbath);
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
    ImGui::Text("⟨½ tr U_□⟩ = %.5f", su2_model::averagePlaquette(m_lattice, m_field));
    ImGui::Text("avg %.2f ms/frame (%.1f FPS)",
                1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

    ImGui::Separator();
    return ImGui::Button("← Back to menu");
}

} // namespace lqft::vis
