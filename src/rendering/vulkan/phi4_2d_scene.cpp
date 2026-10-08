#include "phi4_2d_scene.hpp"

#include "gpu_phi4_sweeper.hpp"

#include <imgui.h>

#include <cmath>
#include <stdexcept>

namespace lqft::vis
{

Phi4_2DScene::Phi4_2DScene(int L, std::uint64_t seed, const VulkanContext* ctx)
    : m_L(L)
    , m_lattice(Lattice<2>::cube(L))
    , m_field(m_lattice)
    , m_rng(seed)
    , m_model(m_lattice, m_m_sq, m_lambda, m_step_size)
    , m_sweeper(m_model)
    , m_upload_buf(static_cast<std::size_t>(L) * L, 0.0f)
{
    resetHot(seed);
    mc::autoTuneStepSize(m_model, m_field, m_rng,
                         /*batch*/40, /*max*/10, /*target*/0.5, /*tol*/0.05);
    m_sweeper.resetCounters();

    if (ctx && ctx->device)
    {
        try
        {
            m_gpu = std::make_unique<GpuPhi4Sweeper>(
                ctx->phys, ctx->device, ctx->queue,
                ctx->queue_family, ctx->cmd_pool,
                L, ctx->shader_dir, seed);
            const int V = L * L;
            for (int i = 0; i < V; ++i)
                m_upload_buf[i] = static_cast<float>(m_field[i]);
            m_gpu->uploadField(m_upload_buf.data());
            m_use_gpu = true;
        }
        catch (const std::exception&) { m_gpu.reset(); m_use_gpu = false; }
    }
}

Phi4_2DScene::~Phi4_2DScene() = default;

void Phi4_2DScene::step()
{
    if (m_paused) return;
    if (m_use_gpu && m_gpu)
        m_gpu->sweep(m_gpu_sweeps_per_frame, m_m_sq, m_lambda, m_step_size);
    else
        m_sweeper.sweepN(m_field, m_rng, m_sweeps_per_frame);
}

void Phi4_2DScene::singleStep()
{
    if (m_use_gpu && m_gpu) m_gpu->sweep(1, m_m_sq, m_lambda, m_step_size);
    else                     m_sweeper.sweep(m_field, m_rng);
}

void Phi4_2DScene::resetHot(std::uint64_t seed)
{
    m_rng = Rng(seed);
    phi4::Phi4Model<2>::hot(m_field, m_rng, 1.0);
    m_sweeper.resetCounters();
    if (m_gpu)
    {
        m_gpu->reseedRng(seed);
        const int V = m_L * m_L;
        for (int i = 0; i < V; ++i)
            m_upload_buf[i] = static_cast<float>(m_field[i]);
        m_gpu->uploadField(m_upload_buf.data());
    }
}

void Phi4_2DScene::resetCold()
{
    phi4::Phi4Model<2>::cold(m_field);
    m_sweeper.resetCounters();
    if (m_gpu)
    {
        const int V = m_L * m_L;
        for (int i = 0; i < V; ++i) m_upload_buf[i] = 0.0f;
        m_gpu->uploadField(m_upload_buf.data());
    }
}

void Phi4_2DScene::syncFromGpu() const
{
    if (m_use_gpu && m_gpu)
    {
        m_gpu->downloadField(const_cast<float*>(m_upload_buf.data()));
        const int V = m_L * m_L;
        for (int i = 0; i < V; ++i)
            const_cast<SiteField<double, 2>&>(m_field)[i]
                = static_cast<double>(m_upload_buf[i]);
    }
}

void Phi4_2DScene::fillVolume(float* out) const
{
    syncFromGpu();
    const int V = m_L * m_L;
    for (int i = 0; i < V; ++i) out[i] = static_cast<float>(m_field[i]);
}

bool Phi4_2DScene::buildControlsUI()
{
    ImGui::Text("Lattice L = %d   V = %d", m_L, m_L * m_L);
    ImGui::TextDisabled("2D φ⁴ — Z₂ broken phase below m_c² ≈ −0.7 at λ = 1");
    ImGui::Separator();

    if (ImGui::CollapsingHeader("Monte Carlo", ImGuiTreeNodeFlags_DefaultOpen))
    {
        float m_sq = static_cast<float>(m_m_sq);
        if (ImGui::SliderFloat("m²", &m_sq, -1.5f, 1.5f, "%.4f"))
        {
            m_m_sq = m_sq;
            m_model.setMSquared(m_sq);
        }
        float lam = static_cast<float>(m_lambda);
        if (ImGui::SliderFloat("λ", &lam, 0.0f, 5.0f, "%.3f"))
        {
            m_lambda = lam;
            m_model.setLambda(lam);
        }
        float ss = static_cast<float>(m_step_size);
        if (ImGui::SliderFloat("step σ", &ss, 0.05f, 4.0f, "%.3f"))
        {
            m_step_size = ss;
            m_model.setStepSize(ss);
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
            ImGui::SliderInt("sweeps/frame", &m_sweeps_per_frame, 1, 20);
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
    ImGui::Text("⟨φ⟩  = %+.5f", obs::mean_field(m_field));
    ImGui::Text("⟨|φ|⟩ = %.5f",  obs::abs_mean_field(m_field));
    ImGui::Text("avg %.2f ms/frame (%.1f FPS)",
                1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

    ImGui::Separator();
    return ImGui::Button("← Back to menu");
}

} // namespace lqft::vis
