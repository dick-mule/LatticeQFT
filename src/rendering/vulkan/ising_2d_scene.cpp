#include "ising_2d_scene.hpp"

#include "gpu_ising_sweeper.hpp"

#include <imgui.h>

#include <stdexcept>

namespace lqft::vis
{

Ising2DScene::Ising2DScene(int L, std::uint64_t seed, const VulkanContext* ctx)
    : m_L(L)
    , m_lattice(Lattice<2>::cube(L))
    , m_field(m_lattice)
    , m_rng(seed)
    , m_model(m_lattice, m_beta)
    , m_sweeper(m_model)
{
    resetHot(seed);
    if (ctx && ctx->device)
    {
        try
        {
            m_gpu = std::make_unique<GpuIsingSweeper>(
                ctx->phys, ctx->device, ctx->queue,
                ctx->queue_family, ctx->cmd_pool,
                L, ctx->shader_dir, seed);
            m_gpu->uploadSpins(m_field.data());
            m_use_gpu = true;
        }
        catch (const std::exception&) { m_gpu.reset(); m_use_gpu = false; }
    }
}

Ising2DScene::~Ising2DScene() = default;

void Ising2DScene::step()
{
    if (m_paused) return;
    if (m_use_gpu && m_gpu)
        m_gpu->sweep(m_gpu_sweeps_per_frame, m_beta);
    else
        m_sweeper.sweepN(m_field, m_rng, m_sweeps_per_frame);
}

void Ising2DScene::singleStep()
{
    if (m_use_gpu && m_gpu) m_gpu->sweep(1, m_beta);
    else                     m_sweeper.sweep(m_field, m_rng);
}

void Ising2DScene::resetHot(std::uint64_t seed)
{
    m_rng = Rng(seed);
    ising::IsingModel<2>::hot(m_field, m_rng);
    m_sweeper.resetCounters();
    if (m_gpu) { m_gpu->reseedRng(seed); m_gpu->uploadSpins(m_field.data()); }
}

void Ising2DScene::resetCold()
{
    ising::IsingModel<2>::cold(m_field);
    m_sweeper.resetCounters();
    if (m_gpu) m_gpu->uploadSpins(m_field.data());
}

void Ising2DScene::syncFromGpu() const
{
    if (m_use_gpu && m_gpu)
        m_gpu->downloadSpins(const_cast<std::int8_t*>(m_field.data()));
}

void Ising2DScene::fillVolume(float* out) const
{
    syncFromGpu();
    const int V = m_L * m_L;
    for (int i = 0; i < V; ++i)
        out[i] = static_cast<float>(m_field[i]); // ±1
}

bool Ising2DScene::buildControlsUI()
{
    ImGui::Text("Lattice L = %d   V = %d", m_L, m_L * m_L);
    ImGui::TextDisabled("2D Ising — Onsager β_c = ½ ln(1+√2) ≈ 0.4407");
    ImGui::Separator();

    if (ImGui::CollapsingHeader("Monte Carlo", ImGuiTreeNodeFlags_DefaultOpen))
    {
        float beta = static_cast<float>(m_beta);
        if (ImGui::SliderFloat("beta", &beta, 0.10f, 0.80f, "%.4f"))
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
                ImGui::SliderInt("GPU sweeps/frame", &m_gpu_sweeps_per_frame, 1, 500);
        }
        else
        {
            ImGui::TextDisabled("(GPU compute unavailable; CPU only)");
        }
        if (!m_use_gpu)
            ImGui::SliderInt("sweeps/frame", &m_sweeps_per_frame, 1, 50);
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
    ImGui::Text("⟨|m|⟩ = %.5f", obs::abs_magnetization(m_field));
    ImGui::Text("avg %.2f ms/frame (%.1f FPS)",
                1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

    ImGui::Separator();
    return ImGui::Button("← Back to menu");
}

} // namespace lqft::vis
