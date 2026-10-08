#include "polyakov_scene.hpp"

#include "../../smearing/ape.hpp"
#include "../../smearing/stout.hpp"
#include "gpu_su2_sweeper.hpp"

#include <imgui.h>

#include <algorithm>
#include <array>
#include <cmath>

namespace lqft::vis
{

PolyakovScene::PolyakovScene(int L, std::uint64_t seed,
                             const VulkanContext* gpu_ctx)
    : m_L(L)
    , m_lattice(Lattice<3>::cube(L))
    , m_field(m_lattice)
    , m_rng(seed)
    , m_model(m_lattice, m_beta, /*step*/0.4)
    , m_sweeper(m_model)
    , m_smeared_field(m_lattice)
    , m_smear_scratch(m_lattice)
    , m_sum_C(static_cast<std::size_t>(L) * L, 0.0)
    , m_p_field(static_cast<std::size_t>(L) * L, 0.0)
{
    resetHot(seed);
    if (gpu_ctx && gpu_ctx->device)
    {
        try
        {
            m_gpu = std::make_unique<GpuSU2Sweeper>(
                gpu_ctx->phys, gpu_ctx->device, gpu_ctx->queue,
                gpu_ctx->queue_family, gpu_ctx->cmd_pool,
                L, /*Dim*/3, gpu_ctx->shader_dir, seed);
            m_gpu->uploadField(m_field);
            m_use_gpu = true;
        }
        catch (const std::exception&)
        {
            m_gpu.reset();
            m_use_gpu = false;
        }
    }
}

PolyakovScene::~PolyakovScene() = default;

void PolyakovScene::resetHot(std::uint64_t seed)
{
    m_rng = Rng(seed);
    su2_model::SU2Model<3>::hot(m_field, m_rng);
    m_sweeper.resetCounters();
    if (m_gpu)
    {
        m_gpu->reseedRng(seed);
        m_gpu->uploadField(m_field);
    }
    resetStats();
}

void PolyakovScene::resetCold()
{
    su2_model::SU2Model<3>::cold(m_field);
    m_sweeper.resetCounters();
    if (m_gpu) m_gpu->uploadField(m_field);
    resetStats();
}

void PolyakovScene::resetStats()
{
    m_n_measurements = 0;
    m_sum_p_global   = 0.0;
    m_sum_p_abs      = 0.0;
    m_sum_p_sq       = 0.0;
    std::fill(m_sum_C.begin(), m_sum_C.end(), 0.0);
    m_frames_since_measure = 0;
}

void PolyakovScene::step()
{
    if (m_paused) return;
    if (m_use_gpu && m_gpu)
    {
        m_gpu->sweep(m_gpu_sweeps_per_frame, m_beta, /*step*/0.4);
        m_gpu->downloadField(m_field);
    }
    else if (m_use_heatbath)
        su2_model::heatBathSweepN(m_model, m_field, m_rng, m_sweeps_per_frame);
    else
        m_sweeper.sweepN(m_field, m_rng, m_sweeps_per_frame);

    ++m_frames_since_measure;
    if (m_frames_since_measure >= m_measure_period)
    {
        m_frames_since_measure = 0;
        measurePolyakov();
    }
}

void PolyakovScene::prepareSmearedField() const
{
    m_smeared_field = m_field;
    if (m_smear_iters > 0)
    {
        const std::array<bool, 3> spatial = { true, true, false };
        switch (m_smear_method)
        {
        case SmearingMethod::APE:
            smearing::apeSmearN<3>(m_lattice,
                                   m_smeared_field, m_smear_scratch,
                                   static_cast<double>(m_smear_alpha),
                                   m_smear_iters, spatial);
            break;
        case SmearingMethod::Stout:
            smearing::stoutSmearN<3>(m_lattice,
                                     m_smeared_field, m_smear_scratch,
                                     static_cast<double>(m_smear_rho),
                                     m_smear_iters, spatial);
            break;
        }
    }
}

double PolyakovScene::polyakovAt(int x, int y,
                                 const LinkField<su2::Element, 3>& U) const
{
    su2::Element P = su2::Element::identity();
    for (int t = 0; t < m_L; ++t)
    {
        const int site = m_lattice.siteIndex({ x, y, t });
        P = su2::multiply(P, U(site, /*ẑ-direction*/2));
    }
    return P.s; // ½ tr P for SU(2)
}

void PolyakovScene::measurePolyakov()
{
    prepareSmearedField();
    const int L = m_L;

    // Pass 1 — compute p(x, y) on the spatial slice.
    for (int y = 0; y < L; ++y)
        for (int x = 0; x < L; ++x)
            m_p_field[static_cast<std::size_t>(y * L + x)]
                = polyakovAt(x, y, m_smeared_field);

    // Pass 2 — translation-averaged correlator C(Δx, Δy) and global ⟨p⟩.
    double sum_p = 0.0;
    for (int y0 = 0; y0 < L; ++y0)
        for (int x0 = 0; x0 < L; ++x0)
        {
            const double p0 = m_p_field[static_cast<std::size_t>(y0 * L + x0)];
            sum_p += p0;
            for (int dy = 0; dy < L; ++dy)
                for (int dx = 0; dx < L; ++dx)
                {
                    const int x = (x0 + dx) % L;
                    const int y = (y0 + dy) % L;
                    const double p1
                        = m_p_field[static_cast<std::size_t>(y * L + x)];
                    m_sum_C[static_cast<std::size_t>(dy * L + dx)] += p0 * p1;
                }
        }
    const double p_bar = sum_p / static_cast<double>(L * L);
    m_sum_p_global += p_bar;
    m_sum_p_abs    += std::abs(p_bar);
    m_sum_p_sq     += p_bar * p_bar;
    ++m_n_measurements;
}

void PolyakovScene::fillVolume(float* out) const
{
    const int        L  = m_L;
    const std::size_t V2 = static_cast<std::size_t>(L) * L;
    const std::size_t V3 = V2 * static_cast<std::size_t>(L);
    if (m_n_measurements == 0)
    {
        // Placeholder: a faint Gaussian glow at the centered origin so the
        // user actually SEES the volume render before the first MC
        // measurement arrives — otherwise an all-zero buffer reads as
        // "where's the cube?".
        for (int z = 0; z < L; ++z)
            for (int y = 0; y < L; ++y)
                for (int x = 0; x < L; ++x)
                {
                    const double dx = (x - L / 2);
                    const double dy = (y - L / 2);
                    const double r2 = dx * dx + dy * dy;
                    out[static_cast<std::size_t>((z * L + y) * L + x)]
                        = static_cast<float>(0.25 * std::exp(-r2 / 8.0));
                }
        return;
    }
    const double inv_N   = 1.0 / static_cast<double>(m_n_measurements);
    const double inv_V2  = 1.0 / static_cast<double>(V2);
    const double p_mean  = m_sum_p_global * inv_N;

    // Build the centered & normalized 2D correlator slice once, then
    // replicate it down the ẑ-axis. Centering = roll by L/2 in both
    // dimensions so the peak (Δ = 0, which on a periodic lattice is also
    // the four corners) lands in the middle of the rendered volume; that
    // matches the BSS flux-tube scene's visual convention and avoids
    // the disorienting four-corner-wraparound look of a raw periodic
    // correlator.
    std::vector<float> slice(V2, 0.0f);
    double f_max = 0.0;
    for (std::size_t i = 0; i < V2; ++i)
    {
        const double C = m_sum_C[i] * inv_N * inv_V2 - p_mean * p_mean;
        if (C > f_max) f_max = C;
    }
    const double norm  = (f_max > 1e-12) ? (1.0 / f_max) : 0.0;
    const double scale = static_cast<double>(m_amplification) * 0.01;
    // C(Δ) decays exponentially (~ exp(−σ·L_t·R)) while the volume renderer
    // maps voxel values linearly to opacity. Without correction, only the
    // single-voxel peak clears the opacity threshold and you see a thin
    // sliver. Apply a fractional power so the exponential halo around the
    // peak survives the [0,1] mapping — this is just a log-scale plot, made
    // explicit so the physics reads correctly.
    constexpr double kGamma = 0.35;
    for (int dy = 0; dy < L; ++dy)
    {
        const int dy_centered = (dy + L / 2) % L;
        for (int dx = 0; dx < L; ++dx)
        {
            const int dx_centered = (dx + L / 2) % L;
            const std::size_t src_idx
                = static_cast<std::size_t>(dy * L + dx);
            const double C
                = m_sum_C[src_idx] * inv_N * inv_V2 - p_mean * p_mean;
            const double val = std::max(0.0, C) * norm;          // ∈ [0, 1]
            const double boosted = std::pow(val, kGamma);         // log-scale-ish
            const std::size_t dst_idx
                = static_cast<std::size_t>(dy_centered * L + dx_centered);
            slice[dst_idx] = static_cast<float>(scale * boosted);
        }
    }
    // Extrude along x̂ — the same axis the BSS scene uses for the static-
    // QQ̄ separation. The Polyakov correlator C(Δx, Δy) has no preferred
    // direction (it's a translation-averaged radial decay), so the choice
    // of extrusion axis is cosmetic, but matching BSS makes the side-by-
    // side comparison of the two observables read at a glance. The bright
    // peak now appears at (x, L/2, L/2) for every x, drawing a horizontal
    // tube along the x-axis just like BSS.
    for (int z = 0; z < L; ++z)
        for (int y = 0; y < L; ++y)
            for (int x = 0; x < L; ++x)
            {
                const std::size_t dst
                    = static_cast<std::size_t>((z * L + y) * L + x);
                out[dst] = slice[static_cast<std::size_t>(z * L + y)];
            }
}

bool PolyakovScene::buildControlsUI()
{
    ImGui::Text("Lattice L = %d   V₂ = %d²", m_L, m_L);
    ImGui::TextDisabled(
        "3D SU(2) Polyakov-loop correlator. C(Δ) = ⟨P(0) P(Δ)⟩ − ⟨P⟩²\n"
        "with P(x, y) = (1/2) tr ∏ₜ U_ẑ(x, y, t) (closed by ẑ-periodicity).\n"
        "Volume shows C(Δx, Δy) extruded along x̂ (to match the BSS scene's\n"
        "tube orientation) and centered at the volume midpoint. Bright tube\n"
        "along x = charge axis, radial decay perpendicular to it. Sidesteps\n"
        "the W(R,T) area-law noise — same physics as BSS, no noise floor.");
    ImGui::Separator();

    if (ImGui::CollapsingHeader("Monte Carlo", ImGuiTreeNodeFlags_DefaultOpen))
    {
        float beta = static_cast<float>(m_beta);
        if (ImGui::SliderFloat("beta", &beta, 0.5f, 12.0f, "%.3f"))
        {
            m_beta = beta;
            m_model.setBeta(beta);
            resetStats();
        }
        if (m_gpu)
        {
            bool use_gpu = m_use_gpu;
            if (ImGui::Checkbox("GPU Metropolis (vs CPU heat-bath)", &use_gpu))
            {
                m_use_gpu = use_gpu;
                resetStats();
            }
            if (m_use_gpu)
                ImGui::SliderInt("GPU sweeps/frame", &m_gpu_sweeps_per_frame, 1, 200);
        }
        if (!m_use_gpu)
        {
            ImGui::SliderInt("sweeps/frame", &m_sweeps_per_frame, 1, 10);
            ImGui::Checkbox("heat-bath", &m_use_heatbath);
        }
        ImGui::Checkbox("paused", &m_paused);
        ImGui::SameLine();
        if (ImGui::Button("reset hot"))
            resetHot(static_cast<std::uint64_t>(ImGui::GetTime() * 1e6));
        ImGui::SameLine();
        if (ImGui::Button("reset cold")) resetCold();
    }

    if (ImGui::CollapsingHeader("Spatial-link smearing",
                                 ImGuiTreeNodeFlags_DefaultOpen))
    {
        bool smear_changed = false;
        if (ImGui::SliderInt("smearing iterations", &m_smear_iters, 0, 40))
            smear_changed = true;

        int method = static_cast<int>(m_smear_method);
        const char* methods[] = { "APE", "stout (Morningstar-Peardon)" };
        if (ImGui::Combo("smearing method", &method, methods, 2))
        {
            m_smear_method = static_cast<SmearingMethod>(method);
            smear_changed  = true;
        }
        if (m_smear_method == SmearingMethod::APE)
        {
            if (ImGui::SliderFloat("α (APE)", &m_smear_alpha, 0.0f, 1.0f, "%.2f"))
                smear_changed = true;
        }
        else
        {
            if (ImGui::SliderFloat("ρ (stout)", &m_smear_rho, 0.0f, 0.3f, "%.3f"))
                smear_changed = true;
        }
        if (smear_changed) resetStats();

        ImGui::SliderInt("measure every N MC steps", &m_measure_period, 1, 16);
        ImGui::TextDisabled(
            "Spatial (x̂, ŷ) links only; temporal ẑ links are what make up\n"
            "the Polyakov loop and stay untouched. Stout is differentiable\n"
            "and tends to give slightly cleaner signal at the same iter\n"
            "count; APE is the historical workhorse.");
    }

    if (ImGui::CollapsingHeader("Display", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::SliderFloat("amplification", &m_amplification,
                           1.0f, 2000.0f, "%.1f");
    }

    ImGui::Separator();
    ImGui::Text("N_measurements = %lld", m_n_measurements);

    if (m_n_measurements > 0)
    {
        const double inv_N  = 1.0 / static_cast<double>(m_n_measurements);
        const double inv_V2 = 1.0 / static_cast<double>(m_L * m_L);
        const double p_mean = m_sum_p_global * inv_N;
        ImGui::Text("⟨P⟩ = %+.5f   (Z₂-symmetric ⟹ 0 in confined phase)",
                    p_mean);
        // Finite-volume order parameter and its susceptibility, as in the 4D
        // deconfinement scans of the CLI (--nt): the symmetric phase tunnels between
        // the two Z₂ sectors, so ⟨P̄⟩ averages to zero while ⟨|P̄|⟩ ~ 1/√V; in the
        // deconfined phase ⟨|P̄|⟩ is O(1) and χ_P peaks at β_c(N_t).
        const double p_abs = m_sum_p_abs * inv_N;
        const double chi_P = static_cast<double>(m_L * m_L) * (m_sum_p_sq * inv_N - p_abs * p_abs);
        ImGui::Text("⟨|P̄|⟩ = %.5f   χ_P = V₂(⟨P̄²⟩ − ⟨|P̄|⟩²) = %.4f", p_abs, chi_P);
        ImGui::TextDisabled(
            "⟨|P̄|⟩ ~ 1/√V₂ confined, O(1) deconfined; χ_P peaks at β_c(N_t).\n"
            "Here N_t = L (cold, confined). The 4D scans live in the CLI (--nt).");

        // Print V(R) = −(1/L_t) log C(R) along R = (R, 0) for R ∈ [1, L/2].
        ImGui::TextDisabled("Static potential V(R) along (R, 0):");
        const double inv_Lt = 1.0 / static_cast<double>(m_L);
        for (int R = 1; R <= m_L / 2; ++R)
        {
            const double C
                = m_sum_C[static_cast<std::size_t>(R)] * inv_N * inv_V2
                - p_mean * p_mean;
            if (C > 1e-12)
            {
                const double V = -inv_Lt * std::log(C);
                ImGui::Text("  R = %2d   C = %+.4e   V(R) = %+.4f", R, C, V);
            }
            else
            {
                ImGui::Text("  R = %2d   C = %+.4e   (below noise)", R, C);
            }
        }
    }
    ImGui::Text("avg %.2f ms/frame (%.1f FPS)",
                1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

    ImGui::Separator();
    return ImGui::Button("← Back to menu");
}

} // namespace lqft::vis
