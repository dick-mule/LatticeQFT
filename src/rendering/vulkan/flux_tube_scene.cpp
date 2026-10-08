#include "flux_tube_scene.hpp"

#include "../../smearing/ape.hpp"
#include "gpu_su2_sweeper.hpp"

#include <imgui.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace lqft::vis
{

FluxTubeScene::FluxTubeScene(int L, std::uint64_t seed,
                             const VulkanContext* gpu_ctx)
    : m_L(L)
    , m_V(static_cast<long long>(L) * L * L)
    , m_lattice(Lattice<3>::cube(L))
    , m_field(m_lattice)
    , m_rng(seed)
    , m_model(m_lattice, m_beta, /*step*/0.4)
    , m_sweeper(m_model)
    , m_smeared_field(m_lattice)
    , m_smear_scratch(m_lattice)
    , m_sum_W_rho(static_cast<std::size_t>(m_V), 0.0)
    , m_rho_field(static_cast<std::size_t>(m_V), 0.0)
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

FluxTubeScene::~FluxTubeScene() = default;

void FluxTubeScene::resetHot(std::uint64_t seed)
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

void FluxTubeScene::resetCold()
{
    su2_model::SU2Model<3>::cold(m_field);
    m_sweeper.resetCounters();
    if (m_gpu) m_gpu->uploadField(m_field);
    resetStats();
}

void FluxTubeScene::resetStats()
{
    m_n_measurements = 0;
    m_sum_W_loops    = 0.0;
    m_sum_rho_mean   = 0.0;
    std::fill(m_sum_W_rho.begin(), m_sum_W_rho.end(), 0.0);
}

// ----------------------------------------------------------------------------
// MC + measurement.
// ----------------------------------------------------------------------------

void FluxTubeScene::step()
{
    if (m_paused) return;
    if (m_use_gpu && m_gpu)
    {
        // GPU runs many Metropolis sweeps; the lower per-sweep efficiency
        // (vs heat-bath) is more than paid back by the parallel speedup.
        m_gpu->sweep(m_gpu_sweeps_per_frame, m_beta, /*step_size*/0.4);
        m_gpu->downloadField(m_field);
    }
    else if (m_use_heatbath)
    {
        su2_model::heatBathSweepN(m_model, m_field, m_rng, m_sweeps_per_frame);
    }
    else
    {
        m_sweeper.sweepN(m_field, m_rng, m_sweeps_per_frame);
    }

    // Smearing + Wilson-loop measurement is the per-frame cost driver when
    // APE is on. Skip measurement on the in-between frames so the MC chain
    // keeps churning at full FPS while the user trades off sampling rate
    // vs framerate via m_measure_period.
    ++m_frames_since_measure;
    if (m_frames_since_measure >= m_measure_period)
    {
        m_frames_since_measure = 0;
        measureCurrentConfig();
    }
}

void FluxTubeScene::computeRhoField() const
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
                        su2::Element P = su2::multiply(m_field(site, mu),
                                                       m_field(s_mu, nu));
                        P = su2::multiply(P, su2::dagger(m_field(s_nu, mu)));
                        P = su2::multiply(P, su2::dagger(m_field(site, nu)));
                        rho += 1.0 - P.s;
                    }
                m_rho_field[static_cast<std::size_t>((z * L + y) * L + x)] = rho;
            }
}

double FluxTubeScene::computeWilsonLoop(
    int                                y_0,
    const LinkField<su2::Element, 3>&  U) const
{
    // Loop in the (x, z) plane starting at corner (0, y_0, 0): R steps in x̂,
    // T steps in ẑ, R steps in −x̂, T steps in −ẑ.
    const int corner = m_lattice.siteIndex({ 0, y_0, 0 });
    return su2_model::wilsonLoopAt(m_lattice, U, corner,
                                   /*mu*/0, /*nu*/2, m_R, m_T);
}

void FluxTubeScene::prepareSmearedField() const
{
    // Snapshot the current gauge field, then run `m_smear_iters` APE steps
    // restricted to the spatial directions x̂ and ŷ. The temporal (ẑ) links
    // are left untouched so the static-charge transfer matrix is preserved.
    m_smeared_field = m_field;
    if (m_smear_iters > 0 && m_smear_alpha > 0.0f)
    {
        const std::array<bool, 3> spatial = { true, true, false };
        smearing::apeSmearN<3>(m_lattice,
                               m_smeared_field, m_smear_scratch,
                               static_cast<double>(m_smear_alpha),
                               m_smear_iters,
                               spatial);
    }
}

void FluxTubeScene::measureCurrentConfig()
{
    computeRhoField();           // ρ uses the unsmeared field (local probe).
    prepareSmearedField();       // W uses spatially smeared links.

    const int L = m_L;
    double sum_W_this_config = 0.0;

    // For each y_0, measure W(y_0) and accumulate Σ_{x,Δy,z} W·ρ(x, y_0+Δy, z).
    // Equivalently: store at relative index (x, Δy = y − y_0 mod L, z).
    for (int y_0 = 0; y_0 < L; ++y_0)
    {
        const double W = computeWilsonLoop(y_0, m_smeared_field);
        sum_W_this_config += W;
        for (int z = 0; z < L; ++z)
            for (int y = 0; y < L; ++y)
            {
                const int Δy = ((y - y_0) % L + L) % L;
                for (int x = 0; x < L; ++x)
                {
                    const std::size_t src = static_cast<std::size_t>((z * L + y) * L + x);
                    const std::size_t dst = static_cast<std::size_t>((z * L + Δy) * L + x);
                    m_sum_W_rho[dst] += W * m_rho_field[src];
                }
            }
    }

    double rho_total = 0.0;
    for (std::size_t s = 0; s < static_cast<std::size_t>(m_V); ++s)
        rho_total += m_rho_field[s];
    const double rho_mean_this = rho_total / static_cast<double>(m_V);

    m_sum_W_loops  += sum_W_this_config;
    m_sum_rho_mean += rho_mean_this;
    ++m_n_measurements;
}

// ----------------------------------------------------------------------------
// Volume fill (connected correlator at the current accumulators).
// ----------------------------------------------------------------------------

void FluxTubeScene::fillVolume(float* out) const
{
    const std::size_t V = static_cast<std::size_t>(m_V);
    if (m_n_measurements == 0)
    {
        std::fill(out, out + V, 0.0f);
        return;
    }
    const double inv_N   = 1.0 / static_cast<double>(m_n_measurements);
    const double avg_W   = m_sum_W_loops  * inv_N / static_cast<double>(m_L);
    const double avg_rho = m_sum_rho_mean * inv_N;
    const double inv_W   = (std::abs(avg_W) < 1e-12) ? 0.0 : (1.0 / avg_W);

    // First pass: compute the raw connected correlator f(x), and track its
    // peak so we can normalize the volume to [0, 1] before applying the
    // user's amplification. Without this, switching smearing on changes the
    // dynamic range of f by orders of magnitude and the colormap saturates
    // across the entire volume.
    double f_max = 0.0;
    for (std::size_t s = 0; s < V; ++s)
    {
        const double avg_W_rho = m_sum_W_rho[s] * inv_N / static_cast<double>(m_L);
        const double f         = std::max(0.0, avg_W_rho * inv_W - avg_rho);
        if (f > f_max) f_max = f;
    }
    const double norm  = (f_max > 1e-12) ? (1.0 / f_max) : 0.0;
    const double scale = static_cast<double>(m_amplification) * 0.02; // unit-area knob

    // Translate by (L/2 − R/2, L/2, L/2 − T/2) so the QQ̄ pair lands at the
    // volume center instead of the (Δy=0, z=0) corner — that way this
    // scene's tube and the Polyakov scene's tube overlay on the same
    // screen position for direct comparison.
    const int L      = m_L;
    const int x_off  = (L - m_R) / 2;
    const int y_off  = L / 2;
    const int z_off  = (L - m_T) / 2;
    for (int z = 0; z < L; ++z)
        for (int y = 0; y < L; ++y)
            for (int x = 0; x < L; ++x)
            {
                const int xs = (x - x_off + L) % L;
                const int ys = (y - y_off + L) % L;
                const int zs = (z - z_off + L) % L;
                const std::size_t src = static_cast<std::size_t>((zs * L + ys) * L + xs);
                const double avg_W_rho
                    = m_sum_W_rho[src] * inv_N / static_cast<double>(m_L);
                const double f
                    = std::max(0.0, avg_W_rho * inv_W - avg_rho);
                out[static_cast<std::size_t>((z * L + y) * L + x)]
                    = static_cast<float>(scale * f * norm);
            }
}

// ----------------------------------------------------------------------------
// ImGui controls.
// ----------------------------------------------------------------------------

bool FluxTubeScene::buildControlsUI()
{
    ImGui::Text("Lattice L = %d   V = %lld", m_L, m_V);
    ImGui::TextUnformatted("3D SU(2) Bali–Schilling–Schlichter flux-tube correlator");
    ImGui::TextDisabled(
        "Connected correlator  f(x) = ⟨W·ρ(x)⟩ / ⟨W⟩ − ⟨ρ⟩,  averaged over\n"
        "y-translations of the Wilson loop.");
    if (ImGui::CollapsingHeader("Why this scene mostly looks like noise"))
    {
        ImGui::PushStyleColor(ImGuiCol_Text,
                              ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
        ImGui::TextWrapped(
            "This is the *original* BSS observable from Bali, Schilling, "
            "Schlichter (1995). What you're seeing is exactly the problem "
            "they spent the next decade fighting:\n"
            "\n"
            "• ⟨W(R, T)⟩ ~ exp(−σ·R·T − μ·(R+T)). Exponentially small.\n"
            "\n"
            "• f(x) = ⟨W·ρ⟩/⟨W⟩ − ⟨ρ⟩ divides by that small ⟨W⟩, which "
            "amplifies noise in ⟨W·ρ⟩ by 1/⟨W⟩ ≈ 5×–100×. Per-voxel SE "
            "balloons accordingly.\n"
            "\n"
            "• At our sample rate (~10⁴ measurements), the tube signal is "
            "buried under noise of comparable magnitude. The whole volume "
            "lights up with random positive blobs because max(0, f) of a "
            "zero-mean noisy quantity gives positive bias everywhere.\n"
            "\n"
            "• Smearing the spatial links (the panel below) helps μ a lot "
            "but doesn't change σ — and overshoots into ⟨W⟩ → 1 if you "
            "push too far. The window where this measurement \"works\" is "
            "narrow.\n"
            "\n"
            "The Polyakov scene next door is the cleanest pedagogical fix: "
            "its observable has no extended T, so no area-law amplification "
            "in the denominator. The textbook fix for *keeping* BSS is "
            "Lüscher-Weisz multilevel sampling (2001), which reduces "
            "variance by exp(−c·K) instead of 1/√N — that's Round B in our "
            "roadmap.");
        ImGui::PopStyleColor();
    }
    ImGui::TextDisabled(
        "β = 6 in 3D SU(2) is the practical sweet spot — confining (a²σ ≈ \n"
        "0.25) but with ⟨W(4,2)⟩ above the noise floor; drop β to 2.4 to\n"
        "see the strong-coupling regime where the tube is even more deeply\n"
        "buried in the area-law suppression.");
    ImGui::Separator();

    if (ImGui::CollapsingHeader("Monte Carlo", ImGuiTreeNodeFlags_DefaultOpen))
    {
        float beta = static_cast<float>(m_beta);
        if (ImGui::SliderFloat("beta", &beta, 0.5f, 8.0f, "%.3f"))
        {
            m_beta = beta;
            m_model.setBeta(beta);
            resetStats(); // β change invalidates the running averages
        }

        if (m_gpu)
        {
            bool use_gpu = m_use_gpu;
            if (ImGui::Checkbox("GPU Metropolis (vs CPU heat-bath)", &use_gpu))
            {
                m_use_gpu = use_gpu;
                resetStats(); // change of algorithm shouldn't mix histories
            }
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
        if (ImGui::Button("reset hot"))
            resetHot(static_cast<std::uint64_t>(ImGui::GetTime() * 1e6));
        ImGui::SameLine();
        if (ImGui::Button("reset cold")) resetCold();
    }

    if (ImGui::CollapsingHeader("Wilson loop geometry",
                                 ImGuiTreeNodeFlags_DefaultOpen))
    {
        const int R_max = m_L / 2;
        const int T_max = m_L / 2;
        int R = m_R, T = m_T;
        bool geom_changed = false;
        if (ImGui::SliderInt("QQ̄ separation R (along x)", &R, 1, R_max))
        {
            m_R = R; geom_changed = true;
        }
        if (ImGui::SliderInt("loop time T (along z)", &T, 1, T_max))
        {
            m_T = T; geom_changed = true;
        }
        if (geom_changed) resetStats();
        ImGui::TextDisabled("Charges at (0, *, 0) and (R, *, 0); y-translation averaged.");
    }

    if (ImGui::CollapsingHeader("APE link smearing",
                                 ImGuiTreeNodeFlags_DefaultOpen))
    {
        bool smear_changed = false;
        if (ImGui::SliderInt("smearing iterations", &m_smear_iters, 0, 40))
            smear_changed = true;
        if (ImGui::SliderFloat("smearing α", &m_smear_alpha, 0.0f, 1.0f, "%.2f"))
            smear_changed = true;
        if (smear_changed) resetStats();
        ImGui::SliderInt("measure every N MC steps", &m_measure_period, 1, 16);
        ImGui::TextDisabled(
            "Spatial-only APE smearing of the gauge field before measuring W.\n"
            "Filters short-distance UV noise so the area law is visible in\n"
            "fewer MC samples. Temporal (ẑ) links untouched.\n"
            "\n"
            "GOLDILOCKS WARNING: BSS's connected correlator f(x) needs ⟨W⟩\n"
            "to be both above noise AND clearly below 1. Too little smearing\n"
            "→ ⟨W⟩ below noise floor (area-law-buried). Too much smearing\n"
            "→ ⟨W⟩ → 1 (loop ≈ identity, f(x) flattens to noise around 0).\n"
            "Sweet spot at β = 6, R = 4, T = 2: ⟨W⟩ ≈ 0.1–0.3. Watch the\n"
            "⟨W⟩ readout below and tune to stay in that band.\n"
            "Typical: α ≈ 0.5, iters 4–8. The Polyakov scene avoids this\n"
            "knob entirely — its observable is smearing-monotone.");
    }

    if (ImGui::CollapsingHeader("Display", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::SliderFloat("amplification", &m_amplification, 1.0f, 2000.0f, "%.1f");
        ImGui::TextDisabled(
            "Multiplies max(0, f(x)) before writing to the volume texture.\n"
            "Tune together with the Render-controls density-scale slider.");
    }

    ImGui::Separator();
    ImGui::Text("N_measurements = %lld", m_n_measurements);
    if (m_n_measurements > 0)
    {
        const double avg_W = m_sum_W_loops
                           / (static_cast<double>(m_L) * m_n_measurements);
        const double avg_rho = m_sum_rho_mean
                             / static_cast<double>(m_n_measurements);
        // Rough Wilson-loop SE from a unit-variance bound on ½ tr U_□.
        const double se_W = 1.0 / std::sqrt(
            static_cast<double>(m_L) * m_n_measurements);

        ImGui::Text("⟨W(R=%d, T=%d)⟩ = %.6f  (±%.1g)", m_R, m_T, avg_W, se_W);
        ImGui::Text("⟨ρ⟩            = %.6f", avg_rho);

        // Diagnose: if |⟨W⟩| is below the noise estimate, the connected
        // correlator denominator is essentially zero and the visualization
        // is dominated by noise. Tell the user directly instead of
        // letting them stare at a yellow blob.
        if (std::abs(avg_W) < 2.0 * se_W)
        {
            ImGui::TextColored(ImVec4(0.95f, 0.55f, 0.35f, 1.0f),
                "⟨W⟩ is below the noise floor at this (R, T) — the\n"
                "correlator f(x) = ⟨W·ρ⟩/⟨W⟩ − ⟨ρ⟩ is dominated by\n"
                "noise. Try smaller R/T (the loop falls exponentially\n"
                "in area), more smearing iterations, or wait for more\n"
                "samples.");
        }
    }
    if (ImGui::Button("Reset statistics")) resetStats();
    ImGui::Text("avg %.2f ms/frame (%.1f FPS)",
                1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

    ImGui::Separator();
    return ImGui::Button("← Back to menu");
}

} // namespace lqft::vis
