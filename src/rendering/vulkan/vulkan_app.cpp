#include "vulkan_app.hpp"

#include "camera.hpp"
#include "flux_tube_scene.hpp"
#include "polyakov_scene.hpp"
#include "heatmap_2d_renderer.hpp"
#include "ising_2d_scene.hpp"
#include "phi4_2d_scene.hpp"
#include "scene.hpp"
#include "schwinger_hmc_scene.hpp"
#include "schwinger_quenched_scene.hpp"
#include "su2_2d_scene.hpp"
#include "su2_3d_scene.hpp"
#include "u1_2d_scene.hpp"
#include "u1_3d_scene.hpp"
#include "volume_renderer.hpp"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include <algorithm>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <utility>

namespace lqft::vis
{

namespace
{

// Validation layers — debug only; enabled here if VK_LAYER_KHRONOS_validation
// is available on the system, silently dropped otherwise.
constexpr bool kEnableValidation = true;
constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";

void throwIfFailed(vk::Result r, const char* what)
{
    if (r != vk::Result::eSuccess)
    {
        throw std::runtime_error(std::string(what) + " failed: "
            + vk::to_string(r));
    }
}

bool validationLayerAvailable()
{
    auto layers = vk::enumerateInstanceLayerProperties();
    for (const auto& layer : layers)
        if (std::strcmp(layer.layerName, kValidationLayer) == 0) return true;
    return false;
}

} // anonymous namespace

VulkanApp::VulkanApp(int width, int height, std::string title, int lattice_L)
    : m_width(width), m_height(height), m_title(std::move(title))
    , m_lattice_L(lattice_L)
{
    initWindow();
    initVulkan();
    initImGui();

    m_camera = std::make_unique<Camera>();

    // Resolve shader directory relative to the executable so launch from
    // any cwd Just Works.
    std::filesystem::path exe = std::filesystem::current_path();
    if (const char* env = std::getenv("LATTICEQFT_SHADERS"); env && *env)
        exe = env;
    m_shader_dir =
        std::filesystem::exists(exe / "shaders")
            ? (exe / "shaders").string()
            : "shaders";

    // Renderers are built lazily on scene activation. The menu screen
    // doesn't need either.
    registerScenes();
}

void VulkanApp::registerScenes()
{
    m_scene_descriptors.push_back(SceneDescriptor{
        "3D compact U(1) gauge",
        "Wilson plaquette action  S = β Σ_□ (1 − cos θ_□)  on a 24³ lattice. "
        "Live volumetric ray-march of the per-site action density Σ_{μ<ν} "
        "(1 − cos θ_□). **GPU Metropolis** runs ~30 link sweeps per frame "
        "in parallel (toggle in the panel) vs CPU heat-bath / Metropolis. "
        "Drag β between 0.5 (strong coupling, bright random) and 5.0 (weak "
        "coupling, dim and uniform).",
        {
            "Watch the volume go from bright-random (strong coupling) to "
            "dim-smooth (weak coupling) as β increases. There's no phase "
            "transition in compact U(1) in 3D — it's a smooth crossover.",
            "Toggle 'GPU Metropolis' on/off: same physics, ~50× faster "
            "Monte Carlo chain on GPU.",
            "⟨S⟩ in the panel is the spatial average; it should equilibrate "
            "within a few hundred sweeps.",
        },
        [](const VulkanContext& ctx) {
            return std::make_unique<U1Scene3D>(24, 12345u, &ctx);
        }
    });
    m_scene_descriptors.push_back(SceneDescriptor{
        "3D SU(2) Yang-Mills",
        "Non-Abelian gauge theory: SU(2) link variables as unit quaternions, "
        "Wilson plaquette action  S = β Σ_□ (1 − ½ tr U_□)  on a 20³ lattice. "
        "Live volumetric ray-march of the per-site action density. "
        "**GPU Metropolis** parallel compute pipeline (default) vs CPU "
        "Kennedy-Pendleton heat-bath. Below β ≈ 2 the field is strongly "
        "fluctuating; above β ≈ 5 it smooths toward the continuum.",
        {
            "Same strong→weak coupling crossover as U(1), but the field has "
            "3 parameters per link (the SU(2) generators), so there's more "
            "noise per voxel at the same β.",
            "Non-Abelian means there's no gauge-invariant local 'direction' "
            "to read — only the local action density ½ tr U_□ is meaningful, "
            "which is what the volume shows.",
            "Try CPU heat-bath vs GPU Metropolis: heat-bath converges in "
            "~½ as many sweeps per unit β-change but the GPU dispatches "
            "many more sweeps per frame, so the chain mixes faster overall.",
        },
        [](const VulkanContext& ctx) {
            return std::make_unique<SU2Scene3D>(20, 67890u, &ctx);
        }
    });
    m_scene_descriptors.push_back(SceneDescriptor{
        "2D Ising (Onsager)",
        "Classic 2D Ising spins on a 64² periodic lattice. Diverging "
        "blue/yellow heatmap so the two ordered phases pop against the "
        "disordered background. **GPU Metropolis** runs ~50 site sweeps "
        "per frame; default β centered on Onsager β_c = ½ ln(1+√2) ≈ "
        "0.4407 — drag below 0.35 (disorder) or above 0.55 (one ordered "
        "domain wins).",
        {
            "The cleanest visible phase transition in the project: at β ≈ "
            "0.44 the lattice fractures into yellow/blue domains at every "
            "scale (diverging correlation length).",
            "Drag β to 0.30: paramagnet — yellow and blue noise, ⟨m⟩ ≈ 0.",
            "Drag β to 0.55: ferromagnet — one color (yellow or blue) wins "
            "the whole lattice within tens of seconds.",
            "Right at β_c, watch domains slowly anneal: small clusters get "
            "absorbed by big ones — that's the algorithm 'paying' the "
            "diverging autocorrelation time of local-update Monte Carlo.",
        },
        [](const VulkanContext& ctx) {
            return std::make_unique<Ising2DScene>(64, 1234u, &ctx);
        }
    });
    m_scene_descriptors.push_back(SceneDescriptor{
        "2D φ⁴ scalar field",
        "Real scalar  S = ½(∂φ)² + ½ m² φ² + (λ/24) φ⁴  on a 64² lattice. "
        "Diverging colormap so the Z₂ broken phase (one sign wins) reads "
        "cleanly. Drag m² below ≈ −0.7 at λ = 1 to enter the broken phase; "
        "**GPU Metropolis** with a fixed-σ proposal kernel.",
        {
            "m² > 0 (default): symmetric phase, field fluctuates around 0 "
            "(uniform gray-with-noise).",
            "m² ≈ −1, λ = 1: spontaneous Z₂ breaking — one sign wins a "
            "macroscopic region, then engulfs the lattice. Wait for it; "
            "the symmetry breaks via slow domain coarsening.",
            "After the field commits to a sign, watch domain walls slowly "
            "annihilate — that's the (1+1)D φ⁴ kink/anti-kink dynamics.",
        },
        [](const VulkanContext& ctx) {
            return std::make_unique<Phi4_2DScene>(64, 5678u, &ctx);
        }
    });
    m_scene_descriptors.push_back(SceneDescriptor{
        "2D compact U(1) gauge",
        "Wilson plaquette  S = β Σ_□ (1 − cos θ_□)  on a 64² lattice. "
        "Heatmap of the per-site action density (1 − cos θ_□). The 2D "
        "partition function factorizes, giving the exact Bessel-ratio "
        "⟨cos θ_□⟩ = I_1(β) / I_0(β). **GPU Metropolis** (same shader as "
        "the 3D scene, instantiated with Dim = 2) or CPU heat-bath.",
        {
            "No phase transition in 2D compact U(1) — smooth crossover only. "
            "Read ⟨cos θ_□⟩ in the panel and compare to the exact Bessel "
            "ratio I_1(β)/I_0(β) printed alongside.",
            "Strong coupling (β ≈ 0.5): heatmap is bright random. Weak "
            "coupling (β ≈ 5): nearly uniform dark.",
            "This 2D scene is the 'free training wheels' version of the 3D "
            "U(1) scene — same dynamics, much cheaper, exact reference.",
        },
        [](const VulkanContext& ctx) {
            return std::make_unique<U1_2DScene>(64, 9876u, &ctx);
        }
    });
    m_scene_descriptors.push_back(SceneDescriptor{
        "2D SU(2) Yang-Mills",
        "Non-Abelian gauge theory  S = β Σ_□ (1 − ½ tr U_□)  on a 48² "
        "lattice. Heatmap of the per-site action density. **GPU Metropolis** "
        "(same SU(2) compute shader, Dim = 2) or CPU Kennedy-Pendleton "
        "heat-bath.",
        {
            "Like 2D U(1) but non-Abelian: 3 parameters per link instead "
            "of 1, so the action density is noisier at the same β.",
            "Smooth crossover from strong (β ≈ 1) to weak (β ≈ 6) coupling. "
            "No phase transition in 2D.",
            "Compare CPU Kennedy-Pendleton heat-bath (exact conditional, "
            "100% acceptance) to GPU Metropolis (lower per-sweep mixing, "
            "way more sweeps per frame).",
        },
        [](const VulkanContext& ctx) {
            return std::make_unique<SU2_2DScene>(48, 13579u, &ctx);
        }
    });
    m_scene_descriptors.push_back(SceneDescriptor{
        "2D Schwinger (quenched)",
        "U(1) gauge field on a 32² lattice + stochastic measurement of the "
        "Wilson-Dirac condensate ⟨ψ̄ψ⟩ via Z₂-noise sources and conjugate "
        "gradient on D†D. Heatmap shows the gauge action density; side "
        "panel runs ⟨ψ̄ψ⟩(β, m). **GPU gauge sweep** (U(1) shader, Dim = 2) "
        "+ CPU CG measurement.",
        {
            "Look at the ⟨ψ̄ψ⟩(β, m) curve in the side panel. At m = 0.2 it "
            "should grow as β decreases (stronger coupling → more chiral "
            "symmetry breaking).",
            "Drop fermion mass: CG iterations/source grows ~1/m (panel "
            "shows it). The lattice IR scale ~1/L bounds the smallest "
            "Dirac eigenvalue, so on L=32 you should see ~3-6× growth "
            "from m=0.4 down to m=0.01 — not infinite, but visibly "
            "tracking the chiral limit.",
            "QUENCHED means there's no fermion back-reaction on the gauge "
            "chain — this systematically overestimates ⟨ψ̄ψ⟩. The HMC "
            "scene next door is the dynamical fix.",
            "Z₂ stochastic sources: more sources → less noise but slower. "
            "Defaults are balanced; bump if curves look jittery.",
        },
        [](const VulkanContext& ctx) {
            return std::make_unique<SchwingerQuenchedScene>(32, 31415u, &ctx);
        }
    });
    m_scene_descriptors.push_back(SceneDescriptor{
        "2D Schwinger (dynamical HMC)",
        "Full dynamical Schwinger model — gauge + pseudofermion + leapfrog "
        "HMC on a 10² lattice. Each trajectory refreshes the pseudofermion "
        "and momenta, integrates with a fermion force from a CG solve per "
        "leapfrog step, then Metropolis-accepts on ΔH. CPU only (HMC "
        "fermion force on GPU is its own project). Live readouts of HMC "
        "acceptance, ⟨ΔH⟩, and the dynamical-fermion condensate.",
        {
            "Four colored validation lines at the top: Wilson-Dirac apply, "
            "CG solve, full Schwinger force, and HMC trajectory ΔH. All "
            "green = the GPU HMC pipeline matches CPU end-to-end (to "
            "single-precision GPU rounding ~1e-4 to 1e-3).",
            "Toggle 'GPU HMC backend (vs CPU)': identical physics, but every "
            "leapfrog force evaluation runs CG + Wilson-Dirac + gauge force "
            "+ fermion force on GPU. Watch ⟨ψ̄ψ⟩ converge to the same value "
            "either way.",
            "HMC acceptance < 50% means HMC dt is too large (integrator "
            "drifts in energy). > 95% means dt is too small (wasted work). "
            "Tune to land near 70-85%.",
            "Drop fermion mass to 0.1: each CG inversion needs many more "
            "iterations (near-chiral D†D is ill-conditioned). The "
            "trajectory slows visibly.",
        },
        [](const VulkanContext& ctx) {
            return std::make_unique<SchwingerHMCScene>(10, 27182u, &ctx);
        }
    });
    m_scene_descriptors.push_back(SceneDescriptor{
        "3D SU(2) flux tube (Bali–Schilling–Schlichter)",
        "Static QQ̄ pair connected by a rectangular Wilson loop in the (x̂, ẑ) "
        "plane. The chromoelectric flux distribution is reconstructed from the "
        "*connected* correlator  f(x) = ⟨W·ρ(x)⟩/⟨W⟩ − ⟨ρ⟩  with translation "
        "averaging over the y-axis to reduce noise. **GPU Metropolis** runs "
        "30+ link sweeps per frame in parallel on the GPU compute pipeline — "
        "5-50× faster MC than CPU heat-bath; toggle in the panel. In the "
        "confining phase (β = 2.4 default) the field localizes into a tube "
        "of finite transverse width — turn amplification up to see early "
        "signal as samples accumulate.",
        {
            "WHY THIS SCENE MOSTLY LOOKS LIKE NOISE — and why that's the point. "
            "The BSS connected correlator f(x) = ⟨W·ρ⟩/⟨W⟩ − ⟨ρ⟩ amplifies "
            "noise in ⟨W·ρ⟩ by 1/⟨W⟩, which is exponentially small from the "
            "area law exp(−σ·R·T). At ~10⁴ measurements the tube signal is "
            "buried under noise of comparable magnitude — the whole volume "
            "lights up because max(0, f) of zero-mean noise has positive "
            "bias everywhere. The scene's panel has a 'Why this scene "
            "mostly looks like noise' section explaining the physics.",
            "This IS the original 1995 observable. It's the problem that "
            "drove a decade of variance-reduction methods (Lüscher-Weisz "
            "multilevel, HYP smearing, GEVP variational basis). The "
            "Polyakov scene next door shows what the same physics looks "
            "like with an observable that sidesteps the area-law-in-the-"
            "denominator pathology.",
            "GOLDILOCKS smearing for what little signal you can extract: "
            "⟨W⟩ readout in the panel should land around 0.1-0.3. Above "
            "~0.4, the loop ≈ identity and f(x) flattens completely; below "
            "~0.02, area-law-buried.",
            "DECONFINEMENT check: drop β to 1.0 → no tube. The chromo-"
            "electric field stays diffusely noisy because the static "
            "charges are no longer confined.",
            "WIDEN the tube: slide R up to 8. The tube length grows; its "
            "transverse width stays approximately fixed — that's the "
            "string-tension area law signature.",
            "APE smearing iters > 0: tube SHARPENS (UV noise filtered). If "
            "FPS tanks, bump 'measure every N MC steps' — chain keeps "
            "running, measurement throttles. Build release for full speed.",
            "Bali-Schilling-Schlichter (1995) measured this exact "
            "observable on SU(2) and SU(3) lattices; this scene is the "
            "qualitative version of their plots.",
        },
        [](const VulkanContext& ctx) {
            return std::make_unique<FluxTubeScene>(16, 24601u, &ctx);
        }
    });
    m_scene_descriptors.push_back(SceneDescriptor{
        "3D SU(2) Polyakov loop correlator",
        "Alternative to BSS that completely sidesteps area-law noise: "
        "P(x, y) = ½ tr ∏ₜ U_ẑ(x, y, t) is a single ẑ-wrapping loop per "
        "spatial site, gauge-invariant by lattice periodicity. The "
        "translation-averaged correlator C(Δ) = ⟨P(0)P(Δ)⟩ − ⟨P⟩² gives the "
        "static QQ̄ potential via V(R) = −(1/L_t) log C(R). 16³ lattice, "
        "GPU SU(2) Metropolis, choice of APE or stout smearing on the "
        "spatial links.",
        {
            "Volume shows C(Δx, Δy) (translation-averaged) extruded along "
            "ẑ — bright vertical tube at the centered origin, decaying "
            "radially. Confined phase: peak at origin, exponential decay "
            "rate = string tension σ. Deconfined: ⟨P⟩ ≠ 0 and C(Δ) "
            "plateaus at large Δ.",
            "Side panel prints V(R) = −(1/L_t) log C(R) for R = 1..L/2. "
            "Take ratios V(R+1)−V(R) along the table — at large R this "
            "approaches σ·a in lattice units.",
            "Try APE vs stout in the smearing combo. Both push C(R) up "
            "by reducing the link-self-energy contribution; stout is "
            "differentiable (used in HMC), APE is the historical workhorse.",
            "β = 6 keeps confinement clean while C(R) stays above noise "
            "for the first half-dozen R values. Drop β below 4 and the "
            "signal at large R drops off the cliff (closer to bulk noise "
            "floor); raise β above 8 and σ shrinks so V(R) gets flat.",
            "Compare to the BSS scene next door: same physics (static "
            "quark potential), totally different observable. This one "
            "has constant O(1/√N) noise; BSS has noise ~ exp(σRT) / √N.",
        },
        [](const VulkanContext& ctx) {
            return std::make_unique<PolyakovScene>(16, 11235u, &ctx);
        }
    });
}

void VulkanApp::rebuildVolumeRendererForL(int L)
{
    if (m_device) m_device.waitIdle();
    m_volume.reset();
    m_heatmap2d.reset();
    m_lattice_L = L;
    m_volume = std::make_unique<VolumeRenderer>(
        m_physical_device, m_device, m_queue, m_queue_family,
        m_command_pool, m_render_pass,
        static_cast<std::uint32_t>(L), m_shader_dir);
}

void VulkanApp::rebuildHeatmap2DRendererForL(int L)
{
    if (m_device) m_device.waitIdle();
    m_volume.reset();
    m_heatmap2d.reset();
    m_lattice_L = L;
    m_heatmap2d = std::make_unique<Heatmap2DRenderer>(
        m_physical_device, m_device, m_queue, m_queue_family,
        m_command_pool, m_render_pass,
        static_cast<std::uint32_t>(L), m_shader_dir);
}

void VulkanApp::tearDownActiveRenderers()
{
    if (m_device) m_device.waitIdle();
    m_volume.reset();
    m_heatmap2d.reset();
}

void VulkanApp::activateScene(std::unique_ptr<Scene> scene)
{
    if (!scene) return;
    // Snapshot the descriptor whose factory just produced this scene so the
    // in-scene "Things to look for" panel has access to the same tips the
    // menu page showed.
    if (m_selected_scene_idx >= 0
        && m_selected_scene_idx < static_cast<int>(m_scene_descriptors.size()))
    {
        const auto& d = m_scene_descriptors[
            static_cast<std::size_t>(m_selected_scene_idx)];
        m_active_scene_name        = d.name;
        m_active_scene_description = d.description;
        m_active_scene_tips        = d.tips;
    }
    else
    {
        m_active_scene_name.clear();
        m_active_scene_description.clear();
        m_active_scene_tips.clear();
    }

    const int  L         = scene->latticeSize();
    const bool wants_2d  = !scene->isVolumetric();

    if (wants_2d)
    {
        rebuildHeatmap2DRendererForL(L);
        m_volume_buffer.assign(static_cast<std::size_t>(L) * L, 0.0f);
    }
    else
    {
        rebuildVolumeRendererForL(L);
        m_volume_buffer.assign(
            static_cast<std::size_t>(L) * L * L, 0.0f);
    }
    m_active_is_2d = wants_2d;
    m_active_scene = std::move(scene);
    m_show_menu    = false;
}

void VulkanApp::returnToMenu()
{
    if (m_device) m_device.waitIdle();
    m_active_scene.reset();
    m_show_menu = true;
}

VulkanApp::~VulkanApp()
{
    cleanup();
}

void VulkanApp::run()
{
    mainLoop();
}

// ----------------------------------------------------------------------------
// initWindow / initVulkan / initImGui
// ----------------------------------------------------------------------------

void VulkanApp::initWindow()
{
    if (!glfwInit())
        throw std::runtime_error("glfwInit failed");
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    m_window = glfwCreateWindow(m_width, m_height, m_title.c_str(), nullptr, nullptr);
    if (!m_window)
        throw std::runtime_error("glfwCreateWindow failed");
    glfwSetWindowUserPointer(m_window, this);
    glfwSetFramebufferSizeCallback(m_window, &VulkanApp::framebufferResizeCallback);
    // NOTE: ImGui_ImplGlfw_InitForVulkan(install_callbacks=true) will install
    // chained callbacks for scroll / cursor / mouse-button. We register our
    // own *after* ImGui init so we still own the dispatch; we manually call
    // the ImGui callback at the top of each handler to keep its state alive.
}

void VulkanApp::framebufferResizeCallback(GLFWwindow* w, int, int)
{
    auto* self = static_cast<VulkanApp*>(glfwGetWindowUserPointer(w));
    if (self) self->m_framebuffer_resized = true;
}

void VulkanApp::scrollCallback(GLFWwindow* w, double xoff, double yoff)
{
    ImGui_ImplGlfw_ScrollCallback(w, xoff, yoff);
    if (ImGui::GetIO().WantCaptureMouse) return;
    auto* self = static_cast<VulkanApp*>(glfwGetWindowUserPointer(w));
    if (self && self->m_camera)
        self->m_camera->onScroll(static_cast<float>(yoff));
}

void VulkanApp::cursorPosCallback(GLFWwindow* w, double x, double y)
{
    ImGui_ImplGlfw_CursorPosCallback(w, x, y);
    auto* self = static_cast<VulkanApp*>(glfwGetWindowUserPointer(w));
    if (!self) return;

    const double dx = x - self->m_last_mouse_x;
    const double dy = y - self->m_last_mouse_y;
    self->m_last_mouse_x = x;
    self->m_last_mouse_y = y;

    if (ImGui::GetIO().WantCaptureMouse) return;
    if (self->m_mouse_left_down && self->m_camera)
        self->m_camera->onMouseDrag(static_cast<float>(dx),
                                    static_cast<float>(dy));
}

void VulkanApp::mouseButtonCallback(GLFWwindow* w, int button, int action, int mods)
{
    ImGui_ImplGlfw_MouseButtonCallback(w, button, action, mods);
    if (ImGui::GetIO().WantCaptureMouse) return;
    auto* self = static_cast<VulkanApp*>(glfwGetWindowUserPointer(w));
    if (!self) return;
    if (button == GLFW_MOUSE_BUTTON_LEFT)
        self->m_mouse_left_down = (action == GLFW_PRESS);
}

void VulkanApp::initVulkan()
{
    createInstance();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapchain();
    createImageViews();
    createRenderPass();
    createFramebuffers();
    createCommandPool();
    createCommandBuffers();
    createSyncObjects();
    createImGuiDescriptorPool();
}

void VulkanApp::initImGui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();

    // Replace the ASCII-only ProggyClean default with a system TTF that
    // covers Greek + math + arrow + bracket + superscript ranges. Without
    // this, every β / Σ / μ / ≈ / ⟨⟩ / □ / Δ / ² / → in the scene tips and
    // panel labels renders as a '?'.
    {
        ImFontGlyphRangesBuilder builder;
        builder.AddRanges(io.Fonts->GetGlyphRangesDefault());
        builder.AddRanges(io.Fonts->GetGlyphRangesGreek());
        const ImWchar extra_ranges[] = {
            0x00A0, 0x00FF,  // Latin-1 supplement (° × ÷ ± ¹ ² · ¬)
            0x0300, 0x036F,  // Combining diacritics (̄ ̂  — Q̄ ẑ x̂ ŷ)
            0x1E00, 0x1EFF,  // Latin Extended Additional (ẑ ẑ ŷ ŝ x̂ etc.)
            0x2000, 0x206F,  // General Punctuation (– — ' ' " " … • ‰ ′ ″)
            0x2070, 0x209F,  // Superscripts and subscripts (⁰ ¹ ² ³ ₀ ₁ ₂)
            0x2100, 0x214F,  // Letterlike symbols (ℏ ℝ ℂ ℤ ℕ)
            0x2190, 0x21FF,  // Arrows (→ ← ↔ ⇒)
            0x2200, 0x22FF,  // Mathematical operators (∂ ≤ ≥ ≠ ≈ √ ∑ ∏ · × ÷ ∈ ∇)
            0x2300, 0x23FF,  // Misc technical (⌈ ⌉ ⌊ ⌋ ⟦ ⟧)
            0x25A0, 0x25FF,  // Geometric shapes (□ ◇ ○ ▢ ■)
            0x27E0, 0x27FF,  // Misc math symbols A (⟨ ⟩ ⟦ ⟧)
            0x2980, 0x29FF,  // Misc math symbols B
            0,
        };
        builder.AddRanges(extra_ranges);
        static ImVector<ImWchar> ranges;
        ranges.clear();
        builder.BuildRanges(&ranges);

        ImFontConfig cfg{};
        cfg.OversampleH = 2;
        cfg.OversampleV = 2;
        cfg.PixelSnapH  = false;

        // PRIMARY: a clean monospace UI font (Latin + Greek). On macOS
        // SFNSMono / Menlo cover everything we need for body text, but
        // they're MISSING the rarer math angle brackets ⟨⟩ at U+27E8/9
        // and a few other math glyphs — those come from a fallback below.
        const char* font_candidates[] = {
            "/System/Library/Fonts/SFNSMono.ttf",
            "/System/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Geneva.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:\\Windows\\Fonts\\consola.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
        };
        ImFont* primary = nullptr;
        for (const char* path : font_candidates)
        {
            if (!std::filesystem::exists(path)) continue;
            primary = io.Fonts->AddFontFromFileTTF(
                path, /*size*/15.0f, &cfg, ranges.Data);
            if (primary) break;
        }
        if (!primary) io.Fonts->AddFontDefault();

        // FALLBACK: merge in a comprehensive Unicode font for the rare
        // math glyphs the primary font is missing. Apple Symbols + Arial
        // Unicode together cover essentially every mathematical-bracket /
        // operator / arrow in BMP. Order matters: the FIRST font in the
        // merge chain that has a given glyph wins, so primary stays the
        // visual default and the math fallbacks only fill in holes.
        ImFontConfig merge_cfg{};
        merge_cfg.MergeMode  = true;
        merge_cfg.OversampleH = 2;
        merge_cfg.OversampleV = 2;
        merge_cfg.PixelSnapH  = false;
        const char* merge_candidates[] = {
            "/System/Library/Fonts/Apple Symbols.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
        };
        for (const char* path : merge_candidates)
        {
            if (!std::filesystem::exists(path)) continue;
            (void)io.Fonts->AddFontFromFileTTF(
                path, /*size*/15.0f, &merge_cfg, ranges.Data);
        }
    }

    ImGui_ImplGlfw_InitForVulkan(m_window, /*install_callbacks*/ true);

    ImGui_ImplVulkan_InitInfo init_info{};
    init_info.Instance        = m_instance;
    init_info.PhysicalDevice  = m_physical_device;
    init_info.Device          = m_device;
    init_info.QueueFamily     = m_queue_family;
    init_info.Queue           = m_queue;
    init_info.PipelineCache   = VK_NULL_HANDLE;
    init_info.DescriptorPool  = m_imgui_desc_pool;
    init_info.RenderPass      = m_render_pass;
    init_info.Subpass         = 0;
    init_info.MinImageCount   = static_cast<std::uint32_t>(m_swapchain_images.size());
    init_info.ImageCount      = static_cast<std::uint32_t>(m_swapchain_images.size());
    init_info.MSAASamples     = VK_SAMPLE_COUNT_1_BIT;
    init_info.Allocator       = nullptr;
    init_info.CheckVkResultFn = nullptr;

    if (!ImGui_ImplVulkan_Init(&init_info))
        throw std::runtime_error("ImGui_ImplVulkan_Init failed");

    // ImGui v1.91+: fonts are uploaded automatically on first NewFrame, but
    // an explicit CreateFontsTexture forces it now so any error surfaces
    // before the first frame.
    if (!ImGui_ImplVulkan_CreateFontsTexture())
        throw std::runtime_error("ImGui_ImplVulkan_CreateFontsTexture failed");

    m_imgui_initialized = true;

    // Install our own chained callbacks AFTER ImGui's install pass so
    // we own the dispatch. We call the ImGui handler explicitly at the
    // top of each callback to keep its event state coherent.
    glfwSetScrollCallback     (m_window, &VulkanApp::scrollCallback);
    glfwSetCursorPosCallback  (m_window, &VulkanApp::cursorPosCallback);
    glfwSetMouseButtonCallback(m_window, &VulkanApp::mouseButtonCallback);
}

// ----------------------------------------------------------------------------
// Instance / surface / device
// ----------------------------------------------------------------------------

void VulkanApp::createInstance()
{
    vk::ApplicationInfo app_info{
        m_title.c_str(), VK_MAKE_VERSION(0, 1, 0),
        "LatticeQFT",    VK_MAKE_VERSION(0, 1, 0),
        VK_API_VERSION_1_2 };

    std::uint32_t glfw_count = 0;
    const char** glfw_exts   = glfwGetRequiredInstanceExtensions(&glfw_count);
    std::vector<const char*> extensions(glfw_exts, glfw_exts + glfw_count);

    vk::InstanceCreateFlags flags{};
#ifdef __APPLE__
    // MoltenVK needs the portability extension + flag.
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
#endif

    std::vector<const char*> layers;
    if (kEnableValidation && validationLayerAvailable())
        layers.push_back(kValidationLayer);

    vk::InstanceCreateInfo ci{};
    ci.flags                   = flags;
    ci.pApplicationInfo        = &app_info;
    ci.enabledLayerCount       = static_cast<std::uint32_t>(layers.size());
    ci.ppEnabledLayerNames     = layers.data();
    ci.enabledExtensionCount   = static_cast<std::uint32_t>(extensions.size());
    ci.ppEnabledExtensionNames = extensions.data();

    m_instance = vk::createInstance(ci);
}

void VulkanApp::createSurface()
{
    VkSurfaceKHR raw_surface = VK_NULL_HANDLE;
    if (glfwCreateWindowSurface(m_instance, m_window, nullptr, &raw_surface) != VK_SUCCESS)
        throw std::runtime_error("glfwCreateWindowSurface failed");
    m_surface = raw_surface;
}

void VulkanApp::pickPhysicalDevice()
{
    auto devices = m_instance.enumeratePhysicalDevices();
    if (devices.empty()) throw std::runtime_error("no Vulkan-capable GPU");

    for (const auto& dev : devices)
    {
        auto families = dev.getQueueFamilyProperties();
        for (std::uint32_t i = 0; i < families.size(); ++i)
        {
            const bool graphics = (families[i].queueFlags & vk::QueueFlagBits::eGraphics)
                                  == vk::QueueFlagBits::eGraphics;
            const bool present = dev.getSurfaceSupportKHR(i, m_surface) == VK_TRUE;
            if (graphics && present)
            {
                m_physical_device = dev;
                m_queue_family    = i;
                return;
            }
        }
    }
    throw std::runtime_error("no graphics+present queue family found");
}

void VulkanApp::createLogicalDevice()
{
    float priority = 1.0f;
    vk::DeviceQueueCreateInfo qci{};
    qci.queueFamilyIndex = m_queue_family;
    qci.queueCount       = 1;
    qci.pQueuePriorities = &priority;

    std::vector<const char*> dev_extensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
#ifdef __APPLE__
    dev_extensions.push_back("VK_KHR_portability_subset");
#endif

    vk::DeviceCreateInfo dci{};
    dci.queueCreateInfoCount    = 1;
    dci.pQueueCreateInfos       = &qci;
    dci.enabledExtensionCount   = static_cast<std::uint32_t>(dev_extensions.size());
    dci.ppEnabledExtensionNames = dev_extensions.data();

    m_device = m_physical_device.createDevice(dci);
    m_queue  = m_device.getQueue(m_queue_family, 0);
}

// ----------------------------------------------------------------------------
// Swapchain + image views + render pass + framebuffers
// ----------------------------------------------------------------------------

void VulkanApp::createSwapchain()
{
    auto caps    = m_physical_device.getSurfaceCapabilitiesKHR(m_surface);
    auto formats = m_physical_device.getSurfaceFormatsKHR(m_surface);
    auto modes   = m_physical_device.getSurfacePresentModesKHR(m_surface);

    // Format: prefer 32-bit sRGB BGRA, fall back to first available.
    vk::SurfaceFormatKHR chosen = formats[0];
    for (const auto& f : formats)
        if (f.format == vk::Format::eB8G8R8A8Srgb
            && f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
        { chosen = f; break; }

    // Present mode: prefer mailbox (low-latency vsync), fall back to FIFO (required).
    vk::PresentModeKHR present_mode = vk::PresentModeKHR::eFifo;
    for (auto m : modes)
        if (m == vk::PresentModeKHR::eMailbox) { present_mode = m; break; }

    vk::Extent2D extent = caps.currentExtent;
    if (extent.width == UINT32_MAX)
    {
        int w = 0, h = 0;
        glfwGetFramebufferSize(m_window, &w, &h);
        extent.width  = std::clamp(static_cast<std::uint32_t>(w),
            caps.minImageExtent.width, caps.maxImageExtent.width);
        extent.height = std::clamp(static_cast<std::uint32_t>(h),
            caps.minImageExtent.height, caps.maxImageExtent.height);
    }

    std::uint32_t image_count = caps.minImageCount + 1;
    if (caps.maxImageCount > 0)
        image_count = std::min(image_count, caps.maxImageCount);

    vk::SwapchainCreateInfoKHR sci{};
    sci.surface          = m_surface;
    sci.minImageCount    = image_count;
    sci.imageFormat      = chosen.format;
    sci.imageColorSpace  = chosen.colorSpace;
    sci.imageExtent      = extent;
    sci.imageArrayLayers = 1;
    sci.imageUsage       = vk::ImageUsageFlagBits::eColorAttachment;
    sci.imageSharingMode = vk::SharingMode::eExclusive;
    sci.preTransform     = caps.currentTransform;
    sci.compositeAlpha   = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    sci.presentMode      = present_mode;
    sci.clipped          = VK_TRUE;
    sci.oldSwapchain     = nullptr;

    m_swapchain        = m_device.createSwapchainKHR(sci);
    m_swapchain_format = chosen.format;
    m_swapchain_extent = extent;
    m_swapchain_images = m_device.getSwapchainImagesKHR(m_swapchain);
}

void VulkanApp::createImageViews()
{
    m_swapchain_views.resize(m_swapchain_images.size());
    for (std::size_t i = 0; i < m_swapchain_images.size(); ++i)
    {
        vk::ImageViewCreateInfo vci{};
        vci.image    = m_swapchain_images[i];
        vci.viewType = vk::ImageViewType::e2D;
        vci.format   = m_swapchain_format;
        vci.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
        vci.subresourceRange.baseMipLevel   = 0;
        vci.subresourceRange.levelCount     = 1;
        vci.subresourceRange.baseArrayLayer = 0;
        vci.subresourceRange.layerCount     = 1;
        m_swapchain_views[i] = m_device.createImageView(vci);
    }
}

void VulkanApp::createRenderPass()
{
    vk::AttachmentDescription color{};
    color.format         = m_swapchain_format;
    color.samples        = vk::SampleCountFlagBits::e1;
    color.loadOp         = vk::AttachmentLoadOp::eClear;
    color.storeOp        = vk::AttachmentStoreOp::eStore;
    color.stencilLoadOp  = vk::AttachmentLoadOp::eDontCare;
    color.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    color.initialLayout  = vk::ImageLayout::eUndefined;
    color.finalLayout    = vk::ImageLayout::ePresentSrcKHR;

    vk::AttachmentReference color_ref{ 0, vk::ImageLayout::eColorAttachmentOptimal };

    vk::SubpassDescription subpass{};
    subpass.pipelineBindPoint    = vk::PipelineBindPoint::eGraphics;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments    = &color_ref;

    vk::SubpassDependency dep{};
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass    = 0;
    dep.srcStageMask  = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    dep.dstStageMask  = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    dep.srcAccessMask = {};
    dep.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;

    vk::RenderPassCreateInfo rci{};
    rci.attachmentCount = 1;
    rci.pAttachments    = &color;
    rci.subpassCount    = 1;
    rci.pSubpasses      = &subpass;
    rci.dependencyCount = 1;
    rci.pDependencies   = &dep;

    m_render_pass = m_device.createRenderPass(rci);
}

void VulkanApp::createFramebuffers()
{
    m_framebuffers.resize(m_swapchain_views.size());
    for (std::size_t i = 0; i < m_swapchain_views.size(); ++i)
    {
        vk::FramebufferCreateInfo fci{};
        fci.renderPass      = m_render_pass;
        fci.attachmentCount = 1;
        fci.pAttachments    = &m_swapchain_views[i];
        fci.width           = m_swapchain_extent.width;
        fci.height          = m_swapchain_extent.height;
        fci.layers          = 1;
        m_framebuffers[i]   = m_device.createFramebuffer(fci);
    }
}

// ----------------------------------------------------------------------------
// Command pool + buffers + sync
// ----------------------------------------------------------------------------

void VulkanApp::createCommandPool()
{
    vk::CommandPoolCreateInfo pci{};
    pci.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
    pci.queueFamilyIndex = m_queue_family;
    m_command_pool = m_device.createCommandPool(pci);
}

void VulkanApp::createCommandBuffers()
{
    vk::CommandBufferAllocateInfo ai{};
    ai.commandPool        = m_command_pool;
    ai.level              = vk::CommandBufferLevel::ePrimary;
    ai.commandBufferCount = kMaxFramesInFlight;
    m_command_buffers     = m_device.allocateCommandBuffers(ai);
}

void VulkanApp::createSyncObjects()
{
    m_image_available .resize(kMaxFramesInFlight);
    m_render_finished .resize(kMaxFramesInFlight);
    m_in_flight_fences.resize(kMaxFramesInFlight);
    for (int i = 0; i < kMaxFramesInFlight; ++i)
    {
        m_image_available[i]  = m_device.createSemaphore({});
        m_render_finished[i]  = m_device.createSemaphore({});
        m_in_flight_fences[i] = m_device.createFence({ vk::FenceCreateFlagBits::eSignaled });
    }
}

void VulkanApp::createImGuiDescriptorPool()
{
    // ImGui's Vulkan backend bills for COMBINED_IMAGE_SAMPLER and a few others.
    // The pool is oversized but cheap — keeps us from poking at the limits
    // when 4b adds the lattice-state texture sampler.
    std::array<vk::DescriptorPoolSize, 11> pool_sizes{{
        { vk::DescriptorType::eSampler,              1000 },
        { vk::DescriptorType::eCombinedImageSampler, 1000 },
        { vk::DescriptorType::eSampledImage,         1000 },
        { vk::DescriptorType::eStorageImage,         1000 },
        { vk::DescriptorType::eUniformTexelBuffer,   1000 },
        { vk::DescriptorType::eStorageTexelBuffer,   1000 },
        { vk::DescriptorType::eUniformBuffer,        1000 },
        { vk::DescriptorType::eStorageBuffer,        1000 },
        { vk::DescriptorType::eUniformBufferDynamic, 1000 },
        { vk::DescriptorType::eStorageBufferDynamic, 1000 },
        { vk::DescriptorType::eInputAttachment,      1000 },
    }};
    vk::DescriptorPoolCreateInfo pci{};
    pci.flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
    pci.maxSets       = 1000 * static_cast<std::uint32_t>(pool_sizes.size());
    pci.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
    pci.pPoolSizes    = pool_sizes.data();
    m_imgui_desc_pool = m_device.createDescriptorPool(pci);
}

// ----------------------------------------------------------------------------
// Main loop
// ----------------------------------------------------------------------------

void VulkanApp::mainLoop()
{
    while (!glfwWindowShouldClose(m_window))
    {
        glfwPollEvents();
        drawFrame();
    }
    if (m_device) m_device.waitIdle();
}

void VulkanApp::buildUI()
{
    if (m_show_menu) buildMenuUI();
    else             buildSceneUI();
}

void VulkanApp::buildMenuUI()
{
    ImGuiViewport* vp = ImGui::GetMainViewport();
    const float    w = std::min(960.0f, vp->WorkSize.x * 0.92f);
    const float    h = vp->WorkSize.y * 0.88f;
    ImGui::SetNextWindowSize(ImVec2(w, h), ImGuiCond_Always);
    ImGui::SetNextWindowPos(vp->GetCenter(), ImGuiCond_Always, ImVec2(0.5f, 0.5f));

    ImGui::Begin("LatticeQFT — scene selector", nullptr,
                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse
               | ImGuiWindowFlags_NoMove);

    ImGui::TextUnformatted("Pick a scene to launch");
    ImGui::TextDisabled(
        "Lattice QFT visualizations look like noise without context — pick "
        "a scene on the left to see what it is, why it's there, and what "
        "to watch for while the chain runs.");
    ImGui::Separator();

    // Clamp the selection in case the descriptor list changes.
    if (m_selected_scene_idx < 0
        || m_selected_scene_idx >= static_cast<int>(m_scene_descriptors.size()))
        m_selected_scene_idx = 0;

    const float left_w  = 240.0f;
    const float pad     = 8.0f;

    // ---- left: scene list ----
    ImGui::BeginChild("scene_list", ImVec2(left_w, 0.0f),
                      ImGuiChildFlags_Border);
    for (std::size_t i = 0; i < m_scene_descriptors.size(); ++i)
    {
        const auto& s = m_scene_descriptors[i];
        const bool selected = (static_cast<int>(i) == m_selected_scene_idx);
        ImGui::PushID(static_cast<int>(i));
        if (ImGui::Selectable(s.name.c_str(), selected,
                              ImGuiSelectableFlags_AllowDoubleClick))
        {
            m_selected_scene_idx = static_cast<int>(i);
            if (ImGui::IsMouseDoubleClicked(0))
            {
                VulkanContext ctx;
                ctx.phys         = m_physical_device;
                ctx.device       = m_device;
                ctx.queue        = m_queue;
                ctx.queue_family = m_queue_family;
                ctx.cmd_pool     = m_command_pool;
                ctx.shader_dir   = m_shader_dir;
                activateScene(s.factory(ctx));
                ImGui::PopID();
                ImGui::EndChild();
                ImGui::End();
                return;
            }
        }
        ImGui::PopID();
    }
    ImGui::EndChild();

    ImGui::SameLine(0.0f, pad);

    // ---- right: details for the selected scene ----
    ImGui::BeginChild("scene_details", ImVec2(0.0f, 0.0f),
                      ImGuiChildFlags_Border);

    const auto& sel = m_scene_descriptors[
        static_cast<std::size_t>(m_selected_scene_idx)];

    ImGui::PushFont(nullptr); // default font; section header
    ImGui::TextUnformatted(sel.name.c_str());
    ImGui::PopFont();
    ImGui::Separator();

    ImGui::TextUnformatted("Description");
    ImGui::PushStyleColor(ImGuiCol_Text,
                          ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
    ImGui::TextWrapped("%s", sel.description.c_str());
    ImGui::PopStyleColor();
    ImGui::Spacing();

    if (!sel.tips.empty())
    {
        ImGui::TextUnformatted("Things to look for");
        ImGui::PushStyleColor(ImGuiCol_Text,
                              ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
        for (const auto& tip : sel.tips)
        {
            ImGui::Bullet();
            ImGui::SameLine();
            ImGui::TextWrapped("%s", tip.c_str());
            ImGui::Spacing();
        }
        ImGui::PopStyleColor();
    }

    ImGui::Spacing();
    ImGui::Separator();

    if (ImGui::Button("Launch", ImVec2(120.0f, 0.0f)))
    {
        VulkanContext ctx;
        ctx.phys         = m_physical_device;
        ctx.device       = m_device;
        ctx.queue        = m_queue;
        ctx.queue_family = m_queue_family;
        ctx.cmd_pool     = m_command_pool;
        ctx.shader_dir   = m_shader_dir;
        activateScene(sel.factory(ctx));
        ImGui::EndChild();
        ImGui::End();
        return;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(or double-click the scene name)");

    ImGui::EndChild();
    ImGui::End();
}

void VulkanApp::dumpActiveVolume()
{
    if (!m_active_scene || m_active_is_2d) return;
    const int L = m_active_scene->latticeSize();
    const std::size_t V3
        = static_cast<std::size_t>(L) * L * L;

    // Re-fill the volume buffer so the dump matches what's currently
    // rendered (cheap — fillVolume is at most O(L³)).
    if (m_volume_buffer.size() < V3) m_volume_buffer.assign(V3, 0.0f);
    m_active_scene->fillVolume(m_volume_buffer.data());

    // Build a safe filename inside the current working dir.
    std::string safe = m_active_scene_name;
    for (char& c : safe)
        if (c == ' ' || c == '/' || c == '(' || c == ')' || c == ',') c = '_';
    const std::string path = "volume_dump_" + safe + ".txt";

    std::ofstream f(path);
    if (!f.is_open())
    {
        m_last_dump_path = "(failed to open " + path + ")";
        return;
    }
    f << "# LatticeQFT volume dump — " << m_active_scene_name << "\n";
    f << "# L = " << L << "  V³ = " << V3 << "\n";

    // Statistics.
    float vmin =  std::numeric_limits<float>::infinity();
    float vmax = -std::numeric_limits<float>::infinity();
    double vsum = 0.0;
    for (std::size_t i = 0; i < V3; ++i)
    {
        const float v = m_volume_buffer[i];
        vmin = std::min(vmin, v);
        vmax = std::max(vmax, v);
        vsum += static_cast<double>(v);
    }
    f << "# min = " << vmin
      << "  max = " << vmax
      << "  mean = " << (vsum / static_cast<double>(V3)) << "\n\n";

    // Top-30 brightest voxels with (x, y, z) coords.
    f << "## Top 30 brightest voxels (x, y, z, value)\n";
    std::vector<std::size_t> idx(V3);
    std::iota(idx.begin(), idx.end(), std::size_t{0});
    std::partial_sort(idx.begin(),
                      idx.begin() + std::min<std::size_t>(30, V3),
                      idx.end(),
                      [&](std::size_t a, std::size_t b)
                      { return m_volume_buffer[a] > m_volume_buffer[b]; });
    for (int k = 0; k < std::min<int>(30, static_cast<int>(V3)); ++k)
    {
        const std::size_t i = idx[static_cast<std::size_t>(k)];
        const int x = static_cast<int>(i % L);
        const int y = static_cast<int>((i / L) % L);
        const int z = static_cast<int>(i / (L * L));
        f << "  (" << x << ", " << y << ", " << z << ")  "
          << m_volume_buffer[i] << "\n";
    }
    f << "\n";

    // Three orthogonal central slices.
    auto slice = [&](const std::string& title,
                     const std::string& xlabel,
                     const std::string& ylabel,
                     std::function<float(int, int)> at)
    {
        f << "## " << title << "\n";
        f << "   columns = " << xlabel
          << ", rows = " << ylabel << "\n";
        for (int j = 0; j < L; ++j)
        {
            for (int i = 0; i < L; ++i)
                f << std::setw(7) << std::fixed << std::setprecision(3)
                  << at(i, j) << " ";
            f << "\n";
        }
        f << "\n";
    };
    const int mid = L / 2;
    slice("Central z-slice  (z = L/2)", "x", "y",
          [&](int x, int y)
          {
              return m_volume_buffer[
                  (static_cast<std::size_t>(mid) * L + y) * L + x];
          });
    slice("Central y-slice  (y = L/2)", "x", "z",
          [&](int x, int z)
          {
              return m_volume_buffer[
                  (static_cast<std::size_t>(z) * L + mid) * L + x];
          });
    slice("Central x-slice  (x = L/2)", "y", "z",
          [&](int y, int z)
          {
              return m_volume_buffer[
                  (static_cast<std::size_t>(z) * L + y) * L + mid];
          });

    f.close();

    // Resolve absolute path for the UI hint.
    try
    {
        m_last_dump_path = std::filesystem::absolute(path).string();
    }
    catch (...) { m_last_dump_path = path; }
}

void VulkanApp::buildSceneUI()
{
    ImGui::Begin("LatticeQFT");
    if (m_active_scene)
    {
        if (m_active_scene->buildControlsUI())
        {
            ImGui::End();
            returnToMenu();
            return;
        }
    }
    else
    {
        ImGui::TextUnformatted("(no active scene)");
    }
    ImGui::End();

    // Auxiliary panels (tips, transfer function, camera). Hidden until a
    // scene is active so the menu stays uncluttered.
    if (m_active_scene)
    {
        // Things-to-look-for panel — carries the menu's tip bullets into
        // the running scene so the user has visual-context guidance without
        // going back to the menu. Default-closed to keep the screen clean.
        if (!m_active_scene_tips.empty() || !m_active_scene_description.empty())
        {
            ImGui::Begin("Scene notes");
            if (ImGui::CollapsingHeader("Description"))
            {
                ImGui::PushStyleColor(ImGuiCol_Text,
                    ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
                ImGui::TextWrapped("%s", m_active_scene_description.c_str());
                ImGui::PopStyleColor();
            }
            if (!m_active_scene_tips.empty())
            {
                if (ImGui::CollapsingHeader("Things to look for"))
                {
                    ImGui::PushStyleColor(ImGuiCol_Text,
                        ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
                    for (const auto& tip : m_active_scene_tips)
                    {
                        ImGui::Bullet();
                        ImGui::SameLine();
                        ImGui::TextWrapped("%s", tip.c_str());
                        ImGui::Spacing();
                    }
                    ImGui::PopStyleColor();
                }
            }
            ImGui::End();
        }

        ImGui::Begin("Render controls");
        if (ImGui::CollapsingHeader("Transfer function",
                                    ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::SliderFloat("density scale",     &m_knobs.density_scale,     0.05f, 8.0f, "%.3f");
            ImGui::SliderFloat("opacity threshold", &m_knobs.opacity_threshold, 0.0f,  3.0f, "%.3f");
            ImGui::SliderFloat("colormap gamma",    &m_knobs.gamma,             0.3f,  3.0f, "%.3f");
        }
        if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (m_camera)
            {
                ImGui::Text("distance = %.2f   polar = %.2f   azim = %.2f",
                            m_camera->distance(), m_camera->polar(),
                            m_camera->azimuthal());
                if (ImGui::Button("reset view")) m_camera->reset();
            }
            ImGui::TextUnformatted("(drag viewport to rotate, scroll to zoom)");
        }
        if (!m_active_is_2d
            && ImGui::CollapsingHeader("Diagnostics", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::TextDisabled(
                "Dumps the current volume buffer to a text file: top-30\n"
                "brightest voxels with (x, y, z) coordinates plus three\n"
                "orthogonal central slices, so the orientation of bright\n"
                "regions is unambiguous and inspectable numerically.");
            if (ImGui::Button("Dump volume to file"))
                dumpActiveVolume();
            ImGui::SameLine();
            ImGui::TextDisabled("→ %s", m_last_dump_path.empty()
                ? "(none yet)" : m_last_dump_path.c_str());
        }
        ImGui::End();
    }
}

void VulkanApp::drawFrame()
{
    // Wait for previous frame to finish.
    throwIfFailed(m_device.waitForFences(
        1, &m_in_flight_fences[m_current_frame], VK_TRUE, UINT64_MAX),
        "waitForFences");

    // Acquire next image.
    auto acq = m_device.acquireNextImageKHR(m_swapchain, UINT64_MAX,
        m_image_available[m_current_frame], nullptr);
    if (acq.result == vk::Result::eErrorOutOfDateKHR)
    {
        recreateSwapchain();
        return;
    }
    if (acq.result != vk::Result::eSuccess
        && acq.result != vk::Result::eSuboptimalKHR)
        throwIfFailed(acq.result, "acquireNextImageKHR");
    const std::uint32_t img_index = acq.value;

    throwIfFailed(m_device.resetFences(1, &m_in_flight_fences[m_current_frame]),
        "resetFences");

    // Run MC sweeps + fill the per-frame texture buffer. The active scene's
    // own state decides whether the buffer is L² (2D heatmap) or L³ (3D
    // volume); the renderer pointer that's alive tells us which path to take.
    const bool scene_active_3d = (!m_show_menu) && m_active_scene && m_volume && !m_active_is_2d;
    const bool scene_active_2d = (!m_show_menu) && m_active_scene && m_heatmap2d && m_active_is_2d;
    const bool scene_active    = scene_active_3d || scene_active_2d;
    // Capture per-frame inputs that depend on the scene BEFORE buildUI(),
    // because the user's "← Back to Menu" click can null `m_active_scene`
    // out from under us.
    Scene::HeatmapStyle cached_heatmap_style{};
    if (scene_active)
    {
        m_active_scene->step();
        m_active_scene->fillVolume(m_volume_buffer.data());
        if (scene_active_2d) cached_heatmap_style = m_active_scene->heatmapStyle();
    }

    // ImGui frame.
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    buildUI();
    ImGui::Render();

    // Record command buffer.
    auto& cmd = m_command_buffers[m_current_frame];
    cmd.reset();
    vk::CommandBufferBeginInfo bi{};
    cmd.begin(bi);

    // Re-check after buildUI: the scene may have been deactivated by a
    // "Back to Menu" click during the ImGui pass. The renderers themselves
    // remain alive until the next scene activation, so we can drop the
    // upload/draw safely without touching their state.
    const bool render_3d = scene_active_3d && m_active_scene;
    const bool render_2d = scene_active_2d && m_active_scene;
    if (render_3d) m_volume   ->upload(cmd, m_volume_buffer.data());
    if (render_2d) m_heatmap2d->upload(cmd, m_volume_buffer.data());

    vk::ClearValue clear{};
    clear.color = vk::ClearColorValue{ std::array<float, 4>{ 0.08f, 0.10f, 0.14f, 1.0f }};

    vk::RenderPassBeginInfo rpi{};
    rpi.renderPass        = m_render_pass;
    rpi.framebuffer       = m_framebuffers[img_index];
    rpi.renderArea.offset = vk::Offset2D{ 0, 0 };
    rpi.renderArea.extent = m_swapchain_extent;
    rpi.clearValueCount   = 1;
    rpi.pClearValues      = &clear;

    cmd.beginRenderPass(rpi, vk::SubpassContents::eInline);
    if (render_3d && m_camera)
    {
        const float aspect = static_cast<float>(m_swapchain_extent.width)
                           / static_cast<float>(m_swapchain_extent.height);
        VolumeRenderer::PushConstants pc{};
        pc.inv_view_proj     = m_camera->inverseViewProj(aspect);
        pc.cam_pos           = m_camera->position();
        pc.density_scale     = m_knobs.density_scale;
        pc.opacity_threshold = m_knobs.opacity_threshold;
        pc.gamma             = m_knobs.gamma;
        m_volume->recordDraw(cmd, m_swapchain_extent, pc);
    }
    else if (render_2d)
    {
        Heatmap2DRenderer::PushConstants pc2d{};
        pc2d.gain       = cached_heatmap_style.gain;
        pc2d.bias       = cached_heatmap_style.bias;
        pc2d.gamma      = cached_heatmap_style.gamma;
        pc2d.signed_map = cached_heatmap_style.signed_map;
        m_heatmap2d->recordDraw(cmd, m_swapchain_extent, pc2d);
    }
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    cmd.endRenderPass();
    cmd.end();

    // Submit.
    vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    vk::SubmitInfo si{};
    si.waitSemaphoreCount   = 1;
    si.pWaitSemaphores      = &m_image_available[m_current_frame];
    si.pWaitDstStageMask    = &wait_stage;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores    = &m_render_finished[m_current_frame];

    throwIfFailed(m_queue.submit(1, &si, m_in_flight_fences[m_current_frame]),
                  "queue.submit");

    // Present.
    vk::PresentInfoKHR pi{};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores    = &m_render_finished[m_current_frame];
    pi.swapchainCount     = 1;
    pi.pSwapchains        = &m_swapchain;
    pi.pImageIndices      = &img_index;
    auto present_result   = m_queue.presentKHR(&pi);
    if (present_result == vk::Result::eErrorOutOfDateKHR
        || present_result == vk::Result::eSuboptimalKHR
        || m_framebuffer_resized)
    {
        m_framebuffer_resized = false;
        recreateSwapchain();
    }
    else if (present_result != vk::Result::eSuccess)
    {
        throwIfFailed(present_result, "presentKHR");
    }

    m_current_frame = (m_current_frame + 1) % kMaxFramesInFlight;
}

// ----------------------------------------------------------------------------
// Swapchain rebuild on resize / out-of-date
// ----------------------------------------------------------------------------

void VulkanApp::destroySwapchainObjects()
{
    if (!m_device) return;
    for (auto fb : m_framebuffers) if (fb) m_device.destroyFramebuffer(fb);
    m_framebuffers.clear();
    for (auto v : m_swapchain_views) if (v) m_device.destroyImageView(v);
    m_swapchain_views.clear();
    if (m_swapchain) { m_device.destroySwapchainKHR(m_swapchain); m_swapchain = nullptr; }
}

void VulkanApp::recreateSwapchain()
{
    // Block until window has a non-zero framebuffer (minimization).
    int w = 0, h = 0;
    glfwGetFramebufferSize(m_window, &w, &h);
    while (w == 0 || h == 0)
    {
        glfwGetFramebufferSize(m_window, &w, &h);
        glfwWaitEvents();
    }
    m_device.waitIdle();
    destroySwapchainObjects();
    createSwapchain();
    createImageViews();
    createFramebuffers();
}

// ----------------------------------------------------------------------------
// Cleanup
// ----------------------------------------------------------------------------

void VulkanApp::cleanup()
{
    if (m_device)
        m_device.waitIdle();

    // Drop 4c owners before tearing down their Vulkan dependencies.
    m_active_scene.reset();
    m_volume.reset();
    m_heatmap2d.reset();
    m_camera.reset();

    if (m_imgui_initialized)
    {
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        m_imgui_initialized = false;
    }
    if (m_device)
    {
        if (m_imgui_desc_pool) m_device.destroyDescriptorPool(m_imgui_desc_pool);

        for (auto f : m_in_flight_fences) if (f) m_device.destroyFence(f);
        for (auto s : m_render_finished)  if (s) m_device.destroySemaphore(s);
        for (auto s : m_image_available)  if (s) m_device.destroySemaphore(s);
        m_in_flight_fences.clear();
        m_render_finished.clear();
        m_image_available.clear();

        if (!m_command_buffers.empty() && m_command_pool)
            m_device.freeCommandBuffers(m_command_pool, m_command_buffers);
        if (m_command_pool) m_device.destroyCommandPool(m_command_pool);

        destroySwapchainObjects();
        if (m_render_pass) m_device.destroyRenderPass(m_render_pass);

        m_device.destroy();
        m_device = nullptr;
    }

    if (m_instance)
    {
        if (m_surface) m_instance.destroySurfaceKHR(m_surface);
        m_instance.destroy();
        m_instance = nullptr;
    }

    if (m_window) { glfwDestroyWindow(m_window); m_window = nullptr; }
    glfwTerminate();
}

} // namespace lqft::vis
