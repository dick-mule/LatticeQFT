#pragma once

/**
 * @file scene.hpp
 * @brief Abstract scene interface for the LatticeQFT visualizer.
 *
 * A `Scene` encapsulates one runnable Monte Carlo model wired to the
 * volumetric renderer:
 *
 *   - `latticeSize()` reports the side length L of the L³ volume texture
 *     the scene wants to drive. The application creates / resizes the
 *     `VolumeRenderer` to match before the scene becomes active.
 *
 *   - `step()` runs one frame's worth of CPU-side MC sweeps. Called per
 *     frame; the scene decides whether the chain advances (e.g. honors
 *     a paused flag internally).
 *
 *   - `fillVolume(float*)` writes L³ floats — typically a per-site
 *     scalar diagnostic like action density — into the buffer the
 *     application then uploads to the texture.
 *
 *   - `buildControlsUI()` renders the scene's ImGui control panel
 *     contents (inside a `Begin`/`End` pair the application owns).
 *     Returns `true` to signal "go back to the menu".
 *
 * Scenes are registered with a `SceneDescriptor` (display name +
 * description + factory) and listed on the menu page.
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace lqft::vis
{

/// Bundle of Vulkan handles a scene needs to spin up its own compute
/// resources. Populated by VulkanApp after initVulkan and passed to every
/// scene factory. Scenes that don't need GPU compute (Ising, the basic
/// U(1)/SU(2) viewers) simply ignore it.
struct VulkanContext
{
    vk::PhysicalDevice phys;
    vk::Device         device;
    vk::Queue          queue;
    std::uint32_t      queue_family = 0;
    vk::CommandPool    cmd_pool;
    std::string        shader_dir;
};

class Scene
{
public:
    virtual ~Scene() = default;

    /// Side length of the volume / heatmap texture the scene drives.
    /// 3D scenes draw into an L³ texture; 2D scenes into L² — see
    /// `isVolumetric()`.
    virtual int latticeSize() const = 0;

    /// `true` (default) → app routes the scene through the 3D `VolumeRenderer`
    /// and calls `fillVolume(L*L*L floats)`.
    /// `false` → 2D `Heatmap2DRenderer`, `fillVolume(L*L floats)`.
    virtual bool isVolumetric() const { return true; }

    /// Push-constant knobs for the 2D heatmap path. Ignored for 3D scenes.
    struct HeatmapStyle
    {
        float gain        = 1.0f;
        float bias        = 0.0f;
        float gamma       = 1.0f;
        int   signed_map  = 0;   // 1 = diverging blue/yellow (Ising, signed φ)
    };
    virtual HeatmapStyle heatmapStyle() const { return {}; }

    /// One frame of MC work on the CPU.
    virtual void step() = 0;

    /// Fill L³ floats (row-major (x,y,z): `out[(z*L + y)*L + x]`) with the
    /// current per-site diagnostic for visualization.
    virtual void fillVolume(float* out) const = 0;

    /// Render scene-specific controls inside an ImGui window the caller
    /// has already begun. Should include a "Back to Menu" button. Return
    /// `true` exactly when the user requests returning to the menu.
    virtual bool buildControlsUI() = 0;
};

struct SceneDescriptor
{
    std::string                                                       name;
    std::string                                                       description;
    /// Concrete "what to look for while the chain runs" bullets. These
    /// matter much more for lattice scenes than for the path tracer —
    /// the visual signal is action density / Wilson-loop accumulators,
    /// which read as undifferentiated noise without context.
    std::vector<std::string>                                          tips;
    std::function<std::unique_ptr<Scene>(const VulkanContext&)>       factory;
};

} // namespace lqft::vis
