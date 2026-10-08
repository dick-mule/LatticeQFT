#pragma once

/**
 * @file vulkan_app.hpp
 * @brief Minimal Vulkan + GLFW + ImGui application scaffold for LatticeQFT.
 *
 * Round 4a deliverable: a window that opens, renders ImGui on a clear-color
 * background, and shuts down cleanly. No physics on-screen yet — that's 4b.
 *
 * The class owns every Vulkan object it creates and destroys them in
 * destructor order. Resize is disabled at the window level for the MVP;
 * swapchain rebuild will land in 4b together with the heatmap texture path.
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lqft::vis
{

class VolumeRenderer;
class Heatmap2DRenderer;
class Camera;
class Scene;
struct SceneDescriptor;

/// Per-frame knobs for the volumetric transfer function. Owned by the app,
/// driven by ImGui, copied into the VolumeRenderer's push constants each frame.
struct VolumeFrameKnobs
{
    float density_scale     = 1.4f;
    float opacity_threshold = 0.40f;
    float gamma             = 1.0f;
};

class VulkanApp
{
public:
    explicit VulkanApp(int width = 1280, int height = 800,
                       std::string title = "LatticeQFT",
                       int lattice_L = 24);
    ~VulkanApp();

    VulkanApp(const VulkanApp&)            = delete;
    VulkanApp& operator=(const VulkanApp&) = delete;
    VulkanApp(VulkanApp&&)                 = delete;
    VulkanApp& operator=(VulkanApp&&)      = delete;

    /// Block on the main render loop until the window is closed.
    void run();

private:
    // ----------------------------------------------------------------------
    // Set-up phases (called in order from the constructor).
    // ----------------------------------------------------------------------
    void initWindow();
    void initVulkan();
    void initImGui();

    // Per-phase building blocks.
    void createInstance();
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapchain();
    void createImageViews();
    void createRenderPass();
    void createFramebuffers();
    void createCommandPool();
    void createCommandBuffers();
    void createSyncObjects();
    void createImGuiDescriptorPool();

    // ----------------------------------------------------------------------
    // Main loop.
    // ----------------------------------------------------------------------
    void mainLoop();
    void drawFrame();
    void buildUI();
    void buildMenuUI();
    void buildSceneUI();
    /// Snapshot the current 3D volume buffer to a text file: top-30 brightest
    /// voxels with coordinates + three orthogonal central slices. Used to
    /// inspect tube orientation numerically when the on-screen rendering is
    /// ambiguous.
    void dumpActiveVolume();

    void registerScenes();
    void activateScene(std::unique_ptr<Scene> scene);
    void returnToMenu();
    void rebuildVolumeRendererForL(int L);
    void rebuildHeatmap2DRendererForL(int L);
    void tearDownActiveRenderers();

    // Swapchain rebuild for window resize.
    void recreateSwapchain();
    void destroySwapchainObjects();

    static void framebufferResizeCallback(GLFWwindow* w, int width, int height);
    static void scrollCallback           (GLFWwindow* w, double xoff, double yoff);
    static void cursorPosCallback        (GLFWwindow* w, double x, double y);
    static void mouseButtonCallback      (GLFWwindow* w, int button, int action, int mods);

    // ----------------------------------------------------------------------
    // Tear-down (destructor delegates here).
    // ----------------------------------------------------------------------
    void cleanup();

    // ----------------------------------------------------------------------
    // State.
    // ----------------------------------------------------------------------
    int           m_width;
    int           m_height;
    std::string   m_title;
    GLFWwindow*   m_window           = nullptr;

    vk::Instance       m_instance;
    vk::SurfaceKHR     m_surface;
    vk::PhysicalDevice m_physical_device;
    std::uint32_t      m_queue_family = 0;
    vk::Device         m_device;
    vk::Queue          m_queue;

    vk::SwapchainKHR             m_swapchain;
    vk::Format                   m_swapchain_format = vk::Format::eUndefined;
    vk::Extent2D                 m_swapchain_extent;
    std::vector<vk::Image>       m_swapchain_images;
    std::vector<vk::ImageView>   m_swapchain_views;

    vk::RenderPass               m_render_pass;
    std::vector<vk::Framebuffer> m_framebuffers;

    vk::CommandPool                m_command_pool;
    std::vector<vk::CommandBuffer> m_command_buffers;

    static constexpr int           kMaxFramesInFlight = 2;
    std::vector<vk::Semaphore>     m_image_available;
    std::vector<vk::Semaphore>     m_render_finished;
    std::vector<vk::Fence>         m_in_flight_fences;
    int                            m_current_frame = 0;

    vk::DescriptorPool             m_imgui_desc_pool;
    bool                           m_imgui_initialized = false;

    int                                m_lattice_L = 24; // current renderer L
    std::unique_ptr<Scene>             m_active_scene;
    std::vector<SceneDescriptor>       m_scene_descriptors;
    int                                m_selected_scene_idx = 0;
    bool                               m_show_menu = true;

    // Copied from the SceneDescriptor at launch time so the in-scene "Things
    // to look for" panel doesn't have to chase the descriptor through the
    // factory lambda.
    std::string                        m_active_scene_name;
    std::string                        m_active_scene_description;
    std::vector<std::string>           m_active_scene_tips;
    std::string                        m_last_dump_path;

    std::unique_ptr<VolumeRenderer>    m_volume;
    std::unique_ptr<Heatmap2DRenderer> m_heatmap2d;
    bool                               m_active_is_2d = false;
    std::unique_ptr<Camera>            m_camera;
    std::vector<float>                 m_volume_buffer; // L*L*L or L*L

    bool        m_framebuffer_resized = false;
    bool        m_mouse_left_down     = false;
    double      m_last_mouse_x        = 0.0;
    double      m_last_mouse_y        = 0.0;

    VolumeFrameKnobs m_knobs;
    std::string      m_shader_dir;
};

} // namespace lqft::vis
