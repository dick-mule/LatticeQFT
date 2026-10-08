/**
 * @file vis_main.cpp
 * @brief Entry point for the Vulkan visualizer.
 *
 * Round 4a only opens a window and lets you click around an ImGui panel.
 * Round 4b will wire the U(1) sweep into a live heatmap.
 */

#include "rendering/vulkan/vulkan_app.hpp"

#include <cstdio>
#include <exception>

int main()
{
    try
    {
        lqft::vis::VulkanApp app(/*w*/1280, /*h*/800,
                                  /*title*/"LatticeQFT — Visualizer");
        app.run();
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return 1;
    }
    return 0;
}
