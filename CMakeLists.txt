cmake_minimum_required(VERSION 3.20)
project(LatticeQFT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# MSVC's <cmath> does not define M_PI etc. unless this is set before the include.
if(MSVC)
    add_compile_definitions(_USE_MATH_DEFINES)
endif()

# =============================================================================
# Dependencies via FetchContent (modern, no submodules required)
# =============================================================================
include(FetchContent)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

FetchContent_Declare(
    magic_enum
    GIT_REPOSITORY https://github.com/Neargye/magic_enum.git
    GIT_TAG        v0.9.7
)
FetchContent_MakeAvailable(magic_enum)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# =============================================================================
# Core library (header-only for now; Dim-templated lattice infrastructure)
# =============================================================================
add_library(lqft_core INTERFACE)
target_include_directories(lqft_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(lqft_core INTERFACE
    nlohmann_json::nlohmann_json
    magic_enum::magic_enum
)
target_compile_features(lqft_core INTERFACE cxx_std_23)

# =============================================================================
# Headless CLI driver: ising / phi4 / u1 / schwinger-quenched / schwinger-hmc
# =============================================================================
add_executable(LatticeQFT
    src/main.cpp
)
target_link_libraries(LatticeQFT PRIVATE lqft_core)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(LatticeQFT PRIVATE -Wall -Wextra -Wpedantic)
endif()

# =============================================================================
# Vulkan visualizer (Round 4a scaffold; optional)
# =============================================================================
option(BUILD_VISUALIZER "Build the Vulkan visualizer (needs Vulkan SDK + GLFW + GLM)" ON)

if(BUILD_VISUALIZER)
    find_package(Vulkan QUIET)
    find_package(glfw3 QUIET)
    find_package(glm    QUIET)

    if(NOT Vulkan_FOUND OR NOT TARGET glfw OR NOT TARGET glm::glm)
        message(STATUS "Visualizer dependencies missing — turn off with -DBUILD_VISUALIZER=OFF "
                       "or install Vulkan SDK / GLFW / GLM (e.g. vcpkg).")
        set(BUILD_VISUALIZER OFF)
    endif()
endif()

if(BUILD_VISUALIZER)
    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG        v1.91.0
    )
    FetchContent_MakeAvailable(imgui)

    set(IMGUI_SOURCES
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
    )

    add_executable(LatticeQFTVis
        src/vis_main.cpp
        src/rendering/vulkan/vulkan_app.cpp
        src/rendering/vulkan/volume_renderer.cpp
        src/rendering/vulkan/heatmap_2d_renderer.cpp
        src/rendering/vulkan/u1_3d_scene.cpp
        src/rendering/vulkan/su2_3d_scene.cpp
        src/rendering/vulkan/flux_tube_scene.cpp
        src/rendering/vulkan/ising_2d_scene.cpp
        src/rendering/vulkan/phi4_2d_scene.cpp
        src/rendering/vulkan/u1_2d_scene.cpp
        src/rendering/vulkan/su2_2d_scene.cpp
        src/rendering/vulkan/gpu_wilson_dirac_2d.cpp
        src/rendering/vulkan/gpu_cg_2d.cpp
        src/rendering/vulkan/gpu_schwinger_force.cpp
        src/rendering/vulkan/polyakov_scene.cpp
        src/rendering/vulkan/schwinger_quenched_scene.cpp
        src/rendering/vulkan/schwinger_hmc_scene.cpp
        src/rendering/vulkan/gpu_su2_sweeper.cpp
        src/rendering/vulkan/gpu_u1_sweeper.cpp
        src/rendering/vulkan/gpu_ising_sweeper.cpp
        src/rendering/vulkan/gpu_phi4_sweeper.cpp
        ${IMGUI_SOURCES}
    )

    target_include_directories(LatticeQFTVis PRIVATE
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
    )

    target_link_libraries(LatticeQFTVis PRIVATE
        lqft_core
        Vulkan::Vulkan
        glfw
        glm::glm
    )

    if(APPLE)
        target_compile_definitions(LatticeQFTVis PRIVATE VK_USE_PLATFORM_MACOS_MVK)
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(LatticeQFTVis PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    # -------------------------------------------------------------------------
    # Shader compilation: GLSL → SPIR-V via glslc.
    # The two .spv files are copied next to the executable so loading from
    # `shaders/` relative to cwd Just Works after a normal launch.
    # -------------------------------------------------------------------------
    find_program(GLSLC glslc HINTS
        $ENV{VULKAN_SDK}/bin
        $ENV{VK_SDK_PATH}/bin
    )
    if(GLSLC)
        set(SHADER_VERT_SRC    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/fullscreen.vert)
        set(SHADER_FRAG_SRC    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/volume.frag)
        set(SHADER_HM2D_SRC    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/heatmap_2d.frag)
        set(SHADER_COMP_SRC    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/su2_metropolis.comp)
        set(SHADER_U1_SRC      ${CMAKE_CURRENT_SOURCE_DIR}/shaders/u1_metropolis.comp)
        set(SHADER_ISING_SRC   ${CMAKE_CURRENT_SOURCE_DIR}/shaders/ising_metropolis.comp)
        set(SHADER_PHI4_SRC    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/phi4_metropolis.comp)
        set(SHADER_WD2D_SRC    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/wilson_dirac_2d.comp)
        set(SHADER_AXPBY_SRC   ${CMAKE_CURRENT_SOURCE_DIR}/shaders/spinor_axpby_real.comp)
        set(SHADER_INNER_SRC   ${CMAKE_CURRENT_SOURCE_DIR}/shaders/spinor_inner_real.comp)
        set(SHADER_GFRC_SRC    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/gauge_force_u1.comp)
        set(SHADER_FFRC_SRC    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/fermion_force_u1.comp)
        set(SHADER_PLAQ_SRC    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/plaquette_action_u1.comp)
        set(SHADER_VERT_SPV    ${CMAKE_CURRENT_BINARY_DIR}/shaders/fullscreen.vert.spv)
        set(SHADER_FRAG_SPV    ${CMAKE_CURRENT_BINARY_DIR}/shaders/volume.frag.spv)
        set(SHADER_HM2D_SPV    ${CMAKE_CURRENT_BINARY_DIR}/shaders/heatmap_2d.frag.spv)
        set(SHADER_COMP_SPV    ${CMAKE_CURRENT_BINARY_DIR}/shaders/su2_metropolis.comp.spv)
        set(SHADER_U1_SPV      ${CMAKE_CURRENT_BINARY_DIR}/shaders/u1_metropolis.comp.spv)
        set(SHADER_ISING_SPV   ${CMAKE_CURRENT_BINARY_DIR}/shaders/ising_metropolis.comp.spv)
        set(SHADER_PHI4_SPV    ${CMAKE_CURRENT_BINARY_DIR}/shaders/phi4_metropolis.comp.spv)
        set(SHADER_WD2D_SPV    ${CMAKE_CURRENT_BINARY_DIR}/shaders/wilson_dirac_2d.comp.spv)
        set(SHADER_AXPBY_SPV   ${CMAKE_CURRENT_BINARY_DIR}/shaders/spinor_axpby_real.comp.spv)
        set(SHADER_INNER_SPV   ${CMAKE_CURRENT_BINARY_DIR}/shaders/spinor_inner_real.comp.spv)
        set(SHADER_GFRC_SPV    ${CMAKE_CURRENT_BINARY_DIR}/shaders/gauge_force_u1.comp.spv)
        set(SHADER_FFRC_SPV    ${CMAKE_CURRENT_BINARY_DIR}/shaders/fermion_force_u1.comp.spv)
        set(SHADER_PLAQ_SPV    ${CMAKE_CURRENT_BINARY_DIR}/shaders/plaquette_action_u1.comp.spv)

        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/shaders)

        add_custom_command(
            OUTPUT  ${SHADER_VERT_SPV}
            COMMAND ${GLSLC} -fshader-stage=vertex   ${SHADER_VERT_SRC} -o ${SHADER_VERT_SPV}
            DEPENDS ${SHADER_VERT_SRC}
            COMMENT "glslc fullscreen.vert → SPIR-V"
            VERBATIM
        )
        add_custom_command(
            OUTPUT  ${SHADER_FRAG_SPV}
            COMMAND ${GLSLC} -fshader-stage=fragment ${SHADER_FRAG_SRC} -o ${SHADER_FRAG_SPV}
            DEPENDS ${SHADER_FRAG_SRC}
            COMMENT "glslc volume.frag → SPIR-V"
            VERBATIM
        )
        add_custom_command(
            OUTPUT  ${SHADER_COMP_SPV}
            COMMAND ${GLSLC} -fshader-stage=compute  ${SHADER_COMP_SRC} -o ${SHADER_COMP_SPV}
            DEPENDS ${SHADER_COMP_SRC}
            COMMENT "glslc su2_metropolis.comp → SPIR-V"
            VERBATIM
        )
        add_custom_command(
            OUTPUT  ${SHADER_U1_SPV}
            COMMAND ${GLSLC} -fshader-stage=compute  ${SHADER_U1_SRC} -o ${SHADER_U1_SPV}
            DEPENDS ${SHADER_U1_SRC}
            COMMENT "glslc u1_metropolis.comp → SPIR-V"
            VERBATIM
        )
        add_custom_command(
            OUTPUT  ${SHADER_HM2D_SPV}
            COMMAND ${GLSLC} -fshader-stage=fragment ${SHADER_HM2D_SRC} -o ${SHADER_HM2D_SPV}
            DEPENDS ${SHADER_HM2D_SRC}
            COMMENT "glslc heatmap_2d.frag → SPIR-V"
            VERBATIM
        )
        add_custom_command(
            OUTPUT  ${SHADER_ISING_SPV}
            COMMAND ${GLSLC} -fshader-stage=compute  ${SHADER_ISING_SRC} -o ${SHADER_ISING_SPV}
            DEPENDS ${SHADER_ISING_SRC}
            COMMENT "glslc ising_metropolis.comp → SPIR-V"
            VERBATIM
        )
        add_custom_command(
            OUTPUT  ${SHADER_PHI4_SPV}
            COMMAND ${GLSLC} -fshader-stage=compute  ${SHADER_PHI4_SRC} -o ${SHADER_PHI4_SPV}
            DEPENDS ${SHADER_PHI4_SRC}
            COMMENT "glslc phi4_metropolis.comp → SPIR-V"
            VERBATIM
        )
        add_custom_command(
            OUTPUT  ${SHADER_WD2D_SPV}
            COMMAND ${GLSLC} -fshader-stage=compute  ${SHADER_WD2D_SRC} -o ${SHADER_WD2D_SPV}
            DEPENDS ${SHADER_WD2D_SRC}
            COMMENT "glslc wilson_dirac_2d.comp → SPIR-V"
            VERBATIM
        )
        add_custom_command(
            OUTPUT  ${SHADER_AXPBY_SPV}
            COMMAND ${GLSLC} -fshader-stage=compute  ${SHADER_AXPBY_SRC} -o ${SHADER_AXPBY_SPV}
            DEPENDS ${SHADER_AXPBY_SRC}
            COMMENT "glslc spinor_axpby_real.comp → SPIR-V"
            VERBATIM
        )
        add_custom_command(
            OUTPUT  ${SHADER_INNER_SPV}
            COMMAND ${GLSLC} -fshader-stage=compute  ${SHADER_INNER_SRC} -o ${SHADER_INNER_SPV}
            DEPENDS ${SHADER_INNER_SRC}
            COMMENT "glslc spinor_inner_real.comp → SPIR-V"
            VERBATIM
        )
        add_custom_command(
            OUTPUT  ${SHADER_GFRC_SPV}
            COMMAND ${GLSLC} -fshader-stage=compute  ${SHADER_GFRC_SRC} -o ${SHADER_GFRC_SPV}
            DEPENDS ${SHADER_GFRC_SRC}
            COMMENT "glslc gauge_force_u1.comp → SPIR-V"
            VERBATIM
        )
        add_custom_command(
            OUTPUT  ${SHADER_FFRC_SPV}
            COMMAND ${GLSLC} -fshader-stage=compute  ${SHADER_FFRC_SRC} -o ${SHADER_FFRC_SPV}
            DEPENDS ${SHADER_FFRC_SRC}
            COMMENT "glslc fermion_force_u1.comp → SPIR-V"
            VERBATIM
        )
        add_custom_command(
            OUTPUT  ${SHADER_PLAQ_SPV}
            COMMAND ${GLSLC} -fshader-stage=compute  ${SHADER_PLAQ_SRC} -o ${SHADER_PLAQ_SPV}
            DEPENDS ${SHADER_PLAQ_SRC}
            COMMENT "glslc plaquette_action_u1.comp → SPIR-V"
            VERBATIM
        )
        add_custom_target(LatticeQFTShaders ALL
            DEPENDS ${SHADER_VERT_SPV} ${SHADER_FRAG_SPV}
                    ${SHADER_HM2D_SPV}
                    ${SHADER_COMP_SPV} ${SHADER_U1_SPV}
                    ${SHADER_ISING_SPV} ${SHADER_PHI4_SPV}
                    ${SHADER_WD2D_SPV}
                    ${SHADER_AXPBY_SPV} ${SHADER_INNER_SPV}
                    ${SHADER_GFRC_SPV} ${SHADER_FFRC_SPV}
                    ${SHADER_PLAQ_SPV})
        add_custom_command(
            TARGET LatticeQFTShaders POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:LatticeQFTVis>/shaders"
            COMMAND ${CMAKE_COMMAND} -E copy ${SHADER_VERT_SPV}   "$<TARGET_FILE_DIR:LatticeQFTVis>/shaders/"
            COMMAND ${CMAKE_COMMAND} -E copy ${SHADER_FRAG_SPV}   "$<TARGET_FILE_DIR:LatticeQFTVis>/shaders/"
            COMMAND ${CMAKE_COMMAND} -E copy ${SHADER_HM2D_SPV}   "$<TARGET_FILE_DIR:LatticeQFTVis>/shaders/"
            COMMAND ${CMAKE_COMMAND} -E copy ${SHADER_COMP_SPV}   "$<TARGET_FILE_DIR:LatticeQFTVis>/shaders/"
            COMMAND ${CMAKE_COMMAND} -E copy ${SHADER_U1_SPV}     "$<TARGET_FILE_DIR:LatticeQFTVis>/shaders/"
            COMMAND ${CMAKE_COMMAND} -E copy ${SHADER_ISING_SPV}  "$<TARGET_FILE_DIR:LatticeQFTVis>/shaders/"
            COMMAND ${CMAKE_COMMAND} -E copy ${SHADER_PHI4_SPV}   "$<TARGET_FILE_DIR:LatticeQFTVis>/shaders/"
            COMMAND ${CMAKE_COMMAND} -E copy ${SHADER_WD2D_SPV}   "$<TARGET_FILE_DIR:LatticeQFTVis>/shaders/"
            COMMAND ${CMAKE_COMMAND} -E copy ${SHADER_AXPBY_SPV}  "$<TARGET_FILE_DIR:LatticeQFTVis>/shaders/"
            COMMAND ${CMAKE_COMMAND} -E copy ${SHADER_INNER_SPV}  "$<TARGET_FILE_DIR:LatticeQFTVis>/shaders/"
            COMMAND ${CMAKE_COMMAND} -E copy ${SHADER_GFRC_SPV}   "$<TARGET_FILE_DIR:LatticeQFTVis>/shaders/"
            COMMAND ${CMAKE_COMMAND} -E copy ${SHADER_FFRC_SPV}   "$<TARGET_FILE_DIR:LatticeQFTVis>/shaders/"
            COMMAND ${CMAKE_COMMAND} -E copy ${SHADER_PLAQ_SPV}   "$<TARGET_FILE_DIR:LatticeQFTVis>/shaders/"
            COMMENT "Copying compiled shaders next to LatticeQFTVis"
        )
        add_dependencies(LatticeQFTVis LatticeQFTShaders)
    else()
        message(WARNING "glslc not found — visualizer shaders won't compile. "
                        "Install Vulkan SDK or place pre-built .spv files in shaders/.")
    endif()
endif()

# =============================================================================
# Testing (GoogleTest)
# =============================================================================
option(BUILD_TESTS "Build the unit test suite" ON)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
