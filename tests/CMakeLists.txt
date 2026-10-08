# Tests CMakeLists.txt

add_executable(LatticeQFTTests
    test_lattice.cpp
    test_ising.cpp
    test_phi4.cpp
    test_u1.cpp
    test_dirac_2d.cpp
    test_cg.cpp
    test_condensate.cpp
    test_hmc.cpp
    test_schwinger.cpp
    test_su2.cpp
    test_deflation.cpp
    test_ape.cpp
    test_su3.cpp
    test_stout.cpp
    test_multilevel.cpp
)

target_link_libraries(LatticeQFTTests PRIVATE
    lqft_core
    GTest::gtest_main
    GTest::gmock
)

include(GoogleTest)
gtest_discover_tests(LatticeQFTTests)
