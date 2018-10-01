# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in ${CMAKE_BINARY_DIR}/googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download)
if (result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif ()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download)
if (result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif ()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src
        ${CMAKE_BINARY_DIR}/googletest-build
        EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include")
endif ()

set(CUDA_NVCC_FLAGS –dlink --ftz=true --prec-div=true --prec-sqrt=true --fmad=true -O3 -gencode arch=compute_61,code=sm_61 -lcusparse -lcublas -lcusolver)

add_executable(kinect_client_pc kinect_client_pc.cpp renderer.cpp renderer.h model.cpp model.h FrameSequencer.h FrameSequencer.cpp)

set_property(TARGET kinect_client_pc PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(kinect_client_pc PUBLIC framework glfw ${GLFW_LIBRARIES} imgui tinyply ${CUDA_cublas_LIBRARY} ${CUDA_cusolver_LIBRARY} ${CUDA_cusparse_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES})
install(TARGETS kinect_client_pc DESTINATION bin)

add_executable(unit_tests unit_tests.cpp)

set_property(TARGET unit_tests PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(unit_tests PUBLIC framework glfw ${GLFW_LIBRARIES} imgui tinyply ${CUDA_cublas_LIBRARY} ${CUDA_cusolver_LIBRARY} ${CUDA_cusparse_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES} gtest_main)
add_test(NAME unit_test COMMAND unit_tests)
install(TARGETS unit_tests DESTINATION bin)