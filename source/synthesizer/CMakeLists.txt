find_package(ASSIMP REQUIRED)

file(GLOB_RECURSE FRAMEWORK_SOURCE_CUDA *.cu)

set(CUDA_NVCC_FLAGS –dlink -use_fast_math -O3 -gencode arch=compute_61,code=sm_61 -lcusparse -lcublas -lcusolver)

add_executable(synthetic_rgbd main.cpp Renderer.h Renderer.cpp Controller.h Controller.cpp FileBuffer.hpp FileBuffer.cpp Choreographer.h Choreographer.cpp ../FrameSequencer.h ../FrameSequencer.cpp ${FRAMEWORK_SOURCE_CUDA})

set_property(TARGET synthetic_rgbd PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(synthetic_rgbd PUBLIC ${OPENGL_INCLUDE_DIR} ${ASSIMP_INCLUDE_DIR})
target_link_libraries(synthetic_rgbd PUBLIC framework glfw ${GLFW_LIBRARIES} imgui tinyply ${CUDA_cublas_LIBRARY} ${CUDA_cusolver_LIBRARY} ${CUDA_cusparse_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES})
target_link_libraries(synthetic_rgbd PUBLIC ${OPENGL_gl_LIBRARY} glfw ${ASSIMP_LIBRARIES})
target_link_libraries(synthetic_rgbd PUBLIC globjects::globjects glbinding::glbinding)

install(TARGETS synthetic_rgbd DESTINATION bin)