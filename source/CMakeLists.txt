add_executable(kinect_client_pc kinect_client_pc.cpp renderer.cpp renderer.h model.cpp model.h)

set_property(TARGET kinect_client_pc PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(kinect_client_pc PUBLIC framework glfw ${GLFW_LIBRARIES} imgui tinyply)
install(TARGETS kinect_client_pc DESTINATION bin)