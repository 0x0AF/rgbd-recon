add_executable(kinect_client_pc kinect_client_pc.cpp renderer.cpp renderer.h model.cpp model.h)
target_link_libraries(kinect_client_pc framework glfw ${GLFW_LIBRARIES} imgui)
install(TARGETS kinect_client_pc DESTINATION bin)

add_executable(calib_inverter calib_inverter.cpp)
target_link_libraries(calib_inverter framework)
install(TARGETS calib_inverter DESTINATION bin)