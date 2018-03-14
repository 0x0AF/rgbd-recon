add_executable(kinect_client kinect_client.cpp)
target_link_libraries(kinect_client framework glfw ${GLFW_LIBRARIES} imgui)
install(TARGETS kinect_client DESTINATION bin)

add_executable(kinect_client_mc kinect_client_mc.cpp)
target_link_libraries(kinect_client_mc framework glfw ${GLFW_LIBRARIES} imgui)
install(TARGETS kinect_client_mc DESTINATION bin)

add_executable(calib_inverter calib_inverter.cpp)
target_link_libraries(calib_inverter framework)
install(TARGETS calib_inverter DESTINATION bin)