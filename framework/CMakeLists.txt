file(GLOB_RECURSE FRAMEWORK_SOURCE *.cpp *.c)
file(GLOB_RECURSE FRAMEWORK_SOURCE_CUDA *.cu *.cuh)
file(GLOB_RECURSE FRAMEWORK_INLINE *.inl)
file(GLOB_RECURSE FRAMEWORK_HEADER *.hpp)

source_group("Inline Files" FILES ${FRAMEWORK_INLINE})

add_library(framework STATIC ${FRAMEWORK_SOURCE} ${FRAMEWORK_SOURCE_CUDA} ${FRAMEWORK_INLINE} ${FRAMEWORK_HEADER})

set_target_properties(framework PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# self built libraries
target_link_libraries(framework gloost fastdxt ${SQUISH_LIBRARIES} globjects::globjects tinyply ${CUDA_cublas_LIBRARY} ${CUDA_cusolver_LIBRARY} ${CUDA_cusparse_LIBRARY})
# global libraries
target_link_libraries(framework GLU zmq ${BOOST_LIBRARIES} gmp mpfr)

target_include_directories(framework PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(framework PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/calibration)
target_include_directories(framework PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/navigation)
target_include_directories(framework PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/measurement)
target_include_directories(framework PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/rendering)
target_include_directories(framework PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/io)
target_include_directories(framework PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction)
