file(GLOB_RECURSE FRAMEWORK_SOURCE *.cpp *.c)
file(GLOB_RECURSE FRAMEWORK_SOURCE_CUDA *.cu)
file(GLOB_RECURSE FRAMEWORK_INLINE *.inl)
file(GLOB_RECURSE FRAMEWORK_HEADER *.hpp)

source_group("Inline Files" FILES ${FRAMEWORK_INLINE})

set(CUDA_NVCC_FLAGS –dlink -use_fast_math -O3 -gencode arch=compute_61,code=sm_61 -lcusparse -lcublas -lcusolver)

add_library(framework STATIC ${FRAMEWORK_SOURCE} ${FRAMEWORK_SOURCE_CUDA} ${FRAMEWORK_INLINE} ${FRAMEWORK_HEADER})

set_target_properties(framework PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# self built libraries
target_link_libraries(framework gloost fastdxt ${SQUISH_LIBRARIES} globjects::globjects glbinding::glbinding tinyply ${CUDA_cublas_LIBRARY} ${CUDA_cusolver_LIBRARY} ${CUDA_cusparse_LIBRARY})
# global libraries
target_link_libraries(framework GLU zmq ${BOOST_LIBRARIES} gmp mpfr CGAL::CGAL)

target_include_directories(framework PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(framework PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/calibration)
target_include_directories(framework PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/navigation)
target_include_directories(framework PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/measurement)
target_include_directories(framework PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/rendering)
target_include_directories(framework PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/io)
target_include_directories(framework PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction)
