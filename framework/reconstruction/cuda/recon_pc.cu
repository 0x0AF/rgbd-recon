#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <GL/gl.h>
#include <GL/glext.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <cublas_v2.h>
#include <cuda_gl_interop.h>
#include <cusparse_v2.h>

#include <reconstruction/cuda/glm.cuh>
#include <reconstruction/cuda/resources.cuh>
#include <reconstruction/cuda/util.cuh>

#include <reconstruction/cuda/copy_reference.cuh>
#include <reconstruction/cuda/ed_sample.cuh>
#include <reconstruction/cuda/fuse_data.cuh>
#include <reconstruction/cuda/pcg_solve.cuh>
#include <reconstruction/cuda/preprocess.cuh>
#include <reconstruction/cuda/sift.cuh>

extern "C" void init_cuda(glm::uvec3 &volume_res, struct_measures &measures, struct_native_handles &native_handles)
{
    cudaDeviceReset();

    int devicesCount;
    cudaGetDeviceCount(&devicesCount);
    for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
    {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, deviceIndex);
        if(deviceProperties.major >= 6 && deviceProperties.minor >= 1)
        {
            cudaSetDevice(deviceIndex);
            cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
            cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
            cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
        }
    }

    size_t value;

    cudaDeviceGetLimit(&value, cudaLimitMallocHeapSize);
    printf("\n\nLimit Malloc Heap Size: %lu\n", value);

    cudaDeviceGetLimit(&value, cudaLimitStackSize);
    printf("\nLimit Stack Size: %lu\n", value);

    cudaDeviceGetLimit(&value, cudaLimitDevRuntimeSyncDepth);
    printf("\nLimit Sync Depth: %lu\n", value);

    cudaDeviceGetLimit(&value, cudaLimitDevRuntimePendingLaunchCount);
    printf("\nLimit Pending Launch: %lu\n\n", value);

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cgr.buffer_bricks, native_handles.buffer_bricks, cudaGraphicsRegisterFlagsReadOnly));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cgr.buffer_occupied, native_handles.buffer_occupied, cudaGraphicsRegisterFlagsReadOnly));

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cgr.buffer_vertex_counter, native_handles.buffer_vertex_counter, cudaGraphicsRegisterFlagsReadOnly));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cgr.buffer_reference_mesh_vertices, native_handles.buffer_reference_vertices, cudaGraphicsRegisterFlagsNone));

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cgr.buffer_ed_nodes_debug, native_handles.buffer_ed_nodes_debug, cudaGraphicsRegisterFlagsWriteDiscard));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cgr.buffer_sorted_vertices_debug, native_handles.buffer_sorted_vertices_debug, cudaGraphicsRegisterFlagsWriteDiscard));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cgr.buffer_correspondences_debug, native_handles.buffer_correspondences_debug, cudaGraphicsRegisterFlagsWriteDiscard));

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cgr.pbo_kinect_rgbs, native_handles.pbo_kinect_rgbs, cudaGraphicsRegisterFlagsReadOnly));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cgr.pbo_kinect_depths, native_handles.pbo_kinect_depths, cudaGraphicsRegisterFlagsReadOnly));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cgr.pbo_kinect_silhouettes, native_handles.pbo_kinect_silhouettes, cudaGraphicsRegisterFlagsReadOnly));

#ifdef PIPELINE_DEBUG_TEXTURE_SILHOUETTES
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cgr.pbo_kinect_silhouettes_debug, native_handles.pbo_kinect_silhouettes_debug, cudaGraphicsRegisterFlagsWriteDiscard));
#endif

#ifdef PIPELINE_DEBUG_TEXTURE_CORRESPONDENCES
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cgr.pbo_kinect_intens_debug, native_handles.pbo_kinect_intens_debug, cudaGraphicsRegisterFlagsWriteDiscard));
#endif

    for(unsigned int i = 0; i < 4; i++)
    {
        checkCudaErrors(cudaGraphicsGLRegisterImage(&_cgr.volume_cv_xyz_inv[i], native_handles.volume_cv_xyz_inv[i], GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
        checkCudaErrors(cudaGraphicsGLRegisterImage(&_cgr.volume_cv_xyz[i], native_handles.volume_cv_xyz[i], GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    }

    checkCudaErrors(cudaGraphicsGLRegisterImage(&_cgr.volume_tsdf_data, native_handles.volume_tsdf_data, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    checkCudaErrors(cudaGraphicsGLRegisterImage(&_cgr.volume_tsdf_ref, native_handles.volume_tsdf_ref, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore));

    memcpy(&_host_res.measures, &measures, sizeof(struct_measures));

    checkCudaErrors(cudaMalloc(&_dev_res.kinect_rgbs, _host_res.measures.color_res.x * _host_res.measures.color_res.y * 4 * sizeof(float4)));
    checkCudaErrors(cudaMalloc(&_dev_res.kinect_intens, _host_res.measures.depth_res.x * _host_res.measures.depth_res.y * 4 * sizeof(float1)));
    checkCudaErrors(cudaMalloc(&_dev_res.kinect_depths, _host_res.measures.depth_res.x * _host_res.measures.depth_res.y * 4 * sizeof(float2)));
    checkCudaErrors(cudaMalloc(&_dev_res.kinect_depths_prev, _host_res.measures.depth_res.x * _host_res.measures.depth_res.y * 4 * sizeof(float2)));
    checkCudaErrors(cudaMalloc(&_dev_res.kinect_silhouettes, _host_res.measures.depth_res.x * _host_res.measures.depth_res.y * 4 * sizeof(float1)));

    _host_res.kernel_gauss = (float *)malloc(KERNEL_LENGTH * sizeof(float));

    _host_res.kernel_gauss[0] = 0.012318f;
    _host_res.kernel_gauss[1] = 0.014381f;
    _host_res.kernel_gauss[2] = 0.016624f;
    _host_res.kernel_gauss[3] = 0.019024f;
    _host_res.kernel_gauss[4] = 0.021555f;
    _host_res.kernel_gauss[5] = 0.02418f;
    _host_res.kernel_gauss[6] = 0.026854f;
    _host_res.kernel_gauss[7] = 0.029528f;
    _host_res.kernel_gauss[8] = 0.032145f;
    _host_res.kernel_gauss[9] = 0.034647f;
    _host_res.kernel_gauss[10] = 0.036972f;
    _host_res.kernel_gauss[11] = 0.03906f;
    _host_res.kernel_gauss[12] = 0.040857f;
    _host_res.kernel_gauss[13] = 0.042311f;
    _host_res.kernel_gauss[14] = 0.043381f;
    _host_res.kernel_gauss[15] = 0.044036f;
    _host_res.kernel_gauss[16] = 0.044256f;
    _host_res.kernel_gauss[17] = 0.044036f;
    _host_res.kernel_gauss[18] = 0.043381f;
    _host_res.kernel_gauss[19] = 0.042311f;
    _host_res.kernel_gauss[20] = 0.040857f;
    _host_res.kernel_gauss[21] = 0.03906f;
    _host_res.kernel_gauss[22] = 0.036972f;
    _host_res.kernel_gauss[23] = 0.034647f;
    _host_res.kernel_gauss[24] = 0.032145f;
    _host_res.kernel_gauss[25] = 0.029528f;
    _host_res.kernel_gauss[26] = 0.026854f;
    _host_res.kernel_gauss[27] = 0.02418f;
    _host_res.kernel_gauss[28] = 0.021555f;
    _host_res.kernel_gauss[29] = 0.019024f;
    _host_res.kernel_gauss[30] = 0.016624f;
    _host_res.kernel_gauss[31] = 0.014381f;
    _host_res.kernel_gauss[32] = 0.012318f;

    cublasCreate(&cublas_handle);
    getLastCudaError("cublasCreate failure");

    cusparseCreate(&cusparse_handle);
    getLastCudaError("cusparseCreate failure");

    cusolverSpCreate(&cusolver_handle);
    getLastCudaError("cusolverSpCreate failure");

    allocate_brick_resources();
    allocate_ed_resources();
    allocate_correspondence_resources();

    sift_front = (SiftData *)malloc(4 * sizeof(SiftData));
    sift_back = (SiftData *)malloc(4 * sizeof(SiftData));

    for(int i = 0; i < 4; i++)
    {
        InitSiftData(sift_front[i], SIFT_MAX_CORRESPONDENCES, true, true);
        InitSiftData(sift_back[i], SIFT_MAX_CORRESPONDENCES, true, true);
    }
}

extern "C" void deinit_cuda()
{
    for(int i = 0; i < 4; i++)
    {
        FreeSiftData(sift_front[i]);
        FreeSiftData(sift_back[i]);
    }

    free(sift_back);
    free(sift_front);

    free_correspondence_resources();
    free_ed_resources();
    free_brick_resources();

    free(_host_res.kernel_gauss);

    cusparseDestroy(cusparse_handle);
    cublasDestroy(cublas_handle);
    cusolverSpDestroy(cusolver_handle);

    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.buffer_vertex_counter));
    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.buffer_reference_mesh_vertices));
    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.buffer_bricks));
    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.buffer_occupied));
    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.buffer_ed_nodes_debug));
    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.buffer_sorted_vertices_debug));
    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.buffer_correspondences_debug));

    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.volume_tsdf_data));
    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.volume_tsdf_ref));

    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.pbo_kinect_rgbs));
    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.pbo_kinect_depths));
    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.pbo_kinect_silhouettes));

#ifdef PIPELINE_DEBUG_TEXTURE_SILHOUETTES
    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.pbo_kinect_silhouettes_debug));
#endif

#ifdef PIPELINE_DEBUG_TEXTURE_CORRESPONDENCES
    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.pbo_kinect_intens_debug));
#endif

    for(unsigned int i = 0; i < 4; i++)
    {
        checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.volume_cv_xyz_inv[i]));
        checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.volume_cv_xyz[i]));
    }

    if(_dev_res.kinect_rgbs != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.kinect_rgbs));
    }

    if(_dev_res.kinect_intens != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.kinect_intens));
    }

    if(_dev_res.kinect_depths != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.kinect_depths));
    }

    if(_dev_res.kinect_depths_prev != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.kinect_depths_prev));
    }

    if(_dev_res.kinect_silhouettes != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.kinect_silhouettes));
    }

    free_brick_resources();
    free_ed_resources();

    cudaDeviceReset();
}
