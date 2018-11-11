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

#include <reconstruction/cuda/SIFT/cudautils.h>

#include <reconstruction/cuda/copy_reference.cuh>
#include <reconstruction/cuda/ed_sample.cuh>
#include <reconstruction/cuda/fuse_data.cuh>
#include <reconstruction/cuda/mc.cuh>
#include <reconstruction/cuda/preprocess.cuh>
#include <reconstruction/cuda/solve.cuh>

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

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cgr.buffer_ed_nodes_debug, native_handles.buffer_ed_nodes_debug, cudaGraphicsRegisterFlagsWriteDiscard));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cgr.buffer_sorted_vertices_debug, native_handles.buffer_sorted_vertices_debug, cudaGraphicsRegisterFlagsWriteDiscard));

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cgr.pbo_kinect_depths, native_handles.pbo_kinect_depths, cudaGraphicsRegisterFlagsReadOnly));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cgr.pbo_kinect_silhouettes, native_handles.pbo_kinect_silhouettes, cudaGraphicsRegisterFlagsReadOnly));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cgr.pbo_kinect_normals, native_handles.pbo_kinect_normals, cudaGraphicsRegisterFlagsReadOnly));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cgr.pbo_opticflow, native_handles.pbo_opticflow, cudaGraphicsRegisterFlagsReadOnly));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cgr.pbo_quality, native_handles.pbo_quality, cudaGraphicsRegisterFlagsReadOnly));

#ifdef PIPELINE_DEBUG_TEXTURE_SILHOUETTES
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cgr.pbo_kinect_silhouettes_debug, native_handles.pbo_kinect_silhouettes_debug, cudaGraphicsRegisterFlagsWriteDiscard));
#endif

#ifdef PIPELINE_DEBUG_TEXTURE_ALIGNMENT_ERROR
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cgr.pbo_kinect_alignment_error_debug, native_handles.pbo_kinect_alignment_error_debug, cudaGraphicsRegisterFlagsWriteDiscard));
#endif

    for(unsigned int i = 0; i < 4; i++)
    {
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cgr.pbo_cv_xyz_inv[i], native_handles.pbo_cv_xyz_inv[i], cudaGraphicsRegisterFlagsReadOnly));
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cgr.pbo_cv_xyz[i], native_handles.pbo_cv_xyz[i], cudaGraphicsRegisterFlagsReadOnly));
    }

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cgr.posvbo, native_handles.posvbo, cudaGraphicsMapFlagsWriteDiscard));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cgr.normalvbo, native_handles.normalvbo, cudaGraphicsMapFlagsWriteDiscard));

    checkCudaErrors(cudaGraphicsGLRegisterImage(&_cgr.volume_tsdf_data, native_handles.volume_tsdf_data, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    checkCudaErrors(cudaGraphicsGLRegisterImage(&_cgr.volume_tsdf_ref_grad, native_handles.volume_tsdf_ref_grad, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore));

    memcpy(&_host_res.measures, &measures, sizeof(struct_measures));

    for(unsigned int i = 0; i < 4; i++)
    {
        checkCudaErrors(cudaMallocPitch(&_dev_res.kinect_depths[i], &_dev_res.pitch_kinect_depths, _host_res.measures.depth_res.x * sizeof(float), _host_res.measures.depth_res.y));
        checkCudaErrors(cudaMallocPitch(&_dev_res.kinect_depths_prev[i], &_dev_res.pitch_kinect_depths_prev, _host_res.measures.depth_res.x * sizeof(float), _host_res.measures.depth_res.y));
        checkCudaErrors(cudaMallocPitch(&_dev_res.kinect_silhouettes[i], &_dev_res.pitch_kinect_silhouettes, _host_res.measures.depth_res.x * sizeof(float), _host_res.measures.depth_res.y));
        checkCudaErrors(cudaMallocPitch(&_dev_res.kinect_normals[i], &_dev_res.pitch_kinect_normals, _host_res.measures.depth_res.x * sizeof(float4), _host_res.measures.depth_res.y));
        checkCudaErrors(cudaMallocPitch(&_dev_res.alignment_error[i], &_dev_res.pitch_alignment_error, _host_res.measures.depth_res.x * sizeof(float), _host_res.measures.depth_res.y));
        checkCudaErrors(cudaMallocPitch(&_dev_res.quality[i], &_dev_res.pitch_quality, _host_res.measures.depth_res.x * sizeof(float), _host_res.measures.depth_res.y));
        checkCudaErrors(cudaMallocPitch(&_dev_res.optical_flow[i], &_dev_res.pitch_optical_flow, 2048 * sizeof(float2), 1696));
    }

    checkCudaErrors(cudaMalloc(&_dev_res.cloud_noise, _host_res.measures.depth_res.x * _host_res.measures.depth_res.y * sizeof(float)));

    float *cloud_noise = (float *)malloc(_host_res.measures.depth_res.x * _host_res.measures.depth_res.y * sizeof(float));
    char *cloud_data_ptr = cloud_data;

    for(int i = 0; i < _host_res.measures.depth_res.x * _host_res.measures.depth_res.y; i++)
    {
        int pixel = (((cloud_data_ptr[0] - 33) << 2) | ((cloud_data_ptr[1] - 33) >> 4));
        cloud_noise[i] = ((float)pixel) / 256.f;

        // printf("\npixel: %i\n", pixel);

        cloud_data_ptr += 4;
    }

    cudaMemcpy(&_dev_res.cloud_noise[0], &cloud_noise[0], _host_res.measures.depth_res.x * _host_res.measures.depth_res.y * sizeof(float), cudaMemcpyHostToDevice);

    free(cloud_noise);

    // printf("\npitch: %lu\n", _dev_res.pitch_kinect_depths);

    checkCudaErrors(cudaMalloc(&_dev_res.tsdf_ref, _host_res.measures.data_volume_res.x * _host_res.measures.data_volume_res.y * _host_res.measures.data_volume_res.z * sizeof(float2)));
    checkCudaErrors(cudaMalloc(&_dev_res.tsdf_ref_warped, _host_res.measures.data_volume_res.x * _host_res.measures.data_volume_res.y * _host_res.measures.data_volume_res.z * sizeof(float2)));
    checkCudaErrors(cudaMalloc(&_dev_res.tsdf_ref_warped_marks, _host_res.measures.data_volume_res.x * _host_res.measures.data_volume_res.y * _host_res.measures.data_volume_res.z * sizeof(float1)));
    checkCudaErrors(cudaMalloc(&_dev_res.tsdf_fused, _host_res.measures.data_volume_res.x * _host_res.measures.data_volume_res.y * _host_res.measures.data_volume_res.z * sizeof(float2)));

    checkCudaErrors(cudaMalloc(&_dev_res.out_tsdf_data, _host_res.measures.data_volume_res.x * _host_res.measures.data_volume_res.y * _host_res.measures.data_volume_res.z * sizeof(uchar)));
    checkCudaErrors(cudaMalloc(&_dev_res.out_tsdf_ref, _host_res.measures.data_volume_res.x * _host_res.measures.data_volume_res.y * _host_res.measures.data_volume_res.z * sizeof(uchar)));
    checkCudaErrors(cudaMalloc(&_dev_res.out_tsdf_warped_ref, _host_res.measures.data_volume_res.x * _host_res.measures.data_volume_res.y * _host_res.measures.data_volume_res.z * sizeof(uchar)));
    checkCudaErrors(cudaMalloc(&_dev_res.out_tsdf_fused, _host_res.measures.data_volume_res.x * _host_res.measures.data_volume_res.y * _host_res.measures.data_volume_res.z * sizeof(uchar)));

    checkCudaErrors(cudaMalloc(&_dev_res.pos, MAX_REFERENCE_VERTICES * 4 * sizeof(float)));
    checkCudaErrors(cudaMalloc(&_dev_res.normal, MAX_REFERENCE_VERTICES * 4 * sizeof(float)));

    _host_res.silhouette = (float *)malloc(_host_res.measures.depth_res.x * _host_res.measures.depth_res.y * sizeof(float));
    _host_res.vx_error_map = (float *)malloc(_host_res.measures.depth_res.x * _host_res.measures.depth_res.y * sizeof(float));
    _host_res.vx_error_values = (float *)malloc(MAX_REFERENCE_VERTICES * sizeof(float));

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
    allocate_pcg_resources();

    /*sift_front = (SiftData *)malloc(4 * sizeof(SiftData));
    sift_back = (SiftData *)malloc(4 * sizeof(SiftData));

    for(int i = 0; i < 4; i++)
    {
        InitSiftData(sift_front[i], SIFT_MAX_CORRESPONDENCES, true, true);
        InitSiftData(sift_back[i], SIFT_MAX_CORRESPONDENCES, true, true);
    }*/

    map_calibration_volumes();

    init_mc(volume_res, measures, native_handles);
}

extern "C" void deinit_cuda()
{
    unmap_calibration_volumes();

    /*for(int i = 0; i < 4; i++)
    {
        FreeSiftData(sift_front[i]);
        FreeSiftData(sift_back[i]);
    }

    free(sift_back);
    free(sift_front);*/

    free_pcg_resources();
    free_ed_resources();
    free_brick_resources();

    free(_host_res.vx_error_values);
    free(_host_res.vx_error_map);
    free(_host_res.silhouette);

    free(_host_res.kernel_gauss);

    cusparseDestroy(cusparse_handle);
    cublasDestroy(cublas_handle);
    cusolverSpDestroy(cusolver_handle);

    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.posvbo));
    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.normalvbo));

    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.buffer_bricks));
    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.buffer_occupied));
    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.buffer_ed_nodes_debug));
    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.buffer_sorted_vertices_debug));

    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.volume_tsdf_data));
    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.volume_tsdf_ref_grad));

    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.pbo_kinect_depths));
    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.pbo_kinect_silhouettes));

#ifdef PIPELINE_DEBUG_TEXTURE_SILHOUETTES
    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.pbo_kinect_silhouettes_debug));
#endif

#ifdef PIPELINE_DEBUG_TEXTURE_SILHOUETTES
    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.pbo_kinect_alignment_error_debug));
#endif

    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.pbo_opticflow));
    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.pbo_quality));

    for(unsigned int i = 0; i < 4; i++)
    {
        checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.pbo_cv_xyz_inv[i]));
        checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.pbo_cv_xyz[i]));
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

    if(_dev_res.quality != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.quality));
    }

    if(_dev_res.optical_flow != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.optical_flow));
    }

    if(_dev_res.alignment_error != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.alignment_error));
    }

    if(_dev_res.tsdf_ref_warped != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.tsdf_ref_warped));
    }

    if(_dev_res.tsdf_ref_warped_marks != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.tsdf_ref_warped_marks));
    }

    if(_dev_res.tsdf_ref != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.tsdf_ref));
    }

    if(_dev_res.out_tsdf_data != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.out_tsdf_data));
    }

    if(_dev_res.out_tsdf_ref != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.out_tsdf_ref));
    }

    if(_dev_res.out_tsdf_warped_ref != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.out_tsdf_warped_ref));
    }

    if(_dev_res.out_tsdf_fused != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.out_tsdf_fused));
    }

    cudaDeviceReset();
}
