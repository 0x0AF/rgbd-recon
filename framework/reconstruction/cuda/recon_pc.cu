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

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cgr.buffer_vertex_counter, native_handles.buffer_vertex_counter, cudaGraphicsRegisterFlagsNone));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cgr.buffer_reference_mesh_vertices, native_handles.buffer_reference_vertices, cudaGraphicsRegisterFlagsNone));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cgr.buffer_bricks, native_handles.buffer_bricks, cudaGraphicsRegisterFlagsNone));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cgr.buffer_occupied, native_handles.buffer_occupied, cudaGraphicsRegisterFlagsNone));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cgr.buffer_ed_nodes_debug, native_handles.buffer_ed_nodes_debug, cudaGraphicsRegisterFlagsNone));

    // TODO: rgbs output
    /*checkCudaErrors(cudaGraphicsGLRegisterImage(&_cgr.texture_kinect_rgbs, native_handles.texture_kinect_rgbs,GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore));*/
    checkCudaErrors(cudaGraphicsGLRegisterImage(&_cgr.texture_kinect_depths, native_handles.texture_kinect_depths, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    checkCudaErrors(cudaGraphicsGLRegisterImage(&_cgr.texture_kinect_silhouettes, native_handles.texture_kinect_silhouettes, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore));

    for(unsigned int i = 0; i < 4; i++)
    {
        checkCudaErrors(cudaGraphicsGLRegisterImage(&_cgr.volume_cv_xyz_inv[i], native_handles.volume_cv_xyz_inv[i], GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
        checkCudaErrors(cudaGraphicsGLRegisterImage(&_cgr.volume_cv_xyz[i], native_handles.volume_cv_xyz[i], GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    }

    checkCudaErrors(cudaGraphicsGLRegisterImage(&_cgr.volume_tsdf_data, native_handles.volume_tsdf_data, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore));

    memcpy(&_host_res.measures, &measures, sizeof(struct_measures));

    cudaExtent volume_extent = make_cudaExtent(volume_res.x, volume_res.y, volume_res.z);
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
    checkCudaErrors(cudaMalloc3DArray(&_dev_res.volume_array_tsdf_ref, &channel_desc, volume_extent, cudaArraySurfaceLoadStore));

    cublasCreate(&cublas_handle);
    getLastCudaError("cublasCreate failure");

    cusparseCreate(&cusparse_handle);
    getLastCudaError("cusparseCreate failure");

    cusolverSpCreate(&cusolver_handle);
    getLastCudaError("cusolverSpCreate failure");
}

extern "C" void deinit_cuda()
{
    cusparseDestroy(cusparse_handle);
    cublasDestroy(cublas_handle);
    cusolverSpDestroy(cusolver_handle);

    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.buffer_vertex_counter));
    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.buffer_reference_mesh_vertices));
    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.buffer_bricks));
    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.buffer_occupied));
    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.buffer_ed_nodes_debug));
    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.volume_tsdf_data));

    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.texture_kinect_depths));
    // TODO: rgbs output
    /*checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.texture_kinect_rgbs));*/
    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.texture_kinect_silhouettes));

    for(unsigned int i = 0; i < 4; i++)
    {
        checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.volume_cv_xyz_inv[i]));
        checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.volume_cv_xyz[i]));
    }

    if(_dev_res.volume_array_tsdf_ref != nullptr)
    {
        checkCudaErrors(cudaFreeArray(_dev_res.volume_array_tsdf_ref));
    }

    if(_dev_res.ed_graph != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.ed_graph));
    }

    if(_dev_res.jtf != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.jtf));
    }

    if(_dev_res.jtj_vals != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.jtj_vals));
    }

    if(_dev_res.jtj_rows != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.jtj_rows));
    }

    if(_dev_res.jtj_cols != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.jtj_cols));
    }

    if(_dev_res.h != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.h));
    }

    if(_dev_res.pcg_Ax != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.pcg_Ax));
    }

    if(_dev_res.pcg_omega != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.pcg_omega));
    }

    if(_dev_res.pcg_p != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.pcg_p));
    }

    cudaDeviceReset();
}
