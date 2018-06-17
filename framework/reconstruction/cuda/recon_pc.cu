#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <GL/gl.h>
#include <GL/glext.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <cuda_gl_interop.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>

#include <reconstruction/cuda/glm.cuh>
#include <reconstruction/cuda/resources.cuh>
#include <reconstruction/cuda/util.cuh>

#include <reconstruction/cuda/copy_reference.cuh>

#include <reconstruction/cuda/ed_sample.cuh>

#include <reconstruction/cuda/pcg_solve.cuh>

#include <reconstruction/cuda/fuse_data.cuh>

extern "C" void init_cuda(glm::uvec3 &volume_res, struct_native_handles &native_handles)
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

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cgr.buffer_vertex_counter, native_handles.buffer_vertex_counter, cudaGraphicsRegisterFlagsReadOnly));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cgr.buffer_reference_mesh_vertices, native_handles.buffer_reference_vertices, cudaGraphicsRegisterFlagsReadOnly));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cgr.buffer_bricks, native_handles.buffer_bricks, cudaGraphicsRegisterFlagsReadOnly));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&_cgr.buffer_occupied, native_handles.buffer_occupied, cudaGraphicsRegisterFlagsReadOnly));
    checkCudaErrors(cudaGraphicsGLRegisterImage(&_cgr.volume_tsdf_data, native_handles.volume_tsdf_data, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    checkCudaErrors(cudaGraphicsGLRegisterImage(&_cgr.array2d_kinect_depths, native_handles.array2d_kinect_depths, GL_TEXTURE_2D_ARRAY_EXT, cudaGraphicsRegisterFlagsReadOnly));

    for(unsigned int i = 0; i < 4; i++)
    {
        checkCudaErrors(cudaGraphicsGLRegisterImage(&_cgr.volume_cv_xyz_inv[i], native_handles.volume_cv_xyz_inv[i], GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    }

    cudaExtent volume_extent = make_cudaExtent(volume_res.x, volume_res.y, volume_res.z);
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
    checkCudaErrors(cudaMalloc3DArray(&_volume_array_tsdf_ref, &channel_desc, volume_extent, cudaArraySurfaceLoadStore));

    cublasCreate(&cublas_handle);
    getLastCudaError("cublasCreate failure");

    cusparseCreate(&cusparse_handle);
    getLastCudaError("cusparseCreate failure");
}

extern "C" void deinit_cuda()
{
    cusparseDestroy(cusparse_handle);
    cublasDestroy(cublas_handle);

    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.buffer_vertex_counter));
    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.buffer_reference_mesh_vertices));
    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.buffer_bricks));
    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.buffer_occupied));
    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.volume_tsdf_data));
    checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.array2d_kinect_depths));

    for(unsigned int i = 0; i < 4; i++)
    {
        checkCudaErrors(cudaGraphicsUnregisterResource(_cgr.volume_cv_xyz_inv[i]));
    }

    if(_volume_array_tsdf_ref != nullptr)
    {
        checkCudaErrors(cudaFree(_volume_array_tsdf_ref));
    }

    if(_ed_graph != nullptr)
    {
        checkCudaErrors(cudaFree(_ed_graph));
    }

    if(_jtf != nullptr)
    {
        checkCudaErrors(cudaFree(_jtf));
    }

  if(_jtj_vals != nullptr)
    {
      checkCudaErrors(cudaFree(_jtj_vals));
    }

  if(_jtj_rows != nullptr)
    {
      checkCudaErrors(cudaFree(_jtj_rows));
    }

  if(_jtj_cols != nullptr)
    {
      checkCudaErrors(cudaFree(_jtj_cols));
    }

    if(_h != nullptr)
    {
        checkCudaErrors(cudaFree(_h));
    }

    if(pcg_Ax != nullptr)
    {
        checkCudaErrors(cudaFree(pcg_Ax));
    }

    if(pcg_omega != nullptr)
    {
        checkCudaErrors(cudaFree(pcg_omega));
    }

    if(pcg_p != nullptr)
    {
        checkCudaErrors(cudaFree(pcg_p));
    }

    cudaDeviceReset();
}
