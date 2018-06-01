#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <GL/gl.h>
#include <GL/glext.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_math.h>

#include <device_launch_parameters.h>
#include <driver_types.h>
#include <glm/vec3.hpp>
#include <iostream>
#include <texture_types.h>
#include <vector_types.h>

surface<void, 3> volume_tsdf_data;
surface<void, 3> volume_tsdf_ref;

cudaExtent volume_res;

struct struct_vertex
{
    glm::vec3 _position;
    uint32_t pad_1;
    glm::vec3 _normal;
    uint32_t pad_2;
};

__global__ void kernel_copy_reference(cudaExtent volume_res)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;

    uint32_t u = (uint32_t)((float)x / (float)64 * volume_res.width);
    uint32_t v = (uint32_t)((float)y / (float)64 * volume_res.height);
    uint32_t d = (uint32_t)((float)z / (float)64 * volume_res.depth);

    float2 data;
    surf3Dread(&data, volume_tsdf_data, u * sizeof(float2), v, d);
    surf3Dwrite(data, volume_tsdf_ref, u * sizeof(float2), v, d);
}

__global__ void kernel_warp(cudaExtent volume_res, GLuint *vx_counter, struct_vertex *vx_ptr)
{
    // TODO: calculate JTJ and JTf in CUDA

    // TODO: run linear solver (cuSOLVER)

    const unsigned long long int block_id = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    const unsigned long long int thread_id = block_id * blockDim.x + threadIdx.x;

    GLuint counter = vx_counter[0];

    // printf("\nvx count: %u, thread: %llu\n", counter, thread_id);

    if(thread_id < counter)
    {
        // printf("\nvx: (%f,%f,%f):(%f,%f,%f)\n", vx_ptr[thread_id]._position.x, vx_ptr[thread_id]._position.y, vx_ptr[thread_id]._position.z,
        //  vx_ptr[thread_id]._normal.x, vx_ptr[thread_id]._normal.y, vx_ptr[thread_id]._normal.z);

        uint32_t u = (uint32_t)(vx_ptr[thread_id]._position.x * volume_res.width);
        uint32_t v = (uint32_t)(vx_ptr[thread_id]._position.y * volume_res.height);
        uint32_t d = (uint32_t)(vx_ptr[thread_id]._position.z * volume_res.depth);

        // float data = 0.0;
        // surf3Dwrite(data, volume_tsdf_data, u * sizeof(float), v, d);
    }
}

__global__ void kernel_fuse_volume(cudaExtent volume_res)
{
    // TODO: blend with warped reference mesh using gradient

    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;

    uint32_t u = (uint32_t)((float)x / (float)64 * volume_res.width);
    uint32_t v = (uint32_t)((float)y / (float)64 * volume_res.height);
    uint32_t d = (uint32_t)((float)z / (float)64 * volume_res.depth);

    float2 data, ref;
    surf3Dread(&data, volume_tsdf_data, u * sizeof(float2), v, d);
    surf3Dread(&ref, volume_tsdf_ref, u * sizeof(float2), v, d);

    float2 fused;

    fused.y = ref.y + data.y;

    if(fused.y > 0.01f)
    {
        fused.x = (data.x * data.y + ref.x * ref.y) / (ref.y + data.y);
    }
    else
    {
        fused.x = data.x;
    }

    surf3Dwrite(fused, volume_tsdf_data, u * sizeof(float2), v, d);
}

extern "C" void init_cuda(uint16_t res_x, uint16_t res_y, uint16_t res_z)
{
    volume_res = make_cudaExtent(res_x, res_y, res_z);

    int devicesCount;
    cudaGetDeviceCount(&devicesCount);
    for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
    {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, deviceIndex);
        if(deviceProperties.major >= 2 && deviceProperties.minor >= 0)
        {
            cudaSetDevice(deviceIndex);
        }
    }
}

extern "C" void deinit_cuda() {}

extern "C" void copy_reference_volume(GLuint volume_tsdf_data_id, GLuint volume_tsdf_reference_id)
{
    cudaGraphicsResource *cgr_volume_tsdf_data = nullptr;
    cudaGraphicsResource *cgr_volume_tsdf_ref = nullptr;

    cudaArray *volume_array_tsdf_data = nullptr;
    cudaArray *volume_array_tsdf_ref = nullptr;

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);

    checkCudaErrors(cudaGraphicsGLRegisterImage(&cgr_volume_tsdf_data, volume_tsdf_data_id, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    checkCudaErrors(cudaGraphicsGLRegisterImage(&cgr_volume_tsdf_ref, volume_tsdf_reference_id, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore));

    checkCudaErrors(cudaGraphicsMapResources(1, &cgr_volume_tsdf_data, 0));
    checkCudaErrors(cudaGraphicsMapResources(1, &cgr_volume_tsdf_ref, 0));

    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&volume_array_tsdf_data, cgr_volume_tsdf_data, 0, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&volume_array_tsdf_ref, cgr_volume_tsdf_ref, 0, 0));

    checkCudaErrors(cudaBindSurfaceToArray(&volume_tsdf_data, volume_array_tsdf_data, &channel_desc));
    checkCudaErrors(cudaBindSurfaceToArray(&volume_tsdf_ref, volume_array_tsdf_ref, &channel_desc));

    uint16_t res_blocks = (uint16_t)(volume_res.height / 4);
    kernel_copy_reference<<<dim3(res_blocks, res_blocks, res_blocks), dim3(4, 4, 4)>>>(volume_res);

    getLastCudaError("render kernel failed");

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cgr_volume_tsdf_ref, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cgr_volume_tsdf_data, 0));

    checkCudaErrors(cudaGraphicsUnregisterResource(cgr_volume_tsdf_ref));
    checkCudaErrors(cudaGraphicsUnregisterResource(cgr_volume_tsdf_data));
}

extern "C" void align_non_rigid(GLuint buffer_reference_mesh_vertices, GLuint buffer_vertex_counter, GLuint volume_tsdf_data_id, GLuint volume_tsdf_reference_id)
{
    // TODO: bind both buffers, extract pointer to reference mesh vertices, fuse into data volume

    cudaGraphicsResource *cgr_buffer_reference_mesh_vertices = nullptr;
    cudaGraphicsResource *cgr_buffer_vertex_counter = nullptr;
    cudaGraphicsResource *cgr_volume_tsdf_data = nullptr;
    cudaGraphicsResource *cgr_volume_tsdf_ref = nullptr;

    cudaArray *volume_array_tsdf_data = nullptr;
    cudaArray *volume_array_tsdf_ref = nullptr;

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);

    size_t vx_bytes;
    GLuint *vx_counter;
    struct_vertex *vx_ptr;

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cgr_buffer_vertex_counter, buffer_vertex_counter, cudaGraphicsRegisterFlagsReadOnly));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cgr_buffer_reference_mesh_vertices, buffer_reference_mesh_vertices, cudaGraphicsRegisterFlagsReadOnly));
    checkCudaErrors(cudaGraphicsGLRegisterImage(&cgr_volume_tsdf_data, volume_tsdf_data_id, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    checkCudaErrors(cudaGraphicsGLRegisterImage(&cgr_volume_tsdf_ref, volume_tsdf_reference_id, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore));

    checkCudaErrors(cudaGraphicsMapResources(1, &cgr_buffer_vertex_counter, 0));
    checkCudaErrors(cudaGraphicsMapResources(1, &cgr_buffer_reference_mesh_vertices, 0));
    checkCudaErrors(cudaGraphicsMapResources(1, &cgr_volume_tsdf_data, 0));
    checkCudaErrors(cudaGraphicsMapResources(1, &cgr_volume_tsdf_ref, 0));

    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&vx_counter, &vx_bytes, cgr_buffer_vertex_counter));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&vx_ptr, &vx_bytes, cgr_buffer_reference_mesh_vertices));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&volume_array_tsdf_data, cgr_volume_tsdf_data, 0, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&volume_array_tsdf_ref, cgr_volume_tsdf_ref, 0, 0));

    checkCudaErrors(cudaBindSurfaceToArray(&volume_tsdf_data, volume_array_tsdf_data, &channel_desc));
    checkCudaErrors(cudaBindSurfaceToArray(&volume_tsdf_ref, volume_array_tsdf_ref, &channel_desc));

    kernel_warp<<<dim3(4, 4, 4), dim3(4, 4, 4)>>>(volume_res, vx_counter, vx_ptr);

    getLastCudaError("render kernel failed");

    uint16_t res_blocks = (uint16_t)(volume_res.height / 4);
    kernel_fuse_volume<<<dim3(res_blocks, res_blocks, res_blocks), dim3(4, 4, 4)>>>(volume_res);

    getLastCudaError("render kernel failed");

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cgr_volume_tsdf_ref, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cgr_volume_tsdf_data, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cgr_buffer_reference_mesh_vertices, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cgr_buffer_vertex_counter, 0));

    checkCudaErrors(cudaGraphicsUnregisterResource(cgr_volume_tsdf_ref));
    checkCudaErrors(cudaGraphicsUnregisterResource(cgr_volume_tsdf_data));
    checkCudaErrors(cudaGraphicsUnregisterResource(cgr_buffer_reference_mesh_vertices));
    checkCudaErrors(cudaGraphicsUnregisterResource(cgr_buffer_vertex_counter));
}
