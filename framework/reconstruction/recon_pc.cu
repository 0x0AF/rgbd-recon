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
#include <iostream>
#include <texture_types.h>
#include <vector_types.h>

surface<void, 3> volume_tsdf_data;

cudaGraphicsResource *cuda_graphics_resource = nullptr;
cudaArray *volume_array_tsdf_data = nullptr;
cudaExtent volume_res;

__global__ void kernel_warp(void *vx_ptr, uint64_t num_vertices)
{
    // TODO: calculate JTJ and JTf in CUDA

    // TODO: run linear solver (cuSOLVER)
}

__global__ void kernel_fuse_volume(cudaExtent volume_res)
{
    // TODO: blend with warped reference mesh

    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;

    uint32_t u = (uint32_t)((float)x / (float)8 * volume_res.width);
    uint32_t v = (uint32_t)((float)y / (float)8 * volume_res.height);
    uint32_t d = (uint32_t)((float)z / (float)8 * volume_res.depth);

    float data = 0.0;
    surf3Dwrite(data, volume_tsdf_data, u * sizeof(float), v, d);
}

extern "C" void init_cuda(uint16_t res_x, uint16_t res_y, uint16_t res_z) { volume_res = make_cudaExtent(res_x, res_y, res_z); }

extern "C" void deinit_cuda() {}

extern "C" void align_non_rigid(GLuint buffer_reference_mesh_vertices, GLuint buffer_vertex_counter, GLuint volume_tsdf_data_id)
{
    // TODO: bind both buffers, extract pointer to reference mesh vertices, fuse into data volume

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_graphics_resource, volume_tsdf_data_id, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_graphics_resource, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&volume_array_tsdf_data, cuda_graphics_resource, 0, 0));
    checkCudaErrors(cudaBindSurfaceToArray(&volume_tsdf_data, volume_array_tsdf_data, &channel_desc));

    kernel_fuse_volume<<<dim3(2, 2, 2), dim3(4, 4, 4)>>>(volume_res);

    getLastCudaError("render kernel failed");

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_graphics_resource, 0));
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_graphics_resource));
}