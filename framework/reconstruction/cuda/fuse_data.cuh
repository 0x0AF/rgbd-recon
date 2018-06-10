#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <reconstruction/cuda/glm.cuh>
#include <reconstruction/cuda/resources.cuh>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>

__global__ void kernel_fuse_data(GLuint *occupied_bricks, size_t occupied_brick_count)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= occupied_brick_count)
    {
        return;
    }

    unsigned int brick_id = occupied_bricks[idx];

    if(brick_id == 0u)
    {
        return;
    }

    glm::uvec3 brick = glm::uvec3(0u);
    brick.z = brick_id / (BRICK_RES * BRICK_RES);
    brick_id %= (BRICK_RES * BRICK_RES);
    brick.y = brick_id / BRICK_RES;
    brick_id %= BRICK_RES;
    brick.x = brick_id;

    // printf("\nbrick %u: (%u,%u,%u)\n", brick_id, brick.x, brick.y, brick.z);

    for(unsigned int i = 0u; i < BRICK_VOXELS; i++)
    {
        unsigned int position_id = i;

        glm::uvec3 position = glm::uvec3(0u);
        position.z = position_id / (BRICK_VOXEL_DIM * BRICK_VOXEL_DIM);
        position_id %= (BRICK_VOXEL_DIM * BRICK_VOXEL_DIM);
        position.y = position_id / BRICK_VOXEL_DIM;
        position_id %= (BRICK_VOXEL_DIM);
        position.x = position_id;

        glm::uvec3 world = brick * BRICK_VOXEL_DIM + position;

        if(world.x >= VOLUME_VOXEL_DIM || world.y >= VOLUME_VOXEL_DIM || world.z >= VOLUME_VOXEL_DIM)
        {
            continue;
        }

        // printf("\nbrick %u, position %u: (%u,%u,%u)\n", occupied_bricks[idx], i, world.x, world.y, world.z);

        float2 data, ref;
        surf3Dread(&data, _volume_tsdf_data, world.x * sizeof(float2), world.y, world.z);
        surf3Dread(&ref, _volume_tsdf_ref, world.x * sizeof(float2), world.y, world.z);

        float2 fused;

        fused.y = ref.y + data.y;

        if(fused.y > 0.001f)
        {
            fused.x = data.x * data.y / fused.y + ref.x * ref.y / fused.y;
        }
        else
        {
            fused.x = data.y > ref.y ? data.x : ref.x;
        }

        surf3Dwrite(ref, _volume_tsdf_data, world.x * sizeof(float2), world.y, world.z);
    }
}

extern "C" void fuse_data()
{
    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.volume_tsdf_data, 0));
    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.buffer_occupied, 0));

    cudaArray *volume_array_tsdf_data = nullptr;
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&volume_array_tsdf_data, _cgr.volume_tsdf_data, 0, 0));

    size_t occupied_brick_bytes;
    GLuint *brick_list;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&brick_list, &occupied_brick_bytes, _cgr.buffer_occupied));

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
    checkCudaErrors(cudaBindSurfaceToArray(&_volume_tsdf_data, volume_array_tsdf_data, &channel_desc));
    checkCudaErrors(cudaBindSurfaceToArray(&_volume_tsdf_ref, _volume_array_tsdf_ref, &channel_desc));

    int block_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_fuse_data, 0, 0);

    unsigned max_bricks = ((unsigned)occupied_brick_bytes) / sizeof(unsigned);
    size_t grid_size = (max_bricks + block_size - 1) / block_size;
    kernel_fuse_data<<<grid_size, block_size>>>(brick_list, max_bricks);

    getLastCudaError("render kernel failed");

    cudaDeviceSynchronize();

    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.buffer_occupied, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.volume_tsdf_data, 0));
}
