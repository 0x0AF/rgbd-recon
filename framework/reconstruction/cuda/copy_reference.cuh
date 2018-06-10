#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <reconstruction/cuda/glm.cuh>
#include <reconstruction/cuda/resources.cuh>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>

__global__ void kernel_copy_reference(unsigned int *active_bricks, unsigned int *bricks_inv_index, GLuint *occupied_bricks, size_t occupied_brick_count)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= occupied_brick_count)
    {
        return;
    }

    unsigned int brick_id = occupied_bricks[idx];

    // printf("\nidx %u, brick %u", idx, brick_id);

    if(brick_id == 0u)
    {
        return;
    }

    unsigned int brick_position = atomicAdd(active_bricks, 1u);
    bricks_inv_index[brick_id] = brick_position;

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

        float2 data;
        surf3Dread(&data, _volume_tsdf_data, world.x * sizeof(float2), world.y, world.z);
        surf3Dwrite(data, _volume_tsdf_ref, world.x * sizeof(float2), world.y, world.z);
    }
}

extern "C" void copy_reference()
{
    unsigned int *active_bricks_count;
    cudaMallocManaged(&active_bricks_count, sizeof(unsigned int));
    *active_bricks_count = 0u;

    if(_bricks_inv_index != nullptr)
    {
        cudaFree(_bricks_inv_index);
    }

    cudaMalloc((void **)&_bricks_inv_index, BRICK_RES * BRICK_RES * BRICK_RES * sizeof(unsigned int));

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
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_copy_reference, 0, 0);

    unsigned max_bricks = ((unsigned)occupied_brick_bytes) / sizeof(unsigned);
    size_t gridSize = (max_bricks + block_size - 1) / block_size;
    kernel_copy_reference<<<gridSize, block_size>>>(active_bricks_count, _bricks_inv_index, brick_list, max_bricks);

    checkCudaErrors(cudaMemcpy(&_active_bricks_count, active_bricks_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    printf("\nactive_bricks: %u\n", _active_bricks_count);

    _ed_nodes_count = _active_bricks_count * ED_CELL_RES * ED_CELL_RES * ED_CELL_RES;
    _ed_nodes_component_count = _ed_nodes_count * 10u;

    printf("\ned_nodes_count: %u\n", _ed_nodes_count);
    printf("\ned_nodes_component_count: %u\n", _ed_nodes_component_count);

    getLastCudaError("render kernel failed");

    cudaDeviceSynchronize();

    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.buffer_occupied, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.volume_tsdf_data, 0));
}