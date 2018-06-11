#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <reconstruction/cuda/glm.cuh>
#include <reconstruction/cuda/resources.cuh>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>

__global__ void kernel_fuse_data(GLuint *occupied_bricks, size_t occupied_brick_count, unsigned int *bricks_inv_index, unsigned int ed_nodes_count, struct_ed_node *ed_graph)
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

        glm::uvec3 ed_cell_index3d = position / ED_CELL_VOXEL_DIM;
        unsigned int ed_cell = ed_cell_index3d.z * ED_CELL_RES * ED_CELL_RES + ed_cell_index3d.y * ED_CELL_RES + ed_cell_index3d.x;
        unsigned int brick_pos_inv_index = bricks_inv_index[occupied_bricks[idx]];
        unsigned int ed_cell_pos = brick_pos_inv_index * ED_CELL_RES * ED_CELL_RES * ED_CELL_RES + ed_cell;

        if(ed_cell_pos > ed_nodes_count)
        {
            continue;
        }

        if(ed_graph[ed_cell_pos].position.z * ED_CELL_RES * ED_CELL_RES + ed_graph[ed_cell_pos].position.y * ED_CELL_RES + ed_graph[ed_cell_pos].position.x == 0u)
        {
            // unset ed_node
            continue;
        }

        glm::vec3 dist = glm::vec3(world) - ed_graph[ed_cell_pos].position;

        // printf("\n|dist|: %f\n", glm::length(dist));

        const float skinning_weight = 1.f; //expf(glm::length(dist) * glm::length(dist) * 2 / (ED_CELL_VOXEL_DIM * ED_CELL_VOXEL_DIM));
        glm::uvec3 warped_position = glm::uvec3(warp_position(dist, ed_graph[ed_cell_pos], skinning_weight));

        if(warped_position.x >= VOLUME_VOXEL_DIM || warped_position.y >= VOLUME_VOXEL_DIM || warped_position.z >= VOLUME_VOXEL_DIM)
        {
            continue;
        }

        float2 data, ref;
        surf3Dread(&data, _volume_tsdf_data, warped_position.x * sizeof(float2), warped_position.y, warped_position.z);
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
            fused.y = data.y > ref.y ? data.y : ref.y;
        }

        surf3Dwrite(fused, _volume_tsdf_data, warped_position.x * sizeof(float2), warped_position.y, warped_position.z);
        surf3Dwrite(fused, _volume_tsdf_ref, world.x * sizeof(float2), world.y, world.z);
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
    kernel_fuse_data<<<grid_size, block_size>>>(brick_list, max_bricks, _bricks_inv_index, _ed_nodes_count, _ed_graph);

    getLastCudaError("render kernel failed");

    cudaDeviceSynchronize();

    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.buffer_occupied, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.volume_tsdf_data, 0));
}
