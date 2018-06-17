#include <reconstruction/cuda/resources.cuh>

__global__ void kernel_brick_indexing(unsigned int *active_bricks, unsigned int *bricks_inv_index, unsigned int *bricks_dense_index, GLuint *occupied_bricks, size_t occupied_brick_count)
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
    bricks_dense_index[brick_position] = brick_id;
    bricks_inv_index[brick_id] = brick_position;
}

__global__ void kernel_copy_reference(unsigned int active_bricks_count, const unsigned int *bricks_inv_index, const unsigned int *bricks_dense_index)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= active_bricks_count * BRICK_VOXELS)
    {
        // printf("\nactive voxel count overshot: %u, active_bricks_count * BRICK_VOXELS: %u\n", idx, active_bricks_count * BRICK_VOXELS);
        return;
    }

    unsigned int brick_id = bricks_dense_index[idx / BRICK_VOXELS];
    unsigned int position_id = idx % BRICK_VOXELS;

    // printf("\nbrick %u: (%u,%u,%u)\n", brick_id, brick.x, brick.y, brick.z);

    glm::uvec3 brick = index_3d(brick_id) * BRICK_VOXEL_DIM;
    glm::uvec3 position = position_3d(position_id);
    glm::uvec3 world = brick + position;

    //    if(position_id == 0)
    //    {
    //        printf("\nbrick %u: (%u,%u,%u)\n", brick_id, brick.x, brick.y, brick.z);
    //    }

    if(world.x >= VOLUME_VOXEL_DIM_X || world.y >= VOLUME_VOXEL_DIM_Y || world.z >= VOLUME_VOXEL_DIM_Z)
    {
        // printf("\nworld position out of volume: (%u,%u,%u)\n", world.x, world.y, world.z);
        return;
    }

    //    if(position_id == 0)
    //    {
    //        printf("\nbrick %u, position %u: (%u,%u,%u)\n", brick_id, position_id, world.x, world.y, world.z);
    //    }

    float2 data;
    surf3Dread(&data, _volume_tsdf_data, world.x * sizeof(float2), world.y, world.z);
    surf3Dwrite(data, _volume_tsdf_ref, world.x * sizeof(float2), world.y, world.z);
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

    if(_bricks_dense_index != nullptr)
    {
        cudaFree(_bricks_dense_index);
    }

    cudaMalloc((void **)&_bricks_inv_index, BRICK_RES_X * BRICK_RES_Y * BRICK_RES_Z * sizeof(unsigned int));
    cudaMalloc((void **)&_bricks_dense_index, BRICK_RES_X * BRICK_RES_Y * BRICK_RES_Z * sizeof(unsigned int));

    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.volume_tsdf_data, 0));
    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.buffer_occupied, 0));

    cudaArray *volume_array_tsdf_data = nullptr;
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&volume_array_tsdf_data, _cgr.volume_tsdf_data, 0, 0));

    size_t occupied_brick_bytes;
    GLuint *brick_sparse_list;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&brick_sparse_list, &occupied_brick_bytes, _cgr.buffer_occupied));

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
    checkCudaErrors(cudaBindSurfaceToArray(&_volume_tsdf_data, volume_array_tsdf_data, &channel_desc));
    checkCudaErrors(cudaBindSurfaceToArray(&_volume_tsdf_ref, _volume_array_tsdf_ref, &channel_desc));

    int block_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_brick_indexing, 0, 0);

    unsigned occupied_brick_count = ((unsigned)occupied_brick_bytes) / sizeof(unsigned);
    size_t grid_size = (occupied_brick_count + block_size - 1) / block_size;
    kernel_brick_indexing<<<grid_size, block_size>>>(active_bricks_count, _bricks_inv_index, _bricks_dense_index, brick_sparse_list, occupied_brick_count);

    checkCudaErrors(cudaMemcpy(&_active_bricks_count, active_bricks_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    printf("\nactive_bricks: %u\n", _active_bricks_count);

    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_copy_reference, 0, 0);

    unsigned active_brick_voxels = _active_bricks_count * BRICK_VOXELS;
    grid_size = (active_brick_voxels + block_size - 1) / block_size;
    kernel_copy_reference<<<grid_size, block_size>>>(_active_bricks_count, _bricks_inv_index, _bricks_dense_index);

    _ed_nodes_count = _active_bricks_count * ED_CELL_RES * ED_CELL_RES * ED_CELL_RES;
    _ed_nodes_component_count = _ed_nodes_count * 10u;

    printf("\ned_nodes_count: %u\n", _ed_nodes_count);
    printf("\ned_nodes_component_count: %u\n", _ed_nodes_component_count);

    getLastCudaError("render kernel failed");

    cudaDeviceSynchronize();

    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.buffer_occupied, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.volume_tsdf_data, 0));
}