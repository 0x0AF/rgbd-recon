#include <reconstruction/cuda/resources.cuh>

__global__ void kernel_brick_indexing(unsigned int *active_bricks, GLuint *occupied_bricks, size_t occupied_brick_count, struct_device_resources dev_res, struct_measures measures)
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
    dev_res.bricks_dense_index[brick_position] = brick_id;
    dev_res.bricks_inv_index[brick_id] = brick_position;
}

__global__ void kernel_copy_reference(unsigned int active_bricks_count, struct_device_resources dev_res, struct_measures measures)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= active_bricks_count * measures.brick_num_voxels)
    {
        // printf("\nactive voxel count overshot: %u, active_bricks_count * BRICK_VOXELS: %u\n", idx, active_bricks_count * BRICK_VOXELS);
        return;
    }

    unsigned int brick_id = dev_res.bricks_dense_index[idx / measures.brick_num_voxels];
    unsigned int position_id = idx % measures.brick_num_voxels;

    // printf("\nbrick %u: (%u,%u,%u)\n", brick_id, brick.x, brick.y, brick.z);

    glm::uvec3 brick = index_3d(brick_id, measures) * measures.brick_dim_voxels;
    glm::uvec3 position = position_3d(position_id, measures);
    glm::uvec3 world = brick + position;

    //    if(position_id == 0)
    //    {
    //        printf("\nbrick %u: (%u,%u,%u)\n", brick_id, brick.x, brick.y, brick.z);
    //    }

    if(!in_data_volume(world, measures))
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

    free_brick_resources();

    cudaDeviceSynchronize();

    allocate_brick_resources();

    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.volume_tsdf_data, 0));
    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.buffer_occupied, 0));

    cudaArray *volume_array_tsdf_data = nullptr;
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&volume_array_tsdf_data, _cgr.volume_tsdf_data, 0, 0));

    size_t occupied_brick_bytes;
    GLuint *brick_sparse_list;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&brick_sparse_list, &occupied_brick_bytes, _cgr.buffer_occupied));

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
    checkCudaErrors(cudaBindSurfaceToArray(&_volume_tsdf_data, volume_array_tsdf_data, &channel_desc));
    checkCudaErrors(cudaBindSurfaceToArray(&_volume_tsdf_ref, _dev_res.volume_array_tsdf_ref, &channel_desc));

    int block_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_brick_indexing, 0, 0);
    unsigned occupied_brick_count = ((unsigned)occupied_brick_bytes) / sizeof(unsigned);
    size_t grid_size = (occupied_brick_count + block_size - 1) / block_size;
    kernel_brick_indexing<<<grid_size, block_size>>>(active_bricks_count, brick_sparse_list, occupied_brick_count, _dev_res, _host_res.measures);
    getLastCudaError("render kernel failed");
    cudaDeviceSynchronize();

    checkCudaErrors(cudaMemcpy(&_host_res.active_bricks_count, active_bricks_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    printf("\nactive_bricks: %u\n", _host_res.active_bricks_count);

    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_copy_reference, 0, 0);
    unsigned active_brick_voxels = _host_res.active_bricks_count * _host_res.measures.brick_num_voxels;
    grid_size = (active_brick_voxels + block_size - 1) / block_size;
    kernel_copy_reference<<<grid_size, block_size>>>(_host_res.active_bricks_count, _dev_res, _host_res.measures);
    getLastCudaError("render kernel failed");
    cudaDeviceSynchronize();

    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.buffer_occupied, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.volume_tsdf_data, 0));

    if(active_bricks_count != nullptr)
    {
        cudaFree(active_bricks_count);
    }
}