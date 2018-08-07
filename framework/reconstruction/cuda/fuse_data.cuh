#include <reconstruction/cuda/resources.cuh>

__global__ void kernel_fuse_data(unsigned int active_ed_nodes_count, struct_device_resources dev_res, struct_measures measures)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= active_ed_nodes_count * measures.ed_cell_num_voxels)
    {
        // printf("\nactive voxel count overshot: %u, active_bricks_count * BRICK_VOXELS: %lu\n", idx, active_bricks_count * BRICK_VOXELS);
        return;
    }

    unsigned int ed_node_offset = idx / measures.ed_cell_num_voxels;

    struct_ed_node ed_node = dev_res.ed_graph[ed_node_offset];
    struct_ed_meta_entry ed_entry = dev_res.ed_graph_meta[ed_node_offset];

    unsigned int voxel_id = idx % measures.ed_cell_num_voxels;

    glm::vec3 dist = glm::vec3(ed_cell_voxel_3d(voxel_id, measures) - glm::uvec3(1u)) * measures.size_voxel;
    glm::vec3 world = glm::clamp(ed_node.position + dist, glm::vec3(0.f), glm::vec3(1.f));
    glm::vec3 warped_position = glm::clamp(warp_position(dist, ed_node, 1.f, measures), glm::vec3(0.f), glm::vec3(1.f));

    glm::uvec3 world_voxel = glm::uvec3(world * glm::vec3(measures.data_volume_res));
    glm::uvec3 warped_position_voxel = glm::uvec3(warped_position * glm::vec3(measures.data_volume_res));

    if(!in_data_volume(world_voxel, measures) || !in_data_volume(warped_position_voxel, measures))
    {
        // printf("\nout of volume: world_voxel(%u,%u,%u), warped_voxel(%u,%u,%u)\n", world_voxel.x, world_voxel.y, world_voxel.z, warped_position_voxel.x, warped_position_voxel.y,
        // warped_position_voxel.z);
        return;
    }

    if(ed_entry.rejected)
    {
        float2 data;
        surf3Dread(&data, _volume_tsdf_data, warped_position_voxel.x * sizeof(float2), warped_position_voxel.y, warped_position_voxel.z);
        surf3Dwrite(data, _volume_tsdf_ref, world_voxel.x * sizeof(float2), world_voxel.y, world_voxel.z);
        return;
    }

    __shared__ float2 voxels_reference[27];
    surf3Dread(&voxels_reference[voxel_id], _volume_tsdf_ref, world_voxel.x * sizeof(float2), world_voxel.y, world_voxel.z);
    __syncthreads();

    __shared__ glm::vec3 warped_gradient;

    if(voxel_id == 0)
    {
        glm::vec3 gradient;

        unsigned int x_pos = ed_cell_voxel_id(glm::uvec3(2, 1, 1), measures);
        unsigned int x_neg = ed_cell_voxel_id(glm::uvec3(0, 1, 1), measures);
        unsigned int y_pos = ed_cell_voxel_id(glm::uvec3(1, 2, 1), measures);
        unsigned int y_neg = ed_cell_voxel_id(glm::uvec3(1, 0, 1), measures);
        unsigned int z_pos = ed_cell_voxel_id(glm::uvec3(1, 1, 2), measures);
        unsigned int z_neg = ed_cell_voxel_id(glm::uvec3(1, 1, 0), measures);

        float two_voxels = 2.0f * measures.size_voxel;

        gradient.x = voxels_reference[x_pos].x / two_voxels - voxels_reference[x_neg].x / two_voxels;
        gradient.y = voxels_reference[y_pos].x / two_voxels - voxels_reference[y_neg].x / two_voxels;
        gradient.z = voxels_reference[z_pos].x / two_voxels - voxels_reference[z_neg].x / two_voxels;

        glm::vec3 gradient_vector = glm::normalize(gradient);
        glm::vec3 warped_gradient_vector = warp_normal(gradient_vector, ed_node, 1.0f, measures);

        warped_gradient = warped_gradient_vector * glm::length(gradient);

        glm::bvec3 is_nan = glm::isnan(warped_gradient);

        if(is_nan.x || is_nan.y || is_nan.z)
        {
#ifdef DEBUG_NANS
            printf("\nNaN in gradient warp evaluation\n");
            warped_gradient = glm::vec3(0.f);
#endif
        }
    }

    __syncthreads();

    __shared__ float2 voxels_prediction[27];

    float2 prediction{0.f, 0.f};
    prediction.x = voxels_reference[voxel_id].x + glm::dot(warped_gradient, warped_position - world);
    prediction.y = 1.0f; // TODO: pull misalignment error here

    voxels_prediction[voxel_id] = prediction;

    __syncthreads();

    // TODO: blending

    float2 data;
    surf3Dread(&data, _volume_tsdf_data, warped_position_voxel.x * sizeof(float2), warped_position_voxel.y, warped_position_voxel.z);

    float2 fused;

    fused.y = voxels_prediction[voxel_id].y + data.y;

    if(fused.y > 0.001f)
    {
        fused.x = data.x * data.y / fused.y + voxels_prediction[voxel_id].x * voxels_prediction[voxel_id].y / fused.y;
    }
    else
    {
        fused.x = data.y > voxels_prediction[voxel_id].y ? data.x : voxels_prediction[voxel_id].x;
        fused.y = data.y > voxels_prediction[voxel_id].y ? data.y : voxels_prediction[voxel_id].y;
    }

    __syncthreads();

    // surf3Dwrite(voxels_reference[voxel_id], _volume_tsdf_data, world_voxel.x * sizeof(float2), world_voxel.y, world_voxel.z);
    // surf3Dwrite(voxels_reference[voxel_id], _volume_tsdf_data, warped_position_voxel.x * sizeof(float2), warped_position_voxel.y, warped_position_voxel.z);
    // surf3Dwrite(voxels_prediction[voxel_id], _volume_tsdf_data, world_voxel.x * sizeof(float2), world_voxel.y, world_voxel.z);
    surf3Dwrite(fused, _volume_tsdf_data, warped_position_voxel.x * sizeof(float2), warped_position_voxel.y, warped_position_voxel.z);
}

extern "C" void fuse_data()
{
    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.volume_tsdf_ref, 0));
    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.volume_tsdf_data, 0));

    cudaArray *volume_array_tsdf_data = nullptr;
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&volume_array_tsdf_data, _cgr.volume_tsdf_data, 0, 0));

    cudaArray *volume_array_tsdf_ref = nullptr;
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&volume_array_tsdf_ref, _cgr.volume_tsdf_ref, 0, 0));

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
    checkCudaErrors(cudaBindSurfaceToArray(&_volume_tsdf_data, volume_array_tsdf_data, &channel_desc));
    checkCudaErrors(cudaBindSurfaceToArray(&_volume_tsdf_ref, volume_array_tsdf_ref, &channel_desc));

    unsigned int active_ed_voxels = _host_res.active_ed_nodes_count * _host_res.measures.ed_cell_num_voxels;
    size_t grid_size = (active_ed_voxels + _host_res.measures.ed_cell_num_voxels - 1) / _host_res.measures.ed_cell_num_voxels;

    // printf("\ngrid_size: %lu, block_size: %u\n", grid_size, ED_CELL_VOXELS);

    kernel_fuse_data<<<grid_size, _host_res.measures.ed_cell_num_voxels>>>(_host_res.active_ed_nodes_count, _dev_res, _host_res.measures);
    getLastCudaError("render kernel failed");
    cudaDeviceSynchronize();

    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.volume_tsdf_data, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.volume_tsdf_ref, 0));
}
