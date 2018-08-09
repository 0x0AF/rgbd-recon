#include <reconstruction/cuda/resources.cuh>

__global__ void kernel_clean_ref_warped(struct_host_resources host_res, struct_device_resources dev_res, struct_measures measures)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= measures.data_volume_num_bricks * measures.brick_num_voxels)
    {
        return;
    }

    unsigned int brick_id = idx / measures.brick_num_voxels;
    unsigned int pos_id = idx % measures.brick_num_voxels;

    glm::uvec3 brick = index_3d(brick_id, measures) * measures.brick_dim_voxels;
    glm::uvec3 position = position_3d(pos_id, measures);
    glm::uvec3 world = brick + position;

    if(!in_data_volume(world, measures))
    {
        return;
    }

    /// Clean with -limit

    unsigned int offset = world.x + world.y * measures.data_volume_res.x + world.z * measures.data_volume_res.x * measures.data_volume_res.y;

    float2 voxel{-0.03f, 0.f};
    memcpy(&dev_res.tsdf_ref_warped[offset], &voxel, sizeof(float2));
}

__global__ void kernel_warp_reference(struct_host_resources host_res, struct_device_resources dev_res, struct_measures measures)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= host_res.active_ed_nodes_count * measures.ed_cell_num_voxels)
    {
        return;
    }

    /// Retrieve ED node

    unsigned int ed_node_offset = idx / measures.ed_cell_num_voxels;

    struct_ed_node ed_node = dev_res.ed_graph[ed_node_offset];
    struct_ed_meta_entry ed_entry = dev_res.ed_graph_meta[ed_node_offset];

    /// Retrieve voxel position

    unsigned int voxel_id = idx % measures.ed_cell_num_voxels;

    glm::uvec3 world_voxel =
        index_3d(ed_entry.brick_id, measures) * measures.brick_dim_voxels + ed_cell_3d(ed_entry.ed_cell_id, measures) * measures.ed_cell_dim_voxels + ed_cell_voxel_3d(voxel_id, measures);

    /// Warp voxel

    glm::vec3 world = data_2_norm(world_voxel, measures);
    glm::vec3 dist = world - ed_node.position;
    glm::vec3 warped_position = glm::clamp(warp_position(dist, ed_node, 1.f, measures), glm::vec3(0.f), glm::vec3(1.f));
    glm::uvec3 warped_position_voxel = norm_2_data(warped_position, measures);

    /// Refresh misaligned voxel

    if(ed_entry.rejected)
    {
        float2 data;
        surf3Dread(&data, _volume_tsdf_data, warped_position_voxel.x * sizeof(float2), warped_position_voxel.y, warped_position_voxel.z);
        surf3Dwrite(data, _volume_tsdf_ref, world_voxel.x * sizeof(float2), world_voxel.y, world_voxel.z);
        return;
    }

    /*if(warped_position_voxel != world_voxel)
    {
        printf("\nworld_voxel(%i,%i,%i), warped_voxel(%i,%i,%i)\n", world_voxel.x, world_voxel.y, world_voxel.z, warped_position_voxel.x, warped_position_voxel.y, warped_position_voxel.z);
    }*/

    if(!in_data_volume(world_voxel, measures) || !in_data_volume(warped_position_voxel, measures))
    {
#ifdef VERBOSE
        printf("\nout of volume: world_voxel(%i,%i,%i), warped_voxel(%i,%i,%i)\n", world_voxel.x, world_voxel.y, world_voxel.z, warped_position_voxel.x, warped_position_voxel.y,
               warped_position_voxel.z);
#endif
        return;
    }

    /// Retrieve SDF value

    float2 voxel;
    surf3Dread(&voxel, _volume_tsdf_ref, world_voxel.x * sizeof(float2), world_voxel.y, world_voxel.z);

    /// Retrieve gradient

    float4 grad;
    surf3Dread(&grad, _volume_tsdf_ref_grad, world_voxel.x * sizeof(float4), world_voxel.y, world_voxel.z);

    /// Warp gradient

    glm::vec3 gradient = glm::vec3(grad.x, grad.y, grad.z);
    glm::vec3 gradient_vector = glm::normalize(gradient);
    glm::vec3 warped_gradient_vector = warp_normal(gradient_vector, ed_node, 1.0f, measures);
    glm::vec3 warped_gradient = warped_gradient_vector * glm::length(gradient);

    glm::bvec3 is_nan = glm::isnan(warped_gradient);

    if(is_nan.x || is_nan.y || is_nan.z)
    {
#ifdef DEBUG_NANS
        printf("\nNaN in gradient warp evaluation\n");
#endif
        warped_gradient = glm::vec3(0.f);
    }

    // TODO: voxel collision detection

    /// Write prediction to 27-neighborhood

    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            for(int k = 0; k < 3; k++)
            {
                glm::uvec3 vote_target = glm::uvec3(glm::ivec3(warped_position_voxel) + glm::ivec3(i - 1, j - 1, k - 1));

                if(!in_data_volume(vote_target, measures))
                {
                    return;
                }

                glm::vec3 diff = measures.size_voxel * (data_2_norm(warped_position_voxel, measures) - data_2_norm(vote_target, measures));
                float prediction = voxel.x + glm::dot(warped_gradient, diff);
                float weight = exp(-glm::length(diff) * glm::length(diff) / (2.0f * measures.sigma * measures.sigma));

                // printf("\nprediction: %f\n", prediction);
                // printf("\nweight: %e, length(diff): %f\n", weight, glm::length(diff));

                if(glm::isnan(prediction))
                {
#ifdef DEBUG_NANS
                    printf("\nNaN in gradient-based prediction\n");
#endif
                    prediction = -0.03f;
                }

                float2 value = sample_ref_warped_ptr(dev_res.tsdf_ref_warped, vote_target, measures);

                value.x = value.x * value.y / (value.y + weight) + prediction * weight / (value.y + weight);
                value.y += weight;

                unsigned int offset = vote_target.x + vote_target.y * measures.data_volume_res.x + vote_target.z * measures.data_volume_res.x * measures.data_volume_res.y;

                memcpy(&dev_res.tsdf_ref_warped[offset], &value, sizeof(float2));

                __syncthreads();
            }
        }
    }
}

extern "C" void fuse_data()
{
    map_tsdf_volumes();

#ifdef PIPELINE_DEBUG_WARPED_REFERENCE_VOLUME
    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.pbo_tsdf_ref_warped_debug, 0));

    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&_dev_res.mapped_pbo_tsdf_ref_warped_debug, &_dev_res.mapped_bytes_ref_warped, _cgr.pbo_tsdf_ref_warped_debug));
    checkCudaErrors(cudaDeviceSynchronize());
#endif

    kernel_clean_ref_warped<<<_host_res.measures.data_volume_num_bricks, _host_res.measures.brick_num_voxels>>>(_host_res, _dev_res, _host_res.measures);
    getLastCudaError("kernel_clean_ref_warped");
    cudaDeviceSynchronize();

    kernel_warp_reference<<<_host_res.active_ed_nodes_count, _host_res.measures.ed_cell_num_voxels>>>(_host_res, _dev_res, _host_res.measures);
    getLastCudaError("kernel_warp_reference");
    cudaDeviceSynchronize();

    /*unsigned int active_ed_voxels = _host_res.active_ed_nodes_count * _host_res.measures.ed_cell_num_voxels;
    size_t grid_size = (active_ed_voxels + _host_res.measures.ed_cell_num_voxels - 1) / _host_res.measures.ed_cell_num_voxels;

    // printf("\ngrid_size: %lu, block_size: %u\n", grid_size, ED_CELL_VOXELS);

    kernel_fuse_data<<<grid_size, _host_res.measures.ed_cell_num_voxels>>>(_host_res.active_ed_nodes_count, _dev_res, _host_res.measures);
    getLastCudaError("render kernel failed");
    cudaDeviceSynchronize();*/

#ifdef PIPELINE_DEBUG_WARPED_REFERENCE_VOLUME
    checkCudaErrors(cudaMemcpy(&_dev_res.mapped_pbo_tsdf_ref_warped_debug[0], &_dev_res.tsdf_ref_warped[0], _dev_res.mapped_bytes_ref_warped, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.pbo_tsdf_ref_warped_debug));
#endif

    unmap_tsdf_volumes();
}
