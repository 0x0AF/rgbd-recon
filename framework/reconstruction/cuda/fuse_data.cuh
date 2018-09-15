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
    memcpy(&dev_res.tsdf_fused[offset], &voxel, sizeof(float2));
}

__global__ void kernel_evaluate_gradient_field(struct_host_resources host_res, struct_device_resources dev_res, struct_measures measures)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float brick_data[729];

    if(idx >= host_res.active_bricks_count * measures.brick_num_voxels)
    {
        return;
    }

    unsigned int brick_id = dev_res.bricks_dense_index[idx / measures.brick_num_voxels];
    unsigned int pos_id = idx % measures.brick_num_voxels;

    glm::uvec3 brick = index_3d(brick_id, measures) * measures.brick_dim_voxels;
    glm::uvec3 position = position_3d(pos_id, measures);
    glm::uvec3 world = brick + position;

    if(!in_data_volume(world, measures))
    {
        return;
    }

    /// Retrieve SDF value

    unsigned int offset = world.x + world.y * measures.data_volume_res.x + world.z * measures.data_volume_res.x * measures.data_volume_res.y;
    float2 voxel = dev_res.tsdf_ref[offset];
    brick_data[pos_id] = voxel.x;
    __syncthreads();

    // printf("\nvoxel: %f\n)", voxel.x);

    /// Evaluate gradient

    float4 gradient{0.f, 0.f, 0.f, 0.f};
    glm::uvec3 sampling_position = position;

    if(position.x == 0)
    {
        sampling_position.x += 1u;
    }

    if(position.y == 0)
    {
        sampling_position.y += 1u;
    }

    if(position.z == 0)
    {
        sampling_position.z += 1u;
    }

    if(position.x == measures.brick_dim_voxels - 1)
    {
        sampling_position.x -= 1u;
    }

    if(position.y == measures.brick_dim_voxels - 1)
    {
        sampling_position.y -= 1u;
    }

    if(position.z == measures.brick_dim_voxels - 1)
    {
        sampling_position.z -= 1u;
    }

    // printf("\nsampling: (%u,%u,%u)\n", sampling_position.x, sampling_position.y, sampling_position.z);

    unsigned int x_pos = position_id(sampling_position + glm::uvec3(1u, 0u, 0u), measures);
    unsigned int x_neg = position_id(sampling_position - glm::uvec3(1u, 0u, 0u), measures);
    unsigned int y_pos = position_id(sampling_position + glm::uvec3(0u, 1u, 0u), measures);
    unsigned int y_neg = position_id(sampling_position - glm::uvec3(0u, 1u, 0u), measures);
    unsigned int z_pos = position_id(sampling_position + glm::uvec3(0u, 0u, 1u), measures);
    unsigned int z_neg = position_id(sampling_position - glm::uvec3(0u, 0u, 1u), measures);

    //    if(z_pos >= host_res.measures.brick_num_voxels)
    //    {
    //        printf("\nz_pos: %u, sampling: (%u,%u,%u)\n", z_pos, sampling_position.x, sampling_position.y, sampling_position.z);
    //    }

    float two_voxels = 2.0f * measures.size_voxel;

    gradient.x = brick_data[x_pos] / two_voxels - brick_data[x_neg] / two_voxels;
    gradient.y = brick_data[y_pos] / two_voxels - brick_data[y_neg] / two_voxels;
    gradient.z = brick_data[z_pos] / two_voxels - brick_data[z_neg] / two_voxels;

    /// Write gradient

    // printf("\ngradient: (%f,%f,%f)\n", gradient.x, gradient.y, gradient.z);

    surf3Dwrite(gradient, _volume_tsdf_ref_grad, world.x * sizeof(float4), world.y, world.z);
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

    if(!in_data_volume(world_voxel, measures))
    {
/*#ifdef VERBOSE
        printf("\nout of volume: world_voxel(%i,%i,%i)\n", world_voxel.x, world_voxel.y, world_voxel.z);
#endif*/
        return;
    }

    /// Warp voxel

    glm::vec3 world = data_2_norm(world_voxel, measures);
    glm::vec3 dist = world - ed_entry.position;
    glm::vec3 warped_position = glm::clamp(warp_position(dist, ed_node, ed_entry, 1.f, measures), glm::vec3(0.f), glm::vec3(1.f));
    glm::uvec3 warped_position_voxel = norm_2_data(warped_position, measures);

    if(!in_data_volume(warped_position_voxel, measures))
    {
/*#ifdef VERBOSE
        printf("\nout of volume: warped_position_voxel(%i,%i,%i)\n", warped_position_voxel.x, warped_position_voxel.y, warped_position_voxel.z);
#endif*/
        return;
    }

    /*if(warped_position_voxel != world_voxel)
    {
        printf("\nworld_voxel(%i,%i,%i), warped_voxel(%i,%i,%i)\n", world_voxel.x, world_voxel.y, world_voxel.z, warped_position_voxel.x, warped_position_voxel.y, warped_position_voxel.z);
    }*/

    /// Retrieve SDF value

    unsigned int offset = world_voxel.x + world_voxel.y * measures.data_volume_res.x + world_voxel.z * measures.data_volume_res.x * measures.data_volume_res.y;
    float2 voxel = dev_res.tsdf_ref[offset];

    /// Retrieve gradient

    float4 grad;
    surf3Dread(&grad, _volume_tsdf_ref_grad, world_voxel.x * sizeof(float4), world_voxel.y, world_voxel.z);

    /// Warp gradient

    glm::vec3 gradient = glm::vec3(grad.x, grad.y, grad.z);
    glm::vec3 gradient_vector = glm::normalize(gradient);
    glm::vec3 warped_gradient_vector = warp_normal(gradient_vector, ed_node, ed_entry, 1.0f, measures);
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

    // TODO: eliminate race condition, cast vote on neighborhood

    for(int i = 1; i < 2; i++)
    {
        for(int j = 1; j < 2; j++)
        {
            for(int k = 1; k < 2; k++)
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
                    prediction = voxel.x;
                }

                float2 value = sample_ref_warped_ptr(dev_res.tsdf_ref_warped, vote_target, measures);

                value.x = prediction * voxel.y * weight + value.x * value.y;
                value.y = voxel.y * weight + value.y;
                value.x /= value.y;

                unsigned int offset = vote_target.x + vote_target.y * measures.data_volume_res.x + vote_target.z * measures.data_volume_res.x * measures.data_volume_res.y;

                memcpy(&dev_res.tsdf_ref_warped[offset], &value, sizeof(float2));

                __syncthreads();
            }
        }
    }

    float2 value = dev_res.tsdf_ref_warped[offset];
    dev_res.out_tsdf_warped_ref[offset] = tsdf_2_8bit(value.x);
}

__global__ void kernel_fuse_data(struct_host_resources host_res, struct_device_resources dev_res, struct_measures measures)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= host_res.active_bricks_count * measures.brick_num_voxels)
    {
        return;
    }

    unsigned int brick_id = dev_res.bricks_dense_index[idx / measures.brick_num_voxels];
    unsigned int pos_id = idx % measures.brick_num_voxels;

    glm::uvec3 brick = index_3d(brick_id, measures) * measures.brick_dim_voxels;
    glm::uvec3 position = position_3d(pos_id, measures);
    glm::uvec3 world = brick + position;

    if(!in_data_volume(world, measures))
    {
        return;
    }

    float2 data;
    surf3Dread(&data, _volume_tsdf_data, world.x * sizeof(float2), world.y, world.z);

    // printf("\ndata: %.3f\n",data.x);

    glm::vec3 norm_pos = data_2_norm(world, measures);

    unsigned int offset = world.x + world.y * measures.data_volume_res.x + world.z * measures.data_volume_res.x * measures.data_volume_res.y;

    float2 ref_warped = dev_res.tsdf_ref_warped[offset];

    if(glm::abs(ref_warped.x) < 0.03f)
    {
        float aggregated_average_error = 0.f;

        for(unsigned int i = 0; i < 4; i++)
        {
            float4 projection = sample_cv_xyz_inv(dev_res.cv_xyz_inv_tex[i], norm_pos);
            glm::vec2 voxel_proj = glm::vec2(projection.x, projection.y);
            float alignment_error = sample_error(dev_res.alignment_error_tex[i], voxel_proj).x;

            aggregated_average_error += alignment_error / 4.f;
        }

        float weight = ref_warped.y * (1.f - aggregated_average_error) + data.y;

        if(glm::abs(weight) < 0.00001f || aggregated_average_error > 0.9999f)
        {
            dev_res.tsdf_fused[offset] = data;
            dev_res.out_tsdf_fused[offset] = tsdf_2_8bit(data.x);
        }
        else
        {
            float value = ref_warped.x * ref_warped.y * (1.f - aggregated_average_error) + data.x * data.y;
            value /= weight;

            dev_res.tsdf_fused[offset] = float2{value, weight};
            dev_res.out_tsdf_fused[offset] = tsdf_2_8bit(value);
        }
    }
    else
    {
        dev_res.tsdf_fused[offset] = data;
        dev_res.out_tsdf_fused[offset] = tsdf_2_8bit(data.x);
    }
}

__global__ void kernel_refresh_misaligned(struct_host_resources host_res, struct_device_resources dev_res, struct_measures measures)
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

    if(!in_data_volume(world_voxel, measures))
    {
/*#ifdef VERBOSE
        printf("\nout of volume: world_voxel(%i,%i,%i)\n", world_voxel.x, world_voxel.y, world_voxel.z);
#endif*/
        return;
    }

    /// Refresh misaligned voxel

    if(ed_entry.misalignment_error > host_res.configuration.rejection_threshold)
    {
        unsigned int offset = world_voxel.x + world_voxel.y * measures.data_volume_res.x + world_voxel.z * measures.data_volume_res.x * measures.data_volume_res.y;

        float2 value = dev_res.tsdf_fused[offset];

        dev_res.tsdf_ref[offset] = value;
        dev_res.out_tsdf_ref[offset] = tsdf_2_8bit(value.x);
        return;
    }
}

extern "C" double fuse_data()
{
    checkCudaErrors(cudaMemset(_dev_res.out_tsdf_warped_ref, 0, _host_res.measures.data_volume_res.x * _host_res.measures.data_volume_res.y * _host_res.measures.data_volume_res.z * sizeof(uchar)));
    checkCudaErrors(cudaMemset(_dev_res.out_tsdf_fused, 0, _host_res.measures.data_volume_res.x * _host_res.measures.data_volume_res.y * _host_res.measures.data_volume_res.z * sizeof(uchar)));

    TimerGPU timer(0);

    map_tsdf_volumes();
    map_error_texture();

    kernel_evaluate_gradient_field<<<_host_res.active_bricks_count, _host_res.measures.brick_num_voxels>>>(_host_res, _dev_res, _host_res.measures);
    getLastCudaError("kernel_evaluate_gradient_field");
    cudaDeviceSynchronize();

    kernel_clean_ref_warped<<<_host_res.measures.data_volume_num_bricks, _host_res.measures.brick_num_voxels>>>(_host_res, _dev_res, _host_res.measures);
    getLastCudaError("kernel_clean_ref_warped");
    cudaDeviceSynchronize();

    kernel_warp_reference<<<_host_res.active_ed_nodes_count, _host_res.measures.ed_cell_num_voxels>>>(_host_res, _dev_res, _host_res.measures);
    getLastCudaError("kernel_warp_reference");
    cudaDeviceSynchronize();

    kernel_fuse_data<<<_host_res.active_bricks_count, _host_res.measures.brick_num_voxels>>>(_host_res, _dev_res, _host_res.measures);
    getLastCudaError("kernel_fuse_data");
    cudaDeviceSynchronize();

    kernel_refresh_misaligned<<<_host_res.active_ed_nodes_count, _host_res.measures.ed_cell_num_voxels>>>(_host_res, _dev_res, _host_res.measures);
    getLastCudaError("kernel_warp_reference");
    cudaDeviceSynchronize();

    unmap_error_texture();
    unmap_tsdf_volumes();

    checkCudaErrors(cudaThreadSynchronize());
    return timer.read();
}

__global__ void kernel_copy_data(struct_host_resources host_res, struct_device_resources dev_res, struct_measures measures)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= host_res.active_bricks_count * measures.brick_num_voxels)
    {
        return;
    }

    unsigned int brick_id = dev_res.bricks_dense_index[idx / measures.brick_num_voxels];
    unsigned int pos_id = idx % measures.brick_num_voxels;

    glm::uvec3 brick = index_3d(brick_id, measures) * measures.brick_dim_voxels;
    glm::uvec3 position = position_3d(pos_id, measures);
    glm::uvec3 world = brick + position;

    if(!in_data_volume(world, measures))
    {
        return;
    }

    float2 data;
    surf3Dread(&data, _volume_tsdf_data, world.x * sizeof(float2), world.y, world.z);

    unsigned int offset = world.x + world.y * measures.data_volume_res.x + world.z * measures.data_volume_res.x * measures.data_volume_res.y;

    dev_res.out_tsdf_data[offset] = tsdf_2_8bit(data.x);
}

extern "C" double extract_data()
{
    checkCudaErrors(cudaMemset(_dev_res.out_tsdf_data, 0, _host_res.measures.data_volume_res.x * _host_res.measures.data_volume_res.y * _host_res.measures.data_volume_res.z * sizeof(uchar)));

    TimerGPU timer(0);

    map_tsdf_volumes();

    kernel_copy_data<<<_host_res.active_bricks_count, _host_res.measures.brick_num_voxels>>>(_host_res, _dev_res, _host_res.measures);
    getLastCudaError("kernel_fuse_data");
    cudaDeviceSynchronize();

    unmap_tsdf_volumes();

    checkCudaErrors(cudaThreadSynchronize());
    return timer.read();
}
