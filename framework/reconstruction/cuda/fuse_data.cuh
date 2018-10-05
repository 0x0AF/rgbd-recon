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
    memcpy(&dev_res.tsdf_ref_warped_marks[offset], &(voxel.x), sizeof(float1));
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

    if(position.x >= measures.brick_dim_voxels - 1)
    {
        sampling_position.x -= 1u;
    }

    if(position.y >= measures.brick_dim_voxels - 1)
    {
        sampling_position.y -= 1u;
    }

    if(position.z >= measures.brick_dim_voxels - 1)
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

    bool is_grad_nan = glm::isnan(gradient.x) || glm::isnan(gradient.y) || glm::isnan(gradient.z);

    if(is_grad_nan)
    {
        gradient = float4{0.f, 0.f, 0.f, 0.f};
    }

    /// Write gradient

    // printf("\ngradient: (%f,%f,%f)\n", gradient.x, gradient.y, gradient.z);

    surf3Dwrite(gradient, _volume_tsdf_ref_grad, world.x * sizeof(float4), world.y, world.z);
}

/***
 * Parallel implementation, suffers from race condition
 *
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
#ifdef VERBOSE
        printf("\nout of volume: world_voxel(%i,%i,%i)\n", world_voxel.x, world_voxel.y, world_voxel.z);
#endif
        return;
    }

    /// Warp voxel

    glm::fvec3 world = data_2_norm(world_voxel, measures);
    glm::fvec3 dist = world - ed_entry.position;
    glm::fvec3 warped_position = glm::clamp(warp_position(dist, ed_node, ed_entry, 1.f, measures), glm::fvec3(0.f), glm::fvec3(1.f));
    glm::uvec3 warped_position_voxel = norm_2_data(warped_position, measures);

    if(!in_data_volume(warped_position_voxel, measures))
    {
#ifdef VERBOSE
        printf("\nout of volume: warped_position_voxel(%i,%i,%i)\n", warped_position_voxel.x, warped_position_voxel.y, warped_position_voxel.z);
#endif
        return;
    }

    *//*if(warped_position_voxel != world_voxel)
    {
        printf("\nworld_voxel(%i,%i,%i), warped_voxel(%i,%i,%i)\n", world_voxel.x, world_voxel.y, world_voxel.z, warped_position_voxel.x, warped_position_voxel.y, warped_position_voxel.z);
    }*//*

    /// Retrieve SDF value

    unsigned int offset = world_voxel.x + world_voxel.y * measures.data_volume_res.x + world_voxel.z * measures.data_volume_res.x * measures.data_volume_res.y;
    float2 voxel = dev_res.tsdf_ref[offset];

    /// Retrieve gradient

    float4 grad;
    surf3Dread(&grad, _volume_tsdf_ref_grad, world_voxel.x * sizeof(float4), world_voxel.y, world_voxel.z);

    /// Warp gradient

    glm::fvec3 gradient = glm::fvec3(grad.x, grad.y, grad.z);
    glm::fvec3 gradient_vector = glm::normalize(gradient);
    glm::fvec3 warped_gradient_vector = warp_normal(gradient_vector, ed_node, ed_entry, 1.0f, measures);
    glm::fvec3 warped_gradient = warped_gradient_vector * glm::length(gradient);

    glm::bvec3 is_nan = glm::isnan(warped_gradient);

    if(is_nan.x || is_nan.y || is_nan.z)
    {
#ifdef DEBUG_NANS
        printf("\nNaN in gradient warp evaluation\n");
#endif
        warped_gradient = glm::fvec3(0.f);
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

                glm::fvec3 diff = measures.size_voxel * (data_2_norm(warped_position_voxel, measures) - data_2_norm(vote_target, measures));
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
}*/

__global__ void kernel_warp_reference_single_ed(int ed_node_cell_index, bool mark_smallest_sdf, struct_host_resources host_res, struct_device_resources dev_res, struct_measures measures)
{
    if(ed_node_cell_index >= host_res.active_ed_nodes_count * measures.ed_cell_num_voxels)
    {
        return;
    }

    /// Retrieve ED node

    unsigned int ed_node_offset = ed_node_cell_index / measures.ed_cell_num_voxels;

    struct_ed_node ed_node = dev_res.ed_graph[ed_node_offset];
    struct_ed_meta_entry ed_entry = dev_res.ed_graph_meta[ed_node_offset];

    if(ed_entry.misalignment_error > host_res.configuration.rejection_threshold)
    {
        return;
    }

    /// Retrieve voxel position

    unsigned int voxel_id = ed_node_cell_index % measures.ed_cell_num_voxels;

    glm::uvec3 world_voxel =
        index_3d(ed_entry.brick_id, measures) * measures.brick_dim_voxels + ed_cell_3d(ed_entry.ed_cell_id, measures) * measures.ed_cell_dim_voxels + ed_cell_voxel_3d(voxel_id, measures);

    if(!in_data_volume(world_voxel, measures))
    {
#ifdef VERBOSE
        printf("\nout of volume: world_voxel(%i,%i,%i)\n", world_voxel.x, world_voxel.y, world_voxel.z);
#endif
        return;
    }

    /// Warp voxel

    glm::fvec3 world = data_2_norm(world_voxel, measures);
    glm::fvec3 dist = world - ed_entry.position;
    glm::fvec3 warped_position = glm::clamp(warp_position(dist, ed_node, ed_entry, 1.f, measures), glm::fvec3(0.f), glm::fvec3(1.f));
    glm::uvec3 warped_position_voxel = norm_2_data(warped_position, measures);

    if(!in_data_volume(warped_position_voxel, measures))
    {
#ifdef VERBOSE
        printf("\nout of volume: warped_position_voxel(%i,%i,%i)\n", warped_position_voxel.x, warped_position_voxel.y, warped_position_voxel.z);
#endif
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

    glm::fvec3 gradient = glm::fvec3(0.f);

    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            for(int k = 0; k < 3; k++)
            {
                glm::uvec3 grad_voxel((world_voxel.x + i - 1), (world_voxel.y + j - 1), (world_voxel.z + k - 1));

                if(!in_data_volume(grad_voxel, measures))
                {
#ifdef VERBOSE
                    printf("\nout of volume: world_voxel(%i,%i,%i)\n", world_voxel.x, world_voxel.y, world_voxel.z);
#endif
                    return;
                }

                float4 grad;
                surf3Dread(&grad, _volume_tsdf_ref_grad, (grad_voxel.x) * sizeof(float4), (grad_voxel.y), (grad_voxel.z));

                bool is_grad_nan = glm::isnan(grad.x) || glm::isnan(grad.y) || glm::isnan(grad.z);

                if(is_grad_nan)
                {
                    grad = float4{0.f, 0.f, 0.f, 0.f};
                }

                gradient += glm::fvec3(grad.x, grad.y, grad.z);
            }
        }
    }

    gradient /= 27.f;

    /// Warp gradient

    glm::fvec3 gradient_vector = glm::normalize(gradient);
    glm::fvec3 warped_gradient_vector = warp_normal(gradient_vector, ed_node, ed_entry, 1.0f, measures);
    glm::fvec3 warped_gradient = warped_gradient_vector * glm::length(gradient);

    bool is_nan = isnan(warped_gradient.x) || isnan(warped_gradient.y) || isnan(warped_gradient.z);

    if(is_nan)
    {
#ifdef DEBUG_NANS
        printf("\nNaN in gradient warp evaluation\n");
#endif
        warped_gradient = glm::fvec3(0.f);
    }

    /// Write prediction to 27-neighborhood

    for(int i = 0; i < 5; i++)
    {
        for(int j = 0; j < 5; j++)
        {
            for(int k = 0; k < 5; k++)
            {
                glm::uvec3 vote_target = glm::uvec3(glm::ivec3(warped_position_voxel) + glm::ivec3(i - 2, j - 2, k - 2));

                if(!in_data_volume(vote_target, measures))
                {
                    __syncthreads();
                    continue;
                }

                if(mark_smallest_sdf)
                {
                    unsigned int offset = vote_target.x + vote_target.y * measures.data_volume_res.x + vote_target.z * measures.data_volume_res.x * measures.data_volume_res.y;
                    float1 mark = sample_ref_warped_marks_ptr(dev_res.tsdf_ref_warped_marks, vote_target, measures);
                    if(glm::abs(mark.x) > glm::abs(voxel.x))
                    {
                        mark.x = voxel.x;
                        memcpy(&dev_res.tsdf_ref_warped_marks[offset], &mark, sizeof(float1));
                    }

                    __syncthreads();
                    continue;
                }

                float1 mark = sample_ref_warped_marks_ptr(dev_res.tsdf_ref_warped_marks, vote_target, measures);
                if(glm::abs(voxel.x - mark.x) > 0.001f /* TODO: figure out threshold */)
                {
                    __syncthreads();
                    continue;
                }

                glm::fvec3 diff = glm::fvec3(glm::ivec3(warped_position_voxel) - glm::ivec3(vote_target)) * measures.size_voxel;
                float prediction = voxel.x - glm::dot(diff, warped_gradient);
                float weight = exp(-glm::length(diff) * glm::length(diff) / (2.0f * measures.sigma * measures.sigma));

                // printf("\noriginal: %.3f, diff: (%.3f,%.3f,%.3f), grad: (%.3f,%.3f,%.3f), prediction:%.3f\n", voxel.x, diff.x, diff.y, diff.z, warped_gradient.x, warped_gradient.y,
                // warped_gradient.z, prediction);
                // printf("\nprediction: %f\n", prediction);
                // printf("\nweight: %e, length(diff): %f\n", weight, glm::length(diff));

                if(isnan(prediction))
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

    glm::fvec3 norm_pos = data_2_norm(world, measures);

    unsigned int offset = world.x + world.y * measures.data_volume_res.x + world.z * measures.data_volume_res.x * measures.data_volume_res.y;

    float2 ref_warped = dev_res.tsdf_ref_warped[offset];
    dev_res.out_tsdf_warped_ref[offset] = tsdf_2_8bit(ref_warped.x);

    if(glm::abs(ref_warped.x) < 0.03f)
    {
        float aggregated_average_error = 0.f;

        for(unsigned int i = 0; i < 4; i++)
        {
            float4 projection = sample_cv_xyz_inv(dev_res.cv_xyz_inv_tex[i], norm_pos);
            glm::vec2 voxel_proj = glm::vec2(projection.x, projection.y);

            if(voxel_proj.x < 0.f || voxel_proj.y < 0.f)
            {
                aggregated_average_error += 0.25f;
            }

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

__global__ void kernel_override_ref_warp(struct_host_resources host_res, struct_device_resources dev_res, struct_measures measures)
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

    unsigned int offset = world.x + world.y * measures.data_volume_res.x + world.z * measures.data_volume_res.x * measures.data_volume_res.y;

#ifdef UNIT_TEST_ERROR_INFORMED_BLENDING_IDENTICAL
    dev_res.tsdf_ref_warped[offset] = data;
#endif

#ifdef UNIT_TEST_ERROR_INFORMED_BLENDING_COMPLEMENTARY
    if(world.x > measures.data_volume_res.x / 2)
    {
        dev_res.tsdf_ref_warped[offset] = data;
    }
    else
    {
        dev_res.tsdf_ref_warped[offset] = float2{-0.03, 0.f};
    }
#endif
}

__global__ void kernel_compare_fused(float *error, struct_host_resources host_res, struct_device_resources dev_res, struct_measures measures)
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

    unsigned int offset = world.x + world.y * measures.data_volume_res.x + world.z * measures.data_volume_res.x * measures.data_volume_res.y;
    float2 fused = dev_res.tsdf_fused[offset];
    float2 ref_warped = dev_res.tsdf_ref_warped[offset];

    float value_difference = 0.f;

#ifdef UNIT_TEST_ERROR_INFORMED_BLENDING_IDENTICAL
    value_difference = glm::abs(data.x - fused.x) + glm::abs(ref_warped.x - fused.x);
#endif

#ifdef UNIT_TEST_ERROR_INFORMED_BLENDING_COMPLEMENTARY
    if(world.x > measures.data_volume_res.x / 2)
    {
        value_difference = glm::abs(ref_warped.x - fused.x);
    }
    else
    {
        value_difference = glm::abs(data.x - fused.x);
    }
#endif

    if(!isnan(value_difference))
    {
        atomicAdd(error, value_difference);
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

__device__ static float atomicFloatMax(float *address, float val)
{
    int *address_as_i = (int *)address;
    int old = *address_as_i, assumed;
    do
    {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed, __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while(assumed != old);
    return __int_as_float(old);
}

__global__ void kernel_calculate_warped_reference_metrics(bool calculate_stdev, float *metrics, unsigned int active_ed_nodes_count, struct_device_resources dev_res, struct_host_resources host_res)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= active_ed_nodes_count)
    {
        return;
    }

    float mean = 0.f;

    if(calculate_stdev)
    {
        mean = glm::abs(metrics[1]) / host_res.active_ed_vx_count;
    }

    struct_ed_meta_entry ed_entry = dev_res.ed_graph_meta[idx];

#pragma unroll
    for(unsigned int vx_idx = 0; vx_idx < ed_entry.vx_length; vx_idx++)
    {
        unsigned int address = ed_entry.vx_offset + vx_idx;
        struct_vertex vx = dev_res.warped_sorted_vx_ptr[address];

        glm::ivec3 world = norm_2_data(vx.position, host_res.measures);

        unsigned int offset = world.x + world.y * host_res.measures.data_volume_res.x + world.z * host_res.measures.data_volume_res.x * host_res.measures.data_volume_res.y;
        float2 ref_warped = dev_res.tsdf_ref_warped[offset];

        float value = glm::abs(ref_warped.x);

        if(!isnan(value))
        {
            if(!calculate_stdev)
            {
                // max
                atomicFloatMax(&metrics[0], value);

                // mean
                atomicAdd(&metrics[1], glm::abs(ref_warped.x));
            }
            else
            {
                value -= mean;
                value = value * value;

                // printf("\n value: %.3f, mean: %.3f\n", value, mean);

                // stdev
                atomicAdd(&metrics[2], value);
            }
        }
    }
}

__host__ void calculate_warped_reference_metrics(float *metrics)
{
    int block_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_calculate_warped_reference_metrics, 0, 0);
    size_t grid_size = (_host_res.active_ed_nodes_count + block_size - 1) / block_size;

    kernel_calculate_warped_reference_metrics<<<grid_size, block_size>>>(false, metrics, _host_res.active_ed_nodes_count, _dev_res, _host_res);
    getLastCudaError("kernel_calculate_warped_reference_metrics failed");
    cudaDeviceSynchronize();

    kernel_calculate_warped_reference_metrics<<<grid_size, block_size>>>(true, metrics, _host_res.active_ed_nodes_count, _dev_res, _host_res);
    getLastCudaError("kernel_calculate_warped_reference_metrics failed");
    cudaDeviceSynchronize();
}

__global__ void kernel_calculate_alignment_map_error(float *error, unsigned int active_ed_nodes_count, struct_device_resources dev_res, struct_host_resources host_res)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= active_ed_nodes_count)
    {
        // printf("\ned_node_offset overshot: %u, ed_nodes_count: %u\n", ed_node_offset, ed_nodes_count);
        return;
    }

    struct_ed_meta_entry ed_entry = dev_res.ed_graph_meta[idx];

    // printf("\ned_node position: (%f,%f,%f)\n", ed_node.position.x, ed_node.position.y, ed_node.position.z);

#pragma unroll
    for(unsigned int vx_idx = 0; vx_idx < ed_entry.vx_length; vx_idx++)
    {
        // printf("\ned_node + vertex match\n");

        struct_projection warped_projection = dev_res.warped_sorted_vx_projections[ed_entry.vx_offset + vx_idx];

        for(int i = 0; i < 4; i++)
        {
            float alignment_error_vx = dev_res.warped_sorted_vx_error[i][ed_entry.vx_offset + vx_idx];
            float silhouette = sample_silhouette(dev_res.silhouette_tex[i], warped_projection.projection[i]).x;
            float alignment_error_texture = sample_error(dev_res.alignment_error_tex[i], warped_projection.projection[i]).x;

            bool is_valid_silhouette = (silhouette == 1.f);

            if(!is_valid_silhouette)
            {
                continue;
            }

            float value_difference = glm::abs(alignment_error_vx - alignment_error_texture);

            if(!glm::isnan(value_difference))
            {
                atomicAdd(error, value_difference);
            }
        }
    }
}

__host__ void calculate_alignment_map_error(float *error)
{
    int block_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_calculate_alignment_map_error, 0, 0);
    size_t grid_size = (_host_res.active_ed_nodes_count + block_size - 1) / block_size;
    kernel_calculate_alignment_map_error<<<grid_size, block_size>>>(error, _host_res.active_ed_nodes_count, _dev_res, _host_res);
    getLastCudaError("kernel_calculate_alignment_statistics failed");
    cudaDeviceSynchronize();
}

#ifdef OUTPUT_PLY_SEQUENCE

#include "tinyply.h"
using namespace tinyply;
extern "C" unsigned long int compute_isosurface(IsoSurfaceVolume target);

extern "C" double write_ply(int frame_number)
{
    compute_isosurface(IsoSurfaceVolume::Fused);

    cudaDeviceSynchronize();

    TimerGPU timer(0);

    std::vector<glm::fvec4> padded_vertices(_host_res.total_verts);
    std::vector<glm::fvec4> padded_normals(_host_res.total_verts);

    checkCudaErrors(cudaMemcpy(&padded_vertices[0], &_dev_res.pos[0], _host_res.total_verts * sizeof(glm::fvec4), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&padded_normals[0], &_dev_res.normal[0], _host_res.total_verts * sizeof(glm::fvec4), cudaMemcpyDeviceToHost));

    std::vector<glm::fvec3> vertices(_host_res.total_verts);
    std::vector<glm::fvec3> normals(_host_res.total_verts);

    for(int i = 0; i < _host_res.total_verts; i++)
    {
        vertices[i] = glm::fvec3(padded_vertices[i]);
        normals[i] = glm::normalize(glm::fvec3(padded_normals[i]));
    }

    std::vector<glm::uvec3> triangles(_host_res.total_verts / 3);

    for(unsigned int i = 0; i < _host_res.total_verts / 3; i++)
    {
        triangles[i] = glm::uvec3(i * 3, i * 3 + 1, i * 3 + 2);
    }

    std::filebuf fb_ascii;
    fb_ascii.open("Frame_" + std::to_string(frame_number) + ".ply", std::ios::out);
    std::ostream outstream_ascii(&fb_ascii);
    if(outstream_ascii.fail())
        return 0.;

    PlyFile cube_file;

    cube_file.add_properties_to_element("vertex", {"x", "y", "z"}, Type::FLOAT32, _host_res.total_verts, reinterpret_cast<uint8_t *>(&vertices[0]), Type::INVALID, 0);
    cube_file.add_properties_to_element("vertex", {"nx", "ny", "nz"}, Type::FLOAT32, _host_res.total_verts, reinterpret_cast<uint8_t *>(&normals[0]), Type::INVALID, 0);
    cube_file.add_properties_to_element("face", {"vertex_indices"}, Type::UINT32, _host_res.total_verts / 3, reinterpret_cast<uint8_t *>(&triangles[0]), Type::UINT8, 3);

    cube_file.get_comments().push_back("Frame: " + std::to_string(frame_number) + ", resolution: " + std::to_string(_host_res.measures.data_volume_res.x));

    cube_file.write(outstream_ascii, false);

    checkCudaErrors(cudaThreadSynchronize());
    return timer.read();
}

#endif

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

    for(int i = 0; i < _host_res.active_ed_nodes_count * _host_res.measures.ed_cell_num_voxels; i++)
    {
        kernel_warp_reference_single_ed<<<1, 1>>>(i, true, _host_res, _dev_res, _host_res.measures);
        getLastCudaError("kernel_warp_reference");
        cudaDeviceSynchronize();
    }

    for(int i = 0; i < _host_res.active_ed_nodes_count * _host_res.measures.ed_cell_num_voxels; i++)
    {
        kernel_warp_reference_single_ed<<<1, 1>>>(i, false, _host_res, _dev_res, _host_res.measures);
        getLastCudaError("kernel_warp_reference");
        cudaDeviceSynchronize();
    }

    /*kernel_warp_reference<<<_host_res.active_ed_nodes_count, _host_res.measures.ed_cell_num_voxels>>>(_host_res, _dev_res, _host_res.measures);
    getLastCudaError("kernel_warp_reference");
    cudaDeviceSynchronize();*/

#ifdef UNIT_TEST_REF_WARP

    float *metrics = nullptr;
    checkCudaErrors(cudaMalloc(&metrics, 3 * sizeof(float)));
    checkCudaErrors(cudaMemset(metrics, 0, 3 * sizeof(float)));

    calculate_warped_reference_metrics(metrics);

    float max_distance = 0.f;
    float mean_distance = 0.f;
    float standard_deviation = 0.f;
    cudaMemcpy(&max_distance, &metrics[0], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&mean_distance, &metrics[1], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&standard_deviation, &metrics[2], sizeof(float), cudaMemcpyDeviceToHost);

    mean_distance /= (float)_host_res.active_ed_vx_count;
    standard_deviation /= (float)(_host_res.active_ed_vx_count - 1);
    standard_deviation = glm::sqrt(standard_deviation);

    printf("\nWarped reference mesh match: <.>: %.3f, max: %.3f, sigma: %.3f\n", mean_distance, max_distance, standard_deviation);

    checkCudaErrors(cudaFree(metrics));

#endif

#ifdef UNIT_TEST_ALIGNMENT_ERROR_MAP

    float error = 0.f;
    float *d_error = nullptr;
    checkCudaErrors(cudaMalloc(&d_error, sizeof(float)));
    checkCudaErrors(cudaMemset(d_error, 0, sizeof(float)));

    calculate_alignment_map_error(d_error);

    checkCudaErrors(cudaMemcpy(&error, d_error, sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_error));

    error /= (float)_host_res.active_ed_vx_count;
    error /= 4.f;

    printf("\nalignment map resampling error: %.3f\n", error);

#endif

#ifdef UNIT_TEST_ERROR_INFORMED_BLENDING_IDENTICAL

    kernel_override_ref_warp<<<_host_res.active_bricks_count, _host_res.measures.brick_num_voxels>>>(_host_res, _dev_res, _host_res.measures);
    getLastCudaError("kernel_fuse_data");
    cudaDeviceSynchronize();

#endif

#ifdef UNIT_TEST_ERROR_INFORMED_BLENDING_COMPLEMENTARY

    kernel_override_ref_warp<<<_host_res.active_bricks_count, _host_res.measures.brick_num_voxels>>>(_host_res, _dev_res, _host_res.measures);
    getLastCudaError("kernel_fuse_data");
    cudaDeviceSynchronize();

#endif

    kernel_fuse_data<<<_host_res.active_bricks_count, _host_res.measures.brick_num_voxels>>>(_host_res, _dev_res, _host_res.measures);
    getLastCudaError("kernel_fuse_data");
    cudaDeviceSynchronize();

#ifdef UNIT_TEST_ERROR_INFORMED_BLENDING_IDENTICAL

    float *d_error = nullptr;
    checkCudaErrors(cudaMalloc(&d_error, sizeof(float)));
    checkCudaErrors(cudaMemset(d_error, 0, sizeof(float)));

    kernel_compare_fused<<<_host_res.active_bricks_count, _host_res.measures.brick_num_voxels>>>(d_error, _host_res, _dev_res, _host_res.measures);
    getLastCudaError("kernel_fuse_data");
    cudaDeviceSynchronize();

    float error = 0.f;
    checkCudaErrors(cudaMemcpy(&error, d_error, sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_error));

    printf("\nidentical volumes blending error: %.3f\n", error);

#endif

#ifdef UNIT_TEST_ERROR_INFORMED_BLENDING_COMPLEMENTARY

    float *d_error = nullptr;
    checkCudaErrors(cudaMalloc(&d_error, sizeof(float)));
    checkCudaErrors(cudaMemset(d_error, 0, sizeof(float)));

    kernel_compare_fused<<<_host_res.active_bricks_count, _host_res.measures.brick_num_voxels>>>(d_error, _host_res, _dev_res, _host_res.measures);
    getLastCudaError("kernel_fuse_data");
    cudaDeviceSynchronize();

    float error = 0.f;
    checkCudaErrors(cudaMemcpy(&error, d_error, sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_error));

    printf("\nidentical volumes blending error: %.3f\n", error);

#endif

    kernel_refresh_misaligned<<<_host_res.active_ed_nodes_count, _host_res.measures.ed_cell_num_voxels>>>(_host_res, _dev_res, _host_res.measures);
    getLastCudaError("kernel_warp_reference");
    cudaDeviceSynchronize();

    unmap_error_texture();
    unmap_tsdf_volumes();

    /// Save a copy of an ED graph for reference
    cudaMemcpy(_dev_res.prev_ed_graph, _dev_res.ed_graph, _host_res.active_ed_nodes_count * sizeof(struct_ed_node), cudaMemcpyDeviceToDevice);
    cudaMemcpy(_dev_res.prev_ed_graph_meta, _dev_res.ed_graph_meta, _host_res.active_ed_nodes_count * sizeof(struct_ed_meta_entry), cudaMemcpyDeviceToDevice);
    _host_res.prev_frame_ed_nodes_count = _host_res.active_ed_nodes_count;

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
    getLastCudaError("kernel_copy_data");
    cudaDeviceSynchronize();

    unmap_tsdf_volumes();

    checkCudaErrors(cudaThreadSynchronize());
    return timer.read();
}
