#include <reconstruction/cuda/resources.cuh>

#ifdef DEBUG_JTJ

#include "../../../external/csv/ostream.hpp"

#endif

#ifdef DEBUG_JTF

#include "../../../external/csv/ostream.hpp"

#endif

#ifdef DEBUG_H

#include "../../../external/csv/ostream.hpp"

#endif

__global__ void kernel_project(unsigned int active_ed_nodes_count, struct_device_resources dev_res, struct_measures measures)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= active_ed_nodes_count)
    {
        return;
    }

    struct_ed_meta_entry ed_entry = dev_res.ed_graph_meta[idx];

#pragma unroll
    for(unsigned int vx_idx = 0; vx_idx < ed_entry.vx_length; vx_idx++)
    {
        unsigned int address = ed_entry.vx_offset + vx_idx;
        struct_vertex vx = dev_res.sorted_vx_ptr[address];

        struct_projection vx_projection = dev_res.sorted_vx_projections[address];
        for(unsigned int i = 0; i < 4; i++)
        {
            float4 projection = sample_cv_xyz_inv(dev_res.cv_xyz_inv_tex[i], vx.position);

            vx_projection.projection[i].x = projection.x;
            vx_projection.projection[i].y = projection.y;

            /*

            TODO: investigate if inv projection is enough for data component evaluation (!)

            float depth = sample_depth(dev_res.depth_tex[i], vx_projection.projection[i]).x;
            printf("\nprojection: (%f,%f) %f:%f\n", projection.x, projection.y, projection.z, depth);*/
        }
        memcpy(&dev_res.sorted_vx_projections[address], &vx_projection, sizeof(struct_projection));
    }
}

__global__ void kernel_project_warped(unsigned int active_ed_nodes_count, struct_device_resources dev_res, struct_measures measures)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= active_ed_nodes_count)
    {
        return;
    }

    struct_ed_meta_entry ed_entry = dev_res.ed_graph_meta[idx];

#pragma unroll
    for(unsigned int vx_idx = 0; vx_idx < ed_entry.vx_length; vx_idx++)
    {
        unsigned int address = ed_entry.vx_offset + vx_idx;
        struct_vertex vx = dev_res.warped_sorted_vx_ptr[address];

        struct_projection vx_projection = dev_res.warped_sorted_vx_projections[address];
        for(unsigned int i = 0; i < 4; i++)
        {
            float4 projection = sample_cv_xyz_inv(dev_res.cv_xyz_inv_tex[i], vx.position);
            vx_projection.projection[i].x = projection.x;
            vx_projection.projection[i].y = projection.y;
        }
        memcpy(&dev_res.warped_sorted_vx_projections[address], &vx_projection, sizeof(struct_projection));
    }
}

__global__ void kernel_warp(unsigned int active_ed_nodes_count, struct_device_resources dev_res, struct_measures measures)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= active_ed_nodes_count)
    {
        return;
    }

    struct_ed_node ed_node = dev_res.ed_graph[idx];
    struct_ed_meta_entry ed_entry = dev_res.ed_graph_meta[idx];

#pragma unroll
    for(unsigned int vx_idx = 0; vx_idx < ed_entry.vx_length; vx_idx++)
    {
        unsigned int address = ed_entry.vx_offset + vx_idx;
        struct_vertex vx = dev_res.sorted_vx_ptr[address];

        glm::vec3 dist = vx.position - ed_entry.position;

        struct_vertex warped_vx = dev_res.warped_sorted_vx_ptr[address];
        warped_vx.position = glm::clamp(warp_position(dist, ed_node, ed_entry, 1.f, measures), glm::vec3(0.f), glm::vec3(1.f));
        warped_vx.brick_id = vx.brick_id;
        warped_vx.normal = warp_normal(vx.normal, ed_node, ed_entry, 1.f, measures);
        warped_vx.ed_cell_id = vx.ed_cell_id;
        memcpy(&dev_res.warped_sorted_vx_ptr[address], &warped_vx, sizeof(struct_vertex));
    }
}

__global__ void kernel_evaluate_alignment_error(unsigned int active_ed_nodes_count, struct_device_resources dev_res, struct_host_resources host_res)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= active_ed_nodes_count)
    {
        // printf("\ned_node_offset overshot: %u, ed_nodes_count: %u\n", ed_node_offset, ed_nodes_count);
        return;
    }

    __syncthreads();

    struct_ed_meta_entry ed_entry = dev_res.ed_graph_meta[idx];

    // printf("\ned_node position: (%f,%f,%f)\n", ed_node.position.x, ed_node.position.y, ed_node.position.z);

#pragma unroll
    for(unsigned int vx_idx = 0; vx_idx < ed_entry.vx_length; vx_idx++)
    {
        struct_vertex warped_vertex = dev_res.warped_sorted_vx_ptr[ed_entry.vx_offset + vx_idx];
        struct_projection warped_projection = dev_res.warped_sorted_vx_projections[ed_entry.vx_offset + vx_idx];

        // printf("\ned_node + vertex match\n");

        for(int i = 0; i < 4; i++)
        {
            // printf("\nsampling depth maps: (%f,%f)\n", warped_projection.projection[i].x, warped_projection.projection[i].y);

            if(warped_projection.projection[i].x >= 1.0f || warped_projection.projection[i].y >= 1.0f)
            {
/*#ifdef VERBOSE
                printf("\nprojected out of depth map: (%u,%u)\n", warped_projection.projection[i].x, warped_projection.projection[i].y);
#endif*/
                continue;
            }

            float depth = sample_depth(dev_res.depth_tex[i], warped_projection.projection[i]).x;

            // printf("\ndepth (%f,%f) = %f\n", warped_projection.projection[i].x, warped_projection.projection[i].y, depth);

            if(depth == 0)
            {
                // printf("\n depth is off! \n");
                continue;
            }

            // printf("\ndepth %f\n", depth);

            glm::vec3 coordinate = glm::vec3(warped_projection.projection[i].x, warped_projection.projection[i].y, depth);

            if(!in_normal_space(coordinate))
            {
/*#ifdef VERBOSE
                printf("\nprojected out of direct calibration volume: (%f,%f,%f)\n", coordinate.x, coordinate.y, coordinate.z);
#endif*/
                continue;
            }

            // printf("\nsampling direct calibration volume: (%f,%f,%f)\n", coordinate.x, coordinate.y, coordinate.z);

            float4 projected = sample_cv_xyz(dev_res.cv_xyz_tex[i], coordinate);

            // printf("\nprojected (%f,%f,%f,%f)\n", projected.x, projected.y, projected.z, projected.w);

            //        if(depth_voxel_space == 45u)
            //        {
            //            printf("\nprojected (x,y, depth): (%u,%u,%u) = (%f,%f,%f)\n", pixel.x, pixel.y, depth_voxel_space, projected.x, projected.y, projected.z);
            //        }

            glm::vec3 extracted_position = glm::vec3(projected.x, projected.y, projected.z);
            extracted_position = bbox_transform_position(extracted_position, host_res.measures);

            // printf("\nextracted_position (%.2f, %.2f, %.2f)\n", extracted_position.x, extracted_position.y, extracted_position.z);

            if(!in_normal_space(extracted_position))
            {
                continue;
            }

            //        printf("\nextracted_position (%.2f, %.2f, %.2f): (%.2f,%.2f,%.2f) = (%.2f,%.2f,%.2f)\n", coordinate.x, coordinate.y, coordinate.z, warped_position.x, warped_position.y,
            //        warped_position.z, extracted_position.x, extracted_position.y, extracted_position.z);

            glm::vec3 diff = warped_vertex.position - extracted_position;

            float vx_error = glm::length(glm::dot(warped_vertex.normal, diff));

            // printf("\nvx_error: %f\n", vx_error);

            if(isnan(vx_error))
            {
#ifdef DEBUG_NANS
                printf("\nvx_residual is NaN!\n");
#endif

                vx_error = 0.f;
            }

            glm::uvec2 warped_projection_rounding = glm::uvec2(warped_projection.projection[i] * glm::vec2(host_res.measures.depth_res.x, host_res.measures.depth_res.y));
            size_t pitched_offset = warped_projection_rounding.x + warped_projection_rounding.y * dev_res.pitch_alignment_error / sizeof(float);
            size_t pitched_offset_bins = warped_projection_rounding.x + warped_projection_rounding.y * dev_res.pitch_alignment_error_bins / sizeof(int);

            int count = 1 + atomicAdd(&dev_res.alignment_error_bins[i][pitched_offset_bins], 1);
            float curr_mean = dev_res.alignment_error[i][pitched_offset];
            float inc_average = curr_mean + (vx_error - curr_mean) / ((float)count);

            atomicExch(&dev_res.alignment_error[i][pitched_offset], inc_average);
            //atomicExch(&dev_res.alignment_error[i][pitched_offset], 0.99f);
        }
    }
}

__global__ void kernel_normalize_alignment_errors(int max_pos, struct_device_resources dev_res, int layer, struct_measures measures)
{
    const int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

    if(ix >= measures.depth_res.x || iy >= measures.depth_res.y)
    {
        return;
    }

    if(iy < 0 || ix < 0)
    {
        return;
    }

    // printf("\nmax: %f\n", dev_res.alignment_error[layer][max_pos]);

    float value = sample_pitched_ptr(dev_res.alignment_error[layer], dev_res.pitch_alignment_error, ix, iy) / dev_res.alignment_error[layer][max_pos];

    if(value == 0)
    {
        value = 1.0f;
    }

    write_pitched_ptr(value, dev_res.alignment_error[layer], dev_res.pitch_alignment_error, ix, iy);
}

__global__ void kernel_reject_misaligned_deformations(unsigned int active_ed_nodes_count, struct_device_resources dev_res, struct_host_resources host_res)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= active_ed_nodes_count)
    {
        // printf("\ned_node_offset overshot: %u, ed_nodes_count: %u\n", ed_node_offset, ed_nodes_count);
        return;
    }

    struct_ed_meta_entry ed_entry = dev_res.ed_graph_meta[idx];

    // printf("\ned_node position: (%f,%f,%f)\n", ed_node.position.x, ed_node.position.y, ed_node.position.z);

    float energy = 0.f;

#pragma unroll
    for(unsigned int vx_idx = 0; vx_idx < ed_entry.vx_length; vx_idx++)
    {
        struct_vertex vx = dev_res.warped_sorted_vx_ptr[ed_entry.vx_offset + vx_idx];
        struct_projection vx_proj = dev_res.warped_sorted_vx_projections[ed_entry.vx_offset + vx_idx];

        // printf("\ned_node + vertex match\n");

        float vx_misalignment = 0.f;

#ifdef EVALUATE_DATA
        vx_misalignment = evaluate_vx_misalignment(vx, host_res.measures);

#ifdef EVALUATE_VISUAL_HULL
        vx_misalignment = glm::min(vx_misalignment, evaluate_hull_residual(vx_proj, dev_res, host_res.measures));
#endif
#else
#ifdef EVALUATE_VISUAL_HULL
        vx_misalignment = evaluate_hull_residual(vx_proj, dev_res, host_res.measures);
#endif
#endif

        // printf("\nvx_residual: %f\n", vx_residual);

        if(isnan(vx_misalignment))
        {
#ifdef DEBUG_NANS
            printf("\nvx_residual is NaN!\n");
#endif

            vx_misalignment = 0.f;
        }

        energy += vx_misalignment;
    }

    energy /= (float)ed_entry.vx_length;

    ed_entry.misalignment_error = energy;

    // printf("\nenergy: %f\n", energy);

    memcpy(&dev_res.ed_graph_meta[idx], &ed_entry, sizeof(struct_ed_meta_entry));
}

__device__ float evaluate_vx_residual(struct_vertex &vx, struct_projection &vx_proj, struct_device_resources &dev_res, struct_host_resources &host_res)
{
    float vx_residual = 0.f;

#ifdef EVALUATE_DATA
    vx_residual += host_res.configuration.weight_data * evaluate_data_residual(vx, vx_proj, dev_res, host_res.measures);
#endif

#ifdef EVALUATE_VISUAL_HULL
    vx_residual += host_res.configuration.weight_hull * evaluate_hull_residual(vx_proj, dev_res, host_res.measures);
#endif

#ifdef EVALUATE_CORRESPONDENCE_FIELD
    vx_residual += host_res.configuration.weight_correspondence * evaluate_cf_residual(vx_proj, vx_proj, dev_res, host_res.measures);
#endif

    // printf("\nvx_residual: %f\n", vx_residual);

    if(isnan(vx_residual))
    {
#ifdef DEBUG_NANS
        printf("\nvx_residual is NaN!\n");
#endif

        vx_residual = 0.f;
    }

    return vx_residual;
}

__global__ void kernel_energy(float *energy, unsigned int active_ed_nodes_count, struct_device_resources dev_res, struct_host_resources host_res, struct_measures measures)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= active_ed_nodes_count)
    {
        // printf("\ned_node_offset overshot: %u, ed_nodes_count: %u\n", ed_node_offset, ed_nodes_count);
        return;
    }

    struct_ed_meta_entry ed_entry = dev_res.ed_graph_meta[idx];

#ifdef EVALUATE_ED_REGULARIZATION
    struct_ed_node ed_node = dev_res.ed_graph[idx];

    float ed_residual = host_res.configuration.weight_regularization * evaluate_ed_node_residual(ed_node, ed_entry, dev_res, measures);

    if(isnan(ed_residual))
    {
#ifdef DEBUG_NANS
        printf("\ned_residual is NaN!\n");
#endif

        ed_residual = 0.f;
    }

    atomicAdd(energy, ed_residual);
#endif

    // printf("\ned_node position: (%f,%f,%f)\n", ed_node.position.x, ed_node.position.y, ed_node.position.z);

#pragma unroll
    for(unsigned int vx_idx = 0; vx_idx < ed_entry.vx_length; vx_idx++)
    {
        struct_vertex vx = dev_res.sorted_vx_ptr[ed_entry.vx_offset + vx_idx];
        struct_projection vx_proj = dev_res.sorted_vx_projections[ed_entry.vx_offset + vx_idx];

        // printf("\ned_node + vertex match\n");

        float vx_residual = evaluate_vx_residual(vx, vx_proj, dev_res, host_res);

        atomicAdd(energy, vx_residual);
    }
}

__global__ void kernel_step_energy(float *energy, unsigned int active_ed_nodes_count, struct_device_resources dev_res, struct_host_resources host_res, struct_measures measures)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= active_ed_nodes_count)
    {
        // printf("\ned_node_offset overshot: %u, ed_nodes_count: %u\n", ed_node_offset, ed_nodes_count);
        return;
    }

    struct_ed_meta_entry ed_entry = dev_res.ed_graph_meta[idx];
    struct_ed_node ed_node = dev_res.ed_graph_step[idx];

#ifdef EVALUATE_ED_REGULARIZATION
    float ed_residual = host_res.configuration.weight_regularization * evaluate_ed_node_residual(ed_node, ed_entry, dev_res, measures);

    if(isnan(ed_residual))
    {
#ifdef DEBUG_NANS
        printf("\ned_residual is NaN!\n");
#endif

        ed_residual = 0.f;
    }

    atomicAdd(energy, ed_residual);
#endif

    // printf("\ned_node position: (%f,%f,%f)\n", ed_node.position.x, ed_node.position.y, ed_node.position.z);

#pragma unroll
    for(unsigned int vx_idx = 0; vx_idx < ed_entry.vx_length; vx_idx++)
    {
        struct_vertex vx = dev_res.sorted_vx_ptr[ed_entry.vx_offset + vx_idx];

        glm::vec3 dist = vx.position - ed_entry.position;

        struct_vertex warped_vx;
        warped_vx.position = glm::clamp(warp_position(dist, ed_node, ed_entry, 1.f, measures), glm::vec3(0.f), glm::vec3(1.f));
        warped_vx.brick_id = vx.brick_id;
        warped_vx.normal = warp_normal(vx.normal, ed_node, ed_entry, 1.f, measures);
        warped_vx.ed_cell_id = vx.ed_cell_id;

        struct_projection vx_projection;
        for(unsigned int i = 0; i < 4; i++)
        {
            float4 projection = sample_cv_xyz_inv(dev_res.cv_xyz_inv_tex[i], warped_vx.position);
            vx_projection.projection[i].x = projection.x;
            vx_projection.projection[i].y = projection.y;
        }

        // printf("\ned_node + vertex match\n");

        float vx_residual = evaluate_vx_residual(warped_vx, vx_projection, dev_res, host_res);

        atomicAdd(energy, vx_residual);
    }

    // printf("\njtf[%u]\n", ed_node_offset * 10u);
}

__device__ float evaluate_vx_partial_derivative(int component, struct_ed_node &ed_node, struct_ed_meta_entry &ed_entry, struct_vertex &vx, struct_projection &vx_proj, struct_device_resources &dev_res,
                                                struct_host_resources &host_res)
{
    float pd = 0.f;

    struct_ed_node ed_node_new;
    memcpy(&ed_node_new, &ed_node, sizeof(struct_ed_node));

    struct_vertex warped_vx;
    memset(&warped_vx, 0, sizeof(struct_vertex));

    struct_projection warped_vx_proj;
    memset(&warped_vx_proj, 0, sizeof(struct_projection));

    glm::vec3 dist = vx.position - ed_entry.position;

    float ds = derivative_step(component, host_res.measures);
    shift_component(ed_node_new, component, ds);

    warped_vx.position = glm::clamp(warp_position(dist, ed_node_new, ed_entry, 1.f, host_res.measures), glm::vec3(0.f), glm::vec3(1.f));
    warped_vx.normal = vx.normal; /// Per-step approximation
    for(unsigned int i = 0; i < 4; i++)
    {
        float4 projection = sample_cv_xyz_inv(dev_res.cv_xyz_inv_tex[i], warped_vx.position);
        warped_vx_proj.projection[i].x = projection.x;
        warped_vx_proj.projection[i].y = projection.y;
    }

    float vx_res_pos = 0.f;

#ifdef EVALUATE_DATA
    vx_res_pos += host_res.configuration.weight_data * evaluate_data_residual(warped_vx, vx_proj /** per-step approximation **/, dev_res, host_res.measures);
#endif

#ifdef EVALUATE_VISUAL_HULL
    vx_res_pos += host_res.configuration.weight_hull * evaluate_hull_residual(warped_vx_proj, dev_res, host_res.measures);
#endif

#ifdef EVALUATE_CORRESPONDENCE_FIELD
    vx_res_pos += host_res.configuration.weight_correspondence * evaluate_cf_residual(vx_proj, warped_vx_proj, dev_res, host_res.measures);
#endif

    // printf("\nvx_residual: %f\n", vx_residual);

    if(isnan(vx_res_pos))
    {
#ifdef DEBUG_NANS
        printf("\nvx_residual is NaN!\n");
#endif

        vx_res_pos = 0.f;
    }

    shift_component(ed_node_new, component, -2.f * ds);

    warped_vx.position = glm::clamp(warp_position(dist, ed_node_new, ed_entry, 1.f, host_res.measures), glm::vec3(0.f), glm::vec3(1.f));
    warped_vx.normal = vx.normal; /// Per-step approximation

    float vx_res_neg = 0.f;

#ifdef EVALUATE_DATA
    vx_res_neg += host_res.configuration.weight_data * evaluate_data_residual(warped_vx, vx_proj /** per-step approximation **/, dev_res, host_res.measures);
#endif

#ifdef EVALUATE_VISUAL_HULL
    vx_res_neg += host_res.configuration.weight_hull * evaluate_hull_residual(warped_vx_proj, dev_res, host_res.measures);
#endif

#ifdef EVALUATE_CORRESPONDENCE_FIELD
    vx_res_neg += host_res.configuration.weight_correspondence * evaluate_cf_residual(vx_proj, warped_vx_proj, dev_res, host_res.measures);
#endif

    // printf("\nvx_residual: %f\n", vx_residual);

    if(isnan(vx_res_neg))
    {
#ifdef DEBUG_NANS
        printf("\nvx_residual is NaN!\n");
#endif

        vx_res_neg = 0.f;
    }

    ds *= 2.f;
    pd = vx_res_pos / ds - vx_res_neg / ds;

    if(isnan(pd))
    {
#ifdef DEBUG_NANS
        printf("\nvx_pds[%u] is NaN!\n", component);
#endif

        pd = 0.f;
    }
    return pd;
}

__global__ void kernel_jtj_jtf(unsigned long long int active_ed_vx_count, unsigned int active_ed_nodes_count, const float mu, struct_device_resources dev_res, struct_host_resources host_res,
                               struct_measures measures)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float shared_jtj_coo_val_block[ED_COMPONENT_COUNT * ED_COMPONENT_COUNT];
    __shared__ float shared_jtf_block[ED_COMPONENT_COUNT];

    unsigned int ed_node_offset = idx / ED_COMPONENT_COUNT;
    if(ed_node_offset >= active_ed_nodes_count)
    {
        // printf("\ned_node_offset overshot: %u, ed_nodes_count: %u\n", ed_node_offset, ed_nodes_count);
        return;
    }

    struct_ed_meta_entry ed_entry = dev_res.ed_graph_meta[ed_node_offset];
    struct_ed_node ed_node = dev_res.ed_graph[ed_node_offset];

    unsigned int component = idx % ED_COMPONENT_COUNT;

#ifdef EVALUATE_ED_REGULARIZATION
    __shared__ float ed_pds[ED_COMPONENT_COUNT];

    float ed_residual = host_res.configuration.weight_regularization * evaluate_ed_node_residual(ed_node, ed_entry, dev_res, measures);

    if(isnan(ed_residual))
    {
#ifdef DEBUG_NANS
        printf("\ned_residual is NaN!\n");
#endif

        ed_residual = 0.f;
    }

    ed_pds[component] = host_res.configuration.weight_regularization * evaluate_ed_pd(ed_node_new, ed_entry, component, dev_res, measures);

    if(isnan(ed_pds[component]))
    {
#ifdef DEBUG_NANS
        printf("\ned_pds[%u] is NaN!\n", component);
#endif

        ed_pds[component] = 0.f;
    }

    __syncthreads();

    float jtf_value = -ed_pds[component] * ed_residual;

    if(isnan(jtf_value))
    {
#ifdef DEBUG_NANS
        printf("\njtf_value[%u] is NaN!\n", component);
#endif

        jtf_value = 0.f;
    }

    shared_jtf_block[component] = jtf_value;

    for(unsigned int component_k = 0; component_k < ED_COMPONENT_COUNT; component_k++)
    {
        unsigned int jtj_pos = component * ED_COMPONENT_COUNT + component_k;

        float jtj_value = ed_pds[component] * ed_pds[component_k];

        if(isnan(jtj_value))
        {
#ifdef DEBUG_NANS
            printf("\njtj_value[%u] is NaN!\n", component);
#endif

            jtj_value = 0.f;
        }

        shared_jtj_coo_val_block[jtj_pos] = jtj_value;
    }
#endif

    __syncthreads();

    // printf("\ned_node position: (%f,%f,%f)\n", ed_node.position.x, ed_node.position.y, ed_node.position.z);
    for(unsigned int vx_idx = 0; vx_idx < ed_entry.vx_length; vx_idx++)
    {
        struct_vertex vx = dev_res.sorted_vx_ptr[ed_entry.vx_offset + vx_idx];
        struct_projection vx_proj = dev_res.sorted_vx_projections[ed_entry.vx_offset + vx_idx];

        // printf("\ned_node + vertex match\n");

        __shared__ float vx_residual;

        if(component == 0)
        {
            vx_residual = evaluate_vx_residual(vx, vx_proj, dev_res, host_res);
        }

        __shared__ float pds[ED_COMPONENT_COUNT];

        pds[component] = evaluate_vx_partial_derivative(component, ed_node, ed_entry, vx, vx_proj, dev_res, host_res);

        __syncthreads();

        //        if(component != 0)
        //        {
        //            printf("\nvx_res: %f\n", vx_residual);
        //        }

        //        if(component == 0)
        //        {
        //            printf("\npds: {%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f}\n", pds[0], pds[1], pds[2], pds[3], pds[4], pds[5], pds[6]);
        //        }

        //        if(component == 6)
        //        {
        //            printf("\npds: {%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f}\n", pds[0], pds[1], pds[2], pds[3], pds[4], pds[5], pds[6]);
        //        }

        float jtf_value = -pds[component] * vx_residual;

        if(isnan(jtf_value))
        {
#ifdef DEBUG_NANS
            printf("\njtf_value[%u] is NaN!\n", component);
#endif

            jtf_value = 0.f;
        }

        atomicAdd(&shared_jtf_block[component], jtf_value);

        for(unsigned int component_k = 0; component_k < ED_COMPONENT_COUNT; component_k++)
        {
            unsigned int jtj_pos = UMAD(component, ED_COMPONENT_COUNT, component_k);

            float jtj_value = pds[component] * pds[component_k];

            //            if(jtf_value == 0.f)
            //            {
            //                printf("\ncoords:(%u,%u), v(%.2f,%.2f,%.2f)\n", component, component_k, vx.position.x, vx.position.y, vx.position.z);
            //            }

            if(isnan(jtj_value))
            {
#ifdef DEBUG_NANS
                printf("\njtj_value[%u] is NaN!\n", component);
#endif

                jtj_value = 0.f;
            }

            atomicAdd(&shared_jtj_coo_val_block[jtj_pos], jtj_value);
        }

        __syncthreads();
    }

    // printf("\njtf[%u]\n", ed_node_offset * 10u);

    unsigned int jtj_pos = UMAD(component, ED_COMPONENT_COUNT, component);
#ifdef JTJ_HESSIAN_DIAG
    atomicAdd(&shared_jtj_coo_val_block[jtj_pos], mu * shared_jtj_coo_val_block[jtj_pos]);
#else
    atomicAdd(&shared_jtj_coo_val_block[jtj_pos], mu);
#endif

    __syncthreads();

    //    if(idx == 0)
    //    {
    //        for(int i = 0; i < ED_COMPONENT_COUNT; i++)
    //        {
    //            printf("\nshared_jtf_block[%u]: %f\n", i, shared_jtf_block[i]);
    //        }
    //    }

#ifdef DEBUG_JTJ_PUSH_ORDERED_INTEGERS

    for(unsigned int component_k = 0; component_k < ED_COMPONENT_COUNT; component_k++)
    {
        unsigned int jtj_pos = component * ED_COMPONENT_COUNT + component_k;
        shared_jtj_coo_val_block[jtj_pos] = jtj_pos;
    }

    __syncthreads();

#endif

    memcpy(&dev_res.jtf[idx], &shared_jtf_block[component], sizeof(float));
    memcpy(&dev_res.jtj_vals[idx * ED_COMPONENT_COUNT], &shared_jtj_coo_val_block[component * ED_COMPONENT_COUNT], sizeof(float) * ED_COMPONENT_COUNT);
}

__global__ void kernel_jtj_coo_cols_rows(unsigned int active_ed_nodes_count, struct_device_resources dev_res, struct_measures measures)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    int jtj_diag_coo_row_strip[ED_COMPONENT_COUNT];
    int jtj_diag_coo_col_strip[ED_COMPONENT_COUNT];

    // printf("\ned_per_thread: %u\n", ed_per_thread);

    unsigned int ed_node_offset = idx / ED_COMPONENT_COUNT;
    if(ed_node_offset >= active_ed_nodes_count)
    {
        // printf("\ned_node_offset overshot: %u, ed_nodes_count: %u\n", ed_node_offset, ed_nodes_count);
        return;
    }

    // printf("\ned_node_offset: %u, ed_nodes_count: %u\n", ed_node_offset, ed_nodes_count);

#pragma unroll
    for(unsigned int col = 0; col < ED_COMPONENT_COUNT; col++)
    {
        jtj_diag_coo_col_strip[col] = ed_node_offset * ED_COMPONENT_COUNT + col;
        jtj_diag_coo_row_strip[col] = idx;
    }

    // printf("\njtf[%u]\n", ed_node_offset * 10u);

    memcpy(&dev_res.jtj_rows[idx * ED_COMPONENT_COUNT], &jtj_diag_coo_row_strip[0], sizeof(int) * ED_COMPONENT_COUNT);
    memcpy(&dev_res.jtj_cols[idx * ED_COMPONENT_COUNT], &jtj_diag_coo_col_strip[0], sizeof(int) * ED_COMPONENT_COUNT);
}

__host__ void convert_to_csr()
{
    int csr_nnz = _host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT;
    int N = (int)_host_res.active_ed_nodes_count * ED_COMPONENT_COUNT;

    cusparseStatus_t status;

    // printf("\nmxn: %ix%i, nnz_dev_mem: %u\n", N, N, csr_nnz);

    cudaDeviceSynchronize();
    status = cusparseXcoo2csr(cusparse_handle, _dev_res.jtj_rows, csr_nnz, N, _dev_res.jtj_rows_csr, CUSPARSE_INDEX_BASE_ZERO);
    cudaDeviceSynchronize();

    if(status != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
    {
        printf("\ncusparseStatus_t: %u\n", status);
    }

    getLastCudaError("cusparseXcoo2csr failure");
}

#ifdef SOLVER_DIRECT_CHOL
__host__ int solve_for_h()
{
    cusparseMatDescr_t descr = nullptr;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    getLastCudaError("cusparseCreateMatDescr failure");
    cudaDeviceSynchronize();

    int csr_nnz = _host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT;
    int N = (int)_host_res.active_ed_nodes_count * ED_COMPONENT_COUNT;

    float tol = 1.e-6f;
    const int reorder = 0;
    int singularity = 0;

    cudaDeviceSynchronize();

    cusolverStatus_t solver_status =
        cusolverSpScsrlsvchol(cusolver_handle, N, csr_nnz, descr, _dev_res.jtj_vals, _dev_res.jtj_rows_csr, _dev_res.jtj_cols, _dev_res.jtf, tol, reorder, _dev_res.h, &singularity);

    cudaDeviceSynchronize();

    if(solver_status != cusolverStatus_t::CUSOLVER_STATUS_SUCCESS)
    {
        printf("\ncusolverStatus_t: %u\n", solver_status);
    }

    getLastCudaError("cusolverSpScsrlsvchol failure");

    cusparseDestroyMatDescr(descr);

    return singularity;
}
#endif

#ifdef SOLVER_DIRECT_QR
__host__ int solve_for_h()
{
    cusparseMatDescr_t descr = nullptr;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    getLastCudaError("cusparseCreateMatDescr failure");
    cudaDeviceSynchronize();

    int csr_nnz = _host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT;
    int N = (int)_host_res.active_ed_nodes_count * ED_COMPONENT_COUNT;

    // printf("\nmxn: %ix%i, nnz_dev_mem: %u\n", N, N, csr_nnz);

    float tol = 1.e-6f;
    const int reorder = 0;
    int singularity = 0;

    float *residual = NULL;
    checkCudaErrors(cudaMalloc((void **)&residual, sizeof(float) * N));

    cudaDeviceSynchronize();

    cusolverStatus_t solver_status =
        cusolverSpScsrlsvqr(cusolver_handle, N, csr_nnz, descr, _dev_res.jtj_vals, _dev_res.jtj_rows_csr, _dev_res.jtj_cols, _dev_res.jtf, tol, reorder, _dev_res.h, &singularity);

    cudaDeviceSynchronize();

    if(solver_status != cusolverStatus_t::CUSOLVER_STATUS_SUCCESS)
    {
        printf("\ncusolverStatus_t: %u\n", solver_status);
    }

    getLastCudaError("cusolverSpScsrlsvlu failure");

    cusparseDestroyMatDescr(descr);

    return singularity;
}
#endif

#ifdef SOLVER_CG
__host__ int solve_for_h()
{
    cusparseMatDescr_t descr = nullptr;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    getLastCudaError("cusparseCreateMatDescr failure");
    cudaDeviceSynchronize();

    int csr_nnz = _host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT;

    int N = (int)_host_res.active_ed_nodes_count * ED_COMPONENT_COUNT;

    // printf("\nmxn: %ix%i, nnz_dev_mem: %u\n", N, N, csr_nnz);

    const int max_iter = _host_res.configuration.solver_cg_steps;
    const float tol = 1e-6f;

    float r0, r1, alpha, alpham1, beta;
    float dot;

    float a, b, na;

    alpha = 1.0f;
    alpham1 = -1.0f;
    beta = 0.f;
    r0 = 0.f;

    cudaDeviceSynchronize();
    cusparseStatus_t status = cusparseScsrmv(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, csr_nnz, &alpha, descr, _dev_res.jtj_vals, _dev_res.jtj_rows_csr, _dev_res.jtj_cols, _dev_res.h,
                                             &beta, _dev_res.pcg_Ax);
    cudaDeviceSynchronize();

    if(status != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
    {
        printf("\ncusparseStatus_t: %u\n", status);
    }

    getLastCudaError("cusparseScsrmv failure");

    cublasSaxpy(cublas_handle, N, &alpham1, _dev_res.pcg_Ax, 1, _dev_res.jtf, 1);
    cublasSdot(cublas_handle, N, _dev_res.jtf, 1, _dev_res.jtf, 1, &r1);

    float init_res = sqrt(r1);

#ifdef VERBOSE
    printf("\ninitial residual = %e\n", sqrt(r1));
#endif

    if(isnanf(sqrt(r1)))
    {
        fprintf(stderr, "\nnan in initial residual!\n");
    }

    int k = 1;
    float r1_last = 0.f;

    while(r1 > tol * tol && k <= max_iter)
    {
        if(k > 1)
        {
            b = r1 / r0;
            cublasSscal(cublas_handle, N, &b, _dev_res.pcg_p, 1);
            cublasSaxpy(cublas_handle, N, &alpha, _dev_res.jtf, 1, _dev_res.pcg_p, 1);
        }
        else
        {
            cublasScopy(cublas_handle, N, _dev_res.jtf, 1, _dev_res.pcg_p, 1);
        }

        cudaDeviceSynchronize();
        status = cusparseScsrmv(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, csr_nnz, &alpha, descr, _dev_res.jtj_vals, _dev_res.jtj_rows_csr, _dev_res.jtj_cols, _dev_res.pcg_p, &beta,
                                _dev_res.pcg_Ax);
        cudaDeviceSynchronize();

        if(status != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
        {
            printf("\ncusparseStatus_t: %u\n", status);
        }

        getLastCudaError("cusparseScsrmv failure");

        cublasSdot(cublas_handle, N, _dev_res.pcg_p, 1, _dev_res.pcg_Ax, 1, &dot);
        a = r1 / dot;

        cublasSaxpy(cublas_handle, N, &a, _dev_res.pcg_p, 1, _dev_res.h, 1);
        na = -a;
        cublasSaxpy(cublas_handle, N, &na, _dev_res.pcg_Ax, 1, _dev_res.jtf, 1);

        r0 = r1;
        cublasSdot(cublas_handle, N, _dev_res.jtf, 1, _dev_res.jtf, 1, &r1);
        k++;

        if(isnanf(sqrt(r1)))
        {
            fprintf(stderr, "\nnan in solution!\n");
            return 0;
        }

        float step = sqrt(r1) - r1_last;

        if(step < 0.f || glm::abs(step) < (0.01f * sqrt(r1)))
        {
            break;
        }

        r1_last = sqrt(r1);

#ifdef VERBOSE
        printf("\niteration = %3d, residual = %e\n", k, sqrt(r1));
#endif
    }

#ifdef VERBOSE
    printf("\niteration = %3d, residual = %e\n", k, sqrt(r1));
#endif

    cusparseDestroyMatDescr(descr);

    return -1;
}
#endif

#ifdef SOLVER_PCG
__host__ int solve_for_h()
{
    // TODO
    return -1;
}
#endif

#ifdef DEBUG_JTJ
__host__ void print_out_jtj()
{
    float *host_jtj_vals;
    int *host_jtj_rows;
    int *host_jtj_cols;

    host_jtj_vals = (float *)malloc(_host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT * sizeof(float));
    host_jtj_rows = (int *)malloc(_host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT * sizeof(int));
    host_jtj_cols = (int *)malloc(_host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT * sizeof(int));

    cudaMemcpy(&host_jtj_vals[0], &_dev_res.jtj_vals[0], _host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_jtj_rows[0], &_dev_res.jtj_rows[0], _host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_jtj_cols[0], &_dev_res.jtj_cols[0], _host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT * sizeof(int), cudaMemcpyDeviceToHost);

    std::ofstream fs("jtj_" + std::to_string(rand()) + ".csv");
    text::csv::csv_ostream csvs(fs);

#ifdef DEBUG_JTJ_COO
    for(int i = 0; i < _host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT; i++)
    {
        csvs << host_jtj_vals[i];
    }
    csvs << text::csv::endl;

    for(int i = 0; i < _host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT; i++)
    {
        csvs << host_jtj_rows[i];
    }
    csvs << text::csv::endl;

    for(int i = 0; i < _host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT; i++)
    {
        csvs << host_jtj_cols[i];
    }
    csvs << text::csv::endl;
#endif

#ifdef DEBUG_JTJ_DENSE
    int row = 0;
    int col = 0;
    for(int i = 0; i < _host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT; i++)
    {
        while(host_jtj_cols[i] > col)
        {
            csvs << 0;
            col++;
        }

        while(host_jtj_rows[i] > row)
        {
            csvs << text::csv::endl;
            row++;
        }

        csvs << host_jtj_vals[i];
        col++;
    }
#endif

    fs.close();

    free(host_jtj_vals);
    free(host_jtj_rows);
    free(host_jtj_cols);
}
#endif

#ifdef DEBUG_JTF

__host__ void print_out_jtf()
{
    float *host_jtf;

    host_jtf = (float *)malloc(_host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * sizeof(float));

    cudaMemcpy(&host_jtf[0], &_dev_res.jtf[0], _host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * sizeof(float), cudaMemcpyDeviceToHost);

    std::ofstream fs("jtf_" + std::to_string(rand()) + ".csv");
    text::csv::csv_ostream csvs(fs);

    for(int i = 0; i < _host_res.active_ed_nodes_count * ED_COMPONENT_COUNT; i++)
    {
        csvs << host_jtf[i];
    }
    csvs << text::csv::endl;

    fs.close();

    free(host_jtf);
}

#endif

#ifdef DEBUG_H

__host__ void print_out_h()
{
    float *host_h;

    host_h = (float *)malloc(_host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * sizeof(float));

    cudaMemcpy(&host_h[0], &_dev_res.h[0], _host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * sizeof(float), cudaMemcpyDeviceToHost);

    std::ofstream fs("h_" + std::to_string(rand()) + ".csv");
    text::csv::csv_ostream csvs(fs);

    for(int i = 0; i < _host_res.active_ed_nodes_count * ED_COMPONENT_COUNT; i++)
    {
        csvs << host_h[i];
    }
    csvs << text::csv::endl;

    fs.close();

    free(host_h);
}

#endif

__host__ void evaluate_jtj_jtf(const float mu)
{
    size_t grid_size = _host_res.active_ed_nodes_count;

    kernel_jtj_jtf<<<grid_size, ED_COMPONENT_COUNT>>>(_host_res.active_ed_vx_count, _host_res.active_ed_nodes_count, mu, _dev_res, _host_res, _host_res.measures);
    getLastCudaError("render kernel failed");
    cudaDeviceSynchronize();

    kernel_jtj_coo_cols_rows<<<grid_size, ED_COMPONENT_COUNT>>>(_host_res.active_ed_nodes_count, _dev_res, _host_res.measures);
    getLastCudaError("render kernel failed");
    cudaDeviceSynchronize();
};

__host__ void evaluate_misalignment_energy(float &misalignment_energy)
{
    float *device_misalignment_energy = nullptr;
    cudaMallocManaged(&device_misalignment_energy, sizeof(float));
    *device_misalignment_energy = 0.f;

    int block_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_energy, 0, 0);
    size_t grid_size = (_host_res.active_ed_nodes_count + block_size - 1) / block_size;
    kernel_energy<<<grid_size, block_size>>>(device_misalignment_energy, _host_res.active_ed_nodes_count, _dev_res, _host_res, _host_res.measures);
    getLastCudaError("render kernel failed");
    cudaDeviceSynchronize();

    cudaMemcpy(&misalignment_energy, &device_misalignment_energy[0], sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(device_misalignment_energy);
}

__host__ void evaluate_step_misalignment_energy(float &misalignment_energy, const float mu)
{
    float *device_misalignment_energy = nullptr;
    cudaMallocManaged(&device_misalignment_energy, sizeof(float));
    *device_misalignment_energy = 0.f;

    cudaMemcpy(&_dev_res.ed_graph_step[0], &_dev_res.ed_graph[0], _host_res.active_ed_nodes_count * sizeof(struct_ed_node), cudaMemcpyDeviceToDevice);

    cudaDeviceSynchronize();

    int N = (int)_host_res.active_ed_nodes_count * ED_COMPONENT_COUNT;
    float one = 1.f;
    cublasSaxpy(cublas_handle, N, &one, _dev_res.h, 1, (float *)&_dev_res.ed_graph_step[0], 1);

    cudaDeviceSynchronize();

    int block_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_step_energy, 0, 0);
    size_t grid_size = (_host_res.active_ed_nodes_count + block_size - 1) / block_size;
    kernel_step_energy<<<grid_size, block_size>>>(device_misalignment_energy, _host_res.active_ed_nodes_count, _dev_res, _host_res, _host_res.measures);
    getLastCudaError("render kernel failed");
    cudaDeviceSynchronize();

    cudaMemcpy(&misalignment_energy, &device_misalignment_energy[0], sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(device_misalignment_energy);
}

__host__ void evaluate_alignment_error()
{
    for(unsigned int i = 0; i < 4; i++)
    {
        checkCudaErrors(cudaMemset2D(_dev_res.alignment_error[i], _dev_res.pitch_alignment_error, 0, _host_res.measures.depth_res.x * sizeof(float), _host_res.measures.depth_res.y));
        checkCudaErrors(cudaMemset2D(_dev_res.alignment_error_bins[i], _dev_res.pitch_alignment_error_bins, 0, _host_res.measures.depth_res.x * sizeof(int), _host_res.measures.depth_res.y));
    }

    int block_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_reject_misaligned_deformations, 0, 0);
    size_t grid_size = (_host_res.active_ed_nodes_count + block_size - 1) / block_size;
    kernel_evaluate_alignment_error<<<grid_size, block_size>>>(_host_res.active_ed_nodes_count, _dev_res, _host_res);
    getLastCudaError("render kernel failed");
    cudaDeviceSynchronize();

    dim3 threads(8, 8);
    dim3 blocks(iDivUp(_host_res.measures.depth_res.x, threads.x), iDivUp(_host_res.measures.depth_res.y, threads.y));

    for(unsigned int i = 0; i < 4; i++)
    {
        thrust::device_ptr<float> d_ptr = thrust::device_pointer_cast(&_dev_res.alignment_error[i][0]);
        thrust::device_vector<float>::iterator iter = thrust::max_element(d_ptr, d_ptr + _dev_res.pitch_alignment_error * _host_res.measures.depth_res.y);
        int pos = thrust::device_pointer_cast(&(iter[0])) - d_ptr;

        // printf("\npos: %i\n", pos);

        kernel_normalize_alignment_errors<<<blocks, threads>>>(pos, _dev_res, i, _host_res.measures);
        getLastCudaError("kernel_normalize_alignment_errors execution failed\n");
    }
}

__host__ void reject_misaligned_deformations()
{
    int block_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_reject_misaligned_deformations, 0, 0);
    size_t grid_size = (_host_res.active_ed_nodes_count + block_size - 1) / block_size;
    kernel_reject_misaligned_deformations<<<grid_size, block_size>>>(_host_res.active_ed_nodes_count, _dev_res, _host_res);
    getLastCudaError("render kernel failed");
    cudaDeviceSynchronize();
}

__host__ void project_vertices()
{
    int block_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_warp, 0, 0);
    size_t grid_size = (_host_res.active_ed_nodes_count + block_size - 1) / block_size;
    kernel_project<<<grid_size, block_size>>>(_host_res.active_ed_nodes_count, _dev_res, _host_res.measures);
    getLastCudaError("render kernel failed");
    cudaDeviceSynchronize();
}

__host__ void project_warped_vertices()
{
    int block_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_warp, 0, 0);
    size_t grid_size = (_host_res.active_ed_nodes_count + block_size - 1) / block_size;
    kernel_project_warped<<<grid_size, block_size>>>(_host_res.active_ed_nodes_count, _dev_res, _host_res.measures);
    getLastCudaError("render kernel failed");
    cudaDeviceSynchronize();
}

__host__ void apply_warp()
{
    int block_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_warp, 0, 0);
    size_t grid_size = (_host_res.active_ed_nodes_count + block_size - 1) / block_size;
    kernel_warp<<<grid_size, block_size>>>(_host_res.active_ed_nodes_count, _dev_res, _host_res.measures);
    getLastCudaError("render kernel failed");
    cudaDeviceSynchronize();
}

extern "C" double pcg_solve(struct_native_handles &native_handles)
{
    TimerGPU timer(0);

    map_tsdf_volumes();
    map_kinect_textures();

    const unsigned int max_iterations = _host_res.configuration.solver_lma_steps;
    unsigned int iterations = 0u;
    float mu = _host_res.configuration.solver_mu;
    float initial_misalignment_energy, solution_misalignment_energy;
    int singularity = 0;

    project_vertices();
    evaluate_misalignment_energy(initial_misalignment_energy);

    while(iterations < max_iterations)
    {
        clean_pcg_resources();
        cudaDeviceSynchronize();

        evaluate_jtj_jtf(mu);
        cudaDeviceSynchronize();

        convert_to_csr();

#ifdef DEBUG_JTJ

        print_out_jtj();

#endif

#ifdef DEBUG_JTF

        print_out_jtf();

#endif

#ifdef DEBUG_H

        print_out_h();

#endif

        singularity = solve_for_h();
        cudaDeviceSynchronize();

        if(singularity != -1)
        {
#ifdef VERBOSE
            printf("\nsingularity encountered\n");
#endif

            break;
        }

        evaluate_step_misalignment_energy(solution_misalignment_energy, mu);

#ifdef VERBOSE
        printf("\ninitial E: % f, solution E: %f\n", initial_misalignment_energy, solution_misalignment_energy);
#endif

        if(solution_misalignment_energy < initial_misalignment_energy && (unsigned int)(solution_misalignment_energy * 1000) != 0u)
        {
#ifdef VERBOSE
            printf("\naccepted step, initial E: % f, solution E: %f\n", initial_misalignment_energy, solution_misalignment_energy);
#endif

            int N = (int)_host_res.active_ed_nodes_count * ED_COMPONENT_COUNT;
            float one = 1.f;
            cublasSaxpy(cublas_handle, N, &one, _dev_res.h, 1, (float *)&_dev_res.ed_graph[0], 1);
            cudaDeviceSynchronize();

            mu -= _host_res.configuration.solver_mu_step;
            initial_misalignment_energy = solution_misalignment_energy;
#ifdef VERBOSE
            printf("\nmu lowered: %f\n", mu);
#endif
        }
        else
        {
#ifdef VERBOSE
            printf("\nrejected step, initial E: % f, solution E: %f\n", initial_misalignment_energy, solution_misalignment_energy);
#endif

            mu += _host_res.configuration.solver_mu_step;
#ifdef VERBOSE
            printf("\nmu raised: %f\n", mu);
#endif
        }

        iterations++;
    }

    apply_warp();
    project_warped_vertices();

    evaluate_alignment_error();

#ifdef REJECT_MISALIGNED
    reject_misaligned_deformations();
#endif

    unmap_kinect_textures();
    unmap_tsdf_volumes();

    checkCudaErrors(cudaThreadSynchronize());
    return timer.read();
}