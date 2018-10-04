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

#ifdef DEBUG_CONVERGENCE

#include "../../../external/csv/ostream.hpp"

#endif

CUDA_HOST_DEVICE float evaluate_ed_residual(struct_ed_node &ed_node, struct_ed_meta_entry &ed_entry, struct_device_resources &dev_res, struct_host_resources &host_res)
{
    float residual = 0.f;

    glm::vec3 translation(ed_node.translation[0], ed_node.translation[1], ed_node.translation[2]);

#pragma unroll
    for(unsigned int i = 0; i < host_res.active_ed_nodes_count; i++)
    {
        struct_ed_node neighbor_node = dev_res.ed_graph[i];
        struct_ed_meta_entry neighbor_node_meta = dev_res.ed_graph_meta[i];

        float dist = glm::length(neighbor_node_meta.position - ed_entry.position);

        if(dist > 2.f * host_res.measures.size_ed_cell)
        {
            continue;
        }

        glm::vec3 neighbor_translation(neighbor_node.translation[0], neighbor_node.translation[1], neighbor_node.translation[2]);
        glm::fquat neighbor_rotation(1.f, neighbor_node.rotation[0], neighbor_node.rotation[1], neighbor_node.rotation[2]);

        float residual_component = 0.f;

        glm::fvec3 diff = ed_entry.position - neighbor_node_meta.position;

#ifdef FAST_QUAT_OPS
        residual_component = glm::length(qtransform(neighbor_rotation, diff) + neighbor_node_meta.position + neighbor_translation - ed_entry.position - translation);
#else
        residual_component = glm::length(glm::mat3_cast(neighbor_rotation) * diff + neighbor_node_meta.position + neighbor_translation - ed_entry.position - translation);
#endif

#ifdef ED_NODES_ROBUSTIFY
        residual_component = glm::sqrt(robustify(glm::pow(residual_component, 2.f)));
#endif

        if(isnan(residual_component))
        {
#ifdef DEBUG_NANS
            printf("\ned residual component is NaN!\n");
#endif

            residual_component = 0.f;
        }

        if(isinf(residual_component))
        {
#ifdef DEBUG_NANS
            printf("\ned residual component is Inf!\n");
#endif

            residual_component = 0.f;
        }

        residual += residual_component;
    }

    if(isnan(residual))
    {
#ifdef DEBUG_NANS
        printf("\ned residual is NaN!\n");
#endif

        residual = 0.f;
    }

    if(isinf(residual))
    {
#ifdef DEBUG_NANS
        printf("\ned residual is Inf!\n");
#endif

        residual = 0.f;
    }

    return host_res.configuration.weight_regularization * residual;
}

__device__ float evaluate_ed_partial_derivative(int component, struct_ed_node &ed_node, struct_ed_meta_entry &ed_entry, struct_device_resources &dev_res, struct_host_resources &host_res)
{
    float pd = 0.f;

    struct_ed_node ed_node_new;
    memcpy(&ed_node_new, &ed_node, sizeof(struct_ed_node));

    float ds = derivative_step(component, host_res.measures);
    shift_component(ed_node_new, component, ds);

    float ed_res_pos = evaluate_ed_residual(ed_node_new, ed_entry, dev_res, host_res);

    // printf("\nvx_residual: %f\n", vx_residual);

    if(isnan(ed_res_pos))
    {
#ifdef DEBUG_NANS
        printf("\nvx_residual is NaN!\n");
#endif

        ed_res_pos = 0.f;
    }

    shift_component(ed_node_new, component, ds);

    float ed_res_pos_pos = evaluate_ed_residual(ed_node_new, ed_entry, dev_res, host_res);

    // printf("\nvx_residual: %f\n", vx_residual);

    if(isnan(ed_res_pos_pos))
    {
#ifdef DEBUG_NANS
        printf("\nvx_residual is NaN!\n");
#endif

        ed_res_pos_pos = 0.f;
    }

    shift_component(ed_node_new, component, -3.f * ds);

    float ed_res_neg = evaluate_ed_residual(ed_node_new, ed_entry, dev_res, host_res);

    // printf("\nvx_residual: %f\n", vx_residual);

    if(isnan(ed_res_neg))
    {
#ifdef DEBUG_NANS
        printf("\nvx_residual is NaN!\n");
#endif

        ed_res_neg = 0.f;
    }

    shift_component(ed_node_new, component, -1.f * ds);

    float ed_res_neg_neg = evaluate_ed_residual(ed_node_new, ed_entry, dev_res, host_res);

    // printf("\nvx_residual: %f\n", vx_residual);

    if(isnan(ed_res_neg_neg))
    {
#ifdef DEBUG_NANS
        printf("\nvx_residual is NaN!\n");
#endif

        ed_res_neg_neg = 0.f;
    }

    ds *= 12.f;
    pd = (-ed_res_pos_pos + 8.f * ed_res_pos - 8.f * ed_res_neg + ed_res_neg_neg) / ds;

    if(isnan(pd))
    {
#ifdef DEBUG_NANS
        printf("\nvx_pds[%u] is NaN!\n", component);
#endif

        pd = 0.f;
    }

    return pd;
}

__global__ void kernel_calculate_opticflow_guess(unsigned int active_ed_nodes_count, struct_device_resources dev_res, struct_measures measures)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= active_ed_nodes_count)
    {
        return;
    }

    struct_ed_meta_entry ed_entry = dev_res.ed_graph_meta[idx];

    glm::fvec3 average_translation{0.f, 0.f, 0.f};
    int translations = 0;

#pragma unroll
    for(unsigned int vx_idx = 0; vx_idx < ed_entry.vx_length; vx_idx++)
    {
        unsigned int address = ed_entry.vx_offset + vx_idx;
        struct_vertex vx = dev_res.prev_frame_warped_sorted_vx_ptr[address];

        struct_projection vx_projection = dev_res.prev_frame_warped_sorted_vx_projections[address];
        for(unsigned int i = 0; i < 4; i++)
        {
            float4 projection = sample_cv_xyz_inv(dev_res.cv_xyz_inv_tex[i], vx.position);

            vx_projection.projection[i].x = projection.x;
            vx_projection.projection[i].y = projection.y;
        }

        for(int layer = 0; layer < 4; layer++)
        {
            if(vx_projection.projection[layer].x < 0.f || vx_projection.projection[layer].y < 0.f)
            {
#ifdef VERBOSE
                printf("\nprojected out of optical flow map: (%f,%f)\n", vx_projection.projection[layer].x, vx_projection.projection[layer].y);
#endif
                continue;
            }

            glm::fvec3 backprojected_position;
            if(!sample_prev_depth_projection(backprojected_position, layer, vx_projection.projection[layer], dev_res, measures))
            {
                continue;
            }

            glm::vec3 diff = backprojected_position - vx.position;

            if(glm::length(diff) > 0.03f)
            {
                continue;
            }

            float2 flow = sample_opticflow(dev_res.optical_flow_tex[layer], vx_projection.projection[layer]);

            // printf("\n(x,y): (%f,%f)\n", flow.x, flow.y);

            glm::vec2 new_projection = vx_projection.projection[layer] - glm::vec2(flow.x / ((float)measures.depth_res.x), flow.y / ((float)measures.depth_res.y));

            glm::fvec3 flow_position;
            if(!sample_depth_projection(flow_position, layer, new_projection, dev_res, measures))
            {
                continue;
            }

            glm::fvec3 optical_flow_vector = flow_position - vx.position;

            bool is_nan = isnan(optical_flow_vector.x) || isnan(optical_flow_vector.y) || isnan(optical_flow_vector.z);

            if(is_nan)
            {
#ifdef DEBUG_NANS
                printf("\nNaN in optical flow guess evaluation\n");
#endif
                optical_flow_vector = glm::fvec3(0.f);
            }

            average_translation += optical_flow_vector;
            translations++;
        }
    }

    average_translation /= (float)translations;

    bool is_nan = isnan(average_translation.x) || isnan(average_translation.y) || isnan(average_translation.z);

    if(is_nan)
    {
#ifdef DEBUG_NANS
        printf("\nNaN in optical flow guess evaluation\n");
#endif
        average_translation = glm::fvec3(0.f);
    }

    memcpy(&dev_res.h[idx * ED_COMPONENT_COUNT], &average_translation, 3 * sizeof(float));
}

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
        }
        memcpy(&dev_res.sorted_vx_projections[address], &vx_projection, sizeof(struct_projection));
    }
}

__global__ void kernel_project_prev_frame_warped(unsigned int active_ed_nodes_count, struct_device_resources dev_res, struct_measures measures)
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
        struct_vertex vx = dev_res.prev_frame_warped_sorted_vx_ptr[address];

        struct_projection vx_projection = dev_res.prev_frame_warped_sorted_vx_projections[address];
        for(unsigned int i = 0; i < 4; i++)
        {
            float4 projection = sample_cv_xyz_inv(dev_res.cv_xyz_inv_tex[i], vx.position);

            vx_projection.projection[i].x = projection.x;
            vx_projection.projection[i].y = projection.y;
        }
        memcpy(&dev_res.prev_frame_warped_sorted_vx_projections[address], &vx_projection, sizeof(struct_projection));
    }
}

__global__ void kernel_prev_frame_warp(struct_device_resources dev_res, struct_host_resources host_res)
{
    unsigned long long int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int vx_per_thread = (unsigned int)max(1u, (unsigned int)(host_res.active_ed_vx_count / (blockDim.x * gridDim.x)));

    for(unsigned int i = 0; i < vx_per_thread; i++)
    {
        unsigned long long int vertex_position = idx * vx_per_thread + i;

        if(vertex_position >= host_res.active_ed_vx_count)
        {
            return;
        }

        struct_vertex vx = dev_res.sorted_vx_ptr[vertex_position];

        unsigned long long int sorted_vx_ptr_offset = 0u;
        bool found_ed = false; // TODO: what if not?

        for(unsigned int ed_position = 0; ed_position < host_res.prev_frame_ed_nodes_count; ed_position++)
        {
            struct_ed_meta_entry ed_entry = dev_res.prev_ed_graph_meta[ed_position];

            if(ed_entry.brick_id != vx.brick_id)
            {
                continue;
            }

            if(ed_entry.ed_cell_id != vx.ed_cell_id)
            {
                continue;
            }

            struct_ed_node ed_node = dev_res.prev_ed_graph[ed_position];

            glm::fvec3 dist = vx.position - ed_entry.position;

            struct_vertex warped_vx;
            memset(&warped_vx, 0, sizeof(struct_vertex));
            warped_vx.position = glm::clamp(warp_position(dist, ed_node, ed_entry, 1.f, host_res.measures), glm::fvec3(0.f), glm::fvec3(1.f));
            warped_vx.brick_id = vx.brick_id;
            warped_vx.normal = warp_normal(vx.normal, ed_node, ed_entry, 1.f, host_res.measures);
            warped_vx.ed_cell_id = vx.ed_cell_id;

            memcpy(&dev_res.prev_frame_warped_sorted_vx_ptr[vertex_position], &warped_vx, sizeof(struct_vertex));

            found_ed = true;

            break;
        }

        __syncthreads();
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

        glm::fvec3 dist = vx.position - ed_entry.position;

        struct_vertex warped_vx = dev_res.warped_sorted_vx_ptr[address];
        warped_vx.position = glm::clamp(warp_position(dist, ed_node, ed_entry, 1.f, measures), glm::fvec3(0.f), glm::fvec3(1.f));
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

            if(warped_projection.projection[i].x < 0.f || warped_projection.projection[i].y < 0.f)
            {
                /*#ifdef VERBOSE
                                printf("\nprojected out of depth map: (%u,%u)\n", warped_projection.projection[i].x, warped_projection.projection[i].y);
                #endif*/
                continue;
            }

            float vx_error = 0.f;

            glm::fvec3 extracted_position;
            if(!sample_depth_projection(extracted_position, i, warped_projection.projection[i], dev_res, host_res.measures))
            {
                vx_error = 1.f;
            }
            else
            {
                //        printf("\nextracted_position (%.2f, %.2f, %.2f): (%.2f,%.2f,%.2f) = (%.2f,%.2f,%.2f)\n", coordinate.x, coordinate.y, coordinate.z, warped_position.x, warped_position.y,
                //        warped_position.z, extracted_position.x, extracted_position.y, extracted_position.z);

                glm::fvec3 diff = warped_vertex.position - extracted_position;

                if(glm::length(diff) > 0.03f)
                {
                    vx_error = 1.f;
                }
                else
                {
                    vx_error = glm::length(diff);

                    // printf("\nvx_error: %f\n", vx_error);

                    if(isnan(vx_error))
                    {
#ifdef DEBUG_NANS
                        printf("\nvx_residual is NaN!\n");
#endif

                        vx_error = 0.f;
                    }
                }
            }

            dev_res.warped_sorted_vx_error[i][ed_entry.vx_offset + vx_idx] = vx_error;
        }
    }
}

__global__ void kernel_reject_misaligned_deformations(int *misaligned_vx_count, unsigned int active_ed_nodes_count, struct_device_resources dev_res, struct_host_resources host_res)
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

    float data_term = 0.f;
    float hull_term = 0.f;
    float correspondence_term = 0.f;

#pragma unroll
    for(unsigned int vx_idx = 0; vx_idx < ed_entry.vx_length; vx_idx++)
    {
        struct_vertex vx = dev_res.warped_sorted_vx_ptr[ed_entry.vx_offset + vx_idx];
        struct_projection vx_proj = dev_res.warped_sorted_vx_projections[ed_entry.vx_offset + vx_idx];

        // printf("\ned_node + vertex match\n");

        float vx_misalignment = 0.f;
        float data_misalignment = 0.f;
        float hull_misalignment = 0.f;

#ifdef EVALUATE_DATA
        data_misalignment = evaluate_vx_misalignment(vx, host_res.measures);
#endif

#ifdef EVALUATE_VISUAL_HULL
        hull_misalignment = evaluate_hull_residual(vx_proj, dev_res, host_res.measures);
#endif

        if(hull_misalignment == 0.f)
        {
            vx_misalignment = data_misalignment;
        }
        else
        {
            vx_misalignment = glm::min(data_misalignment, hull_misalignment);
        }

        // printf("\nvx_residual: %f\n", vx_residual);

        if(isnan(vx_misalignment))
        {
#ifdef DEBUG_NANS
            printf("\nvx_residual is NaN!\n");
#endif

            vx_misalignment = 0.f;
        }

        if(vx_misalignment > host_res.configuration.rejection_threshold)
        {
            atomicAdd(misaligned_vx_count, 1);
        }

        energy += vx_misalignment;
        data_term += evaluate_data_residual(vx, vx_proj, dev_res, host_res.measures);
        hull_term += hull_misalignment;
        correspondence_term += evaluate_cf_residual(vx, dev_res.prev_frame_warped_sorted_vx_ptr[ed_entry.vx_offset + vx_idx], vx_proj, dev_res, host_res.measures);
    }

    energy /= (float)ed_entry.vx_length;
    data_term /= (float)ed_entry.vx_length;
    hull_term /= (float)ed_entry.vx_length;
    correspondence_term /= (float)ed_entry.vx_length;

    ed_entry.misalignment_error = energy;
    ed_entry.data_term = data_term;
    ed_entry.hull_term = hull_term;
    ed_entry.correspondence_term = correspondence_term;
    ed_entry.regularization_term = evaluate_ed_residual(dev_res.ed_graph[idx], ed_entry, dev_res, host_res);

    // printf("\nreg: %f\n", ed_entry.regularization_term);
    // printf("\nhull: %f\n", ed_entry.hull_term);

    // printf("\nenergy: %f\n", energy);

    memcpy(&dev_res.ed_graph_meta[idx], &ed_entry, sizeof(struct_ed_meta_entry));
}

__device__ float evaluate_vx_residual(struct_vertex &prev_warped_vx, struct_projection &prev_warped_vx_proj, struct_vertex &ref_vx, struct_projection &ref_vx_proj, struct_device_resources &dev_res,
                                      struct_host_resources &host_res)
{
    // return host_res.configuration.weight_data * glm::length(glm::fvec3(0.5f, 0.64f, 0.36f) - vx.position);
    // return host_res.configuration.weight_data * glm::abs(0.64f - vx.position.y);

    float vx_residual = 0.f;

#ifdef EVALUATE_DATA
    vx_residual += host_res.configuration.weight_data * evaluate_data_residual(prev_warped_vx, prev_warped_vx_proj, dev_res, host_res.measures);
#endif

#ifdef EVALUATE_VISUAL_HULL
    vx_residual += host_res.configuration.weight_hull * evaluate_hull_residual(prev_warped_vx_proj, dev_res, host_res.measures);
#endif

#ifdef EVALUATE_CORRESPONDENCE_FIELD
    vx_residual += host_res.configuration.weight_correspondence * evaluate_cf_residual(prev_warped_vx, ref_vx, ref_vx_proj, dev_res, host_res.measures);
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

    float ed_residual = evaluate_ed_residual(ed_node, ed_entry, dev_res, host_res);

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
        struct_vertex vx = dev_res.warped_sorted_vx_ptr[ed_entry.vx_offset + vx_idx];
        struct_projection vx_proj = dev_res.warped_sorted_vx_projections[ed_entry.vx_offset + vx_idx];
        struct_vertex ref_vx = dev_res.prev_frame_warped_sorted_vx_ptr[ed_entry.vx_offset + vx_idx];
        struct_projection ref_vx_proj = dev_res.prev_frame_warped_sorted_vx_projections[ed_entry.vx_offset + vx_idx];

        // printf("\ned_node + vertex match\n");

        float vx_residual = evaluate_vx_residual(vx, vx_proj, ref_vx, ref_vx_proj, dev_res, host_res);

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
    float ed_residual = evaluate_ed_residual(ed_node, ed_entry, dev_res, host_res);

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

        glm::fvec3 dist = vx.position - ed_entry.position;

        struct_vertex warped_vx;
        warped_vx.position = glm::clamp(warp_position(dist, ed_node, ed_entry, 1.f, measures), glm::fvec3(0.f), glm::fvec3(1.f));
        warped_vx.brick_id = vx.brick_id;
        warped_vx.normal = warp_normal(vx.normal, ed_node, ed_entry, 1.f, measures);
        warped_vx.ed_cell_id = vx.ed_cell_id;

        struct_projection warped_vx_projection;
        for(unsigned int i = 0; i < 4; i++)
        {
            float4 projection = sample_cv_xyz_inv(dev_res.cv_xyz_inv_tex[i], warped_vx.position);

            warped_vx_projection.projection[i].x = projection.x;
            warped_vx_projection.projection[i].y = projection.y;
        }

        struct_vertex ref_vx = dev_res.prev_frame_warped_sorted_vx_ptr[ed_entry.vx_offset + vx_idx];
        struct_projection ref_vx_proj = dev_res.prev_frame_warped_sorted_vx_projections[ed_entry.vx_offset + vx_idx];

        // printf("\ned_node + vertex match\n");

        float vx_residual = evaluate_vx_residual(warped_vx, warped_vx_projection, ref_vx, ref_vx_proj, dev_res, host_res);

        atomicAdd(energy, vx_residual);
    }

    // printf("\njtf[%u]\n", ed_node_offset * 10u);
}

__device__ float evaluate_vx_partial_derivative(int component, struct_ed_node &ed_node, struct_ed_meta_entry &ed_entry, struct_vertex &initial_vx, struct_vertex &prev_warped_vx,
                                                struct_projection &prev_warped_vx_proj, struct_vertex &ref_vx, struct_projection &ref_vx_proj, struct_device_resources &dev_res,
                                                struct_host_resources &host_res)
{
    float pd = 0.f;

    struct_ed_node ed_node_new;
    memcpy(&ed_node_new, &ed_node, sizeof(struct_ed_node));

    glm::fvec3 dist = initial_vx.position - ed_entry.position;

    struct_vertex warped_vx;
    struct_projection warped_vx_proj;
    memset(&warped_vx, 0, sizeof(struct_vertex));
    memset(&warped_vx_proj, 0, sizeof(struct_projection));

    float ds = derivative_step(component, host_res.measures);
    shift_component(ed_node_new, component, ds);

    warped_vx.position = glm::clamp(warp_position(dist, ed_node_new, ed_entry, 1.f, host_res.measures), glm::fvec3(0.f), glm::fvec3(1.f));
    warped_vx.normal = prev_warped_vx.normal; /// Per-step approximation
    for(unsigned int i = 0; i < 4; i++)
    {
        float4 projection = sample_cv_xyz_inv(dev_res.cv_xyz_inv_tex[i], warped_vx.position);
        warped_vx_proj.projection[i].x = projection.x;
        warped_vx_proj.projection[i].y = projection.y;
    }

    float vx_res_pos = 0.f;

#ifdef EVALUATE_DATA
    vx_res_pos += host_res.configuration.weight_data * evaluate_data_residual(warped_vx, prev_warped_vx_proj /* per-step approximation */, dev_res, host_res.measures);
#endif

#ifdef EVALUATE_VISUAL_HULL
    vx_res_pos += host_res.configuration.weight_hull * evaluate_hull_residual(warped_vx_proj, dev_res, host_res.measures);
#endif

#ifdef EVALUATE_CORRESPONDENCE_FIELD
    vx_res_pos += host_res.configuration.weight_correspondence * evaluate_cf_residual(warped_vx, ref_vx, ref_vx_proj, dev_res, host_res.measures);
#endif

    // printf("\nvx_residual: %f\n", vx_residual);

    if(isnan(vx_res_pos))
    {
#ifdef DEBUG_NANS
        printf("\nvx_residual is NaN!\n");
#endif

        vx_res_pos = 0.f;
    }

    shift_component(ed_node_new, component, ds);

    warped_vx.position = glm::clamp(warp_position(dist, ed_node_new, ed_entry, 1.f, host_res.measures), glm::fvec3(0.f), glm::fvec3(1.f));
    warped_vx.normal = prev_warped_vx.normal; /// Per-step approximation
    for(unsigned int i = 0; i < 4; i++)
    {
        float4 projection = sample_cv_xyz_inv(dev_res.cv_xyz_inv_tex[i], warped_vx.position);
        warped_vx_proj.projection[i].x = projection.x;
        warped_vx_proj.projection[i].y = projection.y;
    }

    float vx_res_pos_pos = 0.f;

#ifdef EVALUATE_DATA
    vx_res_pos_pos += host_res.configuration.weight_data * evaluate_data_residual(warped_vx, prev_warped_vx_proj /* per-step approximation */, dev_res, host_res.measures);
#endif

#ifdef EVALUATE_VISUAL_HULL
    vx_res_pos_pos += host_res.configuration.weight_hull * evaluate_hull_residual(warped_vx_proj, dev_res, host_res.measures);
#endif

#ifdef EVALUATE_CORRESPONDENCE_FIELD
    vx_res_pos_pos += host_res.configuration.weight_correspondence * evaluate_cf_residual(warped_vx, ref_vx, ref_vx_proj, dev_res, host_res.measures);
#endif

    // printf("\nvx_residual: %f\n", vx_residual);

    if(isnan(vx_res_pos_pos))
    {
#ifdef DEBUG_NANS
        printf("\nvx_residual is NaN!\n");
#endif

        vx_res_pos_pos = 0.f;
    }

    shift_component(ed_node_new, component, -3.f * ds);

    warped_vx.position = glm::clamp(warp_position(dist, ed_node_new, ed_entry, 1.f, host_res.measures), glm::fvec3(0.f), glm::fvec3(1.f));
    warped_vx.normal = prev_warped_vx.normal; /// Per-step approximation
    for(unsigned int i = 0; i < 4; i++)
    {
        float4 projection = sample_cv_xyz_inv(dev_res.cv_xyz_inv_tex[i], warped_vx.position);
        warped_vx_proj.projection[i].x = projection.x;
        warped_vx_proj.projection[i].y = projection.y;
    }

    float vx_res_neg = 0.f;

#ifdef EVALUATE_DATA
    vx_res_neg += host_res.configuration.weight_data * evaluate_data_residual(warped_vx, prev_warped_vx_proj /* per-step approximation */, dev_res, host_res.measures);
#endif

#ifdef EVALUATE_VISUAL_HULL
    vx_res_neg += host_res.configuration.weight_hull * evaluate_hull_residual(warped_vx_proj, dev_res, host_res.measures);
#endif

#ifdef EVALUATE_CORRESPONDENCE_FIELD
    vx_res_neg += host_res.configuration.weight_correspondence * evaluate_cf_residual(warped_vx, ref_vx, ref_vx_proj, dev_res, host_res.measures);
#endif

    // printf("\nvx_residual: %f\n", vx_residual);

    if(isnan(vx_res_neg))
    {
#ifdef DEBUG_NANS
        printf("\nvx_residual is NaN!\n");
#endif

        vx_res_neg = 0.f;
    }

    shift_component(ed_node_new, component, -1.f * ds);

    warped_vx.position = glm::clamp(warp_position(dist, ed_node_new, ed_entry, 1.f, host_res.measures), glm::fvec3(0.f), glm::fvec3(1.f));
    warped_vx.normal = prev_warped_vx.normal; /// Per-step approximation
    for(unsigned int i = 0; i < 4; i++)
    {
        float4 projection = sample_cv_xyz_inv(dev_res.cv_xyz_inv_tex[i], warped_vx.position);
        warped_vx_proj.projection[i].x = projection.x;
        warped_vx_proj.projection[i].y = projection.y;
    }

    float vx_res_neg_neg = 0.f;

#ifdef EVALUATE_DATA
    vx_res_neg_neg += host_res.configuration.weight_data * evaluate_data_residual(warped_vx, prev_warped_vx_proj /* per-step approximation */, dev_res, host_res.measures);
#endif

#ifdef EVALUATE_VISUAL_HULL
    vx_res_neg_neg += host_res.configuration.weight_hull * evaluate_hull_residual(warped_vx_proj, dev_res, host_res.measures);
#endif

#ifdef EVALUATE_CORRESPONDENCE_FIELD
    vx_res_neg_neg += host_res.configuration.weight_correspondence * evaluate_cf_residual(warped_vx, ref_vx, ref_vx_proj, dev_res, host_res.measures);
#endif

    // printf("\nvx_residual: %f\n", vx_residual);

    if(isnan(vx_res_neg_neg))
    {
#ifdef DEBUG_NANS
        printf("\nvx_residual is NaN!\n");
#endif

        vx_res_neg_neg = 0.f;
    }

    ds *= 12.f;
    pd = (-vx_res_pos_pos + 8.f * vx_res_pos - 8.f * vx_res_neg + vx_res_neg_neg) / ds;

    if(isnan(pd))
    {
#ifdef DEBUG_NANS
        printf("\nvx_pds[%u] is NaN!\n", component);
#endif

        pd = 0.f;
    }

    return pd;
}

__global__ void kernel_jtj_mu(const float mu, struct_device_resources dev_res, struct_host_resources host_res, struct_measures measures)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float shared_jtj_coo_val_block[ED_COMPONENT_COUNT * ED_COMPONENT_COUNT];

    unsigned int ed_node_offset = idx / ED_COMPONENT_COUNT;
    if(ed_node_offset >= host_res.active_ed_nodes_count)
    {
        // printf("\ned_node_offset overshot: %u, ed_nodes_count: %u\n", ed_node_offset, ed_nodes_count);
        return;
    }

    unsigned int component = idx % ED_COMPONENT_COUNT;

    memcpy(&shared_jtj_coo_val_block[component * ED_COMPONENT_COUNT], &dev_res.jtj_vals[idx * ED_COMPONENT_COUNT], sizeof(float) * ED_COMPONENT_COUNT);

    __syncthreads();

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

    memcpy(&dev_res.jtj_mu_vals[idx * ED_COMPONENT_COUNT], &shared_jtj_coo_val_block[component * ED_COMPONENT_COUNT], sizeof(float) * ED_COMPONENT_COUNT);
}

__global__ void kernel_jtj_jtf(struct_device_resources dev_res, struct_host_resources host_res, struct_measures measures)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float shared_jtj_coo_val_block[ED_COMPONENT_COUNT * ED_COMPONENT_COUNT];
    __shared__ float shared_jtf_block[ED_COMPONENT_COUNT];

    unsigned int ed_node_offset = idx / ED_COMPONENT_COUNT;
    if(ed_node_offset >= host_res.active_ed_nodes_count)
    {
        // printf("\ned_node_offset overshot: %u, ed_nodes_count: %u\n", ed_node_offset, ed_nodes_count);
        return;
    }

    struct_ed_meta_entry ed_entry = dev_res.ed_graph_meta[ed_node_offset];
    struct_ed_node ed_node = dev_res.ed_graph[ed_node_offset];

    unsigned int component = idx % ED_COMPONENT_COUNT;

#ifdef EVALUATE_ED_REGULARIZATION

    __shared__ float ed_residual;

    if(component == 0)
    {
        ed_residual = evaluate_ed_residual(ed_node, ed_entry, dev_res, host_res);
    }

    __shared__ float ed_pds[ED_COMPONENT_COUNT];

    ed_pds[component] = evaluate_ed_partial_derivative(component, ed_node, ed_entry, dev_res, host_res);

    __syncthreads();

    if(isnan(ed_residual))
    {
#ifdef DEBUG_NANS
        printf("\ned_residual is NaN!\n");
#endif

        ed_residual = 0.f;
    }

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

    atomicAdd(&shared_jtf_block[component], jtf_value);

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

        atomicAdd(&shared_jtj_coo_val_block[jtj_pos], jtj_value);
    }
#endif

    __syncthreads();

    // printf("\ned_node position: (%f,%f,%f)\n", ed_node.position.x, ed_node.position.y, ed_node.position.z);
    for(unsigned int vx_idx = 0; vx_idx < ed_entry.vx_length; vx_idx++)
    {
        struct_vertex warped_vx = dev_res.warped_sorted_vx_ptr[ed_entry.vx_offset + vx_idx];
        struct_projection warped_vx_proj = dev_res.warped_sorted_vx_projections[ed_entry.vx_offset + vx_idx];

        struct_vertex initial_vx = dev_res.sorted_vx_ptr[ed_entry.vx_offset + vx_idx];
        struct_vertex ref_vx = dev_res.prev_frame_warped_sorted_vx_ptr[ed_entry.vx_offset + vx_idx];
        struct_projection ref_vx_proj = dev_res.prev_frame_warped_sorted_vx_projections[ed_entry.vx_offset + vx_idx];

        // printf("\ned_node + vertex match\n");

        __shared__ float vx_residual;

        if(component == 0)
        {
            vx_residual = evaluate_vx_residual(warped_vx, warped_vx_proj, ref_vx, ref_vx_proj, dev_res, host_res);
        }

        __shared__ float pds[ED_COMPONENT_COUNT];

        pds[component] = evaluate_vx_partial_derivative(component, ed_node, ed_entry, initial_vx, warped_vx, warped_vx_proj, ref_vx, ref_vx_proj, dev_res, host_res);

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

    __syncthreads();

    // printf("\njtf[%u]\n", ed_node_offset * 10u);

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
        cusolverSpScsrlsvchol(cusolver_handle, N, csr_nnz, descr, _dev_res.jtj_mu_vals, _dev_res.jtj_rows_csr, _dev_res.jtj_cols, _dev_res.jtf, tol, reorder, _dev_res.h, &singularity);

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
        cusolverSpScsrlsvqr(cusolver_handle, N, csr_nnz, descr, _dev_res.jtj_mu_vals, _dev_res.jtj_rows_csr, _dev_res.jtj_cols, _dev_res.jtf, tol, reorder, _dev_res.h, &singularity);

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
    cusparseStatus_t status = cusparseScsrmv(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, csr_nnz, &alpha, descr, _dev_res.jtj_mu_vals, _dev_res.jtj_rows_csr, _dev_res.jtj_cols,
                                             _dev_res.h, &beta, _dev_res.pcg_Ax);
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

    cudaMemcpy(&host_jtj_vals[0], &_dev_res.jtj_mu_vals[0], _host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT * sizeof(float), cudaMemcpyDeviceToHost);
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

__host__ void evaluate_step_jtj_jtf()
{
    size_t grid_size = _host_res.active_ed_nodes_count;

    kernel_jtj_jtf<<<grid_size, ED_COMPONENT_COUNT>>>(_dev_res, _host_res, _host_res.measures);
    getLastCudaError("render kernel failed");
    cudaDeviceSynchronize();

    kernel_jtj_coo_cols_rows<<<grid_size, ED_COMPONENT_COUNT>>>(_host_res.active_ed_nodes_count, _dev_res, _host_res.measures);
    getLastCudaError("render kernel failed");
    cudaDeviceSynchronize();
};

__host__ void add_jtj_step_mu(const float mu)
{
    size_t grid_size = _host_res.active_ed_nodes_count;

    kernel_jtj_mu<<<grid_size, ED_COMPONENT_COUNT>>>(mu, _dev_res, _host_res, _host_res.measures);
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

__host__ void evaluate_step_misalignment_energy(float &misalignment_energy)
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

__global__ void kernel_calculate_alignment_statistics(float *mean_deviation, unsigned int active_ed_nodes_count, struct_device_resources dev_res, struct_host_resources host_res)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= active_ed_nodes_count)
    {
        return;
    }

    struct_ed_node ed_node = dev_res.ed_graph[idx];

    glm::fvec3 translation(ed_node.translation[0], ed_node.translation[1], ed_node.translation[2]);

    glm::bvec3 ed_nan = glm::isnan(translation);

    if(ed_nan.x || ed_nan.y || ed_nan.z)
    {
        printf("\nNaN in deformation estimation\n");
        translation = glm::fvec3(0.f);
    }

    glm::fvec3 step_translation = glm::fvec3(0.0714f, 0.f, 0.0714f);

    // printf("\nstep_translation: (%.2f,%.2f,%.2f)\n", step_translation.x, step_translation.y, step_translation.z);

    float deviation = glm::length(translation - step_translation);

    // printf("\ned translation: (%.2f,%.2f,%.2f)\n", translation.x, translation.y, translation.z);

    if(glm::isnan(deviation))
    {
        printf("\nNaN in deviation\n");
        deviation = 0.f;
    }
    else
    {
        atomicAdd(mean_deviation, deviation);
    }

    /*dev_res.ed_graph[idx].translation = glm::fvec3(0., 0.2, 0.);
    dev_res.ed_graph[idx].rotation = glm::fquat(0., 0., 0., 0.);*/
}

__host__ void calculate_alignment_statistics(float *mad)
{
    int block_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_calculate_alignment_statistics, 0, 0);
    size_t grid_size = (_host_res.active_ed_nodes_count + block_size - 1) / block_size;
    kernel_calculate_alignment_statistics<<<grid_size, block_size>>>(mad, _host_res.active_ed_nodes_count, _dev_res, _host_res);
    getLastCudaError("kernel_calculate_alignment_statistics failed");
    cudaDeviceSynchronize();
}

#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Interpolation_traits_2.h>
#include <CGAL/interpolation_functions.h>
#include <CGAL/natural_neighbor_coordinates_2.h>
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_triangulation_2<K> Delaunay_triangulation;
typedef CGAL::Interpolation_traits_2<K> Traits;
typedef K::FT Coord_type;
typedef K::Point_2 Point;

template <class T>
struct normalize_alignment_error_functor
{
    T _max;
    normalize_alignment_error_functor(T max) { _max = max; }
    __host__ __device__ T operator()(T &x) const { return glm::min(1.f, x / _max); }
};

__host__ void evaluate_alignment_error()
{
    for(unsigned int i = 0; i < 4; i++)
    {
        checkCudaErrors(cudaMemset2D(_dev_res.alignment_error[i], _dev_res.pitch_alignment_error, 0, _host_res.measures.depth_res.x * sizeof(float), _host_res.measures.depth_res.y));
        checkCudaErrors(cudaMemset(_dev_res.warped_sorted_vx_error[i], 0, _host_res.active_ed_vx_count * sizeof(float)));
    }

    int block_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_reject_misaligned_deformations, 0, 0);
    size_t grid_size = (_host_res.active_ed_nodes_count + block_size - 1) / block_size;
    kernel_evaluate_alignment_error<<<grid_size, block_size>>>(_host_res.active_ed_nodes_count, _dev_res, _host_res);
    getLastCudaError("render kernel failed");
    cudaDeviceSynchronize();

    struct_projection *vx_projections = (struct_projection *)malloc(_host_res.active_ed_vx_count * sizeof(struct_projection));
    checkCudaErrors(cudaMemcpy(vx_projections, &_dev_res.warped_sorted_vx_projections[0], _host_res.active_ed_vx_count * sizeof(struct_projection), cudaMemcpyDeviceToHost));

    dim3 threads(8, 8);
    dim3 blocks(iDivUp(_host_res.measures.depth_res.x, threads.x), iDivUp(_host_res.measures.depth_res.y, threads.y));

    // TODO: speed up

    for(unsigned int i = 0; i < 4; i++)
    {
        thrust::device_ptr<float> d_ptr = thrust::device_pointer_cast(&_dev_res.warped_sorted_vx_error[i][0]);
        thrust::device_vector<float>::iterator iter = thrust::max_element(d_ptr, d_ptr + _host_res.active_ed_vx_count);

        normalize_alignment_error_functor<float> nf(*iter);
        thrust::transform(d_ptr, d_ptr + _host_res.active_ed_vx_count, d_ptr, nf);

#ifdef UNIT_TEST_ALIGNMENT_ERROR_MAP

        thrust::copy(d_ptr, d_ptr + _host_res.active_ed_vx_count, &_dev_res.warped_sorted_vx_error[i][0]);

#endif

        Delaunay_triangulation T;
        std::map<Point, Coord_type, K::Less_xy_2> function_values;
        typedef CGAL::Data_access<std::map<Point, Coord_type, K::Less_xy_2>> Value_access;

        /// Copy device resources to host pointers

        memset(_host_res.vx_error_values, 0, _host_res.active_ed_vx_count * sizeof(float));
        checkCudaErrors(cudaMemcpy(_host_res.vx_error_values, &_dev_res.warped_sorted_vx_error[i][0], _host_res.active_ed_vx_count * sizeof(float), cudaMemcpyDeviceToHost));

        memset(_host_res.silhouette, 0, _host_res.measures.depth_res.x * _host_res.measures.depth_res.y * sizeof(float));
        checkCudaErrors(cudaMemcpy(_host_res.silhouette, &_dev_res.kinect_silhouettes[i][0], _host_res.measures.depth_res.x * _host_res.measures.depth_res.y * sizeof(float), cudaMemcpyDeviceToHost));

        /// Populate scatter data

        for(int vx = 0; vx < _host_res.active_ed_vx_count; vx++)
        {
            K::Point_2 p(vx_projections[vx].projection[i].x, vx_projections[vx].projection[i].y);
            T.insert(p);
            function_values.insert(std::make_pair(p, _host_res.vx_error_values[vx]));
        }

        /// Evaluate the map

        memset(_host_res.vx_error_map, 0, _host_res.measures.depth_res.x * _host_res.measures.depth_res.y * sizeof(float));

        for(int x = 0; x < _host_res.measures.depth_res.x; x++)
        {
            for(int y = 0; y < _host_res.measures.depth_res.y; y++)
            {
                K::Point_2 p(x / ((float)_host_res.measures.depth_res.x), y / ((float)_host_res.measures.depth_res.y));
                std::vector<std::pair<Point, Coord_type>> coords;
                Coord_type norm = CGAL::natural_neighbor_coordinates_2(T, p, std::back_inserter(coords)).second;
                float res = (float)CGAL::linear_interpolation(coords.begin(), coords.end(), norm, Value_access(function_values));

                bool is_valid_silhouette = _host_res.silhouette[x + y * _host_res.measures.depth_res.x] == 1.f;
                _host_res.vx_error_map[x + y * _host_res.measures.depth_res.x] = is_valid_silhouette ? res : 1.f;
            }
        }

        checkCudaErrors(cudaMemcpy2D(&_dev_res.alignment_error[i][0], _dev_res.pitch_alignment_error, &_host_res.vx_error_map[0], _host_res.measures.depth_res.x * sizeof(float),
                                     _host_res.measures.depth_res.x * sizeof(float), _host_res.measures.depth_res.y, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    free(vx_projections);
}

__host__ void reject_misaligned_deformations(int &misaligned_vx)
{
    int *d_misaligned_vx = nullptr;
    cudaMallocManaged(&d_misaligned_vx, sizeof(int));
    *d_misaligned_vx = 0;

    int block_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_reject_misaligned_deformations, 0, 0);
    size_t grid_size = (_host_res.active_ed_nodes_count + block_size - 1) / block_size;
    kernel_reject_misaligned_deformations<<<grid_size, block_size>>>(d_misaligned_vx, _host_res.active_ed_nodes_count, _dev_res, _host_res);
    getLastCudaError("render kernel failed");
    cudaDeviceSynchronize();

    misaligned_vx = d_misaligned_vx[0];

    checkCudaErrors(cudaFree(d_misaligned_vx));
}

#ifdef INITIAL_GUESS_OPTICFLOW

__host__ void calculate_opticflow_guess()
{
    int block_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_warp, 0, 0);
    size_t grid_size = (_host_res.active_ed_nodes_count + block_size - 1) / block_size;
    kernel_calculate_opticflow_guess<<<grid_size, block_size>>>(_host_res.active_ed_nodes_count, _dev_res, _host_res.measures);
    getLastCudaError("render kernel failed");
    cudaDeviceSynchronize();
}

#endif

__host__ void project_vertices()
{
    int block_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_project, 0, 0);
    size_t grid_size = (_host_res.active_ed_nodes_count + block_size - 1) / block_size;
    kernel_project<<<grid_size, block_size>>>(_host_res.active_ed_nodes_count, _dev_res, _host_res.measures);
    getLastCudaError("render kernel failed");
    cudaDeviceSynchronize();
}

__host__ void project_warped_vertices()
{
    int block_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_project_warped, 0, 0);
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

__host__ void project_prev_frame_warped_vertices()
{
    int block_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_project_prev_frame_warped, 0, 0);
    size_t grid_size = (_host_res.active_ed_nodes_count + block_size - 1) / block_size;
    kernel_project_prev_frame_warped<<<grid_size, block_size>>>(_host_res.active_ed_nodes_count /** Not a mistake! **/, _dev_res, _host_res.measures);
    getLastCudaError("render kernel failed");
    cudaDeviceSynchronize();
}

__host__ void apply_prev_frame_warp()
{
    int block_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_prev_frame_warp, 0, 0);
    size_t grid_size = (_host_res.active_ed_vx_count + block_size - 1) / block_size;
    kernel_prev_frame_warp<<<grid_size, block_size>>>(_dev_res, _host_res);
    getLastCudaError("render kernel failed");
    cudaDeviceSynchronize();
}

extern "C" double pcg_solve(struct_native_handles &native_handles)
{
    TimerGPU timer(0);

    map_tsdf_volumes();
    map_kinect_textures();

    const unsigned int max_iterations = _host_res.configuration.solver_lma_max_iter;
    unsigned int iterations = 0u;
    float mu = _host_res.configuration.solver_mu;
    float initial_misalignment_energy, solution_misalignment_energy;

    float misaligned_vx_share = 1.f;
    int singularity = 0;

    project_vertices();

    apply_prev_frame_warp();
    project_prev_frame_warped_vertices();

#ifdef INITIAL_GUESS_OPTICFLOW

    /// Clean step h
    clean_pcg_resources();

    /// Fill step h with initial guess from optical flow
    calculate_opticflow_guess();

    /// Calculate initial energy after guess
    evaluate_step_misalignment_energy(initial_misalignment_energy);

    /// Update ED graph
    int N = (int)_host_res.active_ed_nodes_count * ED_COMPONENT_COUNT;
    float one = 1.f;
    cublasSaxpy(cublas_handle, N, &one, (float *)_dev_res.h, 1, (float *)&_dev_res.ed_graph[0], 1);
    cudaDeviceSynchronize();

    /// Apply deformations
    apply_warp();
    project_warped_vertices();

#else

    evaluate_misalignment_energy(initial_misalignment_energy);

#endif

#ifdef DEBUG_CONVERGENCE
    std::ofstream fs("data_" + std::to_string(_host_res.configuration.weight_data) + "hull_" + std::to_string(_host_res.configuration.weight_hull) + "corr_" +
                     std::to_string(_host_res.configuration.weight_correspondence) + "reg_" + std::to_string(_host_res.configuration.weight_regularization) + ".csv");
    text::csv::csv_ostream csvs(fs);

    csvs << "Iteration"
         << "Mu"
         << "E(G_0)"
         << "E(G)"
         << "Misaligned Vx. %";
    csvs << text::csv::endl;
#endif

    /// Produce JTJ (in CSR) & JTF
    clean_pcg_resources();
    cudaDeviceSynchronize();
    evaluate_step_jtj_jtf();
    cudaDeviceSynchronize();
    convert_to_csr();

    unsigned int MAX_ED_CELLS = _host_res.measures.data_volume_num_bricks * _host_res.measures.brick_num_ed_cells;
    unsigned long int JTJ_ROWS = MAX_ED_CELLS * ED_COMPONENT_COUNT;
    unsigned long int JTJ_NNZ = JTJ_ROWS * ED_COMPONENT_COUNT;

    while(iterations < max_iterations)
    {
        /// Clean [JTJ + Mu*I]
        checkCudaErrors(cudaMemset(_dev_res.jtj_mu_vals, 0, JTJ_NNZ * sizeof(float)));

        /// Produce [JTJ + Mu*I]
        add_jtj_step_mu(mu);
        cudaDeviceSynchronize();

#ifdef DEBUG_JTJ

        print_out_jtj();

#endif

#ifdef DEBUG_JTF

        print_out_jtf();

#endif

        singularity = solve_for_h();
        cudaDeviceSynchronize();

        if(singularity != -1)
        {
#ifdef VERBOSE
            printf("\nsingularity encountered\n");

            mu += _host_res.configuration.solver_mu_step;
            printf("\nmu raised: %f\n", mu);
#endif

            continue;
        }

#ifdef DEBUG_H

        print_out_h();

#endif

        evaluate_step_misalignment_energy(solution_misalignment_energy);

#ifdef VERBOSE
        printf("\ninitial E per vx: %.3f, solution E per vx: %.3f\n", initial_misalignment_energy / _host_res.active_ed_vx_count, solution_misalignment_energy / _host_res.active_ed_vx_count);
#endif

#ifdef VERBOSE
        printf("\ninitial E: %f, solution E: %f\n", initial_misalignment_energy, solution_misalignment_energy);
#endif

        if(solution_misalignment_energy < initial_misalignment_energy && (unsigned int)(solution_misalignment_energy * 1000) != 0u)
        {
#ifdef VERBOSE
            printf("\naccepted step, initial E: %f, solution E: %f\n", initial_misalignment_energy, solution_misalignment_energy);
#endif

            /// Update ED graph
            int N = (int)_host_res.active_ed_nodes_count * ED_COMPONENT_COUNT;
            float one = 1.f;
            cublasSaxpy(cublas_handle, N, &one, (float *)_dev_res.h, 1, (float *)&_dev_res.ed_graph[0], 1);
            cudaDeviceSynchronize();

            /// Produce JTJ (in CSR) & JTF
            clean_pcg_resources();
            cudaDeviceSynchronize();
            evaluate_step_jtj_jtf();
            cudaDeviceSynchronize();
            convert_to_csr();

            /// Lower Mu
            mu -= _host_res.configuration.solver_mu_step;
            initial_misalignment_energy = solution_misalignment_energy;
#ifdef VERBOSE
            printf("\nmu lowered: %f\n", mu);
#endif

            /// Assess Vx. Misalignment
            apply_warp();
            project_warped_vertices();
#ifdef REJECT_MISALIGNED
            int misaligned_vx;
            reject_misaligned_deformations(misaligned_vx);
            misaligned_vx_share = (float)misaligned_vx / (float)_host_res.active_ed_vx_count;

#ifdef VERBOSE
            printf("\nmisaligned_vx_share: %.3f\n", misaligned_vx_share * 100.f);
#endif

            if(misaligned_vx_share < 0.2f)
            {
#ifdef VERBOSE
                printf("\nmisalignment vx count criterion satisfied: %.3f\n", misaligned_vx_share * 100.f);
#endif
                break;
            }
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

        if(mu < 0.f)
        {
#ifdef VERBOSE
            printf("\nmu crossed zero: %f\n", mu);
#endif
            break;
        }

        if(mu > 14700.f)
        {
#ifdef VERBOSE
            printf("\nmu crossed 14700: %f\n", mu);
#endif
            break;
        }

#ifdef DEBUG_CONVERGENCE
        csvs << (float)iterations << mu << initial_misalignment_energy / _host_res.active_ed_vx_count << solution_misalignment_energy / _host_res.active_ed_vx_count << misaligned_vx_share * 100.f;
        csvs << text::csv::endl;
#endif

        iterations++;
    }

#ifdef DEBUG_CONVERGENCE
    fs.close();
#endif

#ifndef UNIT_TEST_ERROR_INFORMED_BLENDING_COMPLEMENTARY
    evaluate_alignment_error();
#endif

#ifdef UNIT_TEST_NRA

    float *md = nullptr;
    checkCudaErrors(cudaMalloc(&md, sizeof(float)));
    checkCudaErrors(cudaMemset(md, 0, sizeof(float)));

    calculate_alignment_statistics(md);

    float mad = 0.f;
    cudaMemcpy(&mad, &md[0], sizeof(float), cudaMemcpyDeviceToHost);
    printf("\nMean deviation from expected deformation: %.3f%\n", 100.f * mad / (float)_host_res.active_ed_nodes_count / glm::length(glm::fvec3(0.0714f, 0.f, 0.0714f)));

    checkCudaErrors(cudaFree(md));

#endif

    unmap_kinect_textures();
    unmap_tsdf_volumes();

    checkCudaErrors(cudaThreadSynchronize());
    return timer.read();
}