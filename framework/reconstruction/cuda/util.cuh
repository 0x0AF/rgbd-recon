#include <reconstruction/cuda/resources.cuh>

#define safeCall(err) __safeCall(err, __FILE__, __LINE__)
#define safeThreadSync() __safeThreadSync(__FILE__, __LINE__)
#define checkMsg(msg) __checkMsg(msg, __FILE__, __LINE__)

inline void __safeCall(cudaError err, const char *file, const int line)
{
    if(cudaSuccess != err)
    {
        fprintf(stderr, "safeCall() Runtime API error in file <%s>, line %i : %s.\n", file, line, cudaGetErrorString(err));
        exit(-1);
    }
}

inline void __safeThreadSync(const char *file, const int line)
{
    cudaError err = cudaDeviceSynchronize();
    if(cudaSuccess != err)
    {
        fprintf(stderr, "threadSynchronize() Driver API error in file '%s' in line %i : %s.\n", file, line, cudaGetErrorString(err));
        exit(-1);
    }
}

inline void __checkMsg(const char *errorMessage, const char *file, const int line)
{
    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf(stderr, "checkMsg() CUDA error: %s in file <%s>, line %i : %s.\n", errorMessage, file, line, cudaGetErrorString(err));
        exit(-1);
    }
}

__device__ float4 sample_colors_ptr(float4 *colors_ptr, glm::uvec2 &pos, int layer, struct_measures measures)
{
    return colors_ptr[pos.x + pos.y * measures.depth_res.x + layer * measures.depth_res.x * measures.depth_res.y];
}
/*__device__ float2 sample_depths_ptr(float2 *depths_ptr, glm::uvec2 &pos, int layer, struct_measures measures)
{
    return depths_ptr[pos.x + pos.y * measures.depth_res.x + layer * measures.depth_res.x * measures.depth_res.y];
}
__device__ float sample_silhouettes_ptr(float *silhouettes_ptr, glm::uvec2 &pos, int layer, struct_measures measures)
{
    return silhouettes_ptr[pos.x + pos.y * measures.depth_res.x + layer * measures.depth_res.x * measures.depth_res.y];
}*/

__device__ float2 sample_ref_warped_ptr(float2 *ref_warped_ptr, glm::uvec3 &pos, struct_measures measures)
{
    return ref_warped_ptr[pos.x + pos.y * measures.data_volume_res.x + pos.z * measures.data_volume_res.x * measures.data_volume_res.y];
}

__device__ float1 sample_silhouette(cudaTextureObject_t &silhouette_tex, glm::vec2 &pos) { return tex2D<float1>(silhouette_tex, pos.x, pos.y); }
__device__ float2 sample_depth(cudaTextureObject_t &depth_tex, glm::vec2 &pos) { return tex2D<float2>(depth_tex, pos.x, pos.y); }
__device__ float4 sample_cv_xyz(cudaTextureObject_t &cv_xyz_tex, glm::vec3 &pos) { return tex3D<float4>(cv_xyz_tex, pos.x, pos.y, pos.z); }
__device__ float4 sample_cv_xyz_inv(cudaTextureObject_t &cv_xyz_inv_tex, glm::vec3 &pos) { return tex3D<float4>(cv_xyz_inv_tex, pos.x, pos.y, pos.z); }

CUDA_HOST_DEVICE glm::vec3 bbox_transform_position(glm::vec3 pos, struct_measures &measures)
{
    pos -= measures.bbox_translation;
    pos /= measures.bbox_dimensions;
    return pos;
}

CUDA_HOST_DEVICE glm::vec3 bbox_transform_vector(glm::vec3 vec, struct_measures &measures)
{
    vec /= measures.bbox_dimensions;
    return vec;
}

CUDA_HOST_DEVICE bool in_normal_space(glm::vec3 pos) { return pos.x < 1.f && pos.y < 1.f && pos.z < 1.f && pos.x >= 0.f && pos.y >= 0.f && pos.z >= 0.f; }

CUDA_HOST_DEVICE bool in_data_volume(glm::uvec3 pos, struct_measures &measures)
{
    return pos.x < measures.data_volume_res.x && pos.y < measures.data_volume_res.y && pos.z < measures.data_volume_res.z;
}

CUDA_HOST_DEVICE bool in_cv_xyz(glm::uvec3 pos, struct_measures &measures) { return pos.x < measures.cv_xyz_res.x && pos.y < measures.cv_xyz_res.y && pos.z < measures.cv_xyz_res.z; }

CUDA_HOST_DEVICE bool in_cv_xyz_inv(glm::uvec3 pos, struct_measures &measures) { return pos.x < measures.cv_xyz_inv_res.x && pos.y < measures.cv_xyz_inv_res.y && pos.z < measures.cv_xyz_inv_res.z; }

CUDA_HOST_DEVICE glm::uvec3 norm_2_cv_xyz(glm::vec3 pos, struct_measures &measures) { return glm::uvec3(pos * glm::vec3(measures.cv_xyz_res)); }
CUDA_HOST_DEVICE glm::uvec3 norm_2_cv_xyz_inv(glm::vec3 pos, struct_measures &measures) { return glm::uvec3(pos * glm::vec3(measures.cv_xyz_inv_res)); }
CUDA_HOST_DEVICE glm::uvec3 norm_2_data(glm::vec3 pos, struct_measures &measures) { return glm::uvec3(pos * glm::vec3(measures.data_volume_res)); }
CUDA_HOST_DEVICE glm::vec3 data_2_norm(glm::uvec3 pos, struct_measures &measures) { return glm::vec3(pos) / glm::vec3(measures.data_volume_res); }

CUDA_HOST_DEVICE uchar tsdf_2_8bit(float tsdf) { return (uchar)(255.f * (tsdf / 0.06f + 0.5f)); }

CUDA_HOST_DEVICE glm::uvec3 index_3d(unsigned int brick_id, struct_measures &measures)
{
    glm::uvec3 brick = glm::uvec3(0u);
    brick.z = brick_id / (measures.data_volume_bricked_res.x * measures.data_volume_bricked_res.y);
    brick_id %= (measures.data_volume_bricked_res.x * measures.data_volume_bricked_res.y);
    brick.y = brick_id / measures.data_volume_bricked_res.x;
    brick_id %= measures.data_volume_bricked_res.x;
    brick.x = brick_id;

    return brick;
}

CUDA_HOST_DEVICE glm::uvec3 position_3d(unsigned int position_id, struct_measures &measures)
{
    glm::uvec3 position = glm::uvec3(0u);
    position.z = position_id / (measures.brick_dim_voxels * measures.brick_dim_voxels);
    position_id %= measures.brick_dim_voxels * measures.brick_dim_voxels;
    position.y = position_id / measures.brick_dim_voxels;
    position_id %= measures.brick_dim_voxels;
    position.x = position_id;

    return position;
}

CUDA_HOST_DEVICE unsigned int position_id(glm::uvec3 position_3d, struct_measures &measures)
{
    return position_3d.z * measures.brick_dim_voxels * measures.brick_dim_voxels + position_3d.y * measures.brick_dim_voxels + position_3d.x;
}

CUDA_HOST_DEVICE glm::uvec3 ed_cell_3d(unsigned int ed_cell_id, struct_measures &measures)
{
    glm::uvec3 ed_cell_position = glm::uvec3(0u);
    ed_cell_position.z = ed_cell_id / (measures.brick_dim_ed_cells * measures.brick_dim_ed_cells);
    ed_cell_id %= measures.brick_dim_ed_cells * measures.brick_dim_ed_cells;
    ed_cell_position.y = ed_cell_id / measures.brick_dim_ed_cells;
    ed_cell_id %= measures.brick_dim_ed_cells;
    ed_cell_position.x = ed_cell_id;

    return ed_cell_position;
}

CUDA_HOST_DEVICE unsigned int ed_cell_id(glm::uvec3 ed_cell_3d, struct_measures &measures)
{
    return ed_cell_3d.z * measures.brick_dim_ed_cells * measures.brick_dim_ed_cells + ed_cell_3d.y * measures.brick_dim_ed_cells + ed_cell_3d.x;
}

CUDA_HOST_DEVICE glm::uvec3 ed_cell_voxel_3d(unsigned int ed_cell_voxel_id, struct_measures &measures)
{
    glm::uvec3 ed_cell_voxel_3d = glm::uvec3(0u);
    ed_cell_voxel_3d.z = ed_cell_voxel_id / (measures.ed_cell_dim_voxels * measures.ed_cell_dim_voxels);
    ed_cell_voxel_id %= measures.ed_cell_dim_voxels * measures.ed_cell_dim_voxels;
    ed_cell_voxel_3d.y = ed_cell_voxel_id / measures.ed_cell_dim_voxels;
    ed_cell_voxel_id %= measures.ed_cell_dim_voxels;
    ed_cell_voxel_3d.x = ed_cell_voxel_id;

    return ed_cell_voxel_3d;
}

CUDA_HOST_DEVICE unsigned int ed_cell_voxel_id(glm::uvec3 ed_cell_voxel_3d, struct_measures &measures)
{
    return ed_cell_voxel_3d.z * measures.ed_cell_dim_voxels * measures.ed_cell_dim_voxels + ed_cell_voxel_3d.y * measures.ed_cell_dim_voxels + ed_cell_voxel_3d.x;
}

CUDA_HOST_DEVICE unsigned int identify_brick_id(const glm::vec3 position, struct_measures &measures)
{
#ifdef VERBOSE
    if(!in_normal_space(position))
    {
        printf("\nsampled position out of volume: (%f,%f,%f)\n", position.x, position.y, position.z);
    }
#endif

    glm::uvec3 pos_voxel_space = norm_2_data(position, measures);
    glm::uvec3 brick_index3d = pos_voxel_space / measures.brick_dim_voxels;

    return brick_index3d.z * measures.data_volume_bricked_res.x * measures.data_volume_bricked_res.y + brick_index3d.y * measures.data_volume_bricked_res.x + brick_index3d.x;
}

CUDA_HOST_DEVICE unsigned int identify_ed_cell_id(const glm::vec3 position, struct_measures &measures)
{
#ifdef VERBOSE
    if(!in_normal_space(position))
    {
        printf("\nsampled position out of volume: (%f,%f,%f)\n", position.x, position.y, position.z);
    }
#endif

    glm::uvec3 pos_voxel_space = norm_2_data(position, measures);
    glm::uvec3 ed_cell_index3d = (pos_voxel_space % measures.brick_dim_voxels) / measures.ed_cell_dim_voxels;

    return ed_cell_id(ed_cell_index3d, measures);
}

CUDA_HOST_DEVICE unsigned int identify_depth_cell_id(const glm::uvec2 pos, unsigned int layer, struct_measures &measures)
{
    glm::uvec2 depth_cell_index2d = pos / measures.size_depth_cell;

    return layer * depth_cell_index2d.x * depth_cell_index2d.y + depth_cell_index2d.y * measures.depth_cell_res.x + depth_cell_index2d.x;
}

CUDA_HOST_DEVICE glm::vec3 qtransform(glm::quat &q, glm::vec3 &v) { return v + 2.f * glm::cross(glm::cross(v, glm::vec3(q.x, q.y, q.z)) + q.w * v, glm::vec3(q.x, q.y, q.z)); }

/** Warp a position with a single ED node **/
CUDA_HOST_DEVICE glm::vec3 warp_position(glm::vec3 &dist, struct_ed_node &ed_node, struct_ed_meta_entry &ed_meta, const float &skinning_weight, struct_measures &measures)
{
#ifdef FAST_QUAT_OPS
    return skinning_weight * (qtransform(ed_node.rotation, dist) + ed_meta.position + ed_node.translation);
#else
    return skinning_weight * (glm::mat3(ed_node.affine) * dist + ed_node.position + ed_node.translation);
#endif
}

/** Warp a normal with a single ED node **/
CUDA_HOST_DEVICE glm::vec3 warp_normal(glm::vec3 &normal, struct_ed_node &ed_node, struct_ed_meta_entry &ed_meta, const float &skinning_weight, struct_measures &measures)
{
#ifdef FAST_QUAT_OPS
    return skinning_weight * (qtransform(ed_node.rotation, normal));
#else
    return glm::normalize(skinning_weight * (glm::transpose(glm::inverse(glm::mat3(ed_node.affine))) * normal));
#endif
}

__device__ float evaluate_vx_misalignment(struct_vertex &warped_vertex, struct_measures &measures)
{
    glm::uvec3 wp_voxel_space = norm_2_data(warped_vertex.position, measures);
    // printf("\n (x,y,z): (%u,%u,%u)\n", wp_voxel_space.x, wp_voxel_space.y, wp_voxel_space.z);

    if(!in_data_volume(wp_voxel_space, measures))
    {
        // printf("\nwarped out of volume: (%u,%u,%u)\n", wp_voxel_space.x, wp_voxel_space.y, wp_voxel_space.z);
        return 1.f;
    }

    float2 data{0.f, 0.f};
    surf3Dread(&data, _volume_tsdf_data, wp_voxel_space.x * sizeof(float2), wp_voxel_space.y, wp_voxel_space.z);

    return glm::abs(data.x);
}

__device__ float evaluate_data_residual(struct_vertex &warped_vertex, struct_projection &warped_projection, struct_device_resources &dev_res, struct_measures &measures)
{
    float residual = 0.f;

    //    if(!in_normal_space(warped_vertex.position))
    //    {
    //        // printf("\nwarped out of volume: (%.2f,%.2f,%.2f)]\n", warped_vertex.position.x, warped_vertex.position.y, warped_vertex.position.z);
    //        return 1.f;
    //    }

    // printf("\nwarped position: (%f,%f,%f)\n", warped_position.x, warped_position.y, warped_position.z);
    // printf("\nwarped normal: (%.2f,%.2f,%.2f) [%.2f]\n", warped_normal.x, warped_normal.y, warped_normal.z, glm::length(warped_normal));

    for(int i = 0; i < 4; i++)
    {
        // printf("\nsampling depth maps: (%u,%u)\n", pixel.x, pixel.y);

        if(warped_projection.projection[i].x >= 1.0f || warped_projection.projection[i].y >= 1.0f)
        {
            // printf("\nprojected out of depth map: (%u,%u)\n", pixel.x, pixel.y);
            continue;
        }

        float depth = sample_depth(dev_res.depth_tex[i], warped_projection.projection[i]).x;

        if(depth == 0)
        {
            continue;
        }

        // printf("\ndepth %f\n", depth);

        glm::vec3 coordinate = glm::vec3(warped_projection.projection[i].x, warped_projection.projection[i].y, depth);

        if(!in_normal_space(coordinate))
        {
            // printf("\nprojected out of direct calibration volume: (%f,%f,%f)\n", coordinate.x, coordinate.y, coordinate.z);
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
        extracted_position = bbox_transform_position(extracted_position, measures);

        // printf("\nextracted_position (%.2f, %.2f, %.2f)\n", extracted_position.x, extracted_position.y, extracted_position.z);

        if(!in_normal_space(extracted_position))
        {
            continue;
        }

        //        printf("\nextracted_position (%.2f, %.2f, %.2f): (%.2f,%.2f,%.2f) = (%.2f,%.2f,%.2f)\n", coordinate.x, coordinate.y, coordinate.z, warped_position.x, warped_position.y,
        //        warped_position.z, extracted_position.x, extracted_position.y, extracted_position.z);

        glm::vec3 diff = warped_vertex.position - extracted_position;

        // printf("\ndiff: %f\n", glm::length(diff));

        if(glm::length(diff) > 0.03f || glm::length(diff) == 0.f)
        {
            continue;
        }

        float residual_component = glm::abs(glm::dot(warped_vertex.normal, diff));

        // printf("\nresidual_component: %f, warped_normal: (%f,%f,%f), diff: (%f,%f,%f)\n", residual_component, warped_normal.x, warped_normal.y, warped_normal.z, diff.x, diff.y, diff.z);

        if(isnan(residual_component))
        {
#ifdef DEBUG_NANS
            printf("\nresidual_component is NaN!\n");
#endif

            residual_component = 0.f;
        }

        residual += residual_component;
    }

    return residual;
}

CUDA_HOST_DEVICE float derivative_step(const int partial_derivative_index, struct_measures &measures)
{
    float step = 0.f;
    switch(partial_derivative_index)
    {
    case 0:
    case 1:
    case 2:
        step = measures.size_voxel * 0.5f; // half-voxel step
        break;
    case 3:
    case 4:
    case 5:
    case 6:
        step = 0.125f * glm::pi<float>(); // small quaternion twists
        break;
    default:
        printf("\nfatal sampling error: wrong ed component id\n");
        break;
    }

    return step;
}

CUDA_HOST_DEVICE void shift_component(struct_ed_node &ed_node, const int partial_derivative_index, float step)
{
    switch(partial_derivative_index)
    {
    case 0:
        ed_node.translation.x += step;
        break;
    case 1:
        ed_node.translation.y += step;
        break;
    case 2:
        ed_node.translation.z += step;
        break;
    case 3:
        ed_node.rotation.x += step;
        break;
    case 4:
        ed_node.rotation.y += step;
        break;
    case 5:
        ed_node.rotation.z += step;
        break;
    case 6:
        ed_node.rotation.w += step;
        break;
    default:
        printf("\nfatal sampling error: wrong ed component id\n");
        break;
    }
}

CUDA_HOST_DEVICE float robustify(float value)
{
#ifdef ED_NODES_ROBUSTIFY
    return 0.9375f * glm::pow(1 - glm::pow(value, 2), 2);
#else
    return value;
#endif
}

CUDA_HOST_DEVICE float evaluate_ed_node_residual(struct_ed_node &ed_node, struct_ed_meta_entry &ed_entry, struct_device_resources &dev_res, struct_measures &measures)
{
    float residual = 0.f;

    /** Quad-style encoding of an affine transformation prohibits non-rotational transformations by design, hence the expression below is always 0 **/

    /*glm::mat3 mat_1 = glm::transpose(glm::toMat3(ed_node.affine)) * glm::toMat3(ed_node.affine) - glm::mat3(1.0f);

    for(int i = 0; i < 3; i++)
    {
        for(int k = 0; k < 3; k++)
        {
            residual += mat_1[i][k] * mat_1[i][k];
        }
    }

    residual = (float)sqrt(residual);
    residual += glm::determinant(glm::toMat3(ed_node.affine)) - 1;*/

    unsigned int valid_neghbors = 0u;
    float average_distance = -1.f;
    float weights[27];
    glm::vec3 comparative_vectors[27];

    for(unsigned int i = 0; i < 27; i++)
    {
        if(ed_entry.ed_cell_id == i || ed_entry.neighbors[i] == -1)
        {
            continue;
        }

        struct_ed_node n_ed_node = dev_res.ed_graph[ed_entry.neighbors[i]];
        struct_ed_meta_entry n_ed_entry = dev_res.ed_graph_meta[ed_entry.neighbors[i]];

        float dist = glm::length(n_ed_entry.position - ed_entry.position);

        if(dist > measures.size_voxel * measures.data_volume_res.x)
        {
            // TODO: investigate distances further than volume extent
            //            printf("\ndist: %f, ed_node.position: (%g,%g,%g), n_ed_node.position: (%g,%g,%g)\n",
            //                   dist, ed_node.position.x, ed_node.position.y, ed_node.position.z, n_ed_node.position.x, n_ed_node.position.y, n_ed_node.position.z);

            return 1.f;
        }

        valid_neghbors++;
        weights[i] = -dist * dist;

        if(average_distance < 0.f)
        {
            average_distance = dist;
        }
        else
        {
            average_distance = average_distance + (dist - average_distance) / (float)(valid_neghbors);
        };

        comparative_vectors[i] = glm::toMat3(n_ed_node.rotation) * (n_ed_entry.position - ed_entry.position) + n_ed_entry.position + n_ed_node.translation - ed_entry.position - ed_node.translation;
    }

    if(valid_neghbors == 0u)
    {
        return 0.f;
    }

    for(unsigned int i = 0; i < 27; i++)
    {
        if(ed_entry.ed_cell_id == i || ed_entry.neighbors[i] == -1)
        {
            continue;
        }

        // printf ("\nw: %f\n",expf(weights[i] / (2.0f * average_distance * average_distance)));

        float residual_component = expf(weights[i] / (2.0f * average_distance * average_distance)) * robustify(glm::length(comparative_vectors[i]));

        // printf ("\nresidual_component[%u]: %f\n", i, residual_component);

        if(isnan(residual_component))
        {
#ifdef DEBUG_NANS
            printf("\nresidual_component[%u] is NaN! ad: %f, w: %f, lcv: %f\n", i, average_distance, weights[i], robustify(glm::length(comparative_vectors[i])));
#endif

            residual_component = 0.f;
        }

        if(isinf(residual_component))
        {
#ifdef DEBUG_NANS
            printf("\nresidual_component[%u] is Inf! ad: %f, w: %f, lcv: %f\n", i, average_distance, weights[i], robustify(glm::length(comparative_vectors[i])));
#endif

            residual_component = 1.f;
        }

        residual += residual_component;
    }

    if(isnan(residual))
    {
#ifdef DEBUG_NANS
        printf("\nresidual is NaN!\n");
#endif

        residual = 0.f;
    }

    if(isinf(residual))
    {
#ifdef DEBUG_NANS
        printf("\nresidual is Inf!\n");
#endif

        residual = 1.f;
    }

    return residual;
}

CUDA_HOST_DEVICE float evaluate_ed_pd(struct_ed_node ed_node_new, struct_ed_meta_entry &ed_entry, const int &partial_derivative_index, struct_device_resources &dev_res, struct_measures &measures)
{
    float ds = derivative_step(partial_derivative_index, measures);

    float *mapped_ed_node = (float *)&ed_node_new;

    mapped_ed_node[partial_derivative_index] += ds;
    float residual_pos = evaluate_ed_node_residual(ed_node_new, ed_entry, dev_res, measures);

    mapped_ed_node[partial_derivative_index] -= ds;
    float residual_neg = evaluate_ed_node_residual(ed_node_new, ed_entry, dev_res, measures);

    // printf("\nresidual_pos: %f\n", residual_pos);

    if(isnan(residual_pos))
    {
#ifdef DEBUG_NANS
        printf("\nresidual_pos is NaN!\n");
#endif

        residual_pos = 0.f;
    }

    if(isnan(residual_neg))
    {
#ifdef DEBUG_NANS
        printf("\nresidual_pos is NaN!\n");
#endif

        residual_pos = 0.f;
    }

    float partial_derivative = residual_pos / (2.0f * ds) - residual_neg / (2.0f * ds);

    // printf("\npartial_derivative: %f\n", partial_derivative);

    if(isnan(partial_derivative))
    {
#ifdef DEBUG_NANS
        printf("\npartial_derivative is NaN!\n");
#endif

        partial_derivative = 0.f;
    }

    return partial_derivative;
}

__device__ float evaluate_hull_residual(struct_projection &warped_projection, struct_device_resources &dev_res, struct_measures &measures)
{
    float residual = 0.f;

    for(int i = 0; i < 4; i++)
    {
        if(warped_projection.projection[i].x >= 1.0f || warped_projection.projection[i].y >= 1.0f)
        {
#ifdef VERBOSE
            printf("\nprojected out of depth map: (%u,%u)\n", warped_projection.projection[i].x, warped_projection.projection[i].y);
#endif
            continue;
        }

        float occupancy = sample_silhouette(dev_res.silhouette_tex[i], warped_projection.projection[i]).x;

        // printf("\n (x,y): (%u,%u) = %f\n", pixel.x, pixel.y, occupancy);

        float residual_component = measures.data_volume_res.x * measures.size_voxel * (1.0f - occupancy);

        if(isnan(residual_component))
        {
#ifdef DEBUG_NANS
            printf("\nresidual_component is NaN!\n");
#endif

            residual_component = 0.f;
        }

        residual += residual_component;
    }

    //    if(residual == 4.0f)
    //    {
    //        printf("\n residual: %f\n, v(%.2f,%.2f,%.2f)", residual, vertex.position.x, vertex.position.y, vertex.position.z);
    //    }

    return residual;
}

__device__ float evaluate_cf_residual(struct_vertex &warped_vertex, struct_projection &warped_projection, struct_device_resources &dev_res, struct_measures &measures)
{
    float residual = 0.f;

    for(int layer = 0; layer < 4; layer++)
    {
        glm::uvec2 pixel = glm::uvec2(warped_projection.projection[layer]);
        unsigned int cell_id = identify_depth_cell_id(pixel, layer, measures);

        if(cell_id > measures.num_depth_cells)
        {
            printf("\ncell_id out of depth_cells range: %u\n", cell_id);
            continue;
        }

        struct_depth_cell_meta meta = dev_res.depth_cell_meta[cell_id];

        if(meta.cp_length == 0)
        {
            continue;
        }

        unsigned int argmin = 0u;

        float distance = 16.f;

        /*#pragma unroll
                for(unsigned int i = 0; i < meta.cp_length; i++)
                {
                    float delta = glm::length(dev_res.sorted_correspondences[meta.cp_offset + i].previous_proj - warped_projection.projection[layer]);
                    if(delta < distance)
                    {
                        argmin = i;
                        distance = delta;
                    }
                }*/

#pragma unroll
        for(unsigned int i = 0; i < meta.cp_length; i++)
        {
            float delta = glm::length(dev_res.sorted_correspondences[meta.cp_offset + i].previous - warped_vertex.position);
            if(delta < distance)
            {
                argmin = i;
                distance = delta;
            }
        }

        // printf ("\nargmin: %u\n", argmin);

        glm::vec3 cp = dev_res.sorted_correspondences[meta.cp_offset + argmin].current;
        float motion = glm::length(warped_vertex.position - cp);

        if(motion > 0.01f)
        {
            continue;
        }

        float residual_component = robustify(motion);

        // printf("\nresidual_component: %f, cp: (%f,%f,%f), vx: (%f,%f,%f)\n", residual_component, cp.x, cp.y, cp.z, warped_vertex.position.x, warped_vertex.position.y, warped_vertex.position.z);

        if(isnan(residual_component))
        {
#ifdef DEBUG_NANS
            printf("\nresidual_component is NaN!\n");
#endif

            residual_component = 0.f;
        }

        residual += residual_component;
    }

    return residual;
}

extern "C" glm::uvec3 test_index_3d(unsigned int brick_id, struct_measures &measures) { return index_3d(brick_id, measures); }
extern "C" glm::uvec3 test_position_3d(unsigned int position_id, struct_measures &measures) { return position_3d(position_id, measures); }
extern "C" glm::uvec3 test_ed_cell_3d(unsigned int ed_cell_id, struct_measures &measures) { return ed_cell_3d(ed_cell_id, measures); }
extern "C" unsigned int test_ed_cell_id(glm::uvec3 ed_cell_3d, struct_measures &measures) { return ed_cell_id(ed_cell_3d, measures); };
extern "C" unsigned int test_ed_cell_voxel_id(glm::uvec3 ed_cell_voxel_3d, struct_measures &measures) { return ed_cell_voxel_id(ed_cell_voxel_3d, measures); }
/*extern "C" glm::vec3 test_warp_position(glm::vec3 &dist, struct_ed_node &ed_node, const float &skinning_weight, struct_measures &measures)
{
    return warp_position(dist, ed_node, skinning_weight, measures);
}*/
/*extern "C" glm::vec3 test_warp_normal(glm::vec3 &normal, struct_ed_node &ed_node, const float &skinning_weight, struct_measures &measures)
{
    return warp_normal(normal, ed_node, skinning_weight, measures);
}*/
extern "C" unsigned int test_identify_brick_id(const glm::vec3 position, struct_measures &measures) { return identify_brick_id(position, measures); }
extern "C" unsigned int test_identify_ed_cell_id(const glm::vec3 position, struct_measures &measures) { return identify_ed_cell_id(position, measures); }
// extern "C" float test_evaluate_ed_node_residual(struct_ed_node &ed_node, struct_ed_meta_entry &ed_entry, struct_measures &measures)
//{
//    return evaluate_ed_node_residual(ed_node, ed_entry, measures);
//}