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
    cudaError err = cudaThreadSynchronize();
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

// CUDA_HOST_DEVICE glm::vec3 bbox_original_position(glm::vec3 &pos, struct_measures &measures)
//{
//    pos = pos + measures.bbox_translation;
//    pos = pos * measures.bbox_dimensions;
//    return pos;
//}
//
// CUDA_HOST_DEVICE glm::vec3 bbox_original_vector(glm::vec3 &vec, struct_measures &measures)
//{
//    vec = vec * measures.bbox_dimensions;
//    return vec;
//}

CUDA_HOST_DEVICE glm::vec3 bbox_transform_position(glm::vec3 &pos, struct_measures &measures)
{
    pos = pos * measures.bbox_dimensions;
    //pos = pos - measures.bbox_translation;
    return pos;
}

CUDA_HOST_DEVICE glm::vec3 bbox_transform_vector(glm::vec3 &vec, struct_measures &measures)
{
    vec = vec * measures.bbox_dimensions;
    return vec;
}

CUDA_HOST_DEVICE bool in_data_volume(glm::uvec3 &pos, struct_measures &measures)
{
    return pos.x < measures.data_volume_res.x && pos.y < measures.data_volume_res.y && pos.z < measures.data_volume_res.z;
}

CUDA_HOST_DEVICE bool in_cv_xyz(glm::uvec3 &pos, struct_measures &measures) { return pos.x < measures.cv_xyz_res.x && pos.y < measures.cv_xyz_res.y && pos.z < measures.cv_xyz_res.z; }

CUDA_HOST_DEVICE bool in_cv_xyz_inv(glm::uvec3 &pos, struct_measures &measures) { return pos.x < measures.cv_xyz_inv_res.x && pos.y < measures.cv_xyz_inv_res.y && pos.z < measures.cv_xyz_inv_res.z; }

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
    position_id %= (measures.brick_dim_voxels * measures.brick_dim_voxels);
    position.y = position_id / measures.brick_dim_voxels;
    position_id %= (measures.brick_dim_voxels);
    position.x = position_id;

    return position;
}

CUDA_HOST_DEVICE glm::uvec3 ed_cell_3d(unsigned int ed_cell_id, struct_measures &measures)
{
    glm::uvec3 ed_cell_position = glm::uvec3(0u);
    ed_cell_position.z = ed_cell_id / (measures.brick_dim_ed_cells * measures.brick_dim_ed_cells);
    ed_cell_id %= (measures.brick_dim_ed_cells * measures.brick_dim_ed_cells);
    ed_cell_position.y = ed_cell_id / measures.brick_dim_ed_cells;
    ed_cell_id %= (measures.brick_dim_ed_cells);
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
    ed_cell_voxel_id %= (measures.ed_cell_dim_voxels * measures.ed_cell_dim_voxels);
    ed_cell_voxel_3d.y = ed_cell_voxel_id / measures.ed_cell_dim_voxels;
    ed_cell_voxel_id %= (measures.ed_cell_dim_voxels);
    ed_cell_voxel_3d.x = ed_cell_voxel_id;

    return ed_cell_voxel_3d;
}

CUDA_HOST_DEVICE unsigned int ed_cell_voxel_id(glm::uvec3 ed_cell_voxel_3d, struct_measures &measures)
{
    return ed_cell_voxel_3d.z * measures.ed_cell_dim_voxels * measures.ed_cell_dim_voxels + ed_cell_voxel_3d.y * measures.ed_cell_dim_voxels + ed_cell_voxel_3d.x;
}

CUDA_HOST_DEVICE unsigned int identify_brick_id(const glm::vec3 position, struct_measures &measures)
{
    glm::uvec3 pos_voxel_space = glm::uvec3(position / measures.size_voxel);

    if(!in_data_volume(pos_voxel_space, measures))
    {
        printf("\nsampled position out of volume: (%u,%u,%u)\n", pos_voxel_space.x, pos_voxel_space.y, pos_voxel_space.z);
    }

    glm::uvec3 brick_index3d = pos_voxel_space / measures.brick_dim_voxels;

    return brick_index3d.z * measures.data_volume_bricked_res.x * measures.data_volume_bricked_res.y + brick_index3d.y * measures.data_volume_bricked_res.x + brick_index3d.x;
}

CUDA_HOST_DEVICE unsigned int identify_ed_cell_id(const glm::vec3 position, struct_measures &measures)
{
    glm::uvec3 pos_voxel_space = glm::uvec3(position / measures.size_voxel);

    if(!in_data_volume(pos_voxel_space, measures))
    {
        printf("\nsampled position out of volume: (%u,%u,%u)\n", pos_voxel_space.x, pos_voxel_space.y, pos_voxel_space.z);
    }

    glm::uvec3 ed_cell_index3d = (pos_voxel_space % measures.brick_dim_voxels) / measures.ed_cell_dim_voxels;
    return ed_cell_id(ed_cell_index3d, measures);
}

/** Warp a position with a single ED node **/
CUDA_HOST_DEVICE glm::vec3 warp_position(glm::vec3 &dist, struct_ed_node &ed_node, const float &skinning_weight, struct_measures &measures)
{
    return skinning_weight * (glm::mat3(ed_node.affine) * dist + ed_node.position + ed_node.translation);
}

/** Warp a normal with a single ED node **/
CUDA_HOST_DEVICE glm::vec3 warp_normal(glm::vec3 &normal, struct_ed_node &ed_node, const float &skinning_weight, struct_measures &measures)
{
    return skinning_weight * (glm::transpose(glm::inverse(glm::mat3(ed_node.affine))) * normal);
}

__device__ float evaluate_vx_misalignment(struct_vertex &vertex, struct_ed_node &ed_node, struct_measures &measures)
{
    glm::vec3 dist = vertex.position - ed_node.position;
    const float skinning_weight = 1.f; // TODO
    glm::vec3 warped_position = warp_position(dist, ed_node, skinning_weight, measures);

    glm::uvec3 wp_voxel_space = glm::uvec3(warped_position / measures.size_voxel);
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

__device__ float evaluate_vx_residual(struct_vertex &vertex, struct_ed_node &ed_node_new, struct_ed_node &ed_node, struct_measures &measures)
{
    glm::vec3 dist = vertex.position - ed_node_new.position;
    const float skinning_weight = 1.f; // TODO expf(glm::length(dist) * glm::length(dist) * 2 / (ED_CELL_VOXEL_DIM * ED_CELL_VOXEL_DIM));

    glm::vec3 warped_position = warp_position(dist, ed_node_new, skinning_weight, measures);
    glm::vec3 warped_normal = warp_normal(vertex.normal, ed_node, skinning_weight, measures);

    float residual = 0.f;

    glm::uvec3 wp_cv_inv_space = glm::uvec3(warped_position * (float)measures.cv_xyz_inv_res.x);

    if(!in_cv_xyz_inv(wp_cv_inv_space, measures))
    {
        // printf("\nwarped out of volume: (%u,%u,%u)\n", wp_voxel_space.x, wp_voxel_space.y, wp_voxel_space.z);
        return 0.f;
    }

    for(int i = 0; i < 4; i++)
    {
        float4 data{0.f, 0.f, 0.f, 0.f};

        switch(i)
        {
        case 0:
            surf3Dread(&data, _volume_cv_xyz_inv_0, wp_cv_inv_space.x * sizeof(float4), wp_cv_inv_space.y, wp_cv_inv_space.z);
            break;
        case 1:
            surf3Dread(&data, _volume_cv_xyz_inv_1, wp_cv_inv_space.x * sizeof(float4), wp_cv_inv_space.y, wp_cv_inv_space.z);
            break;
        case 2:
            surf3Dread(&data, _volume_cv_xyz_inv_2, wp_cv_inv_space.x * sizeof(float4), wp_cv_inv_space.y, wp_cv_inv_space.z);
            break;
        case 3:
            surf3Dread(&data, _volume_cv_xyz_inv_3, wp_cv_inv_space.x * sizeof(float4), wp_cv_inv_space.y, wp_cv_inv_space.z);
            break;
        }

        // printf("\n (x,y): (%f,%f)\n", data.x, data.y);

        uint2 pixel{0u, 0u};
        pixel.x = (unsigned int)(data.x * measures.depth_res.x);
        pixel.y = (unsigned int)(data.y * measures.depth_res.y);

        if(pixel.x >= measures.depth_res.x || pixel.y >= measures.depth_res.y)
        {
            printf("\nprojected out of depth map: (%u,%u)\n", pixel.x, pixel.y);
            continue;
        }

        float1 depth{0.f};
        surf3Dread(&depth, _kinect_depths, pixel.x * sizeof(float1), pixel.y, i);

        // printf("\n (x,y): (%u,%u) = %f\n", pixel.x, pixel.y, depth.x);

        float normalized_depth = (depth.x - measures.depth_limits[i].x) / (measures.depth_limits[i].y - measures.depth_limits[i].x);

        // printf("\n normalized depth (x,y): (%u,%u) = %f\n", pixel.x, pixel.y, normalized_depth);

        unsigned int depth_voxel_space = (unsigned int)(normalized_depth * measures.cv_xyz_res.z);

        // printf("\n depth_voxel_space (x,y) [i]: (%u,%u) [%u] = %u\n", pixel.x, pixel.y, i, depth_voxel_space);

        uint2 scaled_pixel{0u, 0u};
        scaled_pixel.x = (unsigned int)(data.x * measures.cv_xyz_res.x);
        scaled_pixel.y = (unsigned int)(data.y * measures.cv_xyz_res.y);

        glm::uvec3 coordinate = glm::uvec3(scaled_pixel.x, scaled_pixel.y, depth_voxel_space);

        if(!in_cv_xyz(coordinate, measures))
        {
            continue;
        }

        float4 projected{0.f, 0.f, 0.f, 0.f};

        switch(i)
        {
        case 0:
            surf3Dread(&projected, _volume_cv_xyz_0, coordinate.x * sizeof(float4), coordinate.y, coordinate.z);
            break;
        case 1:
            surf3Dread(&projected, _volume_cv_xyz_1, coordinate.x * sizeof(float4), coordinate.y, coordinate.z);
            break;
        case 2:
            surf3Dread(&projected, _volume_cv_xyz_2, coordinate.x * sizeof(float4), coordinate.y, coordinate.z);
            break;
        case 3:
            surf3Dread(&projected, _volume_cv_xyz_3, coordinate.x * sizeof(float4), coordinate.y, coordinate.z);
            break;
        }

        // printf("\nprojected (%f,%f,%f, %f)\n", projected.x, projected.y, projected.z, projected.w);

        //        if(depth_voxel_space == 45u)
        //        {
        //            printf("\nprojected (x,y, depth): (%u,%u,%u) = (%f,%f,%f)\n", pixel.x, pixel.y, depth_voxel_space, projected.x, projected.y, projected.z);
        //        }

        glm::vec3 extracted_position = glm::vec3(projected.x, projected.y, projected.z);
        extracted_position = bbox_transform_position(extracted_position, measures);

        glm::uvec3 epvs = extracted_position / measures.size_voxel;

        if(!in_data_volume(epvs, measures))
        {
            continue;
        }

        //        printf("\nextracted_position (%u, %u, %u): (%u,%u,%u) = (%f,%f,%f)\n", pixel.x, pixel.y, depth_voxel_space, wp_voxel_space.x, wp_voxel_space.y, wp_voxel_space.z,
        //        extracted_position.x, extracted_position.y, extracted_position.z);

        glm::vec3 diff = warped_position - extracted_position;

        // printf("\ndiff: %f\n", glm::length(diff));

        if(glm::length(diff) > 0.1f || glm::length(diff) == 0.f)
        {
            continue;
        }

        float residual_component = glm::abs(glm::dot(warped_normal, diff));

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

CUDA_HOST_DEVICE float derivative_step(const int &partial_derivative_index, struct_measures &measures)
{
    float step = 0.f;
    switch(partial_derivative_index)
    {
    case 0:
    case 1:
    case 2:
    case 7:
    case 8:
    case 9:
        step = measures.size_voxel; // one voxel step
        break;
    case 3:
    case 4:
    case 5:
    case 6:
        step = 0.01f; // small quaternion twists
        break;
    default:
        printf("\nfatal sampling error: wrong ed component id\n");
        break;
    }

    return step;
}

__device__ float evaluate_vx_pd(struct_vertex &vertex, struct_ed_node ed_node_new, struct_ed_node ed_node, const int &partial_derivative_index, const float &vx_residual, struct_measures &measures)
{
    float ds = derivative_step(partial_derivative_index, measures);

    float *mapped_ed_node = (float *)&ed_node_new;

    mapped_ed_node[partial_derivative_index] += ds;

    float residual_pos = evaluate_vx_residual(vertex, ed_node_new, ed_node, measures);

    // printf("\nresidual_pos: %f\n", residual_pos);

    if(isnan(residual_pos))
    {
#ifdef DEBUG_NANS
        printf("\nresidual_pos is NaN!\n");
#endif

        residual_pos = 0.f;
    }

    float partial_derivative = residual_pos / (2.0f * ds) - vx_residual / (2.0f * ds);

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

CUDA_HOST_DEVICE float robustify(float value)
{
    // TODO
    return value;
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

        float dist = glm::length(n_ed_node.position - ed_node.position);

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

        comparative_vectors[i] = glm::toMat3(n_ed_node.affine) * (n_ed_node.position - ed_node.position) + n_ed_node.position + n_ed_node.translation - ed_node.position - ed_node.translation;
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

CUDA_HOST_DEVICE float evaluate_ed_pd(struct_ed_node ed_node, struct_ed_meta_entry &ed_entry, const int &partial_derivative_index, const float &ed_residual, struct_device_resources &dev_res,
                                      struct_measures &measures)
{
    float ds = derivative_step(partial_derivative_index, measures);

    float *mapped_ed_node = (float *)&ed_node;

    mapped_ed_node[partial_derivative_index] += ds;

    float residual_pos = evaluate_ed_node_residual(ed_node, ed_entry, dev_res, measures);

    // printf("\nresidual_pos: %f\n", residual_pos);

    if(isnan(residual_pos))
    {
#ifdef DEBUG_NANS
        printf("\nresidual_pos is NaN!\n");
#endif

        residual_pos = 0.f;
    }

    float partial_derivative = residual_pos / (2.0f * ds) - ed_residual / (2.0f * ds);

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

__device__ float evaluate_hull_residual(struct_vertex &vertex, struct_ed_node &ed_node, struct_measures &measures)
{
    glm::vec3 dist = vertex.position - ed_node.position;
    const float skinning_weight = 1.f; // expf(glm::length(dist) * glm::length(dist) * 2 / (ED_CELL_VOXEL_DIM * ED_CELL_VOXEL_DIM));

    glm::vec3 warped_position = warp_position(dist, ed_node, skinning_weight, measures);

    float residual = 0.f;

    glm::uvec3 wp_cv_inv_space = glm::uvec3(warped_position * 200.f / 140.f);

    if(!in_cv_xyz_inv(wp_cv_inv_space, measures))
    {
        return 1.f;
    }

    for(int i = 0; i < 4; i++)
    {
        float4 data{0.f, 0.f, 0.f, 0.f};

        switch(i)
        {
        case 0:
            surf3Dread(&data, _volume_cv_xyz_inv_0, wp_cv_inv_space.x * sizeof(float4), wp_cv_inv_space.y, wp_cv_inv_space.z);
            break;
        case 1:
            surf3Dread(&data, _volume_cv_xyz_inv_1, wp_cv_inv_space.x * sizeof(float4), wp_cv_inv_space.y, wp_cv_inv_space.z);
            break;
        case 2:
            surf3Dread(&data, _volume_cv_xyz_inv_2, wp_cv_inv_space.x * sizeof(float4), wp_cv_inv_space.y, wp_cv_inv_space.z);
            break;
        case 3:
            surf3Dread(&data, _volume_cv_xyz_inv_3, wp_cv_inv_space.x * sizeof(float4), wp_cv_inv_space.y, wp_cv_inv_space.z);
            break;
        }

        // printf("\n (x,y): (%f,%f)\n", data.x, data.y);

        uint2 pixel{0u, 0u};
        pixel.x = (unsigned int)(data.x * measures.depth_res.x);
        pixel.y = (unsigned int)(data.y * measures.depth_res.y);

        if(pixel.x >= measures.depth_res.x || pixel.y >= measures.depth_res.y)
        {
            printf("\nprojected out of depth map: (%u,%u)\n", pixel.x, pixel.y);
            continue;
        }

        float1 occupancy{0.f};
        surf3Dread(&occupancy, _kinect_silhouettes, pixel.x * sizeof(float1), pixel.y, i);

        // printf("\n (x,y): (%u,%u) = %f\n", pixel.x, pixel.y, occupancy.x);

        float residual_component = 1.0f - occupancy.x;

        if(isnan(residual_component))
        {
            // printf("\nresidual_component is NaN!\n");

            residual_component = 0.f;
        }

        residual += residual_component;
    }

    return residual;
}

__device__ float evaluate_hull_pd(struct_vertex &vertex, struct_ed_node ed_node, const int &partial_derivative_index, const float &vx_residual, struct_measures &measures)
{
    float ds = derivative_step(partial_derivative_index, measures);

    float *mapped_ed_node = (float *)&ed_node;

    mapped_ed_node[partial_derivative_index] += ds;

    float residual_pos = evaluate_hull_residual(vertex, ed_node, measures);

    // printf("\nresidual_pos: %f\n", residual_pos);

    if(isnan(residual_pos))
    {
#ifdef DEBUG_NANS
        printf("\nresidual_pos is NaN!\n");
#endif

        residual_pos = 0.f;
    }

    float partial_derivative = residual_pos / (2.0f * ds) - vx_residual / (2.0f * ds);

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

extern "C" glm::uvec3 test_index_3d(unsigned int brick_id, struct_measures &measures) { return index_3d(brick_id, measures); }
extern "C" glm::uvec3 test_position_3d(unsigned int position_id, struct_measures &measures) { return position_3d(position_id, measures); }
extern "C" glm::uvec3 test_ed_cell_3d(unsigned int ed_cell_id, struct_measures &measures) { return ed_cell_3d(ed_cell_id, measures); }
extern "C" unsigned int test_ed_cell_id(glm::uvec3 ed_cell_3d, struct_measures &measures) { return ed_cell_id(ed_cell_3d, measures); };
extern "C" unsigned int test_ed_cell_voxel_id(glm::uvec3 ed_cell_voxel_3d, struct_measures &measures) { return ed_cell_voxel_id(ed_cell_voxel_3d, measures); }
extern "C" glm::vec3 test_warp_position(glm::vec3 &dist, struct_ed_node &ed_node, const float &skinning_weight, struct_measures &measures)
{
    return warp_position(dist, ed_node, skinning_weight, measures);
}
extern "C" glm::vec3 test_warp_normal(glm::vec3 &normal, struct_ed_node &ed_node, const float &skinning_weight, struct_measures &measures)
{
    return warp_normal(normal, ed_node, skinning_weight, measures);
}
extern "C" unsigned int test_identify_brick_id(const glm::vec3 position, struct_measures &measures) { return identify_brick_id(position, measures); }
extern "C" unsigned int test_identify_ed_cell_id(const glm::vec3 position, struct_measures &measures) { return identify_ed_cell_id(position, measures); }
// extern "C" float test_evaluate_ed_node_residual(struct_ed_node &ed_node, struct_ed_meta_entry &ed_entry, struct_measures &measures)
//{
//    return evaluate_ed_node_residual(ed_node, ed_entry, measures);
//}