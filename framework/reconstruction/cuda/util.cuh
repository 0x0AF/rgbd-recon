#include <reconstruction/cuda/resources.cuh>

__device__ __host__ glm::uvec3 index_3d(unsigned int brick_id)
{
    glm::uvec3 brick = glm::uvec3(0u);
    brick.z = brick_id / (BRICK_RES_X * BRICK_RES_Y);
    brick_id %= (BRICK_RES_X * BRICK_RES_Y);
    brick.y = brick_id / BRICK_RES_X;
    brick_id %= BRICK_RES_X;
    brick.x = brick_id;

    return brick;
}

__device__ __host__ glm::uvec3 position_3d(unsigned int position_id)
{
    glm::uvec3 position = glm::uvec3(0u);
    position.z = position_id / (BRICK_VOXEL_DIM * BRICK_VOXEL_DIM);
    position_id %= (BRICK_VOXEL_DIM * BRICK_VOXEL_DIM);
    position.y = position_id / BRICK_VOXEL_DIM;
    position_id %= (BRICK_VOXEL_DIM);
    position.x = position_id;

    return position;
}

__device__ __host__ glm::uvec3 ed_cell_3d(unsigned int ed_cell_id)
{
    glm::uvec3 ed_cell_position = glm::uvec3(0u);
    ed_cell_position.z = ed_cell_id / (ED_CELL_RES * ED_CELL_RES);
    ed_cell_id %= (ED_CELL_RES * ED_CELL_RES);
    ed_cell_position.y = ed_cell_id / ED_CELL_RES;
    ed_cell_id %= (ED_CELL_RES);
    ed_cell_position.x = ed_cell_id;

    return ed_cell_position;
}

__device__ __host__ unsigned int ed_cell_id(glm::uvec3 ed_cell_3d) { return ed_cell_3d.z * ED_CELL_RES * ED_CELL_RES + ed_cell_3d.y * ED_CELL_RES + ed_cell_3d.x; }

__device__ glm::uvec3 ed_cell_voxel_3d(unsigned int ed_cell_voxel_id)
{
    glm::uvec3 ed_cell_voxel_3d = glm::uvec3(0u);
    ed_cell_voxel_3d.z = ed_cell_voxel_id / (ED_CELL_VOXEL_DIM * ED_CELL_VOXEL_DIM);
    ed_cell_voxel_id %= (ED_CELL_VOXEL_DIM * ED_CELL_VOXEL_DIM);
    ed_cell_voxel_3d.y = ed_cell_voxel_id / ED_CELL_VOXEL_DIM;
    ed_cell_voxel_id %= (ED_CELL_VOXEL_DIM);
    ed_cell_voxel_3d.x = ed_cell_voxel_id;

    return ed_cell_voxel_3d;
}

__device__ __host__ unsigned int ed_cell_voxel_id(glm::uvec3 ed_cell_voxel_3d)
{
    return ed_cell_voxel_3d.z * ED_CELL_VOXEL_DIM * ED_CELL_VOXEL_DIM + ed_cell_voxel_3d.y * ED_CELL_VOXEL_DIM + ed_cell_voxel_3d.x;
}

__device__ __host__ unsigned int identify_brick_id(const glm::vec3 position)
{
    glm::uvec3 pos_voxel_space = glm::uvec3(position);
    glm::uvec3 brick_index3d = pos_voxel_space / BRICK_VOXEL_DIM;

    return brick_index3d.z * BRICK_RES_X * BRICK_RES_Y + brick_index3d.y * BRICK_RES_X + brick_index3d.x;
}

__device__ __host__ unsigned int identify_ed_cell_id(const glm::vec3 position, unsigned int brick_id)
{
    glm::uvec3 pos_voxel_space = glm::uvec3(position);
    glm::uvec3 brick_index3d = pos_voxel_space / BRICK_VOXEL_DIM;

    // printf("\nbrick_id  %u\n", brick_id);

    glm::uvec3 relative_pos = pos_voxel_space - brick_index3d * BRICK_VOXEL_DIM;
    glm::uvec3 ed_cell_index3d = relative_pos / ED_CELL_VOXEL_DIM;

    // printf("\nrelative_pos (%u,%u,%u)\n", relative_pos.x, relative_pos.y, relative_pos.z);

    return ed_cell_id(ed_cell_index3d);
}

/*
 * Warp a position in volume voxel space with a single ED node
 * */
__device__ __host__ glm::vec3 warp_position(glm::vec3 &dist, struct_ed_node &ed_node, const float &skinning_weight)
{
    return skinning_weight * (glm::mat3(ed_node.affine) * dist + ed_node.position + ed_node.translation);
}

/*
 * Warp a normal in volume voxel space with a single ED node
 * */
__device__ __host__ glm::vec3 warp_normal(glm::vec3 &normal, struct_ed_node &ed_node, const float &skinning_weight)
{
    return skinning_weight * (glm::transpose(glm::inverse(glm::mat3(ed_node.affine))) * normal);
}

__device__ float evaluate_vx_misalignment(struct_vertex &vertex, struct_ed_node &ed_node, struct_measures *measures)
{
    glm::vec3 dist = vertex.position - ed_node.position;
    const float skinning_weight = 1.f;
    glm::vec3 warped_position = warp_position(dist, ed_node, skinning_weight);

    glm::uvec3 wp_voxel_space = glm::uvec3(warped_position);
    // printf("\n (x,y,z): (%u,%u,%u)\n", wp_voxel_space.x, wp_voxel_space.y, wp_voxel_space.z);

    if(wp_voxel_space.x >= VOLUME_VOXEL_DIM_X || wp_voxel_space.y >= VOLUME_VOXEL_DIM_Y || wp_voxel_space.z >= VOLUME_VOXEL_DIM_Z)
    {
        // printf("\nwarped out of volume: (%u,%u,%u)\n", wp_voxel_space.x, wp_voxel_space.y, wp_voxel_space.z);
        return 0.f;
    }

    float2 data{0.f, 0.f};
    surf3Dread(&data, _volume_tsdf_data, wp_voxel_space.x * sizeof(float2), wp_voxel_space.y, wp_voxel_space.z);

    return glm::abs(data.x);
}

__device__ float evaluate_vx_residual(struct_vertex &vertex, struct_ed_node &ed_node_new, struct_ed_node &ed_node, struct_measures *measures)
{
    glm::vec3 dist = vertex.position - ed_node.position;
    const float skinning_weight = 1.f; // expf(glm::length(dist) * glm::length(dist) * 2 / (ED_CELL_VOXEL_DIM * ED_CELL_VOXEL_DIM));

    glm::vec3 warped_position = warp_position(dist, ed_node, skinning_weight);
    glm::vec3 warped_normal = warp_normal(vertex.normal, ed_node, skinning_weight);

    float residual = 0.000001f;

    glm::vec3 warped_position_new = warp_position(dist, ed_node_new, skinning_weight);

    glm::uvec3 wp_voxel_space = glm::uvec3(warped_position_new);
    // printf("\n (x,y,z): (%u,%u,%u)\n", wp_voxel_space.x, wp_voxel_space.y, wp_voxel_space.z);

    if(wp_voxel_space.x >= VOLUME_VOXEL_DIM_X || wp_voxel_space.y >= VOLUME_VOXEL_DIM_Y || wp_voxel_space.z >= VOLUME_VOXEL_DIM_Z)
    {
        // printf("\nwarped out of volume: (%u,%u,%u)\n", wp_voxel_space.x, wp_voxel_space.y, wp_voxel_space.z);
        return 0.f;
    }

    for(int i = 0; i < 1; i++)
    {
        float4 data{0.f, 0.f, 0.f, 0.f};

        switch(i)
        {
        case 0:
            surf3Dread(&data, _volume_cv_xyz_inv_0, wp_voxel_space.x * sizeof(float4), wp_voxel_space.y, wp_voxel_space.z);
            break;
            // TODO
            /*case 1:
                surf3Dread(&data, _volume_cv_xyz_inv_1, wp_voxel_space.x * sizeof(float4), wp_voxel_space.y, wp_voxel_space.z);
                break;
            case 2:
                surf3Dread(&data, _volume_cv_xyz_inv_2, wp_voxel_space.x * sizeof(float4), wp_voxel_space.y, wp_voxel_space.z);
                break;
            case 3:
                surf3Dread(&data, _volume_cv_xyz_inv_3, wp_voxel_space.x * sizeof(float4), wp_voxel_space.y, wp_voxel_space.z);
                break;*/
        }

        // printf("\ncamera_positions[%u]: (%f,%f,%f)\n", i, camera_positions[i].x,camera_positions[i].y,camera_positions[i].z);

        // printf("\n (x,y): (%f,%f)\n", data.x, data.y);

        uint2 pixel{0u, 0u};
        pixel.x = (unsigned int)(data.x * measures->depth_resolution.x);
        pixel.y = (unsigned int)(data.y * measures->depth_resolution.y);

        if(pixel.x >= measures->depth_resolution.x || pixel.y >= measures->depth_resolution.y)
        {
            printf("\nprojected out of depth map: (%u,%u)\n", pixel.x, pixel.y);
            continue;
        }

        float1 depth{0.f};

        switch(i)
        {
        case 0:
            surf2DLayeredread(&depth, _array2d_kinect_depths_0, pixel.x * sizeof(float1), pixel.y, i);
            break;
            // TODO
            /*case 1:
                surf2DLayeredread(&depth, _array2d_kinect_depths_1, pixel.x * sizeof(float1), pixel.y, i);
                break;
            case 2:
                surf2DLayeredread(&depth, _array2d_kinect_depths_2, pixel.x * sizeof(float1), pixel.y, i);
                break;
            case 3:
                surf2DLayeredread(&depth, _array2d_kinect_depths_3, pixel.x * sizeof(float1), pixel.y, i);
                break;*/
        }

        // printf("\n (x,y): (%u,%u) = %f\n", pixel.x, pixel.y, depth.x);

        float normalized_depth = (depth.x - measures->depth_limits[i].x) / (measures->depth_limits[i].y - measures->depth_limits[i].x);

        // printf("\n normalized depth (x,y): (%u,%u) = %f\n", pixel.x, pixel.y, normalized_depth);

        unsigned int depth_voxel_space = (unsigned int)(normalized_depth / 0.01f);

        // printf("\n depth_voxel_space (x,y): (%u,%u) = %u\n", pixel.x, pixel.y, depth_voxel_space);

        if(depth_voxel_space >= 128)
        {
            // printf("\n depth_voxel_space out of bounds: %u\n", depth_voxel_space);
            continue;
        }

        float3 projected{0.f, 0.f, 0.f};

        switch(i)
        {
        case 0:
            surf3Dread(&projected, _volume_cv_xyz_0, pixel.x * sizeof(float3), pixel.y, depth_voxel_space);
            break;
            // TODO
            /*case 1:
                surf3Dread(&projected, _volume_cv_xyz_1, pixel.x * sizeof(float3), pixel.y, depth_voxel_space);
                break;
            case 2:
                surf3Dread(&projected, _volume_cv_xyz_2, pixel.x * sizeof(float3), pixel.y, depth_voxel_space);
                break;
            case 3:
                surf3Dread(&projected, _volume_cv_xyz_3, pixel.x * sizeof(float3), pixel.y, depth_voxel_space);
                break;*/
        }

        //        if(depth_voxel_space == 45u)
        //        {
        //            printf("\nprojected (x,y, depth): (%u,%u,%u) = (%f,%f,%f)\n", pixel.x, pixel.y, depth_voxel_space, projected.x, projected.y, projected.z);
        //        }

        glm::vec3 extracted_position = glm::vec3(projected.x, projected.y, projected.z);

        //        printf("\nextracted_position: (%f,%f,%f), wp_voxel_space: (%u,%u,%u)\n", extracted_position.x, extracted_position.y, extracted_position.z, wp_voxel_space.x, wp_voxel_space.y,
        //        wp_voxel_space.z);

        glm::vec3 diff = glm::vec3(wp_voxel_space) - extracted_position;

        if(glm::length(diff) > 3.f || glm::length(diff) == 0.f)
        {
            continue;
        }

        float residual_component = glm::abs(glm::dot(warped_normal, diff));

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

__device__ __host__ float derivative_step(const int &partial_derivative_index)
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
        step = 1.0f; // one voxel step
        break;
    case 3:
    case 4:
    case 5:
    case 6:
        step = 0.1f; // small quaternion twists
        break;
    default:
        printf("\nfatal sampling error: wrong ed component id\n");
        break;
    }

    return step;
}

__device__ float evaluate_vx_pd(struct_vertex &vertex, struct_ed_node ed_node_new, struct_ed_node ed_node, const int &partial_derivative_index, const float &vx_residual, struct_measures *measures)
{
    float ds = derivative_step(partial_derivative_index);

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

__device__ __host__ float robustify(float value)
{
    // TODO
    return value;
}

__device__ __host__ float evaluate_ed_node_residual(struct_ed_node &ed_node, struct_ed_meta_entry &ed_entry, struct_ed_node *ed_graph, struct_ed_meta_entry *ed_metas)
{
    float residual = 0.f;

    // Quad-style encoding of an affine transformation prohibits non-rotational transformations by design, hence the expression below is always 0

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

        struct_ed_node n_ed_node = ed_graph[ed_entry.neighbors[i]];

        float dist = glm::length(n_ed_node.position - ed_node.position);

        if(dist > 240.f)
        {
            // TODO: investigate distances further than volume extent
            //            printf("\ndist: %f, ed_node.position: (%g,%g,%g), n_ed_node.position: (%g,%g,%g)\n",
            //                   dist, ed_node.position.x, ed_node.position.y, ed_node.position.z, n_ed_node.position.x, n_ed_node.position.y, n_ed_node.position.z);

            return residual;
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
        return residual;
    }

    for(unsigned int i = 0; i < 27; i++)
    {
        if(ed_entry.ed_cell_id == i || ed_entry.neighbors[i] == -1)
        {
            continue;
        }

        // printf ("\nw: %f\n",expf(weights[i] / (2.0f * average_distance * average_distance)));

        float residual_component = expf(weights[i] / (2.0f * average_distance * average_distance)) * robustify(glm::length(comparative_vectors[i]));

        // printf ("\nresidual_component[%u]: %f\n", i,residual_component);

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

            residual_component = 0.f;
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

        residual = 0.f;
    }

    return residual;
}

__device__ __host__ float evaluate_ed_pd(struct_ed_node ed_node, struct_ed_meta_entry &ed_entry, struct_ed_node *ed_graph, struct_ed_meta_entry *ed_graph_meta, const int &partial_derivative_index,
                                         const float &ed_residual)
{
    float ds = derivative_step(partial_derivative_index);

    float *mapped_ed_node = (float *)&ed_node;

    mapped_ed_node[partial_derivative_index] += ds;

    float residual_pos = evaluate_ed_node_residual(ed_node, ed_entry, ed_graph, ed_graph_meta);

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

__device__ float evaluate_hull_residual(struct_vertex &vertex, struct_ed_node &ed_node, struct_measures *measures)
{
    glm::vec3 dist = vertex.position - ed_node.position;
    const float skinning_weight = 1.f; // expf(glm::length(dist) * glm::length(dist) * 2 / (ED_CELL_VOXEL_DIM * ED_CELL_VOXEL_DIM));

    glm::vec3 warped_position = warp_position(dist, ed_node, skinning_weight);

    float residual = 0.000001f;

    glm::uvec3 wp_voxel_space = glm::uvec3(warped_position);
    // printf("\n (x,y,z): (%u,%u,%u)\n", wp_voxel_space.x, wp_voxel_space.y, wp_voxel_space.z);

    if(wp_voxel_space.x >= VOLUME_VOXEL_DIM_X || wp_voxel_space.y >= VOLUME_VOXEL_DIM_Y || wp_voxel_space.z >= VOLUME_VOXEL_DIM_Z)
    {
        // printf("\nwarped out of volume: (%u,%u,%u)\n", wp_voxel_space.x, wp_voxel_space.y, wp_voxel_space.z);
        return 0.f;
    }

    for(int i = 0; i < 1; i++)
    {
        float4 data{0.f, 0.f, 0.f, 0.f};

        switch(i)
        {
        case 0:
            surf3Dread(&data, _volume_cv_xyz_inv_0, wp_voxel_space.x * sizeof(float4), wp_voxel_space.y, wp_voxel_space.z);
            break;
            // TODO
            /*case 1:
                surf3Dread(&data, _volume_cv_xyz_inv_1, wp_voxel_space.x * sizeof(float4), wp_voxel_space.y, wp_voxel_space.z);
                break;
            case 2:
                surf3Dread(&data, _volume_cv_xyz_inv_2, wp_voxel_space.x * sizeof(float4), wp_voxel_space.y, wp_voxel_space.z);
                break;
            case 3:
                surf3Dread(&data, _volume_cv_xyz_inv_3, wp_voxel_space.x * sizeof(float4), wp_voxel_space.y, wp_voxel_space.z);
                break;*/
        }

        // printf("\ncamera_positions[%u]: (%f,%f,%f)\n", i, camera_positions[i].x,camera_positions[i].y,camera_positions[i].z);

        // printf("\n (x,y): (%f,%f)\n", data.x, data.y);

        uint2 pixel{0u, 0u};
        pixel.x = (unsigned int)(data.x * measures->depth_resolution.x);
        pixel.y = (unsigned int)(data.y * measures->depth_resolution.y);

        if(pixel.x >= measures->depth_resolution.x || pixel.y >= measures->depth_resolution.y)
        {
            printf("\nprojected out of depth map: (%u,%u)\n", pixel.x, pixel.y);
            continue;
        }

        float1 occupancy{0.f};

        switch(i)
        {
        case 0:
            surf2DLayeredread(&occupancy, _array2d_silhouettes_0, pixel.x * sizeof(float1), pixel.y, i);
            break;
            // TODO
            /*case 1:
                surf2DLayeredread(&occupancy, _array2d_kinect_depths_1, pixel.x * sizeof(float1), pixel.y, i);
                break;
            case 2:
                surf2DLayeredread(&occupancy, _array2d_kinect_depths_2, pixel.x * sizeof(float1), pixel.y, i);
                break;
            case 3:
                surf2DLayeredread(&occupancy, _array2d_kinect_depths_3, pixel.x * sizeof(float1), pixel.y, i);
                break;*/
        }

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

__device__ float evaluate_hull_pd(struct_vertex &vertex, struct_ed_node ed_node, const int &partial_derivative_index, const float &vx_residual, struct_measures *measures)
{
    float ds = derivative_step(partial_derivative_index);

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

extern "C" glm::uvec3 test_index_3d(unsigned int brick_id) { return index_3d(brick_id); }
extern "C" glm::uvec3 test_position_3d(unsigned int position_id) { return position_3d(position_id); }
extern "C" glm::uvec3 test_ed_cell_3d(unsigned int ed_cell_id) { return ed_cell_3d(ed_cell_id); }
extern "C" unsigned int test_ed_cell_id(glm::uvec3 ed_cell_3d) { return ed_cell_id(ed_cell_3d); };
extern "C" unsigned int test_ed_cell_voxel_id(glm::uvec3 ed_cell_voxel_3d) { return ed_cell_voxel_id(ed_cell_voxel_3d); }
extern "C" glm::vec3 test_warp_position(glm::vec3 &dist, struct_ed_node &ed_node, const float &skinning_weight) { return warp_position(dist, ed_node, skinning_weight); }
extern "C" glm::vec3 test_warp_normal(glm::vec3 &normal, struct_ed_node &ed_node, const float &skinning_weight) { return warp_normal(normal, ed_node, skinning_weight); }
extern "C" float test_evaluate_ed_node_residual(struct_ed_node &ed_node, struct_ed_meta_entry &ed_entry, struct_ed_node *ed_neighborhood, struct_ed_meta_entry *ed_neighborhood_meta)
{
    return evaluate_ed_node_residual(ed_node, ed_entry, ed_neighborhood, ed_neighborhood_meta);
}
