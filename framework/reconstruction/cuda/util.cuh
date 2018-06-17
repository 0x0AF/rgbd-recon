#include <reconstruction/cuda/resources.cuh>

#define TSDF_LOOKUP

__device__ glm::uvec3 index_3d(unsigned int brick_id)
{
    glm::uvec3 brick = glm::uvec3(0u);
    brick.z = brick_id / (BRICK_RES_X * BRICK_RES_Y);
    brick_id %= (BRICK_RES_X * BRICK_RES_Y);
    brick.y = brick_id / BRICK_RES_X;
    brick_id %= BRICK_RES_X;
    brick.x = brick_id;

    return brick;
}

__device__ glm::uvec3 position_3d(unsigned int position_id)
{
    glm::uvec3 position = glm::uvec3(0u);
    position.z = position_id / (BRICK_VOXEL_DIM * BRICK_VOXEL_DIM);
    position_id %= (BRICK_VOXEL_DIM * BRICK_VOXEL_DIM);
    position.y = position_id / BRICK_VOXEL_DIM;
    position_id %= (BRICK_VOXEL_DIM);
    position.x = position_id;

    return position;
}

__device__ glm::uvec3 ed_cell_3d(unsigned int ed_cell_id)
{
    glm::uvec3 ed_cell_position = glm::uvec3(0u);
    ed_cell_position.z = ed_cell_id / (ED_CELL_RES * ED_CELL_RES);
    ed_cell_id %= (ED_CELL_RES * ED_CELL_RES);
    ed_cell_position.y = ed_cell_id / ED_CELL_RES;
    ed_cell_id %= (ED_CELL_RES);
    ed_cell_position.x = ed_cell_id;

    return ed_cell_position;
}

__device__ unsigned int ed_cell_id(glm::uvec3 ed_cell_3d) { return ed_cell_3d.z * ED_CELL_RES * ED_CELL_RES + ed_cell_3d.y * ED_CELL_RES + ed_cell_3d.x; }

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

__device__ unsigned int ed_cell_voxel_id(glm::uvec3 ed_cell_voxel_3d)
{
    return ed_cell_voxel_3d.z * ED_CELL_VOXEL_DIM * ED_CELL_VOXEL_DIM + ed_cell_voxel_3d.y * ED_CELL_VOXEL_DIM + ed_cell_voxel_3d.x;
}

/*
 * Identify enclosing ED cell in volume voxel space
 * */
__device__ unsigned int identify_ed_cell_pos(const glm::vec3 position, const unsigned int *bricks_inv_index)
{
    glm::uvec3 pos_voxel_space = glm::uvec3(position);
    glm::uvec3 brick_index3d = pos_voxel_space / BRICK_VOXEL_DIM;

    unsigned int brick_id = brick_index3d.z * BRICK_RES_X * BRICK_RES_Y + brick_index3d.y * BRICK_RES_X + brick_index3d.x;

    // printf("\nbrick_id  %u\n", brick_id);

    glm::uvec3 relative_pos = pos_voxel_space - brick_index3d * BRICK_VOXEL_DIM;
    glm::uvec3 ed_cell_index3d = relative_pos / ED_CELL_VOXEL_DIM;

    // printf("\nrelative_pos (%u,%u,%u)\n", relative_pos.x, relative_pos.y, relative_pos.z);

    unsigned int ed_cell = ed_cell_index3d.z * ED_CELL_RES * ED_CELL_RES + ed_cell_index3d.y * ED_CELL_RES + ed_cell_index3d.x;

    // printf("\ned_cell %u\n", ed_cell);

    unsigned int brick_pos_inv_index = bricks_inv_index[brick_id];

    // printf("\nbrick_id, brick_pos_inv_index [%u,%u]\n", brick_id, brick_pos_inv_index);

    unsigned int ed_cell_pos = brick_pos_inv_index * ED_CELL_RES * ED_CELL_RES * ED_CELL_RES + ed_cell;

    // printf("\ned_cell_pos %u\n", ed_cell_pos);

    return ed_cell_pos;
}

/*
 * Warp a position in volume voxel space with a single ED node
 * */
__device__ glm::vec3 warp_position(glm::vec3 &dist, struct_ed_node &ed_node, const float &skinning_weight)
{
    return skinning_weight * (glm::mat3(ed_node.affine) * dist + ed_node.position + ed_node.translation);
}

/*
 * Warp a normal in volume voxel space with a single ED node
 * */
__device__ glm::vec3 warp_normal(glm::vec3 &normal, struct_ed_node &ed_node, const float &skinning_weight)
{
    return skinning_weight * (glm::transpose(glm::inverse(glm::mat3(ed_node.affine))) * normal);
}

__device__ float evaluate_vx_residual(struct_vertex &vertex, struct_ed_node &ed_node)
{
    glm::vec3 dist = vertex.position - ed_node.position;
    const float skinning_weight = 1.f; // expf(glm::length(dist) * glm::length(dist) * 2 / (ED_CELL_VOXEL_DIM * ED_CELL_VOXEL_DIM));

    glm::vec3 warped_position = warp_position(dist, ed_node, skinning_weight);

    float residual = 0.f;

    glm::uvec3 wp_voxel_space = glm::uvec3(warped_position);
    // printf("\n (x,y,z): (%u,%u,%u)\n", wp_voxel_space.x, wp_voxel_space.y, wp_voxel_space.z);

    if(wp_voxel_space.x >= VOLUME_VOXEL_DIM_X || wp_voxel_space.y >= VOLUME_VOXEL_DIM_Y || wp_voxel_space.z >= VOLUME_VOXEL_DIM_Z)
    {
        // TODO: warped out of volume!
        return 0.f;
    }

#ifdef TSDF_LOOKUP
    float2 data;
    surf3Dread(&data, _volume_tsdf_data, wp_voxel_space.x * sizeof(float2), wp_voxel_space.y, wp_voxel_space.z);

    if(data.x > 0. && data.x < 0.03f)
    {
        residual += data.x;
    }
#endif

#ifdef CV_LOOKUP
    glm::vec3 warped_normal = warp_normal(vertex.normal, ed_node, skinning_weight);

    for(int i = 0; i < 4; i++)
    {
        float4 data;

        switch(i)
        {
        case 0:
            surf3Dread(&data, _volume_cv_xyz_inv_0, wp_voxel_space.x * sizeof(float4), wp_voxel_space.y, wp_voxel_space.z);
            break;
        case 1:
            surf3Dread(&data, _volume_cv_xyz_inv_1, wp_voxel_space.x * sizeof(float4), wp_voxel_space.y, wp_voxel_space.z);
            break;
        case 2:
            surf3Dread(&data, _volume_cv_xyz_inv_2, wp_voxel_space.x * sizeof(float4), wp_voxel_space.y, wp_voxel_space.z);
            break;
        case 3:
            surf3Dread(&data, _volume_cv_xyz_inv_3, wp_voxel_space.x * sizeof(float4), wp_voxel_space.y, wp_voxel_space.z);
            break;
        }

        // printf("\n (x,y): (%f,%f)\n", data.x, data.y);

        if(data.y > 1.f || data.x > 1.f || data.y < 0.f || data.x < 0.f)
        {
            // TODO: projects out of depth map!
            continue;
        }

        // uint2 pixel;
        // pixel.x = (unsigned int)(data.x * 512);
        // pixel.y = (unsigned int)(data.y * 424);

        // TODO: lookup kinect depths

        // float depth;
        // surf2DLayeredread(&depth, _array2d_kinect_depths, pixel.x * sizeof(float), pixel.y, i);

        // printf("\n (x,y): (%u,%u) = %f\n", pixel.x, pixel.y, /*depth*/data.z);

        glm::vec3 extracted_position = glm::vec3(wp_voxel_space) + glm::vec3(1.f - data.z) * (float)VOLUME_VOXEL_DIM;
        // extracted_position *= (1 + 0.1 * fracf(sinf(warped_position.x)));

        residual += glm::abs(glm::dot(warped_normal, glm::vec3(wp_voxel_space) - extracted_position));
    }
#endif

    return residual;
}

__device__ float evaluate_vx_pd(struct_vertex &vertex, struct_ed_node ed_node, const int &partial_derivative_index, const float &vx_residual)
{
    float *mapped_ed_node = (float *)&ed_node;

    mapped_ed_node[partial_derivative_index] += 0.0001f;

    float residual_pos = evaluate_vx_residual(vertex, ed_node);

    mapped_ed_node[partial_derivative_index] -= 0.0001f;

    // printf("\nresidual_pos: %f\n", residual_pos);

    if(isnan(residual_pos))
    {
        printf("\nresidual_pos is NaN!\n");

        residual_pos = 0.f;
    }

    float partial_derivative = residual_pos / 0.0002f - vx_residual / 0.0002f;

    // printf("\npartial_derivative: %f\n", partial_derivative);

    if(isnan(partial_derivative))
    {
        printf("\npartial_derivative is NaN!\n");

        partial_derivative = 0.f;
    }

    return partial_derivative;
}

__device__ float evaluate_ed_node_residual(struct_ed_node &ed_node)
{
    float residual = 0.f;

    glm::mat3 mat_1 = (glm::transpose(glm::toMat3(ed_node.affine)) * glm::toMat3(ed_node.affine) - glm::mat3());

    for(int i = 0; i < 3; i++)
    {
        for(int k = 0; k < 3; k++)
        {
            residual += mat_1[i][k] * mat_1[i][k];
        }
    }

    residual = (float)sqrt(residual);
    residual += glm::determinant(glm::toMat3(ed_node.affine)) - 1;

    // TODO: figure out smooth component
    // residuals[1] = 0.f;

    return residual;
}