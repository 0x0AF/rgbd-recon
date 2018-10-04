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

/*__device__ float4 sample_colors_ptr(float4 *colors_ptr, glm::uvec2 &pos, int layer, struct_measures measures)
{
    return colors_ptr[pos.x + pos.y * measures.depth_res.x + layer * measures.depth_res.x * measures.depth_res.y];
}*/
/*__device__ float2 sample_depths_ptr(float2 *depths_ptr, glm::uvec2 &pos, int layer, struct_measures measures)
{
    return depths_ptr[pos.x + pos.y * measures.depth_res.x + layer * measures.depth_res.x * measures.depth_res.y];
}
__device__ float sample_silhouettes_ptr(float *silhouettes_ptr, glm::uvec2 &pos, int layer, struct_measures measures)
{
    return silhouettes_ptr[pos.x + pos.y * measures.depth_res.x + layer * measures.depth_res.x * measures.depth_res.y];
}*/

template <class data_type>
CUDA_HOST_DEVICE data_type sample_pitched_ptr(data_type *pitched_ptr, size_t pitch, unsigned int x, unsigned int y)
{
    return pitched_ptr[x + y * pitch / sizeof(data_type)];
}

template <class data_type>
CUDA_HOST_DEVICE void write_pitched_ptr(data_type value, data_type *pitched_ptr, size_t pitch, unsigned int x, unsigned int y)
{
    pitched_ptr[x + y * pitch / sizeof(data_type)] = value;
}

__device__ float2 sample_ref_warped_ptr(float2 *ref_warped_ptr, glm::uvec3 &pos, struct_measures measures)
{
    return ref_warped_ptr[pos.x + pos.y * measures.data_volume_res.x + pos.z * measures.data_volume_res.x * measures.data_volume_res.y];
}

__device__ float1 sample_ref_warped_marks_ptr(float1 *ref_warped_marks_ptr, glm::uvec3 &pos, struct_measures measures)
{
    return ref_warped_marks_ptr[pos.x + pos.y * measures.data_volume_res.x + pos.z * measures.data_volume_res.x * measures.data_volume_res.y];
}

__device__ float2 sample_opticflow(cudaTextureObject_t &opticflow_tex, glm::vec2 &pos) { return tex2D<float2>(opticflow_tex, pos.x, pos.y); }
__device__ float1 sample_silhouette(cudaTextureObject_t &silhouette_tex, glm::vec2 &pos) { return tex2D<float1>(silhouette_tex, pos.x, pos.y); }
__device__ float1 sample_depth(cudaTextureObject_t &depth_tex, glm::vec2 &pos) { return tex2D<float1>(depth_tex, pos.x, pos.y); }
__device__ float1 sample_prev_depth(cudaTextureObject_t &depth_tex_prev, glm::vec2 &pos) { return tex2D<float1>(depth_tex_prev, pos.x, pos.y); }
__device__ float4 sample_normals(cudaTextureObject_t &normals_tex, glm::vec2 &pos) { return tex2D<float4>(normals_tex, pos.x, pos.y); }
__device__ float1 sample_error(cudaTextureObject_t &alignment_error_tex, glm::vec2 &pos) { return tex2D<float1>(alignment_error_tex, pos.x, pos.y); }
__device__ float4 sample_cv_xyz(cudaTextureObject_t &cv_xyz_tex, glm::fvec3 &pos) { return tex3D<float4>(cv_xyz_tex, pos.x, pos.y, pos.z); }
__device__ float4 sample_cv_xyz_inv(cudaTextureObject_t &cv_xyz_inv_tex, glm::fvec3 &pos) { return tex3D<float4>(cv_xyz_inv_tex, pos.x, pos.y, pos.z); }

CUDA_HOST_DEVICE glm::fvec3 bbox_transform_position(glm::fvec3 pos, struct_measures &measures)
{
    pos -= measures.bbox_translation;
    pos /= measures.bbox_dimensions;
    return pos;
}

CUDA_HOST_DEVICE glm::fvec3 bbox_transform_vector(glm::fvec3 vec, struct_measures &measures)
{
    vec /= measures.bbox_dimensions;
    return vec;
}

CUDA_HOST_DEVICE bool in_normal_space(glm::fvec3 pos) { return pos.x < 1.f && pos.y < 1.f && pos.z < 1.f && pos.x >= 0.f && pos.y >= 0.f && pos.z >= 0.f; }

CUDA_HOST_DEVICE bool in_data_volume(glm::uvec3 pos, struct_measures &measures)
{
    return pos.x < measures.data_volume_res.x && pos.y < measures.data_volume_res.y && pos.z < measures.data_volume_res.z;
}

CUDA_HOST_DEVICE bool in_cv_xyz(glm::uvec3 pos, struct_measures &measures) { return pos.x < measures.cv_xyz_res.x && pos.y < measures.cv_xyz_res.y && pos.z < measures.cv_xyz_res.z; }

CUDA_HOST_DEVICE bool in_cv_xyz_inv(glm::uvec3 pos, struct_measures &measures) { return pos.x < measures.cv_xyz_inv_res.x && pos.y < measures.cv_xyz_inv_res.y && pos.z < measures.cv_xyz_inv_res.z; }

CUDA_HOST_DEVICE glm::uvec3 norm_2_cv_xyz(glm::fvec3 pos, struct_measures &measures) { return glm::uvec3(pos * glm::fvec3(measures.cv_xyz_res)); }
CUDA_HOST_DEVICE glm::uvec3 norm_2_cv_xyz_inv(glm::fvec3 pos, struct_measures &measures) { return glm::uvec3(pos * glm::fvec3(measures.cv_xyz_inv_res)); }
CUDA_HOST_DEVICE glm::uvec3 norm_2_data(glm::fvec3 pos, struct_measures &measures) { return glm::uvec3(pos * glm::fvec3(measures.data_volume_res)); }
CUDA_HOST_DEVICE glm::fvec3 data_2_norm(glm::uvec3 pos, struct_measures &measures) { return glm::fvec3(pos) / glm::fvec3(measures.data_volume_res); }

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

CUDA_HOST_DEVICE unsigned int identify_brick_id(const glm::fvec3 position, struct_measures &measures)
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

CUDA_HOST_DEVICE unsigned int identify_ed_cell_id(const glm::fvec3 position, struct_measures &measures)
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

CUDA_HOST_DEVICE glm::fvec3 qtransform(glm::fquat q, glm::fvec3 &v) { return v + 2.f * glm::cross(glm::cross(v, glm::fvec3(q.x, q.y, q.z)) + q.w * v, glm::fvec3(q.x, q.y, q.z)); }

/** Warp a position with a single ED node **/
CUDA_HOST_DEVICE glm::fvec3 warp_position(glm::fvec3 &dist, struct_ed_node &ed_node, struct_ed_meta_entry &ed_meta, const float &skinning_weight, struct_measures &measures)
{
#ifdef FAST_QUAT_OPS
    return skinning_weight * (qtransform(glm::fquat(1., ed_node.rotation[0], ed_node.rotation[1], ed_node.rotation[2]), dist) + ed_meta.position +
                              glm::fvec3(ed_node.translation[0], ed_node.translation[1], ed_node.translation[2]));
#else
    return skinning_weight * (glm::mat3_cast(glm::fquat(1., ed_node.rotation[0], ed_node.rotation[1], ed_node.rotation[2])) * dist + ed_meta.position +
                              glm::fvec3(ed_node.translation[0], ed_node.translation[1], ed_node.translation[2]));
#endif
}

/** Warp a normal with a single ED node **/
CUDA_HOST_DEVICE glm::fvec3 warp_normal(glm::fvec3 &normal, struct_ed_node &ed_node, struct_ed_meta_entry &ed_meta, const float &skinning_weight, struct_measures &measures)
{
#ifdef FAST_QUAT_OPS
    return glm::normalize(skinning_weight * (qtransform(glm::fquat(1., ed_node.rotation[0], ed_node.rotation[1], ed_node.rotation[2]), normal)));
#else
    return glm::normalize(skinning_weight * (glm::transpose(glm::inverse(glm::mat3_cast(glm::fquat(1., ed_node.rotation[0], ed_node.rotation[1], ed_node.rotation[2])))) * normal));
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

__device__ bool sample_prev_depth_projection(glm::fvec3 &depth_projection, int layer, glm::vec2 projection, struct_device_resources dev_res, struct_measures measures)
{
    float depth = sample_prev_depth(dev_res.depth_tex_prev[layer], projection).x;

    // printf("\ndepth (%f,%f) = %f\n", warped_projection.projection[layer].x, warped_projection.projection[layer].y, depth);

    if(depth == 0)
    {
        return false;
    }

    // printf("\ndepth %f\n", depth);

    glm::fvec3 coordinate = glm::fvec3(projection.x, projection.y, depth);

    if(!in_normal_space(coordinate))
    {
        /*#ifdef VERBOSE
                    printf("\nprojected out of direct calibration volume: (%f,%f,%f)\n", coordinate.x, coordinate.y, coordinate.z);
        #endif*/
        return false;
    }

    // printf("\nsampling direct calibration volume: (%f,%f,%f)\n", coordinate.x, coordinate.y, coordinate.z);

    float4 projected = sample_cv_xyz(dev_res.cv_xyz_tex[layer], coordinate);

    // printf("\nprojected (%f,%f,%f,%f)\n", projected.x, projected.y, projected.z, projected.w);

    //        if(depth_voxel_space == 45u)
    //        {
    //            printf("\nprojected (x,y, depth): (%u,%u,%u) = (%f,%f,%f)\n", pixel.x, pixel.y, depth_voxel_space, projected.x, projected.y, projected.z);
    //        }

    glm::fvec3 extracted_position = glm::fvec3(projected.x, projected.y, projected.z);
    extracted_position = bbox_transform_position(extracted_position, measures);

    // printf("\nextracted_position (%.2f, %.2f, %.2f)\n", extracted_position.x, extracted_position.y, extracted_position.z);

    if(!in_normal_space(extracted_position))
    {
        return false;
    }

    //        printf("\nextracted_position (%.2f, %.2f, %.2f): (%.2f,%.2f,%.2f) = (%.2f,%.2f,%.2f)\n", coordinate.x, coordinate.y, coordinate.z, warped_position.x, warped_position.y,
    //        warped_position.z, extracted_position.x, extracted_position.y, extracted_position.z);

    depth_projection = extracted_position;

    return true;
}

__device__ bool sample_depth_projection(glm::fvec3 &depth_projection, int layer, glm::vec2 projection, struct_device_resources dev_res, struct_measures measures)
{
    float depth = sample_depth(dev_res.depth_tex[layer], projection).x;

    // printf("\ndepth (%f,%f) = %f\n", warped_projection.projection[layer].x, warped_projection.projection[layer].y, depth);

    if(depth == 0)
    {
        return false;
    }

    // printf("\ndepth %f\n", depth);

    glm::fvec3 coordinate = glm::fvec3(projection.x, projection.y, depth);

    if(!in_normal_space(coordinate))
    {
        /*#ifdef VERBOSE
                    printf("\nprojected out of direct calibration volume: (%f,%f,%f)\n", coordinate.x, coordinate.y, coordinate.z);
        #endif*/
        return false;
    }

    // printf("\nsampling direct calibration volume: (%f,%f,%f)\n", coordinate.x, coordinate.y, coordinate.z);

    float4 projected = sample_cv_xyz(dev_res.cv_xyz_tex[layer], coordinate);

    // printf("\nprojected (%f,%f,%f,%f)\n", projected.x, projected.y, projected.z, projected.w);

    //        if(depth_voxel_space == 45u)
    //        {
    //            printf("\nprojected (x,y, depth): (%u,%u,%u) = (%f,%f,%f)\n", pixel.x, pixel.y, depth_voxel_space, projected.x, projected.y, projected.z);
    //        }

    glm::fvec3 extracted_position = glm::fvec3(projected.x, projected.y, projected.z);
    extracted_position = bbox_transform_position(extracted_position, measures);

    // printf("\nextracted_position (%.2f, %.2f, %.2f)\n", extracted_position.x, extracted_position.y, extracted_position.z);

    if(!in_normal_space(extracted_position))
    {
        return false;
    }

    //        printf("\nextracted_position (%.2f, %.2f, %.2f): (%.2f,%.2f,%.2f) = (%.2f,%.2f,%.2f)\n", coordinate.x, coordinate.y, coordinate.z, warped_position.x, warped_position.y,
    //        warped_position.z, extracted_position.x, extracted_position.y, extracted_position.z);

    depth_projection = extracted_position;

    return true;
}

__device__ float evaluate_data_residual(struct_vertex &warped_vertex, struct_projection &projection, struct_device_resources &dev_res, struct_measures &measures)
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
        // printf("\nsampling depth maps: (%.2f,%.2f)\n", warped_projection.projection[i].x, warped_projection.projection[i].y);

        if(projection.projection[i].x < 0.f || projection.projection[i].y < 0.f)
        {
#ifdef VERBOSE
            printf("\nprojected out of depth map: (%f,%f)\n", projection.projection[i].x, projection.projection[i].y);
#endif
            continue;
        }

        float4 kinect_normal = sample_normals(dev_res.normals_tex[i], projection.projection[i]);

        if(kinect_normal.x == 0.f)
        {
            continue;
        }

        glm::fvec3 kinect_normal_normalized = glm::normalize(glm::fvec3(kinect_normal.x, kinect_normal.y, kinect_normal.z));

        /*printf("\nnormal: (%.2f,%.2f,%.2f), kinect normal: (%.2f,%.2f,%.2f)\n", warped_vertex.normal.x, warped_vertex.normal.y, warped_vertex.normal.z, kinect_normal_normalized.x,
               kinect_normal_normalized.y, kinect_normal_normalized.z);*/

        float normals_alignment = glm::dot(warped_vertex.normal, kinect_normal_normalized);

        // printf("\nnormals alignment: %.3f\n", normals_alignment);

        if(normals_alignment < 0.f)
        {
            continue;
        }

        glm::fvec3 extracted_position;
        if(!sample_depth_projection(extracted_position, i, projection.projection[i], dev_res, measures))
        {
            continue;
        }

        glm::fvec3 diff = warped_vertex.position - extracted_position;

        // printf("\ndiff: %f\n", glm::length(diff));

        if(glm::length(diff) > 0.03f || glm::length(diff) == 0.f)
        {
            continue;
        }

        float residual_component = glm::length(glm::dot(warped_vertex.normal, diff));

        // printf("\nresidual_component: %f, warped_normal: (%f,%f,%f), diff: (%f,%f,%f)\n", residual_component, warped_vertex.normal.x, warped_vertex.normal.y, warped_vertex.normal.z, diff.x, diff.y,
        // diff.z);

        if(isnan(residual_component))
        {
#ifdef DEBUG_NANS
            printf("\ndata residual component is NaN!\n");
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
        step = 0.05f; // small quaternion twists
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
        ed_node.translation[0] += step;
        break;
    case 1:
        ed_node.translation[1] += step;
        break;
    case 2:
        ed_node.translation[2] += step;
        break;
    case 3:
        ed_node.rotation[0] += step;
        break;
    case 4:
        ed_node.rotation[1] += step;
        break;
    case 5:
        ed_node.rotation[2] += step;
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

__device__ float evaluate_hull_residual(struct_projection &warped_projection, struct_device_resources &dev_res, struct_measures &measures)
{
    float residual = 0.f;

    for(int i = 0; i < 4; i++)
    {
        if(warped_projection.projection[i].x < 0.f || warped_projection.projection[i].y < 0.f)
        {
#ifdef VERBOSE
            printf("\nprojected out of depth map: (%f,%f)\n", warped_projection.projection[i].x, warped_projection.projection[i].y);
#endif
            residual += 1.f;
            continue;
        }

        float occupancy = sample_silhouette(dev_res.silhouette_tex[i], warped_projection.projection[i]).x;

        // printf("\n (x,y): (%f,%f) = %f\n", warped_projection.projection[i].x, warped_projection.projection[i].y, occupancy);

        float residual_component = 1.0f - occupancy;

        if(isnan(residual_component))
        {
#ifdef DEBUG_NANS
            printf("\nhull residual component is NaN!\n");
#endif

            residual_component = 0.f;
        }

        residual += residual_component;
    }

    // residual /= 4.f;

    //    if(residual == 4.0f)
    //    {
    //        printf("\n residual: %f\n, v(%.2f,%.2f,%.2f)", residual, vertex.position.x, vertex.position.y, vertex.position.z);
    //    }

    return residual;
}

__device__ float evaluate_cf_residual(struct_vertex &warped_vertex, struct_vertex &reference_vertex, struct_projection &reference_vertex_projection, struct_device_resources &dev_res, struct_measures &measures)
{
    float residual = 0.f;

    for(int layer = 0; layer < 4; layer++)
    {
        if(reference_vertex_projection.projection[layer].x < 0.f || reference_vertex_projection.projection[layer].y < 0.f)
        {
#ifdef VERBOSE
            printf("\nprojected out of optical flow map: (%f,%f)\n", reference_vertex_projection.projection[layer].x, reference_vertex_projection.projection[layer].y);
#endif
            continue;
        }

        glm::fvec3 backprojected_position;
        if(!sample_prev_depth_projection(backprojected_position, layer, reference_vertex_projection.projection[layer], dev_res, measures))
        {
            continue;
        }

        glm::vec3 diff = backprojected_position - reference_vertex.position;

        if(glm::length(diff) > 0.03f || glm::length(diff) == 0.f)
        {
            continue;
        }

        float2 flow = sample_opticflow(dev_res.optical_flow_tex[layer], reference_vertex_projection.projection[layer]);

        // printf("\n(x,y): (%f,%f)\n", flow.x, flow.y);

        glm::vec2 new_projection = reference_vertex_projection.projection[layer] - glm::vec2(flow.x / ((float)measures.depth_res.x), flow.y / ((float)measures.depth_res.y));

        glm::fvec3 flow_position;
        if(!sample_depth_projection(flow_position, layer, new_projection, dev_res, measures))
        {
            continue;
        }

        float residual_component = glm::length(flow_position - warped_vertex.position);

        // printf("\nresidual_component: %f\n", residual_component);

        if(isnan(residual_component))
        {
#ifdef DEBUG_NANS
            printf("\ncorrespondence residual component is NaN!\n");
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
/*extern "C" glm::fvec3 test_warp_position(glm::fvec3 &dist, struct_ed_node &ed_node, const float &skinning_weight, struct_measures &measures)
{
    return warp_position(dist, ed_node, skinning_weight, measures);
}*/
/*extern "C" glm::fvec3 test_warp_normal(glm::fvec3 &normal, struct_ed_node &ed_node, const float &skinning_weight, struct_measures &measures)
{
    return warp_normal(normal, ed_node, skinning_weight, measures);
}*/
extern "C" unsigned int test_identify_brick_id(const glm::fvec3 position, struct_measures &measures) { return identify_brick_id(position, measures); }
extern "C" unsigned int test_identify_ed_cell_id(const glm::fvec3 position, struct_measures &measures) { return identify_ed_cell_id(position, measures); }
// extern "C" float test_evaluate_ed_node_residual(struct_ed_node &ed_node, struct_ed_meta_entry &ed_entry, struct_measures &measures)
//{
//    return evaluate_ed_node_residual(ed_node, ed_entry, measures);
//}