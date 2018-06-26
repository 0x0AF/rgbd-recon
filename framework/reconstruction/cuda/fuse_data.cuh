#include <reconstruction/cuda/resources.cuh>

__global__ void kernel_fuse_data(unsigned int active_ed_nodes_count, struct_ed_meta_entry *ed_dense_index, struct_ed_node *ed_graph)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= active_ed_nodes_count * ED_CELL_VOXELS)
    {
        // printf("\nactive voxel count overshot: %u, active_bricks_count * BRICK_VOXELS: %lu\n", idx, active_bricks_count * BRICK_VOXELS);
        return;
    }

    unsigned int ed_node_offset = idx / ED_CELL_VOXELS;

    struct_ed_node ed_node = ed_graph[ed_node_offset];
    struct_ed_meta_entry ed_entry = ed_dense_index[ed_node_offset];

    unsigned int brick_id = ed_entry.brick_id;
    unsigned int ed_cell_id = ed_entry.ed_cell_id;
    unsigned int edc_voxel_id = idx % ED_CELL_VOXELS;

    glm::uvec3 brick = index_3d(brick_id) * BRICK_VOXEL_DIM;
    glm::uvec3 ed_cell_index3d = ed_cell_voxel_3d(edc_voxel_id);
    glm::uvec3 position = ed_cell_3d(ed_cell_id) + ed_cell_index3d;
    glm::uvec3 world = brick + position;

    //    if(position_id == 0)
    //    {
    //        printf("\nbrick %u: (%u,%u,%u)\n", brick_id, brick.x, brick.y, brick.z);
    //    }

    if(world.x >= VOLUME_VOXEL_DIM_X || world.y >= VOLUME_VOXEL_DIM_Y || world.z >= VOLUME_VOXEL_DIM_Z)
    {
        printf("\nout of volume: w(%u,%u,%u) = b(%u,%u,%u) + p(%u,%u,%u)\n", world.x, world.y, world.z, brick.x, brick.y, brick.z, position.x, position.y, position.z);
        return;
    }

    //    if(edc_voxel_id == 0)
    //    {
    //        printf("\ned node %u: (%f,%f,%f)\n", ed_node_offset, ed_node.position.x, ed_node.position.y, ed_node.position.z);
    //    }

    //    if(edc_voxel_id == 0)
    //    {
    //        printf("\ned node %u, position: (%u,%u,%u)\n", idx / ED_CELL_VOXELS, world.x, world.y, world.z);
    //    }

    // warped position

    glm::vec3 dist = glm::vec3(world) - ed_node.position;

    //    if(edc_voxel_id == 0)
    //    {
    //        printf("\n|dist|: %f\n", glm::length(dist));
    //    }

    const float skinning_weight = 1.f; // expf(glm::length(dist) * glm::length(dist) * 2 / (ED_CELL_VOXEL_DIM * ED_CELL_VOXEL_DIM));
    glm::uvec3 warped_position = glm::uvec3(warp_position(dist, ed_node, skinning_weight));

    if(warped_position.x >= VOLUME_VOXEL_DIM_X || warped_position.y >= VOLUME_VOXEL_DIM_Y || warped_position.z >= VOLUME_VOXEL_DIM_Z)
    {
        //        printf("\nwarped out of volume: (%u,%u,%u), w(%u,%u,%u) = b(%u,%u,%u) + p(%u,%u,%u)\n", warped_position.x, warped_position.y, warped_position.z, world.x, world.y, world.z, brick.x,
        //        brick.y,
        //               brick.z, position.x, position.y, position.z);
        return;
    }

    if(ed_entry.rejected)
    {
        float2 data;
        surf3Dread(&data, _volume_tsdf_data, warped_position.x * sizeof(float2), warped_position.y, warped_position.z);
        surf3Dwrite(data, _volume_tsdf_ref, world.x * sizeof(float2), world.y, world.z);
        return;
    }

    __shared__ float2 ed_cell_voxels[ED_CELL_VOXELS];
    __shared__ float2 edc_predictions[ED_CELL_VOXELS];

    surf3Dread(&ed_cell_voxels[edc_voxel_id], _volume_tsdf_ref, world.x * sizeof(float2), world.y, world.z);

    __syncthreads();

    //    if(edc_voxel_id == 0)
    //    {
    //        glm::vec3 diff = glm::vec3(warped_position) - glm::vec3(world);
    //        printf("\ndiff %u: (%u,%u,%u)\n", glm::length(diff), warped_position.x, warped_position.y, warped_position.z);
    //    }

    // warped gradient

    glm::vec3 gradient;

    if(ed_cell_index3d.x != ED_CELL_VOXEL_DIM)
    {
        unsigned int ed_cell_voxel_id_x = ed_cell_voxel_id(ed_cell_index3d + glm::uvec3(1, 0, 0));
        gradient.x = ed_cell_voxels[ed_cell_voxel_id_x].x / 2.0f - ed_cell_voxels[edc_voxel_id].x / 2.0f;
    }
    else
    {
        float2 voxel_x;
        surf3Dread(&voxel_x, _volume_tsdf_ref, (world.x + 1) * sizeof(float2), world.y, world.z);

        gradient.x = voxel_x.x / 2.0f - ed_cell_voxels[edc_voxel_id].x / 2.0f;
    }

    if(ed_cell_index3d.y != ED_CELL_VOXEL_DIM)
    {
        unsigned int ed_cell_voxel_id_y = ed_cell_voxel_id(ed_cell_index3d + glm::uvec3(0, 1, 0));
        gradient.x = ed_cell_voxels[ed_cell_voxel_id_y].x / 2.0f - ed_cell_voxels[edc_voxel_id].x / 2.0f;
    }
    else
    {
        float2 voxel_y;
        surf3Dread(&voxel_y, _volume_tsdf_ref, world.x * sizeof(float2), world.y + 1, world.z);

        gradient.y = voxel_y.x / 2.0f - ed_cell_voxels[edc_voxel_id].x / 2.0f;
    }

    if(ed_cell_index3d.z != ED_CELL_VOXEL_DIM)
    {
        unsigned int ed_cell_voxel_id_z = ed_cell_voxel_id(ed_cell_index3d + glm::uvec3(0, 0, 1));
        gradient.z = ed_cell_voxels[ed_cell_voxel_id_z].x / 2.0f - ed_cell_voxels[edc_voxel_id].x / 2.0f;
    }
    else
    {
        float2 voxel_z;
        surf3Dread(&voxel_z, _volume_tsdf_ref, world.x * sizeof(float2), world.y, world.z + 1);

        gradient.z = voxel_z.x / 2.0f - ed_cell_voxels[edc_voxel_id].x / 2.0f;
    }

    glm::vec3 gradient_vector = glm::normalize(gradient);
    glm::vec3 warped_gradient_vector = warp_normal(gradient_vector, ed_node, skinning_weight);
    glm::vec3 warped_gradient = warped_gradient_vector * glm::length(gradient);

    glm::bvec3 is_nan = glm::isnan(warped_gradient);

    if(is_nan.x || is_nan.y || is_nan.z)
    {
        // printf("\nNaN in gradient warp evaluation\n");
        warped_gradient = glm::vec3(0.f);
    }

    //    if(position_id == 0)
    //    {
    //        printf("\ngradient: (%f,%f,%f), warped gradient: (%f,%f,%f)\n", gradient.x, gradient.y, gradient.z, warped_gradient.x, warped_gradient.y, warped_gradient.z);
    //    }

    // reference frame SDF prediction

    float2 prediction{0.f, 0.f};
    prediction.x = ed_cell_voxels[edc_voxel_id].x + glm::dot(warped_gradient, glm::vec3(warped_position) - glm::vec3(world));
    prediction.y = 1.0f;

    edc_predictions[edc_voxel_id] = prediction;

    __syncthreads();

    for(unsigned int i = 0; i < ED_CELL_VOXELS; i++)
    {
        if(i == edc_voxel_id)
        {
            continue;
        }
        glm::uvec3 edc_voxel_3d = ed_cell_voxel_3d(edc_voxel_id);
        float dist = glm::length(glm::vec3(edc_voxel_3d));
        float weight = expf(-dist * dist / 2.0f / (float)ED_CELL_VOXEL_DIM / (float)ED_CELL_VOXEL_DIM);
        edc_predictions[i].x = edc_predictions[i].x * edc_predictions[i].y + prediction.x * weight / (edc_predictions[i].y + weight);
        edc_predictions[i].y = edc_predictions[i].y + weight;
    }

    __syncthreads();

    //    if(position_id == 0)
    //    {
    //        printf("\nprediction: %f\n", prediction.x);
    //    }

    // TODO: blending

    float2 data;
    surf3Dread(&data, _volume_tsdf_data, warped_position.x * sizeof(float2), warped_position.y, warped_position.z);

    float2 fused;

    fused.y = edc_predictions[edc_voxel_id].y + data.y;

    if(fused.y > 0.001f)
    {
        fused.x = data.x * data.y / fused.y + edc_predictions[edc_voxel_id].x * edc_predictions[edc_voxel_id].y / fused.y;
    }
    else
    {
        fused.x = data.y > edc_predictions[edc_voxel_id].y ? data.x : edc_predictions[edc_voxel_id].x;
        fused.y = data.y > edc_predictions[edc_voxel_id].y ? data.y : edc_predictions[edc_voxel_id].y;
    }

    __syncthreads();

    //    if(edc_voxel_id == 0)
    //    {
    //        printf("\nref: %f, prediction: %f, fused: %f\n", ed_cell_voxels[edc_voxel_id].x, prediction.x, fused.x);
    //    }

    // surf3Dwrite(float2{0.00f, 1.00f}, _volume_tsdf_data, warped_position.x * sizeof(float2), warped_position.y, warped_position.z);
    // surf3Dwrite(float2{0.00f, 1.00f}, _volume_tsdf_data, world.x * sizeof(float2), world.y, world.z);
    // surf3Dwrite(ed_cell_voxels[edc_voxel_id].x, _volume_tsdf_data, warped_position.x * sizeof(float2), warped_position.y, warped_position.z);
    // surf3Dwrite(prediction, _volume_tsdf_data, warped_position.x * sizeof(float2), warped_position.y, warped_position.z);
    surf3Dwrite(fused, _volume_tsdf_data, warped_position.x * sizeof(float2), warped_position.y, warped_position.z);
}

extern "C" void fuse_data()
{
    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.volume_tsdf_data, 0));

    cudaArray *volume_array_tsdf_data = nullptr;
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&volume_array_tsdf_data, _cgr.volume_tsdf_data, 0, 0));

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
    checkCudaErrors(cudaBindSurfaceToArray(&_volume_tsdf_data, volume_array_tsdf_data, &channel_desc));
    checkCudaErrors(cudaBindSurfaceToArray(&_volume_tsdf_ref, _volume_array_tsdf_ref, &channel_desc));

    unsigned int active_ed_voxels = _active_ed_nodes_count * ED_CELL_VOXELS;
    size_t grid_size = (active_ed_voxels + ED_CELL_VOXELS - 1) / ED_CELL_VOXELS;

    // printf("\ngrid_size: %lu, block_size: %u\n", grid_size, ED_CELL_VOXELS);

    kernel_fuse_data<<<grid_size, ED_CELL_VOXELS>>>(_active_ed_nodes_count, _ed_graph_meta, _ed_graph);

    getLastCudaError("render kernel failed");

    cudaDeviceSynchronize();

    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.volume_tsdf_data, 0));
}
