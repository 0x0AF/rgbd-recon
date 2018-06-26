#include <reconstruction/cuda/resources.cuh>

__global__ void kernel_mark_ed_nodes(GLuint *vx_counter, struct_vertex *vx_ptr, unsigned int ed_nodes_count, unsigned int *bricks_inv_index, int *ed_reference_counter)
{
    unsigned long long int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int vx_per_thread = (unsigned int)max(1u, vx_counter[0] / (blockDim.x * gridDim.x));

    for(unsigned int i = 0; i < vx_per_thread; i++)
    {
        unsigned long long int vertex_position = idx * vx_per_thread + i;

        if(vertex_position >= vx_counter[0])
        {
            return;
        }

        glm::vec3 position_voxel_space = vx_ptr[vertex_position].position * glm::vec3(VOLUME_VOXEL_DIM_X, VOLUME_VOXEL_DIM_Y, VOLUME_VOXEL_DIM_Z);

        const unsigned int brick_id = identify_brick_id(position_voxel_space);
        const unsigned int ed_cell_id = identify_ed_cell_id(position_voxel_space, brick_id);
        const unsigned int ed_node_offset = bricks_inv_index[brick_id] * (ED_CELL_RES * ED_CELL_RES * ED_CELL_RES) + ed_cell_id;

        if(ed_node_offset >= ed_nodes_count)
        {
            printf("\ned_node_offset overshot: %u, ed_nodes_count: %u\n", ed_node_offset, ed_nodes_count);
            return;
        }

        atomicAdd(&ed_reference_counter[ed_node_offset], 1);

        vx_ptr[vertex_position].brick_id = brick_id;
        vx_ptr[vertex_position].ed_cell_id = ed_cell_id;
    }
}

__global__ void kernel_retrieve_active_ed_nodes(unsigned int *active_ed_nodes, unsigned long long int *active_ed_vx, GLuint *vx_counter, struct_vertex *vx_ptr, unsigned int ed_nodes_count,
                                                unsigned int *bricks_dense_index, int *ed_reference_counter, struct_ed_meta_entry *ed_dense_index)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= ed_nodes_count)
    {
        // printf("\ned_nodes_count overshot: %u, ed_nodes_count: %u\n", idx, ed_nodes_count);
        return;
    }

    __shared__ int brick_ed_cell_positions[ED_CELL_RES * ED_CELL_RES * ED_CELL_RES];

    unsigned long long int ed_vx_counter = ed_reference_counter[idx];

    unsigned int ed_cell_id = idx % (ED_CELL_RES * ED_CELL_RES * ED_CELL_RES);
    unsigned int brick_pos = idx / (ED_CELL_RES * ED_CELL_RES * ED_CELL_RES);
    unsigned int brick_id = bricks_dense_index[brick_pos];

    if(ed_vx_counter == 0)
    {
        brick_ed_cell_positions[ed_cell_id] = -1;
        return;
    }

    unsigned int ed_position = atomicAdd(active_ed_nodes, 1u);
    unsigned long long int ed_vx_position = atomicAdd(active_ed_vx, ed_vx_counter);

    brick_ed_cell_positions[ed_cell_id] = ed_position;

    __syncthreads();

    ed_dense_index[ed_position].brick_id = brick_id;
    ed_dense_index[ed_position].ed_cell_id = ed_cell_id;
    ed_dense_index[ed_position].vx_offset = ed_vx_position;
    ed_dense_index[ed_position].vx_length = ed_vx_counter;
    ed_dense_index[ed_position].rejected = false;

    for(unsigned int i = 0; i < 27; i++)
    {
        ed_dense_index[ed_position].neighbors[i] = brick_ed_cell_positions[i];
    }
}

__global__ void kernel_sort_active_ed_vx(unsigned int active_ed_nodes, unsigned long long int active_ed_vx, GLuint *vx_counter, struct_vertex *vx_ptr, unsigned int *bricks_inv_index,
                                         int *ed_reference_counter, struct_ed_meta_entry *ed_dense_index, struct_vertex *sorted_vx_ptr)
{
    unsigned long long int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int vx_per_thread = (unsigned int)max(1u, (unsigned int)(active_ed_vx / (blockDim.x * gridDim.x)));

    for(unsigned int i = 0; i < vx_per_thread; i++)
    {
        unsigned long long int vertex_position = idx * vx_per_thread + i;

        if(vertex_position >= vx_counter[0])
        {
            return;
        }

        struct_vertex vx = vx_ptr[vertex_position];
        vx.position = vx.position * glm::vec3(VOLUME_VOXEL_DIM_X, VOLUME_VOXEL_DIM_Y, VOLUME_VOXEL_DIM_Z);

        unsigned long long int sorted_vx_ptr_offset = 0u;
        bool found_ed = false;

        //        if(idx % 100 == 0)
        //        {
        //            printf("\nidx: %u, vx.brick_id: %u, vx.ed_cell_id: %u, vx.position: (%f,%f,%f)\n", idx, vx.brick_id, vx.ed_cell_id, vx.position.x, vx.position.y, vx.position.z);
        //        }

        for(unsigned int ed_position = 0; ed_position < active_ed_nodes; ed_position++)
        {
            struct_ed_meta_entry ed_entry = ed_dense_index[ed_position];

            // if(idx == 100)
            // {
            // printf("\ned_entry.brick_id: %u, ed_entry.ed_cell_id: %u, ed_entry.vx_offset: %lu, ed_entry.vx_length: %lu\n",
            //                               ed_entry.brick_id, ed_entry.ed_cell_id, ed_entry.vx_offset, ed_entry.vx_length);
            // printf("\nvx.brick_id: %u, vx.ed_cell_id: %u, ed_entry.brick_id: %u, ed_entry.ed_cell_id: %u\n", vx.brick_id, vx.ed_cell_id, ed_entry.brick_id, ed_entry.ed_cell_id);
            // }

            if(ed_entry.brick_id != vx.brick_id)
            {
                continue;
            }

            if(ed_entry.ed_cell_id != vx.ed_cell_id)
            {
                continue;
            }

            //            if(ed_entry.vx_length < 1)
            //            {
            //                printf("\nfatal ed index error, active ed_entry.vx_length: %u\n", ed_entry.vx_length);
            //                return;
            //            }

            // printf("\ned_position: %u, ed_cell_id: %u, ed_brick_id: %u\n", ed_position, vx.ed_cell_id, vx.brick_id);

            const unsigned int ed_node_offset = bricks_inv_index[ed_entry.brick_id] * (ED_CELL_RES * ED_CELL_RES * ED_CELL_RES) + ed_entry.ed_cell_id;
            const int vx_sub_offset = atomicSub(&ed_reference_counter[ed_node_offset], 1);
            const int vx_sub_position = ed_entry.vx_length - vx_sub_offset;

            //            if(vx_sub_position < 0 || vx_sub_position > ed_entry.vx_length)
            //            {
            //                printf("\nvx_sub_position: %i, vx_sub_offset: %i, ed_entry.vx_length: %u\n", vx_sub_position, vx_sub_offset, ed_entry.vx_length);
            //            }

            found_ed = true;
            sorted_vx_ptr_offset = ed_entry.vx_offset + vx_sub_position;

            break;
        }

        __syncthreads();

        if(found_ed)
        {
            memcpy(&sorted_vx_ptr[sorted_vx_ptr_offset], &vx, sizeof(struct_vertex));
        }

        //        if((int)vx.position.x == 0 || (int)sorted_vx_ptr[sorted_vx_ptr_offset].position.x == 0)
        //        {
        //            printf("\nvx.position.x == %f || sorted_vx.x == %f, node_idx: %u\n", vx.position.x, sorted_vx_ptr[sorted_vx_ptr_offset].position.x, idx);
        //        }
    }
}

__global__ void kernel_sample_ed_nodes(unsigned int active_ed_nodes, unsigned long long int active_ed_vx, struct_vertex *sorted_vx_ptr, struct_ed_meta_entry *ed_dense_index,
                                       struct_ed_node *ed_graph)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= active_ed_nodes)
    {
        // printf("\ned_nodes_count overshot: %u, ed_nodes_count: %u\n", idx, ed_nodes_count);
        return;
    }

    glm::vec3 position{0.f};
    struct_ed_meta_entry ed_entry = ed_dense_index[idx];

    // printf("\ned_entry.vx_offset %lu, ed_entry.vx_length %u\n", ed_entry.vx_offset, ed_entry.vx_length);

    for(unsigned int vx_position = 0; vx_position < ed_entry.vx_length; vx_position++)
    {
        unsigned long long int sorted_vx_ptr_offset = ed_entry.vx_offset + vx_position;

        if(sorted_vx_ptr_offset >= active_ed_vx)
        {
            printf("\nsorted_vx_ptr_offset overshot: %u, active_ed_vx: %u\n", sorted_vx_ptr_offset, active_ed_vx);
            continue;
        }

        //        if((int)sorted_vx_ptr[sorted_vx_ptr_offset].position.x == 0)
        //        {
        //            printf("\nsorted_vx.x == 0, node_idx: %u\n", idx);
        //        }

        position = position + sorted_vx_ptr[sorted_vx_ptr_offset].position / (float)(ed_entry.vx_length);

        //        if(idx == 100)
        //        {
        //            glm::vec3 vx_pos = sorted_vx_ptr[sorted_vx_ptr_offset].position;
        //            printf("\nposition %u, length %u: (%f,%f,%f), av (%f,%f,%f)\n", vx_position, ed_entry.vx_length, vx_pos.x, vx_pos.y, vx_pos.z, position.x, position.y, position.z);
        //        }
    }

    __syncthreads();

    memcpy(&ed_graph[idx], &position, sizeof(glm::vec3));
}

extern "C" void sample_ed_nodes()
{
    unsigned int ed_nodes_count = _active_bricks_count * ED_CELL_RES * ED_CELL_RES * ED_CELL_RES;

    unsigned int *active_ed_nodes_count;
    cudaMallocManaged(&active_ed_nodes_count, sizeof(unsigned int));
    *active_ed_nodes_count = 0u;

    unsigned long long int *active_ed_vx_count;
    cudaMallocManaged(&active_ed_vx_count, sizeof(unsigned long long int));
    *active_ed_vx_count = 0u;

    int *ed_reference_counter;
    checkCudaErrors(cudaMalloc(&ed_reference_counter, ed_nodes_count * sizeof(int)));
    checkCudaErrors(cudaMemset(ed_reference_counter, 0, ed_nodes_count * sizeof(int)));

    free_ed_resources();

    checkCudaErrors(cudaMalloc(&_ed_graph_meta, ed_nodes_count * sizeof(struct_ed_meta_entry)));
    checkCudaErrors(cudaMemset(_ed_graph_meta, 0, ed_nodes_count * sizeof(struct_ed_meta_entry)));

    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.buffer_vertex_counter, 0));
    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.buffer_reference_mesh_vertices, 0));

    size_t vx_bytes;
    GLuint *vx_counter;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&vx_counter, &vx_bytes, _cgr.buffer_vertex_counter));

    struct_vertex *vx_ptr;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&vx_ptr, &vx_bytes, _cgr.buffer_reference_mesh_vertices));

    // printf("\nvx_bytes: %zu\n", vx_bytes);

    int block_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_mark_ed_nodes, 0, 0);

    unsigned max_vertices = ((unsigned)vx_bytes) / sizeof(struct_vertex);
    size_t grid_size = (max_vertices + block_size - 1) / block_size;
    kernel_mark_ed_nodes<<<grid_size, block_size>>>(vx_counter, vx_ptr, ed_nodes_count, _bricks_inv_index, ed_reference_counter);

    getLastCudaError("render kernel failed");

    cudaDeviceSynchronize();

    block_size = ED_CELL_RES * ED_CELL_RES * ED_CELL_RES;
    grid_size = (ed_nodes_count + block_size - 1) / block_size;
    kernel_retrieve_active_ed_nodes<<<grid_size, block_size>>>(active_ed_nodes_count, active_ed_vx_count, vx_counter, vx_ptr, ed_nodes_count, _bricks_dense_index, ed_reference_counter,
                                                               _ed_graph_meta);

    getLastCudaError("render kernel failed");

    cudaDeviceSynchronize();

    checkCudaErrors(cudaMemcpy(&_active_ed_nodes_count, active_ed_nodes_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    printf("\nactive_ed_nodes_count: %u\n", _active_ed_nodes_count);

    checkCudaErrors(cudaMemcpy(&_active_ed_vx_count, active_ed_vx_count, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
    printf("\nactive_ed_vx_count: %lu\n", _active_ed_vx_count);

    allocate_ed_resources();

    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_sort_active_ed_vx, 0, 0);
    grid_size = (_active_ed_vx_count + block_size - 1) / block_size;
    kernel_sort_active_ed_vx<<<grid_size, block_size>>>(_active_ed_nodes_count, _active_ed_vx_count, vx_counter, vx_ptr, _bricks_inv_index, ed_reference_counter, _ed_graph_meta,
                                                        _sorted_vx_ptr);

    getLastCudaError("render kernel failed");

    cudaDeviceSynchronize();

    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_sample_ed_nodes, 0, 0);
    grid_size = (_active_ed_nodes_count + block_size - 1) / block_size;
    kernel_sample_ed_nodes<<<grid_size, block_size>>>(_active_ed_nodes_count, _active_ed_vx_count, _sorted_vx_ptr, _ed_graph_meta, _ed_graph);

    getLastCudaError("render kernel failed");

    cudaDeviceSynchronize();

    if(ed_reference_counter != nullptr)
    {
        checkCudaErrors(cudaFree(ed_reference_counter));
    }

    if(active_ed_nodes_count != nullptr)
    {
        checkCudaErrors(cudaFree(active_ed_nodes_count));
    }

    if(active_ed_vx_count != nullptr)
    {
        checkCudaErrors(cudaFree(active_ed_vx_count));
    }

    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.buffer_reference_mesh_vertices, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.buffer_vertex_counter, 0));
}