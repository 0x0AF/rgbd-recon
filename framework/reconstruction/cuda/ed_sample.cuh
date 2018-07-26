#include <reconstruction/cuda/resources.cuh>

__global__ void kernel_mark_ed_nodes(unsigned long long int *active_vx, GLuint *vx_counter, struct_vertex *vx_ptr, unsigned int ed_nodes_count, int *ed_reference_counter,
                                     struct_device_resources dev_res, struct_measures measures)
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

        struct_vertex vx = vx_ptr[vertex_position];

        vx.position = glm::clamp(vx.position, glm::vec3(0.f), glm::vec3(1.f));

        if(!in_normal_space(vx.position))
        {
            printf("\nspawned vertex out of normal space, vx.position: (%f,%f,%f)\n", vx.position.x, vx.position.y, vx.position.z);
            continue;
        }

        const unsigned int brick_id = identify_brick_id(vx.position, measures);

        if(brick_id >= measures.data_volume_num_bricks)
        {
            printf("\nspawned vertex outside of volume bricks, brick id: %u / %u\n", brick_id, measures.data_volume_num_bricks);
            continue;
        }

        const unsigned int brick_pos = dev_res.bricks_inv_index[brick_id];

        if(brick_pos == 0 && dev_res.bricks_dense_index[0] != brick_id)
        {
            // printf("\nspawned vertex outside of occupied bricks, brick id: %u\n", brick_id);
            continue;
        }

        const unsigned int ed_cell_id = identify_ed_cell_id(vx.position, measures);
        const unsigned int ed_node_offset = brick_pos * measures.brick_num_ed_cells + ed_cell_id;

        if(ed_node_offset >= ed_nodes_count)
        {
            printf("\ned_node_offset overshot: %u = %u * 27 + %u, ed_nodes_count: %u\n", ed_node_offset, brick_id, ed_cell_id, ed_nodes_count);
            return;
        }

        atomicAdd(&ed_reference_counter[ed_node_offset], 1);

        vx.brick_id = brick_id;
        vx.ed_cell_id = ed_cell_id;

        unsigned long long int unsorted_vx_position = atomicAdd(active_vx, 1);
        memcpy(&dev_res.unsorted_vx_ptr[unsorted_vx_position], &vx, sizeof(struct_vertex));
    }
}

__global__ void kernel_retrieve_active_ed_nodes(unsigned int *active_ed_nodes, unsigned long long int *active_ed_vx, unsigned int ed_nodes_count, int *ed_reference_counter,
                                                struct_device_resources dev_res, struct_measures measures)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= ed_nodes_count)
    {
        // printf("\ned_nodes_count overshot: %u, ed_nodes_count: %u\n", idx, ed_nodes_count);
        return;
    }

    __shared__ int brick_ed_cell_positions[27]; // point of error

    unsigned long long int ed_vx_counter = ed_reference_counter[idx];

    unsigned int ed_cell_id = idx % measures.brick_num_ed_cells;
    unsigned int brick_pos = idx / measures.brick_num_ed_cells;
    unsigned int brick_id = dev_res.bricks_dense_index[brick_pos];

    if(ed_vx_counter == 0)
    {
        brick_ed_cell_positions[ed_cell_id] = -1;
        return;
    }

    unsigned int ed_position = atomicAdd(active_ed_nodes, 1u);
    unsigned long long int ed_vx_position = atomicAdd(active_ed_vx, ed_vx_counter);

    brick_ed_cell_positions[ed_cell_id] = ed_position;

    __syncthreads();

    struct_ed_meta_entry ed_meta = dev_res.ed_graph_meta[ed_position];

    ed_meta.brick_id = brick_id;
    ed_meta.ed_cell_id = ed_cell_id;
    ed_meta.vx_offset = ed_vx_position;
    ed_meta.vx_length = ed_vx_counter;
    ed_meta.rejected = false;

    for(unsigned int i = 0; i < 27; i++)
    {
        ed_meta.neighbors[i] = brick_ed_cell_positions[i];
    }

    memcpy(&dev_res.ed_graph_meta[ed_position], &ed_meta, sizeof(struct_ed_meta_entry));
}

__global__ void kernel_sort_active_ed_vx(unsigned int active_ed_nodes, unsigned long long int active_ed_vx, int *ed_reference_counter, struct_device_resources dev_res, struct_measures measures)
{
    unsigned long long int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int vx_per_thread = (unsigned int)max(1u, (unsigned int)(active_ed_vx / (blockDim.x * gridDim.x)));

    for(unsigned int i = 0; i < vx_per_thread; i++)
    {
        unsigned long long int vertex_position = idx * vx_per_thread + i;

        if(vertex_position >= active_ed_vx)
        {
            return;
        }

        struct_vertex vx = dev_res.unsorted_vx_ptr[vertex_position];

        unsigned long long int sorted_vx_ptr_offset = 0u;
        bool found_ed = false;

        //        if(idx % 100 == 0)
        //        {
        //            printf("\nidx: %u, vx.brick_id: %u, vx.ed_cell_id: %u, vx.position: (%f,%f,%f)\n", idx, vx.brick_id, vx.ed_cell_id, vx.position.x, vx.position.y, vx.position.z);
        //        }

        for(unsigned int ed_position = 0; ed_position < active_ed_nodes; ed_position++)
        {
            struct_ed_meta_entry ed_entry = dev_res.ed_graph_meta[ed_position];

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

            const unsigned int ed_node_offset = dev_res.bricks_inv_index[ed_entry.brick_id] * measures.brick_num_ed_cells + ed_entry.ed_cell_id;
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
            memcpy(&dev_res.sorted_vx_ptr[sorted_vx_ptr_offset], &vx, sizeof(struct_vertex));

            //      printf("\nvx.position.x == %f || sorted_vx.x == %f, node_idx: %u\n", vx.position.x, dev_res.sorted_vx_ptr[sorted_vx_ptr_offset].position.x, idx);
            //      printf("\nvx.normal.x == %f || sorted_vx.normal.x == %f, node_idx: %u\n", vx.normal.x, dev_res.sorted_vx_ptr[sorted_vx_ptr_offset].normal.x, idx);
        }
        //        else
        //        {
        //            if(vx.brick_id < measures.data_volume_num_bricks)
        //            {
        //                printf("\ncrsp. ed node not found, vx.brick_id: %u, vx.ed_cell_id: %u, vx.position: (%f,%f,%f), brick inv index entry: %u\n", vx.brick_id, vx.ed_cell_id, vx.position.x,
        //                vx.position.y,
        //                       vx.position.z, dev_res.bricks_inv_index[vx.brick_id]);
        //            }
        //            else
        //            {
        //                printf("\ncrsp. ed node not found, vx.brick_id overshot: %u / %u\n", vx.brick_id, measures.data_volume_num_bricks);
        //            }
        //        }
    }
}

__global__ void kernel_sample_ed_nodes(unsigned int active_ed_nodes, unsigned long long int active_ed_vx, struct_device_resources dev_res, struct_measures measures)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= active_ed_nodes)
    {
        // printf("\ned_nodes_count overshot: %u, ed_nodes_count: %u\n", idx, ed_nodes_count);
        return;
    }

    struct_ed_node ed_node = dev_res.ed_graph[idx];
    struct_ed_meta_entry ed_entry = dev_res.ed_graph_meta[idx];

    ed_node.position = glm::vec3(0.f);

    // printf("\ned_entry.vx_offset %lu, ed_entry.vx_length %u\n", ed_entry.vx_offset, ed_entry.vx_length);

    for(unsigned int vx_position = 0; vx_position < ed_entry.vx_length; vx_position++)
    {
        unsigned long long int sorted_vx_ptr_offset = ed_entry.vx_offset + vx_position;

        if(sorted_vx_ptr_offset >= active_ed_vx)
        {
            printf("\nsorted_vx_ptr_offset overshot: %u, active_ed_vx: %u\n", sorted_vx_ptr_offset, active_ed_vx);
            continue;
        }

        // printf("\nsorted_vx.normal: (%f,%f,%f)\n", dev_res.sorted_vx_ptr[sorted_vx_ptr_offset].normal.x, dev_res.sorted_vx_ptr[sorted_vx_ptr_offset].normal.y,
        // dev_res.sorted_vx_ptr[sorted_vx_ptr_offset].normal.z);

        //        if(dev_res.sorted_vx_ptr[sorted_vx_ptr_offset].brick_id != ed_entry.brick_id)
        //        {
        //            printf("\nvx.brick_id != ed_entry.brick_id [%u!=%u], node_idx: %u\n", dev_res.sorted_vx_ptr[sorted_vx_ptr_offset].brick_id, ed_entry.brick_id, idx);
        //        }

        ed_node.position = ed_node.position + dev_res.sorted_vx_ptr[sorted_vx_ptr_offset].position / (float)(ed_entry.vx_length);

        /*if(idx == 100)
        {
            glm::vec3 vx_pos = sorted_vx_ptr[sorted_vx_ptr_offset].position;
            printf("\nposition %u, length %u: (%f,%f,%f), av (%f,%f,%f)\n", vx_position, ed_entry.vx_length, vx_pos.x, vx_pos.y, vx_pos.z, position.x, position.y, position.z);
        }*/
    }

    __syncthreads();

    ed_node.translation = glm::vec3(0.f);
    ed_node.affine = glm::quat(1.f, 0.f, 0.f, 0.f);

    memcpy(&dev_res.ed_graph[idx], &ed_node, sizeof(struct_ed_node));
}

extern "C" void sample_ed_nodes()
{
    unsigned int ed_nodes_count = _host_res.active_bricks_count * _host_res.measures.brick_num_ed_cells;

    unsigned int *active_ed_nodes_count;
    cudaMallocManaged(&active_ed_nodes_count, sizeof(unsigned int));
    *active_ed_nodes_count = 0u;

    unsigned long long int *active_ed_vx_count;
    cudaMallocManaged(&active_ed_vx_count, sizeof(unsigned long long int));
    *active_ed_vx_count = 0u;

    unsigned long long int *active_vx_count;
    cudaMallocManaged(&active_vx_count, sizeof(unsigned long long int));
    *active_vx_count = 0u;

    int *ed_reference_counter;
    checkCudaErrors(cudaMalloc(&ed_reference_counter, ed_nodes_count * sizeof(int)));
    checkCudaErrors(cudaMemset(ed_reference_counter, 0, ed_nodes_count * sizeof(int)));

    clean_ed_resources ();

    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.buffer_vertex_counter));
    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.buffer_reference_mesh_vertices));

    size_t vx_bytes;
    GLuint *vx_counter;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&vx_counter, &vx_bytes, _cgr.buffer_vertex_counter));

    struct_vertex *vx_ptr;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&vx_ptr, &vx_bytes, _cgr.buffer_reference_mesh_vertices));

    checkCudaErrors(cudaMalloc(&_dev_res.unsorted_vx_ptr, vx_bytes));
    checkCudaErrors(cudaMemset(_dev_res.unsorted_vx_ptr, 0, vx_bytes));

    // printf("\nvx_bytes: %zu\n", vx_bytes);

    int block_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_mark_ed_nodes, 0, 0);

    unsigned max_vertices = ((unsigned)vx_bytes) / sizeof(struct_vertex);
    size_t grid_size = (max_vertices + block_size - 1) / block_size;
    kernel_mark_ed_nodes<<<grid_size, block_size>>>(active_vx_count, vx_counter, vx_ptr, ed_nodes_count, ed_reference_counter, _dev_res, _host_res.measures);
    getLastCudaError("render kernel failed");
    cudaDeviceSynchronize();

    block_size = _host_res.measures.brick_num_ed_cells;
    grid_size = (ed_nodes_count + block_size - 1) / block_size;
    kernel_retrieve_active_ed_nodes<<<grid_size, block_size>>>(active_ed_nodes_count, active_ed_vx_count, ed_nodes_count, ed_reference_counter, _dev_res, _host_res.measures);
    getLastCudaError("render kernel failed");
    cudaDeviceSynchronize();

    checkCudaErrors(cudaMemcpy(&_host_res.active_ed_nodes_count, active_ed_nodes_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    printf("\nactive_ed_nodes_count: %u\n", _host_res.active_ed_nodes_count);

    checkCudaErrors(cudaMemcpy(&_host_res.active_ed_vx_count, active_ed_vx_count, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
    printf("\nactive_ed_vx_count: %lu\n", _host_res.active_ed_vx_count);

    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_sort_active_ed_vx, 0, 0);
    grid_size = (_host_res.active_ed_vx_count + block_size - 1) / block_size;
    kernel_sort_active_ed_vx<<<grid_size, block_size>>>(_host_res.active_ed_nodes_count, _host_res.active_ed_vx_count, ed_reference_counter, _dev_res, _host_res.measures);
    getLastCudaError("render kernel failed");
    cudaDeviceSynchronize();

    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_sample_ed_nodes, 0, 0);
    grid_size = (_host_res.active_ed_nodes_count + block_size - 1) / block_size;
    kernel_sample_ed_nodes<<<grid_size, block_size>>>(_host_res.active_ed_nodes_count, _host_res.active_ed_vx_count, _dev_res, _host_res.measures);
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

    if(active_vx_count != nullptr)
    {
        checkCudaErrors(cudaFree(active_vx_count));
    }

    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.buffer_reference_mesh_vertices));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.buffer_vertex_counter));
}

__global__ void kernel_push_debug_ed_nodes(struct_ed_node_debug *ed_ptr, unsigned int ed_node_count, struct_device_resources dev_res, struct_measures measures)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int ed_per_thread = (unsigned int)max(1u, ed_node_count / (blockDim.x * gridDim.x));

    for(unsigned int i = 0; i < ed_per_thread; i++)
    {
        unsigned int ed_position = idx * ed_per_thread + i;

        if(ed_position >= ed_node_count)
        {
            return;
        }

        struct_ed_node_debug node = ed_ptr[ed_position];

        node.position = dev_res.ed_graph[ed_position].position;
        node.brick_id = dev_res.ed_graph_meta[ed_position].brick_id;
        node.translation = dev_res.ed_graph[ed_position].translation;
        node.ed_cell_id = dev_res.ed_graph_meta[ed_position].ed_cell_id;
        node.affine = dev_res.ed_graph[ed_position].affine;
        node.vx_offset = (unsigned int)dev_res.ed_graph_meta[ed_position].vx_offset;
        node.vx_length = dev_res.ed_graph_meta[ed_position].vx_length;

        memcpy(&ed_ptr[ed_position], &node, sizeof(struct_ed_node_debug));
    }
}

extern "C" unsigned int push_debug_ed_nodes()
{
    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.buffer_ed_nodes_debug));

    struct_ed_node_debug *ed_ptr;
    size_t ed_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&ed_ptr, &ed_bytes, _cgr.buffer_ed_nodes_debug));

    int block_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_push_debug_ed_nodes, 0, 0);

    size_t grid_size = (_host_res.active_ed_nodes_count + block_size - 1) / block_size;
    kernel_push_debug_ed_nodes<<<grid_size, block_size>>>(ed_ptr, _host_res.active_ed_nodes_count, _dev_res, _host_res.measures);
    getLastCudaError("render kernel failed");
    cudaDeviceSynchronize();

    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.buffer_ed_nodes_debug));

    return _host_res.active_ed_nodes_count;
}

__global__ void kernel_push_debug_sorted_vertices(struct_vertex *vx_ptr, unsigned int vx_count, struct_device_resources dev_res, struct_measures measures)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int vx_per_thread = (unsigned int)max(1u, vx_count / (blockDim.x * gridDim.x));

    for(unsigned int i = 0; i < vx_per_thread; i++)
    {
        unsigned int vx_position = idx * vx_per_thread + i;

        if(vx_position >= vx_count)
        {
            return;
        }

        memcpy(&vx_ptr[vx_position], &dev_res.sorted_vx_ptr[vx_position], sizeof(struct_vertex));
    }
}

extern "C" unsigned long push_debug_sorted_vertices()
{
    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.buffer_sorted_vertices_debug));

    struct_vertex *vx_ptr;
    size_t vx_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&vx_ptr, &vx_bytes, _cgr.buffer_sorted_vertices_debug));

    int block_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_push_debug_sorted_vertices, 0, 0);
    size_t grid_size = (_host_res.active_ed_vx_count + block_size - 1) / block_size;
    kernel_push_debug_sorted_vertices<<<grid_size, block_size>>>(vx_ptr, _host_res.active_ed_vx_count, _dev_res, _host_res.measures);
    getLastCudaError("render kernel failed");
    cudaDeviceSynchronize();

    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.buffer_sorted_vertices_debug));

    return _host_res.active_ed_vx_count;
}