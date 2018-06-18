#include <reconstruction/cuda/resources.cuh>

__global__ void kernel_jtj_jtf(float *jtj, float *jtf, GLuint *vx_counter, struct_vertex *vx_ptr, unsigned int ed_nodes_count, struct_ed_node *ed_graph)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float jtj_diag_coo_val_blocks[6400]; // 25,6 Kb
    __shared__ float jtf_blocks[640];               // 2,56 Kb

    unsigned int ed_node_offset = idx / ED_COMPONENT_COUNT;
    if(ed_node_offset >= ed_nodes_count)
    {
        // printf("\ned_node_offset overshot: %u, ed_nodes_count: %u\n", ed_node_offset, ed_nodes_count);
        return;
    }

    struct_ed_node ed_node = ed_graph[ed_node_offset];

    if(ed_node.position.z * ED_CELL_RES * ED_CELL_RES + ed_node.position.y * ED_CELL_RES + ed_node.position.x == 0u)
    {
        // unset ed_node
        return;
    }

    unsigned int block_ed_node_component_offset = idx % JTJ_JTF_BLOCK_SIZE;
    unsigned int block_ed_node_offset = block_ed_node_component_offset / ED_COMPONENT_COUNT;
    unsigned int component = idx % ED_COMPONENT_COUNT;

    // printf("\ned_node position: (%f,%f,%f)\n", ed_node.position.x, ed_node.position.y, ed_node.position.z);

    for(unsigned long vx_idx = 0; vx_idx < vx_counter[0]; vx_idx++)
    {
        struct_vertex vx = vx_ptr[vx_idx];

        if(vx.ed_cell_id != ed_node_offset)
        {
            // is not influenced by ed_node
            continue;
        }

        vx.position = vx.position * glm::vec3(VOLUME_VOXEL_DIM_X, VOLUME_VOXEL_DIM_Y, VOLUME_VOXEL_DIM_Z);

        // printf("\ned_node + vertex match\n");

        float vx_residual = evaluate_vx_residual(vx, ed_node);

        if(isnan(vx_residual))
        {
            printf("\nvx_residual is NaN!\n");

            vx_residual = 0.f;
        }

        __shared__ float vx_pds[640]; // 2,56 Kb

        vx_pds[block_ed_node_component_offset + component] = evaluate_vx_pd(vx, ed_node, component, vx_residual);

        if(isnan(vx_pds[block_ed_node_component_offset + component]))
        {
            printf("\nvx_pds[%u] is NaN!\n", component);

            vx_pds[block_ed_node_component_offset + component] = 0.f;
        }

        __syncthreads();

        float jtf_value = vx_pds[block_ed_node_component_offset + component] * vx_residual;

        if(isnan(jtf_value))
        {
            printf("\njtf_value[%u] is NaN!\n", component);

            jtf_value = 0.f;
        }

        jtf_blocks[block_ed_node_component_offset + component] += jtf_value;

        for(unsigned int component_k = 0; component_k < ED_COMPONENT_COUNT; component_k++)
        {
            unsigned int jtj_pos = block_ed_node_offset * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT + component * ED_COMPONENT_COUNT + component_k;

            float jtj_value = vx_pds[block_ed_node_component_offset + component] * vx_pds[block_ed_node_component_offset + component_k];

            if(isnan(jtj_value))
            {
                printf("\njtj_value[%u] is NaN!\n", component);

                jtj_value = 0.f;
            }

            jtj_diag_coo_val_blocks[jtj_pos] += jtj_value;
        }

        __syncthreads();
    }

    // printf("\njtf[%u]\n", ed_node_offset * 10u);

    __syncthreads();

    unsigned int block_id = idx / JTJ_JTF_BLOCK_SIZE;

    memcpy(jtf + block_id * JTJ_JTF_BLOCK_SIZE, jtf_blocks, sizeof(float) * 640);
    memcpy(jtj + block_id * JTJ_JTF_BLOCK_SIZE * ED_COMPONENT_COUNT, jtj_diag_coo_val_blocks, sizeof(float) * 6400);
}

__global__ void kernel_jtj_coo_rows(int *jtj_rows, unsigned int ed_nodes_count)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float jtj_diag_coo_row_blocks[6400]; // 25,6 Kb

    // printf("\ned_per_thread: %u\n", ed_per_thread);

    unsigned int ed_node_offset = idx / ED_COMPONENT_COUNT;
    if(ed_node_offset >= ed_nodes_count)
    {
        // printf("\ned_node_offset overshot: %u, ed_nodes_count: %u\n", ed_node_offset, ed_nodes_count);
        return;
    }

    unsigned int block_ed_node_component_offset = idx % JTJ_JTF_BLOCK_SIZE;
    unsigned int component = idx % ED_COMPONENT_COUNT;
    unsigned int block_ed_node_offset = block_ed_node_component_offset / ED_COMPONENT_COUNT;

    // printf("\ned_node_offset: %u, ed_nodes_count: %u\n", ed_node_offset, ed_nodes_count);

    for(unsigned int component_k = 0; component_k < ED_COMPONENT_COUNT; component_k++)
    {
        unsigned int jtj_pos = block_ed_node_offset * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT + component * ED_COMPONENT_COUNT + component_k;
        jtj_diag_coo_row_blocks[jtj_pos] = ed_node_offset * ED_COMPONENT_COUNT + component_k;
    }

    // printf("\njtf[%u]\n", ed_node_offset * 10u);

    __syncthreads();

    unsigned int block_id = idx / JTJ_JTF_BLOCK_SIZE;

    memcpy(jtj_rows + block_id * JTJ_JTF_BLOCK_SIZE * ED_COMPONENT_COUNT, jtj_diag_coo_row_blocks, sizeof(float) * 6400);
}

__global__ void kernel_jtj_coo_cols(int *jtj_cols, unsigned int ed_nodes_count)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float jtj_diag_coo_col_blocks[6400]; // 25,6 Kb

    // printf("\ned_per_thread: %u\n", ed_per_thread);

    unsigned int ed_node_offset = idx / ED_COMPONENT_COUNT;
    if(ed_node_offset >= ed_nodes_count)
    {
        printf("\ned_node_offset overshot: %u, ed_nodes_count: %u\n", ed_node_offset, ed_nodes_count);
        return;
    }

    unsigned int block_ed_node_component_offset = idx % JTJ_JTF_BLOCK_SIZE;
    unsigned int component = idx % ED_COMPONENT_COUNT;
    unsigned int block_ed_node_offset = block_ed_node_component_offset / ED_COMPONENT_COUNT;

    // printf("\ned_node_offset: %u, ed_nodes_count: %u\n", ed_node_offset, ed_nodes_count);

    for(unsigned int component_k = 0; component_k < ED_COMPONENT_COUNT; component_k++)
    {
        unsigned int jtj_pos = block_ed_node_offset * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT + component * ED_COMPONENT_COUNT + component_k;
        jtj_diag_coo_col_blocks[jtj_pos] = ed_node_offset * ED_COMPONENT_COUNT + component;
    }

    // printf("\njtf[%u]\n", ed_node_offset * 10u);

    __syncthreads();

    unsigned int block_id = idx / JTJ_JTF_BLOCK_SIZE;

    memcpy(jtj_cols + block_id * JTJ_JTF_BLOCK_SIZE * ED_COMPONENT_COUNT, jtj_diag_coo_col_blocks, sizeof(float) * 6400);
}

__host__ void solve_for_h()
{
    cusparseStatus_t status;

    cusparseMatDescr_t descr = nullptr;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    getLastCudaError("cusparseCreateMatDescr failure");
    cudaDeviceSynchronize();

    int *csr_row_ptr_jtj = nullptr;
    int csr_nnz = _ed_nodes_component_count * ED_COMPONENT_COUNT;

    int N = (int)_ed_nodes_component_count;

    // printf("\nmxn: %ix%i, nnz_dev_mem: %u\n", N, N, csr_nnz);

    checkCudaErrors(cudaMalloc(&csr_row_ptr_jtj, sizeof(int) * (N + 1)));

    cudaDeviceSynchronize();
    status = cusparseXcoo2csr(cusparse_handle, _jtj_rows, csr_nnz, N, csr_row_ptr_jtj, CUSPARSE_INDEX_BASE_ZERO);
    cudaDeviceSynchronize();

    if(status != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
    {
        printf("\ncusparseStatus_t: %u\n", status);
    }

    getLastCudaError("cusparseXcoo2csr failure");

    const int max_iter = 16;
    const float tol = 1e-12f;

    float r0, r1, alpha, alpham1, beta;
    float dot;

    float a, b, na;

    alpha = 1.0f;
    alpham1 = -1.0f;
    beta = 0.f;
    r0 = 0.f;

    cudaDeviceSynchronize();
    status = cusparseScsrmv(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, csr_nnz, &alpha, descr, _jtj_vals, csr_row_ptr_jtj, _jtj_cols, _h, &beta, pcg_Ax);
    cudaDeviceSynchronize();

    if(status != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
    {
        printf("\ncusparseStatus_t: %u\n", status);
    }

    getLastCudaError("cusparseScsrmv failure");

    // TODO: use for dampening
    //    float mu = 1.0f;
    //
    //    for(unsigned int lm_step = 0; lm_step < 16; lm_step++)
    //    {
    cublasSaxpy(cublas_handle, N, &alpham1, pcg_Ax, 1, _jtf, 1);
    cublasSdot(cublas_handle, N, _jtf, 1, _jtf, 1, &r1);

    float init_res = sqrt(r1);

    printf("\ninitial residual = %e\n", sqrt(r1));

    if(isnanf(sqrt(r1)))
    {
        fprintf(stderr, "\nnan in initial residual!\n");
    }

    int k = 1;

    while(r1 > tol * tol && k <= max_iter)
    {
        if(k > 1)
        {
            b = r1 / r0;
            cublasSscal(cublas_handle, N, &b, pcg_p, 1);
            cublasSaxpy(cublas_handle, N, &alpha, _jtf, 1, pcg_p, 1);
        }
        else
        {
            cublasScopy(cublas_handle, N, _jtf, 1, pcg_p, 1);
        }

        cudaDeviceSynchronize();
        status = cusparseScsrmv(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, csr_nnz, &alpha, descr, _jtj_vals, csr_row_ptr_jtj, _jtj_cols, pcg_p, &beta, pcg_Ax);
        cudaDeviceSynchronize();

        if(status != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
        {
            printf("\ncusparseStatus_t: %u\n", status);
        }

        getLastCudaError("cusparseScsrmv failure");

        cublasSdot(cublas_handle, N, pcg_p, 1, pcg_Ax, 1, &dot);
        a = r1 / dot;

        cublasSaxpy(cublas_handle, N, &a, pcg_p, 1, _h, 1);
        na = -a;
        cublasSaxpy(cublas_handle, N, &na, pcg_Ax, 1, _jtf, 1);

        r0 = r1;
        cublasSdot(cublas_handle, N, _jtf, 1, _jtf, 1, &r1);
        k++;

        if(isnanf(sqrt(r1)))
        {
            fprintf(stderr, "\nnan in solution!\n");
        }
    }

    printf("\niteration = %3d, residual = %e\n", k, sqrt(r1));

    //        if(sqrt(r1) < init_res)
    //        {
    // sort of dampening with mu
    cublasSaxpy(cublas_handle, N, &alpha, _h, 1, (float *)_ed_graph, 1);
    //            mu -= 0.06;
    //        }
    //        else
    //        {
    //            mu += 0.06;
    //        }
    //    }

    checkCudaErrors(cudaFree(csr_row_ptr_jtj));
    cusparseDestroyMatDescr(descr);
}

extern "C" void pcg_solve()
{
    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.volume_tsdf_data, 0));
    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.buffer_vertex_counter, 0));
    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.buffer_reference_mesh_vertices, 0));
    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.buffer_occupied, 0));

    for(unsigned int i = 0; i < 4; i++)
    {
        checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.volume_cv_xyz_inv[i], 0));
    }

    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.array2d_kinect_depths, 0));

    cudaArray *volume_array_tsdf_data = nullptr;
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&volume_array_tsdf_data, _cgr.volume_tsdf_data, 0, 0));

    size_t vx_bytes;
    GLuint *vx_counter;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&vx_counter, &vx_bytes, _cgr.buffer_vertex_counter));

    struct_vertex *vx_ptr;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&vx_ptr, &vx_bytes, _cgr.buffer_reference_mesh_vertices));

    size_t brick_bytes;
    GLuint *brick_list;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&brick_list, &brick_bytes, _cgr.buffer_occupied));

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
    checkCudaErrors(cudaBindSurfaceToArray(&_volume_tsdf_data, volume_array_tsdf_data, &channel_desc));

    cudaArray *volume_array_cv_xyz_inv[4] = {nullptr, nullptr, nullptr, nullptr};
    for(unsigned int i = 0; i < 4; i++)
    {
        checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&volume_array_cv_xyz_inv[i], _cgr.volume_cv_xyz_inv[i], 0, 0));
    }
    cudaChannelFormatDesc channel_desc_cv = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    checkCudaErrors(cudaBindSurfaceToArray(&_volume_cv_xyz_inv_0, volume_array_cv_xyz_inv[0], &channel_desc_cv));
    checkCudaErrors(cudaBindSurfaceToArray(&_volume_cv_xyz_inv_1, volume_array_cv_xyz_inv[1], &channel_desc_cv));
    checkCudaErrors(cudaBindSurfaceToArray(&_volume_cv_xyz_inv_2, volume_array_cv_xyz_inv[2], &channel_desc_cv));
    checkCudaErrors(cudaBindSurfaceToArray(&_volume_cv_xyz_inv_3, volume_array_cv_xyz_inv[3], &channel_desc_cv));

    cudaArray *_array_kd = nullptr;
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&_array_kd, _cgr.array2d_kinect_depths, 0, 0));
    // cudaChannelFormatDesc channel_desc_kd = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    // checkCudaErrors(cudaBindSurfaceToArray(&_array2d_kinect_depths, _array_kd, &channel_desc_kd));

    size_t grid_size = (_ed_nodes_component_count + JTJ_JTF_BLOCK_SIZE - 1) / JTJ_JTF_BLOCK_SIZE;
    kernel_jtj_jtf<<<grid_size, JTJ_JTF_BLOCK_SIZE>>>(_jtj_vals, _jtf, vx_counter, vx_ptr, _ed_nodes_count, _ed_graph);

    getLastCudaError("render kernel failed");

    cudaDeviceSynchronize();

    //    kernel_jtj_coo_rows<<<grid_size, JTJ_JTF_BLOCK_SIZE>>>(_jtj_rows, _ed_nodes_count);
    //
    //    getLastCudaError("render kernel failed");
    //
    //    cudaDeviceSynchronize();
    //
    //    kernel_jtj_coo_cols<<<grid_size, JTJ_JTF_BLOCK_SIZE>>>(_jtj_cols, _ed_nodes_count);
    //
    //    getLastCudaError("render kernel failed");
    //
    //    cudaDeviceSynchronize();

    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.array2d_kinect_depths, 0));

    for(unsigned int i = 0; i < 4; i++)
    {
        checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.volume_cv_xyz_inv[i], 0));
    }

    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.volume_tsdf_data, 0));

    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.buffer_occupied, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.buffer_reference_mesh_vertices, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.buffer_vertex_counter, 0));

    cudaDeviceSynchronize();

    solve_for_h();

    getLastCudaError("render kernel failed");

    cudaDeviceSynchronize();
}