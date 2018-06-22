#include <reconstruction/cuda/resources.cuh>

// #define DEBUG_JTJ

#ifdef DEBUG_JTJ

#include "../../../external/csv/ostream.hpp"

#endif

__global__ void kernel_jtj_jtf(float *jtj, float *jtf, unsigned long long int active_ed_vx_count, struct_vertex *sorted_vx_ptr, unsigned int active_ed_nodes_count,
                               struct_ed_dense_index_entry *ed_dense_index, struct_ed_node *ed_graph, struct_measures *measures)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float shared_jtj_coo_val_block[ED_COMPONENT_COUNT * ED_COMPONENT_COUNT];
    __shared__ float shared_jtf_block[ED_COMPONENT_COUNT];

    unsigned int ed_node_offset = idx / ED_COMPONENT_COUNT;
    if(ed_node_offset >= active_ed_nodes_count)
    {
        // printf("\ned_node_offset overshot: %u, ed_nodes_count: %u\n", ed_node_offset, ed_nodes_count);
        return;
    }

    struct_ed_node ed_node = ed_graph[ed_node_offset];
    struct_ed_dense_index_entry ed_entry = ed_dense_index[ed_node_offset];

    unsigned int component = idx % ED_COMPONENT_COUNT;

    // printf("\ned_node position: (%f,%f,%f)\n", ed_node.position.x, ed_node.position.y, ed_node.position.z);

    for(unsigned int vx_idx = 0; vx_idx < ed_entry.vx_length; vx_idx++)
    {
        struct_vertex vx = sorted_vx_ptr[ed_entry.vx_offset + vx_idx];

        // printf("\ned_node + vertex match\n");

        float vx_residual = evaluate_vx_residual(vx, ed_node, measures);

        // printf("\nvx_residual: %f\n", vx_residual);

        if(isnan(vx_residual))
        {
            printf("\nvx_residual is NaN!\n");

            vx_residual = 0.f;
        }

        __shared__ float vx_pds[ED_COMPONENT_COUNT];

        vx_pds[component] = evaluate_vx_pd(vx, ed_node, component, vx_residual, measures);

        if(isnan(vx_pds[component]))
        {
            printf("\nvx_pds[%u] is NaN!\n", component);

            vx_pds[component] = 0.f;
        }

        __syncthreads();

        //        if(idx == 0)
        //        {
        //            for(int i = 0; i < ED_COMPONENT_COUNT; i++)
        //            {
        //                printf("\nvx_pds[%u]: %f\n", i, vx_pds[i]);
        //            }
        //        }

        float jtf_value = vx_pds[component] * vx_residual;

        if(isnan(jtf_value))
        {
            printf("\njtf_value[%u] is NaN!\n", component);

            jtf_value = 0.f;
        }

        shared_jtf_block[component] += jtf_value;

        for(unsigned int component_k = 0; component_k < ED_COMPONENT_COUNT; component_k++)
        {
            unsigned int jtj_pos = component * ED_COMPONENT_COUNT + component_k;

            float jtj_value = vx_pds[component] * vx_pds[component_k];

            if(isnan(jtj_value))
            {
                printf("\njtj_value[%u] is NaN!\n", component);

                jtj_value = 0.f;
            }

            shared_jtj_coo_val_block[jtj_pos] += jtj_value;
        }

        __syncthreads();
    }

    // printf("\njtf[%u]\n", ed_node_offset * 10u);

    __syncthreads();

    //    if(idx == 0)
    //    {
    //        for(int i = 0; i < ED_COMPONENT_COUNT; i++)
    //        {
    //            printf("\nshared_jtf_block[%u]: %f\n", i, shared_jtf_block[i]);
    //        }
    //    }

    memcpy(&jtf[ed_node_offset * ED_COMPONENT_COUNT], &shared_jtf_block[0], sizeof(float) * ED_COMPONENT_COUNT);
    memcpy(&jtj[ed_node_offset * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT], &shared_jtj_coo_val_block[0], sizeof(float) * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT);
}

__global__ void kernel_jtj_coo_rows(int *jtj_rows, unsigned int active_ed_nodes_count)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ int jtj_diag_coo_row_blocks[ED_COMPONENT_COUNT * ED_COMPONENT_COUNT];

    // printf("\ned_per_thread: %u\n", ed_per_thread);

    unsigned int ed_node_offset = idx / ED_COMPONENT_COUNT;
    if(ed_node_offset >= active_ed_nodes_count)
    {
        // printf("\ned_node_offset overshot: %u, ed_nodes_count: %u\n", ed_node_offset, ed_nodes_count);
        return;
    }

    unsigned int component = idx % ED_COMPONENT_COUNT;

    // printf("\ned_node_offset: %u, ed_nodes_count: %u\n", ed_node_offset, ed_nodes_count);

    for(unsigned int component_k = 0; component_k < ED_COMPONENT_COUNT; component_k++)
    {
        unsigned int jtj_pos = component * ED_COMPONENT_COUNT + component_k;
        jtj_diag_coo_row_blocks[jtj_pos] = ed_node_offset * ED_COMPONENT_COUNT + component_k;
    }

    // printf("\njtf[%u]\n", ed_node_offset * 10u);

    __syncthreads();

    memcpy(&jtj_rows[ed_node_offset * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT], &jtj_diag_coo_row_blocks[0], sizeof(int) * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT);
}

__global__ void kernel_jtj_coo_cols(int *jtj_cols, unsigned int ed_nodes_count)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ int jtj_diag_coo_col_blocks[ED_COMPONENT_COUNT * ED_COMPONENT_COUNT];

    // printf("\ned_per_thread: %u\n", ed_per_thread);

    unsigned int ed_node_offset = idx / ED_COMPONENT_COUNT;
    if(ed_node_offset >= ed_nodes_count)
    {
        // printf("\ned_node_offset overshot: %u, ed_nodes_count: %u\n", ed_node_offset, ed_nodes_count);
        return;
    }

    unsigned int component = idx % ED_COMPONENT_COUNT;

    // printf("\ned_node_offset: %u, ed_nodes_count: %u\n", ed_node_offset, ed_nodes_count);

    for(unsigned int component_k = 0; component_k < ED_COMPONENT_COUNT; component_k++)
    {
        unsigned int jtj_pos = component * ED_COMPONENT_COUNT + component_k;
        jtj_diag_coo_col_blocks[jtj_pos] = ed_node_offset * ED_COMPONENT_COUNT + component;
    }

    // printf("\njtf[%u]\n", ed_node_offset * 10u);

    __syncthreads();

    memcpy(&jtj_cols[ed_node_offset * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT], &jtj_diag_coo_col_blocks[0], sizeof(int) * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT);
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
    int csr_nnz = _active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT;

    int N = (int)_active_ed_nodes_count * ED_COMPONENT_COUNT;

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

#ifdef DEBUG_JTJ

__host__ void print_out_jtj()
{
    float *host_jtj_vals;
    int *host_jtj_rows;
    int *host_jtj_cols;

    host_jtj_vals = (float *)malloc(_active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT * sizeof(float));
    host_jtj_rows = (int *)malloc(_active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT * sizeof(int));
    host_jtj_cols = (int *)malloc(_active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT * sizeof(int));

    cudaMemcpy(&host_jtj_vals[0], &_jtj_vals[0], _active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_jtj_rows[0], &_jtj_rows[0], _active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_jtj_cols[0], &_jtj_cols[0], _active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT * sizeof(int), cudaMemcpyDeviceToHost);

    std::ofstream fs("jtj_" + std::to_string(rand()) + ".csv");
    text::csv::csv_ostream csvs(fs);

    for(int i = 0; i < _active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT; i++)
    {
        csvs << host_jtj_vals[i];
    }
    csvs << text::csv::endl;

    for(int i = 0; i < _active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT; i++)
    {
        csvs << host_jtj_rows[i];
    }
    csvs << text::csv::endl;

    for(int i = 0; i < _active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT; i++)
    {
        csvs << host_jtj_cols[i];
    }
    csvs << text::csv::endl;

    fs.close();

    free(host_jtj_vals);
    free(host_jtj_rows);
    free(host_jtj_cols);
}

#endif

extern "C" void pcg_solve()
{
    for(unsigned int i = 0; i < 4; i++)
    {
        checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.volume_cv_xyz_inv[i], 0));
        checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.volume_cv_xyz[i], 0));
    }

    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.array2d_kinect_depths, 0));

    cudaArray *array_kd[4] = {nullptr, nullptr, nullptr, nullptr};
    cudaArray *volume_array_cv_xyz_inv[4] = {nullptr, nullptr, nullptr, nullptr};
    cudaArray *volume_array_cv_xyz[4] = {nullptr, nullptr, nullptr, nullptr};

    for(unsigned int i = 0; i < 4; i++)
    {
        checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&volume_array_cv_xyz_inv[i], _cgr.volume_cv_xyz_inv[i], 0, 0));
        checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&volume_array_cv_xyz[i], _cgr.volume_cv_xyz[i], 0, 0));
        checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&array_kd[i], _cgr.array2d_kinect_depths, i, 0));
    }

    cudaChannelFormatDesc channel_desc_cv_xyz_inv = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    checkCudaErrors(cudaBindSurfaceToArray(&_volume_cv_xyz_inv_0, volume_array_cv_xyz_inv[0], &channel_desc_cv_xyz_inv));
    checkCudaErrors(cudaBindSurfaceToArray(&_volume_cv_xyz_inv_1, volume_array_cv_xyz_inv[1], &channel_desc_cv_xyz_inv));
    checkCudaErrors(cudaBindSurfaceToArray(&_volume_cv_xyz_inv_2, volume_array_cv_xyz_inv[2], &channel_desc_cv_xyz_inv));
    checkCudaErrors(cudaBindSurfaceToArray(&_volume_cv_xyz_inv_3, volume_array_cv_xyz_inv[3], &channel_desc_cv_xyz_inv));

    cudaChannelFormatDesc channel_desc_cv_xyz = cudaCreateChannelDesc(32, 32, 32, 0, cudaChannelFormatKindFloat);
    checkCudaErrors(cudaBindSurfaceToArray(&_volume_cv_xyz_0, volume_array_cv_xyz[0], &channel_desc_cv_xyz));
    checkCudaErrors(cudaBindSurfaceToArray(&_volume_cv_xyz_1, volume_array_cv_xyz[1], &channel_desc_cv_xyz));
    checkCudaErrors(cudaBindSurfaceToArray(&_volume_cv_xyz_2, volume_array_cv_xyz[2], &channel_desc_cv_xyz));
    checkCudaErrors(cudaBindSurfaceToArray(&_volume_cv_xyz_3, volume_array_cv_xyz[3], &channel_desc_cv_xyz));

    cudaChannelFormatDesc channel_desc_kd = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    checkCudaErrors(cudaBindSurfaceToArray(&_array2d_kinect_depths_0, array_kd[0], &channel_desc_kd));
    checkCudaErrors(cudaBindSurfaceToArray(&_array2d_kinect_depths_1, array_kd[1], &channel_desc_kd));
    checkCudaErrors(cudaBindSurfaceToArray(&_array2d_kinect_depths_2, array_kd[2], &channel_desc_kd));
    checkCudaErrors(cudaBindSurfaceToArray(&_array2d_kinect_depths_3, array_kd[3], &channel_desc_kd));

    size_t grid_size = (_active_ed_nodes_count * ED_COMPONENT_COUNT + ED_COMPONENT_COUNT - 1) / ED_COMPONENT_COUNT;
    kernel_jtj_jtf<<<grid_size, ED_COMPONENT_COUNT>>>(_jtj_vals, _jtf, _active_ed_vx_count, _sorted_vx_ptr, _active_ed_nodes_count, _ed_nodes_dense_index, _ed_graph, _measures);

    getLastCudaError("render kernel failed");

    cudaDeviceSynchronize();

    kernel_jtj_coo_rows<<<grid_size, ED_COMPONENT_COUNT>>>(_jtj_rows, _active_ed_nodes_count);

    getLastCudaError("render kernel failed");

    cudaDeviceSynchronize();

    kernel_jtj_coo_cols<<<grid_size, ED_COMPONENT_COUNT>>>(_jtj_cols, _active_ed_nodes_count);

    getLastCudaError("render kernel failed");

    cudaDeviceSynchronize();

    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.array2d_kinect_depths, 0));

    for(unsigned int i = 0; i < 4; i++)
    {
        checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.volume_cv_xyz[i], 0));
        checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.volume_cv_xyz_inv[i], 0));
    }

    cudaDeviceSynchronize();

    solve_for_h();

#ifdef DEBUG_JTJ

    print_out_jtj();

#endif

    getLastCudaError("render kernel failed");

    cudaDeviceSynchronize();
}