#include <reconstruction/cuda/resources.cuh>

#define EVALUATE_DATA
#define EVALUATE_VISUAL_HULL
// #define EVALUATE_ED_REGULARIZATION

// #define ED_NODES_ROBUSTIFY
#define FAST_QUAT_OPS

// #define DEBUG_JTJ
// #define DEBUG_JTF
// #define DEBUG_H
#define SOLVER_DIRECT_CHOL
// #define SOLVER_DIRECT_QR
// #define SOLVER_PCG

#ifdef DEBUG_JTJ

#include "../../../external/csv/ostream.hpp"

#endif

#ifdef DEBUG_JTF

#include "../../../external/csv/ostream.hpp"

#endif

#ifdef DEBUG_H

#include "../../../external/csv/ostream.hpp"

#endif

__global__ void kernel_reject_misaligned_deformations(unsigned int active_ed_nodes_count, struct_device_resources dev_res, struct_measures measures)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= active_ed_nodes_count)
    {
        // printf("\ned_node_offset overshot: %u, ed_nodes_count: %u\n", ed_node_offset, ed_nodes_count);
        return;
    }

    struct_ed_node ed_node = dev_res.ed_graph[idx];
    struct_ed_meta_entry ed_entry = dev_res.ed_graph_meta[idx];

    // printf("\ned_node position: (%f,%f,%f)\n", ed_node.position.x, ed_node.position.y, ed_node.position.z);

    float energy = 0.f;

    for(unsigned int vx_idx = 0; vx_idx < ed_entry.vx_length; vx_idx++)
    {
        struct_vertex vx = dev_res.sorted_vx_ptr[ed_entry.vx_offset + vx_idx];

        // printf("\ned_node + vertex match\n");

        float vx_misalignment = glm::min(evaluate_vx_misalignment(vx, ed_node, measures), evaluate_hull_residual(vx, ed_node, dev_res.kinect_silhouettes, measures));

        // printf("\nvx_residual: %f\n", vx_residual);

        if(isnan(vx_misalignment))
        {
#ifdef DEBUG_NANS
            printf("\nvx_residual is NaN!\n");
#endif

            vx_misalignment = 0.f;
        }

        energy += vx_misalignment;
    }

    energy /= (float)ed_entry.vx_length;

    ed_entry.rejected = energy > 0.03f; // TODO: figure out threshold

    // printf("\nenergy: %f\n", energy);
}

__global__ void kernel_step_energy(float *energy, unsigned int active_ed_nodes_count, struct_ed_node *step_ed_graph, struct_device_resources dev_res, struct_measures measures)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= active_ed_nodes_count)
    {
        // printf("\ned_node_offset overshot: %u, ed_nodes_count: %u\n", ed_node_offset, ed_nodes_count);
        return;
    }

    struct_ed_meta_entry ed_entry = dev_res.ed_graph_meta[idx];
    struct_ed_node ed_node = step_ed_graph[idx];

#ifdef EVALUATE_ED_REGULARIZATION
    float ed_residual = evaluate_ed_node_residual(ed_node, ed_entry, dev_res, measures);

    if(isnan(ed_residual))
    {
        printf("\ned_residual is NaN!\n");

        ed_residual = 0.f;
    }

    atomicAdd(energy, ed_residual);
#endif

    // printf("\ned_node position: (%f,%f,%f)\n", ed_node.position.x, ed_node.position.y, ed_node.position.z);

    for(unsigned int vx_idx = 0; vx_idx < ed_entry.vx_length; vx_idx++)
    {
        struct_vertex vx = dev_res.sorted_vx_ptr[ed_entry.vx_offset + vx_idx];

        // printf("\ned_node + vertex match\n");

        float vx_residual = 0.f;
#ifdef EVALUATE_DATA
        vx_residual += evaluate_vx_residual(vx, ed_node, ed_node, dev_res.kinect_depths, measures);
#endif

#ifdef EVALUATE_VISUAL_HULL
        vx_residual += evaluate_hull_residual(vx, ed_node, dev_res.kinect_silhouettes, measures);
#endif

        // printf("\nvx_residual: %f\n", vx_residual);

        if(isnan(vx_residual))
        {
            printf("\nvx_residual is NaN!\n");

            vx_residual = 0.f;
        }

        atomicAdd(energy, vx_residual);
    }

    // printf("\njtf[%u]\n", ed_node_offset * 10u);
}

__global__ void kernel_energy(float *energy, unsigned int active_ed_nodes_count, struct_device_resources dev_res, struct_measures measures)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= active_ed_nodes_count)
    {
        // printf("\ned_node_offset overshot: %u, ed_nodes_count: %u\n", ed_node_offset, ed_nodes_count);
        return;
    }

    struct_ed_meta_entry ed_entry = dev_res.ed_graph_meta[idx];
    struct_ed_node ed_node = dev_res.ed_graph[idx];

#ifdef EVALUATE_ED_REGULARIZATION
    float ed_residual = evaluate_ed_node_residual(ed_node, ed_entry, dev_res, measures);

    if(isnan(ed_residual))
    {
        printf("\ned_residual is NaN!\n");

        ed_residual = 0.f;
    }

    atomicAdd(energy, ed_residual);
#endif

    // printf("\ned_node position: (%f,%f,%f)\n", ed_node.position.x, ed_node.position.y, ed_node.position.z);

    for(unsigned int vx_idx = 0; vx_idx < ed_entry.vx_length; vx_idx++)
    {
        struct_vertex vx = dev_res.sorted_vx_ptr[ed_entry.vx_offset + vx_idx];

        // printf("\ned_node + vertex match\n");

        float vx_residual = 0.f;
#ifdef EVALUATE_DATA
        vx_residual += evaluate_vx_residual(vx, ed_node, ed_node, dev_res.kinect_depths, measures);
#endif

#ifdef EVALUATE_VISUAL_HULL
        vx_residual += evaluate_hull_residual(vx, ed_node, dev_res.kinect_silhouettes, measures);
#endif

        // printf("\nvx_residual: %f\n", vx_residual);

        if(isnan(vx_residual))
        {
            printf("\nvx_residual is NaN!\n");

            vx_residual = 0.f;
        }

        atomicAdd(energy, vx_residual);
    }

    // printf("\njtf[%u]\n", ed_node_offset * 10u);
}

__global__ void kernel_jtj_jtf(unsigned long long int active_ed_vx_count, unsigned int active_ed_nodes_count, const float mu, struct_device_resources dev_res, struct_measures measures)
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

    struct_ed_meta_entry ed_entry = dev_res.ed_graph_meta[ed_node_offset];
    struct_ed_node ed_node = dev_res.ed_graph[ed_node_offset];

    unsigned int component = idx % ED_COMPONENT_COUNT;

#ifdef EVALUATE_ED_REGULARIZATION
    __shared__ float ed_pds[ED_COMPONENT_COUNT];

    float ed_residual = evaluate_ed_node_residual(ed_node, ed_entry, dev_res, measures);

    if(isnan(ed_residual))
    {
#ifdef DEBUG_NANS
        printf("\ned_residual is NaN!\n");
#endif

        ed_residual = 0.f;
    }

    ed_pds[component] = evaluate_ed_pd(ed_node, ed_entry, component, ed_residual, dev_res, measures);

    if(isnan(ed_pds[component]))
    {
#ifdef DEBUG_NANS
        printf("\ned_pds[%u] is NaN!\n", component);
#endif

        ed_pds[component] = 0.f;
    }

    __syncthreads();

    float jtf_value = -ed_pds[component] * ed_residual;

    if(isnan(jtf_value))
    {
#ifdef DEBUG_NANS
        printf("\njtf_value[%u] is NaN!\n", component);
#endif

        jtf_value = 0.f;
    }

    shared_jtf_block[component] = jtf_value;

    for(unsigned int component_k = 0; component_k < ED_COMPONENT_COUNT; component_k++)
    {
        unsigned int jtj_pos = component * ED_COMPONENT_COUNT + component_k;

        float jtj_value = ed_pds[component] * ed_pds[component_k];

        if(isnan(jtj_value))
        {
#ifdef DEBUG_NANS
            printf("\njtj_value[%u] is NaN!\n", component);
#endif

            jtj_value = 0.f;
        }

        shared_jtj_coo_val_block[jtj_pos] = jtj_value;
    }

#endif

    __syncthreads();

    // printf("\ned_node position: (%f,%f,%f)\n", ed_node.position.x, ed_node.position.y, ed_node.position.z);

    for(unsigned int vx_idx = 0; vx_idx < ed_entry.vx_length; vx_idx++)
    {
        struct_vertex vx = dev_res.sorted_vx_ptr[ed_entry.vx_offset + vx_idx];

        // printf("\ned_node + vertex match\n");

        float vx_residual = 0.f;
#ifdef EVALUATE_DATA
        vx_residual += evaluate_vx_residual(vx, ed_node, ed_node, dev_res.kinect_depths, measures);
#endif

#ifdef EVALUATE_VISUAL_HULL
        vx_residual += evaluate_hull_residual(vx, ed_node, dev_res.kinect_silhouettes, measures);
#endif

        // printf("\nvx_residual: %f\n", vx_residual);

        if(isnan(vx_residual))
        {
#ifdef DEBUG_NANS
            printf("\nvx_residual is NaN!\n");
#endif

            vx_residual = 0.f;
        }

        __shared__ float pds[ED_COMPONENT_COUNT];

        pds[component] = 0.f;

#ifdef EVALUATE_DATA
        pds[component] += evaluate_vx_pd(vx, ed_node, dev_res.ed_graph[idx], component, vx_residual, dev_res.kinect_depths, measures);
#endif

#ifdef EVALUATE_VISUAL_HULL
        pds[component] += evaluate_hull_pd(vx, ed_node, component, vx_residual, dev_res.kinect_silhouettes, measures);
#endif

        if(isnan(pds[component]))
        {
#ifdef DEBUG_NANS
            printf("\nvx_pds[%u] is NaN!\n", component);
#endif

            pds[component] = 0.f;
        }

        __syncthreads();

        //        if(idx == 0)
        //        {
        //            for(int i = 0; i < ED_COMPONENT_COUNT; i++)
        //            {
        //                printf("\nvx_pds[%u]: %f\n", i, vx_pds[i]);
        //            }
        //        }

        float jtf_value = -pds[component] * vx_residual;

        if(isnan(jtf_value))
        {
#ifdef DEBUG_NANS
            printf("\njtf_value[%u] is NaN!\n", component);
#endif

            jtf_value = 0.f;
        }

        shared_jtf_block[component] += jtf_value;

        for(unsigned int component_k = 0; component_k < ED_COMPONENT_COUNT; component_k++)
        {
            unsigned int jtj_pos = component * ED_COMPONENT_COUNT + component_k;

            float jtj_value = pds[component] * pds[component_k];

            if(isnan(jtj_value))
            {
#ifdef DEBUG_NANS
                printf("\njtj_value[%u] is NaN!\n", component);
#endif

                jtj_value = 0.f;
            }

            atomicAdd(&shared_jtj_coo_val_block[jtj_pos], jtj_value);
        }

        __syncthreads();
    }

    // printf("\njtf[%u]\n", ed_node_offset * 10u);

    for(unsigned int component_k = 0; component_k < ED_COMPONENT_COUNT; component_k++)
    {
        unsigned int jtj_pos = component * ED_COMPONENT_COUNT + component_k;

        if(component == component_k)
        {
            atomicAdd(&shared_jtj_coo_val_block[jtj_pos], mu);
        }
    }

    __syncthreads();

    //    if(idx == 0)
    //    {
    //        for(int i = 0; i < ED_COMPONENT_COUNT; i++)
    //        {
    //            printf("\nshared_jtf_block[%u]: %f\n", i, shared_jtf_block[i]);
    //        }
    //    }

    memcpy(&dev_res.jtf[ed_node_offset * ED_COMPONENT_COUNT], &shared_jtf_block[0], sizeof(float) * ED_COMPONENT_COUNT);
    memcpy(&dev_res.jtj_vals[ed_node_offset * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT], &shared_jtj_coo_val_block[0], sizeof(float) * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT);
}

__global__ void kernel_jtj_coo_rows(unsigned int active_ed_nodes_count, struct_device_resources dev_res, struct_measures measures)
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

    memcpy(&dev_res.jtj_rows[ed_node_offset * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT], &jtj_diag_coo_row_blocks[0], sizeof(int) * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT);
}

__global__ void kernel_jtj_coo_cols(unsigned int ed_nodes_count, struct_device_resources dev_res, struct_measures measures)
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

    memcpy(&dev_res.jtj_cols[ed_node_offset * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT], &jtj_diag_coo_col_blocks[0], sizeof(int) * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT);
}

#ifdef SOLVER_DIRECT_CHOL
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
    int csr_nnz = _host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT;

    int N = (int)_host_res.active_ed_nodes_count * ED_COMPONENT_COUNT;

    // printf("\nmxn: %ix%i, nnz_dev_mem: %u\n", N, N, csr_nnz);

    checkCudaErrors(cudaMalloc(&csr_row_ptr_jtj, sizeof(int) * (N + 1)));

    cudaDeviceSynchronize();
    status = cusparseXcoo2csr(cusparse_handle, _dev_res.jtj_rows, csr_nnz, N, csr_row_ptr_jtj, CUSPARSE_INDEX_BASE_ZERO);
    cudaDeviceSynchronize();

    if(status != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
    {
        printf("\ncusparseStatus_t: %u\n", status);
    }

    getLastCudaError("cusparseXcoo2csr failure");

    float tol = 1.e-6f;
    const int reorder = 0;
    int singularity = 0;

    cudaDeviceSynchronize();

    cusolverStatus_t solver_status =
        cusolverSpScsrlsvchol(cusolver_handle, N, csr_nnz, descr, _dev_res.jtj_vals, csr_row_ptr_jtj, _dev_res.jtj_cols, _dev_res.jtf, tol, reorder, _dev_res.h, &singularity);

    cudaDeviceSynchronize();

    // printf("\nsingularity: %i\n", singularity);

    if(solver_status != cusolverStatus_t::CUSOLVER_STATUS_SUCCESS)
    {
        printf("\ncusolverStatus_t: %u\n", solver_status);
    }

    getLastCudaError("cusolverSpScsrlsvchol failure");

    checkCudaErrors(cudaFree(csr_row_ptr_jtj));
    cusparseDestroyMatDescr(descr);
}
#endif

#ifdef SOLVER_DIRECT_QR
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

    float tol = 1.e-6f;
    const int reorder = 0;
    int singularity = 0;

    float *residual = NULL;
    checkCudaErrors(cudaMalloc((void **)&residual, sizeof(float) * N));

    cudaDeviceSynchronize();

    cusolverStatus_t solver_status = cusolverSpScsrlsvqr(cusolver_handle, N, csr_nnz, descr, _jtj_vals, csr_row_ptr_jtj, _jtj_cols, _jtf, tol, reorder, _h, &singularity);

    cudaDeviceSynchronize();

    // printf("\nsingularity: %i\n", singularity);

    if(solver_status != cusolverStatus_t::CUSOLVER_STATUS_SUCCESS)
    {
        printf("\ncusolverStatus_t: %u\n", solver_status);
    }

    getLastCudaError("cusolverSpScsrlsvlu failure");

    checkCudaErrors(cudaFree(csr_row_ptr_jtj));
    cusparseDestroyMatDescr(descr);
}
#endif

#ifdef SOLVER_PCG
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

    checkCudaErrors(cudaFree(csr_row_ptr_jtj));
    cusparseDestroyMatDescr(descr);
}
#endif

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

#ifdef DEBUG_JTF

__host__ void print_out_jtf()
{
    float *host_jtf;

    host_jtf = (float *)malloc(_active_ed_nodes_count * ED_COMPONENT_COUNT * sizeof(float));

    cudaMemcpy(&host_jtf[0], &_jtf[0], _active_ed_nodes_count * ED_COMPONENT_COUNT * sizeof(float), cudaMemcpyDeviceToHost);

    std::ofstream fs("jtf_" + std::to_string(rand()) + ".csv");
    text::csv::csv_ostream csvs(fs);

    for(int i = 0; i < _active_ed_nodes_count * ED_COMPONENT_COUNT; i++)
    {
        csvs << host_jtf[i];
    }
    csvs << text::csv::endl;

    fs.close();

    free(host_jtf);
}

#endif

#ifdef DEBUG_H

__host__ void print_out_h()
{
    float *host_h;

    host_h = (float *)malloc(_active_ed_nodes_count * ED_COMPONENT_COUNT * sizeof(float));

    cudaMemcpy(&host_h[0], &_h[0], _active_ed_nodes_count * ED_COMPONENT_COUNT * sizeof(float), cudaMemcpyDeviceToHost);

    std::ofstream fs("h_" + std::to_string(rand()) + ".csv");
    text::csv::csv_ostream csvs(fs);

    for(int i = 0; i < _active_ed_nodes_count * ED_COMPONENT_COUNT; i++)
    {
        csvs << host_h[i];
    }
    csvs << text::csv::endl;

    fs.close();

    free(host_h);
}

#endif

void evaluate_jtj_jtf(const float mu)
{
    size_t grid_size = (_host_res.active_ed_nodes_count * ED_COMPONENT_COUNT + ED_COMPONENT_COUNT - 1) / ED_COMPONENT_COUNT;
    kernel_jtj_jtf<<<grid_size, ED_COMPONENT_COUNT>>>(_host_res.active_ed_vx_count, _host_res.active_ed_nodes_count, mu, _dev_res, _host_res.measures);
    getLastCudaError("render kernel failed");
    cudaDeviceSynchronize();

    kernel_jtj_coo_rows<<<grid_size, ED_COMPONENT_COUNT>>>(_host_res.active_ed_nodes_count, _dev_res, _host_res.measures);
    getLastCudaError("render kernel failed");
    cudaDeviceSynchronize();

    kernel_jtj_coo_cols<<<grid_size, ED_COMPONENT_COUNT>>>(_host_res.active_ed_nodes_count, _dev_res, _host_res.measures);
    getLastCudaError("render kernel failed");
    cudaDeviceSynchronize();
};

void evaluate_misalignment_energy(float &misalignment_energy)
{
    float *device_misalignment_energy = nullptr;
    cudaMallocManaged(&device_misalignment_energy, sizeof(float));
    *device_misalignment_energy = 0.f;

    int block_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_energy, 0, 0);
    size_t grid_size = (_host_res.active_ed_nodes_count + block_size - 1) / block_size;
    kernel_energy<<<grid_size, block_size>>>(device_misalignment_energy, _host_res.active_ed_nodes_count, _dev_res, _host_res.measures);
    getLastCudaError("render kernel failed");
    cudaDeviceSynchronize();

    cudaMemcpy(&misalignment_energy, &device_misalignment_energy[0], sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(device_misalignment_energy);
}

void evaluate_step_misalignment_energy(float &misalignment_energy)
{
    float *device_misalignment_energy = nullptr;
    cudaMallocManaged(&device_misalignment_energy, sizeof(float));
    *device_misalignment_energy = 0.f;

    struct_ed_node *step_ed_graph = nullptr;
    cudaMalloc(&step_ed_graph, _host_res.active_ed_nodes_count * sizeof(struct_ed_node));
    cudaMemcpy(step_ed_graph, _dev_res.ed_graph, _host_res.active_ed_nodes_count * sizeof(struct_ed_node), cudaMemcpyDeviceToDevice);

    cudaDeviceSynchronize();

    int N = (int)_host_res.active_ed_nodes_count * ED_COMPONENT_COUNT;
    const float one = 1.0f;
    cublasSaxpy(cublas_handle, N, &one, _dev_res.h, 1, (float *)&step_ed_graph[0], 1);

    cudaDeviceSynchronize();

    int block_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_step_energy, 0, 0);
    size_t grid_size = (_host_res.active_ed_nodes_count + block_size - 1) / block_size;
    kernel_step_energy<<<grid_size, block_size>>>(device_misalignment_energy, _host_res.active_ed_nodes_count, step_ed_graph, _dev_res, _host_res.measures);
    getLastCudaError("render kernel failed");
    cudaDeviceSynchronize();

    cudaMemcpy(&misalignment_energy, &device_misalignment_energy[0], sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(step_ed_graph);
    cudaFree(device_misalignment_energy);
}

void reject_misaligned_deformations()
{
    int block_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_reject_misaligned_deformations, 0, 0);
    size_t grid_size = (_host_res.active_ed_nodes_count + block_size - 1) / block_size;
    kernel_reject_misaligned_deformations<<<grid_size, block_size>>>(_host_res.active_ed_nodes_count, _dev_res, _host_res.measures);
    getLastCudaError("render kernel failed");
    cudaDeviceSynchronize();
}

extern "C" void pcg_solve(struct_native_handles &native_handles)
{
    map_calibration_volumes();
    map_tsdf_volumes();

    const unsigned int max_iterations = 2u;
    unsigned int iterations = 0u;
    float mu = 0.5f;
    float initial_misalignment_energy, solution_misalignment_energy;

    evaluate_misalignment_energy(initial_misalignment_energy);
    cudaDeviceSynchronize();

    while(iterations < max_iterations)
    {
        evaluate_jtj_jtf(mu);

        cudaDeviceSynchronize();

        solve_for_h();
        cudaDeviceSynchronize();

        evaluate_step_misalignment_energy(solution_misalignment_energy);
        cudaDeviceSynchronize();

        if(solution_misalignment_energy < initial_misalignment_energy && (unsigned int)(solution_misalignment_energy * 1000) != 0u)
        {
            printf("\naccepted step, initial E: % f, solution E: %f\n", initial_misalignment_energy, solution_misalignment_energy);

            int N = (int)_host_res.active_ed_nodes_count * ED_COMPONENT_COUNT;
            const float one = 1.0f;
            cublasSaxpy(cublas_handle, N, &one, _dev_res.h, 1, (float *)&_dev_res.ed_graph[0], 1);
            cudaDeviceSynchronize();

            mu -= 0.05f;
            initial_misalignment_energy = solution_misalignment_energy;
            printf("\nmu lowered: %f\n", mu);
        }
        else
        {
            printf("\nrejected step, initial E: % f, solution E: %f\n", initial_misalignment_energy, solution_misalignment_energy);

            mu += 0.05f;
            printf("\nmu raised: %f\n", mu);
        }

        iterations++;
    }

    reject_misaligned_deformations();

#ifdef DEBUG_JTJ

    print_out_jtj();

#endif

#ifdef DEBUG_JTF

    print_out_jtf();

#endif

#ifdef DEBUG_H

    print_out_h();

#endif

    unmap_calibration_volumes();
    unmap_tsdf_volumes();
}