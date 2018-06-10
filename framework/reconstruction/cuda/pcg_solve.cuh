#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <reconstruction/cuda/glm.cuh>
#include <reconstruction/cuda/resources.cuh>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>

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
    const float skinning_weight = expf(glm::length(dist) * glm::length(dist) * 2 / (ED_CELL_VOXEL_DIM * ED_CELL_VOXEL_DIM));

    glm::vec3 warped_position = warp_position(dist, ed_node, skinning_weight);
    glm::vec3 warped_normal = warp_normal(vertex.normal, ed_node, skinning_weight);

    float residuals = 0.f;

    for(int i = 0; i < 4; i++)
    {
        glm::uvec3 wp_voxel_space = glm::uvec3(warped_position);
        // printf("\n (x,y,z): (%u,%u,%u)\n", wp_voxel_space.x, wp_voxel_space.y, wp_voxel_space.z);

        if(wp_voxel_space.x >= VOLUME_VOXEL_DIM || wp_voxel_space.y >= VOLUME_VOXEL_DIM || wp_voxel_space.z >= VOLUME_VOXEL_DIM)
        {
            // TODO: warped out of volume!
            continue;
        }

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

        residuals += glm::abs(glm::dot(warped_normal, glm::vec3(wp_voxel_space) - extracted_position));
    }

    return residuals;
}

__device__ float evaluate_vx_pd(struct_vertex &vertex, struct_ed_node &ed_node, const int &partial_derivative_index, const float &vx_residual)
{
    float *mapped_ed_node = (float *)&ed_node;

    mapped_ed_node[partial_derivative_index] += 0.001f;

    float residual_pos = evaluate_vx_residual(vertex, ed_node);

    mapped_ed_node[partial_derivative_index] -= 0.001f;

    return (residual_pos - vx_residual) / 0.002f;
}

__device__ float *evaluate_ed_node_residuals(struct_ed_node &ed_node)
{
    float *residuals = new float[2];

    glm::mat3 mat_1 = (glm::transpose(glm::toMat3(ed_node.affine)) * glm::toMat3(ed_node.affine) - glm::mat3());

    residuals[0] = 0.f;

    for(int i = 0; i < 3; i++)
    {
        for(int k = 0; k < 3; k++)
        {
            residuals[0] += mat_1[i][k] * mat_1[i][k];
        }
    }

    residuals[0] = (float)sqrt(residuals[0]);
    residuals[0] += glm::determinant(glm::toMat3(ed_node.affine)) - 1;

    // TODO: figure out smooth component
    residuals[1] = 0.f;

    return residuals;
}

__global__ void kernel_jtj_jtf(float *jtj, float *jtf, GLuint *vx_counter, struct_vertex *vx_ptr, unsigned int ed_nodes_count, struct_ed_node *ed_graph)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int ed_per_thread = (unsigned int)max(1u, ed_nodes_count / blockDim.x * gridDim.x);

    // printf("\ned_per_thread: %u\n", ed_per_thread);

    for(unsigned int ed_node_idx = 0; ed_node_idx < ed_per_thread; ed_node_idx++)
    {
        unsigned int offset = idx * ed_per_thread + ed_node_idx;
        if(offset >= ed_nodes_count)
        {
            return;
        }

        // printf("\noffset: %u, ed_nodes_count: %u\n", offset, ed_nodes_count);

        struct_ed_node ed_node = ed_graph[offset];

        if(ed_node.position.z * ED_CELL_RES * ED_CELL_RES + ed_node.position.y * ED_CELL_RES + ed_node.position.x == 0u)
        {
            // unset ed_node
            return;
        }

        // printf("\ned_node position: (%f,%f,%f)\n", ed_node.position.x, ed_node.position.y, ed_node.position.z);

        float *jtf_cell = (float *)malloc(sizeof(float) * 10u);
        float *jtj_cell = (float *)malloc(sizeof(float) * 100u);

        for(unsigned long vx_idx = 0; vx_idx < vx_counter[0]; vx_idx++)
        {
            struct_vertex vx = vx_ptr[vx_idx];

            if(vx.ed_cell_id != offset)
            {
                // is not influenced by ed_node
                continue;
            }

            vx.position = vx.position * (float)VOLUME_VOXEL_DIM;

            // printf("\ned_node + vertex match\n");

            float vx_residual = evaluate_vx_residual(vx, ed_node);
            float *vx_pds = (float *)malloc(sizeof(float) * 10u);

            for(unsigned int component = 0; component < 10u; component++)
            {
                vx_pds[component] = evaluate_vx_pd(vx, ed_node, component, vx_residual);
            }

            for(unsigned int component_i = 0; component_i < 10u; component_i++)
            {
                jtf_cell[component_i] += vx_pds[component_i] * vx_residual;

                for(unsigned int component_k = 0; component_k < 10; component_k++)
                {
                    jtj_cell[component_i * 10u + component_k] += vx_pds[component_i] * vx_pds[component_k];
                }
            }

            free(vx_pds);
        }

        // printf("\njtf[%u]\n", offset * 10u);

        memcpy(jtf + offset * 10u, jtf_cell, sizeof(float) * 10);
        for(unsigned int line_i = 0; line_i < 10u; line_i++)
        {
            // printf("\njtf[%u]\n", ed_nodes * 10u * (offset * 10u + line_i) + offset * 10u);

            memcpy(jtj + ed_nodes_count * 10u * (offset * 10u + line_i) + offset * 10u, jtj_cell + line_i * 10u, sizeof(float) * 10);
        }

        free(jtf_cell);
        free(jtj_cell);
    }
}

__host__ void solve_for_h()
{
    cusparseMatDescr_t descr = nullptr;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    getLastCudaError("cusparseCreateMatDescr failure");
    cudaDeviceSynchronize();

    float *csr_val_jtj = nullptr;
    int *csr_row_ptr_jtj = nullptr;
    int *csr_col_ind_jtj = nullptr;
    int csr_nnz = 0;

    int N = (int)_ed_nodes_component_count;

    int *nnz_per_row_col = nullptr;
    checkCudaErrors(cudaMalloc(&nnz_per_row_col, sizeof(int) * 2));

    cusparseSnnz(cusparse_handle, CUSPARSE_DIRECTION_ROW, N, N, descr, _jtj, N, nnz_per_row_col, &csr_nnz);
    cudaDeviceSynchronize();

    // printf("\ncusparseStatus_t: %u\n", status);

    getLastCudaError("cusparseSnnz failure");

    printf("\nmxn: %ix%i, nnz_dev_mem: %u\n", N, N, csr_nnz);

    checkCudaErrors(cudaMalloc(&csr_val_jtj, sizeof(float) * csr_nnz));
    checkCudaErrors(cudaMalloc(&csr_row_ptr_jtj, sizeof(int) * (N + 1)));
    checkCudaErrors(cudaMalloc(&csr_col_ind_jtj, sizeof(int) * csr_nnz));

    cusparseSdense2csr(cusparse_handle, N, N, descr, _jtj, N, nnz_per_row_col, csr_val_jtj, csr_row_ptr_jtj, csr_col_ind_jtj);
    cudaDeviceSynchronize();

    checkCudaErrors(cudaFree(nnz_per_row_col));

    getLastCudaError("cusparseSdense2csr failure");

    const int max_iter = 256;
    const float tol = 1e-12f;

    float r0, r1, alpha, alpham1, beta;
    float dot;

    float a, b, na;

    alpha = 1.0f;
    alpham1 = -1.0f;
    beta = 0.f;
    r0 = 0.f;

    cudaDeviceSynchronize();
    cusparseStatus_t status = cusparseScsrmv(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, csr_nnz, &alpha, descr, csr_val_jtj, csr_row_ptr_jtj, csr_col_ind_jtj, _h, &beta, pcg_Ax);
    cudaDeviceSynchronize();

    if(status != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
    {
        printf("\ncusparseStatus_t: %u\n", status);
    }

    getLastCudaError("cusparseScsrmv failure");

    cublasSaxpy(cublas_handle, N, &alpham1, pcg_Ax, 1, _jtf, 1);
    cublasSdot(cublas_handle, N, _jtf, 1, _jtf, 1, &r1);

    printf("\ninitial residual = %e\n", sqrt(r1));

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
        cusparseStatus_t status = cusparseScsrmv(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, csr_nnz, &alpha, descr, csr_val_jtj, csr_row_ptr_jtj, csr_col_ind_jtj, pcg_p, &beta, pcg_Ax);
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
        cudaDeviceSynchronize();
        printf("\niteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
    }

    cudaDeviceSynchronize();

    checkCudaErrors(cudaFree(csr_val_jtj));
    checkCudaErrors(cudaFree(csr_row_ptr_jtj));
    checkCudaErrors(cudaFree(csr_col_ind_jtj));
    cusparseDestroyMatDescr(descr);
}

extern "C" void pcg_solve()
{
    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.buffer_vertex_counter, 0));
    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.buffer_reference_mesh_vertices, 0));
    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.buffer_occupied, 0));

    for(unsigned int i = 0; i < 4; i++)
    {
        checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.volume_cv_xyz_inv[i], 0));
    }

    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.array2d_kinect_depths, 0));

    size_t vx_bytes;
    GLuint *vx_counter;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&vx_counter, &vx_bytes, _cgr.buffer_vertex_counter));

    struct_vertex *vx_ptr;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&vx_ptr, &vx_bytes, _cgr.buffer_reference_mesh_vertices));

    size_t brick_bytes;
    GLuint *brick_list;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&brick_list, &brick_bytes, _cgr.buffer_occupied));

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

    int block_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_jtj_jtf, 0, 0);

    size_t grid_size = (_ed_nodes_count + block_size - 1) / block_size;
    kernel_jtj_jtf<<<grid_size, block_size>>>(_jtj, _jtf, vx_counter, vx_ptr, _ed_nodes_count, _ed_graph);

    getLastCudaError("render kernel failed");

    cudaDeviceSynchronize();

    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.array2d_kinect_depths, 0));

    for(unsigned int i = 0; i < 4; i++)
    {
        checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.volume_cv_xyz_inv[i], 0));
    }

    cudaDeviceSynchronize();

    solve_for_h();

    getLastCudaError("render kernel failed");

    cudaDeviceSynchronize();

    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.buffer_occupied, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.buffer_reference_mesh_vertices, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.buffer_vertex_counter, 0));
}