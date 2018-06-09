#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <GL/gl.h>
#include <GL/glext.h>

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/common.hpp>
#include <glm/exponential.hpp>
#include <glm/geometric.hpp>
#include <glm/integer.hpp>
#include <glm/mat2x2.hpp>
#include <glm/mat2x3.hpp>
#include <glm/mat2x4.hpp>
#include <glm/mat3x2.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat3x4.hpp>
#include <glm/mat4x2.hpp>
#include <glm/mat4x3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/matrix.hpp>
#include <glm/packing.hpp>
#include <glm/trigonometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/vector_relational.hpp>

#include <glm/gtx/norm.hpp>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_cusolver.h>
#include <helper_math.h>

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cuda_occupancy.h>
#include <device_launch_parameters.h>
#include <driver_types.h>
#include <iostream>
#include <texture_types.h>
#include <vector_types.h>

#include <recon_pc.cuh>

cudaGraphicsResource *cgr_buffer_reference_mesh_vertices = nullptr;
cudaGraphicsResource *cgr_buffer_vertex_counter = nullptr;
cudaGraphicsResource *cgr_buffer_bricks = nullptr;
cudaGraphicsResource *cgr_buffer_occupied = nullptr;
cudaGraphicsResource *cgr_volume_tsdf_data = nullptr;
cudaGraphicsResource *cgr_array2d_kinect_depths = nullptr;

surface<void, cudaSurfaceType3D> _volume_tsdf_data;
surface<void, cudaSurfaceType3D> _volume_tsdf_ref;

surface<void, cudaSurfaceType2DLayered> _array2d_kinect_depths;

cudaExtent _volume_res;
struct_native_handles _native_handles;

cudaArray *_volume_array_tsdf_ref = nullptr;
unsigned int *_active_bricks_count = nullptr;
unsigned int *_bricks_inv_index = nullptr;
struct_ed_node *_ed_graph = nullptr;
float *_jtj = nullptr;
float *_jtf = nullptr;
float *_h = nullptr;

const unsigned ED_CELL_RES = 2u;
const unsigned ED_CELL_VOXEL_DIM = 3u;
const unsigned ED_CELL_VOXELS = 27u;
const unsigned BRICK_RES = 9u;
const unsigned BRICK_VOXEL_DIM = 6u;
const unsigned BRICK_VOXELS = 216u;
const unsigned VOLUME_VOXEL_DIM = 50u;

__global__ void kernel_copy_reference(unsigned int *active_bricks, unsigned int *bricks_inv_index, GLuint *occupied_bricks, size_t occupied_brick_count)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= occupied_brick_count)
    {
        return;
    }

    unsigned int brick_id = occupied_bricks[idx];

    // printf("\nidx %u, brick %u", idx, brick_id);

    if(brick_id == 0u)
    {
        return;
    }

    unsigned int brick_position = atomicAdd(active_bricks, 1u);
    bricks_inv_index[brick_id] = brick_position;

    glm::uvec3 brick = glm::uvec3(0u);
    brick.z = brick_id / (BRICK_RES * BRICK_RES);
    brick_id %= (BRICK_RES * BRICK_RES);
    brick.y = brick_id / BRICK_RES;
    brick_id %= BRICK_RES;
    brick.x = brick_id;

    // printf("\nbrick %u: (%u,%u,%u)\n", brick_id, brick.x, brick.y, brick.z);

    for(unsigned int i = 0u; i < BRICK_VOXELS; i++)
    {
        unsigned int position_id = i;

        glm::uvec3 position = glm::uvec3(0u);
        position.z = position_id / (BRICK_VOXEL_DIM * BRICK_VOXEL_DIM);
        position_id %= (BRICK_VOXEL_DIM * BRICK_VOXEL_DIM);
        position.y = position_id / BRICK_VOXEL_DIM;
        position_id %= (BRICK_VOXEL_DIM);
        position.x = position_id;

        glm::uvec3 world = brick * BRICK_VOXEL_DIM + position;

        if(world.x >= VOLUME_VOXEL_DIM || world.y >= VOLUME_VOXEL_DIM || world.z >= VOLUME_VOXEL_DIM)
        {
            continue;
        }

        // printf("\nbrick %u, position %u: (%u,%u,%u)\n", occupied_bricks[idx], i, world.x, world.y, world.z);

        float2 data;
        surf3Dread(&data, _volume_tsdf_data, world.x * sizeof(float2), world.y, world.z);
        surf3Dwrite(data, _volume_tsdf_ref, world.x * sizeof(float2), world.y, world.z);
    }
}

/*
 * Identify enclosing ED cell in volume voxel space
 * */
__device__ unsigned int identify_ed_cell_pos(glm::vec3 position, unsigned int *bricks_inv_index)
{
    glm::uvec3 pos_voxel_space = glm::uvec3(position);
    glm::uvec3 brick_index3d = pos_voxel_space / BRICK_VOXEL_DIM;

    unsigned int brick_id = brick_index3d.z * BRICK_RES * BRICK_RES + brick_index3d.y * BRICK_RES + brick_index3d.x;

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

__global__ void kernel_sample_ed_nodes(GLuint *vx_counter, struct_vertex *vx_ptr, unsigned int *bricks_inv_index, struct_ed_node *ed_graph)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    unsigned int vx_per_thread = (unsigned int)max(1u, vx_counter[0] / blockDim.x * gridDim.x);

    for(unsigned long i = 0; i < vx_per_thread; i++)
    {
        unsigned long vertex_position = idx * vx_per_thread + i;

        if(vertex_position >= vx_counter[0])
        {
            return;
        }

        unsigned int ed_cell_pos = identify_ed_cell_pos(vx_ptr[vertex_position].position * (float)VOLUME_VOXEL_DIM, bricks_inv_index);

        vx_ptr[vertex_position].ed_cell_id = ed_cell_pos;

        if(ed_graph[ed_cell_pos].position.x < 0.f)
        {
            ed_graph[ed_cell_pos].position = vx_ptr[vertex_position].position * (float)VOLUME_VOXEL_DIM;
        }
        else
        {
            // TODO: incremental centroid instead of averaging
            ed_graph[ed_cell_pos].position = (ed_graph[ed_cell_pos].position + vx_ptr[vertex_position].position * (float)VOLUME_VOXEL_DIM) / 2.0f;
        }
    }
}

/*
 * Warp a position in volume voxel space with a single ED node
 * */
__device__ glm::vec3 warp_position(glm::vec3 &pos, struct_ed_node &ed_node)
{
    glm::vec3 dist = pos - ed_node.position;
    float skinning_weight = expf(glm::length(dist) * glm::length(dist) * 2 / (ED_CELL_VOXEL_DIM * ED_CELL_VOXEL_DIM));
    return skinning_weight * (glm::mat3(ed_node.affine) * dist + ed_node.position + ed_node.translation);
}

/*
 * Warp a normal in volume voxel space with a single ED node
 * */
__device__ glm::vec3 warp_normal(glm::vec3 &pos, glm::vec3 &normal, struct_ed_node &ed_node)
{
    glm::vec3 dist = pos - ed_node.position;
    float skinning_weight = expf(glm::length(dist) * glm::length(dist) * 2 / (ED_CELL_VOXEL_DIM * ED_CELL_VOXEL_DIM));
    return skinning_weight * (glm::transpose(glm::inverse(glm::mat3(ed_node.affine))) * normal);
}

__device__ float evaluate_vx_residual(struct_vertex &vertex, struct_ed_node &ed_node)
{
    glm::vec3 position = vertex.position * (float)VOLUME_VOXEL_DIM;

    glm::vec3 warped_position = warp_position(position, ed_node);
    glm::vec3 warped_normal = warp_normal(position, vertex.normal, ed_node);

    float residuals = 0.f;

    for(int i = 0; i < 5; i++)
    {
        // TODO: lookup depth maps and extract projective term

        glm::vec3 extracted_position = warped_position;
        extracted_position *= (1 + 0.1 * fracf(sinf(warped_position.x)));

        residuals += glm::dot(warped_normal, warped_position - extracted_position);
    }

    return residuals;
}

__device__ float evaluate_vx_pd(struct_vertex &vertex, struct_ed_node &ed_node, int partial_derivative_index, float vx_residual)
{
    switch(partial_derivative_index)
    {
    case 0:
        ed_node.position.x += 0.035f;
        break;
    case 1:
        ed_node.position.y += 0.035f;
        break;
    case 2:
        ed_node.position.z += 0.035f;
        break;
    case 3:
        ed_node.affine.w += 0.035f;
        break;
    case 4:
        ed_node.affine.x += 0.035f;
        break;
    case 5:
        ed_node.affine.y += 0.035f;
        break;
    case 6:
        ed_node.affine.z += 0.035f;
        break;
    case 7:
        ed_node.translation.x += 0.035f;
        break;
    case 8:
        ed_node.translation.y += 0.035f;
        break;
    case 9:
        ed_node.translation.z += 0.035f;
        break;
    }

    float residual_pos = evaluate_vx_residual(vertex, ed_node);

    switch(partial_derivative_index)
    {
    case 0:
        ed_node.position.x -= 0.035f;
        break;
    case 1:
        ed_node.position.y -= 0.035f;
        break;
    case 2:
        ed_node.position.z -= 0.035f;
        break;
    case 3:
        ed_node.affine.w -= 0.035f;
        break;
    case 4:
        ed_node.affine.x -= 0.035f;
        break;
    case 5:
        ed_node.affine.y -= 0.035f;
        break;
    case 6:
        ed_node.affine.z -= 0.035f;
        break;
    case 7:
        ed_node.translation.x -= 0.035f;
        break;
    case 8:
        ed_node.translation.y -= 0.035f;
        break;
    case 9:
        ed_node.translation.z -= 0.035f;
        break;
    }

    return (residual_pos - vx_residual) / 0.07f;
}

//__device__ float *evaluate_ed_node_residuals(cudaExtent &volume_res, struct_ed_node &ed_node)
//{
//    float *residuals = new float[2];
//
//    glm::mat3 mat_1 = (glm::transpose(glm::toMat3(ed_node.affine)) * glm::toMat3(ed_node.affine) - glm::mat3());
//
//    residuals[0] = 0.f;
//
//    for(int i = 0; i < 3; i++)
//    {
//        for(int k = 0; k < 3; k++)
//        {
//            residuals[0] += mat_1[i][k] * mat_1[i][k];
//        }
//    }
//
//    residuals[0] = (float)sqrt(residuals[0]);
//    residuals[0] += glm::determinant(glm::toMat3(ed_node.affine)) - 1;
//
//    // TODO: figure out smooth component
//    residuals[1] = 0.f;
//
//    return residuals;
//}

__global__ void kernel_jtj_jtf(float *jtj, float *jtf, GLuint *vx_counter, struct_vertex *vx_ptr, unsigned int *active_bricks, unsigned int *bricks_inv_index, struct_ed_node *ed_graph)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int ed_nodes = active_bricks[0] * ED_CELL_RES * ED_CELL_RES * ED_CELL_RES;
    unsigned int ed_per_thread = (unsigned int)max(1u, ed_nodes / blockDim.x * gridDim.x);

    for(unsigned int ed_node_idx = 0; ed_node_idx < ed_per_thread; ed_node_idx++)
    {
        unsigned int offset = idx * ed_per_thread + ed_node_idx;
        if(offset >= ed_nodes)
        {
            break;
        }
        struct_ed_node ed_node = ed_graph[offset];

        float *jtf_cell = (float *)malloc(sizeof(float) * 10);
        float *jtj_cell = (float *)malloc(sizeof(float) * 100);

        for(unsigned long vx_idx = 0; vx_idx < vx_counter[0]; vx_idx++)
        {
            struct_vertex vx = vx_ptr[vx_idx];

            if(vx.ed_cell_id != ed_node_idx)
            {
                // is not influenced by ed_node
                continue;
            }

            float vx_residual = evaluate_vx_residual(vx, ed_node);
            float *vx_pds = (float *)malloc(sizeof(float) * 10);

            for(unsigned int component = 0; component < 10u; component++)
            {
                unsigned int component_offset = ed_node_idx * 10u + component;
                if(component_offset >= ed_nodes * 10u)
                {
                    break;
                }

                vx_pds[vx_idx] = evaluate_vx_pd(vx, ed_node, component, vx_residual);
            }

            for(unsigned int component_i = 0; component_i < 10u; component_i++)
            {
                jtf_cell[component_i] += vx_pds[component_i] * vx_residual;

                for(int component_k = 0; component_k < 10; component_k++)
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
            memcpy(jtj + ed_nodes * 10u * (offset * 10u + line_i) + offset * 10u, jtj_cell + line_i * 10u, sizeof(float) * 10);
        }

        free(jtf_cell);
        free(jtj_cell);
    }
}

//__global__ void kernel_jtj_jtf(float *jtj, float *jtf, GLuint *vx_counter, struct_vertex *vx_ptr, unsigned int *bricks_inv_index, struct_ed_node *ed_graph)
//{
//    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
//
//    unsigned int vx_per_thread = (unsigned int)max(1u, vx_counter[0] / blockDim.x * gridDim.x);
//
//    for(unsigned long ii = 0; ii < vx_per_thread; ii++)
//    {
//        unsigned long vertex_position = idx * vx_per_thread + ii;
//
//        if(vertex_position >= vx_counter[0])
//        {
//            return;
//        }
//
//        struct_vertex vx = vx_ptr[vertex_position];
//
//        unsigned int ed_cell_pos = identify_ed_cell_pos(vx_ptr[vertex_position].position * (float)VOLUME_VOXEL_DIM, bricks_inv_index);
//        struct_ed_node node = ed_graph[ed_cell_pos];
//
//        float vx_residual = evaluate_vx_residual(vx, node);
//        float *vx_pds = (float *)malloc(sizeof(float) * 10);
//
//        for(int i = 0; i < 10; i++)
//        {
//            vx_pds[i] = evaluate_vx_pd(vx, node, i, vx_residual);
//        }
//
//        for(int i = 0; i < 10; i++)
//        {
//            atomicAdd(jtf + ed_cell_pos * 10 + i, vx_pds[i] * vx_residual);
//            // jtf[ed_cell_pos * 10 + i] += vx_pds[i] * vx_residual;
//
//            for(int k = 0; k < 10; k++)
//            {
//                atomicAdd(jtj + (ed_cell_pos * 10 + i) * ED_CELL_RES * ED_CELL_RES * ED_CELL_RES + ed_cell_pos * 10 + k, vx_pds[i] * vx_pds[k]);
//                // jtj[(ed_cell_pos * 10 + i) * ED_CELL_RES * ED_CELL_RES * ED_CELL_RES + ed_cell_pos * 10 + k] += vx_pds[i] * vx_pds[k];
//            }
//        }
//
//        free(vx_pds);
//    }
//}

__global__ void kernel_fuse_volume(GLuint *occupied_bricks, size_t occupied_brick_count)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= occupied_brick_count)
    {
        return;
    }

    unsigned int brick_id = occupied_bricks[idx];

    if(brick_id == 0u)
    {
        return;
    }

    glm::uvec3 brick = glm::uvec3(0u);
    brick.z = brick_id / (BRICK_RES * BRICK_RES);
    brick_id %= (BRICK_RES * BRICK_RES);
    brick.y = brick_id / BRICK_RES;
    brick_id %= BRICK_RES;
    brick.x = brick_id;

    // printf("\nbrick %u: (%u,%u,%u)\n", brick_id, brick.x, brick.y, brick.z);

    for(unsigned int i = 0u; i < BRICK_VOXELS; i++)
    {
        unsigned int position_id = i;

        glm::uvec3 position = glm::uvec3(0u);
        position.z = position_id / (BRICK_VOXEL_DIM * BRICK_VOXEL_DIM);
        position_id %= (BRICK_VOXEL_DIM * BRICK_VOXEL_DIM);
        position.y = position_id / BRICK_VOXEL_DIM;
        position_id %= (BRICK_VOXEL_DIM);
        position.x = position_id;

        glm::uvec3 world = brick * BRICK_VOXEL_DIM + position;

        if(world.x >= VOLUME_VOXEL_DIM || world.y >= VOLUME_VOXEL_DIM || world.z >= VOLUME_VOXEL_DIM)
        {
            continue;
        }

        // printf("\nbrick %u, position %u: (%u,%u,%u)\n", occupied_bricks[idx], i, world.x, world.y, world.z);

        float2 data, ref;
        surf3Dread(&data, _volume_tsdf_data, world.x * sizeof(float2), world.y, world.z);
        surf3Dread(&ref, _volume_tsdf_ref, world.x * sizeof(float2), world.y, world.z);

        float2 fused;

        fused.y = ref.y + data.y;

        if(fused.y > 0.001f)
        {
            fused.x = data.x * data.y / fused.y + ref.x * ref.y / fused.y;
        }
        else
        {
            fused.x = data.y > ref.y ? data.x : ref.x;
        }

        surf3Dwrite(fused, _volume_tsdf_data, world.x * sizeof(float2), world.y, world.z);
    }
}

__host__ void convert_jtj_to_sparse(cusparseHandle_t &cusparseHandle, float *csr_sorted_val_jtj, int *csr_sorted_row_ptr_jtj, int *csr_sorted_col_ind_jtj, cusparseMatDescr_t &descr, int &nnz_dev_mem)
{
    unsigned int active_bricks;
    checkCudaErrors(cudaMemcpy(&active_bricks, _active_bricks_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    unsigned long ed_nodes_count = active_bricks * ED_CELL_RES * ED_CELL_RES * ED_CELL_RES;
    unsigned long ed_nodes_component_count = ed_nodes_count * 10u;

    int m = (int)ed_nodes_component_count;
    int n = (int)ed_nodes_component_count;

    cusparseCreateMatDescr(&descr);

    getLastCudaError("cusparseCreateMatDescr failure");

    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    int *nnz_per_row_col = nullptr;

    checkCudaErrors(cudaMalloc((void **)&nnz_per_row_col, sizeof(int) * 2));

    cusparseSnnz(cusparseHandle, CUSPARSE_DIRECTION_ROW, m, n, descr, _jtj, m, nnz_per_row_col, &nnz_dev_mem);

    getLastCudaError("cusparseSnnz failure");

    printf("\nnnz_dev_mem: %u\n", nnz_dev_mem);

    cudaDeviceSynchronize();

    checkCudaErrors(cudaMalloc((void **)&csr_sorted_val_jtj, sizeof(float) * nnz_dev_mem));
    checkCudaErrors(cudaMalloc((void **)&csr_sorted_row_ptr_jtj, sizeof(int) * (m + 1)));
    checkCudaErrors(cudaMalloc((void **)&csr_sorted_col_ind_jtj, sizeof(int) * nnz_dev_mem));

    cudaDeviceSynchronize();

    cusparseSdense2csr(cusparseHandle, m, n, descr, _jtj, m, nnz_per_row_col, csr_sorted_val_jtj, csr_sorted_row_ptr_jtj, csr_sorted_col_ind_jtj);

    getLastCudaError("cusparseSdense2csr failure");

    cudaDeviceSynchronize();
}

__host__ void pcg(cublasHandle_t &cublasHandle, cusparseHandle_t &cusparseHandle, cusparseMatDescr_t &descr, int csr_nnz, float *csr_sorted_val_jtj, int *csr_sorted_row_ptr_jtj,
                  int *csr_sorted_col_ind_jtj)
{
    unsigned int active_bricks;
    checkCudaErrors(cudaMemcpy(&active_bricks, _active_bricks_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    unsigned long ed_nodes_count = active_bricks * ED_CELL_RES * ED_CELL_RES * ED_CELL_RES;
    unsigned long ed_nodes_component_count = ed_nodes_count * 10u;

    int N = (int)ed_nodes_component_count;

    const int max_iter = 16;
    const float tol = 1e-12f;
    const float floatone = 1.0;
    const float floatzero = 0.0;

    float r0, r1, alpha, beta;
    float dot, nalpha;

    float *d_p, *d_omega, *d_r;

    checkCudaErrors(cudaMalloc((void **)&d_r, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_p, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_omega, N * sizeof(float)));

    checkCudaErrors(cudaFree(d_omega));
    checkCudaErrors(cudaFree(d_p));
    checkCudaErrors(cudaFree(d_r));

    cudaDeviceSynchronize();
}

__host__ void solve_for_h()
{
    cublasHandle_t cublasHandle = nullptr;
    cublasCreate(&cublasHandle);
    getLastCudaError("cublasCreate failure");

    cusparseHandle_t cusparseHandle = nullptr;
    cusparseCreate(&cusparseHandle);
    getLastCudaError("cusparseCreate failure");

    //    cusparseMatDescr_t descr = nullptr;
    //    float *csr_sorted_val_jtj = nullptr;
    //    int *csr_sorted_row_ptr_jtj = nullptr;
    //    int *csr_sorted_col_ind_jtj = nullptr;
    //    int csr_nnz;

    // convert_jtj_to_sparse(cusparseHandle, csr_sorted_val_jtj, csr_sorted_row_ptr_jtj, csr_sorted_col_ind_jtj, descr, csr_nnz);
    // pcg(cublasHandle, cusparseHandle, descr, csr_nnz, csr_sorted_val_jtj, csr_sorted_row_ptr_jtj, csr_sorted_col_ind_jtj);

    //    cudaFree (csr_sorted_val_jtj);
    //    cudaFree (csr_sorted_row_ptr_jtj);
    //    cudaFree (csr_sorted_col_ind_jtj);
    //    cusparseDestroyMatDescr(descr);

    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);
}

extern "C" void init_cuda(glm::uvec3 &volume_res, struct_native_handles &native_handles)
{
    _volume_res = make_cudaExtent(volume_res.x, volume_res.y, volume_res.z);

    int devicesCount;
    cudaGetDeviceCount(&devicesCount);
    for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
    {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, deviceIndex);
        if(deviceProperties.major >= 6 && deviceProperties.minor >= 1)
        {
            cudaSetDevice(deviceIndex);
        }
    }

    size_t value;

    cudaDeviceGetLimit(&value, cudaLimitMallocHeapSize);
    printf("\n\nLimit Malloc Heap Size: %lu\n", value);

    cudaDeviceGetLimit(&value, cudaLimitStackSize);
    printf("\nLimit Stack Size: %lu\n", value);

    cudaDeviceGetLimit(&value, cudaLimitDevRuntimeSyncDepth);
    printf("\nLimit Sync Depth: %lu\n", value);

    cudaDeviceGetLimit(&value, cudaLimitDevRuntimePendingLaunchCount);
    printf("\nLimit Pending Launch: %lu\n\n", value);

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cgr_buffer_vertex_counter, native_handles.buffer_vertex_counter, cudaGraphicsRegisterFlagsReadOnly));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cgr_buffer_reference_mesh_vertices, native_handles.buffer_reference_vertices, cudaGraphicsRegisterFlagsReadOnly));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cgr_buffer_bricks, native_handles.buffer_bricks, cudaGraphicsRegisterFlagsReadOnly));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cgr_buffer_occupied, native_handles.buffer_occupied, cudaGraphicsRegisterFlagsReadOnly));
    checkCudaErrors(cudaGraphicsGLRegisterImage(&cgr_volume_tsdf_data, native_handles.volume_tsdf_data, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    checkCudaErrors(cudaGraphicsGLRegisterImage(&cgr_array2d_kinect_depths, native_handles.array2d_kinect_depths, GL_TEXTURE_2D_ARRAY, cudaGraphicsRegisterFlagsReadOnly));

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
    checkCudaErrors(cudaMalloc3DArray(&_volume_array_tsdf_ref, &channel_desc, _volume_res, cudaArraySurfaceLoadStore));
}

extern "C" void deinit_cuda()
{
    checkCudaErrors(cudaGraphicsUnregisterResource(cgr_buffer_vertex_counter));
    checkCudaErrors(cudaGraphicsUnregisterResource(cgr_buffer_reference_mesh_vertices));
    checkCudaErrors(cudaGraphicsUnregisterResource(cgr_buffer_bricks));
    checkCudaErrors(cudaGraphicsUnregisterResource(cgr_buffer_occupied));
    checkCudaErrors(cudaGraphicsUnregisterResource(cgr_volume_tsdf_data));
    checkCudaErrors(cudaGraphicsUnregisterResource(cgr_array2d_kinect_depths));

    if(_ed_graph != nullptr)
    {
        checkCudaErrors(cudaFree(_ed_graph));
    }

    if(_jtf != nullptr)
    {
        checkCudaErrors(cudaFree(_jtf));
    }

    if(_jtj != nullptr)
    {
        checkCudaErrors(cudaFree(_jtj));
    }

    if(_h != nullptr)
    {
        checkCudaErrors(cudaFree(_h));
    }

    if(_volume_array_tsdf_ref != nullptr)
    {
        checkCudaErrors(cudaFree(_volume_array_tsdf_ref));
    }
}

extern "C" void copy_reference_volume()
{
    if(_active_bricks_count != nullptr)
    {
        cudaFree(_active_bricks_count);
    }

    cudaMallocManaged(&_active_bricks_count, sizeof(unsigned int));
    *_active_bricks_count = 0u;

    if(_bricks_inv_index != nullptr)
    {
        cudaFree(_bricks_inv_index);
    }

    cudaMalloc((void **)&_bricks_inv_index, BRICK_RES * BRICK_RES * BRICK_RES * sizeof(unsigned int));

    checkCudaErrors(cudaGraphicsMapResources(1, &cgr_volume_tsdf_data, 0));
    checkCudaErrors(cudaGraphicsMapResources(1, &cgr_buffer_occupied, 0));

    cudaArray *volume_array_tsdf_data = nullptr;
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&volume_array_tsdf_data, cgr_volume_tsdf_data, 0, 0));

    size_t occupied_brick_bytes;
    GLuint *brick_list;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&brick_list, &occupied_brick_bytes, cgr_buffer_occupied));

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
    checkCudaErrors(cudaBindSurfaceToArray(&_volume_tsdf_data, volume_array_tsdf_data, &channel_desc));
    checkCudaErrors(cudaBindSurfaceToArray(&_volume_tsdf_ref, _volume_array_tsdf_ref, &channel_desc));

    int block_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_copy_reference, 0, 0);

    unsigned max_bricks = ((unsigned)occupied_brick_bytes) / sizeof(unsigned);
    size_t gridSize = (max_bricks + block_size - 1) / block_size;
    kernel_copy_reference<<<gridSize, block_size>>>(_active_bricks_count, _bricks_inv_index, brick_list, max_bricks);

    getLastCudaError("render kernel failed");

    cudaDeviceSynchronize();

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cgr_buffer_occupied, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cgr_volume_tsdf_data, 0));
}

extern "C" void sample_ed_nodes()
{
    if(_ed_graph != nullptr)
    {
        checkCudaErrors(cudaFree(_ed_graph));
    }

    if(_jtf != nullptr)
    {
        checkCudaErrors(cudaFree(_jtf));
    }

    if(_jtj != nullptr)
    {
        checkCudaErrors(cudaFree(_jtj));
    }

    if(_h != nullptr)
    {
        checkCudaErrors(cudaFree(_h));
    }

    unsigned int active_bricks;
    checkCudaErrors(cudaMemcpy(&active_bricks, _active_bricks_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    printf("\nactive_bricks: %u\n", active_bricks);

    unsigned long ed_nodes_count = active_bricks * ED_CELL_RES * ED_CELL_RES * ED_CELL_RES;
    unsigned long ed_nodes_component_count = ed_nodes_count * 10u;

    printf("\ned_nodes_count: %lu\n", ed_nodes_count);
    printf("\ned_nodes_component_count: %lu\n", ed_nodes_component_count);

    cudaDeviceSynchronize();

    checkCudaErrors(cudaMalloc(&_jtj, ed_nodes_component_count * ed_nodes_component_count * sizeof(float)));
    checkCudaErrors(cudaMalloc(&_jtf, ed_nodes_component_count * sizeof(float)));
    checkCudaErrors(cudaMalloc(&_h, ed_nodes_component_count * sizeof(float)));
    checkCudaErrors(cudaMalloc(&_ed_graph, ed_nodes_count * sizeof(struct_ed_node)));

    checkCudaErrors(cudaGraphicsMapResources(1, &cgr_buffer_vertex_counter, 0));
    checkCudaErrors(cudaGraphicsMapResources(1, &cgr_buffer_reference_mesh_vertices, 0));

    size_t vx_bytes;
    GLuint *vx_counter;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&vx_counter, &vx_bytes, cgr_buffer_vertex_counter));

    struct_vertex *vx_ptr;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&vx_ptr, &vx_bytes, cgr_buffer_reference_mesh_vertices));

    // printf("\nvx_bytes: %zu\n", vx_bytes);

    int blockSize;
    int minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel_sample_ed_nodes, 0, 0);

    unsigned max_vertices = ((unsigned)vx_bytes) / sizeof(struct_vertex);
    size_t gridSize = (max_vertices + blockSize - 1) / blockSize;
    kernel_sample_ed_nodes<<<gridSize, blockSize>>>(vx_counter, vx_ptr, _bricks_inv_index, _ed_graph);

    getLastCudaError("render kernel failed");

    cudaDeviceSynchronize();

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cgr_buffer_reference_mesh_vertices, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cgr_buffer_vertex_counter, 0));
}

extern "C" void align_non_rigid()
{
    checkCudaErrors(cudaGraphicsMapResources(1, &cgr_buffer_vertex_counter, 0));
    checkCudaErrors(cudaGraphicsMapResources(1, &cgr_buffer_reference_mesh_vertices, 0));
    checkCudaErrors(cudaGraphicsMapResources(1, &cgr_buffer_occupied, 0));
    checkCudaErrors(cudaGraphicsMapResources(1, &cgr_volume_tsdf_data, 0));

    size_t vx_bytes;
    GLuint *vx_counter;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&vx_counter, &vx_bytes, cgr_buffer_vertex_counter));

    struct_vertex *vx_ptr;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&vx_ptr, &vx_bytes, cgr_buffer_reference_mesh_vertices));

    size_t brick_bytes;
    GLuint *brick_list;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&brick_list, &brick_bytes, cgr_buffer_occupied));

    cudaArray *volume_array_tsdf_data = nullptr;
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&volume_array_tsdf_data, cgr_volume_tsdf_data, 0, 0));

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
    checkCudaErrors(cudaBindSurfaceToArray(&_volume_tsdf_data, volume_array_tsdf_data, &channel_desc));
    checkCudaErrors(cudaBindSurfaceToArray(&_volume_tsdf_ref, _volume_array_tsdf_ref, &channel_desc));

    int block_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_jtj_jtf, 0, 0);

    unsigned int active_bricks;
    checkCudaErrors(cudaMemcpy(&active_bricks, _active_bricks_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    unsigned long ed_nodes_count = active_bricks * ED_CELL_RES * ED_CELL_RES * ED_CELL_RES;
    size_t gridSize = (ed_nodes_count + block_size - 1) / block_size;
    kernel_jtj_jtf<<<gridSize, block_size>>>(_jtj, _jtf, vx_counter, vx_ptr, _active_bricks_count, _bricks_inv_index, _ed_graph);

    getLastCudaError("render kernel failed");

    cudaDeviceSynchronize();

    // solve_for_h();

    getLastCudaError("render kernel failed");

    cudaDeviceSynchronize();

    // int block_size;
    // int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_fuse_volume, 0, 0);

    unsigned max_bricks = ((unsigned)brick_bytes) / sizeof(unsigned);
    /*size_t*/ gridSize = (max_bricks + block_size - 1) / block_size;
    kernel_fuse_volume<<<gridSize, block_size>>>(brick_list, max_bricks);

    getLastCudaError("render kernel failed");

    cudaDeviceSynchronize();

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cgr_volume_tsdf_data, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cgr_buffer_occupied, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cgr_buffer_reference_mesh_vertices, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cgr_buffer_vertex_counter, 0));
}
