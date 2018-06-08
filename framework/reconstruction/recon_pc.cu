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
struct_ed_node *_ed_graph = nullptr;
float *_jtj = nullptr;
float *_jtf = nullptr;
float *_h = nullptr;

const unsigned ED_GRAPH_NODE_RES = 9u;
const unsigned BRICK_RES = 9u;
const unsigned BRICK_VOXEL_DIM = 6u;
const unsigned BRICK_VOXELS = 216u;
const unsigned VOLUME_VOXEL_DIM = 50u;

__global__ void kernel_copy_reference(GLuint *occupied_bricks, size_t occupied_brick_count)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx < occupied_brick_count)
    {
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

            float2 data;
            surf3Dread(&data, _volume_tsdf_data, world.x * sizeof(float2), world.y, world.z);
            surf3Dwrite(data, _volume_tsdf_ref, world.x * sizeof(float2), world.y, world.z);
        }
    }
}

__global__ void kernel_sample_ed_nodes(GLuint *vx_counter, struct_vertex *vx_ptr, struct_ed_node *_ed_graph)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    long int u = x * VOLUME_VOXEL_DIM / ED_GRAPH_NODE_RES;
    long int v = y * VOLUME_VOXEL_DIM / ED_GRAPH_NODE_RES;
    long int d = z * VOLUME_VOXEL_DIM / ED_GRAPH_NODE_RES;

    long int node_id = x * ED_GRAPH_NODE_RES * ED_GRAPH_NODE_RES + y * ED_GRAPH_NODE_RES + z;

    GLuint counter = vx_counter[0];

    // printf("\ncounter: %u\n", counter);

    struct_ed_node *node = _ed_graph + node_id;

    for(long int i = 0; i < (long int)counter; i++)
    {
        glm::vec3 pos = vx_ptr[i].position;
        pos *= VOLUME_VOXEL_DIM;
        if(pos.x > u && pos.x < u + ED_GRAPH_NODE_RES && pos.y > v && pos.y < v + ED_GRAPH_NODE_RES && pos.z > d && pos.z < d + ED_GRAPH_NODE_RES)
        {
            node->position = pos;
            node->set = true;
            // printf("\nfound node: %f,%f,%f at %lu\n", pos.x, pos.y, pos.z, i);
            break;
        }
    }
}

__device__ glm::vec3 warp_position(glm::vec3 &pos, struct_ed_node &ed_node)
{
    pos *= VOLUME_VOXEL_DIM;

    glm::vec3 dist = pos - ed_node.position;
    float skinning_weight = expf(glm::length(dist) * glm::length(dist) * 2 / (ED_GRAPH_NODE_RES * ED_GRAPH_NODE_RES));
    return skinning_weight * (ed_node.affine * dist + ed_node.position + ed_node.translation);
}

__device__ glm::vec3 warp_normal(glm::vec3 &pos, glm::vec3 &normal, struct_ed_node &ed_node)
{
    pos *= VOLUME_VOXEL_DIM;

    glm::vec3 dist = pos - ed_node.position;
    float skinning_weight = expf(glm::length(dist) * glm::length(dist) * 2 / (ED_GRAPH_NODE_RES * ED_GRAPH_NODE_RES));
    return skinning_weight * (glm::transpose(glm::inverse(ed_node.affine)) * normal);
}

__device__ float evaluate_vx_residual(struct_vertex &vertex, struct_ed_node &ed_node)
{
    glm::vec3 warped_position = warp_position(vertex.position, ed_node);
    glm::vec3 warped_normal = warp_normal(vertex.position, vertex.normal, ed_node);

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
        ed_node.affine[0][0] += 0.035f;
        break;
    case 4:
        ed_node.affine[0][1] += 0.035f;
        break;
    case 5:
        ed_node.affine[0][2] += 0.035f;
        break;
    case 6:
        ed_node.affine[1][0] += 0.035f;
        break;
    case 7:
        ed_node.affine[1][1] += 0.035f;
        break;
    case 8:
        ed_node.affine[1][2] += 0.035f;
        break;
    case 9:
        ed_node.affine[2][0] += 0.035f;
        break;
    case 10:
        ed_node.affine[2][1] += 0.035f;
        break;
    case 11:
        ed_node.affine[2][2] += 0.035f;
        break;
    case 12:
        ed_node.translation.x += 0.035f;
        break;
    case 13:
        ed_node.translation.y += 0.035f;
        break;
    case 14:
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
        ed_node.affine[0][0] -= 0.035f;
        break;
    case 4:
        ed_node.affine[0][1] -= 0.035f;
        break;
    case 5:
        ed_node.affine[0][2] -= 0.035f;
        break;
    case 6:
        ed_node.affine[1][0] -= 0.035f;
        break;
    case 7:
        ed_node.affine[1][1] -= 0.035f;
        break;
    case 8:
        ed_node.affine[1][2] -= 0.035f;
        break;
    case 9:
        ed_node.affine[2][0] -= 0.035f;
        break;
    case 10:
        ed_node.affine[2][1] -= 0.035f;
        break;
    case 11:
        ed_node.affine[2][2] -= 0.035f;
        break;
    case 12:
        ed_node.translation.x -= 0.035f;
        break;
    case 13:
        ed_node.translation.y -= 0.035f;
        break;
    case 14:
        ed_node.translation.z -= 0.035f;
        break;
    }

    return (residual_pos - vx_residual) / 0.07f;
}

__device__ float *evaluate_ed_node_residuals(cudaExtent &volume_res, struct_ed_node &ed_node)
{
    float *residuals = new float[2];

    glm::mat3 mat_1 = (glm::transpose(ed_node.affine) * ed_node.affine - glm::mat3());

    residuals[0] = 0.f;

    for(int i = 0; i < 3; i++)
    {
        for(int k = 0; k < 3; k++)
        {
            residuals[0] += mat_1[i][k] * mat_1[i][k];
        }
    }

    residuals[0] = (float)sqrt(residuals[0]);
    residuals[0] += glm::determinant(ed_node.affine) - 1;

    // TODO: figure out smooth component
    residuals[1] = 0.f;

    return residuals;
}

__global__ void kernel_jtj_jtf(float *jtj, float *jtf, GLuint *vx_counter, struct_vertex *vx_ptr, struct_ed_node *_ed_graph)
{
    const unsigned long long int block_id = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    const unsigned long long int thread_id = block_id * blockDim.x + threadIdx.x;

    GLuint vertex_counter = vx_counter[0];

    if(thread_id >= vertex_counter)
    {
        return;
    }

    struct_vertex vx = vx_ptr[thread_id];

    glm::vec3 pos = vx.position;
    pos *= VOLUME_VOXEL_DIM;

    long int node_id = (int)pos.x * ED_GRAPH_NODE_RES + (int)pos.y + (int)pos.z / ED_GRAPH_NODE_RES;
    struct_ed_node node = *(_ed_graph + node_id);

    float vx_residual = evaluate_vx_residual(vx, node);
    float *vx_pds = (float *)malloc(sizeof(float) * 15);

    for(int i = 0; i < 15; i++)
    {
        vx_pds[i] = evaluate_vx_pd(vx, node, i, vx_residual);
    }

    for(int i = 0; i < 15; i++)
    {
        jtf[node_id * 15 + i] += vx_pds[i] * vx_residual;

        for(int k = 0; k < 15; k++)
        {
            jtj[(node_id * 15 + i) * ED_GRAPH_NODE_RES + node_id * 15 + k] += vx_pds[i] * vx_pds[k];
        }
    }

    free(vx_pds);
}

__global__ void kernel_fuse_volume(GLuint *occupied_bricks, size_t occupied_brick_count)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx < occupied_brick_count)
    {
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
}

__host__ void solve_for_h()
{
    int m = 0; // ED_GRAPH_NODE_COUNT * 15;
    int n = 0; // ED_GRAPH_NODE_COUNT * 15;

    cublasHandle_t cublasHandle = nullptr;
    cublasCreate(&cublasHandle);

    getLastCudaError("cublasCreate failure");

    cusparseHandle_t cusparseHandle = nullptr;
    cusparseCreate(&cusparseHandle);

    getLastCudaError("cusparseCreate failure");

    cusparseMatDescr_t descr = nullptr;
    cusparseCreateMatDescr(&descr);

    getLastCudaError("cusparseCreateMatDescr failure");

    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    int *nnz_per_row_col = nullptr;
    int nnz_in_dev_memory;

    checkCudaErrors(cudaMalloc((void **)&nnz_per_row_col, sizeof(int) * 2));

    cusparseSnnz(cusparseHandle, CUSPARSE_DIRECTION_ROW, m, n, descr, _jtj, m, nnz_per_row_col, &nnz_in_dev_memory);

    getLastCudaError("cusparseSnnz failure");

    // printf ("\nNNZ: %u\n", nnz_in_dev_memory);

    cudaDeviceSynchronize();

    float *csr_sorted_val_jtj = nullptr;
    int *csr_sorted_row_ptr_jtj = nullptr;
    int *csr_sorted_col_ind_jtj = nullptr;

    checkCudaErrors(cudaMalloc((void **)&csr_sorted_val_jtj, sizeof(float) * nnz_in_dev_memory));
    checkCudaErrors(cudaMalloc((void **)&csr_sorted_row_ptr_jtj, sizeof(int) * (m + 1)));
    checkCudaErrors(cudaMalloc((void **)&csr_sorted_col_ind_jtj, sizeof(int) * nnz_in_dev_memory));

    cusparseSdense2csr(cusparseHandle, m, n, descr, _jtj, m, nnz_per_row_col, csr_sorted_val_jtj, csr_sorted_row_ptr_jtj, csr_sorted_col_ind_jtj);

    getLastCudaError("cusparseSdense2csr failure");

    cudaDeviceSynchronize();

    float *h = (float *)malloc(sizeof(float) * n);
    float *rhs = (float *)malloc(sizeof(float) * n);

    for(int i = 0; i < n; i++)
    {
        rhs[i] = 1.0;
        h[i] = 0.0;
    }

    float *jtjh = nullptr;
    float *r = nullptr;
    float *p = nullptr;

    checkCudaErrors(cudaMalloc((void **)&jtjh, n * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&r, n * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&p, n * sizeof(float)));

    checkCudaErrors(cudaMemcpy(_h, h, n * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(r, rhs, n * sizeof(float), cudaMemcpyHostToDevice));

    float alpha = 1.0f;
    float alpham1 = -1.0f;
    float beta = 0.0f;
    float r0 = 0.f;

    float a, b, na, r1, dot;

    const float tol = 1e-5f;

    cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz_in_dev_memory, &alpha, descr, csr_sorted_val_jtj, csr_sorted_row_ptr_jtj, csr_sorted_col_ind_jtj, _h, &beta, jtjh);

    cublasSaxpy(cublasHandle, n, &alpham1, jtjh, 1, r, 1);
    cublasSdot(cublasHandle, n, r, 1, r, 1, &r1);

    printf("initial residual = %e\n", sqrt(r1));

    int k = 1;

    while(r1 > tol * tol && k <= 10)
    {
        if(k > 1)
        {
            b = r1 / r0;
            cublasSscal(cublasHandle, n, &b, p, 1);
            cublasSaxpy(cublasHandle, n, &alpha, r, 1, p, 1);
        }
        else
        {
            cublasScopy(cublasHandle, n, r, 1, p, 1);
        }

        cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz_in_dev_memory, &alpha, descr, csr_sorted_val_jtj, csr_sorted_row_ptr_jtj, csr_sorted_col_ind_jtj, p, &beta, jtjh);
        cublasSdot(cublasHandle, n, p, 1, jtjh, 1, &dot);
        a = r1 / dot;

        cublasSaxpy(cublasHandle, n, &a, p, 1, _h, 1);
        na = -a;
        cublasSaxpy(cublasHandle, n, &na, jtjh, 1, r, 1);

        r0 = r1;
        cublasSdot(cublasHandle, n, r, 1, r, 1, &r1);
        cudaDeviceSynchronize();
        printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
        k++;
    }

    cudaDeviceSynchronize();

    if(jtjh != nullptr)
    {
        checkCudaErrors(cudaFree(jtjh));
    }

    if(r != nullptr)
    {
        checkCudaErrors(cudaFree(r));
    }

    if(p != nullptr)
    {
        checkCudaErrors(cudaFree(p));
    }

    if(csr_sorted_col_ind_jtj != nullptr)
    {
        checkCudaErrors(cudaFree(csr_sorted_col_ind_jtj));
    }

    if(csr_sorted_row_ptr_jtj != nullptr)
    {
        checkCudaErrors(cudaFree(csr_sorted_row_ptr_jtj));
    }

    if(csr_sorted_val_jtj != nullptr)
    {
        checkCudaErrors(cudaFree(csr_sorted_val_jtj));
    }

    if(nnz_per_row_col != nullptr)
    {
        checkCudaErrors(cudaFree(nnz_per_row_col));
    }

    if(rhs != nullptr)
    {
        free(rhs);
    }

    if(h != nullptr)
    {
        free(h);
    }

    cusparseDestroyMatDescr(descr);
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

    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel_copy_reference, 0, 0);

    unsigned max_bricks = ((unsigned)occupied_brick_bytes) / sizeof(unsigned);
    size_t gridSize = (max_bricks + blockSize - 1) / blockSize;

    kernel_copy_reference<<<gridSize, blockSize>>>(brick_list, max_bricks);

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

    //    checkCudaErrors(cudaMalloc(&_jtj, ED_GRAPH_NODE_COUNT * 15 * ED_GRAPH_NODE_COUNT * 15 * sizeof(float)));
    //    checkCudaErrors(cudaMalloc(&_jtf, ED_GRAPH_NODE_COUNT * 15 * sizeof(float)));
    //    checkCudaErrors(cudaMalloc(&_h, ED_GRAPH_NODE_COUNT * 15 * sizeof(float)));
    //    checkCudaErrors(cudaMalloc(&_ed_graph, ED_GRAPH_NODE_COUNT * sizeof(struct_ed_node)));

    size_t vx_bytes;
    GLuint *vx_counter;
    struct_vertex *vx_ptr;

    checkCudaErrors(cudaGraphicsMapResources(1, &cgr_buffer_vertex_counter, 0));
    checkCudaErrors(cudaGraphicsMapResources(1, &cgr_buffer_reference_mesh_vertices, 0));

    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&vx_counter, &vx_bytes, cgr_buffer_vertex_counter));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&vx_ptr, &vx_bytes, cgr_buffer_reference_mesh_vertices));

    // printf("\nvx_bytes: %zu\n", vx_bytes);

    kernel_sample_ed_nodes<<<dim3(4, 4, 4), dim3(2, 2, 2)>>>(vx_counter, vx_ptr, _ed_graph);

    getLastCudaError("render kernel failed");

    cudaDeviceSynchronize();

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cgr_buffer_reference_mesh_vertices, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cgr_buffer_vertex_counter, 0));
}

extern "C" void align_non_rigid()
{
    cudaArray *volume_array_tsdf_data = nullptr;

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);

    size_t vx_bytes;
    GLuint *vx_counter;
    struct_vertex *vx_ptr;

    size_t brick_bytes;
    GLuint *brick_list;

    checkCudaErrors(cudaGraphicsMapResources(1, &cgr_buffer_vertex_counter, 0));
    checkCudaErrors(cudaGraphicsMapResources(1, &cgr_buffer_reference_mesh_vertices, 0));
    checkCudaErrors(cudaGraphicsMapResources(1, &cgr_buffer_occupied, 0));
    checkCudaErrors(cudaGraphicsMapResources(1, &cgr_volume_tsdf_data, 0));

    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&vx_counter, &vx_bytes, cgr_buffer_vertex_counter));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&vx_ptr, &vx_bytes, cgr_buffer_reference_mesh_vertices));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&brick_list, &brick_bytes, cgr_buffer_occupied));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&volume_array_tsdf_data, cgr_volume_tsdf_data, 0, 0));

    checkCudaErrors(cudaBindSurfaceToArray(&_volume_tsdf_data, volume_array_tsdf_data, &channel_desc));
    checkCudaErrors(cudaBindSurfaceToArray(&_volume_tsdf_ref, _volume_array_tsdf_ref, &channel_desc));

    // kernel_jtj_jtf<<<dim3(8, 8, 8), dim3(4, 4, 4)>>>(_jtj, _jtf, vx_counter, vx_ptr, _ed_graph);

    getLastCudaError("render kernel failed");

    cudaDeviceSynchronize();

    // solve_for_h();

    getLastCudaError("render kernel failed");

    cudaDeviceSynchronize();

    unsigned max_bricks = ((unsigned)brick_bytes) / sizeof(unsigned);

    int blockSize;
    int minGridSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel_copy_reference, 0, 0);

    size_t gridSize = (max_bricks + blockSize - 1) / blockSize;

    kernel_fuse_volume<<<gridSize, blockSize>>>(brick_list, max_bricks);

    getLastCudaError("render kernel failed");

    cudaDeviceSynchronize();

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cgr_volume_tsdf_data, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cgr_buffer_occupied, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cgr_buffer_reference_mesh_vertices, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cgr_buffer_vertex_counter, 0));
}
