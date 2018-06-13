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
 * Identify enclosing ED cell in volume voxel space
 * */
__device__ unsigned int identify_ed_cell_pos(glm::vec3 position, unsigned int *bricks_inv_index)
{
    glm::uvec3 pos_voxel_space = glm::uvec3(position);
    glm::uvec3 brick_index3d = pos_voxel_space / BRICK_VOXEL_DIM;

    unsigned int brick_id = brick_index3d.z * BRICK_RES_Y * BRICK_RES_Z + brick_index3d.y * BRICK_RES_Y + brick_index3d.x;

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

__global__ void kernel_sample_ed_nodes(GLuint *vx_counter, struct_vertex *vx_ptr, unsigned int *bricks_inv_index, struct_ed_node *ed_graph, unsigned int ed_node_countt)
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

        glm::vec3 position_voxel_space = vx_ptr[vertex_position].position * glm::vec3(VOLUME_VOXEL_DIM_X, VOLUME_VOXEL_DIM_Y, VOLUME_VOXEL_DIM_Z);

        unsigned int ed_cell_pos = identify_ed_cell_pos(position_voxel_space, bricks_inv_index);

        if(ed_cell_pos >= ed_node_countt)
        {
            continue;
        }

        vx_ptr[vertex_position].ed_cell_id = ed_cell_pos;

        if(ed_graph[ed_cell_pos].position.z * ED_CELL_RES * ED_CELL_RES + ed_graph[ed_cell_pos].position.y * ED_CELL_RES + ed_graph[ed_cell_pos].position.x == 0u)
        {
            ed_graph[ed_cell_pos].position = position_voxel_space;
        }
        else
        {
            // TODO: incremental centroid instead of averaging
            ed_graph[ed_cell_pos].position = (ed_graph[ed_cell_pos].position + position_voxel_space) / 2.0f;
        }
    }
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

    if(_jtj_vals != nullptr)
    {
        checkCudaErrors(cudaFree(_jtj_vals));
    }

    if(_jtj_rows != nullptr)
    {
        checkCudaErrors(cudaFree(_jtj_rows));
    }

    if(_jtj_cols != nullptr)
    {
        checkCudaErrors(cudaFree(_jtj_cols));
    }

    if(_h != nullptr)
    {
        checkCudaErrors(cudaFree(_h));
    }

    if(pcg_Ax != nullptr)
    {
        checkCudaErrors(cudaFree(pcg_Ax));
    }

    if(pcg_omega != nullptr)
    {
        checkCudaErrors(cudaFree(pcg_omega));
    }

    if(pcg_p != nullptr)
    {
        checkCudaErrors(cudaFree(pcg_p));
    }

    cudaDeviceSynchronize();

    checkCudaErrors(cudaMalloc(&_ed_graph, _ed_nodes_count * sizeof(struct_ed_node)));

    checkCudaErrors(cudaMalloc(&_jtj_vals, _ed_nodes_component_count * ED_COMPONENT_COUNT * sizeof(float)));
    checkCudaErrors(cudaMalloc(&_jtj_rows, _ed_nodes_component_count * ED_COMPONENT_COUNT * sizeof(int)));
    checkCudaErrors(cudaMalloc(&_jtj_cols, _ed_nodes_component_count * ED_COMPONENT_COUNT * sizeof(int)));

    checkCudaErrors(cudaMalloc(&_jtf, _ed_nodes_component_count * sizeof(float)));
    checkCudaErrors(cudaMalloc(&_h, _ed_nodes_component_count * sizeof(float)));

    checkCudaErrors(cudaMalloc(&pcg_p, _ed_nodes_component_count * sizeof(float)));
    checkCudaErrors(cudaMalloc(&pcg_omega, _ed_nodes_component_count * sizeof(float)));
    checkCudaErrors(cudaMalloc(&pcg_Ax, _ed_nodes_component_count * sizeof(float)));

    cudaMemset(_ed_graph, 0, _ed_nodes_component_count * sizeof(float));

    cudaMemset(_jtj_vals, 0, _ed_nodes_component_count * ED_COMPONENT_COUNT * sizeof(float));
    cudaMemset(_jtj_rows, 0, _ed_nodes_component_count * ED_COMPONENT_COUNT * sizeof(int));
    cudaMemset(_jtj_cols, 0, _ed_nodes_component_count * ED_COMPONENT_COUNT * sizeof(int));

    cudaMemset(_jtf, 0, _ed_nodes_component_count * sizeof(float));
    cudaMemset(_h, 0, _ed_nodes_component_count * sizeof(float));

    cudaMemset(pcg_p, 0, _ed_nodes_component_count * sizeof(float));
    cudaMemset(pcg_omega, 0, _ed_nodes_component_count * sizeof(float));
    cudaMemset(pcg_Ax, 0, _ed_nodes_component_count * sizeof(float));

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
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_sample_ed_nodes, 0, 0);

    unsigned max_vertices = ((unsigned)vx_bytes) / sizeof(struct_vertex);
    size_t grid_size = (max_vertices + block_size - 1) / block_size;
    kernel_sample_ed_nodes<<<grid_size, block_size>>>(vx_counter, vx_ptr, _bricks_inv_index, _ed_graph, _ed_nodes_count);

    getLastCudaError("render kernel failed");

    cudaDeviceSynchronize();

    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.buffer_reference_mesh_vertices, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.buffer_vertex_counter, 0));
}