#ifndef RECON_PC_CUDA_RESOURCES
#define RECON_PC_CUDA_RESOURCES

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cublas_v2.h>
#include <cuda_gl_interop.h>
#include <cusolverSp.h>
#include <cusolver_common.h>
#include <cusparse_v2.h>
#include <reconstruction/cuda/glm.cuh>
#include <reconstruction/cuda/structures.cuh>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <helper_cusolver.h>

struct struct_graphic_resources
{
    cudaGraphicsResource *buffer_reference_mesh_vertices{nullptr};
    cudaGraphicsResource *buffer_vertex_counter{nullptr};
    cudaGraphicsResource *buffer_bricks{nullptr};
    cudaGraphicsResource *buffer_occupied{nullptr};

    cudaGraphicsResource *pbo_kinect_rgbs{nullptr};
    cudaGraphicsResource *pbo_kinect_depths{nullptr};
    cudaGraphicsResource *pbo_kinect_silhouettes{nullptr};

    cudaGraphicsResource *volume_cv_xyz_inv[4]{nullptr, nullptr, nullptr, nullptr};
    cudaGraphicsResource *volume_cv_xyz[4]{nullptr, nullptr, nullptr, nullptr};
    cudaGraphicsResource *volume_tsdf_data{nullptr};
};

cudaArray *_volume_array_tsdf_ref = nullptr;

struct_graphic_resources _cgr;

unsigned int *_bricks_dense_index = nullptr;
unsigned int *_bricks_inv_index = nullptr;
struct_ed_node *_ed_graph = nullptr;
struct_ed_meta_entry *_ed_graph_meta = nullptr;
struct_vertex *_sorted_vx_ptr = nullptr;
struct_vertex_weights *_sorted_vx_weights = nullptr;

struct_measures *_measures = nullptr;

cublasHandle_t cublas_handle = nullptr;
cusparseHandle_t cusparse_handle = nullptr;
cusolverSpHandle_t cusolver_handle = nullptr;

unsigned int _active_bricks_count = 0u;
unsigned int _active_ed_nodes_count = 0u;
unsigned long _active_ed_vx_count = 0u;

float *_jtj_vals = nullptr;
int *_jtj_rows = nullptr;
int *_jtj_cols = nullptr;
float *_jtf = nullptr;
float *_h = nullptr;

float *pcg_p = nullptr;
float *pcg_omega = nullptr;
float *pcg_Ax = nullptr;

surface<void, cudaSurfaceType3D> _volume_tsdf_data;
surface<void, cudaSurfaceType3D> _volume_tsdf_ref;

surface<void, cudaSurfaceType3D> _volume_cv_xyz_inv_0;
surface<void, cudaSurfaceType3D> _volume_cv_xyz_inv_1;
surface<void, cudaSurfaceType3D> _volume_cv_xyz_inv_2;
surface<void, cudaSurfaceType3D> _volume_cv_xyz_inv_3;

surface<void, cudaSurfaceType3D> _volume_cv_xyz_0;
surface<void, cudaSurfaceType3D> _volume_cv_xyz_1;
surface<void, cudaSurfaceType3D> _volume_cv_xyz_2;
surface<void, cudaSurfaceType3D> _volume_cv_xyz_3;

__host__ void free_brick_resources()
{
    if(_bricks_inv_index != nullptr)
    {
        checkCudaErrors(cudaFree(_bricks_inv_index));
    }

    if(_bricks_dense_index != nullptr)
    {
        checkCudaErrors(cudaFree(_bricks_dense_index));
    }
}
__host__ void free_ed_resources()
{
    if(_sorted_vx_ptr != nullptr)
    {
        checkCudaErrors(cudaFree(_sorted_vx_ptr));
    }

    if(_sorted_vx_weights != nullptr)
    {
        checkCudaErrors(cudaFree(_sorted_vx_weights));
    }

    if(_ed_graph != nullptr)
    {
        checkCudaErrors(cudaFree(_ed_graph));
    }

    if(_ed_graph_meta)
    {
        checkCudaErrors(cudaFree(_ed_graph_meta));
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
}
__host__ void allocate_brick_resources()
{
    checkCudaErrors(cudaMalloc((void **)&_bricks_inv_index, BRICK_RES_X * BRICK_RES_Y * BRICK_RES_Z * sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void **)&_bricks_dense_index, BRICK_RES_X * BRICK_RES_Y * BRICK_RES_Z * sizeof(unsigned int)));
}
__host__ void allocate_ed_resources()
{
    checkCudaErrors(cudaMalloc(&_sorted_vx_ptr, _active_ed_vx_count * sizeof(struct_vertex)));
    checkCudaErrors(cudaMalloc(&_sorted_vx_weights, _active_ed_vx_count * sizeof(struct_vertex_weights)));
    checkCudaErrors(cudaMalloc(&_ed_graph, _active_ed_nodes_count * sizeof(struct_ed_node)));

    checkCudaErrors(cudaMalloc(&_jtj_vals, _active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT * sizeof(float)));
    checkCudaErrors(cudaMalloc(&_jtj_rows, _active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT * sizeof(int)));
    checkCudaErrors(cudaMalloc(&_jtj_cols, _active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT * sizeof(int)));

    checkCudaErrors(cudaMalloc(&_jtf, _active_ed_nodes_count * ED_COMPONENT_COUNT * sizeof(float)));
    checkCudaErrors(cudaMalloc(&_h, _active_ed_nodes_count * ED_COMPONENT_COUNT * sizeof(float)));

    checkCudaErrors(cudaMalloc(&pcg_p, _active_ed_nodes_count * ED_COMPONENT_COUNT * sizeof(float)));
    checkCudaErrors(cudaMalloc(&pcg_omega, _active_ed_nodes_count * ED_COMPONENT_COUNT * sizeof(float)));
    checkCudaErrors(cudaMalloc(&pcg_Ax, _active_ed_nodes_count * ED_COMPONENT_COUNT * sizeof(float)));

    cudaMemset(_sorted_vx_ptr, 0, _active_ed_vx_count * sizeof(struct_vertex));
    cudaMemset(_ed_graph, 0, _active_ed_nodes_count * sizeof(struct_ed_node));

    cudaMemset(_jtj_vals, 0, _active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT * sizeof(float));
    cudaMemset(_jtj_rows, 0, _active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT * sizeof(int));
    cudaMemset(_jtj_cols, 0, _active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT * sizeof(int));

    cudaMemset(_jtf, 0, _active_ed_nodes_count * ED_COMPONENT_COUNT * sizeof(float));
    cudaMemset(_h, 0, _active_ed_nodes_count * ED_COMPONENT_COUNT * sizeof(float));

    cudaMemset(pcg_p, 0, _active_ed_nodes_count * ED_COMPONENT_COUNT * sizeof(float));
    cudaMemset(pcg_omega, 0, _active_ed_nodes_count * ED_COMPONENT_COUNT * sizeof(float));
    cudaMemset(pcg_Ax, 0, _active_ed_nodes_count * ED_COMPONENT_COUNT * sizeof(float));
}

#endif // RECON_PC_CUDA_RESOURCES