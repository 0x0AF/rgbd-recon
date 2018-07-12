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
    cudaGraphicsResource *buffer_ed_nodes_debug{nullptr};
    cudaGraphicsResource *buffer_sorted_vertices_debug{nullptr};

    cudaGraphicsResource *texture_kinect_rgbs{nullptr};
    cudaGraphicsResource *texture_kinect_depths{nullptr};
    cudaGraphicsResource *texture_kinect_silhouettes{nullptr};

    cudaGraphicsResource *volume_cv_xyz_inv[4]{nullptr, nullptr, nullptr, nullptr};
    cudaGraphicsResource *volume_cv_xyz[4]{nullptr, nullptr, nullptr, nullptr};
    cudaGraphicsResource *volume_tsdf_data{nullptr};
    cudaGraphicsResource *volume_tsdf_ref{nullptr};
};

struct_graphic_resources _cgr;

struct struct_device_resources
{
    unsigned int *bricks_dense_index = nullptr;
    unsigned int *bricks_inv_index = nullptr;
    struct_ed_node *ed_graph = nullptr;
    struct_ed_meta_entry *ed_graph_meta = nullptr;
    struct_vertex *sorted_vx_ptr = nullptr;
    struct_vertex_weights *sorted_vx_weights = nullptr;

    float *jtj_vals = nullptr;
    int *jtj_rows = nullptr;
    int *jtj_cols = nullptr;
    float *jtf = nullptr;
    float *h = nullptr;

    float *pcg_p = nullptr;
    float *pcg_omega = nullptr;
    float *pcg_Ax = nullptr;
};

struct_device_resources _dev_res;

struct struct_host_resources
{
    struct_measures measures;

    unsigned int active_bricks_count = 0u;
    unsigned int active_ed_nodes_count = 0u;
    unsigned long active_ed_vx_count = 0u;
};

struct_host_resources _host_res;

cublasHandle_t cublas_handle = nullptr;
cusparseHandle_t cusparse_handle = nullptr;
cusolverSpHandle_t cusolver_handle = nullptr;

surface<void, cudaSurfaceType3D> _volume_tsdf_data;
surface<void, cudaSurfaceType3D> _volume_tsdf_ref;

surface<void, cudaSurfaceType3D> _kinect_rgbs;
surface<void, cudaSurfaceType3D> _kinect_depths;
surface<void, cudaSurfaceType3D> _kinect_silhouettes;

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
    if(_dev_res.bricks_inv_index != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.bricks_inv_index));
    }

    if(_dev_res.bricks_dense_index != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.bricks_dense_index));
    }
}
__host__ void free_ed_resources()
{
    if(_dev_res.sorted_vx_ptr != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.sorted_vx_ptr));
    }

    if(_dev_res.sorted_vx_weights != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.sorted_vx_weights));
    }

    if(_dev_res.ed_graph != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.ed_graph));
    }

    if(_dev_res.ed_graph_meta)
    {
        checkCudaErrors(cudaFree(_dev_res.ed_graph_meta));
    }

    if(_dev_res.jtf != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.jtf));
    }

    if(_dev_res.jtj_vals != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.jtj_vals));
    }

    if(_dev_res.jtj_rows != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.jtj_rows));
    }

    if(_dev_res.jtj_cols != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.jtj_cols));
    }

    if(_dev_res.h != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.h));
    }

    if(_dev_res.pcg_Ax != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.pcg_Ax));
    }

    if(_dev_res.pcg_omega != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.pcg_omega));
    }

    if(_dev_res.pcg_p != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.pcg_p));
    }
}
__host__ void allocate_brick_resources()
{
    checkCudaErrors(cudaMalloc(&_dev_res.bricks_inv_index, _host_res.measures.data_volume_num_bricks * sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc(&_dev_res.bricks_dense_index, _host_res.measures.data_volume_num_bricks * sizeof(unsigned int)));

    checkCudaErrors(cudaMemset(_dev_res.bricks_inv_index, 0, _host_res.measures.data_volume_num_bricks * sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(_dev_res.bricks_dense_index, 0, _host_res.measures.data_volume_num_bricks * sizeof(unsigned int)));
}
__host__ void allocate_ed_resources()
{
    checkCudaErrors(cudaMalloc(&_dev_res.sorted_vx_ptr, _host_res.active_ed_vx_count * sizeof(struct_vertex)));
    checkCudaErrors(cudaMalloc(&_dev_res.sorted_vx_weights, _host_res.active_ed_vx_count * sizeof(struct_vertex_weights)));
    checkCudaErrors(cudaMalloc(&_dev_res.ed_graph, _host_res.active_ed_nodes_count * sizeof(struct_ed_node)));

    checkCudaErrors(cudaMalloc(&_dev_res.jtj_vals, _host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT * sizeof(float)));
    checkCudaErrors(cudaMalloc(&_dev_res.jtj_rows, _host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT * sizeof(int)));
    checkCudaErrors(cudaMalloc(&_dev_res.jtj_cols, _host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT * sizeof(int)));

    checkCudaErrors(cudaMalloc(&_dev_res.jtf, _host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * sizeof(float)));
    checkCudaErrors(cudaMalloc(&_dev_res.h, _host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * sizeof(float)));

    checkCudaErrors(cudaMalloc(&_dev_res.pcg_p, _host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * sizeof(float)));
    checkCudaErrors(cudaMalloc(&_dev_res.pcg_omega, _host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * sizeof(float)));
    checkCudaErrors(cudaMalloc(&_dev_res.pcg_Ax, _host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * sizeof(float)));

    checkCudaErrors(cudaMemset(_dev_res.sorted_vx_ptr, 0, _host_res.active_ed_vx_count * sizeof(struct_vertex)));
    checkCudaErrors(cudaMemset(_dev_res.sorted_vx_weights, 0, _host_res.active_ed_vx_count * sizeof(struct_vertex_weights)));
    checkCudaErrors(cudaMemset(_dev_res.ed_graph, 0, _host_res.active_ed_nodes_count * sizeof(struct_ed_node)));

    checkCudaErrors(cudaMemset(_dev_res.jtj_vals, 0, _host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT * sizeof(float)));
    checkCudaErrors(cudaMemset(_dev_res.jtj_rows, 0, _host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT * sizeof(int)));
    checkCudaErrors(cudaMemset(_dev_res.jtj_cols, 0, _host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * ED_COMPONENT_COUNT * sizeof(int)));

    checkCudaErrors(cudaMemset(_dev_res.jtf, 0, _host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * sizeof(float)));
    checkCudaErrors(cudaMemset(_dev_res.h, 0, _host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * sizeof(float)));

    checkCudaErrors(cudaMemset(_dev_res.pcg_p, 0, _host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * sizeof(float)));
    checkCudaErrors(cudaMemset(_dev_res.pcg_omega, 0, _host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * sizeof(float)));
    checkCudaErrors(cudaMemset(_dev_res.pcg_Ax, 0, _host_res.active_ed_nodes_count * ED_COMPONENT_COUNT * sizeof(float)));
}

#endif // RECON_PC_CUDA_RESOURCES