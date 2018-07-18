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
    float4 *kinect_rgbs = nullptr;
    float2 *kinect_depths = nullptr;
    float *kinect_silhouettes = nullptr;

    size_t mapped_bytes_kinect_arrays[3] = {0,0,0};
    float4 *mapped_pbo_rgbs = nullptr;
    float2 *mapped_pbo_depths = nullptr;
    float1 *mapped_pbo_silhouettes = nullptr;

    unsigned int *bricks_dense_index = nullptr;
    unsigned int *bricks_inv_index = nullptr;
    struct_ed_node *ed_graph = nullptr;
    struct_ed_meta_entry *ed_graph_meta = nullptr;
    struct_vertex *sorted_vx_ptr = nullptr;

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

    float *kernel_gauss;
};

struct_host_resources _host_res;

cublasHandle_t cublas_handle = nullptr;
cusparseHandle_t cusparse_handle = nullptr;
cusolverSpHandle_t cusolver_handle = nullptr;

surface<void, cudaSurfaceType3D> _volume_tsdf_data;
surface<void, cudaSurfaceType3D> _volume_tsdf_ref;

texture<float4, cudaTextureType3D, cudaReadModeElementType> _volume_cv_xyz_inv_0;
texture<float4, cudaTextureType3D, cudaReadModeElementType> _volume_cv_xyz_inv_1;
texture<float4, cudaTextureType3D, cudaReadModeElementType> _volume_cv_xyz_inv_2;
texture<float4, cudaTextureType3D, cudaReadModeElementType> _volume_cv_xyz_inv_3;

texture<float4, cudaTextureType3D, cudaReadModeElementType> _volume_cv_xyz_0;
texture<float4, cudaTextureType3D, cudaReadModeElementType> _volume_cv_xyz_1;
texture<float4, cudaTextureType3D, cudaReadModeElementType> _volume_cv_xyz_2;
texture<float4, cudaTextureType3D, cudaReadModeElementType> _volume_cv_xyz_3;

void map_calibration_volumes()
{
    _volume_cv_xyz_inv_0.normalized = true;
    _volume_cv_xyz_inv_0.filterMode = cudaFilterModePoint;
    _volume_cv_xyz_inv_0.addressMode[0] = cudaAddressModeClamp;
    _volume_cv_xyz_inv_0.addressMode[1] = cudaAddressModeClamp;
    _volume_cv_xyz_inv_0.addressMode[2] = cudaAddressModeClamp;

    _volume_cv_xyz_inv_1.normalized = true;
    _volume_cv_xyz_inv_1.filterMode = cudaFilterModePoint;
    _volume_cv_xyz_inv_1.addressMode[0] = cudaAddressModeClamp;
    _volume_cv_xyz_inv_1.addressMode[1] = cudaAddressModeClamp;
    _volume_cv_xyz_inv_1.addressMode[2] = cudaAddressModeClamp;

    _volume_cv_xyz_inv_2.normalized = true;
    _volume_cv_xyz_inv_2.filterMode = cudaFilterModePoint;
    _volume_cv_xyz_inv_2.addressMode[0] = cudaAddressModeClamp;
    _volume_cv_xyz_inv_2.addressMode[1] = cudaAddressModeClamp;
    _volume_cv_xyz_inv_2.addressMode[2] = cudaAddressModeClamp;

    _volume_cv_xyz_inv_3.normalized = true;
    _volume_cv_xyz_inv_3.filterMode = cudaFilterModePoint;
    _volume_cv_xyz_inv_3.addressMode[0] = cudaAddressModeClamp;
    _volume_cv_xyz_inv_3.addressMode[1] = cudaAddressModeClamp;
    _volume_cv_xyz_inv_3.addressMode[2] = cudaAddressModeClamp;

    _volume_cv_xyz_0.normalized = true;
    _volume_cv_xyz_0.filterMode = cudaFilterModePoint;
    _volume_cv_xyz_0.addressMode[0] = cudaAddressModeClamp;
    _volume_cv_xyz_0.addressMode[1] = cudaAddressModeClamp;
    _volume_cv_xyz_0.addressMode[2] = cudaAddressModeClamp;

    _volume_cv_xyz_1.normalized = true;
    _volume_cv_xyz_1.filterMode = cudaFilterModePoint;
    _volume_cv_xyz_1.addressMode[0] = cudaAddressModeClamp;
    _volume_cv_xyz_1.addressMode[1] = cudaAddressModeClamp;
    _volume_cv_xyz_1.addressMode[2] = cudaAddressModeClamp;

    _volume_cv_xyz_2.normalized = true;
    _volume_cv_xyz_2.filterMode = cudaFilterModePoint;
    _volume_cv_xyz_2.addressMode[0] = cudaAddressModeClamp;
    _volume_cv_xyz_2.addressMode[1] = cudaAddressModeClamp;
    _volume_cv_xyz_2.addressMode[2] = cudaAddressModeClamp;

    _volume_cv_xyz_3.normalized = true;
    _volume_cv_xyz_3.filterMode = cudaFilterModePoint;
    _volume_cv_xyz_3.addressMode[0] = cudaAddressModeClamp;
    _volume_cv_xyz_3.addressMode[1] = cudaAddressModeClamp;
    _volume_cv_xyz_3.addressMode[2] = cudaAddressModeClamp;

    checkCudaErrors(cudaGraphicsMapResources(4, _cgr.volume_cv_xyz_inv, 0));
    checkCudaErrors(cudaGraphicsMapResources(4, _cgr.volume_cv_xyz, 0));

    cudaArray *volume_array_cv_xyz_inv[4] = {nullptr, nullptr, nullptr, nullptr};
    cudaArray *volume_array_cv_xyz[4] = {nullptr, nullptr, nullptr, nullptr};

    for(unsigned int i = 0; i < 4; i++)
    {
        checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&volume_array_cv_xyz_inv[i], _cgr.volume_cv_xyz_inv[i], 0, 0));
        checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&volume_array_cv_xyz[i], _cgr.volume_cv_xyz[i], 0, 0));
    }

    cudaChannelFormatDesc channel_desc_cv_xyz_inv = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    checkCudaErrors(cudaBindTextureToArray(&_volume_cv_xyz_inv_0, volume_array_cv_xyz_inv[0], &channel_desc_cv_xyz_inv));
    checkCudaErrors(cudaBindTextureToArray(&_volume_cv_xyz_inv_1, volume_array_cv_xyz_inv[1], &channel_desc_cv_xyz_inv));
    checkCudaErrors(cudaBindTextureToArray(&_volume_cv_xyz_inv_2, volume_array_cv_xyz_inv[2], &channel_desc_cv_xyz_inv));
    checkCudaErrors(cudaBindTextureToArray(&_volume_cv_xyz_inv_3, volume_array_cv_xyz_inv[3], &channel_desc_cv_xyz_inv));

    cudaChannelFormatDesc channel_desc_cv_xyz = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    checkCudaErrors(cudaBindTextureToArray(&_volume_cv_xyz_0, volume_array_cv_xyz[0], &channel_desc_cv_xyz));
    checkCudaErrors(cudaBindTextureToArray(&_volume_cv_xyz_1, volume_array_cv_xyz[1], &channel_desc_cv_xyz));
    checkCudaErrors(cudaBindTextureToArray(&_volume_cv_xyz_2, volume_array_cv_xyz[2], &channel_desc_cv_xyz));
    checkCudaErrors(cudaBindTextureToArray(&_volume_cv_xyz_3, volume_array_cv_xyz[3], &channel_desc_cv_xyz));
}

void unmap_calibration_volumes()
{
    cudaUnbindTexture(&_volume_cv_xyz_inv_0);
    cudaUnbindTexture(&_volume_cv_xyz_inv_1);
    cudaUnbindTexture(&_volume_cv_xyz_inv_2);
    cudaUnbindTexture(&_volume_cv_xyz_inv_3);

    cudaUnbindTexture(&_volume_cv_xyz_0);
    cudaUnbindTexture(&_volume_cv_xyz_1);
    cudaUnbindTexture(&_volume_cv_xyz_2);
    cudaUnbindTexture(&_volume_cv_xyz_3);

    checkCudaErrors(cudaGraphicsUnmapResources(4, _cgr.volume_cv_xyz, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(4, _cgr.volume_cv_xyz_inv, 0));
}

void map_tsdf_volumes()
{
    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.volume_tsdf_data, 0));
    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.volume_tsdf_ref, 0));

    cudaArray *volume_array_tsdf_data = nullptr;
    cudaArray *volume_array_tsdf_ref = nullptr;
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&volume_array_tsdf_data, _cgr.volume_tsdf_data, 0, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&volume_array_tsdf_ref, _cgr.volume_tsdf_ref, 0, 0));

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
    checkCudaErrors(cudaBindSurfaceToArray(&_volume_tsdf_data, volume_array_tsdf_data, &channel_desc));
    checkCudaErrors(cudaBindSurfaceToArray(&_volume_tsdf_ref, volume_array_tsdf_ref, &channel_desc));
}

void unmap_tsdf_volumes()
{
    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.volume_tsdf_data, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.volume_tsdf_ref, 0));
}

void map_kinect_arrays()
{
    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.texture_kinect_rgbs, 0));
    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.texture_kinect_depths, 0));
    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.texture_kinect_silhouettes, 0));

    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&_dev_res.mapped_pbo_rgbs, &_dev_res.mapped_bytes_kinect_arrays[0], _cgr.texture_kinect_rgbs));
    checkCudaErrors(cudaMemcpy(&_dev_res.kinect_rgbs[0], &_dev_res.mapped_pbo_rgbs[0], _dev_res.mapped_bytes_kinect_arrays[0], cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&_dev_res.mapped_pbo_depths, &_dev_res.mapped_bytes_kinect_arrays[1], _cgr.texture_kinect_depths));
    checkCudaErrors(cudaMemcpy(&_dev_res.kinect_depths[0], &_dev_res.mapped_pbo_depths[0], _dev_res.mapped_bytes_kinect_arrays[1], cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&_dev_res.mapped_pbo_silhouettes, &_dev_res.mapped_bytes_kinect_arrays[2], _cgr.texture_kinect_silhouettes));
    checkCudaErrors(cudaMemcpy(&_dev_res.kinect_silhouettes[0], &_dev_res.mapped_pbo_silhouettes[0], _dev_res.mapped_bytes_kinect_arrays[2], cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
}

void unmap_kinect_arrays()
{
    checkCudaErrors(cudaMemcpy(&_dev_res.mapped_pbo_silhouettes[0], &_dev_res.kinect_silhouettes[0], _dev_res.mapped_bytes_kinect_arrays[2], cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.texture_kinect_silhouettes));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.texture_kinect_depths));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.texture_kinect_rgbs));
}

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