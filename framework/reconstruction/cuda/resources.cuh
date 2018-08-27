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
#include <reconstruction/cuda/MC/marchingCubes_kernel.cuh>
#include <reconstruction/cuda/glm.cuh>
#include <reconstruction/cuda/structures.cuh>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <helper_cusolver.h>

#define UMUL(a, b) ((a) * (b))
#define UMAD(a, b, c) (UMUL((a), (b)) + (c))

struct struct_graphic_resources
{
    cudaGraphicsResource *buffer_bricks{nullptr};
    cudaGraphicsResource *buffer_occupied{nullptr};
    cudaGraphicsResource *buffer_ed_nodes_debug{nullptr};
    cudaGraphicsResource *buffer_sorted_vertices_debug{nullptr};
    cudaGraphicsResource *buffer_correspondences_debug{nullptr};

    cudaGraphicsResource *pbo_kinect_rgbs{nullptr};
    cudaGraphicsResource *pbo_kinect_depths{nullptr};
    cudaGraphicsResource *pbo_kinect_silhouettes{nullptr};

    cudaGraphicsResource *pbo_kinect_silhouettes_debug{nullptr};

    cudaGraphicsResource *pbo_cv_xyz_inv[4]{nullptr, nullptr, nullptr, nullptr};
    cudaGraphicsResource *pbo_cv_xyz[4]{nullptr, nullptr, nullptr, nullptr};

    cudaGraphicsResource *volume_tsdf_data{nullptr};
    cudaGraphicsResource *volume_tsdf_ref_grad{nullptr};

    cudaGraphicsResource *posvbo{nullptr};
    cudaGraphicsResource *normalvbo{nullptr};
};

struct_graphic_resources _cgr;

struct struct_device_resources
{
    /// Lab space colors
    float4 *kinect_rgbs = nullptr;

    /// Merged grayscale colors
    float *kinect_intens = nullptr;

    /// Current depth frames
    float2 *kinect_depths = nullptr;

    /// Previous depth frames
    float2 *kinect_depths_prev = nullptr;

    /// (Smooth) silhouettes
    float *kinect_silhouettes = nullptr;

    cudaTextureObject_t depth_tex[4] = {0, 0, 0, 0};
    cudaTextureObject_t depth_tex_prev[4] = {0, 0, 0, 0};
    cudaTextureObject_t silhouette_tex[4] = {0, 0, 0, 0};

    /// Mapped pointers to GL resources
    float4 *mapped_pbo_rgbs = nullptr;
    float2 *mapped_pbo_depths = nullptr;
    float1 *mapped_pbo_silhouettes = nullptr;
    float1 *mapped_pbo_silhouettes_debug = nullptr;
    float4 *mapped_pbo_cv_xyz_inv[4] = {nullptr, nullptr, nullptr, nullptr};
    float4 *mapped_pbo_cv_xyz[4] = {nullptr, nullptr, nullptr, nullptr};

    /// CUDA-side calibration volumes
    cudaArray *cv_xyz[4] = {nullptr, nullptr, nullptr, nullptr};
    cudaArray *cv_xyz_inv[4] = {nullptr, nullptr, nullptr, nullptr};
    cudaTextureObject_t cv_xyz_tex[4] = {0, 0, 0, 0};
    cudaTextureObject_t cv_xyz_inv_tex[4] = {0, 0, 0, 0};

    /// Dense brick index
    unsigned int *bricks_dense_index = nullptr;

    /// Inverted brick index
    unsigned int *bricks_inv_index = nullptr;

    /// Embedded deformation graph
    struct_ed_node *ed_graph = nullptr;

    /// Embedded deformation graph meta data
    struct_ed_meta_entry *ed_graph_meta = nullptr;

    /// Embedded deformation graph step (for energy evaluation)
    struct_ed_node *ed_graph_step = nullptr;

    /// Unsorted vertex pointer
    struct_vertex *unsorted_vx_ptr = nullptr;

    /// ED-sorted vertex pointer
    struct_vertex *sorted_vx_ptr = nullptr;

    /// ED-sorted vertex projections
    struct_projection *sorted_vx_projections = nullptr;

    /// Warped ED-sorted vertex pointer
    struct_vertex *warped_sorted_vx_ptr = nullptr;

    /// Warped ED-sorted vertex projections
    struct_projection *warped_sorted_vx_projections = nullptr;

#ifdef PIPELINE_CORRESPONDENCE
    struct_correspondence *sorted_correspondences = nullptr;
    struct_correspondence *unsorted_correspondences = nullptr;
    unsigned int *depth_cell_counter = nullptr;
    struct_depth_cell_meta *depth_cell_meta = nullptr;
#endif

    /// JTJ sparse pointers
    float *jtj_vals = nullptr;
    int *jtj_rows = nullptr;
    int *jtj_rows_csr = nullptr;
    int *jtj_cols = nullptr;

    /// JTF
    float *jtf = nullptr;

    /// Solver step
    float *h = nullptr;

    /// PCG allocations
    float *pcg_p = nullptr;
    float *pcg_omega = nullptr;
    float *pcg_Ax = nullptr;

    /// MC tables
    uint *edge_table = nullptr;
    uint *tri_table = nullptr;
    uint *num_verts_table = nullptr;

    /// MC allocations
    uint *voxel_verts = nullptr;
    uint *voxel_verts_scan = nullptr;
    uint *voxel_occupied = nullptr;
    uint *voxel_occupied_scan = nullptr;
    uint *comp_voxel_array = nullptr;

    /// TSDF volumes (R <- TSDF, G <- Weight)
    float2 *tsdf_ref = nullptr;
    float2 *tsdf_ref_warped = nullptr;

    /// 8-bit TSDF volumes for surface extraction
    uchar *out_tsdf_data = nullptr;
    uchar *out_tsdf_ref = nullptr;
    uchar *out_tsdf_warped_ref = nullptr;

    /// MC outputs
    float4 *pos = 0;
    float4 *normal = 0;
    float4 *mapped_pos = 0;
    float4 *mapped_normal = 0;
};

struct_device_resources _dev_res;

struct struct_host_resources
{
    struct_measures measures;
    Configuration configuration;

    unsigned int active_bricks_count = 0u;
    unsigned int active_ed_nodes_count = 0u;
    unsigned long active_ed_vx_count = 0u;

    unsigned int valid_correspondences = 0u;

    uint3 grid_size;
    uint3 grid_size_shift;
    uint3 grid_size_mask;
    uint num_voxels;
    float3 voxel_size;
    uint active_voxels;
    uint total_verts;

    float *kernel_gauss;
};

struct_host_resources _host_res;

cublasHandle_t cublas_handle = nullptr;
cusparseHandle_t cusparse_handle = nullptr;
cusolverSpHandle_t cusolver_handle = nullptr;

surface<void, cudaSurfaceType3D> _volume_tsdf_data;
surface<void, cudaSurfaceType3D> _volume_tsdf_ref_grad;

__host__ void map_calibration_volumes()
{
    checkCudaErrors(cudaGraphicsMapResources(4, _cgr.pbo_cv_xyz_inv, 0));
    checkCudaErrors(cudaGraphicsMapResources(4, _cgr.pbo_cv_xyz, 0));

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    cudaExtent extent_cv_xyz = make_cudaExtent(_host_res.measures.cv_xyz_res.x, _host_res.measures.cv_xyz_res.y, _host_res.measures.cv_xyz_res.z);
    cudaExtent extent_cv_xyz_inv = make_cudaExtent(_host_res.measures.cv_xyz_inv_res.x, _host_res.measures.cv_xyz_inv_res.y, _host_res.measures.cv_xyz_inv_res.z);

    for(int i = 0; i < 4; i++)
    {
        size_t bytes = 0;
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&_dev_res.mapped_pbo_cv_xyz[i], &bytes, _cgr.pbo_cv_xyz[i]));

        float4 *h_cv_xyz = (float4 *)malloc(bytes);

        checkCudaErrors(cudaMemcpy(&h_cv_xyz[0], _dev_res.mapped_pbo_cv_xyz[i], bytes, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaMalloc3DArray(&_dev_res.cv_xyz[i], &channel_desc, extent_cv_xyz));
        checkCudaErrors(cudaDeviceSynchronize());

        cudaMemcpy3DParms cv_xyz_copy = {0};
        cv_xyz_copy.srcPos = make_cudaPos(0, 0, 0);
        cv_xyz_copy.dstPos = make_cudaPos(0, 0, 0);
        cv_xyz_copy.srcPtr = make_cudaPitchedPtr(h_cv_xyz, extent_cv_xyz.width * sizeof(float4), extent_cv_xyz.width, extent_cv_xyz.height);
        cv_xyz_copy.dstArray = _dev_res.cv_xyz[i];
        cv_xyz_copy.extent = extent_cv_xyz;
        cv_xyz_copy.kind = cudaMemcpyHostToDevice;
        checkCudaErrors(cudaMemcpy3D(&cv_xyz_copy));
        cudaDeviceSynchronize();

        free(h_cv_xyz);

        struct cudaResourceDesc cv_xyz_descr;
        memset(&cv_xyz_descr, 0, sizeof(cv_xyz_descr));
        cv_xyz_descr.resType = cudaResourceTypeArray;
        cv_xyz_descr.res.array.array = _dev_res.cv_xyz[i];

        struct cudaTextureDesc cv_xyz_tex_descr;
        memset(&cv_xyz_tex_descr, 0, sizeof(cv_xyz_tex_descr));
        cv_xyz_tex_descr.addressMode[0] = cudaAddressModeClamp;
        cv_xyz_tex_descr.addressMode[1] = cudaAddressModeClamp;
        cv_xyz_tex_descr.addressMode[2] = cudaAddressModeClamp;
        cv_xyz_tex_descr.filterMode = cudaFilterModeLinear;
        cv_xyz_tex_descr.readMode = cudaReadModeElementType;
        cv_xyz_tex_descr.normalizedCoords = 1;

        cudaCreateTextureObject(&_dev_res.cv_xyz_tex[i], &cv_xyz_descr, &cv_xyz_tex_descr, NULL);

        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&_dev_res.mapped_pbo_cv_xyz_inv[i], &bytes, _cgr.pbo_cv_xyz_inv[i]));
        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaMalloc3DArray(&_dev_res.cv_xyz_inv[i], &channel_desc, extent_cv_xyz_inv));

        float4 *h_cv_xyz_inv = (float4 *)malloc(bytes);

        checkCudaErrors(cudaMemcpy(&h_cv_xyz_inv[0], _dev_res.mapped_pbo_cv_xyz_inv[i], bytes, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaMalloc3DArray(&_dev_res.cv_xyz_inv[i], &channel_desc, extent_cv_xyz_inv));
        checkCudaErrors(cudaDeviceSynchronize());

        cudaMemcpy3DParms cv_xyz_inv_copy = {0};
        cv_xyz_inv_copy.srcPos = make_cudaPos(0, 0, 0);
        cv_xyz_inv_copy.dstPos = make_cudaPos(0, 0, 0);
        cv_xyz_inv_copy.srcPtr = make_cudaPitchedPtr(h_cv_xyz_inv, extent_cv_xyz_inv.width * sizeof(float4), extent_cv_xyz_inv.width, extent_cv_xyz_inv.height);
        cv_xyz_inv_copy.dstArray = _dev_res.cv_xyz_inv[i];
        cv_xyz_inv_copy.extent = extent_cv_xyz_inv;
        cv_xyz_inv_copy.kind = cudaMemcpyHostToDevice;
        checkCudaErrors(cudaMemcpy3D(&cv_xyz_inv_copy));
        cudaDeviceSynchronize();

        free(h_cv_xyz_inv);

        struct cudaResourceDesc cv_xyz_inv_descr;
        memset(&cv_xyz_inv_descr, 0, sizeof(cv_xyz_inv_descr));
        cv_xyz_inv_descr.resType = cudaResourceTypeArray;
        cv_xyz_inv_descr.res.array.array = _dev_res.cv_xyz_inv[i];

        struct cudaTextureDesc cv_xyz_inv_tex_descr;
        memset(&cv_xyz_inv_tex_descr, 0, sizeof(cv_xyz_inv_tex_descr));
        cv_xyz_inv_tex_descr.addressMode[0] = cudaAddressModeClamp;
        cv_xyz_inv_tex_descr.addressMode[1] = cudaAddressModeClamp;
        cv_xyz_inv_tex_descr.addressMode[2] = cudaAddressModeClamp;
        cv_xyz_inv_tex_descr.filterMode = cudaFilterModeLinear;
        cv_xyz_inv_tex_descr.readMode = cudaReadModeElementType;
        cv_xyz_inv_tex_descr.normalizedCoords = 1;

        cudaCreateTextureObject(&_dev_res.cv_xyz_inv_tex[i], &cv_xyz_inv_descr, &cv_xyz_inv_tex_descr, NULL);
    }
}

__host__ void unmap_calibration_volumes()
{
    for(int i = 0; i < 4; i++)
    {
        cudaDestroyTextureObject(_dev_res.cv_xyz_tex[i]);
        cudaFreeArray(_dev_res.cv_xyz[i]);
        cudaDestroyTextureObject(_dev_res.cv_xyz_inv_tex[i]);
        cudaFreeArray(_dev_res.cv_xyz_inv[i]);
    }

    checkCudaErrors(cudaGraphicsUnmapResources(4, _cgr.pbo_cv_xyz, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(4, _cgr.pbo_cv_xyz_inv, 0));
}

__host__ void map_kinect_textures()
{
    for(int i = 0; i < 4; i++)
    {
        struct cudaResourceDesc depth_descr;
        memset(&depth_descr, 0, sizeof(depth_descr));
        depth_descr.resType = cudaResourceTypePitch2D;
        depth_descr.res.pitch2D.devPtr = _dev_res.kinect_depths + i * _host_res.measures.depth_res.x * _host_res.measures.depth_res.y * sizeof(float2);
        depth_descr.res.pitch2D.width = _host_res.measures.depth_res.x;
        depth_descr.res.pitch2D.height = _host_res.measures.depth_res.y;
        depth_descr.res.pitch2D.pitchInBytes = _host_res.measures.depth_res.x * sizeof(float2);
        depth_descr.res.pitch2D.desc = cudaCreateChannelDesc<float2>();

        struct cudaTextureDesc depth_tex_descr;
        memset(&depth_tex_descr, 0, sizeof(depth_tex_descr));
        depth_tex_descr.addressMode[0] = cudaAddressModeClamp;
        depth_tex_descr.addressMode[1] = cudaAddressModeClamp;
        depth_tex_descr.filterMode = cudaFilterModeLinear;
        depth_tex_descr.readMode = cudaReadModeElementType;
        depth_tex_descr.normalizedCoords = 1;

        cudaCreateTextureObject(&_dev_res.depth_tex[i], &depth_descr, &depth_tex_descr, NULL);

        struct cudaResourceDesc depth_prev_descr;
        memset(&depth_prev_descr, 0, sizeof(depth_prev_descr));
        depth_prev_descr.resType = cudaResourceTypePitch2D;
        depth_prev_descr.res.pitch2D.devPtr = _dev_res.kinect_depths_prev + i * _host_res.measures.depth_res.x * _host_res.measures.depth_res.y * sizeof(float2);
        depth_prev_descr.res.pitch2D.width = _host_res.measures.depth_res.x;
        depth_prev_descr.res.pitch2D.height = _host_res.measures.depth_res.y;
        depth_prev_descr.res.pitch2D.pitchInBytes = _host_res.measures.depth_res.x * sizeof(float2);
        depth_prev_descr.res.pitch2D.desc = cudaCreateChannelDesc<float2>();

        struct cudaTextureDesc depth_prev_tex_descr;
        memset(&depth_prev_tex_descr, 0, sizeof(depth_prev_tex_descr));
        depth_prev_tex_descr.addressMode[0] = cudaAddressModeClamp;
        depth_prev_tex_descr.addressMode[1] = cudaAddressModeClamp;
        depth_prev_tex_descr.filterMode = cudaFilterModeLinear;
        depth_prev_tex_descr.readMode = cudaReadModeElementType;
        depth_prev_tex_descr.normalizedCoords = 1;

        cudaCreateTextureObject(&_dev_res.depth_tex_prev[i], &depth_prev_descr, &depth_prev_tex_descr, NULL);

        struct cudaResourceDesc silhouette_descr;
        memset(&silhouette_descr, 0, sizeof(silhouette_descr));
        silhouette_descr.resType = cudaResourceTypePitch2D;
        silhouette_descr.res.pitch2D.devPtr = _dev_res.kinect_silhouettes + i * _host_res.measures.depth_res.x * _host_res.measures.depth_res.y * sizeof(float);
        silhouette_descr.res.pitch2D.width = _host_res.measures.depth_res.x;
        silhouette_descr.res.pitch2D.height = _host_res.measures.depth_res.y;
        silhouette_descr.res.pitch2D.pitchInBytes = _host_res.measures.depth_res.x * sizeof(float);
        silhouette_descr.res.pitch2D.desc = cudaCreateChannelDesc<float>();

        struct cudaTextureDesc silhouette_tex_descr;
        memset(&silhouette_tex_descr, 0, sizeof(silhouette_tex_descr));
        silhouette_tex_descr.addressMode[0] = cudaAddressModeClamp;
        silhouette_tex_descr.addressMode[1] = cudaAddressModeClamp;
        silhouette_tex_descr.filterMode = cudaFilterModeLinear;
        silhouette_tex_descr.readMode = cudaReadModeElementType;
        silhouette_tex_descr.normalizedCoords = 1;

        cudaCreateTextureObject(&_dev_res.silhouette_tex[i], &silhouette_descr, &silhouette_tex_descr, NULL);
    }
}

__host__ void unmap_kinect_textures()
{
    for(int i = 0; i < 4; i++)
    {
        cudaDestroyTextureObject(_dev_res.depth_tex[i]);
        cudaDestroyTextureObject(_dev_res.depth_tex_prev[i]);
        cudaDestroyTextureObject(_dev_res.silhouette_tex[i]);
    }
}

__host__ void map_tsdf_volumes()
{
    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.volume_tsdf_data, 0));
    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.volume_tsdf_ref_grad, 0));

    cudaArray *volume_array_tsdf_data = nullptr;
    cudaArray *volume_array_tsdf_ref_grad = nullptr;
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&volume_array_tsdf_data, _cgr.volume_tsdf_data, 0, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&volume_array_tsdf_ref_grad, _cgr.volume_tsdf_ref_grad, 0, 0));

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
    checkCudaErrors(cudaBindSurfaceToArray(&_volume_tsdf_data, volume_array_tsdf_data, &channel_desc));

    cudaChannelFormatDesc rgba32_channel_desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    checkCudaErrors(cudaBindSurfaceToArray(&_volume_tsdf_ref_grad, volume_array_tsdf_ref_grad, &rgba32_channel_desc));
}

__host__ void unmap_tsdf_volumes()
{
    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.volume_tsdf_ref_grad, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.volume_tsdf_data, 0));
}

__host__ void map_kinect_arrays()
{
    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.pbo_kinect_rgbs, 0));
    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.pbo_kinect_depths, 0));
    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.pbo_kinect_silhouettes, 0));

    size_t dummy_size = 0;
    size_t depth_size = _host_res.measures.depth_res.x * _host_res.measures.depth_res.y * 4;

    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&_dev_res.mapped_pbo_rgbs, &dummy_size, _cgr.pbo_kinect_rgbs));
    checkCudaErrors(cudaMemcpy(&_dev_res.kinect_rgbs[0], &_dev_res.mapped_pbo_rgbs[0], depth_size * sizeof(float4), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&_dev_res.mapped_pbo_depths, &dummy_size, _cgr.pbo_kinect_depths));
    checkCudaErrors(cudaMemcpy(&_dev_res.kinect_depths_prev[0], &_dev_res.kinect_depths[0], depth_size * sizeof(float2), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(&_dev_res.kinect_depths[0], &_dev_res.mapped_pbo_depths[0], depth_size * sizeof(float2), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&_dev_res.mapped_pbo_silhouettes, &dummy_size, _cgr.pbo_kinect_silhouettes));
    checkCudaErrors(cudaMemcpy(&_dev_res.kinect_silhouettes[0], &_dev_res.mapped_pbo_silhouettes[0], depth_size * sizeof(float1), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaDeviceSynchronize());

#ifdef PIPELINE_DEBUG_TEXTURE_SILHOUETTES
    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.pbo_kinect_silhouettes_debug, 0));

    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&_dev_res.mapped_pbo_silhouettes_debug, &dummy_size, _cgr.pbo_kinect_silhouettes_debug));
    checkCudaErrors(cudaDeviceSynchronize());
#endif
}

__host__ void unmap_kinect_arrays()
{
    size_t depth_size = _host_res.measures.depth_res.x * _host_res.measures.depth_res.y * 4;

#ifdef PIPELINE_DEBUG_TEXTURE_SILHOUETTES
    checkCudaErrors(cudaMemcpy(&_dev_res.mapped_pbo_silhouettes_debug[0], &_dev_res.kinect_silhouettes[0], depth_size * sizeof(float1), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.pbo_kinect_silhouettes_debug));
#endif

    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.pbo_kinect_silhouettes));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.pbo_kinect_depths));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.pbo_kinect_rgbs));
}

__host__ void free_brick_resources()
{
    if(_dev_res.bricks_inv_index != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.bricks_inv_index));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    if(_dev_res.bricks_dense_index != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.bricks_dense_index));
        checkCudaErrors(cudaDeviceSynchronize());
    }
}
__host__ void free_ed_resources()
{
    if(_dev_res.unsorted_vx_ptr != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.unsorted_vx_ptr));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    if(_dev_res.sorted_vx_ptr != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.sorted_vx_ptr));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    if(_dev_res.warped_sorted_vx_ptr != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.warped_sorted_vx_ptr));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    if(_dev_res.sorted_vx_projections != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.sorted_vx_projections));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    if(_dev_res.warped_sorted_vx_projections != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.warped_sorted_vx_projections));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    if(_dev_res.ed_graph != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.ed_graph));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    if(_dev_res.ed_graph_meta)
    {
        checkCudaErrors(cudaFree(_dev_res.ed_graph_meta));
        checkCudaErrors(cudaDeviceSynchronize());
    }
}

__host__ void free_pcg_resources()
{
    if(_dev_res.jtf != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.jtf));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    if(_dev_res.jtj_vals != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.jtj_vals));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    if(_dev_res.jtj_rows != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.jtj_rows));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    if(_dev_res.jtj_cols != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.jtj_cols));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    if(_dev_res.h != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.h));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    if(_dev_res.pcg_Ax != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.pcg_Ax));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    if(_dev_res.pcg_omega != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.pcg_omega));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    if(_dev_res.pcg_p != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.pcg_p));
        checkCudaErrors(cudaDeviceSynchronize());
    }
}
__host__ void allocate_brick_resources()
{
    checkCudaErrors(cudaMalloc(&_dev_res.bricks_inv_index, _host_res.measures.data_volume_num_bricks * sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc(&_dev_res.bricks_dense_index, _host_res.measures.data_volume_num_bricks * sizeof(unsigned int)));
}
__host__ void clean_brick_resources()
{
    checkCudaErrors(cudaMemset(_dev_res.bricks_inv_index, 0, _host_res.measures.data_volume_num_bricks * sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(_dev_res.bricks_dense_index, 0, _host_res.measures.data_volume_num_bricks * sizeof(unsigned int)));
}

__host__ void allocate_ed_resources()
{
    unsigned int MAX_ED_CELLS = _host_res.measures.data_volume_num_bricks * _host_res.measures.brick_num_ed_cells;

    checkCudaErrors(cudaMalloc(&_dev_res.unsorted_vx_ptr, MAX_REFERENCE_VERTICES * sizeof(struct_vertex)));
    checkCudaErrors(cudaMalloc(&_dev_res.sorted_vx_ptr, MAX_REFERENCE_VERTICES * sizeof(struct_vertex)));
    checkCudaErrors(cudaMalloc(&_dev_res.warped_sorted_vx_ptr, MAX_REFERENCE_VERTICES * sizeof(struct_vertex)));
    checkCudaErrors(cudaMalloc(&_dev_res.sorted_vx_projections, MAX_REFERENCE_VERTICES * sizeof(struct_projection)));
    checkCudaErrors(cudaMalloc(&_dev_res.warped_sorted_vx_projections, MAX_REFERENCE_VERTICES * sizeof(struct_projection)));

    checkCudaErrors(cudaMalloc(&_dev_res.ed_graph, MAX_ED_CELLS * sizeof(struct_ed_node)));
    checkCudaErrors(cudaMalloc(&_dev_res.ed_graph_step, MAX_ED_CELLS * sizeof(struct_ed_node)));
    checkCudaErrors(cudaMalloc(&_dev_res.ed_graph_meta, MAX_ED_CELLS * sizeof(struct_ed_meta_entry)));
}

__host__ void allocate_pcg_resources()
{
    unsigned int MAX_ED_CELLS = _host_res.measures.data_volume_num_bricks * _host_res.measures.brick_num_ed_cells;
    unsigned long int JTJ_ROWS = MAX_ED_CELLS * ED_COMPONENT_COUNT;
    unsigned long int JTJ_NNZ = JTJ_ROWS * ED_COMPONENT_COUNT;

    checkCudaErrors(cudaMalloc(&_dev_res.jtf, JTJ_ROWS * sizeof(float)));
    checkCudaErrors(cudaMalloc(&_dev_res.h, JTJ_ROWS * sizeof(float)));

    checkCudaErrors(cudaMalloc(&_dev_res.pcg_p, JTJ_ROWS * sizeof(float)));
    checkCudaErrors(cudaMalloc(&_dev_res.pcg_omega, JTJ_ROWS * sizeof(float)));
    checkCudaErrors(cudaMalloc(&_dev_res.pcg_Ax, JTJ_ROWS * sizeof(float)));

    checkCudaErrors(cudaMalloc(&_dev_res.jtj_vals, JTJ_NNZ * sizeof(float)));
    checkCudaErrors(cudaMalloc(&_dev_res.jtj_rows, JTJ_NNZ * sizeof(int)));
    checkCudaErrors(cudaMalloc(&_dev_res.jtj_rows_csr, JTJ_NNZ * sizeof(int)));
    checkCudaErrors(cudaMalloc(&_dev_res.jtj_cols, JTJ_NNZ * sizeof(int)));
}

__host__ void clean_ed_resources()
{
    unsigned int MAX_ED_CELLS = _host_res.measures.data_volume_num_bricks * _host_res.measures.brick_num_ed_cells;

    checkCudaErrors(cudaMemset(_dev_res.unsorted_vx_ptr, 0, MAX_REFERENCE_VERTICES * sizeof(struct_vertex)));
    checkCudaErrors(cudaMemset(_dev_res.sorted_vx_ptr, 0, MAX_REFERENCE_VERTICES * sizeof(struct_vertex)));
    checkCudaErrors(cudaMemset(_dev_res.warped_sorted_vx_ptr, 0, MAX_REFERENCE_VERTICES * sizeof(struct_vertex)));
    checkCudaErrors(cudaMemset(_dev_res.sorted_vx_projections, 0, MAX_REFERENCE_VERTICES * sizeof(struct_projection)));
    checkCudaErrors(cudaMemset(_dev_res.warped_sorted_vx_projections, 0, MAX_REFERENCE_VERTICES * sizeof(struct_projection)));

    checkCudaErrors(cudaMemset(_dev_res.ed_graph, 0, MAX_ED_CELLS * sizeof(struct_ed_node)));
    checkCudaErrors(cudaMemset(_dev_res.ed_graph_step, 0, MAX_ED_CELLS * sizeof(struct_ed_node)));
    checkCudaErrors(cudaMemset(_dev_res.ed_graph_meta, 0, MAX_ED_CELLS * sizeof(struct_ed_meta_entry)));
}

__host__ void clean_pcg_resources()
{
    unsigned int MAX_ED_CELLS = _host_res.measures.data_volume_num_bricks * _host_res.measures.brick_num_ed_cells;
    unsigned long int JTJ_ROWS = MAX_ED_CELLS * ED_COMPONENT_COUNT;
    unsigned long int JTJ_NNZ = JTJ_ROWS * ED_COMPONENT_COUNT;

    checkCudaErrors(cudaMemset(_dev_res.jtf, 0, JTJ_ROWS * sizeof(float)));
    checkCudaErrors(cudaMemset(_dev_res.h, 0, JTJ_ROWS * sizeof(float)));

    checkCudaErrors(cudaMemset(_dev_res.pcg_p, 0, JTJ_ROWS * sizeof(float)));
    checkCudaErrors(cudaMemset(_dev_res.pcg_omega, 0, JTJ_ROWS * sizeof(float)));
    checkCudaErrors(cudaMemset(_dev_res.pcg_Ax, 0, JTJ_ROWS * sizeof(float)));

    checkCudaErrors(cudaMemset(_dev_res.jtj_vals, 0, JTJ_NNZ * sizeof(float)));
    checkCudaErrors(cudaMemset(_dev_res.jtj_rows, 0, JTJ_NNZ * sizeof(int)));
    checkCudaErrors(cudaMemset(_dev_res.jtj_rows_csr, 0, JTJ_NNZ * sizeof(int)));
    checkCudaErrors(cudaMemset(_dev_res.jtj_cols, 0, JTJ_NNZ * sizeof(int)));
}

__host__ void allocate_correspondence_resources()
{
    checkCudaErrors(cudaMalloc(&_dev_res.sorted_correspondences, 4 * SIFT_MAX_CORRESPONDENCES * sizeof(struct_correspondence)));
    checkCudaErrors(cudaMalloc(&_dev_res.unsorted_correspondences, 4 * SIFT_MAX_CORRESPONDENCES * sizeof(struct_correspondence)));
    checkCudaErrors(cudaMalloc(&_dev_res.depth_cell_counter, 4 * _host_res.measures.num_depth_cells * sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc(&_dev_res.depth_cell_meta, 4 * _host_res.measures.num_depth_cells * sizeof(struct_depth_cell_meta)));
}

__host__ void clean_correspondence_resources()
{
    checkCudaErrors(cudaMemset(_dev_res.sorted_correspondences, 0, 4 * SIFT_MAX_CORRESPONDENCES * sizeof(struct_correspondence)));
    checkCudaErrors(cudaMemset(_dev_res.unsorted_correspondences, 0, 4 * SIFT_MAX_CORRESPONDENCES * sizeof(struct_correspondence)));
    checkCudaErrors(cudaMemset(_dev_res.depth_cell_counter, 0, 4 * _host_res.measures.num_depth_cells * sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(_dev_res.depth_cell_meta, 0, 4 * _host_res.measures.num_depth_cells * sizeof(struct_depth_cell_meta)));
}

__host__ void free_correspondence_resources()
{
    if(_dev_res.sorted_correspondences != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.sorted_correspondences));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    if(_dev_res.unsorted_correspondences != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.unsorted_correspondences));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    if(_dev_res.depth_cell_counter != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.depth_cell_counter));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    if(_dev_res.depth_cell_meta != nullptr)
    {
        checkCudaErrors(cudaFree(_dev_res.depth_cell_meta));
        checkCudaErrors(cudaDeviceSynchronize());
    }
}

extern "C" void update_configuration(Configuration &configuration) { memcpy(&_host_res.configuration, &configuration, sizeof(Configuration)); }

#endif // RECON_PC_CUDA_RESOURCES