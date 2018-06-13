#ifndef RECON_PC_CUDA_RESOURCES
#define RECON_PC_CUDA_RESOURCES

#include <cuda_gl_interop.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <reconstruction/cuda/structures.cuh>

struct struct_graphic_resources
{
  cudaGraphicsResource *buffer_reference_mesh_vertices{nullptr};
  cudaGraphicsResource *buffer_vertex_counter{nullptr};
  cudaGraphicsResource *buffer_bricks{nullptr};
  cudaGraphicsResource *buffer_occupied{nullptr};
  cudaGraphicsResource *array2d_kinect_depths{nullptr};

  cudaGraphicsResource *volume_cv_xyz_inv[4]{nullptr, nullptr, nullptr, nullptr};
  cudaGraphicsResource *volume_tsdf_data{nullptr};
};

cudaArray *_volume_array_tsdf_ref = nullptr;

struct_graphic_resources _cgr;
struct_native_handles _native_handles;

unsigned int *_bricks_inv_index = nullptr;
struct_ed_node *_ed_graph = nullptr;

cublasHandle_t cublas_handle = nullptr;
cusparseHandle_t cusparse_handle = nullptr;

unsigned int _active_bricks_count = 0u;
unsigned int _ed_nodes_count = 0u;
unsigned int _ed_nodes_component_count = 0u;

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

surface<void, cudaTextureType2DLayered> _array2d_kinect_depths;

surface<void, cudaSurfaceType3D> _volume_cv_xyz_inv_0;
surface<void, cudaSurfaceType3D> _volume_cv_xyz_inv_1;
surface<void, cudaSurfaceType3D> _volume_cv_xyz_inv_2;
surface<void, cudaSurfaceType3D> _volume_cv_xyz_inv_3;

#endif // RECON_PC_CUDA_RESOURCES