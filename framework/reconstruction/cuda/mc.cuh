#include <reconstruction/cuda/glm.cuh>
#include <reconstruction/cuda/resources.cuh>

#define DEBUG_BUFFERS 0

template <class T>
void dumpBuffer(T *d_buffer, int nelements, int size_element)
{
    uint bytes = nelements * size_element;
    T *h_buffer = (T *)malloc(bytes);
    checkCudaErrors(cudaMemcpy(h_buffer, d_buffer, bytes, cudaMemcpyDeviceToHost));

    for(int i = 0; i < nelements; i++)
    {
        printf("%d: %u\n", i, h_buffer[i]);
    }

    printf("\n");
    free(h_buffer);
}

void init_mc(glm::uvec3 &volume_res, struct_measures &measures, struct_native_handles &native_handles)
{
    _host_res.grid_size = make_uint3(volume_res.x, volume_res.y, volume_res.z);
    _host_res.grid_size_shift = make_uint3(0, (unsigned int)log2((float)volume_res.x), (unsigned int)(2 * log2((float)volume_res.x)));
    _host_res.grid_size_mask = make_uint3(volume_res.x - 1, volume_res.y - 1, volume_res.z - 1);

    _host_res.num_voxels = _host_res.grid_size.x * _host_res.grid_size.y * _host_res.grid_size.z;
    _host_res.voxel_size = make_float3(2.0f / _host_res.grid_size.x, 2.0f / _host_res.grid_size.y, 2.0f / _host_res.grid_size.z);

    allocateTextures(&_dev_res.edge_table, &_dev_res.tri_table, &_dev_res.num_verts_table);

    unsigned int memSize = sizeof(uint) * _host_res.num_voxels;
    checkCudaErrors(cudaMalloc((void **)&_dev_res.voxel_verts, memSize));
    checkCudaErrors(cudaMalloc((void **)&_dev_res.voxel_verts_scan, memSize));
    checkCudaErrors(cudaMalloc((void **)&_dev_res.voxel_occupied, memSize));
    checkCudaErrors(cudaMalloc((void **)&_dev_res.voxel_occupied_scan, memSize));
    checkCudaErrors(cudaMalloc((void **)&_dev_res.comp_voxel_array, memSize));
}

extern "C" unsigned long int compute_isosurface(IsoSurfaceVolume target)
{
    uchar *target_ptr = nullptr;

    switch(target)
    {
    case IsoSurfaceVolume::Data:
        target_ptr = _dev_res.out_tsdf_data;
        break;
    case IsoSurfaceVolume::Reference:
        target_ptr = _dev_res.out_tsdf_ref;
        break;
    case IsoSurfaceVolume::WarpedReference:
        target_ptr = _dev_res.out_tsdf_warped_ref;
        break;
    }

    bindVolumeTexture(target_ptr);

    int threads = 128;
    dim3 grid(_host_res.num_voxels / threads, 1, 1);

    // get around maximum grid size of 65535 in each dimension
    if(grid.x > 65535)
    {
        grid.y = grid.x / 32768;
        grid.x = 32768;
    }

    // calculate number of vertices need per voxel
    launch_classifyVoxel(grid, threads, _dev_res.voxel_verts, _dev_res.voxel_occupied, target_ptr, _host_res.grid_size, _host_res.grid_size_shift, _host_res.grid_size_mask, _host_res.num_voxels,
                         _host_res.voxel_size, 0.5f);
#if DEBUG_BUFFERS
    printf("voxelVerts:\n");
    dumpBuffer(_dev_res.voxel_verts, _host_res.num_voxels, sizeof(uint));
#endif

#if SKIP_EMPTY_VOXELS
    // scan voxel occupied array
    ThrustScanWrapper(_dev_res.voxel_occupied_scan, _dev_res.voxel_occupied, _host_res.num_voxels);

#if DEBUG_BUFFERS
    printf("voxelOccupiedScan:\n");
    dumpBuffer(_dev_res.voxel_occupied_scan, _host_res.num_voxels, sizeof(uint));
#endif

    // read back values to calculate total number of non-empty voxels
    // since we are using an exclusive scan, the total is the last value of
    // the scan result plus the last value in the input array
    {
        uint lastElement, lastScanElement;
        checkCudaErrors(cudaMemcpy((void *)&lastElement, (void *)(_dev_res.voxel_occupied + _host_res.num_voxels - 1), sizeof(uint), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy((void *)&lastScanElement, (void *)(_dev_res.voxel_occupied_scan + _host_res.num_voxels - 1), sizeof(uint), cudaMemcpyDeviceToHost));
        _host_res.active_voxels = lastElement + lastScanElement;
    }

    if(_host_res.active_voxels == 0)
    {
        // return if there are no full voxels
        _host_res.total_verts = 0;
        return 0;
    }

    // compact voxel index array
    launch_compactVoxels(grid, threads, _dev_res.comp_voxel_array, _dev_res.voxel_occupied, _dev_res.voxel_occupied_scan, _host_res.num_voxels);
    getLastCudaError("compactVoxels failed");

#endif // SKIP_EMPTY_VOXELS

    // scan voxel vertex count array
    ThrustScanWrapper(_dev_res.voxel_verts_scan, _dev_res.voxel_verts, _host_res.num_voxels);

#if DEBUG_BUFFERS
    printf("voxelVertsScan:\n");
    dumpBuffer(_dev_res.voxel_verts_scan, _host_res.num_voxels, sizeof(uint));
#endif

    // readback total number of vertices
    {
        uint lastElement, lastScanElement;
        checkCudaErrors(cudaMemcpy((void *)&lastElement, (void *)(_dev_res.voxel_verts + _host_res.num_voxels - 1), sizeof(uint), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy((void *)&lastScanElement, (void *)(_dev_res.voxel_verts_scan + _host_res.num_voxels - 1), sizeof(uint), cudaMemcpyDeviceToHost));
        _host_res.total_verts = lastElement + lastScanElement;
    }

    // generate triangles, writing to vertex buffers
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.posvbo, 0));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&_dev_res.mapped_pos, &num_bytes, _cgr.posvbo));

    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.normalvbo, 0));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&_dev_res.mapped_normal, &num_bytes, _cgr.normalvbo));

    checkCudaErrors(cudaMemset(_dev_res.pos, 0, num_bytes));
    checkCudaErrors(cudaMemset(_dev_res.normal, 0, num_bytes));

#if SKIP_EMPTY_VOXELS
    dim3 grid2((int)ceil(_host_res.active_voxels / (float)NTHREADS), 1, 1);
#else
    dim3 grid2((int)ceil(_host_res.num_voxels / (float)NTHREADS), 1, 1);
#endif

    while(grid2.x > 65535)
    {
        grid2.x /= 2;
        grid2.y *= 2;
    }

#if SAMPLE_VOLUME
    launch_generateTriangles2(grid2, NTHREADS, _dev_res.pos, _dev_res.normal, _dev_res.comp_voxel_array, _dev_res.voxel_verts_scan, target_ptr, _host_res.grid_size, _host_res.grid_size_shift,
                              _host_res.grid_size_mask, _host_res.voxel_size, 0.5f, _host_res.active_voxels, MAX_REFERENCE_VERTICES);
#else
    launch_generateTriangles(grid2, NTHREADS, _dev_res.pos, _dev_res.normal, _dev_res.comp_voxel_array, _dev_res.voxel_verts_scan, target_ptr, _host_res.grid_size, _host_res.grid_size_shift,
                             _host_res.grid_size_mask, _host_res.voxel_size, 0.5f, _host_res.active_voxels, MAX_REFERENCE_VERTICES);
#endif

    checkCudaErrors(cudaMemcpy(_dev_res.mapped_pos, _dev_res.pos, num_bytes, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(_dev_res.mapped_normal, _dev_res.normal, num_bytes, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.normalvbo, 0));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.posvbo, 0));

    return _host_res.total_verts;
}