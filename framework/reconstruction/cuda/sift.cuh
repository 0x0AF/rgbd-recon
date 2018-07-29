#include <reconstruction/cuda/resources.cuh>

#include "reconstruction/cuda/SIFT/cudaImage.cuh"
#include "reconstruction/cuda/SIFT/cudaSiftH.cuh"
#include "reconstruction/cuda/SIFT/cudautils.h"
#include "reconstruction/cuda/SIFT/matching.cuh"

SiftData *sift_front;
SiftData *sift_back;
CudaImage img;

__global__ void kernel_extract_correspondences(int valid_matches, unsigned int *counter, int layer, SiftData current, SiftData previous, struct_device_resources dev_res, struct_measures measures)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int correspondences_per_thread = (unsigned int)max(1u, valid_matches / (blockDim.x * gridDim.x));

    for(unsigned int i = 0; i < correspondences_per_thread; i++)
    {
        unsigned int index = idx * correspondences_per_thread + i;

        if(index >= valid_matches)
        {
            return;
        }

        if(current.d_data[index].score > SIFT_MINIMAL_SCORE)
        {
            // printf("\ncorrespondence error: %f\n", current.d_data[index].match_error);

            // printf("\ncorrespondence: (%f,%f)\n", current.d_data[index].match_xpos, current.d_data[index].match_ypos);

            struct_correspondence correspondence;

            int match = current.d_data[index].match;
            glm::uvec2 curr(current.d_data[index].xpos, current.d_data[index].ypos);
            glm::uvec2 prev(previous.d_data[match].xpos, previous.d_data[match].ypos);

            float2 depth_curr = sample_depths_ptr(dev_res.kinect_depths, curr, layer, measures);
            float2 depth_prev = sample_depths_ptr(dev_res.kinect_depths_prev, prev, layer, measures);

            if((int)(depth_curr.x * 1000) == 0 || (int)(depth_prev.x * 1000) == 0 || glm::abs(depth_curr.x - depth_prev.x) > SIFT_FILTER_MAX_MOTION)
            {
                continue;
            }

            glm::vec3 norm_curr = glm::vec3(((float)curr.x) / measures.depth_res.x, ((float)curr.y) / measures.depth_res.y, depth_curr.x);
            glm::vec3 norm_prev = glm::vec3(((float)prev.x) / measures.depth_res.x, ((float)prev.y) / measures.depth_res.y, depth_prev.x);

            // printf("\ndepth_curr: %f, depth_prev: %f\n", depth_curr.x, depth_prev.x);

            float4 current = sample_cv_xyz(dev_res.cv_xyz_tex[layer], norm_curr);
            float4 previous = sample_cv_xyz(dev_res.cv_xyz_tex[layer], norm_prev);

            correspondence.current = glm::clamp(bbox_transform_position(glm::vec3(current.x, current.y, current.z), measures), glm::vec3(0.f), glm::vec3(1.f));
            correspondence.previous = glm::clamp(bbox_transform_position(glm::vec3(previous.x, previous.y, previous.z), measures), glm::vec3(0.f), glm::vec3(1.f));

            //            printf("\ncorrespondence: c(%f,%f,%f) === p(%f,%f,%f)\n", correspondence.current.x, correspondence.current.y, correspondence.current.z,
            //                   correspondence.previous.x, correspondence.previous.y, correspondence.previous.z);

            if(glm::length(correspondence.current - correspondence.previous) > SIFT_FILTER_MAX_MOTION)
            {
                continue;
            }

            correspondence.current_proj = curr;
            correspondence.previous_proj = prev;
            correspondence.layer = layer;
            correspondence.cell_id = identify_depth_cell_id(correspondence.previous_proj, layer, measures);

            unsigned int alloc_position = atomicAdd(counter, 1u);
            atomicAdd(&dev_res.depth_cell_counter[correspondence.cell_id], 1u);

            memcpy(&dev_res.unsorted_correspondences[alloc_position], &correspondence, sizeof(struct_correspondence));
        }
    }
}

__global__ void kernel_retrieve_depth_cells(unsigned int *depth_cell_index, unsigned int *cp_index, struct_device_resources dev_res, struct_measures measures)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= measures.num_depth_cells)
    {
        // printf("\ned_nodes_count overshot: %u, ed_nodes_count: %u\n", idx, ed_nodes_count);
        return;
    }

    unsigned int cp_count = dev_res.depth_cell_counter[idx];

    if(cp_count == 0)
    {
        return;
    }

    unsigned int cell_position = atomicAdd(depth_cell_index, 1u);
    unsigned int cell_cp_offset = atomicAdd(cp_index, cp_count);

    struct_depth_cell_meta cp_meta;

    cp_meta.cp_offset = cell_position;
    cp_meta.cp_length = cell_cp_offset;

    memcpy(&dev_res.depth_cell_meta[cell_position], &cp_meta, sizeof(struct_depth_cell_meta));
}

/*__global__ void kernel_sort_correspondences(unsigned int counter, unsigned int offset, int layer, struct_device_resources dev_res, struct_measures measures)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int correspondences_per_thread = (unsigned int)max(1u, counter / (blockDim.x * gridDim.x));

    for(unsigned int i = 0; i < correspondences_per_thread; i++)
    {
        unsigned int index = idx * correspondences_per_thread + i;

        if(index >= counter)
        {
            return;
        }

        struct_correspondence correspondence = dev_res.unsorted_correspondences[offset + index];

        unsigned int position_in_cell = atomicSub(&dev_res.depth_cell_counter[correspondence.cell_id], 1u);
    }
}*/

extern "C" void estimate_correspondence_field()
{
    clean_correspondence_resources();

    int w = (int)_host_res.measures.depth_res.x;
    int h = (int)_host_res.measures.depth_res.y;
    int p = iAlignUp(w, 128);

    unsigned int *counter;
    cudaMallocManaged(&counter, sizeof(unsigned int));
    *counter = 0u;

    for(int i = 0; i < 4; i++)
    {
        img.Allocate(w, h, p, false, _dev_res.kinect_intens + w * h * i, NULL);

        ExtractSift(sift_front[i], img, SIFT_OCTAVES, SIFT_BLUR, SIFT_THRESHOLD, SIFT_LOWEST_SCALE, SIFT_UPSCALE);
        // printf("\nextracted: %i, layer[%i]\n", sift_front[i].numPts, i);
        MatchSiftData(sift_front[i], sift_back[i]);

        int block_size;
        int min_grid_size;
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_extract_correspondences, 0, 0);
        size_t grid_size = (sift_front[i].numPts + block_size - 1) / block_size;
        kernel_extract_correspondences<<<grid_size, block_size>>>(sift_front[i].numPts, counter, i, sift_front[i], sift_back[i], _dev_res, _host_res.measures);
        getLastCudaError("kernel_extract_correspondences failed");
        cudaDeviceSynchronize();
    }

    checkCudaErrors(cudaMemcpy(&_host_res.valid_correspondences, counter, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // printf("\nvalidated correspondences: %u\n", _host_res.valid_correspondences);

    unsigned int *cell_counter, *cp_counter;
    cudaMallocManaged(&cell_counter, sizeof(unsigned int));
    cudaMallocManaged(&cp_counter, sizeof(unsigned int));
    *cell_counter = 0u;
    *cp_counter = 0u;

    int block_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_retrieve_depth_cells, 0, 0);
    size_t grid_size = (w * h + block_size - 1) / block_size;
    kernel_retrieve_depth_cells<<<grid_size, block_size>>>(cell_counter, cp_counter, _dev_res, _host_res.measures);
    getLastCudaError("kernel_extract_correspondences failed");
    cudaDeviceSynchronize();

    /*int block_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_sort_correspondences, 0, 0);
    size_t grid_size = (*counter + block_size - 1) / block_size;
    kernel_sort_correspondences<<<grid_size, block_size>>>(*counter, offset, i, _dev_res, _host_res.measures);
    getLastCudaError("kernel_extract_correspondences failed");
    cudaDeviceSynchronize();*/

    if(counter != nullptr)
    {
        checkCudaErrors(cudaFree(counter));
    }

    if(cell_counter != nullptr)
    {
        checkCudaErrors(cudaFree(cell_counter));
    }

    if(cp_counter != nullptr)
    {
        checkCudaErrors(cudaFree(cp_counter));
    }

    SiftData *tmp = sift_front;
    sift_front = sift_back;
    sift_back = tmp;
}

__global__ void kernel_push_debug_correspondences(struct_correspondence *cp_ptr, unsigned int valid_correspondences, struct_device_resources dev_res, struct_measures measures)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int correspondences_per_thread = (unsigned int)max(1u, valid_correspondences / (blockDim.x * gridDim.x));

    for(unsigned int i = 0; i < correspondences_per_thread; i++)
    {
        unsigned int index = idx * correspondences_per_thread + i;

        if(index >= valid_correspondences)
        {
            return;
        }

        memcpy(&cp_ptr[index], &dev_res.unsorted_correspondences[index], sizeof(struct_correspondence));
    }
}

extern "C" unsigned int push_debug_correspondences()
{
    if(_host_res.valid_correspondences == 0u)
    {
        return 0u;
    }

    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.buffer_correspondences_debug));

    struct_correspondence *cp_ptr;
    size_t bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&cp_ptr, &bytes, _cgr.buffer_correspondences_debug));

    int block_size;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_push_debug_correspondences, 0, 0);
    size_t grid_size = (_host_res.valid_correspondences + block_size - 1) / block_size;
    kernel_push_debug_correspondences<<<grid_size, block_size>>>(cp_ptr, _host_res.valid_correspondences, _dev_res, _host_res.measures);
    getLastCudaError("render kernel failed");
    cudaDeviceSynchronize();

    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.buffer_correspondences_debug));

    return _host_res.valid_correspondences;
}