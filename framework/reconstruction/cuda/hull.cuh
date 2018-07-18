#include <reconstruction/cuda/resources.cuh>

#define KERNEL_RADIUS 16
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

// Maps to a single instruction on G8x / G9x / G10x
#define IMAD(a, b, c) (__mul24((a), (b)) + (c))

// Round a / b to nearest higher integer value
inline int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

// Align a to nearest higher multiple of b
inline int iAlignUp(int a, int b) { return (a % b != 0) ? (a - a % b + b) : a; }

__constant__ float c_Kernel[KERNEL_LENGTH];

template <int i>
__device__ float convolutionRow(float *silhouettes_ptr, int offset)
{
    return silhouettes_ptr[offset + KERNEL_RADIUS - i] * c_Kernel[i] + convolutionRow<i - 1>(silhouettes_ptr, offset);
}

template <>
__device__ float convolutionRow<-1>(float *silhouettes_ptr, int offset)
{
    return 0;
}

template <int i>
__device__ float convolutionColumn(float *silhouettes_ptr, int offset, int stride)
{
    return silhouettes_ptr[offset + (KERNEL_RADIUS - i) * stride] * c_Kernel[i] + convolutionColumn<i - 1>(silhouettes_ptr, offset, stride);
}

template <>
__device__ float convolutionColumn<-1>(float *silhouettes_ptr, int offset, int stride)
{
    return 0;
}

__global__ void kernel_filter_rows(float *silhouettes_ptr, int layer, struct_measures measures)
{
    const int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

    if(ix >= (measures.depth_res.x - KERNEL_RADIUS) || iy >= measures.depth_res.y)
    {
        return;
    }

    if(ix < KERNEL_RADIUS)
    {
        return;
    }

    const int offset = ix + iy * measures.depth_res.x + layer * measures.depth_res.x * measures.depth_res.y;

    float initial_value = silhouettes_ptr[offset];

    if(((int)(initial_value * 10)) == 10)
    {
        return;
    }

    float sum = convolutionRow<2 * KERNEL_RADIUS>(silhouettes_ptr, offset);

    __syncthreads();

    silhouettes_ptr[offset] = sum;
}

__global__ void kernel_filter_cols(float *silhouettes_ptr, int layer, struct_measures measures)
{
    const int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

    if(ix >= measures.depth_res.x || iy >= (measures.depth_res.y - KERNEL_RADIUS))
    {
        return;
    }

    if(iy < KERNEL_RADIUS)
    {
        return;
    }

    const int offset = ix + iy * measures.depth_res.x + layer * measures.depth_res.x * measures.depth_res.y;

    float initial_value = silhouettes_ptr[offset];

    if(((int)(initial_value * 10)) == 10)
    {
        return;
    }

    float sum = convolutionColumn<2 * KERNEL_RADIUS>(silhouettes_ptr, offset, measures.depth_res.x);

    __syncthreads();

    silhouettes_ptr[offset] = sum;
}

extern "C" void generate_smooth_hull()
{
    const int iterations = 20;

    map_kinect_arrays();

    cudaMemcpyToSymbol(c_Kernel, _host_res.kernel_gauss, KERNEL_LENGTH * sizeof(float));

    dim3 threads(8, 8);
    dim3 blocks(iDivUp(_host_res.measures.depth_res.x, threads.x), iDivUp(_host_res.measures.depth_res.y, threads.y));

    for(int iteration = 0; iteration < iterations; iteration++)
    {
        for(int layer = 0; layer < 4; layer++)
        {
            kernel_filter_rows<<<blocks, threads>>>(_dev_res.kinect_silhouettes, layer, _host_res.measures);
            getLastCudaError("kernel_filter_rows execution failed\n");

            kernel_filter_cols<<<blocks, threads>>>(_dev_res.kinect_silhouettes, layer, _host_res.measures);
            getLastCudaError("kernel_filter_cols execution failed\n");
        }
    }

    unmap_kinect_arrays();
}