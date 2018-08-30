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

__global__ void kernel_filter_rows(struct_device_resources dev_res, int layer, struct_measures measures)
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

    const int offset = ix + iy * dev_res.pitch_kinect_silhouettes / sizeof(float);

    float initial_value = sample_pitched_ptr(dev_res.kinect_silhouettes[layer], dev_res.pitch_kinect_silhouettes, ix, iy);

    if(((int)(initial_value * 10)) == 10)
    {
        return;
    }

    float sum = convolutionRow<2 * KERNEL_RADIUS>(dev_res.kinect_silhouettes[layer], offset);

    __syncthreads();

    write_pitched_ptr(sum, dev_res.kinect_silhouettes[layer], dev_res.pitch_kinect_silhouettes, ix, iy);
}

__global__ void kernel_filter_cols(struct_device_resources dev_res, int layer, struct_measures measures)
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

    const int offset = ix + iy * dev_res.pitch_kinect_silhouettes / sizeof(float);

    float initial_value = sample_pitched_ptr(dev_res.kinect_silhouettes[layer], dev_res.pitch_kinect_silhouettes, ix, iy);

    if(((int)(initial_value * 10)) == 10)
    {
        return;
    }

    float sum = convolutionColumn<2 * KERNEL_RADIUS>(dev_res.kinect_silhouettes[layer], offset, dev_res.pitch_kinect_silhouettes / sizeof(float));

    __syncthreads();

    write_pitched_ptr(sum, dev_res.kinect_silhouettes[layer], dev_res.pitch_kinect_silhouettes, ix, iy);
}

__host__ void preprocess_hull()
{
    cudaMemcpyToSymbol(c_Kernel, _host_res.kernel_gauss, KERNEL_LENGTH * sizeof(float));

    dim3 threads(8, 8);
    dim3 blocks(iDivUp(_host_res.measures.depth_res.x, threads.x), iDivUp(_host_res.measures.depth_res.y, threads.y));

    for(int iteration = 0; iteration < _host_res.configuration.textures_silhouettes_iterations; iteration++)
    {
        for(int layer = 0; layer < 4; layer++)
        {
            kernel_filter_rows<<<blocks, threads>>>(_dev_res, layer, _host_res.measures);
            getLastCudaError("kernel_filter_rows execution failed\n");

            kernel_filter_cols<<<blocks, threads>>>(_dev_res, layer, _host_res.measures);
            getLastCudaError("kernel_filter_cols execution failed\n");
        }
    }
}

__global__ void kernel_flatten_rgbs(struct_device_resources dev_res, int layer, struct_measures measures)
{
    const int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

    if(ix >= measures.depth_res.x || iy >= measures.depth_res.y)
    {
        return;
    }

    if(iy < 0 || ix < 0)
    {
        return;
    }

    const int offset = ix + iy * measures.depth_res.x + layer * measures.depth_res.x * measures.depth_res.y;

    // printf("\nx:%u, y:%u\n", measures.depth_res.x,measures.depth_res.y);

    if(dev_res.mapped_pbo_silhouettes[offset].x == 1)
    {
        float4 color = dev_res.mapped_pbo_rgbs[offset];

        // printf ("\ncolor: (%f,%f,%f,%f)\n",color.x,color.y,color.z, color.w);

        /// Following statement maps to two-instruction equivalent of 5.0f * color.x + color.y + 2.5f * color.z
        float intensity = __fmaf_rn(2.5f, color.z, __fmaf_rn(5.f, color.x, color.y));

        write_pitched_ptr(intensity, dev_res.kinect_intens[layer], dev_res.pitch_kinect_intens, ix, iy);
    }
    else
    {
        float cloud = dev_res.cloud_noise[ix + iy * measures.depth_res.x];
        write_pitched_ptr(cloud, dev_res.kinect_intens[layer], dev_res.pitch_kinect_intens, ix, iy);
    }
}

__host__ void preprocess_intensity()
{
    dim3 threads(8, 8);
    dim3 blocks(iDivUp(_host_res.measures.depth_res.x, threads.x), iDivUp(_host_res.measures.depth_res.y, threads.y));

    for(int layer = 0; layer < 4; layer++)
    {
        kernel_flatten_rgbs<<<blocks, threads>>>(_dev_res, layer, _host_res.measures);
        getLastCudaError("kernel_filter_rows execution failed\n");
    }
}

__global__ void kernel_copy_depth(struct_device_resources dev_res, int layer, struct_measures measures)
{
    const int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

    if(ix >= measures.depth_res.x || iy >= measures.depth_res.y)
    {
        return;
    }

    if(iy < 0 || ix < 0)
    {
        return;
    }

    const int offset = ix + iy * measures.depth_res.x + layer * measures.depth_res.x * measures.depth_res.y;

    float depth_prev = sample_pitched_ptr(dev_res.kinect_depths[layer], dev_res.pitch_kinect_depths, ix, iy);
    write_pitched_ptr(depth_prev, dev_res.kinect_depths_prev[layer], dev_res.pitch_kinect_depths_prev, ix, iy);

    float depth = dev_res.mapped_pbo_depths[offset].x;

    if(isnan(depth))
    {
        depth = 0.f;
    }

    if(isinf(depth))
    {
        depth = 0.f;
    }

    write_pitched_ptr(depth, dev_res.kinect_depths[layer], dev_res.pitch_kinect_depths, ix, iy);
}

__host__ void preprocess_depth()
{
    dim3 threads(8, 8);
    dim3 blocks(iDivUp(_host_res.measures.depth_res.x, threads.x), iDivUp(_host_res.measures.depth_res.y, threads.y));

    for(int layer = 0; layer < 4; layer++)
    {
        kernel_copy_depth<<<blocks, threads>>>(_dev_res, layer, _host_res.measures);
        getLastCudaError("kernel_filter_rows execution failed\n");
    }
}

extern "C" double preprocess_textures()
{
    TimerGPU timer(0);

    map_kinect_arrays();

    for(unsigned int i = 0; i < 4; i++)
    {
        checkCudaErrors(cudaMemcpy2D(&_dev_res.kinect_intens_prev[i][0], _dev_res.pitch_kinect_intens, &_dev_res.kinect_intens[i][0], _dev_res.pitch_kinect_intens,
                                     _host_res.measures.depth_res.x * sizeof(float), _host_res.measures.depth_res.y, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    preprocess_intensity();
    preprocess_depth();
    preprocess_hull();

    unmap_kinect_arrays();

    checkCudaErrors(cudaThreadSynchronize());
    return timer.read();
}