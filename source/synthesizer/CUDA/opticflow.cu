#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <GL/gl.h>
#include <GL/glext.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <cublas_v2.h>
#include <cuda_gl_interop.h>
#include <cusparse_v2.h>

#include <reconstruction/cuda/glm.cuh>

#include "brox_optical_flow.cuh"
#include "cuda_calls.cuh"

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

float opticflow_scaling_factor = 0.95;    // 0.95;
int opticflow_num_inner_iterations = 5;   // 5;
int opticflow_num_outer_iterations = 150; // 150;
int opticflow_num_solver_iterations = 10; // 10;

extern "C" void init_of_cuda()
{
    cudaDeviceReset();

    int devicesCount;
    cudaGetDeviceCount(&devicesCount);
    for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
    {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, deviceIndex);
        if(deviceProperties.major >= 6 && deviceProperties.minor >= 1)
        {
            cudaSetDevice(deviceIndex);
            cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
            cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
            cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
        }
    }
}

extern "C" void deinit_of_cuda() { cudaDeviceReset(); }

template <class T>
struct normalize_functor
{
    T _min;
    T _max;
    normalize_functor(T min, T max)
    {
        _min = min;
        _max = max;
    }
    __host__ __device__ T operator()(T &x) const { return (x - _min) / (_max - _min); }
};

extern "C" void evaluate_optical_flow(float *color_frame_previous, float *color_frame, float2 *optical_flow)
{
    int im_width = 512;
    int im_height = 424;
    int32_t im_dimensions[3] = {im_width, im_height, 1};

    std::vector<float> norm_im_0(im_width * im_height, 0.0);
    std::vector<float> norm_im_1(im_width * im_height, 0.0);

    memcpy(&norm_im_0[0], color_frame_previous, im_height * im_width * sizeof(float));
    memcpy(&norm_im_1[0], color_frame, im_height * im_width * sizeof(float));

    thrust::host_vector<float> prev(norm_im_0.begin(), norm_im_0.end());
    thrust::host_vector<float> curr(norm_im_1.begin(), norm_im_1.end());

    thrust::host_vector<float>::iterator max_0 = thrust::max_element(prev.begin(), prev.end());
    thrust::host_vector<float>::iterator max_1 = thrust::max_element(curr.begin(), curr.end());

    thrust::host_vector<float>::iterator min_0 = thrust::min_element(prev.begin(), prev.end());
    thrust::host_vector<float>::iterator min_1 = thrust::min_element(curr.begin(), curr.end());

    float min = std::min(*min_0, *min_1);
    float max = std::max(*max_0, *max_1);

    // printf("\nminmax: %f:%f\n", min, max);

    normalize_functor<float> nf(min, max);

    thrust::transform(prev.begin(), prev.end(), norm_im_0.begin(), nf);
    thrust::transform(curr.begin(), curr.end(), norm_im_1.begin(), nf);

    std::vector<float> computed_flow_x(im_width * im_height, 0.0);
    std::vector<float> computed_flow_y(im_width * im_height, 0.0);

    float smoothness_term = 0.197;
    float edge_term = 50.0;

    std::vector<std::array<int32_t, 3>> pyramid_sizes;
    std::vector<std::vector<float>> scale_pyramid_1;
    std::vector<std::vector<float>> scale_pyramid_2;
    std::vector<std::vector<float>> grad_x_0_pyramid;
    std::vector<std::vector<float>> grad_y_0_pyramid;

    Brox_optical_flow(norm_im_0, norm_im_1, im_dimensions, smoothness_term, edge_term, opticflow_scaling_factor, opticflow_num_inner_iterations, opticflow_num_outer_iterations,
                      opticflow_num_solver_iterations, computed_flow_x, computed_flow_y, scale_pyramid_1, scale_pyramid_2, grad_x_0_pyramid, grad_y_0_pyramid, pyramid_sizes);
    checkCudaErrors(cudaThreadSynchronize());

    for(int k = 0; k < im_width * im_height; k++)
    {
        optical_flow[k].x = computed_flow_x[k];
        optical_flow[k].y = computed_flow_y[k];

        // printf("\n(%i,%i): (%f,%f)\n", k / im_width, k % im_width, optical_flow[k].x, optical_flow[k].y);
    }
}