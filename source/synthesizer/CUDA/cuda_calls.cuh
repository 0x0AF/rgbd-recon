#include <cstdio>
#include <iostream>

__constant__ __device__ float sobel_kernel_x[9] = {-0.125, 0.0, 0.125, -0.250, 0.0, 0.250, -0.125, 0.0, 0.125};

__constant__ __device__ float sobel_kernel_y[9] = {-0.125, -0.250, -0.125, 0.0, 0.0, 0.0, 0.125, 0.250, 0.125};

__constant__ __device__ float spatial_blur_kernel[5] = {0.036, 0.249, 0.431, 0.249, 0.036};

__constant__ __device__ int spatial_kernel_offset = 2;

__constant__ __device__ float spatial_differ_kernel[5] = {-0.108, -0.283, 0.0, 0.283, 0.108};

__constant__ __device__ float temporal_blur_kernel[2] = {1.0, 1.0};

__constant__ __device__ float temporal_differ_kernel[2] = {-0.5, 0.5};

__global__ void compute_spatial_gradient_x(float *image_content) {}

__global__ void iterative_flow_field_solver_kernel(float *d_I_x, float *d_I_y, float *d_I_t, float *in_flow_x, float *in_flow_y, float *out_flow_x, float *out_flow_y, int image_width,
                                                   int image_height)
{
    int global_idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    int global_idx_y = threadIdx.y + blockIdx.y * blockDim.y;

    int grid_offset_x = gridDim.x;
    int grid_offset_y = gridDim.y;

    float lambda = 1.0;

    for(int y_idx = global_idx_y; y_idx < image_height; y_idx += grid_offset_y)
    {
        for(int x_idx = global_idx_x; x_idx < image_width; x_idx += grid_offset_x)
        {
            float avg_velocity_x = 0.0f;
            float avg_velocity_y = 0.0f;

            long long one_dimensional_center_idx = x_idx + y_idx * image_width;

            for(int kernel_offset_y = -1; kernel_offset_y < 2; ++kernel_offset_y)
            {
                for(int kernel_offset_x = -1; kernel_offset_x < 2; ++kernel_offset_x)
                {
                    if(abs(kernel_offset_x) + abs(kernel_offset_y) == 1)
                    {
                        int clamped_x_pos = max(kernel_offset_x - x_idx, min(image_width - 1, kernel_offset_x + x_idx));
                        int clamped_y_pos = max(kernel_offset_y - y_idx, min(image_height - 1, kernel_offset_y + y_idx));

                        long long one_dimensional_blur_idx = clamped_x_pos + clamped_y_pos * image_width;

                        avg_velocity_x += in_flow_x[one_dimensional_blur_idx];
                        avg_velocity_y += in_flow_y[one_dimensional_blur_idx];
                    }
                }
            }

            avg_velocity_x /= 4.0f;
            avg_velocity_y /= 4.0f;

            float I_x = d_I_x[one_dimensional_center_idx];
            float I_y = d_I_y[one_dimensional_center_idx];
            float I_t = d_I_t[one_dimensional_center_idx];

            float alpha = (I_x * avg_velocity_x + I_y * avg_velocity_y + I_t) / (lambda * lambda + I_x * I_x + I_y * I_y);

            float velo_x_out = avg_velocity_x - (I_x * alpha);
            float velo_y_out = avg_velocity_y - (I_y * alpha);

            out_flow_x[one_dimensional_center_idx] = velo_x_out;
            out_flow_y[one_dimensional_center_idx] = velo_y_out;
        }
    }
}

__global__ void differentiate_spatially_2d_kernel(float *d_in_image_content, float *d_out_image_content, int image_width, int image_height, bool along_x)
{
    int global_idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    int global_idx_y = threadIdx.y + blockIdx.y * blockDim.y;

    int grid_offset_x = gridDim.x;
    int grid_offset_y = gridDim.y;

    float *diff_kernel;

    if(along_x)
    {
        diff_kernel = sobel_kernel_x;
    }
    else
    {
        diff_kernel = sobel_kernel_y;
    }

    for(int y_idx = global_idx_y; y_idx < image_height; y_idx += grid_offset_y)
    {
        for(int x_idx = global_idx_x; x_idx < image_width; x_idx += grid_offset_x)
        {
            long long one_dimensional_center_idx = x_idx + y_idx * image_width;

            float diffed_value = 0.0;

            for(int kernel_offset_y = -1; kernel_offset_y < 2; ++kernel_offset_y)
            {
                for(int kernel_offset_x = -1; kernel_offset_x < 2; ++kernel_offset_x)
                {
                    int kernel_width = 3;

                    int kernel_idx = (kernel_offset_y + 1) * kernel_width + (kernel_offset_x + 1);

                    long long one_dimensional_diff_idx = 0;

                    int clamped_x_pos = max(kernel_offset_x - x_idx, min(image_width - 1, kernel_offset_x + x_idx));

                    int clamped_y_pos = max(kernel_offset_y - y_idx, min(image_height - 1, kernel_offset_y + y_idx));

                    one_dimensional_diff_idx = clamped_x_pos + clamped_y_pos * image_width;

                    float looked_up_weight = 0.0;

                    if(along_x)
                    {
                        looked_up_weight = sobel_kernel_x[kernel_idx];
                    }
                    else
                    {
                        looked_up_weight = sobel_kernel_y[kernel_idx];
                    }

                    diffed_value += looked_up_weight * d_in_image_content[one_dimensional_diff_idx];
                }
            }

            d_out_image_content[one_dimensional_center_idx] = diffed_value;
        }
    }
}

__global__ void convolve_spatially_kernel(float *d_in_image_content, float *d_out_image_content, int image_width, int image_height, bool along_x, bool is_blur)
{
    int global_idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    int global_idx_y = threadIdx.y + blockIdx.y * blockDim.y;

    int grid_offset_x = gridDim.x;
    int grid_offset_y = gridDim.y;

    for(int y_idx = global_idx_y; y_idx < image_height; y_idx += grid_offset_y)
    {
        for(int x_idx = global_idx_x; x_idx < image_width; x_idx += grid_offset_x)
        {
            long long one_dimensional_center_idx = x_idx + y_idx * image_width;

            float blurred_value = 0.0;
            for(int kernel_offset = -2; kernel_offset < 3; ++kernel_offset)
            {
                long long one_dimensional_blur_idx = 0;

                if(along_x)
                {
                    int clamped_x_pos = max(0, min(image_width - 1, kernel_offset + x_idx));

                    one_dimensional_blur_idx = clamped_x_pos + y_idx * image_width;
                }
                else
                {
                    int clamped_y_pos = max(0, min(image_height - 1, kernel_offset + y_idx));

                    one_dimensional_blur_idx = x_idx + clamped_y_pos * image_width;
                }

                float looked_up_weight = 0.0;

                if(is_blur)
                {
                    looked_up_weight = spatial_blur_kernel[kernel_offset + spatial_kernel_offset];
                }
                else
                {
                    looked_up_weight = spatial_differ_kernel[kernel_offset + spatial_kernel_offset];
                }

                blurred_value += looked_up_weight * d_in_image_content[one_dimensional_blur_idx];
            }

            d_out_image_content[one_dimensional_center_idx] = blurred_value;
        }
    }
}

__global__ void convolve_temporally_kernel(float *d_in_image_t0_content, float *d_in_image_t1_content, float *d_out_image_content, int image_width, int image_height, bool is_blur)
{
    int global_idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    int global_idx_y = threadIdx.y + blockIdx.y * blockDim.y;

    int grid_offset_x = gridDim.x;
    int grid_offset_y = gridDim.y;

    float convolution_kernel[2];

    if(is_blur)
    {
        convolution_kernel[0] = temporal_blur_kernel[0];
        convolution_kernel[1] = temporal_blur_kernel[1];
    }
    else
    {
        convolution_kernel[0] = temporal_differ_kernel[0];
        convolution_kernel[1] = temporal_differ_kernel[1];
    }

    for(int y_idx = global_idx_y; y_idx < image_height; y_idx += grid_offset_y)
    {
        for(int x_idx = global_idx_x; x_idx < image_width; x_idx += grid_offset_x)
        {
            long long one_dimensional_center_idx = x_idx + y_idx * image_width;

            float convolved_intensity = 0.0;

            long long one_dimensional_blur_idx = 0;

            convolved_intensity = convolution_kernel[0] * d_in_image_t0_content[one_dimensional_center_idx] + convolution_kernel[1] * d_in_image_t1_content[one_dimensional_center_idx];

            d_out_image_content[one_dimensional_center_idx] = convolved_intensity;
        }
    }
}

void launch_spatial_diff_2d_kernel(std::vector<float> const &in_image, std::vector<float> &out_spatially_convolved_image, int32_t *image_dimensions, bool convolve_along_x)
{
    float *device_in_image;
    float *device_out_image;

    int64_t num_image_elements = in_image.size();

    int64_t bytes_to_allocate = num_image_elements * sizeof(float);
    cudaMalloc((void **)&device_in_image, bytes_to_allocate);
    cudaMalloc((void **)&device_out_image, bytes_to_allocate);

    cudaMemcpy(device_in_image, &in_image[0], bytes_to_allocate, cudaMemcpyHostToDevice);

    dim3 block_dims(16, 16, 1);

    dim3 grid_dims(1024, 1024, 1);

    if(convolve_along_x)
    {
        differentiate_spatially_2d_kernel<<<grid_dims, block_dims>>>(device_in_image, device_out_image, image_dimensions[0], image_dimensions[1], convolve_along_x);
    }
    else
    {
        differentiate_spatially_2d_kernel<<<grid_dims, block_dims>>>(device_in_image, device_out_image, image_dimensions[0], image_dimensions[1], convolve_along_x);
    }
    cudaMemcpy(&out_spatially_convolved_image[0], device_out_image, bytes_to_allocate, cudaMemcpyDeviceToHost);

    cudaFree(device_in_image);
    cudaFree(device_out_image);
}

__host__ void launch_spatial_kernel(std::vector<float> const &in_image, std::vector<float> &out_spatially_convolved_image, int32_t *image_dimensions, bool convolve_along_x, bool is_blur)
{
    float *device_in_image;
    float *device_out_image;

    int64_t num_image_elements = in_image.size();

    int64_t bytes_to_allocate = num_image_elements * sizeof(float);
    cudaMalloc((void **)&device_in_image, bytes_to_allocate);
    cudaMalloc((void **)&device_out_image, bytes_to_allocate);

    cudaMemcpy(device_in_image, &in_image[0], bytes_to_allocate, cudaMemcpyHostToDevice);

    dim3 block_dims(16, 16, 1);

    dim3 grid_dims(1024, 1024, 1);

    if(convolve_along_x)
    {
        convolve_spatially_kernel<<<grid_dims, block_dims>>>(device_in_image, device_out_image, image_dimensions[0], image_dimensions[1], convolve_along_x, is_blur);
    }
    else
    {
        convolve_spatially_kernel<<<grid_dims, block_dims>>>(device_in_image, device_out_image, image_dimensions[0], image_dimensions[1], !convolve_along_x, is_blur);
    }
    cudaMemcpy(&out_spatially_convolved_image[0], device_out_image, bytes_to_allocate, cudaMemcpyDeviceToHost);

    cudaFree(device_in_image);
    cudaFree(device_out_image);
}

__host__ void launch_temporal_kernel(std::vector<float> const &in_image_t0, std::vector<float> const &in_image_t1, std::vector<float> &out_temporally_convolved_image, int32_t *image_dimensions,
                                     bool is_blur)
{
    float *device_in_image_t0;
    float *device_in_image_t1;
    float *device_out_image;

    int64_t num_image_elements = in_image_t0.size();

    int64_t bytes_to_allocate = num_image_elements * sizeof(float);
    cudaMalloc((void **)&device_in_image_t0, bytes_to_allocate);
    cudaMalloc((void **)&device_in_image_t1, bytes_to_allocate);
    cudaMalloc((void **)&device_out_image, bytes_to_allocate);

    cudaMemcpy(device_in_image_t0, &in_image_t0[0], bytes_to_allocate, cudaMemcpyHostToDevice);
    cudaMemcpy(device_in_image_t1, &in_image_t1[0], bytes_to_allocate, cudaMemcpyHostToDevice);

    dim3 block_dims(16, 16, 1);

    dim3 grid_dims(1024, 1024, 1);

    convolve_temporally_kernel<<<grid_dims, block_dims>>>(device_in_image_t0, device_in_image_t1, device_out_image, image_dimensions[0], image_dimensions[1], is_blur);

    cudaMemcpy(&out_temporally_convolved_image[0], device_out_image, bytes_to_allocate, cudaMemcpyDeviceToHost);

    cudaFree(device_in_image_t0);
    cudaFree(device_in_image_t1);
    cudaFree(device_out_image);
}

__host__ void launch_flow_field_solver(std::vector<float> const &in_x_grad, std::vector<float> const &in_y_grad, std::vector<float> const &in_t_grad, std::vector<float> &out_flow_x,
                                       std::vector<float> &out_flow_y, int32_t *image_dimensions, int num_iterations)
{
    float *device_in_grad_x;
    float *device_in_grad_y;
    float *device_in_grad_t;

    float *device_temp_flow_x;
    float *device_temp_flow_y;

    float *device_out_flow_x;
    float *device_out_flow_y;

    int64_t num_image_elements = in_x_grad.size();

    int64_t bytes_to_allocate = num_image_elements * sizeof(float);

    std::vector<float> initial_displacement_x_guess(num_image_elements, 100.0f);
    std::vector<float> initial_displacement_y_guess(num_image_elements, 0.0f);
    cudaMalloc((void **)&device_in_grad_x, bytes_to_allocate);
    cudaMalloc((void **)&device_in_grad_y, bytes_to_allocate);
    cudaMalloc((void **)&device_in_grad_t, bytes_to_allocate);

    cudaMemcpy(device_in_grad_x, &in_x_grad[0], bytes_to_allocate, cudaMemcpyHostToDevice);
    cudaMemcpy(device_in_grad_y, &in_y_grad[0], bytes_to_allocate, cudaMemcpyHostToDevice);
    cudaMemcpy(device_in_grad_t, &in_t_grad[0], bytes_to_allocate, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&device_temp_flow_x, bytes_to_allocate);
    cudaMalloc((void **)&device_temp_flow_y, bytes_to_allocate);

    cudaMemcpy(device_temp_flow_x, &initial_displacement_x_guess[0], bytes_to_allocate, cudaMemcpyHostToDevice);
    cudaMemcpy(device_temp_flow_y, &initial_displacement_y_guess[0], bytes_to_allocate, cudaMemcpyHostToDevice);
    // cudaMemset(device_temp_flow_y, 0, bytes_to_allocate);

    // does not need to be allocated, because it will be overridden entirely
    cudaMalloc((void **)&device_out_flow_x, bytes_to_allocate);
    cudaMalloc((void **)&device_out_flow_y, bytes_to_allocate);

    cudaMemset(device_out_flow_x, 0, bytes_to_allocate);
    cudaMemset(device_out_flow_y, 0, bytes_to_allocate);

    // cudaMemset(device_out_flow_x, 1, bytes_to_allocate);
    // cudaMemset(device_out_flow_y, 1, bytes_to_allocate);

    /////
    dim3 block_dims(16, 16, 1);

    dim3 grid_dims(1024, 1024, 1);

    for(int iter_idx = 0; iter_idx < num_iterations; ++iter_idx)
    {
        iterative_flow_field_solver_kernel<<<grid_dims, block_dims>>>(device_in_grad_x, device_in_grad_y, device_in_grad_t, device_temp_flow_x, device_temp_flow_y, device_out_flow_x,
                                                                      device_out_flow_y, image_dimensions[0], image_dimensions[1]);

        if(iter_idx != num_iterations - 1)
        {
            cudaMemcpy(device_temp_flow_x, device_out_flow_x, bytes_to_allocate, cudaMemcpyDeviceToDevice);
            cudaMemcpy(device_temp_flow_y, device_out_flow_y, bytes_to_allocate, cudaMemcpyDeviceToDevice);
        }
    }

    cudaMemcpy(&out_flow_x[0], device_out_flow_x, bytes_to_allocate, cudaMemcpyDeviceToHost);
    cudaMemcpy(&out_flow_y[0], device_out_flow_y, bytes_to_allocate, cudaMemcpyDeviceToHost);

    cudaFree(device_in_grad_x);
    cudaFree(device_in_grad_y);
    cudaFree(device_in_grad_t);

    cudaFree(device_temp_flow_x);
    cudaFree(device_temp_flow_y);
    cudaFree(device_out_flow_x);
    cudaFree(device_out_flow_y);
}