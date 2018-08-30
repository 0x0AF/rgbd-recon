#include <array>
#include <cstdio>
#include <iostream>
#include <limits>
#include <thrust/device_vector.h>

#define GAUSSIAN_WINDOW_SIZE 5
#define PI 3.1415926535

__device__ const float eps2 = 1e-6f;

__global__ void calculate_1d_kernel(float *convolution_kernel, int num_elements, float sigma)
{
    int global_idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    int grid_offset_x = blockDim.x * gridDim.x;

    for(int x_idx = global_idx_x; x_idx < num_elements; x_idx += grid_offset_x)
    {
        convolution_kernel[x_idx] = expf(-(x_idx * x_idx / (2 * sigma * sigma))) * (1.0 / (sqrtf(2.0 * PI) * sigma));
    }
}

__global__ void calc_image_gradient_1d(const float *d_in_image_content, float *d_out_image_content, int image_width, int image_height, int image_depth, float *convolution_kernel,
                                       int num_one_sided_elements, bool along_x)
{
    int threads_per_grid_dim_x = blockDim.x * gridDim.x;
    int threads_per_grid_dim_y = blockDim.y * gridDim.y;

    int global_idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    int global_idx_y = threadIdx.y + blockIdx.y * blockDim.y;

    int num_negative_kernel_elements = (num_one_sided_elements - 1);

    for(int y_idx = global_idx_y; y_idx < image_height; y_idx += threads_per_grid_dim_y)
    {
        for(int x_idx = global_idx_x; x_idx < image_width; x_idx += threads_per_grid_dim_x)
        {
            float accumulated_weight = 0.0f;
            float convolved_value = 0.0f;
            long long one_dimensional_center_idx = x_idx + y_idx * image_width;

            for(int kernel_offset = -(num_negative_kernel_elements); kernel_offset < num_one_sided_elements; ++kernel_offset)
            {
                long long one_dimensional_conv_idx = 0;

                if(along_x)
                {
                    int clamped_x_pos = max(0, min(image_width - 1, x_idx + kernel_offset));

                    one_dimensional_conv_idx = clamped_x_pos + y_idx * image_width;
                }
                else
                {
                    int clamped_y_pos = max(0, min(image_height - 1, y_idx + kernel_offset));

                    one_dimensional_conv_idx = x_idx + clamped_y_pos * image_width;
                }

                float looked_up_weight = 0.0;

                looked_up_weight = convolution_kernel[abs(kernel_offset)];

                if(kernel_offset < 0)
                {
                    looked_up_weight *= -1.0;
                }

                convolved_value += looked_up_weight * d_in_image_content[one_dimensional_conv_idx];
            }

            d_out_image_content[one_dimensional_center_idx] = convolved_value / 12.0;
        }
    }
}

__global__ void convolve_image(float *d_in_image_content, float *d_out_image_content, int image_width, int image_height, int image_depth, float *convolution_kernel, int num_one_sided_elements,
                               bool along_x, bool invert_negative_elements = false)
{
    int threads_per_grid_dim_x = blockDim.x * gridDim.x;
    int threads_per_grid_dim_y = blockDim.y * gridDim.y;

    int global_idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    int global_idx_y = threadIdx.y + blockIdx.y * blockDim.y;

    int num_negative_kernel_elements = (num_one_sided_elements - 1);

    for(int y_idx = global_idx_y; y_idx < image_height; y_idx += threads_per_grid_dim_y)
    {
        for(int x_idx = global_idx_x; x_idx < image_width; x_idx += threads_per_grid_dim_x)
        {
            float accumulated_weight = 0.0f;

            long long one_dimensional_center_idx = x_idx + y_idx * image_width;

            float convolved_value = 0.0;
            for(int kernel_offset = -(num_negative_kernel_elements); kernel_offset < num_one_sided_elements; ++kernel_offset)
            {
                long long one_dimensional_conv_idx = 0;

                if(along_x)
                {
                    int clamped_x_pos = max(0, min(image_width - 1, x_idx + kernel_offset));

                    if(invert_negative_elements)
                    {
                        int x_access = x_idx + kernel_offset;

                        if(x_access < 0)
                        {
                            clamped_x_pos = std::abs(x_access);
                        }

                        if(x_access > image_width - 1)
                        {
                            int diff = std::abs(x_access - (image_width - 1));
                            clamped_x_pos = (image_width - 1) - diff;
                        }
                    }

                    one_dimensional_conv_idx = clamped_x_pos + y_idx * image_width;
                }
                else
                {
                    int clamped_y_pos = max(0, min(image_height - 1, y_idx + kernel_offset));

                    if(invert_negative_elements)
                    {
                        int y_access = y_idx + kernel_offset;

                        if(y_access < 0)
                        {
                            clamped_y_pos = std::abs(y_access);
                        }

                        if(y_access > image_height - 1)
                        {
                            int diff = std::abs(y_access - (image_height - 1));
                            clamped_y_pos = (image_height - 1) - diff;
                        }
                    }

                    one_dimensional_conv_idx = x_idx + clamped_y_pos * image_width;
                }

                float looked_up_weight = 0.0;

                looked_up_weight = convolution_kernel[abs(kernel_offset)];

                if(kernel_offset < 0 && invert_negative_elements)
                    looked_up_weight = -looked_up_weight;

                accumulated_weight += looked_up_weight;
                convolved_value += looked_up_weight * d_in_image_content[one_dimensional_conv_idx];
            }

            if(!invert_negative_elements)
            {
                d_out_image_content[one_dimensional_center_idx] = convolved_value / accumulated_weight;
            }
            else
            {
                d_out_image_content[one_dimensional_center_idx] = 120000000.0 * convolved_value;
            }
        }
    }
}

__host__ void launch_spatial_gaussian_kernel(std::vector<float> &in_out_image, float sigma, int32_t *image_dimensions)
{
    float *device_in_image;
    float *device_out_image;

    int64_t num_image_elements = in_out_image.size();

    int64_t bytes_to_allocate = num_image_elements * sizeof(float);

    cudaError_t err = cudaMalloc((void **)&device_in_image, bytes_to_allocate);

    if(err != cudaSuccess)
    {
        printf("@ Pos 0: The error is %s\n", cudaGetErrorString(err));
    }

    err = cudaMalloc((void **)&device_out_image, bytes_to_allocate);

    if(err != cudaSuccess)
    {
        printf("@ Pos 0: The error is %s\n", cudaGetErrorString(err));
    }

    err = cudaMemcpy(device_in_image, &in_out_image[0], bytes_to_allocate, cudaMemcpyHostToDevice);

    if(err != cudaSuccess)
    {
        printf("@ Pos 0: The error is %s\n", cudaGetErrorString(err));
    }

    err = cudaMemset(device_out_image, 0, bytes_to_allocate);

    if(err != cudaSuccess)
    {
        printf("@ Pos 0: The error is %s\n", cudaGetErrorString(err));
    }

    float *convolution_kernel;

    int one_sided_conv_kernel_size = (GAUSSIAN_WINDOW_SIZE * sigma) + 1;

    err = cudaMalloc((void **)&convolution_kernel, sizeof(float) * one_sided_conv_kernel_size);
    if(err != cudaSuccess)
    {
        printf("convo kernel  %s\n", cudaGetErrorString(err));
    }

    int num_threads_to_launch = std::min(one_sided_conv_kernel_size, 1024);
    calculate_1d_kernel<<<1, num_threads_to_launch>>>(convolution_kernel, one_sided_conv_kernel_size, sigma);

    dim3 block_dims(16, 16, 1);

    dim3 grid_dims(16, 16, 1);

    convolve_image<<<grid_dims, block_dims>>>(device_in_image, device_out_image, image_dimensions[0], image_dimensions[1], image_dimensions[2], convolution_kernel, one_sided_conv_kernel_size, true);

    cudaDeviceSynchronize();

    convolve_image<<<grid_dims, block_dims>>>(device_out_image, device_in_image, image_dimensions[0], image_dimensions[1], image_dimensions[2], convolution_kernel, one_sided_conv_kernel_size, false);

    // Z axis needs to be blurred later as well

    cudaMemcpy(&in_out_image[0], device_in_image, bytes_to_allocate, cudaMemcpyDeviceToHost);

    cudaFree(convolution_kernel);
    cudaFree(device_in_image);
    cudaFree(device_out_image);
}

__forceinline__ __device__ float cubic_interpolation_cell(float *v, float x)
{
    return v[1] + 0.5 * x * (v[2] - v[0] + x * (2.0 * v[0] - 5.0 * v[1] + 4.0 * v[2] - v[3] + x * (3.0 * (v[1] - v[2]) + v[3] - v[0])));
}

__forceinline__ __device__ float bicubically_interpolate_cell(float p[4][4], float x, float y)
{
    float v[4];
    v[0] = cubic_interpolation_cell(p[0], y);
    v[1] = cubic_interpolation_cell(p[1], y);
    v[2] = cubic_interpolation_cell(p[2], y);
    v[3] = cubic_interpolation_cell(p[3], y);

    return cubic_interpolation_cell(v, x);
}

__forceinline__ __device__ float interpolate_value_bicubically(float *image_to_sample, float x_in, float y_in, float z_in, int width, int height, int depth)
{
    int sx = (x_in < 0.0) ? -1 : 1;
    int sy = (y_in < 0.0) ? -1 : 1;
    int x = max(0, min(width - 1, (int)(x_in)));
    int y = max(0, min(height - 1, (int)(y_in)));

    int mx = max(0, min(width - 1, (int)(x_in)-sx));
    int my = max(0, min(height - 1, (int)(y_in)-sy));
    int dx = max(0, min(width - 1, (int)(x_in) + sx));
    int dy = max(0, min(height - 1, (int)(y_in) + sy));
    int ddx = max(0, min(width - 1, (int)(x_in) + 2 * sx));
    int ddy = max(0, min(height - 1, (int)(y_in) + 2 * sy));

    if((x_in - sx < 0.0) || (x_in + 2 * sx < 0.0) || (x_in + 2 * sx > width - 1) || (x_in - sx > width - 1)

       || (y_in - sy < 0.0) || (y_in + 2 * sy < 0.0) || (y_in + 2 * sy > height - 1) || (y_in - sy > height - 1)

    )
    {
        return 0.0;
    }

    float p11 = image_to_sample[mx + width * my];
    float p12 = image_to_sample[x + width * my];
    float p13 = image_to_sample[dx + width * my];
    float p14 = image_to_sample[ddx + width * my];

    float p21 = image_to_sample[mx + width * y];
    float p22 = image_to_sample[x + width * y];
    float p23 = image_to_sample[dx + width * y];
    float p24 = image_to_sample[ddx + width * y];

    float p31 = image_to_sample[mx + width * dy];
    float p32 = image_to_sample[x + width * dy];
    float p33 = image_to_sample[dx + width * dy];
    float p34 = image_to_sample[ddx + width * dy];

    float p41 = image_to_sample[mx + width * ddy];
    float p42 = image_to_sample[x + width * ddy];
    float p43 = image_to_sample[dx + width * ddy];
    float p44 = image_to_sample[ddx + width * ddy];

    float pol[4][4] = {{p11, p21, p31, p41}, {p12, p22, p32, p42}, {p13, p23, p33, p43}, {p14, p24, p34, p44}};

    return bicubically_interpolate_cell(pol, x_in - x, y_in - y);
}

__forceinline__ __device__ float interpolate_value_bilinearly(const float *image_to_sample, const float x_in, const float y_in, const float z_in, const int width, const int height, const int depth)
{
    if(x_in > (float)(width - 1) || x_in < 0.0 || y_in > (float)(height - 1) || y_in < 0.0)
    {
        return 0.0;
    }

    int x_left = max(0, min(width - 1, ((int)x_in)));
    int x_right = max(0, min(width - 1, ((int)x_in) + 1));
    int y_bot = max(0, min(height - 1, ((int)y_in)));
    int y_top = max(0, min(height - 1, ((int)y_in) + 1));

    float weight_x = x_in - (float)(x_left);
    float weight_y = y_in - (float)(y_bot);

    long long bot_left_idx = x_left + y_bot * width;
    long long bot_right_idx = x_right + y_bot * width;
    long long top_left_idx = x_left + y_top * width;
    long long top_right_idx = x_right + y_top * width;

    float bottom_left_intensity = image_to_sample[bot_left_idx];
    float bottom_right_intensity = image_to_sample[bot_right_idx];
    float top_left_intensity = image_to_sample[top_left_idx];
    float top_right_intensity = image_to_sample[top_right_idx];

    float x_interp_bottom = (1.0 - weight_x) * bottom_left_intensity + weight_x * bottom_right_intensity;
    float x_interp_top = (1.0 - weight_x) * top_left_intensity + weight_x * top_right_intensity;

    float interpolated_value = (1.0 - weight_y) * x_interp_bottom + weight_y * x_interp_top;
    return interpolated_value;
}

__global__ void bilinear_interpolation_kernel(float *level_in, float *level_out, int width_in, int height_in, int depth_in, int width_out, int height_out, int depth_out)
{
    int global_idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    int global_idx_y = threadIdx.y + blockIdx.y * blockDim.y;

    int grid_offset_x = blockDim.x * gridDim.x;
    int grid_offset_y = blockDim.y * gridDim.y;

    float x_scale = width_in / (float)width_out;
    float y_scale = height_in / (float)height_out;
    float z_scale = depth_in / (float)depth_out;

    for(int y_idx = global_idx_y; y_idx < height_out; y_idx += grid_offset_y)
    {
        for(int x_idx = global_idx_x; x_idx < width_out; x_idx += grid_offset_x)
        {
            long long one_dimensional_center_idx = x_idx + y_idx * width_out;

            float x_fine_pos = x_idx * x_scale;
            float y_fine_pos = y_idx * y_scale;
            float z_fine_pos = 1 * z_scale;

            level_out[one_dimensional_center_idx] = interpolate_value_bilinearly(level_in, x_fine_pos, y_fine_pos, z_fine_pos, width_in, height_in, depth_in);
        }
    }
}

__global__ void warp_interpolation_kernel(float *device_in_layer, float *device_out_layer, float *device_in_u_vecs, float *device_in_v_vecs, int width, int height, int depth)
{
    int global_idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    int global_idx_y = threadIdx.y + blockIdx.y * blockDim.y;

    int grid_offset_x = blockDim.x * gridDim.x;
    int grid_offset_y = blockDim.y * gridDim.y;

    for(int y_idx = global_idx_y; y_idx < height; y_idx += grid_offset_y)
    {
        for(int x_idx = global_idx_x; x_idx < width; x_idx += grid_offset_x)
        {
            long long one_dimensional_center_idx = x_idx + y_idx * width;

            float u_vec = device_in_u_vecs[one_dimensional_center_idx];
            float v_vec = device_in_v_vecs[one_dimensional_center_idx];

            float warped_position_x = (float)(x_idx) + u_vec;
            float warped_position_y = (float)(y_idx) + v_vec;
            float warped_position_z = 1.0; //(float)(z_idx) + w_vec

            float interpolated_value = 0.0;

            if(warped_position_x >= 0.0 && warped_position_x <= (float)(width - 1) && warped_position_y >= 0.0 && warped_position_y <= (float)(height - 1))
            {
                interpolated_value = interpolate_value_bilinearly(device_in_layer, warped_position_x, warped_position_y, warped_position_z, width, height, depth);
            }

            device_out_layer[one_dimensional_center_idx] = interpolated_value;
        }
    }
}

__host__ void launch_bilinear_warp_interpolation_kernel(float *device_ptr_layer_in, float *device_ptr_same_size_layer_out, float *device_ptr_u_vec_field, float *device_ptr_v_vec_field,
                                                        int32_t *image_dimensions)
{
    dim3 block_dims(16, 16, 1);

    dim3 grid_dims(16, 16, 1);
    // launch kernel here

    warp_interpolation_kernel<<<grid_dims, block_dims>>>(device_ptr_layer_in, device_ptr_same_size_layer_out, device_ptr_u_vec_field, device_ptr_v_vec_field, image_dimensions[0], image_dimensions[1],
                                                         image_dimensions[2]);
}

__host__ void launch_bilinear_layer_downsampling_kernel(std::vector<float> const &pyramid_layer_in, std::vector<float> &pyramid_layer_out, int32_t *dimensions_image_in, int32_t *dimensions_image_out)
{
    float *device_in_layer;
    float *device_out_layer;

    int64_t num_elements_fine_level = pyramid_layer_in.size();
    int64_t num_elements_coarse_level = pyramid_layer_out.size();

    cudaError_t err = cudaPeekAtLastError();

    if(err != cudaSuccess)
    {
        printf("@ Pos X: The error is %s\n", cudaGetErrorString(err));
    }

    int64_t bytes_to_allocate_fine_level = num_elements_fine_level * sizeof(float);
    err = cudaMalloc((void **)&device_in_layer, bytes_to_allocate_fine_level);

    if(err != cudaSuccess)
    {
        printf("@ Pos 1: The error is %s\n", cudaGetErrorString(err));
    }

    int64_t bytes_to_allocate_coarse_level = num_elements_coarse_level * sizeof(float);

    err = cudaMalloc((void **)&device_out_layer, bytes_to_allocate_coarse_level);

    if(err != cudaSuccess)
    {
        printf("@ Pos 2: The error is %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(device_in_layer, &pyramid_layer_in[0], num_elements_fine_level * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(device_out_layer, 0, num_elements_coarse_level * sizeof(float));

    dim3 block_dims(16, 16, 1);

    dim3 grid_dims(16, 16, 1);

    bilinear_interpolation_kernel<<<grid_dims, block_dims>>>(device_in_layer, device_out_layer, dimensions_image_in[0], dimensions_image_in[1], dimensions_image_in[2], dimensions_image_out[0],
                                                             dimensions_image_out[1], dimensions_image_out[2]);

    cudaMemcpy(&pyramid_layer_out[0], device_out_layer, num_elements_coarse_level * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(device_in_layer);
    cudaFree(device_out_layer);
}

__host__ void create_pyramid_layers(std::vector<std::vector<float>> &scale_pyramid_1, std::vector<std::vector<float>> &scale_pyramid_2, std::vector<std::array<int32_t, 3>> &pyramid_sizes,
                                    float scaling_factor)
{
    float gaussian_sigma = 0.6 * sqrt(1.0 / (scaling_factor * scaling_factor) - 1.0);

    for(int downsampling_lvl = 1; downsampling_lvl < scale_pyramid_1.size(); ++downsampling_lvl)
    {
        int level_above = downsampling_lvl - 1;
        int64_t num_elements_in_pyramid = scale_pyramid_1[level_above].size();

        auto const &above_pyramid_level_sizes = pyramid_sizes[level_above];
        int32_t image_dimensions_level_above[3] = {above_pyramid_level_sizes[0], above_pyramid_level_sizes[1], above_pyramid_level_sizes[2]};

        auto const &curr_pyramid_level_sizes = pyramid_sizes[downsampling_lvl];
        int32_t image_dimensions_current_level[3] = {curr_pyramid_level_sizes[0], curr_pyramid_level_sizes[1], curr_pyramid_level_sizes[2]};

        // copy content

        // memcpy( &temp_image_curr_scale[0], &scale_pyramid_1[level_above][0], num_elements_in_pyramid*sizeof(float) );

        {
            std::vector<float> temp_image_curr_scale(scale_pyramid_1[level_above]);
            launch_spatial_gaussian_kernel(temp_image_curr_scale, gaussian_sigma, image_dimensions_level_above);

            // downsample layer 1 here ! otherwise the temp result gets overridden
            launch_bilinear_layer_downsampling_kernel(temp_image_curr_scale, scale_pyramid_1[downsampling_lvl], image_dimensions_level_above, image_dimensions_current_level);
        }
        // copy content
        // memcpy( &temp_image_curr_scale[0], &scale_pyramid_2[level_above][0], num_elements_in_pyramid*sizeof(float) );

        {
            std::vector<float> temp_image_curr_scale(scale_pyramid_2[level_above]);
            launch_spatial_gaussian_kernel(temp_image_curr_scale, gaussian_sigma, image_dimensions_level_above);

            // downsample layer 1 here ! otherwise the temp result gets overridden
            launch_bilinear_layer_downsampling_kernel(temp_image_curr_scale, scale_pyramid_2[downsampling_lvl], image_dimensions_level_above, image_dimensions_current_level);
        }
    }
}

__host__ void calculate_image_pyramid_sizes(std::vector<std::vector<float>> &scale_pyramid_1, std::vector<std::vector<float>> &scale_pyramid_2, std::vector<std::array<int32_t, 3>> &pyramid_sizes,
                                            float scaling_factor, int desired_num_scalings)
{
    std::array<int32_t, 3> prev_level_dims = pyramid_sizes[0];
    for(int scaling_idx = 1; scaling_idx < desired_num_scalings; ++scaling_idx)
    {
        int64_t flattened_level_size = 1;
        for(int dim_idx = 0; dim_idx < 3; ++dim_idx)
        {
            int32_t dimension_size = max(1, (int32_t)(prev_level_dims[dim_idx] * scaling_factor + 0.5));

            pyramid_sizes[scaling_idx][dim_idx] = dimension_size;
            prev_level_dims[dim_idx] = dimension_size;
            flattened_level_size *= dimension_size;
        }

        scale_pyramid_1[scaling_idx].resize(flattened_level_size);
        scale_pyramid_2[scaling_idx].resize(flattened_level_size);
    }
}

__global__ void prepare_SOR_kernel_stage_1(const float *device_in_ptr_I0, const float *device_in_ptr_I1, const float *device_in_ptr_Ix0, const float *device_in_ptr_Iy0, const float *device_in_ptr_Ix1,
                                           const float *device_in_ptr_Iy1, const float *device_in_ptr_Ixx1, const float *device_in_ptr_Iyy1, const float *device_in_ptr_Ixy1,

                                           float *device_out_ptr_diffusivity_x, float *device_out_ptr_diffusivity_y, float *device_out_ptr_denom_u, float *device_out_ptr_denom_v,
                                           float *device_out_ptr_num_dudv, float *device_out_ptr_num_u, float *device_out_ptr_num_v, const float *device_in_out_ptr_flow_field_x,
                                           const float *device_in_out_ptr_flow_field_y, const float *device_in_out_ptr_flow_delta_x, const float *device_in_out_ptr_flow_delta_y, const float alpha,
                                           const float gamma, const int width, const int height, const int depth)
{
    int global_idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    int global_idx_y = threadIdx.y + blockIdx.y * blockDim.y;

    int grid_offset_x = blockDim.x * gridDim.x;
    int grid_offset_y = blockDim.y * gridDim.y;

    for(int y_idx = global_idx_y; y_idx < height; y_idx += grid_offset_y)
    {
        for(int x_idx = global_idx_x; x_idx < width; x_idx += grid_offset_x)
        {
            long long one_dimensional_center_idx = x_idx + y_idx * width;

            float u = device_in_out_ptr_flow_field_x[one_dimensional_center_idx];
            float v = device_in_out_ptr_flow_field_y[one_dimensional_center_idx];

            float warped_x = x_idx + device_in_out_ptr_flow_field_x[one_dimensional_center_idx];
            float warped_y = y_idx + device_in_out_ptr_flow_field_y[one_dimensional_center_idx];

            float Iz =
                interpolate_value_bilinearly(device_in_ptr_I1, warped_x, warped_y, 1.0, width, height, depth) - interpolate_value_bilinearly(device_in_ptr_I0, x_idx, y_idx, 1.0, width, height, depth);

            float Ix = interpolate_value_bilinearly(device_in_ptr_Ix1, warped_x, warped_y, 1.0, width, height, depth);
            float Ixz = Ix - interpolate_value_bilinearly(device_in_ptr_Ix0, x_idx, y_idx, 1.0, width, height, depth);
            float Ixy = interpolate_value_bilinearly(device_in_ptr_Ixy1, warped_x, warped_y, 1.0, width, height, depth);
            float Ixx = interpolate_value_bilinearly(device_in_ptr_Ixx1, warped_x, warped_y, 1.0, width, height, depth);
            float Iy = interpolate_value_bilinearly(device_in_ptr_Iy1, warped_x, warped_y, 1.0, width, height, depth);
            float Iyz = Iy - interpolate_value_bilinearly(device_in_ptr_Iy0, x_idx, y_idx, 1.0, width, height, depth);
            float Iyy = interpolate_value_bilinearly(device_in_ptr_Iyy1, warped_x, warped_y, 1.0, width, height, depth);

            // data term

            float du = device_in_out_ptr_flow_delta_x[one_dimensional_center_idx];
            float dv = device_in_out_ptr_flow_delta_y[one_dimensional_center_idx];

            float q0 = Iz + Ix * du + Iy * dv;
            float q1 = Ixz + Ixx * du + Ixy * dv;
            float q2 = Iyz + Ixy * du + Iyy * dv;

            float data_term = 0.5f * rsqrtf(q0 * q0 + gamma * (q1 * q1 + q2 * q2) + eps2);

            data_term /= alpha;

            float sx = 0.0;
            float sy = 0.0;

            if(x_idx > 0)
            {
                // sx = diffusivity_along_x();

                const int left_idx = max(0, x_idx - 1);
                const int one_d_left_idx = left_idx + y_idx * width;

                float u_left = device_in_out_ptr_flow_field_x[one_d_left_idx];
                float v_left = device_in_out_ptr_flow_field_y[one_d_left_idx];
                float du_left = device_in_out_ptr_flow_delta_x[one_d_left_idx];
                float dv_left = device_in_out_ptr_flow_delta_y[one_d_left_idx];

                // x derivative between pixels (i,j) and (i-1,j
                float u_x = u + du - u_left - du_left;
                float v_x = v + dv - v_left - dv_left;

                const int up_idx = min(height - 1, y_idx + 1);
                const int one_d_up_idx = x_idx + up_idx * width;
                const int one_d_up_left_idx = left_idx + up_idx * width;

                const int down_idx = max(0, y_idx - 1);
                const int one_d_down_idx = x_idx + down_idx * width;
                const int one_d_down_left_idx = left_idx + down_idx * width;

                // y derivative between pixels (i,j) and (i-1,j)
                float u_up = device_in_out_ptr_flow_field_x[one_d_up_idx];
                float du_up = device_in_out_ptr_flow_delta_x[one_d_up_idx];
                float u_up_left = device_in_out_ptr_flow_field_x[one_d_up_left_idx];
                float du_up_left = device_in_out_ptr_flow_delta_x[one_d_up_left_idx];
                float u_down = device_in_out_ptr_flow_field_x[one_d_down_idx];
                float du_down = device_in_out_ptr_flow_delta_x[one_d_down_idx];
                float u_down_left = device_in_out_ptr_flow_field_x[one_d_down_left_idx];
                float du_down_left = device_in_out_ptr_flow_delta_x[one_d_down_left_idx];

                float v_up = device_in_out_ptr_flow_field_y[one_d_up_idx];
                float dv_up = device_in_out_ptr_flow_delta_y[one_d_up_idx];
                float v_up_left = device_in_out_ptr_flow_field_y[one_d_up_left_idx];
                float dv_up_left = device_in_out_ptr_flow_delta_y[one_d_up_left_idx];
                float v_down = device_in_out_ptr_flow_field_y[one_d_down_idx];
                float dv_down = device_in_out_ptr_flow_delta_y[one_d_down_idx];
                float v_down_left = device_in_out_ptr_flow_field_y[one_d_down_left_idx];
                float dv_down_left = device_in_out_ptr_flow_delta_y[one_d_down_left_idx];

                float u_y = 0.25f * (u_up + du_up + u_up_left + du_up_left - u_down - du_down - u_down_left - du_down_left);

                float v_y = 0.25f * (v_up + dv_up + v_up_left + dv_up_left - v_down - dv_down - v_down_left - dv_down_left);
                sx = 0.5f / sqrtf(u_x * u_x + v_x * v_x + u_y * u_y + v_y * v_y + eps2);
            }

            if(y_idx > 0)
            {
                const int left_idx = max(0, x_idx - 1);
                const int right_idx = min(width - 1, x_idx + 1);
                const int down_idx = max(0, y_idx - 1);
                const int one_d_down_idx = x_idx + down_idx * width;

                float u_down = device_in_out_ptr_flow_field_x[one_d_down_idx];
                float v_down = device_in_out_ptr_flow_field_y[one_d_down_idx];
                float du_down = device_in_out_ptr_flow_delta_x[one_d_down_idx];
                float dv_down = device_in_out_ptr_flow_delta_y[one_d_down_idx];

                // x derivative between pixels (i,j) and (i-1,j
                float u_y = u + du - u_down - du_down;
                float v_y = v + dv - v_down - dv_down;

                const int one_d_left_idx = left_idx + y_idx * width;
                const int one_d_down_left_idx = left_idx + down_idx * width;
                const int one_d_down_right_idx = right_idx + down_idx * width;
                const int one_d_right_idx = right_idx + y_idx * width;
                // y derivative between pixels (i,j) and (i-1,j)

                float u_down_right = device_in_out_ptr_flow_field_x[one_d_down_right_idx];
                float du_down_right = device_in_out_ptr_flow_delta_x[one_d_down_right_idx];
                float u_left = device_in_out_ptr_flow_field_x[one_d_left_idx];
                float du_left = device_in_out_ptr_flow_delta_x[one_d_left_idx];
                float u_down_left = device_in_out_ptr_flow_field_x[one_d_down_left_idx];
                float du_down_left = device_in_out_ptr_flow_delta_x[one_d_down_left_idx];
                float u_right = device_in_out_ptr_flow_field_x[one_d_right_idx];
                float du_right = device_in_out_ptr_flow_delta_x[one_d_right_idx];

                float v_down_right = device_in_out_ptr_flow_field_y[one_d_down_right_idx];
                float dv_down_right = device_in_out_ptr_flow_delta_y[one_d_down_right_idx];
                float v_left = device_in_out_ptr_flow_field_y[one_d_left_idx];
                float dv_left = device_in_out_ptr_flow_delta_y[one_d_left_idx];
                float v_down_left = device_in_out_ptr_flow_field_y[one_d_down_left_idx];
                float dv_down_left = device_in_out_ptr_flow_delta_y[one_d_down_left_idx];
                float v_right = device_in_out_ptr_flow_field_y[one_d_right_idx];
                float dv_right = device_in_out_ptr_flow_delta_y[one_d_right_idx];

                float u_x = 0.25f * (u_right + du_right + u_down_right + du_down_right - u_left - du_left - u_down_left - du_down_left);

                float v_x = 0.25f * (v_right + dv_right + v_down_right + dv_down_right - v_left - dv_left - v_down_left - dv_down_left);
                sy = 0.5f / sqrtf(u_x * u_x + v_x * v_x + u_y * u_y + v_y * v_y + eps2);
            }

            device_out_ptr_num_dudv[one_dimensional_center_idx] = data_term * (Ix * Iy + gamma * Ixy * (Ixx + Iyy));
            device_out_ptr_num_u[one_dimensional_center_idx] = data_term * (Ix * Iz + gamma * (Ixx * Ixz + Ixy * Iyz));
            device_out_ptr_num_v[one_dimensional_center_idx] = data_term * (Iy * Iz + gamma * (Iyy * Iyz + Ixy * Ixz));

            device_out_ptr_denom_u[one_dimensional_center_idx] = data_term * (Ix * Ix + gamma * (Ixy * Ixy + Ixx * Ixx));
            device_out_ptr_denom_v[one_dimensional_center_idx] = data_term * (Iy * Iy + gamma * (Ixy * Ixy + Iyy * Iyy));

            device_out_ptr_diffusivity_x[one_dimensional_center_idx] = sx;
            device_out_ptr_diffusivity_y[one_dimensional_center_idx] = sy;
        }
    }
}

__global__ void prepare_SOR_kernel_stage_2(float *device_out_ptr_diffusivity_x, float *device_out_ptr_diffusivity_y, float *device_out_ptr_inv_denom_u, float *device_out_ptr_inv_denom_v,

                                           int width, int height, int depth)
{
    int global_idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    int global_idx_y = threadIdx.y + blockIdx.y * blockDim.y;

    int grid_offset_x = blockDim.x * gridDim.x;
    int grid_offset_y = blockDim.y * gridDim.y;

    for(int y_idx = global_idx_y; y_idx < height; y_idx += grid_offset_y)
    {
        for(int x_idx = global_idx_x; x_idx < width; x_idx += grid_offset_x)
        {
            long long one_dimensional_center_idx = x_idx + y_idx * width;
            int idx_up = min(height - 1, y_idx + 1);

            int idx_right = min(width - 1, x_idx + 1);
            long long one_d_idx_right = idx_right + y_idx * width;
            long long one_d_idx_up = x_idx + idx_up * width;
            float denom_u = device_out_ptr_inv_denom_u[one_dimensional_center_idx];
            float denom_v = device_out_ptr_inv_denom_v[one_dimensional_center_idx];

            float sx_pos = device_out_ptr_diffusivity_x[one_dimensional_center_idx];
            float sy_pos = device_out_ptr_diffusivity_y[one_dimensional_center_idx];

            float sy_up = device_out_ptr_diffusivity_y[one_d_idx_up];
            if(y_idx == height - 1)
            {
                sy_up = 0.0;
            }

            float sx_right = device_out_ptr_diffusivity_x[one_d_idx_right];

            if(x_idx == width - 1)
            {
                sx_right = 0.0;
            }

            float diffusivity_sum = sx_pos + sx_right + sy_pos + sy_up;

            denom_u += diffusivity_sum;
            denom_v += diffusivity_sum;

            device_out_ptr_inv_denom_u[one_dimensional_center_idx] = 1.0f / denom_u;
            device_out_ptr_inv_denom_v[one_dimensional_center_idx] = 1.0f / denom_v;
        }
    }
}

__global__ void sor_pass(int pass, float *new_du, float *new_dv, const float *device_u, const float *device_v, const float *device_du, const float *device_dv, const float *device_diffusivity_x,
                         const float *device_diffusivity_y, const float *device_inv_denominator_u, const float *device_inv_denominator_v, const float *device_numerator_u,
                         const float *device_numerator_v, const float *device_numerator_dudv, float omega, int width, int height, int depth)
{
    int global_idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    int global_idx_y = threadIdx.y + blockIdx.y * blockDim.y;

    int grid_offset_x = blockDim.x * gridDim.x;
    int grid_offset_y = blockDim.y * gridDim.y;

    for(int y_idx = global_idx_y; y_idx < height; y_idx += grid_offset_y)
    {
        for(int x_idx = global_idx_x; x_idx < width; x_idx += grid_offset_x)
        {
            long long one_dimensional_center_idx = x_idx + y_idx * width;

            float s_left = device_diffusivity_x[one_dimensional_center_idx];
            float s_down = device_diffusivity_y[one_dimensional_center_idx];
            float s_right = 0.0;
            float s_up = 0.0;

            if(x_idx < width - 1)
            {
                s_right = device_diffusivity_x[(x_idx + 1) + y_idx * width];
            }

            if(y_idx < height - 1)
            {
                s_up = device_diffusivity_y[x_idx + (y_idx + 1) * width];
            }

            int x_left = x_idx > 0 ? x_idx - 1 : x_idx;
            int x_right = x_idx < width - 1 ? x_idx + 1 : x_idx;
            int y_down = y_idx > 0 ? y_idx - 1 : y_idx;
            int y_up = y_idx < height - 1 ? y_idx + 1 : y_idx;

            long long one_d_left_idx = x_left + y_idx * width;
            long long one_d_right_idx = x_right + y_idx * width;
            long long one_d_down_idx = x_idx + y_down * width;
            long long one_d_up_idx = x_idx + y_up * width;

            float u_up = device_u[one_d_up_idx];
            float u_down = device_u[one_d_down_idx];
            float u_left = device_u[one_d_left_idx];
            float u_right = device_u[one_d_right_idx];
            float u = device_u[one_dimensional_center_idx];

            float v_up = device_v[one_d_up_idx];
            float v_down = device_v[one_d_down_idx];
            float v_left = device_v[one_d_left_idx];
            float v_right = device_v[one_d_right_idx];
            float v = device_v[one_dimensional_center_idx];

            float du_up = device_du[one_d_up_idx];
            float du_down = device_du[one_d_down_idx];
            float du_left = device_du[one_d_left_idx];
            float du_right = device_du[one_d_right_idx];
            float du = device_du[one_dimensional_center_idx];

            float dv_up = device_dv[one_d_up_idx];
            float dv_down = device_dv[one_d_down_idx];
            float dv_left = device_dv[one_d_left_idx];
            float dv_right = device_dv[one_d_right_idx];
            float dv = device_dv[one_dimensional_center_idx];

            float numerator_dudv = device_numerator_dudv[one_dimensional_center_idx];

            if((x_idx + y_idx) % 2 == pass)
            {
                // update du
                float numerator_u = (s_left * (u_left + du_left) + s_up * (u_up + du_up) + s_right * (u_right + du_right) + s_down * (u_down + du_down) - u * (s_left + s_right + s_up + s_down) -
                                     device_numerator_u[one_dimensional_center_idx] - numerator_dudv * dv);

                du = (1.0f - omega) * du + omega * device_inv_denominator_u[one_dimensional_center_idx] * numerator_u;

                // update dv
                float numerator_v = (s_left * (v_left + dv_left) + s_up * (v_up + dv_up) + s_right * (v_right + dv_right) + s_down * (v_down + dv_down) - v * (s_left + s_right + s_up + s_down) -
                                     device_numerator_v[one_dimensional_center_idx] - numerator_dudv * du);

                dv = (1.0f - omega) * dv + omega * device_inv_denominator_v[one_dimensional_center_idx] * numerator_v;
            }

            new_du[one_dimensional_center_idx] = du;
            new_dv[one_dimensional_center_idx] = dv;
        }
    }
}

__global__ void add_delta_to_flow_fields_kernel(float *device_in_out_flow_x, float *device_in_out_flow_y, const float *device_in_delta_x, const float *device_in_delta_y, int width, int height,
                                                int depth)
{
    int global_idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    int global_idx_y = threadIdx.y + blockIdx.y * blockDim.y;

    int grid_offset_x = blockDim.x * gridDim.x;
    int grid_offset_y = blockDim.y * gridDim.y;

    for(int y_idx = global_idx_y; y_idx < height; y_idx += grid_offset_y)
    {
        for(int x_idx = global_idx_x; x_idx < width; x_idx += grid_offset_x)
        {
            long long one_dimensional_center_idx = x_idx + y_idx * width;

            device_in_out_flow_x[one_dimensional_center_idx] += device_in_delta_x[one_dimensional_center_idx];
            device_in_out_flow_y[one_dimensional_center_idx] += device_in_delta_y[one_dimensional_center_idx];
        }
    }
}

__host__ void launch_brox_inner_loop_kernel_tasks(float *device_ptr_I0, float *device_ptr_I1, float *device_ptr_Ix0, float *device_ptr_Iy0, float *device_ptr_Ix1, float *device_ptr_Iy1,
                                                  float *device_ptr_Ixx1, float *device_ptr_Iyy1, float *device_ptr_Ixy1, float *device_out_ptr_diffusivity_x, float *device_out_ptr_diffusivity_y,
                                                  float *device_out_ptr_denom_u, float *device_out_ptr_denom_v, float *device_out_ptr_num_dudv, float *device_out_ptr_num_u,
                                                  float *device_out_ptr_num_v, float *device_in_out_ptr_flow_field_x, float *device_in_out_ptr_flow_field_y,
                                                  float *device_in_out_ptr_flow_field_delta_x, float *device_in_out_ptr_flow_field_delta_y,

                                                  float alpha, float gamma, float scaling_factor, int num_inner_iterations, int num_outer_iterations, int num_solver_iterations,
                                                  int32_t *current_layer_dims, int64_t num_bytes_to_alloc)
{
    dim3 block_dims(16, 16, 1);
    dim3 grid_dims(16, 16, 1);
    prepare_SOR_kernel_stage_1<<<grid_dims, block_dims>>>(
        device_ptr_I0, device_ptr_I1, device_ptr_Ix0, device_ptr_Iy0, device_ptr_Ix1, device_ptr_Iy1, device_ptr_Ixx1, device_ptr_Iyy1, device_ptr_Ixy1,

        device_out_ptr_diffusivity_x, device_out_ptr_diffusivity_y, device_out_ptr_denom_u, device_out_ptr_denom_v, device_out_ptr_num_dudv, device_out_ptr_num_u, device_out_ptr_num_v,
        device_in_out_ptr_flow_field_x, device_in_out_ptr_flow_field_y, device_in_out_ptr_flow_field_delta_x, device_in_out_ptr_flow_field_delta_y, alpha, gamma, current_layer_dims[0],
        current_layer_dims[1], current_layer_dims[2]);

    cudaDeviceSynchronize();

    prepare_SOR_kernel_stage_2<<<grid_dims, block_dims>>>(device_out_ptr_diffusivity_x, device_out_ptr_diffusivity_y, device_out_ptr_denom_u, device_out_ptr_denom_v, current_layer_dims[0],
                                                          current_layer_dims[1], current_layer_dims[2]);

    float *new_du;
    float *new_dv;

    cudaMalloc((char **)&new_du, num_bytes_to_alloc);
    cudaMemcpy(new_du, device_in_out_ptr_flow_field_delta_x, num_bytes_to_alloc, cudaMemcpyDeviceToDevice);
    // cudaMemset(new_du, 0, num_bytes_to_alloc);
    cudaMalloc((char **)&new_dv, num_bytes_to_alloc);
    cudaMemcpy(new_dv, device_in_out_ptr_flow_field_delta_y, num_bytes_to_alloc, cudaMemcpyDeviceToDevice);
    // cudaMemset(new_dv, 0, num_bytes_to_alloc);

    const float omega = 1.99f;
    for(int solver_it_idx = 0; solver_it_idx < num_solver_iterations; ++solver_it_idx)
    {
        // call sor pass 1
        sor_pass<<<grid_dims, block_dims>>>(0, new_du, new_dv, device_in_out_ptr_flow_field_x, device_in_out_ptr_flow_field_y, device_in_out_ptr_flow_field_delta_x,
                                            device_in_out_ptr_flow_field_delta_y, device_out_ptr_diffusivity_x, device_out_ptr_diffusivity_y, device_out_ptr_denom_u, device_out_ptr_denom_v,
                                            device_out_ptr_num_u, device_out_ptr_num_v, device_out_ptr_num_dudv, omega, current_layer_dims[0], current_layer_dims[1], current_layer_dims[2]);

        cudaDeviceSynchronize();

        sor_pass<<<grid_dims, block_dims>>>(1, device_in_out_ptr_flow_field_delta_x, device_in_out_ptr_flow_field_delta_y, device_in_out_ptr_flow_field_x, device_in_out_ptr_flow_field_y, new_du,
                                            new_dv, device_out_ptr_diffusivity_x, device_out_ptr_diffusivity_y, device_out_ptr_denom_u, device_out_ptr_denom_v, device_out_ptr_num_u,
                                            device_out_ptr_num_v, device_out_ptr_num_dudv, omega, current_layer_dims[0], current_layer_dims[1], current_layer_dims[2]);

        cudaDeviceSynchronize();
    }

    cudaFree(new_du);
    cudaFree(new_dv);
}

__host__ void launch_brox_outer_loop_kernel_tasks(std::vector<float> const &current_layer_1, std::vector<float> const &current_layer_2, std::vector<float> &grad_x_0_current_layer,
                                                  std::vector<float> &grad_y_0_current_layer, std::vector<float> &flow_field_in_out_x, std::vector<float> &flow_field_in_out_y, float alpha,
                                                  float gamma, float scaling_factor, int num_inner_iterations, int num_outer_iterations, int num_solver_iterations, int32_t *current_layer_dims)
{
    float derivative_filter[3] = {0.0f, 8.0f, -1.0f};

    float *device_derivative_filter;
    cudaMalloc((void **)&device_derivative_filter, sizeof(float) * 3);
    cudaMemcpy(device_derivative_filter, derivative_filter, sizeof(float) * 3, cudaMemcpyHostToDevice);

    int one_sided_filter_size = 3;

    long long num_elements_per_layer = current_layer_1.size();
    long long num_bytes_to_alloc = num_elements_per_layer * sizeof(float);

    float *device_in_image_layer_0;
    cudaMalloc((void **)&device_in_image_layer_0, num_bytes_to_alloc);
    cudaMemcpy(device_in_image_layer_0, &current_layer_1[0], num_bytes_to_alloc, cudaMemcpyHostToDevice);
    float *device_in_image_layer_1;
    cudaMalloc((void **)&device_in_image_layer_1, num_bytes_to_alloc);
    cudaMemcpy(device_in_image_layer_1, &current_layer_2[0], num_bytes_to_alloc, cudaMemcpyHostToDevice);

    float *device_out_image_0_grad_x;
    cudaMalloc((void **)&device_out_image_0_grad_x, num_bytes_to_alloc);
    float *device_out_image_0_grad_y;
    cudaMalloc((void **)&device_out_image_0_grad_y, num_bytes_to_alloc);

    float *device_out_image_1_grad_x;
    cudaMalloc((void **)&device_out_image_1_grad_x, num_bytes_to_alloc);
    float *device_out_image_1_grad_y;
    cudaMalloc((void **)&device_out_image_1_grad_y, num_bytes_to_alloc);

    float *device_out_image_1_grad_xx;
    cudaMalloc((void **)&device_out_image_1_grad_xx, num_bytes_to_alloc);
    float *device_out_image_1_grad_yy;
    cudaMalloc((void **)&device_out_image_1_grad_yy, num_bytes_to_alloc);
    float *device_out_image_1_grad_xy;
    cudaMalloc((void **)&device_out_image_1_grad_xy, num_bytes_to_alloc);

    float *device_in_out_flow_x;
    float *device_in_out_flow_y;
    cudaMalloc((void **)&device_in_out_flow_x, num_bytes_to_alloc);
    cudaMemcpy(device_in_out_flow_x, &flow_field_in_out_x[0], num_bytes_to_alloc, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&device_in_out_flow_y, num_bytes_to_alloc);
    cudaMemcpy(device_in_out_flow_y, &flow_field_in_out_y[0], num_bytes_to_alloc, cudaMemcpyHostToDevice);

    float *device_in_out_flow_delta_x;
    cudaMalloc((void **)&device_in_out_flow_delta_x, num_bytes_to_alloc);
    cudaMemset(device_in_out_flow_delta_x, 0, num_bytes_to_alloc);

    float *device_in_out_flow_delta_y;
    cudaMalloc((void **)&device_in_out_flow_delta_y, num_bytes_to_alloc);
    cudaMemset(device_in_out_flow_delta_y, 0, num_bytes_to_alloc);

    dim3 block_dims(16, 16, 1);
    dim3 grid_dims(16, 16, 1);

    calc_image_gradient_1d<<<grid_dims, block_dims>>>(device_in_image_layer_0, device_out_image_0_grad_x, current_layer_dims[0], current_layer_dims[1], current_layer_dims[2], device_derivative_filter,
                                                      one_sided_filter_size, true);

    calc_image_gradient_1d<<<grid_dims, block_dims>>>(device_in_image_layer_0, device_out_image_0_grad_y, current_layer_dims[0], current_layer_dims[1], current_layer_dims[2], device_derivative_filter,
                                                      one_sided_filter_size, false);

    cudaMemcpy(&grad_x_0_current_layer[0], device_out_image_0_grad_y, num_bytes_to_alloc, cudaMemcpyDeviceToHost);
    cudaMemcpy(&grad_y_0_current_layer[0], device_out_image_0_grad_y, num_bytes_to_alloc, cudaMemcpyDeviceToHost);

    calc_image_gradient_1d<<<grid_dims, block_dims>>>(device_in_image_layer_1, device_out_image_1_grad_x, current_layer_dims[0], current_layer_dims[1], current_layer_dims[2], device_derivative_filter,
                                                      one_sided_filter_size, true);

    calc_image_gradient_1d<<<grid_dims, block_dims>>>(device_in_image_layer_1, device_out_image_1_grad_y, current_layer_dims[0], current_layer_dims[1], current_layer_dims[2], device_derivative_filter,
                                                      one_sided_filter_size, false);

    calc_image_gradient_1d<<<grid_dims, block_dims>>>(device_out_image_1_grad_x, device_out_image_1_grad_xx, current_layer_dims[0], current_layer_dims[1], current_layer_dims[2],
                                                      device_derivative_filter, one_sided_filter_size, true);

    calc_image_gradient_1d<<<grid_dims, block_dims>>>(device_out_image_1_grad_y, device_out_image_1_grad_yy, current_layer_dims[0], current_layer_dims[1], current_layer_dims[2],
                                                      device_derivative_filter, one_sided_filter_size, false);

    calc_image_gradient_1d<<<grid_dims, block_dims>>>(device_out_image_1_grad_x, device_out_image_1_grad_xy, current_layer_dims[0], current_layer_dims[1], current_layer_dims[2],
                                                      device_derivative_filter, one_sided_filter_size, false);

    std::vector<float> check_on_velocities(num_elements_per_layer);

    cudaMemcpy(&check_on_velocities[0], device_out_image_1_grad_xy, num_bytes_to_alloc, cudaMemcpyDeviceToHost);

    float *device_out_diffusivity_x;
    float *device_out_diffusivity_y;
    float *device_out_denom_u;
    float *device_out_denom_v;
    float *device_out_num_dudv;
    float *device_out_num_u;
    float *device_out_num_v;

    cudaMalloc((void **)&device_out_diffusivity_x, num_bytes_to_alloc);
    cudaMalloc((void **)&device_out_diffusivity_y, num_bytes_to_alloc);
    cudaMalloc((void **)&device_out_denom_u, num_bytes_to_alloc);
    cudaMalloc((void **)&device_out_denom_v, num_bytes_to_alloc);
    cudaMalloc((void **)&device_out_num_dudv, num_bytes_to_alloc);
    cudaMalloc((void **)&device_out_num_u, num_bytes_to_alloc);
    cudaMalloc((void **)&device_out_num_v, num_bytes_to_alloc);

    for(int64_t inner_iteration_idx = 0; inner_iteration_idx < num_inner_iterations; ++inner_iteration_idx)
    {
        launch_brox_inner_loop_kernel_tasks(device_in_image_layer_0, device_in_image_layer_1, device_out_image_0_grad_x, device_out_image_0_grad_y, device_out_image_1_grad_x,
                                            device_out_image_1_grad_y, device_out_image_1_grad_xx, device_out_image_1_grad_yy, device_out_image_1_grad_xy, device_out_diffusivity_x,
                                            device_out_diffusivity_y, device_out_denom_u, device_out_denom_v, device_out_num_dudv, device_out_num_u, device_out_num_v,

                                            device_in_out_flow_x, device_in_out_flow_y, device_in_out_flow_delta_x, device_in_out_flow_delta_y, alpha, gamma, scaling_factor, num_inner_iterations,
                                            num_outer_iterations, num_solver_iterations, current_layer_dims, num_bytes_to_alloc);
    }

    // add du & dv to u and v
    add_delta_to_flow_fields_kernel<<<grid_dims, block_dims>>>(device_in_out_flow_x, device_in_out_flow_y, device_in_out_flow_delta_x, device_in_out_flow_delta_y, current_layer_dims[0],
                                                               current_layer_dims[1], current_layer_dims[2]);

    cudaMemcpy(&check_on_velocities[0], device_in_out_flow_x, num_bytes_to_alloc, cudaMemcpyDeviceToHost);
    // std::cout << "After field addition: " <<  check_on_velocities[ current_layer_dims[0]/2 + (current_layer_dims[1]/2)*current_layer_dims[0] ] << "\n";

    cudaMemcpy(&flow_field_in_out_x[0], device_in_out_flow_x, num_bytes_to_alloc, cudaMemcpyDeviceToHost);
    cudaMemcpy(&flow_field_in_out_y[0], device_in_out_flow_y, num_bytes_to_alloc, cudaMemcpyDeviceToHost);

    cudaFree(device_derivative_filter);

    cudaFree(device_out_diffusivity_x);
    cudaFree(device_out_diffusivity_y);
    cudaFree(device_out_denom_u);
    cudaFree(device_out_denom_v);
    cudaFree(device_out_num_dudv);
    cudaFree(device_out_num_u);
    cudaFree(device_out_num_v);

    cudaFree(device_in_image_layer_0);
    cudaFree(device_in_image_layer_1);
    cudaFree(device_out_image_0_grad_x);
    cudaFree(device_out_image_0_grad_y);
    cudaFree(device_out_image_1_grad_x);
    cudaFree(device_out_image_1_grad_y);
    cudaFree(device_out_image_1_grad_xx);
    cudaFree(device_out_image_1_grad_yy);
    cudaFree(device_out_image_1_grad_xy);

    cudaFree(device_in_out_flow_x);
    cudaFree(device_in_out_flow_y);

    cudaFree(device_in_out_flow_delta_x);
    cudaFree(device_in_out_flow_delta_y);
}

__global__ void vector_field_upsampling_kernel(const float *device_in_low_res_flow_x, const float *device_in_low_res_flow_y, float *device_in_high_res_flow_x, float *device_in_high_res_flow_y,
                                               int low_res_width, int low_res_height, int low_res_depth, int high_res_width, int high_res_height, int high_res_depth)
{
    int global_idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    int global_idx_y = threadIdx.y + blockIdx.y * blockDim.y;

    int grid_offset_x = blockDim.x * gridDim.x;
    int grid_offset_y = blockDim.y * gridDim.y;

    float h_to_l_scale_x = low_res_width / (float)(high_res_width);
    float h_to_l_scale_y = low_res_height / (float)(high_res_height);
    float h_to_l_scale_z = low_res_depth / (float)(high_res_depth);

    float x_vec_scaling = 1.0f / h_to_l_scale_x;
    float y_vec_scaling = 1.0f / h_to_l_scale_y;
    float z_vec_scaling = 1.0f / h_to_l_scale_z;

    for(int y_idx = global_idx_y; y_idx < high_res_height; y_idx += grid_offset_y)
    {
        for(int x_idx = global_idx_x; x_idx < high_res_width; x_idx += grid_offset_x)
        {
            long long one_dimensional_center_idx = x_idx + y_idx * high_res_width;

            float x_sample_pos = max(0.0f, min((float)(low_res_width)-1.0f, (float)(x_idx * h_to_l_scale_x)));
            float y_sample_pos = max(0.0f, min((float)(low_res_height)-1.0f, (float)(y_idx * h_to_l_scale_y)));
            // float z_sample_pos = max(0.0f, min( (float)(low_res_depth)-1.0f, (float)(z_idx * h_to_l_scale_z) ) );

            float looked_up_vec_x_comp = x_vec_scaling * interpolate_value_bilinearly(device_in_low_res_flow_x, x_sample_pos, y_sample_pos, 1.0, low_res_width, low_res_height, low_res_depth);

            float looked_up_vec_y_comp = y_vec_scaling * interpolate_value_bilinearly(device_in_low_res_flow_y, x_sample_pos, y_sample_pos, 1.0, low_res_width, low_res_height, low_res_depth);

            // do the same fow z

            device_in_high_res_flow_x[one_dimensional_center_idx] = looked_up_vec_x_comp;
            device_in_high_res_flow_y[one_dimensional_center_idx] = looked_up_vec_y_comp;
        }
    }
}

__host__ void launch_vector_field_upsampling_kernel(std::vector<float> const &low_res_flow_x, std::vector<float> const &low_res_flow_y, std::vector<float> &high_res_flow_x,
                                                    std::vector<float> &high_res_flow_y, int32_t *low_res_layer_dims, int32_t *high_res_layer_dims)
{
    float *device_low_res_flow_x;
    float *device_low_res_flow_y;
    float *device_high_res_flow_x;
    float *device_high_res_flow_y;

    int64_t num_elements_low_res = low_res_layer_dims[0] * low_res_layer_dims[1] * low_res_layer_dims[2];
    int64_t num_bytes_to_alloc_low_res = num_elements_low_res * sizeof(float);

    int64_t num_elements_high_res = high_res_layer_dims[0] * high_res_layer_dims[1] * high_res_layer_dims[2];
    int64_t num_bytes_to_alloc_high_res = num_elements_high_res * sizeof(float);

    cudaMalloc((char **)&device_low_res_flow_x, num_bytes_to_alloc_low_res);
    cudaMemcpy(device_low_res_flow_x, &low_res_flow_x[0], num_bytes_to_alloc_low_res, cudaMemcpyHostToDevice);
    cudaMalloc((char **)&device_low_res_flow_y, num_bytes_to_alloc_low_res);
    cudaMemcpy(device_low_res_flow_y, &low_res_flow_y[0], num_bytes_to_alloc_low_res, cudaMemcpyHostToDevice);

    cudaMalloc((char **)&device_high_res_flow_x, num_bytes_to_alloc_high_res);

    cudaMalloc((char **)&device_high_res_flow_y, num_bytes_to_alloc_high_res);

    dim3 block_dims(16, 16, 1);
    dim3 grid_dims(16, 16, 1);

    vector_field_upsampling_kernel<<<grid_dims, block_dims>>>(device_low_res_flow_x, device_low_res_flow_y, device_high_res_flow_x, device_high_res_flow_y, low_res_layer_dims[0],
                                                              low_res_layer_dims[1], low_res_layer_dims[2], high_res_layer_dims[0], high_res_layer_dims[1], high_res_layer_dims[2]);

    cudaMemcpy(&high_res_flow_x[0], device_high_res_flow_x, num_bytes_to_alloc_high_res, cudaMemcpyDeviceToHost);
    cudaMemcpy(&high_res_flow_y[0], device_high_res_flow_y, num_bytes_to_alloc_high_res, cudaMemcpyDeviceToHost);
    // cpy results back
    cudaFree(device_low_res_flow_x);
    cudaFree(device_low_res_flow_y);
    cudaFree(device_high_res_flow_x);
    cudaFree(device_high_res_flow_y);
}

__host__ void Brox_optical_flow(std::vector<float> &norm_1c_im1_content, std::vector<float> &norm_1c_im2_content, int32_t *image_dimensions, float alpha, float gamma, float scaling_factor,
                                int num_inner_iterations, int num_outer_iterations, int num_solver_iterations, std::vector<float> &computed_flow_x, std::vector<float> &computed_flow_y,
                                std::vector<std::vector<float>> &scale_pyramid_1, std::vector<std::vector<float>> &scale_pyramid_2, std::vector<std::vector<float>> &grad_x_0_pyramid,
                                std::vector<std::vector<float>> &grad_y_0_pyramid, std::vector<std::array<int32_t, 3>> &pyramid_dimensions_per_layer)
{
    // alpha = smoothness term
    // gamma = edge term

    int max_num_scales_according_to_axes = std::numeric_limits<int>::max();
    for(int dim_idx = 0; dim_idx < 2; ++dim_idx)
    {
        max_num_scales_according_to_axes = std::min(max_num_scales_according_to_axes, (int)(-std::log(image_dimensions[dim_idx]) / std::log(scaling_factor)));
    }

    float desired_max_motion = 150.0; // let us assume that we do not want to detect movements that are larger than 50 pixel/voxel

    int64_t desired_num_scalings = -std::log(desired_max_motion) / std::log(scaling_factor) + 1;

    desired_num_scalings = desired_num_scalings;
    desired_num_scalings = num_outer_iterations;
    // desired_num_scalings = 1;

    desired_num_scalings =
        std::min(num_outer_iterations, std::min((int)(std::log(1.0 / image_dimensions[0]) / std::log(scaling_factor)), (int)(std::log(1.0 / image_dimensions[1]) / std::log(scaling_factor))));

    //---------------------------------- allocate pyramid layers for cpu mem
    scale_pyramid_1.resize(desired_num_scalings);
    scale_pyramid_2.resize(desired_num_scalings);
    grad_x_0_pyramid.resize(desired_num_scalings);
    grad_y_0_pyramid.resize(desired_num_scalings);

    int64_t num_elements_per_leaf_layer = norm_1c_im1_content.size();
    scale_pyramid_1[0].resize(num_elements_per_leaf_layer);
    scale_pyramid_2[0].resize(num_elements_per_leaf_layer);
    grad_x_0_pyramid[0].resize(num_elements_per_leaf_layer);
    grad_y_0_pyramid[0].resize(num_elements_per_leaf_layer);

    pyramid_dimensions_per_layer.resize(desired_num_scalings);

    std::array<int32_t, 3> finest_level_dims;
    for(int dim_idx = 0; dim_idx < 3; ++dim_idx)
    {
        finest_level_dims[dim_idx] = image_dimensions[dim_idx];
    }
    pyramid_dimensions_per_layer[0] = finest_level_dims;

    scale_pyramid_1[0] = norm_1c_im1_content;
    scale_pyramid_2[0] = norm_1c_im2_content;

    // presmoothing with sigma 0.8
    launch_spatial_gaussian_kernel(scale_pyramid_1[0], 0.8, image_dimensions);
    cudaDeviceSynchronize();
    launch_spatial_gaussian_kernel(scale_pyramid_2[0], 0.8, image_dimensions);

    cudaError_t err = cudaPeekAtLastError();
    if(err != cudaSuccess)
    {
        printf("AFTER INITIAL GAUSSIAN BLURS: The error is %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    calculate_image_pyramid_sizes(scale_pyramid_1, scale_pyramid_2, pyramid_dimensions_per_layer, scaling_factor, desired_num_scalings);

    create_pyramid_layers(scale_pyramid_1, scale_pyramid_2, pyramid_dimensions_per_layer, scaling_factor);

    cudaDeviceSynchronize();

    std::vector<std::vector<float>> flow_field_pyramid_x(desired_num_scalings);
    std::vector<std::vector<float>> flow_field_pyramid_y(desired_num_scalings);

    for(int scale_idx = desired_num_scalings - 1; scale_idx >= 0; --scale_idx)
    {
        auto const &current_layer_size = pyramid_dimensions_per_layer[scale_idx];

        int64_t num_elements_in_vector = current_layer_size[0] * current_layer_size[1] * current_layer_size[2];

        int32_t current_layer_dims[3] = {current_layer_size[0], current_layer_size[1], current_layer_size[2]};

        flow_field_pyramid_x[scale_idx].resize(num_elements_in_vector, 0.0f);
        flow_field_pyramid_y[scale_idx].resize(num_elements_in_vector, 0.0f);

        grad_x_0_pyramid[scale_idx].resize(num_elements_in_vector, 0.0f);
        grad_y_0_pyramid[scale_idx].resize(num_elements_in_vector, 0.0f);

        // upsample vector fields from last pyramid step

        if(scale_idx != desired_num_scalings - 1)
        {
            auto const &prev_layer_size = pyramid_dimensions_per_layer[scale_idx + 1];
            int32_t prev_layer_dims[3] = {prev_layer_size[0], prev_layer_size[1], prev_layer_size[2]};

            launch_vector_field_upsampling_kernel(flow_field_pyramid_x[scale_idx + 1], flow_field_pyramid_y[scale_idx + 1], flow_field_pyramid_x[scale_idx], flow_field_pyramid_y[scale_idx],
                                                  prev_layer_dims, current_layer_dims);
        }

        cudaDeviceSynchronize();

        launch_brox_outer_loop_kernel_tasks(scale_pyramid_1[scale_idx], scale_pyramid_2[scale_idx], grad_x_0_pyramid[scale_idx], grad_y_0_pyramid[scale_idx], flow_field_pyramid_x[scale_idx],
                                            flow_field_pyramid_y[scale_idx], alpha, gamma, scaling_factor, num_inner_iterations, num_outer_iterations, num_solver_iterations, current_layer_dims);

        cudaDeviceSynchronize();

        cudaError_t err = cudaPeekAtLastError();
        if(err != cudaSuccess)
        {
            printf("@ Pos END: The error is %s\n", cudaGetErrorString(err));
        }
    }

    computed_flow_x = flow_field_pyramid_x[0];
    computed_flow_y = flow_field_pyramid_y[0];
}