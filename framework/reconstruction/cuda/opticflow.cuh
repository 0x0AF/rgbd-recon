#include <reconstruction/cuda/OPTICFLOW/brox_optical_flow.cu>
#include <reconstruction/cuda/OPTICFLOW/cuda_calls.cu>
#include <reconstruction/cuda/glm.cuh>
#include <reconstruction/cuda/resources.cuh>

extern "C" double evaluate_dense_correspondence_field()
{
    TimerGPU timer(0);

    int im_width = _host_res.measures.depth_res.x;
    int im_height = _host_res.measures.depth_res.y;
    int32_t im_dimensions[3] = {im_width, im_height, 1};

    std::vector<float> norm_im_0(im_width * im_height, 0.0);
    std::vector<float> norm_im_1(im_width * im_height, 0.0);

    std::vector<float> computed_flow_x(im_width * im_height, 0.0);
    std::vector<float> computed_flow_y(im_width * im_height, 0.0);

    float smoothness_term = 0.197;
    float edge_term = 50.0;

    std::vector<std::array<int32_t, 3>> pyramid_sizes;
    std::vector<std::vector<float>> scale_pyramid_1;
    std::vector<std::vector<float>> scale_pyramid_2;
    std::vector<std::vector<float>> grad_x_0_pyramid;
    std::vector<std::vector<float>> grad_y_0_pyramid;

    for(unsigned int i = 0; i < 4; i++)
    {
        checkCudaErrors(
            cudaMemcpy2D(&norm_im_0[0], im_width * sizeof(float), &_dev_res.kinect_intens[i][0], _dev_res.pitch_kinect_intens, im_width * sizeof(float), im_height, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(
            cudaMemcpy2D(&norm_im_1[0], im_width * sizeof(float), &_dev_res.kinect_intens_prev[i][0], _dev_res.pitch_kinect_intens, im_width * sizeof(float), im_height, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaDeviceSynchronize());

        Brox_optical_flow(norm_im_0, norm_im_1, im_dimensions, smoothness_term, edge_term, _host_res.configuration.opticflow_scaling_factor, _host_res.configuration.opticflow_num_inner_iterations,
                          _host_res.configuration.opticflow_num_outer_iterations, _host_res.configuration.opticflow_num_solver_iterations, computed_flow_x, computed_flow_y, scale_pyramid_1,
                          scale_pyramid_2, grad_x_0_pyramid, grad_y_0_pyramid, pyramid_sizes);
        checkCudaErrors(cudaThreadSynchronize());

        float2 *optical_flow = (float2 *)malloc(im_width * im_height * sizeof(float2));

        for(int k = 0; k < im_width * im_height; k++)
        {
            optical_flow[k].x = computed_flow_x[k];
            optical_flow[k].y = computed_flow_y[k];

            // printf("\n(x,y): (%f,%f)\n", optical_flow[i].x, optical_flow[i].y);
        }

        checkCudaErrors(
            cudaMemcpy2D(&_dev_res.optical_flow[i][0], _dev_res.pitch_optical_flow, &optical_flow[0], im_width * sizeof(float2), im_width * sizeof(float2), im_height, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaDeviceSynchronize());

        free(optical_flow);
    }

#ifdef PIPELINE_DEBUG_TEXTURE_SILHOUETTES
    checkCudaErrors(cudaGraphicsMapResources(1, &_cgr.pbo_opticflow_debug, 0));

    size_t dummy_size;

    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&_dev_res.mapped_pbo_opticflow_debug, &dummy_size, _cgr.pbo_opticflow_debug));
    checkCudaErrors(cudaDeviceSynchronize());

    size_t depth_size = _host_res.measures.depth_res.x * _host_res.measures.depth_res.y;

    for(unsigned int i = 0; i < 4; i++)
    {
        checkCudaErrors(cudaMemcpy2D(&_dev_res.mapped_pbo_opticflow_debug[i * depth_size], _host_res.measures.depth_res.x * sizeof(float2), &_dev_res.optical_flow[i][0], _dev_res.pitch_optical_flow,
                                     _host_res.measures.depth_res.x * sizeof(float2), _host_res.measures.depth_res.y, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    checkCudaErrors(cudaGraphicsUnmapResources(1, &_cgr.pbo_opticflow_debug));
#endif

    return timer.read();
}