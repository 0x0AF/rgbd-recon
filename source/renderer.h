//
// Created by xaf on 19.05.18.
//

#ifndef RGBD_RECON_RENDERER_H
#define RGBD_RECON_RENDERER_H

#include <glbinding/Function.h>
#include <glbinding/Meta.h>
#include <glbinding/callbacks.h>
#include <glbinding/gl/gl.h>

using namespace gl;

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <globjects/Buffer.h>
#include <globjects/Query.h>
#include <globjects/base/File.h>
#include <globjects/globjects.h>

#include <cmath>
#include <cstdlib>
#include <imgui.h>
#include <iostream>
#include <memory>
#include <tuple>

#include <io/CMDParser.h>
#include <io/FeedbackReceiver.h>
#include <io/configurator.hpp>
#include <rendering/texture_blitter.hpp>

#include <BoundingBox.h>
#include <PerspectiveCamera.h>
#include <Point3.h>
#include <StereoCamera.h>

#include <NetKinectArray.h>
#include <calibration/CalibVolumes.hpp>
#include <calibration/KinectCalibrationFile.h>
#include <calibration/calibration_files.hpp>
#include <navigation/CameraNavigator.h>

#include <reconstruction/recon_pc.hpp>
#include <reconstruction/recon_points.hpp>
#include <reconstruction/reconstruction.hpp>
#include <rendering/timer_database.hpp>

#include "model.h"
/**
 * @class renderer implements all rendering routines
 */
class renderer
{
  public:
    renderer();
    ~renderer();
    static renderer &get_instance()
    {
        static renderer instance;
        return instance;
    }

    renderer(renderer const &) = delete;
    void operator=(renderer const &) = delete;

    void init();
    void update_gui();
    void update_model_matrix(bool load_ident = true);
    void process_textures();
    void draw3d();
    void next_shading_mode();

  private:
    model *_model;
    model::IO *_io;

    globjects::Buffer *_buffer_shading;
    struct shading_data_t
    {
        int mode = 0;
    } _shading_buffer_data;
};

#endif // RGBD_RECON_RENDERER_H
