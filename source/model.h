//
// Created by xaf on 19.05.18.
//

#ifndef RGBD_RECON_MODEL_H
#define RGBD_RECON_MODEL_H

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

#include <LocalKinectArray.h>
#include <calibration/CalibVolumes.hpp>
#include <calibration/KinectCalibrationFile.h>
#include <calibration/calibration_files.hpp>
#include <navigation/CameraNavigator.h>

#include <reconstruction/recon_pc.hpp>
#include <reconstruction/recon_points.hpp>
#include <reconstruction/reconstruction.hpp>
#include <rendering/timer_database.hpp>

#include <iostream>
#include <memory>
#include <reconstruction/recon_calibs.hpp>
#include <reconstruction/recon_integration.hpp>

/**
 * @class model is responsible for all states and data streams, except window (context) management
 */
class model
{
  public:
    model();
    ~model();

    static model &get_instance()
    {
        static model instance;
        return instance;
    }

    model(model const &) = delete;
    void operator=(model const &) = delete;

    struct IO
    {
        std::string _record_name = "";
        std::vector<std::pair<int, int>> _gui_texture_settings{};

        float _clear_color[4] = {1., 1., 1., 1.};
        bool _splitscreen_comparison = false;
        unsigned _stereo_mode = 0;
        float _screenWidthReal = 1.28;
        float _screenHeightReal = 0.72;
        unsigned _screenWidth = 1280;
        unsigned _screenHeight = 720;
        unsigned _windowWidth = 1280;
        unsigned _windowHeight = 720;
        unsigned _left_pos_x = 0;
        unsigned _left_pos_y = 0;
        unsigned _right_pos_x = 0;
        unsigned _right_pos_y = 0;

        float _aspect = (float)(_screenWidth * 1.0 / _screenHeight);
        bool _play = false;
        bool _draw_frustums = false;
        bool _draw_grid = true;
        bool _animate = false;
        int _recon_mode = 0;
        bool _bilateral = true;
        bool _draw_calibvis = false;
        bool _draw_textures = false;
        int _texture_type = 0;
        int _num_texture = 0;
        bool _processed = false;
        bool _refine = false;
        bool _colorfill = false;
        bool _bricking = true;
        bool _skip_space = true;
        bool _draw_bricks = false;
        bool _watch_errors = true;
        int _num_kinect = 1;
        float _voxel_size = 0.021875f; // 256: 0.00546875f; // 128: 0.0109375f; // 64: 0.021875f
        float _ed_cell_size = _voxel_size * 3;
        float _brick_size = _voxel_size * 9;
        float _tsdf_limit = 0.03f;
        float _zoom = 2.5f;
        double _time_prev = 0.0f;

        int _min_voxels = 0;

        bool _loaded_conf = false;
        unsigned _time_limit = 1;
        std::string _conf_file{};
    };

    static IO &get_io() { return get_instance().io; }
    void cmd(CMDParser &p);
    void init(std::string &file_name, std::string &file_name_flow, gloost::Point3 &bbox_min, gloost::Point3 &bbox_max, std::vector<std::string> &calib_filenames, std::string &resource_path);
    void init_stereo_camera();
    void init_fbr(const char *client_socket);
    void init_config(std::vector<std::string> const &args);
    void load_config(std::string const &file_name);
    void reload_reconstructions();
    void next_reconstruction();
    void prev_reconstruction();
    void update_framebuffer_size(unsigned int width, unsigned int height);

    const std::shared_ptr<gloost::BoundingBox> &get_bbox() const;
    const std::shared_ptr<gloost::PerspectiveCamera> &get_camera() const;
    const std::shared_ptr<gloost::StereoCamera> &get_stereo_camera() const;
    const std::shared_ptr<pmd::CameraNavigator> &get_navi() const;
    const std::shared_ptr<sys::FeedbackReceiver> &get_fbr() const;
    const std::shared_ptr<kinect::LocalKinectArray> &get_nka() const;
    const std::shared_ptr<kinect::CalibVolumes> &get_cv() const;
    const std::shared_ptr<kinect::CalibrationFiles> &get_calib_files() const;
    const std::vector<std::shared_ptr<kinect::Reconstruction>> &get_recons() const;
    const std::shared_ptr<kinect::ReconCalibs> &get_calibvis() const;
    const std::shared_ptr<kinect::ReconPerformanceCapture> &get_recon_pc() const;

  private:
    IO io;

    std::shared_ptr<gloost::BoundingBox> _bbox;
    std::shared_ptr<gloost::PerspectiveCamera> _camera;
    std::shared_ptr<gloost::StereoCamera> _stereo_camera;
    std::shared_ptr<pmd::CameraNavigator> _navi;

    std::shared_ptr<sys::FeedbackReceiver> _fbr;
    std::shared_ptr<kinect::LocalKinectArray> _nka;
    std::shared_ptr<kinect::CalibVolumes> _cv;
    std::shared_ptr<kinect::CalibrationFiles> _calib_files;

    std::shared_ptr<kinect::ReconPoints> _recon_points;
    std::shared_ptr<kinect::ReconPerformanceCapture> _recon_pc;
    // std::shared_ptr<kinect::ReconIntegration> _recon_integration;
    std::vector<std::shared_ptr<kinect::Reconstruction>> _recons;

    std::shared_ptr<kinect::ReconCalibs> _calibvis;
};

#endif // RGBD_RECON_MODEL_H
