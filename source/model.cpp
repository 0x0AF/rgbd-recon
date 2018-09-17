//
// Created by xaf on 19.05.18.
//

#include "model.h"
#include "renderer.h"

void model::cmd(CMDParser &p)
{
    if(p.isOptSet("s"))
    {
        io._screenWidthReal = p.getOptsFloat("s")[0];
        io._screenHeightReal = p.getOptsFloat("s")[1];
    }
    if(p.isOptSet("d"))
    {
        io._screenWidth = (unsigned)p.getOptsInt("d")[0];
        io._screenHeight = (unsigned)p.getOptsInt("d")[1];
    }
    if(p.isOptSet("w"))
    {
        io._windowWidth = (unsigned)p.getOptsInt("w")[0];
        io._windowHeight = (unsigned)p.getOptsInt("w")[1];
    }

    if(p.isOptSet("l"))
    {
        io._left_pos_x = (unsigned)p.getOptsInt("l")[0];
        io._left_pos_y = (unsigned)p.getOptsInt("l")[1];
    }
    if(p.isOptSet("r"))
    {
        io._right_pos_x = (unsigned)p.getOptsInt("r")[0];
        io._right_pos_y = (unsigned)p.getOptsInt("r")[1];
    }

    if(p.isOptSet("m"))
    {
        io._stereo_mode = (unsigned)p.getOptsInt("m")[0];
    }

    if(p.isOptSet("c"))
    {
        io._clear_color[0] = p.getOptsFloat("c")[0];
        io._clear_color[1] = p.getOptsFloat("c")[1];
        io._clear_color[2] = p.getOptsFloat("c")[2];
        io._clear_color[3] = p.getOptsFloat("c")[3];
    }

    if(p.isOptSet("f"))
    {
        io._record_name = p.getOptsString("f")[0].c_str();
    }

    if((1 == io._stereo_mode) || (2 == io._stereo_mode))
    {
        init_stereo_camera();
    }

    // load global variables
    init_config(p.getArgs());
}
void model::init(std::string &file_name, std::string &file_name_flow,gloost::Point3 &bbox_min, gloost::Point3 &bbox_max, std::vector<std::string> &calib_filenames, std::string &resource_path)
{
    _bbox = std::make_shared<gloost::BoundingBox>();
    _camera = std::make_shared<gloost::PerspectiveCamera>(50.0, io._aspect, 0.1, 200.0);
    _navi = std::make_shared<pmd::CameraNavigator>(0.5f);

    _bbox->setPMin(bbox_min);
    _bbox->setPMax(bbox_max);

    _calib_files = std::make_shared<kinect::CalibrationFiles>(calib_filenames);
    _cv = std::make_shared<kinect::CalibVolumes>(calib_filenames, *_bbox);
    _nka = std::make_shared<kinect::LocalKinectArray>(file_name, file_name_flow, _calib_files.get(), _cv.get(), true);

    // binds to unit 1 to 18
    _nka->setStartTextureUnit(1);
    // bind calibration volumes from 12 - 24
    _cv->setStartTextureUnit(12);
    _cv->loadInverseCalibs(resource_path);
    _cv->setStartTextureUnitInv(20);

    _recon_points = std::make_shared<kinect::ReconPoints>(*_calib_files, _cv.get(), *_bbox);
    _recon_pc = std::make_shared<kinect::ReconPerformanceCapture>(*_nka, *_calib_files, _cv.get(), *_bbox, io._tsdf_limit, io._voxel_size, io._ed_cell_size);
    //_recon_integration = std::make_shared<kinect::ReconIntegration>(*_calib_files, _cv.get(), *_bbox, io._tsdf_limit, io._voxel_size);
    _recons.emplace_back(_recon_points);
    _recons.emplace_back(_recon_pc);
    //_recons.emplace_back(_recon_integration);

    _calibvis = std::make_shared<kinect::ReconCalibs>(*_calib_files, _cv.get(), *_bbox);

    renderer::get_instance().init();

    _nka->useProcessedDepths(io._processed);
    _nka->filterTextures(io._bilateral);
    _nka->refineBoundary(io._refine);
    _recon_pc->setTsdfLimit(io._tsdf_limit);
    //    _recon_pc->setVoxelSize(io._voxel_size);
    //    _recon_pc->setBrickSize(io._brick_size);
    //    _recon_integration->setTsdfLimit(io._tsdf_limit);
    //    _recon_integration->setVoxelSize(io._voxel_size);
    //    _recon_integration->setBrickSize(io._brick_size);
    //    _recon_integration->setColorFilling(io._colorfill);
    //    _recon_integration->setSpaceSkip(io._skip_space);
    //    _recon_integration->setDrawBricks(io._draw_bricks);
    //    _recon_integration->setUseBricks(io._bricking);

    io._aspect = (float)(io._screenWidth * 1.0 / io._screenHeight);

    _camera->setAspect(io._aspect);
    _navi->resize(io._screenWidth, io._screenHeight);
    _navi->setZoom(io._zoom);

    for(auto &recon : _recons)
    {
        recon->resize(io._screenWidth, io._screenHeight);
    }
}
void model::init_stereo_camera()
{
    gloost::Matrix eye_matrix;
    eye_matrix.setIdentity();
    eye_matrix.setTranslate(0.0, 0.0, 1.0);
    gloost::Matrix screen_matrix;
    screen_matrix.setIdentity();

    _stereo_camera = std::make_shared<gloost::StereoCamera>(eye_matrix, 0.2, 20.0, 0.064 /*eyesep*/, screen_matrix, io._screenWidthReal, io._screenHeightReal);
}
void model::init_fbr(const char *client_socket)
{
    sys::feedback initial_fb;
    initial_fb.cyclops_mat[3][0] = 0.0;
    initial_fb.cyclops_mat[3][1] = 0.0;
    initial_fb.cyclops_mat[3][2] = 1.0;
    initial_fb.recon_mode = 1;
    _fbr = std::make_shared<sys::FeedbackReceiver>(initial_fb, client_socket);
}
void model::load_config(std::string const &file_name)
{
    configurator().read(file_name);
    configurator().print();
    io._recon_mode = configurator().getUint("recon_mode");
    io._screenWidth = configurator().getUint("screenWidth");
    io._screenHeight = configurator().getUint("screenHeight");
    io._play = configurator().getBool("play");
    io._draw_grid = configurator().getBool("draw_grid");
    io._animate = configurator().getBool("animate");
    io._bilateral = configurator().getBool("bilateral");
    io._processed = configurator().getBool("processed");
    io._refine = configurator().getBool("refine");
    io._colorfill = configurator().getBool("colorfill");
    io._bricking = configurator().getBool("bricking");
    io._skip_space = configurator().getBool("skip_space");
    io._watch_errors = configurator().getBool("watch_errors");
    io._voxel_size = configurator().getFloat("voxel_size");
    io._brick_size = configurator().getFloat("brick_size");
    io._tsdf_limit = configurator().getFloat("tsdf_limit");
    io._zoom = configurator().getFloat("zoom");
    io._time_limit = configurator().getUint("time_limit");
    io._loaded_conf = true;
    io._conf_file = file_name;
}
void model::init_config(std::vector<std::string> const &args)
{
    if(args.size() > 1)
    {
        std::string ext = args[1].substr(args[1].find_last_of(".") + 1);
        if("conf" == ext)
        {
            load_config(args[1]);
        }
        else
        {
            throw std::invalid_argument{"No .conf file specified"};
        }
    }
}
model::model() {}
model::~model()
{
    _bbox.reset();
    _camera.reset();
    _stereo_camera.reset();
    _navi.reset();

    _fbr.reset();
    _nka.reset();
    _cv.reset();
    _calib_files.reset();

    _recon_points.reset();
    _recon_pc.reset();
    //_recon_integration.reset();
}
const std::shared_ptr<gloost::BoundingBox> &model::get_bbox() const { return _bbox; }
const std::shared_ptr<gloost::PerspectiveCamera> &model::get_camera() const { return _camera; }
const std::shared_ptr<gloost::StereoCamera> &model::get_stereo_camera() const { return _stereo_camera; }
const std::shared_ptr<pmd::CameraNavigator> &model::get_navi() const { return _navi; }
const std::shared_ptr<sys::FeedbackReceiver> &model::get_fbr() const { return _fbr; }
const std::shared_ptr<kinect::LocalKinectArray> &model::get_nka() const { return _nka; }
const std::shared_ptr<kinect::CalibVolumes> &model::get_cv() const { return _cv; }
const std::shared_ptr<kinect::CalibrationFiles> &model::get_calib_files() const { return _calib_files; }
void model::reload_reconstructions()
{
    for(auto &recon : _recons)
    {
        recon->reload();
    }
    globjects::File::reloadAll();
}
void model::next_reconstruction()
{
    io._recon_mode = (int)((io._recon_mode + 1) % _recons.size());
    reload_reconstructions();
}
void model::prev_reconstruction()
{
    io._recon_mode = (int)((io._recon_mode + _recons.size() - 1) % _recons.size());
    reload_reconstructions();
}
void model::update_framebuffer_size(unsigned int width, unsigned int height)
{
    io._screenWidth = width;
    io._screenHeight = height;
    io._aspect = (float)(io._screenWidth * 1.0 / io._screenHeight);
    _camera->setAspect(io._aspect);

    for(auto &recon : _recons)
    {
        recon->resize(width, height);
    }

    _navi->resize(width, height);
}
const std::vector<std::shared_ptr<kinect::Reconstruction>> &model::get_recons() const { return _recons; }
const std::shared_ptr<kinect::ReconCalibs> &model::get_calibvis() const { return _calibvis; }
const std::shared_ptr<kinect::ReconPerformanceCapture> &model::get_recon_pc() const { return _recon_pc; }
