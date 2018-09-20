#include <glbinding/Function.h>
#include <glbinding/Meta.h>
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
#include <imgui_impl_glfw_gl3.h>
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

#include "model.h"
#include "renderer.h"

GLFWwindow *_window = nullptr;

model *_model = nullptr;
renderer *_renderer = nullptr;
model::IO *_io = nullptr;

void init(std::vector<std::string> const &args)
{
    std::string ext{args[0].substr(args[0].find_last_of(".") + 1)};
    std::string file_name{};
    if("ks" == ext)
    {
        file_name = args[0];
    }
    else
    {
        throw std::invalid_argument{"No .ks file specified"};
    }

    // read ks file
    std::vector<std::string> calib_filenames;
    gloost::Point3 bbox_min{-1.0f, 0.0f, -1.0f};
    gloost::Point3 bbox_max{1.0f, 2.2f, 1.0f};

    std::string resource_path = file_name.substr(0, file_name.find_last_of("/\\")) + '/';
    std::cout << resource_path << std::endl;
    std::ifstream in(file_name);
    std::string token;
    std::string record_name;
    while(in >> token)
    {
        if(token == "kinect")
        {
            in >> token;
            // detect absolute path
            if(token[0] == '/' || token[1] == ':')
            {
                calib_filenames.push_back(token);
            }
            else
            {
                calib_filenames.push_back(resource_path + token);
            }
        }
        else if(token == "bbx")
        {
            in >> bbox_min[0];
            in >> bbox_min[1];
            in >> bbox_min[2];
            in >> bbox_max[0];
            in >> bbox_max[1];
            in >> bbox_max[2];
        }
    }
    in.close();

    std::string flow_name(_io->_record_name);
    flow_name.replace(flow_name.find("stream"), 6, "flow");

    _model->init(_io->_record_name, flow_name, bbox_min, bbox_max, calib_filenames, resource_path);
}

void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if(key == GLFW_KEY_P && action == GLFW_PRESS)
    {
        _io->_play = true;
        _model->get_recon_pc()->pause(false);
    }

    if(action != GLFW_RELEASE)
        return;

    switch(key)
    {
    case GLFW_KEY_ESCAPE:
    case GLFW_KEY_Q:
        glfwSetWindowShouldClose(_window, 1);
        break;
    case GLFW_KEY_F:
        _io->_draw_frustums = !_io->_draw_frustums;
        break;
    case GLFW_KEY_B:
        _io->_bilateral = !_io->_bilateral;
        _model->get_nka()->filterTextures(_io->_bilateral);
        break;
    case GLFW_KEY_D:
        _io->_processed = !_io->_processed;
        _model->get_nka()->useProcessedDepths(_io->_processed);
        break;
    case GLFW_KEY_N:
        _io->_refine = !_io->_refine;
        _model->get_nka()->refineBoundary(_io->_refine);
        break;
    case GLFW_KEY_J:
        _io->_splitscreen_comparison = !_io->_splitscreen_comparison;

        if(_io->_splitscreen_comparison)
        {
            _io->_aspect = (float)(_io->_screenWidth * 0.5 / _io->_screenHeight);

            _model->get_camera()->setAspect(_io->_aspect);

            for(auto &recon : _model->get_recons())
            {
                recon->resize((unsigned int)glm::round(_io->_screenWidth * 0.5), _io->_screenHeight);
            }
        }
        else
        {
            _io->_aspect = (float)(_io->_screenWidth * 1. / _io->_screenHeight);

            _model->get_camera()->setAspect(_io->_aspect);

            for(auto &recon : _model->get_recons())
            {
                recon->resize((unsigned int)glm::round(_io->_screenWidth), _io->_screenHeight);
            }
        }

        break;
    case GLFW_KEY_G:
        _io->_draw_grid = !_io->_draw_grid;
        break;
    case GLFW_KEY_A:
        _io->_animate = !_io->_animate;
        break;
    case GLFW_KEY_Y:
        _io->_num_texture = (_io->_num_texture + 1) % _model->get_calib_files()->num();
        break;
    case GLFW_KEY_U:
        _io->_texture_type = (_io->_texture_type + 1) % 7;
        break;
    case GLFW_KEY_T:
        _io->_draw_textures = !_io->_draw_textures;
        break;
    case GLFW_KEY_S:
        _model->reload_reconstructions();
        _renderer->process_textures();
        break;
    case GLFW_KEY_1:
        _renderer->next_shading_mode();
        break;
    case GLFW_KEY_V:
        _io->_draw_calibvis = !_io->_draw_calibvis;
        break;
    case GLFW_KEY_C:
        _io->_num_kinect = (_io->_num_kinect + 1) % _model->get_calib_files()->num();
        _model->get_calibvis()->setActiveKinect(_io->_num_kinect);
        break;
    case GLFW_KEY_PAGE_UP:
        _model->next_reconstruction();
        break;
    case GLFW_KEY_PAGE_DOWN:
        _model->prev_reconstruction();
        break;
    default:
        break;
    }
}

void mouse_callback(GLFWwindow *window, double xpos, double ypos) { _model->get_navi()->motion(xpos, ypos); }
void click_callback(GLFWwindow *window, int button, int action, int mods)
{
    if(ImGui::GetIO().WantCaptureMouse)
        return;
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    int mouse_h = (int)xpos;
    int mouse_v = (int)ypos;

    _model->get_navi()->mouse(button, action, mouse_h, mouse_v);
}
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset)
{
    _io->_zoom = std::min(_io->_zoom + (float)yoffset * 0.05f, 2.5f);
    _model->get_navi()->setZoom(_io->_zoom);

    ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
}
static void error_callback(int error, const char *description) { fprintf(stderr, "Error %d: %s\n", error, description); }
void framebuffer_size_callback(GLFWwindow *window, int width, int height) { _model->update_framebuffer_size(width, height); }
void quit(int status)
{
    if(_io->_loaded_conf)
    {
        time_t t = time(0); // get time now
        struct tm *now = localtime(&t);
        std::stringstream file_name;

        file_name << _io->_conf_file.substr(0, _io->_conf_file.length() - 5) << "," << (now->tm_year + 1900) << '-' << (now->tm_mon + 1) << '-' << now->tm_mday << ',' << now->tm_hour << '-'
                  << now->tm_min << ".csv";
        TimerDatabase::instance().writeMean(file_name.str());
        TimerDatabase::instance().writeMin(file_name.str());
        TimerDatabase::instance().writeMax(file_name.str());
    }

    // free globjects
    globjects::detachAllObjects();

    ImGui_ImplGlfwGL3_Shutdown();
    // free glfw resources
    glfwDestroyWindow(_window);
    glfwTerminate();

    std::exit(status);
}

int main(int argc, char *argv[])
{
    _model = &model::get_instance();
    _renderer = &renderer::get_instance();
    _io = &model::get_io();

    CMDParser p("kinect_surface ...");

    p.addOpt("s", 2, "screensize", "set screen size in meter");
    p.addOpt("d", 2, "displaysize", "set display size in pixel");

    p.addOpt("w", 2, "windowsize", "set window size in pixel for stereomode side-by-side");
    p.addOpt("l", 2, "leftpos", "set the position of the left viewport (upper left corner) in pixel for stereomode side-by-side");
    p.addOpt("r", 2, "rightpos", "set the position of the right viewport (upper left corner) in pixel for stereomode side-by-side");

    p.addOpt("m", 1, "stereomode", "set stereo mode 0: none, 1: anaglyph, 2: side-by-side (default: 0)");
    p.addOpt("c", 4, "clearcolor", "set clear color (default: 0.0 0.0 0.0 0.0)");

    p.addOpt("f", 1, "stream", "set stream file to play from");

    p.init(argc, argv);

    _model->cmd(p);

    // Setup window
    glfwSetErrorCallback(error_callback);
    if(!glfwInit())
    {
        std::exit(EXIT_FAILURE);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
    glfwWindowHint(GLFW_CONTEXT_ROBUSTNESS, GLFW_LOSE_CONTEXT_ON_RESET);
#if __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    switch(_io->_stereo_mode)
    {
    case 0:
    case 1:
    {
        _window = glfwCreateWindow(_io->_screenWidth, _io->_screenHeight, "Kinect Reconstruction", NULL, NULL);
    }
    break;
    case 2:
    {
        _window = glfwCreateWindow(_io->_windowWidth, _io->_windowHeight, "Kinect Reconstruction", NULL, NULL);
    }
    break;
    default:
        throw std::runtime_error("Unknown stereo mode: " + std::to_string(_io->_stereo_mode));
    }

    if(!_window)
    {
        glfwTerminate();
        std::exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(_window);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();

    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui_ImplGlfwGL3_Init(_window, true);
    ImGui::StyleColorsDark();

    // disable vsync
    glfwSwapInterval(0);

    glfwSetKeyCallback(_window, key_callback);
    glfwSetCursorPosCallback(_window, mouse_callback);
    glfwSetMouseButtonCallback(_window, click_callback);
    glfwSetScrollCallback(_window, scroll_callback);
    if(0 == _io->_stereo_mode)
    {
        glfwSetFramebufferSizeCallback(_window, framebuffer_size_callback);
    }
    // allow unlimited mouse movement

    // Initialize globjects (internally initializes glbinding, and registers the current context)
    globjects::init();

    renderer::watch_gl_errors(_io->_watch_errors);

    // set some gl states
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    // load and intialize objects
    init(p.getArgs());

    // start of rendering
    auto time_start = std::chrono::high_resolution_clock::now();

    while(!glfwWindowShouldClose(_window))
    {
        glfwPollEvents();

        ImGui_ImplGlfwGL3_NewFrame();

        _renderer->update_gui();
        _renderer->draw3d();

        if(2 != _io->_stereo_mode)
        {
            ImGui::Render();
            ImGui_ImplGlfwGL3_RenderDrawData(ImGui::GetDrawData());
        }

        glfwSwapBuffers(_window);

        // keep track fo time if config was loaded
        if(_io->_loaded_conf)
        {
            long time_in_s = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - time_start).count();
            if(time_in_s >= _io->_time_limit)
            {
                quit(EXIT_SUCCESS);
            }
        }
    }

    quit(EXIT_SUCCESS);
}
