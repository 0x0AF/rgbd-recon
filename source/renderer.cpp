//
// Created by xaf on 19.05.18.
//

#include "renderer.h"
#include <imgui_impl_glfw_gl3.h>

std::ostream &operator<<(std::ostream &os, const glm::mat4 &m)
{
    os << "mat4[" << std::fixed << std::endl;
    os << "       (" << m[0][0] << ", " << m[1][0] << ", " << m[2][0] << ", " << m[3][0] << ")," << std::endl;
    os << "       (" << m[0][1] << ", " << m[1][1] << ", " << m[2][1] << ", " << m[3][1] << ")," << std::endl;
    os << "       (" << m[0][2] << ", " << m[1][2] << ", " << m[2][2] << ", " << m[3][2] << ")," << std::endl;
    os << "       (" << m[0][3] << ", " << m[1][3] << ", " << m[2][3] << ", " << m[3][3] << ") ]" << std::endl;

    return os;
}

gloost::Matrix glm2gloost(const glm::mat4 m)
{
    gloost::Matrix tmp;
    tmp[0] = m[0][0];
    tmp[1] = m[0][1];
    tmp[2] = m[0][2];
    tmp[3] = m[0][3];

    tmp[4] = m[1][0];
    tmp[5] = m[1][1];
    tmp[6] = m[1][2];
    tmp[7] = m[1][3];

    tmp[8] = m[2][0];
    tmp[9] = m[2][1];
    tmp[10] = m[2][2];
    tmp[11] = m[2][3];

    tmp[12] = m[3][0];
    tmp[13] = m[3][1];
    tmp[14] = m[3][2];
    tmp[15] = m[3][3];

    return tmp;
}

renderer::renderer()
{
    _model = &model::get_instance();
    _io = &model::get_io();
}
renderer::~renderer() { _buffer_shading->destroy(); }
void renderer::init()
{
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);

    _buffer_shading = new globjects::Buffer();
    _buffer_shading->ref();
    _buffer_shading->setData(sizeof(shading_data_t), &_shading_buffer_data, GL_STATIC_DRAW);
    _buffer_shading->bindBase(GL_UNIFORM_BUFFER, 1);
}
void renderer::update_gui()
{
    {
        ImGui::SetNextWindowSize(ImVec2(200, 100), ImGuiSetCond_FirstUseEver);
        ImGui::Begin("Settings");
        if(ImGui::Button("Show textures"))
        {
            _io->_gui_texture_settings.emplace_back(0, 0);
        }
        if(ImGui::Checkbox("Watch OpenGL errors", &_io->_watch_errors))
            ;
        ImGui::Text("%.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        if(ImGui::CollapsingHeader("Reconstruction Mode", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::RadioButton("Points", &_io->_recon_mode, 0);
            ImGui::RadioButton("Marching Cubes", &_io->_recon_mode, 1);
        }
        if(ImGui::CollapsingHeader("Visualisation"))
        {
            int prev = _shading_buffer_data.mode;
            ImGui::RadioButton("Textured", &_shading_buffer_data.mode, 0);
            ImGui::RadioButton("Shaded", &_shading_buffer_data.mode, 1);
            ImGui::RadioButton("Normals", &_shading_buffer_data.mode, 2);
            ImGui::RadioButton("Blending", &_shading_buffer_data.mode, 3);
            if(prev != _shading_buffer_data.mode)
            {
                _buffer_shading->setSubData(0, sizeof(shading_data_t), &_shading_buffer_data);
            }
        }
        if(ImGui::CollapsingHeader("Settings"))
        {
            switch(_io->_recon_mode)
            {
            case 0: // points
            {
            }
            break;
            case 1: // performance capture
            {
                std::shared_ptr<kinect::ReconPerformanceCapture> recon_pc = std::dynamic_pointer_cast<kinect::ReconPerformanceCapture>(_model->get_recons().at(1));

                if(ImGui::SliderFloat("TSDF Limit", &_io->_tsdf_limit, 0.001f, 0.03f, "%.5f", 2.71828381f))
                {
                    recon_pc->setTsdfLimit(_io->_tsdf_limit);
                }
                if(ImGui::SliderFloat("Voxel Size", &_io->_voxel_size, 0.01f, 0.1f, "%.5f", 2.71828381f))
                {
                    recon_pc->setVoxelSize(_io->_voxel_size);
                }
                if(ImGui::SliderFloat("Marching Cubes ISO", &_io->_mc_iso, 0.00001f, 0.01f, "%.5f", 2.71828381f))
                {
                    recon_pc->setIso(_io->_mc_iso);
                }
            }
            break;
            default:
                throw std::runtime_error("Unknown reconstruction mode: " + std::to_string(_io->_recon_mode));
            }
        }
        if(ImGui::CollapsingHeader("Processing Performance"))
        {
            ImGui::SameLine();
            ImGui::Text("   %.3f ms", TimerDatabase::instance().duration("1preprocess") / 1000000.0f);
            ImGui::Text("   Reconstruction");
            ImGui::SameLine();
            ImGui::Text("   %.3f ms", TimerDatabase::instance().duration("draw") / 1000000.0f);
        }
        ImGui::End();
    }
    for(unsigned i = 0; i < _io->_gui_texture_settings.size(); ++i)
    {
        auto &setting = _io->_gui_texture_settings[i];
        ImGui::SetNextWindowSize(ImVec2(100, 100), ImGuiSetCond_FirstUseEver);
        bool show_tex = true;
        if(!ImGui::Begin(std::string{"Textures " + std::to_string(i)}.c_str(), &show_tex))
        {
            ImGui::End();
        }
        else
        {
            if(ImGui::CollapsingHeader("Kinect", ImGuiTreeNodeFlags_DefaultOpen))
            {
                static std::vector<const char *> listbox_items = {"1", "2", "3", "4"};
                ImGui::ListBox("Number", &setting.second, listbox_items.data(), listbox_items.size(), listbox_items.size());
            }
            if(ImGui::CollapsingHeader("Texture Type", ImGuiTreeNodeFlags_DefaultOpen))
            {
                static std::vector<const char *> listbox_items = {"Color", "Depth", "Quality", "Normals", "Silhouette", "Orig Depth", "LAB colors"};
                ImGui::ListBox("Type", &setting.first, listbox_items.data(), listbox_items.size(), listbox_items.size());
            }

            TexInfo test = {(uint16_t)(_model->get_nka()->getStartTextureUnit() + setting.first), (uint16_t)(-setting.second - 1)};
            ImTextureID cont;
            std::memcpy(&cont, &test, sizeof(test));
            glm::uvec2 res{_model->get_nka()->getDepthResolution()};

            // float aspect = float(res.x) / res.y;
            // for rotated texture visualization
            float aspect = float(res.y) / res.x;

            float width = ImGui::GetWindowContentRegionWidth();
            ImGui::Image(cont, ImVec2(width, width / aspect), ImVec2(0, 0), ImVec2(1, 1), ImColor(255, 255, 255, 255), ImColor(255, 255, 255, 128));
            ImGui::End();
        }
        if(!show_tex)
        {
            _io->_gui_texture_settings.pop_back();
        }
    }
}
void renderer::update_model_matrix(bool load_ident)
{
    gloost::Point3 speed(0.0, 0.0, 0.0);
    glm::fvec2 speed_button1(_model->get_navi()->getOffset(0));
    glm::fvec2 speed_button2(_model->get_navi()->getOffset(1));
    float fac = 0.005;
    speed[0] = speed_button1.x * fac;
    speed[1] = speed_button1.y * -1.0 * fac;
    speed[2] = speed_button2.y * fac;
    gloost::Matrix camview(_model->get_navi()->get(speed));
    camview.invert();

    glMatrixMode(GL_MODELVIEW);
    if(load_ident)
    {
        glLoadIdentity();
    }
    glMultMatrixf(camview.data());

    static double curr_rot = 0.0;
    const double TAU = 2.0 * 180.0;
    if(_io->_animate)
    {
        curr_rot += ImGui::GetIO().DeltaTime * 10.0;
        if(curr_rot >= TAU)
            curr_rot = 0.0;
    }
    glRotatef(curr_rot, 0.0, 1.0, 0.0);

    _model->get_navi()->resetOffsets();
}
void renderer::process_textures() { _model->get_nka()->processTextures(); }
void renderer::draw3d()
{
    bool update_textures = false;
    if(_io->_play)
    {
        update_textures = update_textures || _model->get_nka()->update();
    }

    if(update_textures)
    {
        process_textures();
    }

    glClearColor(_io->_clear_color[0], _io->_clear_color[1], _io->_clear_color[2], _io->_clear_color[3]);

    switch(_io->_stereo_mode)
    {
    case 0: // MONO
    {
        glViewport(0, 0, _io->_screenWidth, _io->_screenHeight);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        _model->get_camera()->set();
        update_model_matrix();
        _model->get_recons().at(_io->_recon_mode)->drawF();
    }
    break;
    case 1: // ANAGLYPH STEREO
    {
        glViewport(0, 0, _io->_screenWidth, _io->_screenHeight);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        _model->get_stereo_camera()->setLeft();
        update_model_matrix(false);

        _model->get_recons().at(_io->_recon_mode)->setColorMaskMode(1);
        _model->get_recons().at(_io->_recon_mode)->drawF();

        glClear(GL_DEPTH_BUFFER_BIT);
        _model->get_stereo_camera()->setRight();
        update_model_matrix(false);
        _model->get_recons().at(_io->_recon_mode)->setColorMaskMode(2);
        _model->get_recons().at(_io->_recon_mode)->drawF();
    }
    break;
    case 2: // SIDE-BY-SIDE STEREO
    {
        if(_model->get_fbr())
        {
            sys::feedback fb = _model->get_fbr()->get();
            const gloost::Matrix cyclops_mat = glm2gloost(fb.cyclops_mat);
            const gloost::Matrix screen_mat = glm2gloost(fb.screen_mat);
            const gloost::Matrix model_mat = glm2gloost(fb.model_mat);
            _io->_recon_mode = fb.recon_mode;

            glViewport(0, 0, _io->_windowWidth, _io->_windowHeight);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            _model->get_stereo_camera()->setCyclopsMatrix(cyclops_mat);
            _model->get_stereo_camera()->setScreenMatrix(screen_mat);

            glViewport(_io->_left_pos_x, _io->_left_pos_y, _io->_screenWidth, _io->_screenHeight);
            _model->get_stereo_camera()->setLeft();
            glMatrixMode(GL_MODELVIEW);
            glMultMatrixf(model_mat.data());
            _model->get_recons().at(_io->_recon_mode)->setViewportOffset((float)_io->_left_pos_x, (float)_io->_left_pos_y);
            _model->get_recons().at(_io->_recon_mode)->drawF();

            glViewport(_io->_right_pos_x, _io->_right_pos_y, _io->_screenWidth, _io->_screenHeight);
            _model->get_stereo_camera()->setRight();
            glMatrixMode(GL_MODELVIEW);
            glMultMatrixf(model_mat.data());
            _model->get_recons().at(_io->_recon_mode)->setViewportOffset((float)_io->_right_pos_x, (float)_io->_right_pos_y);
            _model->get_recons().at(_io->_recon_mode)->drawF();

            glViewport(0, 0, _io->_screenWidth, _io->_screenHeight);
        }
    }
    break;
    }

    if(0 == _io->_stereo_mode)
    {
        if(_io->_draw_calibvis)
        {
            _model->get_calibvis()->draw();
        }
        if(_io->_draw_frustums)
        {
            _model->get_cv()->drawFrustums();
        }
        if(_io->_draw_grid)
        {
            _model->get_bbox()->draw();
        }
    }

    glMemoryBarrier(GL_ALL_BARRIER_BITS);
}
void renderer::next_shading_mode()
{
    _shading_buffer_data.mode = (_shading_buffer_data.mode + 1) % 4;
    _buffer_shading->setSubData(0, sizeof(shading_data_t), &_shading_buffer_data);
}