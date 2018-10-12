//
// Created by xaf on 19.05.18.
//

#include "renderer.h"
#include <globjects/Sync.h>
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
    _sequencer = new FrameSequencer(FrameSequencer::Type::INCREASING_STEP, 0, 1);
}
renderer::~renderer()
{
    _buffer_shading->destroy();
    delete _sequencer;
}
void renderer::init()
{
    //    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    //    glPixelStorei(GL_PACK_ALIGNMENT, 1);

    glEnable(GL_CULL_FACE);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);

    _buffer_shading = new globjects::Buffer();
    _buffer_shading->ref();
    _buffer_shading->setData(sizeof(shading_data_t), &_shading_buffer_data, GL_STATIC_DRAW);
    _buffer_shading->bindBase(GL_UNIFORM_BUFFER, 1);
}
void renderer::update_gui()
{
    ImGui::SetNextWindowSize(ImVec2(200, 100), ImGuiSetCond_FirstUseEver);
    ImGui::Begin("Settings");
    if(ImGui::Button("Show textures"))
    {
        _io->_gui_texture_settings.emplace_back(0, 0);
    }
    if(ImGui::Checkbox("Watch OpenGL errors", &_io->_watch_errors))
    {
        watch_gl_errors(_io->_watch_errors);
    };
    ImGui::Text("%.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    if(ImGui::CollapsingHeader("Reconstruction Mode", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::RadioButton("Points", &_io->_recon_mode, 0);
        if(ImGui::RadioButton("Performance Capture", &_io->_recon_mode, 1))
        {
            std::shared_ptr<kinect::ReconPerformanceCapture> recon_pc = std::dynamic_pointer_cast<kinect::ReconPerformanceCapture>(_model->get_recons().at(1));
            recon_pc->setTsdfLimit(_io->_tsdf_limit);
            //            recon_pc->setVoxelSize(_io->_voxel_size);
            _io->_brick_size = recon_pc->getBrickSize();
            //            recon_pc->setBrickSize(_io->_brick_size);
            recon_pc->setMinVoxelsPerBrick(_io->_min_voxels);
            recon_pc->updateOccupiedBricks();
            recon_pc->integrate_data_frame();
        }
        //        if(ImGui::RadioButton("Integration", &_io->_recon_mode, 2))
        //        {
        //            std::shared_ptr<kinect::ReconIntegration> recon_integration = std::dynamic_pointer_cast<kinect::ReconIntegration>(_model->get_recons().at(2));
        //            recon_integration->setTsdfLimit(_io->_tsdf_limit);
        //            recon_integration->setVoxelSize(_io->_voxel_size);
        //            _io->_brick_size = recon_integration->getBrickSize();
        //            recon_integration->setBrickSize(_io->_brick_size);
        //            recon_integration->setMinVoxelsPerBrick(_io->_min_voxels);
        //            recon_integration->updateOccupiedBricks();
        //            recon_integration->integrate();
        //        }
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
    if(ImGui::CollapsingHeader("Processing"))
    {
        if(ImGui::Checkbox("Morphological Filter", &_io->_processed))
        {
            _model->get_nka()->useProcessedDepths(_io->_processed);
        }
        if(ImGui::Checkbox("Bilateral Filter", &_io->_bilateral))
        {
            _model->get_nka()->filterTextures(_io->_bilateral);
        }
        if(ImGui::Checkbox("Boundary Refinement", &_io->_refine))
        {
            _model->get_nka()->refineBoundary(_io->_refine);
        }
    }
    if(ImGui::CollapsingHeader("Performance Capture", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::SliderInt("Frame Reset", &_model->get_recon_pc()->_conf.reset_frame_count, 1, 100, "%.0f");

        ImGui::Separator();

        ImGui::Columns(2, NULL, false);
        ImGui::Checkbox("Volume Bricking", &_model->get_recon_pc()->_conf.use_bricks);
        ImGui::NextColumn();
        ImGui::Text("%.3f %% occupied", _model->get_recon_pc()->occupiedRatio() * 100.0f);
        ImGui::Columns(1);

        if(_io->_bricking)
        {
            ImGui::Checkbox("Draw occupied bricks", &_model->get_recon_pc()->_conf.draw_bricks);
        }

        ImGui::Separator();

        ImGui::Checkbox("Debug Texture Silhouettes", &_model->get_recon_pc()->_conf.debug_texture_silhouettes);
        ImGui::Checkbox("Debug Texture Alignment Error", &_model->get_recon_pc()->_conf.debug_texture_alignment_error);
        ImGui::Checkbox("Debug Reference Volume", &_model->get_recon_pc()->_conf.debug_reference_volume);
        ImGui::Checkbox("Debug Reference Mesh", &_model->get_recon_pc()->_conf.debug_reference_mesh);
        ImGui::Checkbox("Debug ED Sampling", &_model->get_recon_pc()->_conf.debug_ed_sampling);
        ImGui::Checkbox("Debug Vertices", &_model->get_recon_pc()->_conf.debug_sorted_vertices);

        if(_model->get_recon_pc()->_conf.debug_sorted_vertices)
        {
            ImGui::RadioButton("Misalignment energy", &_model->get_recon_pc()->_conf.debug_sorted_vertices_mode, 0);
            ImGui::RadioButton("Data Term", &_model->get_recon_pc()->_conf.debug_sorted_vertices_mode, 1);
            ImGui::RadioButton("Hull Term", &_model->get_recon_pc()->_conf.debug_sorted_vertices_mode, 2);
            ImGui::RadioButton("Correspondence Term", &_model->get_recon_pc()->_conf.debug_sorted_vertices_mode, 3);
            ImGui::RadioButton("Regularization Term", &_model->get_recon_pc()->_conf.debug_sorted_vertices_mode, 4);

            ImGui::Checkbox("Debug Vertex Connections", &_model->get_recon_pc()->_conf.debug_sorted_vertices_connections);
        }

        if(_model->get_recon_pc()->_conf.debug_sorted_vertices_connections)
        {
            ImGui::RadioButton("Debug Vertex Vectors", &_model->get_recon_pc()->_conf.debug_sorted_vertices_traces, 0);
            ImGui::RadioButton("Debug Vertex Traces", &_model->get_recon_pc()->_conf.debug_sorted_vertices_traces, 1);
        }

        ImGui::Checkbox("Debug Gradient Field", &_model->get_recon_pc()->_conf.debug_gradient_field);
        ImGui::Checkbox("Debug Warped Reference Volume [Surface]", &_model->get_recon_pc()->_conf.debug_warped_reference_volume_surface);

        ImGui::Separator();

        ImGui::Checkbox("Pipeline Preprocess Textures", &_model->get_recon_pc()->_conf.pipeline_preprocess_textures);
        ImGui::Checkbox("Pipeline Sample", &_model->get_recon_pc()->_conf.pipeline_sample);
        ImGui::Checkbox("Pipeline Correspondence", &_model->get_recon_pc()->_conf.pipeline_correspondence);
        ImGui::Checkbox("Pipeline Align", &_model->get_recon_pc()->_conf.pipeline_align);
        ImGui::Checkbox("Pipeline Fuse", &_model->get_recon_pc()->_conf.pipeline_fuse);

        ImGui::Separator();

        ImGui::SliderInt("Gaussian Iterations", &_model->get_recon_pc()->_conf.textures_silhouettes_iterations, 0, 100, "%.0f");

        ImGui::Separator();

        ImGui::SliderFloat("Weight Data", &_model->get_recon_pc()->_conf.weight_data, 0.f, 1.f, "%.5f");
        ImGui::SliderFloat("Weight Visual Hull", &_model->get_recon_pc()->_conf.weight_hull, 0.f, 1.f, "%.5f");
        ImGui::SliderFloat("Weight Correspondence", &_model->get_recon_pc()->_conf.weight_correspondence, 0.f, 1.f, "%.5f");
        ImGui::SliderFloat("Weight Regularization", &_model->get_recon_pc()->_conf.weight_regularization, 0.f, 1.f, "%.5f");

        ImGui::Separator();

        ImGui::SliderFloat("Starting Mu Value", &_model->get_recon_pc()->_conf.solver_mu, 0.1f, 5000.f, "%.5f");
        ImGui::SliderFloat("Mu step", &_model->get_recon_pc()->_conf.solver_mu_step, 0.01f, 100.0f, "%.5f");
        ImGui::SliderInt("LMA Max Iterations", &_model->get_recon_pc()->_conf.solver_lma_max_iter, 0, 100, "%.0f");
        ImGui::SliderInt("CG Max Iterations", &_model->get_recon_pc()->_conf.solver_cg_steps, 0, 10, "%.0f");

        ImGui::Separator();

        ImGui::SliderFloat("Rejection Threshold", &_model->get_recon_pc()->_conf.rejection_threshold, 0.0001f, 0.03f, "%.5f");
    }
    if(ImGui::CollapsingHeader("Settings"))
    {
        switch(_io->_recon_mode)
        {
        case 0: // points
        {
            if(ImGui::CollapsingHeader("Processing Performance"))
            {
                ImGui::SameLine();
                ImGui::Text("   %.3f ms", TimerDatabase::instance().duration("1preprocess") / 1000000.0f);
                ImGui::Text("   Reconstruction");
                ImGui::SameLine();
                ImGui::Text("   %.3f ms", TimerDatabase::instance().duration("draw") / 1000000.0f);
            }
        }
        break;
        case 1: // performance capture
        {
            std::shared_ptr<kinect::ReconPerformanceCapture> recon_pc = std::dynamic_pointer_cast<kinect::ReconPerformanceCapture>(_model->get_recons().at(1));

            if(ImGui::SliderFloat("TSDF Limit", &_io->_tsdf_limit, 0.001f, 0.03f, "%.5f"))
            {
                recon_pc->setTsdfLimit(_io->_tsdf_limit);
            }
            if(ImGui::SliderFloat("Voxel Size", &_io->_voxel_size, 0.01f, 0.08f, "%.5f"))
            {
                // recon_pc->setVoxelSize(_io->_voxel_size);
                _io->_brick_size = recon_pc->getBrickSize();
            }
            if(ImGui::SliderFloat("Brick Size", &_io->_brick_size, 0.075f, 0.5f, "%.3f"))
            {
                // recon_pc->setBrickSize(_io->_brick_size);
                _io->_brick_size = recon_pc->getBrickSize();
            }
            if(ImGui::SliderInt("Min Brick Voxels", &_io->_min_voxels, 0, 500, "%.0f"))
            {
                recon_pc->setMinVoxelsPerBrick(_io->_min_voxels);
                recon_pc->updateOccupiedBricks();
                recon_pc->integrate_data_frame();
            }
            ImGui::Checkbox("Draw TSDF", &_io->_draw_calibvis);

            if(ImGui::CollapsingHeader("Processing Performance"))
            {
                ImGui::SameLine();
                ImGui::Text("   %.3f ms", TimerDatabase::instance().duration("draw") / 1000000.0f);

                ImGui::Text("   Data volume integration");
                ImGui::Text("   %.3f ms", TimerDatabase::instance().duration(kinect::ReconPerformanceCapture::TIMER_DATA_VOLUME_INTEGRATION) / 1000000.0f);
                ImGui::Text("   Texture processing");
                ImGui::Text("   %.3f ms", _model->get_recon_pc()->_conf.time_preprocess);
                ImGui::Text("   SIFT extraction & filtering");
                ImGui::Text("   %.3f ms", _model->get_recon_pc()->_conf.time_correspondence);
                ImGui::Text("   Reference copy");
                ImGui::Text("   %.3f ms", _model->get_recon_pc()->_conf.time_copy_reference);
                ImGui::Text("   Reference mesh extraction");
                ImGui::Text("   %.3f ms", TimerDatabase::instance().duration(kinect::ReconPerformanceCapture::TIMER_REFERENCE_MESH_EXTRACTION) / 1000000.0f);
                ImGui::Text("   ED graph sampling");
                ImGui::Text("   %.3f ms", _model->get_recon_pc()->_conf.time_sample_ed);
                ImGui::Text("   Non-rigid alignment");
                ImGui::Text("   %.3f ms", _model->get_recon_pc()->_conf.time_nra);
                ImGui::Text("   Fusion");
                ImGui::Text("   %.3f ms", _model->get_recon_pc()->_conf.time_fuse);
                ImGui::Text("   Data mesh draw");
                ImGui::Text("   %.3f ms", TimerDatabase::instance().duration(kinect::ReconPerformanceCapture::TIMER_DATA_MESH_DRAW) / 1000000.0f);
            }
        }
        break;
        case 2: // raymarched integration
        {
            std::shared_ptr<kinect::ReconIntegration> recon_integration = std::dynamic_pointer_cast<kinect::ReconIntegration>(_model->get_recons().at(2));

            if(ImGui::SliderFloat("TSDF Limit", &_io->_tsdf_limit, 0.001f, 0.03f, "%.5f"))
            {
                recon_integration->setTsdfLimit(_io->_tsdf_limit);
            }
            if(ImGui::SliderFloat("Voxel Size", &_io->_voxel_size, 0.01f, 0.08f, "%.5f"))
            {
                recon_integration->setVoxelSize(_io->_voxel_size);
                _io->_brick_size = recon_integration->getBrickSize();
            }
            if(ImGui::SliderFloat("Brick Size", &_io->_brick_size, 0.075f, 0.5f, "%.3f"))
            {
                recon_integration->setBrickSize(_io->_brick_size);
                _io->_brick_size = recon_integration->getBrickSize();
            }
            if(ImGui::SliderInt("Min Brick Voxels", &_io->_min_voxels, 0, 500, "%.0f"))
            {
                recon_integration->setMinVoxelsPerBrick(_io->_min_voxels);
                recon_integration->updateOccupiedBricks();
                recon_integration->integrate();
            }
            if(ImGui::Checkbox("Color hole filling", &_io->_colorfill))
            {
                recon_integration->setColorFilling(_io->_colorfill);
            }
            if(_io->_bricking)
            {
                ImGui::Columns(2, NULL, false);
                if(ImGui::Checkbox("Volume Bricking", &_io->_bricking))
                {
                    recon_integration->setUseBricks(_io->_bricking);
                }
                ImGui::NextColumn();
                ImGui::Text("%.3f %% occupied", recon_integration->occupiedRatio() * 100.0f);
                ImGui::Columns(1);

                if(_io->_bricking)
                {
                    if(ImGui::Checkbox("Skip empty Spaces", &_io->_skip_space))
                    {
                        recon_integration->setSpaceSkip(_io->_skip_space);
                    }
                    if(ImGui::Checkbox("Draw occupied bricks", &_io->_draw_bricks))
                    {
                        recon_integration->setDrawBricks(_io->_draw_bricks);
                    }
                }
            }
            else
            {
                if(ImGui::Checkbox("Volume Bricking", &_io->_bricking))
                {
                    recon_integration->setUseBricks(_io->_bricking);
                    recon_integration->integrate();
                }
            }
            ImGui::Checkbox("Draw TSDF", &_io->_draw_calibvis);

            if(ImGui::CollapsingHeader("Processing Performance"))
            {
                ImGui::SameLine();
                ImGui::Text("   %.3f ms", TimerDatabase::instance().duration("1preprocess") / 1000000.0f);
                ImGui::Text("   Reconstruction");
                ImGui::SameLine();
                ImGui::Text("   %.3f ms", TimerDatabase::instance().duration("draw") / 1000000.0f);
            }
        }
        break;
        default:
            throw std::runtime_error("Unknown reconstruction mode: " + std::to_string(_io->_recon_mode));
        }
    }

    ImGui::End();

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
                static std::vector<const char *> listbox_items = {"Color", "Depth", "Quality", "Normals", "Silhouette", "Orig Depth", "LAB colors", "Optical Flow"};
                ImGui::ListBox("Type", &setting.first, listbox_items.data(), listbox_items.size(), listbox_items.size());
            }

            TexInfo test = {(uint16_t)(_model->get_nka()->getStartTextureUnit() + setting.first), (uint16_t)(-setting.second - 1)};
            ImTextureID cont;
            std::memcpy(&cont, &test, sizeof(TexInfo));
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
void renderer::process_textures()
{
    std::shared_ptr<kinect::ReconPerformanceCapture> recon_pc = std::dynamic_pointer_cast<kinect::ReconPerformanceCapture>(_model->get_recons().at(1));
    recon_pc->clearOccupiedBricks();

    _model->get_nka()->processTextures();

    recon_pc->updateOccupiedBricks();
}
void renderer::draw3d()
{
    if(_io->_play)
    {
        int frame_position = _sequencer->next_frame_position();

        _model->get_recon_pc()->_conf.frame = _sequencer->current_frame();
        _model->get_nka()->update(frame_position);

        process_textures();

        _model->get_nka()->write_out_pbos();

        // TODO: investigate
        /// Have to do it two times, otherwise get black textures
        process_textures();
    }

    glClearColor(_io->_clear_color[0], _io->_clear_color[1], _io->_clear_color[2], _io->_clear_color[3]);

    switch(_io->_stereo_mode)
    {
    case 0: // MONO
    {
        if(!_io->_splitscreen_comparison)
        {
            glViewport(0, 0, _io->_screenWidth, _io->_screenHeight);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            _model->get_camera()->set();
            update_model_matrix();
            _model->get_recons().at(_io->_recon_mode)->drawF();
        }
        else
        {
            _model->get_camera()->set();
            update_model_matrix();
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            glViewport(0, 0, _io->_screenWidth / 2, _io->_screenHeight);
            _model->get_recons().at(_io->_recon_mode)->drawF();

            glViewport(_io->_screenWidth / 2, 0, _io->_screenWidth / 2, _io->_screenHeight);
            _model->get_recons().at(_io->_recon_mode)->drawComparison();
        }
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
        if(!_io->_splitscreen_comparison)
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
        else
        {
            glViewport(0, 0, _io->_screenWidth / 2, _io->_screenHeight);

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

            glViewport(_io->_screenWidth / 2, 0, _io->_screenWidth / 2, _io->_screenHeight);

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
    }

    if((_io->_play && _sequencer->get_type() == FrameSequencer::Type::INCREASING_STEP) || (_io->_play && _sequencer->is_finished()))
    {
        _io->_play = false;
        _model->get_recon_pc()->pause(true);

        if(_sequencer->is_finished())
        {
            _sequencer->rewind();
        }
    }

    glMemoryBarrier(GL_ALL_BARRIER_BITS);
}
void renderer::next_shading_mode()
{
    _shading_buffer_data.mode = (_shading_buffer_data.mode + 1) % 4;
    _buffer_shading->setSubData(0, sizeof(shading_data_t), &_shading_buffer_data);
}
void renderer::watch_gl_errors(bool activate)
{
    if(activate)
    {
        // add callback after each function call
        glbinding::setCallbackMaskExcept(glbinding::CallbackMask::After | glbinding::CallbackMask::ParametersAndReturnValue, {"glGetError", "glVertex3f", "glVertex2f", "glBegin", "glColor3f"});

        glbinding::setAfterCallback([](const glbinding::FunctionCall &call) {
            GLenum error = glGetError();
            if(error != GL_NO_ERROR)
            {
                // print name
                std::cerr << "OpenGL Error: " << call.function->name() << "(";
                // parameters
                for(unsigned i = 0; i < call.parameters.size(); ++i)
                {
                    std::cerr << call.parameters[i]->asString();
                    if(i < call.parameters.size() - 1)
                        std::cerr << ", ";
                }
                std::cerr << ")";
                // return value
                if(call.returnValue)
                {
                    std::cerr << " -> " << call.returnValue->asString();
                }
                // error
                std::cerr << " - " << glbinding::Meta::getString(error) << std::endl;
                throw std::runtime_error("OpenGL error caught");
            }
        });
    }
    else
    {
        glbinding::setCallbackMask(glbinding::CallbackMask::None);
    }
}
