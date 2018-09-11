#include "Renderer.h"
#include "Controller.h"

using namespace glbinding;

void Renderer::resize(int width, int height)
{
    _width = glm::max(1, width);
    _height = glm::max(1, height);

    // Set the viewport to be the entire window
    glViewport(0, 0, _width, _height);

    build_projection_matrix(49.13f, 1.185185185f, 0.5f, 3.f);
}
static int i = 0;
void Renderer::draw()
{
    /// Geometry pass

    for(unsigned i = 0; i < 4; i++)
    {
        _fbo->bind();

        _fbo->setDrawBuffers({(gl::GLenum)GL_COLOR_ATTACHMENT0, (gl::GLenum)GL_DEPTH_ATTACHMENT});
        _fbo->attachTextureLayer((gl::GLenum)GL_COLOR_ATTACHMENT0, _texture_color, 0, i);
        _fbo->attachTextureLayer((gl::GLenum)GL_DEPTH_ATTACHMENT, _texture_depth, 0, i);

        glViewport(0, 0, 1280, 1080);

        if(GL_FRAMEBUFFER_COMPLETE != _fbo->checkStatus())
        {
            std::cerr << "FBO is not happy: " << _fbo->checkStatus() << std::endl;
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        globjects::Framebuffer::clearColor(0.f, 0.2f, 0.f, 1.f);
        globjects::Framebuffer::clearDepth(4.f);

        //std::cout << "Camera " << i << std::endl;
        set_camera(_camera_descriptor[i].pos.x, _camera_descriptor[i].pos.y, _camera_descriptor[i].pos.z, _camera_descriptor[i].look_at.x, _camera_descriptor[i].look_at.y,
                   _camera_descriptor[i].look_at.z);

        set_identity_matrix(_model_matrix, 4);
        translate(_translation.x, _translation.y, _translation.z);

        _program->use();

        // 27 meshes
        /*recursive_render(_controller->get_environment(), _controller->get_environment()->mRootNode);*/
        recursive_render(_controller->get_poi(), _controller->get_poi()->mRootNode);

        globjects::Program::release();

        globjects::Framebuffer::unbind();
    }

    /// Color postprocess pass

    for(unsigned i = 0; i < 4; i++)
    {
        _fbo_color->bind();

        _fbo_color->setDrawBuffers({(gl::GLenum)GL_COLOR_ATTACHMENT0, (gl::GLenum)GL_DEPTH_ATTACHMENT});
        _fbo_color->attachTextureLayer((gl::GLenum)GL_COLOR_ATTACHMENT0, _texture_color_postproc, 0, i);
        _fbo_color->attachTextureLayer((gl::GLenum)GL_DEPTH_ATTACHMENT, _texture_dummy_depth_1280, 0, i);

        glViewport(0, 0, 1280, 1080);

        if(GL_FRAMEBUFFER_COMPLETE != _fbo_color->checkStatus())
        {
            std::cerr << "Color FBO is not happy: " << _fbo_color->checkStatus() << std::endl;
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        globjects::Framebuffer::clearColor(0.f, 0.2f, 0.f, 1.f);
        globjects::Framebuffer::clearDepth(4.f);

        _program_postproc_color->use();
        _program_postproc_color->setUniform("layer", (int)i);

        glBindTexture(GL_TEXTURE_2D_ARRAY, _texture_color->id());

        _vao_fsquad_debug->bind();
        _vao_fsquad_debug->drawArrays((gl::GLenum)GL_TRIANGLE_STRIP, 0, 4);
        globjects::VertexArray::unbind();

        glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

        globjects::Program::release();

        globjects::Framebuffer::unbind();
    }

    /// Depth postprocess pass

    for(unsigned i = 0; i < 4; i++)
    {
        _fbo_depth->bind();

        _fbo_depth->setDrawBuffers({(gl::GLenum)GL_COLOR_ATTACHMENT0, (gl::GLenum)GL_DEPTH_ATTACHMENT});
        _fbo_depth->attachTextureLayer((gl::GLenum)GL_COLOR_ATTACHMENT0, _texture_depth_postproc, 0, i);
        _fbo_depth->attachTextureLayer((gl::GLenum)GL_DEPTH_ATTACHMENT, _texture_dummy_depth_512, 0, i);

        glViewport(0, 0, 512, 424);

        if(GL_FRAMEBUFFER_COMPLETE != _fbo_depth->checkStatus())
        {
            std::cerr << "Depth FBO is not happy: " << _fbo_depth->checkStatus() << std::endl;
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        globjects::Framebuffer::clearColor(0.f, 0.2f, 0.f, 1.f);
        globjects::Framebuffer::clearDepth(4.f);

        _program_postproc_depth->use();
        _program_postproc_depth->setUniform("layer", (int)i);

        glBindTexture(GL_TEXTURE_2D_ARRAY, _texture_depth->id());

        _vao_fsquad_debug->bind();
        _vao_fsquad_debug->drawArrays((gl::GLenum)GL_TRIANGLE_STRIP, 0, 4);
        globjects::VertexArray::unbind();

        glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

        globjects::Program::release();

        globjects::Framebuffer::unbind();
    }

    /// Write to RAM
    {
        _texture_color_postproc->getImage(0, (gl::GLenum)GL_RGB, (gl::GLenum)GL_UNSIGNED_BYTE, _frame_color);
        _texture_depth_postproc->getImage(0, (gl::GLenum)GL_RED, (gl::GLenum)GL_FLOAT, _frame_depth);
    }

#define WRITE_RGBD

#ifdef WRITE_RGBD

    if(i < 101)
    {
        size_t colorsize = 1280 * 1080 * 3 * sizeof(byte);
        size_t depthsize = 512 * 424 * sizeof(float);

        for(unsigned i = 0; i < 4; ++i)
        {
            _fb->write((byte *)_frame_color + colorsize * i, colorsize);
            _fb->write((byte *)_frame_depth + depthsize * i, depthsize);
        }
    }

#endif

    /// Visualization pass

    {
        glViewport(0, 0, _width, _height);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        globjects::Framebuffer::clearColor(0.f, 0.2f, 0.f, 1.f);
        globjects::Framebuffer::clearDepth(4.f);

        _program_debug_texture->use();

        switch(_mode)
        {
        case 0:
        {
            glBindTexture(GL_TEXTURE_2D_ARRAY, _texture_color_postproc->id());
            _program_debug_texture->setUniform("texture_2d_array", 12);
        }
        break;
        case 1:
        {
            glBindTexture(GL_TEXTURE_2D_ARRAY, _texture_depth_postproc->id());
            _program_debug_texture->setUniform("texture_2d_array", 13);
        }
        break;
        }

        _vao_fsquad_debug->bind();
        _vao_fsquad_debug->drawArrays((gl::GLenum)GL_TRIANGLE_STRIP, 0, 4);
        globjects::VertexArray::unbind();

        glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

        globjects::Program::release();
    }

    i++;
}
Renderer::Renderer(Controller *controller)
{
    _fb = new FileBuffer(std::string("/home/xaf/Desktop/MSc/data/synthetic_rgbd/record.stream").c_str());
    if(!_fb->open("w", 0))
    {
        std::cerr << "error opening "
                  << "record.stream"
                  << " exiting..." << std::endl;
    }

    _frame_color = (unsigned char *)malloc(4 * (1280 * 1080) * 3 * sizeof(unsigned char));
    _frame_depth = (float *)malloc(4 * (512 * 424) * sizeof(float));
    _vao_fsquad_debug = new globjects::VertexArray();

    const std::array<glm::vec2, 4> raw{{glm::vec2(1.f, -1.f), glm::vec2(1.f, 1.f), glm::vec2(-1.f, -1.f), glm::vec2(-1.f, 1.f)}};

    _buffer_fsquad_debug = new globjects::Buffer();
    _buffer_fsquad_debug->setData(raw, (gl::GLenum)GL_STATIC_DRAW); // needed for some drivers

    auto binding = _vao_fsquad_debug->binding(0);
    binding->setAttribute(0);
    binding->setBuffer(_buffer_fsquad_debug, 0, sizeof(float) * 2);
    binding->setFormat(2, (gl::GLenum)GL_FLOAT, GL_FALSE, 0);
    _vao_fsquad_debug->enable(0);

    _fbo = new globjects::Framebuffer();
    _fbo_color = new globjects::Framebuffer();
    _fbo_depth = new globjects::Framebuffer();

    _texture_color = globjects::Texture::createDefault((gl::GLenum)GL_TEXTURE_2D_ARRAY);
    _texture_color->image3D(0, (gl::GLenum)GL_RGB8, 1280, 1080, 4, 0, (gl::GLenum)GL_RGB, (gl::GLenum)GL_UNSIGNED_BYTE, (void *)nullptr);
    _texture_color->setParameter((gl::GLenum)GL_TEXTURE_MIN_FILTER, (gl::GLenum)GL_LINEAR);
    _texture_color->setParameter((gl::GLenum)GL_TEXTURE_MAG_FILTER, (gl::GLenum)GL_LINEAR);

    _texture_depth = globjects::Texture::createDefault((gl::GLenum)GL_TEXTURE_2D_ARRAY);
    _texture_depth->image3D(0, (gl::GLenum)GL_DEPTH_COMPONENT16, 1280, 1080, 4, 0, (gl::GLenum)GL_DEPTH_COMPONENT, (gl::GLenum)GL_FLOAT, (void *)nullptr);
    _texture_depth->setParameter((gl::GLenum)GL_TEXTURE_MIN_FILTER, (gl::GLenum)GL_NEAREST);
    _texture_depth->setParameter((gl::GLenum)GL_TEXTURE_MAG_FILTER, (gl::GLenum)GL_NEAREST);

    _texture_color_postproc = globjects::Texture::createDefault((gl::GLenum)GL_TEXTURE_2D_ARRAY);
    _texture_color_postproc->image3D(0, (gl::GLenum)GL_RGB8, 1280, 1080, 4, 0, (gl::GLenum)GL_RGB, (gl::GLenum)GL_UNSIGNED_BYTE, (void *)nullptr);
    _texture_color_postproc->setParameter((gl::GLenum)GL_TEXTURE_MIN_FILTER, (gl::GLenum)GL_LINEAR);
    _texture_color_postproc->setParameter((gl::GLenum)GL_TEXTURE_MAG_FILTER, (gl::GLenum)GL_LINEAR);

    _texture_depth_postproc = globjects::Texture::createDefault((gl::GLenum)GL_TEXTURE_2D_ARRAY);
    _texture_depth_postproc->image3D(0, (gl::GLenum)GL_LUMINANCE32F_ARB, 512, 424, 4, 0, (gl::GLenum)GL_RED, (gl::GLenum)GL_FLOAT, (void *)nullptr);
    _texture_depth_postproc->setParameter((gl::GLenum)GL_TEXTURE_MIN_FILTER, (gl::GLenum)GL_NEAREST);
    _texture_depth_postproc->setParameter((gl::GLenum)GL_TEXTURE_MAG_FILTER, (gl::GLenum)GL_NEAREST);

    _texture_dummy_depth_1280 = globjects::Texture::createDefault((gl::GLenum)GL_TEXTURE_2D_ARRAY);
    _texture_dummy_depth_1280->image3D(0, (gl::GLenum)GL_DEPTH_COMPONENT16, 1280, 1080, 4, 0, (gl::GLenum)GL_DEPTH_COMPONENT, (gl::GLenum)GL_FLOAT, (void *)nullptr);
    _texture_dummy_depth_1280->setParameter((gl::GLenum)GL_TEXTURE_MIN_FILTER, (gl::GLenum)GL_NEAREST);
    _texture_dummy_depth_1280->setParameter((gl::GLenum)GL_TEXTURE_MAG_FILTER, (gl::GLenum)GL_NEAREST);

    _texture_dummy_depth_512 = globjects::Texture::createDefault((gl::GLenum)GL_TEXTURE_2D_ARRAY);
    _texture_dummy_depth_512->image3D(0, (gl::GLenum)GL_DEPTH_COMPONENT16, 512, 424, 4, 0, (gl::GLenum)GL_DEPTH_COMPONENT, (gl::GLenum)GL_FLOAT, (void *)nullptr);
    _texture_dummy_depth_512->setParameter((gl::GLenum)GL_TEXTURE_MIN_FILTER, (gl::GLenum)GL_NEAREST);
    _texture_dummy_depth_512->setParameter((gl::GLenum)GL_TEXTURE_MAG_FILTER, (gl::GLenum)GL_NEAREST);

    gl::glBindTextureUnit(10, _texture_color->id());
    gl::glBindTextureUnit(11, _texture_depth->id());
    gl::glBindTextureUnit(12, _texture_color_postproc->id());
    gl::glBindTextureUnit(13, _texture_depth_postproc->id());

    _program_debug_texture = new globjects::Program();
    _program_debug_texture->attach(globjects::Shader::fromFile((gl::GLenum)GL_VERTEX_SHADER, "/home/xaf/Desktop/MSc/impl/rgbd-recon/source/synthesizer/textured_quad.vs"));
    _program_debug_texture->attach(globjects::Shader::fromFile((gl::GLenum)GL_FRAGMENT_SHADER, "/home/xaf/Desktop/MSc/impl/rgbd-recon/source/synthesizer/debug_texture.fs"));

    _program_postproc_color = new globjects::Program();
    _program_postproc_color->attach(globjects::Shader::fromFile((gl::GLenum)GL_VERTEX_SHADER, "/home/xaf/Desktop/MSc/impl/rgbd-recon/source/synthesizer/textured_quad.vs"));
    _program_postproc_color->attach(globjects::Shader::fromFile((gl::GLenum)GL_FRAGMENT_SHADER, "/home/xaf/Desktop/MSc/impl/rgbd-recon/source/synthesizer/postproc_color.fs"));
    _program_postproc_color->setUniform("texture_2d_array", 10);

    _program_postproc_depth = new globjects::Program();
    _program_postproc_depth->attach(globjects::Shader::fromFile((gl::GLenum)GL_VERTEX_SHADER, "/home/xaf/Desktop/MSc/impl/rgbd-recon/source/synthesizer/textured_quad.vs"));
    _program_postproc_depth->attach(globjects::Shader::fromFile((gl::GLenum)GL_FRAGMENT_SHADER, "/home/xaf/Desktop/MSc/impl/rgbd-recon/source/synthesizer/postproc_depth.fs"));
    _program_postproc_depth->setUniform("texture_2d_array", 11);

    _program = new globjects::Program();

    _program->attach(globjects::Shader::fromFile((gl::GLenum)GL_VERTEX_SHADER, "/home/xaf/Desktop/MSc/impl/rgbd-recon/source/synthesizer/rgbd.vs"));
    _program->attach(globjects::Shader::fromFile((gl::GLenum)GL_FRAGMENT_SHADER, "/home/xaf/Desktop/MSc/impl/rgbd-recon/source/synthesizer/rgbd.fs"));

    _program->bindFragDataLocation(0, "out_color");

    _program->bindAttributeLocation(0, "position");
    _program->bindAttributeLocation(1, "normal");
    _program->bindAttributeLocation(2, "texCoord");

    _ub_matrices = new globjects::UniformBlock(_program, globjects::LocationIdentity("Matrices"));
    _ub_matrices->setBinding(1);
    _ub_materials = new globjects::UniformBlock(_program, globjects::LocationIdentity("Material"));
    _ub_materials->setBinding(2);

    _program->setUniform("texUnit", 0);

    _controller = controller;
    /*gen_VAOs_and_uniform_buffer(_controller->get_environment());*/
    // std::cout << _meshes.size ()<<std::endl;
    gen_VAOs_and_uniform_buffer(_controller->get_poi());
    // std::cout << _meshes.size ()<<std::endl;

    gl::glGenBuffers(1, &_matrices_uni_buffer);
    gl::glBindBuffer((gl::GLenum)GL_UNIFORM_BUFFER, _matrices_uni_buffer);
    gl::glBufferData((gl::GLenum)GL_UNIFORM_BUFFER, MatricesUniBufferSize, NULL, (gl::GLenum)GL_DYNAMIC_DRAW);
    gl::glBindBufferRange((gl::GLenum)GL_UNIFORM_BUFFER, 1, _matrices_uni_buffer, 0, MatricesUniBufferSize);
    gl::glBindBuffer((gl::GLenum)GL_UNIFORM_BUFFER, 0);

    _camera_descriptor.emplace_back(CameraDescriptor({-1.6f, 1.0f, 1.6f}, {0.f, -0.2f, 0.f}));
    _camera_descriptor.emplace_back(CameraDescriptor({-1.6f, 1.0f, -1.6f}, {0.f, -0.2f, 0.f}));
    _camera_descriptor.emplace_back(CameraDescriptor({1.6f, 1.0f, -1.6f}, {0.f, -0.2f, 0.f}));
    _camera_descriptor.emplace_back(CameraDescriptor({1.6f, 1.0f, 1.6f}, {0.f, -0.2f, 0.f}));

    set_identity_matrix(_model_matrix, 4);
}
Renderer::~Renderer()
{
    free(_frame_color);
    free(_frame_depth);

    _fb->close();
    delete _fb;

    _vao_fsquad_debug->destroy();
    _buffer_fsquad_debug->destroy();
    _fbo->destroy();

    _program->destroy();

    _texture_id_map.clear();

    for(unsigned int i = 0; i < _meshes.size(); ++i)
    {
        gl::glDeleteVertexArrays(1, &(_meshes[i].vao));
        gl::glDeleteTextures(1, &(_meshes[i].texIndex));
        gl::glDeleteBuffers(1, &(_meshes[i].uniformBlockIndex));
    }

    gl::glDeleteBuffers(1, &_matrices_uni_buffer);
}
void Renderer::recursive_render(const aiScene *sc, const aiNode *nd, int offset)
{
    aiMatrix4x4 m = nd->mTransformation;
    m.Transpose();

    push_matrix();

    float aux[16];
    memcpy(aux, &m, sizeof(float) * 16);
    mult_matrix(_model_matrix, aux);
    set_model_matrix();

    for(unsigned int n = 0; n < nd->mNumMeshes; ++n)
    {
        gl::glBindBufferRange((gl::GLenum)GL_UNIFORM_BUFFER, 2, _meshes[offset + nd->mMeshes[n]].uniformBlockIndex, 0, sizeof(struct MaterialDescriptor));
        gl::glBindTexture((gl::GLenum)GL_TEXTURE_2D, _meshes[offset + nd->mMeshes[n]].texIndex);
        gl::glBindVertexArray(_meshes[offset + nd->mMeshes[n]].vao);
        gl::glDrawElements((gl::GLenum)GL_TRIANGLES, _meshes[offset + nd->mMeshes[n]].numFaces * 3, (gl::GLenum)GL_UNSIGNED_INT, 0);
    }

    for(unsigned int n = 0; n < nd->mNumChildren; ++n)
    {
        recursive_render(sc, nd->mChildren[n], offset);
    }
    pop_matrix();
}
void Renderer::gen_VAOs_and_uniform_buffer(const aiScene *sc)
{
    struct MeshDescriptor a_mesh;
    struct MaterialDescriptor a_mat;
    GLuint buffer;

    // For each mesh
    for(unsigned int n = 0; n < sc->mNumMeshes; ++n)
    {
        const aiMesh *mesh = sc->mMeshes[n];

        // create array with faces
        // have to convert from Assimp format to array
        unsigned int *faceArray;
        faceArray = (unsigned int *)malloc(sizeof(unsigned int) * mesh->mNumFaces * 3);
        unsigned int faceIndex = 0;

        for(unsigned int t = 0; t < mesh->mNumFaces; ++t)
        {
            const aiFace *face = &mesh->mFaces[t];

            memcpy(&faceArray[faceIndex], face->mIndices, 3 * sizeof(unsigned int));
            faceIndex += 3;
        }
        a_mesh.numFaces = sc->mMeshes[n]->mNumFaces;

        // generate Vertex Array for mesh
        gl::glGenVertexArrays(1, &(a_mesh.vao));
        gl::glBindVertexArray(a_mesh.vao);

        // buffer for faces
        gl::glGenBuffers(1, &buffer);
        gl::glBindBuffer((gl::GLenum)GL_ELEMENT_ARRAY_BUFFER, buffer);
        gl::glBufferData((gl::GLenum)GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * mesh->mNumFaces * 3, faceArray, (gl::GLenum)GL_STATIC_DRAW);

        // buffer for vertex positions
        if(mesh->HasPositions())
        {
            gl::glGenBuffers(1, &buffer);
            gl::glBindBuffer((gl::GLenum)GL_ARRAY_BUFFER, buffer);
            gl::glBufferData((gl::GLenum)GL_ARRAY_BUFFER, sizeof(float) * 3 * mesh->mNumVertices, mesh->mVertices, (gl::GLenum)GL_STATIC_DRAW);
            gl::glEnableVertexAttribArray(0);
            gl::glVertexAttribPointer(0, 3, (gl::GLenum)GL_FLOAT, 0, 0, 0);
        }

        // buffer for vertex normals
        if(mesh->HasNormals())
        {
            gl::glGenBuffers(1, &buffer);
            gl::glBindBuffer((gl::GLenum)GL_ARRAY_BUFFER, buffer);
            gl::glBufferData((gl::GLenum)GL_ARRAY_BUFFER, sizeof(float) * 3 * mesh->mNumVertices, mesh->mNormals, (gl::GLenum)GL_STATIC_DRAW);
            gl::glEnableVertexAttribArray(1);
            gl::glVertexAttribPointer(1, 3, (gl::GLenum)GL_FLOAT, 0, 0, 0);
        }

        // buffer for vertex texture coordinates
        if(mesh->HasTextureCoords(0))
        {
            float *texCoords = (float *)malloc(sizeof(float) * 2 * mesh->mNumVertices);
            for(unsigned int k = 0; k < mesh->mNumVertices; ++k)
            {
                texCoords[k * 2] = mesh->mTextureCoords[0][k].x;
                texCoords[k * 2 + 1] = mesh->mTextureCoords[0][k].y;
            }
            gl::glGenBuffers(1, &buffer);
            gl::glBindBuffer((gl::GLenum)GL_ARRAY_BUFFER, buffer);
            gl::glBufferData((gl::GLenum)GL_ARRAY_BUFFER, sizeof(float) * 2 * mesh->mNumVertices, texCoords, (gl::GLenum)GL_STATIC_DRAW);
            gl::glEnableVertexAttribArray(2);
            gl::glVertexAttribPointer(2, 2, (gl::GLenum)GL_FLOAT, 0, 0, 0);
        }

        // unbind buffers
        gl::glBindVertexArray(0);
        gl::glBindBuffer((gl::GLenum)GL_ARRAY_BUFFER, 0);
        gl::glBindBuffer((gl::GLenum)GL_ELEMENT_ARRAY_BUFFER, 0);

        // create material uniform buffer
        aiMaterial *mtl = sc->mMaterials[mesh->mMaterialIndex];

        aiString texPath; // contains filename of texture
        if(AI_SUCCESS == mtl->GetTexture(aiTextureType_DIFFUSE, 0, &texPath))
        {
            // bind texture
            unsigned int texId = _texture_id_map[texPath.data];
            a_mesh.texIndex = texId;
            a_mat.texCount = 1;
        }
        else
            a_mat.texCount = 0;

        float c[4];
        set_float4(c, 0.8f, 0.8f, 0.8f, 1.0f);
        aiColor4D diffuse;
        if(AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_DIFFUSE, &diffuse))
            color4_to_float4(&diffuse, c);
        memcpy(a_mat.diffuse, c, sizeof(c));

        set_float4(c, 0.2f, 0.2f, 0.2f, 1.0f);
        aiColor4D ambient;
        if(AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_AMBIENT, &ambient))
            color4_to_float4(&ambient, c);
        memcpy(a_mat.ambient, c, sizeof(c));

        set_float4(c, 0.0f, 0.0f, 0.0f, 1.0f);
        aiColor4D specular;
        if(AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_SPECULAR, &specular))
            color4_to_float4(&specular, c);
        memcpy(a_mat.specular, c, sizeof(c));

        set_float4(c, 0.0f, 0.0f, 0.0f, 1.0f);
        aiColor4D emission;
        if(AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_EMISSIVE, &emission))
            color4_to_float4(&emission, c);
        memcpy(a_mat.emissive, c, sizeof(c));

        float shininess = 0.0;
        unsigned int max;
        aiGetMaterialFloatArray(mtl, AI_MATKEY_SHININESS, &shininess, &max);
        a_mat.shininess = shininess;

        gl::glGenBuffers(1, &(a_mesh.uniformBlockIndex));
        gl::glBindBuffer((gl::GLenum)GL_UNIFORM_BUFFER, a_mesh.uniformBlockIndex);
        gl::glBufferData((gl::GLenum)GL_UNIFORM_BUFFER, sizeof(a_mat), (void *)(&a_mat), (gl::GLenum)GL_STATIC_DRAW);

        _meshes.push_back(a_mesh);
    }
}
void Renderer::set_float4(float *f, float a, float b, float c, float d)
{
    f[0] = a;
    f[1] = b;
    f[2] = c;
    f[3] = d;
}
void Renderer::color4_to_float4(const aiColor4D *c, float *f)
{
    f[0] = c->r;
    f[1] = c->g;
    f[2] = c->b;
    f[3] = c->a;
}
void Renderer::push_matrix()
{
    float *aux = (float *)malloc(sizeof(float) * 16);
    memcpy(aux, _model_matrix, sizeof(float) * 16);
    _matrix_stack.push_back(aux);
}
void Renderer::pop_matrix()
{
    float *m = _matrix_stack[_matrix_stack.size() - 1];
    memcpy(_model_matrix, m, sizeof(float) * 16);
    _matrix_stack.pop_back();
    free(m);
}
void Renderer::set_identity_matrix(float *mat, int size)
{
    // fill matrix with 0s
    for(int i = 0; i < size * size; ++i)
        mat[i] = 0.0f;

    // fill diagonal with 1s
    for(int i = 0; i < size; ++i)
        mat[i + i * size] = 1.0f;
}
void Renderer::mult_matrix(float *a, float *b)
{
    float res[16];

    for(int i = 0; i < 4; ++i)
    {
        for(int j = 0; j < 4; ++j)
        {
            res[j * 4 + i] = 0.0f;
            for(int k = 0; k < 4; ++k)
            {
                res[j * 4 + i] += a[k * 4 + i] * b[j * 4 + k];
            }
        }
    }
    memcpy(a, res, 16 * sizeof(float));
}
void Renderer::set_translation_matrix(float *mat, float x, float y, float z)
{
    set_identity_matrix(mat, 4);
    mat[12] = x;
    mat[13] = y;
    mat[14] = z;
}
void Renderer::set_scale_matrix(float *mat, float sx, float sy, float sz)
{
    set_identity_matrix(mat, 4);
    mat[0] = sx;
    mat[5] = sy;
    mat[10] = sz;
}
void Renderer::set_rotation_matrix(float *mat, float angle, float x, float y, float z)
{
    float radAngle = deg_2_rad(angle);
    float co = cos(radAngle);
    float si = sin(radAngle);
    float x2 = x * x;
    float y2 = y * y;
    float z2 = z * z;

    mat[0] = x2 + (y2 + z2) * co;
    mat[4] = x * y * (1 - co) - z * si;
    mat[8] = x * z * (1 - co) + y * si;
    mat[12] = 0.0f;

    mat[1] = x * y * (1 - co) + z * si;
    mat[5] = y2 + (x2 + z2) * co;
    mat[9] = y * z * (1 - co) - x * si;
    mat[13] = 0.0f;

    mat[2] = x * z * (1 - co) - y * si;
    mat[6] = y * z * (1 - co) + x * si;
    mat[10] = z2 + (x2 + y2) * co;
    mat[14] = 0.0f;

    mat[3] = 0.0f;
    mat[7] = 0.0f;
    mat[11] = 0.0f;
    mat[15] = 1.0f;
}
void Renderer::set_model_matrix()
{
    gl::glBindBuffer((gl::GLenum)GL_UNIFORM_BUFFER, _matrices_uni_buffer);
    gl::glBufferSubData((gl::GLenum)GL_UNIFORM_BUFFER, ModelMatrixOffset, MatrixSize, _model_matrix);
    gl::glBindBuffer((gl::GLenum)GL_UNIFORM_BUFFER, 0);
}
void Renderer::translate(float x, float y, float z)
{
    float aux[16];

    set_translation_matrix(aux, x, y, z);
    mult_matrix(_model_matrix, aux);
    set_model_matrix();
}
void Renderer::rotate(float angle, float x, float y, float z)
{
    float aux[16];

    set_rotation_matrix(aux, angle, x, y, z);
    mult_matrix(_model_matrix, aux);
    set_model_matrix();
}
void Renderer::scale(float x, float y, float z)
{
    float aux[16];

    set_scale_matrix(aux, x, y, z);
    mult_matrix(_model_matrix, aux);
    set_model_matrix();
}
void Renderer::build_projection_matrix(float fov, float ratio, float nearp, float farp)
{
    float projMatrix[16];

    float f = 1.0f / tan(fov * (M_PI / 360.0f));

    set_identity_matrix(projMatrix, 4);

    projMatrix[0] = f / ratio;
    projMatrix[1 * 4 + 1] = f;
    projMatrix[2 * 4 + 2] = (farp + nearp) / (nearp - farp);
    projMatrix[3 * 4 + 2] = (2.0f * farp * nearp) / (nearp - farp);
    projMatrix[2 * 4 + 3] = -1.0f;
    projMatrix[3 * 4 + 3] = 0.0f;

    gl::glBindBuffer((gl::GLenum)GL_UNIFORM_BUFFER, _matrices_uni_buffer);
    gl::glBufferSubData((gl::GLenum)GL_UNIFORM_BUFFER, ProjMatrixOffset, MatrixSize, projMatrix);
    gl::glBindBuffer((gl::GLenum)GL_UNIFORM_BUFFER, 0);

    /*std::cout << std::endl;
    std::cout << projMatrix[0] << std::endl;
    std::cout << projMatrix[1] << std::endl;
    std::cout << projMatrix[2] << std::endl;
    std::cout << projMatrix[3] << std::endl;
    std::cout << projMatrix[4] << std::endl;
    std::cout << projMatrix[5] << std::endl;
    std::cout << projMatrix[6] << std::endl;
    std::cout << projMatrix[7] << std::endl;
    std::cout << projMatrix[8] << std::endl;
    std::cout << projMatrix[9] << std::endl;
    std::cout << projMatrix[10] << std::endl;
    std::cout << projMatrix[11] << std::endl;
    std::cout << projMatrix[12] << std::endl;
    std::cout << projMatrix[13] << std::endl;
    std::cout << projMatrix[14] << std::endl;
    std::cout << projMatrix[15] << std::endl;
    std::cout << std::endl;*/
}
void Renderer::set_camera(float posX, float posY, float posZ, float lookAtX, float lookAtY, float lookAtZ)
{
    float dir[3], right[3], up[3];

    up[0] = 0.0f;
    up[1] = 1.0f;
    up[2] = 0.0f;

    dir[0] = (lookAtX - posX);
    dir[1] = (lookAtY - posY);
    dir[2] = (lookAtZ - posZ);
    normalize(dir);

    cross_product(dir, up, right);
    normalize(right);

    cross_product(right, dir, up);
    normalize(up);

    float viewMatrix[16], aux[16];

    viewMatrix[0] = right[0];
    viewMatrix[4] = right[1];
    viewMatrix[8] = right[2];
    viewMatrix[12] = 0.0f;

    viewMatrix[1] = up[0];
    viewMatrix[5] = up[1];
    viewMatrix[9] = up[2];
    viewMatrix[13] = 0.0f;

    viewMatrix[2] = -dir[0];
    viewMatrix[6] = -dir[1];
    viewMatrix[10] = -dir[2];
    viewMatrix[14] = 0.0f;

    viewMatrix[3] = 0.0f;
    viewMatrix[7] = 0.0f;
    viewMatrix[11] = 0.0f;
    viewMatrix[15] = 1.0f;

    set_translation_matrix(aux, -posX, -posY, -posZ);

    mult_matrix(viewMatrix, aux);

    gl::glBindBuffer((gl::GLenum)GL_UNIFORM_BUFFER, _matrices_uni_buffer);
    gl::glBufferSubData((gl::GLenum)GL_UNIFORM_BUFFER, ViewMatrixOffset, MatrixSize, viewMatrix);
    gl::glBindBuffer((gl::GLenum)GL_UNIFORM_BUFFER, 0);

    /*std::cout << std::endl;
    std::cout << viewMatrix[0] << std::endl;
    std::cout << viewMatrix[1] << std::endl;
    std::cout << viewMatrix[2] << std::endl;
    std::cout << viewMatrix[3] << std::endl;
    std::cout << viewMatrix[4] << std::endl;
    std::cout << viewMatrix[5] << std::endl;
    std::cout << viewMatrix[6] << std::endl;
    std::cout << viewMatrix[7] << std::endl;
    std::cout << viewMatrix[8] << std::endl;
    std::cout << viewMatrix[9] << std::endl;
    std::cout << viewMatrix[10] << std::endl;
    std::cout << viewMatrix[11] << std::endl;
    std::cout << viewMatrix[12] << std::endl;
    std::cout << viewMatrix[13] << std::endl;
    std::cout << viewMatrix[14] << std::endl;
    std::cout << viewMatrix[15] << std::endl;
    std::cout << std::endl;*/
}
void Renderer::cross_product(float *a, float *b, float *res)
{
    res[0] = a[1] * b[2] - b[1] * a[2];
    res[1] = a[2] * b[0] - b[2] * a[0];
    res[2] = a[0] * b[1] - b[0] * a[1];
}
void Renderer::normalize(float *a)
{
    float mag = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);

    a[0] /= mag;
    a[1] /= mag;
    a[2] /= mag;
}
glm::vec3 &Renderer::get_translation() { return _translation; }
void Renderer::set_color() { _mode = 0; }
void Renderer::set_depth() { _mode = 1; }
