#ifndef SYNTHETIC_RGBD_RENDERER_H
#define SYNTHETIC_RGBD_RENDERER_H

#include <glm/glm.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/projection.hpp>

#include <glbinding/Binding.h>
#include <glbinding/ContextInfo.h>
#include <glbinding/Version.h>
#include <glbinding/callbacks.h>

#include <glbinding/gl/gl.h>

#include <globjects/Buffer.h>
#include <globjects/Query.h>
#include <globjects/Texture.h>
#include <globjects/UniformBlock.h>
#include <globjects/VertexArray.h>
#include <globjects/VertexAttributeBinding.h>
#include <globjects/base/File.h>
#include <globjects/globjects.h>

#include <GL/gl.h>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <fstream>
#include <map>
#include <math.h>
#include <string>
#include <vector>

#include "FileBuffer.hpp"
#include "Choreographer.h"

#define MatricesUniBufferSize sizeof(float) * 16 * 3
#define ProjMatrixOffset 0
#define ViewMatrixOffset sizeof(float) * 16
#define ModelMatrixOffset sizeof(float) * 16 * 2
#define MatrixSize sizeof(float) * 16

class Controller;

class Renderer
{
  public:
    explicit Renderer(Controller *controller, Choreographer*choreographer);
    ~Renderer();

    void resize(int width, int height);
    void draw();

    glm::vec3 &get_translation();

    struct CameraDescriptor
    {
        CameraDescriptor(glm::vec3 pos, glm::vec3 look_at, glm::vec3 up)
        {
            this->pos = pos;
            this->look_at = look_at;
            this->up = up;
        }
        glm::vec3 pos;
        glm::vec3 look_at;
        glm::vec3 up;
    };

    struct MeshDescriptor
    {
        GLuint vao;
        GLuint texIndex;
        GLuint uniformBlockIndex;
        int numFaces;
    };

    struct MaterialDescriptor
    {
        float diffuse[4];
        float ambient[4];
        float specular[4];
        float emissive[4];
        float shininess;
        int texCount;
    };

    void set_color();
    void set_depth();

  private:
    FileBuffer * _fb;
    unsigned char* _frame_color;
    float* _frame_depth;

    int _mode = 0;
    int _width = 1, _height = 1;
    glm::vec3 _translation{0.f}; //{8.60003f, 11.4501f, -6.89923f};
    std::vector<CameraDescriptor> _camera_descriptor;

    GLuint _matrices_uni_buffer;

    float _model_matrix[16];
    std::vector<float *> _matrix_stack;

    globjects::Buffer *_buffer_fsquad_debug;
    globjects::VertexArray *_vao_fsquad_debug;

    globjects::Framebuffer *_fbo, *_fbo_color, *_fbo_depth;
    globjects::Texture *_texture_color;
    globjects::Texture *_texture_depth;
    globjects::Texture *_texture_color_postproc;
    globjects::Texture *_texture_depth_postproc;
    globjects::Texture *_texture_dummy_depth_1280, *_texture_dummy_depth_512;

    globjects::UniformBlock *_ub_matrices;
    globjects::UniformBlock *_ub_materials;
    Controller *_controller;
    Choreographer *_choreographer;
    globjects::Program *_program, *_program_postproc_color, *_program_postproc_depth, *_program_debug_texture;

    std::vector<MeshDescriptor> _meshes;
    std::map<std::string, GLuint> _texture_id_map;

    static inline float deg_2_rad(float degrees) { return (float)(degrees * (M_PI / 180.0f)); };

    void set_float4(float f[4], float a, float b, float c, float d);
    void color4_to_float4(const aiColor4D *c, float f[4]);

    void cross_product(float *a, float *b, float *res);
    void normalize(float *a);

    void push_matrix();
    void pop_matrix();
    void set_identity_matrix(float *mat, int size);
    void mult_matrix(float *a, float *b);
    void set_translation_matrix(float *mat, float x, float y, float z);
    void set_scale_matrix(float *mat, float sx, float sy, float sz);
    void set_rotation_matrix(float *mat, float angle, float x, float y, float z);
    void set_model_matrix();
    void translate(float x, float y, float z);
    void rotate(float angle, float x, float y, float z);
    void scale(float x, float y, float z);
    void build_projection_matrix(float fov, float ratio, float nearp, float farp);
    void set_camera(float posX, float posY, float posZ, float lookAtX, float lookAtY, float lookAtZ, float upX, float upY, float upZ);

    void gen_VAOs_and_uniform_buffer(const aiScene *sc);
    void recursive_render(const aiScene *sc, const aiNode *nd, int offset = 0);
};

#endif // SYNTHETIC_RGBD_RENDERER_H
