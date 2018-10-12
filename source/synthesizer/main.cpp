#include <iostream>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <glbinding/Binding.h>
#include <glbinding/ContextInfo.h>
#include <glbinding/Version.h>
#include <glbinding/callbacks.h>

#include <glbinding/gl/gl.h>

#include <globjects/Buffer.h>
#include <globjects/Query.h>
#include <globjects/base/File.h>
#include <globjects/globjects.h>

#include "Choreographer.h"
#include "Controller.h"
#include "Renderer.h"

#include <assimp/scene.h>

using namespace gl;
using namespace glbinding;

void error(int errnum, const char *errmsg) { std::cerr << errnum << ": " << errmsg << std::endl; }

struct WindowPointer
{
    Renderer *_renderer;
};

void size_callback(GLFWwindow *window, int width, int height)
{
    WindowPointer *ptr = (WindowPointer *)glfwGetWindowUserPointer(window);
    ptr->_renderer->resize(width, height);
}

void key_callback(GLFWwindow *window, int key, int /*scancode*/, int action, int /*mods*/)
{
    if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, 1);

    if(action == GLFW_PRESS || action == GLFW_REPEAT)
    {
        glm::vec3 direction(0., 0.f, 1.f);
        glm::vec3 right(0.f, 1.f, 0.f);
        float speed = 0.1f;

        WindowPointer *ptr = (WindowPointer *)glfwGetWindowUserPointer(window);

        switch(key)
        {
        case GLFW_KEY_W:
        {
            ptr->_renderer->get_translation() += direction * speed;
        }
        break;
        case GLFW_KEY_S:
        {
            ptr->_renderer->get_translation() -= direction * speed;
        }
        break;
        case GLFW_KEY_A:
        {
            ptr->_renderer->get_translation() -= right * speed;
        }
        break;
        case GLFW_KEY_D:
        {
            ptr->_renderer->get_translation() += right * speed;
        }
        break;
        case GLFW_KEY_1:
        {
            ptr->_renderer->set_color();
        }
        break;
        case GLFW_KEY_2:
        {
            ptr->_renderer->set_depth();
        }
        break;
        case GLFW_KEY_3:
        {
            ptr->_renderer->set_grayscale();
        }
        break;
        case GLFW_KEY_4:
        {
            ptr->_renderer->set_opticflow();
        }
        break;
        }
    }
}

int main(int, char *[])
{
    if(!glfwInit())
        return 1;

    glfwSetErrorCallback(error);

    glfwDefaultWindowHints();

    GLFWwindow *window = glfwCreateWindow(1280, 1080, "Video Avatars RGB-D Synthesizer", nullptr, nullptr);
    if(!window)
    {
        glfwTerminate();
        return -1;
    }

    glfwSetKeyCallback(window, key_callback);
    glfwMakeContextCurrent(window);

    globjects::init();

    setAfterCallback([](const FunctionCall &) {
        gl::GLenum error = gl::glGetError();
        if(error != GL_NO_ERROR)
            std::cout << "error: " << error << std::endl;
    });

    std::cout << std::endl
              << "OpenGL Version:  " << ContextInfo::version() << std::endl
              << "OpenGL Vendor:   " << ContextInfo::vendor() << std::endl
              << "OpenGL Renderer: " << ContextInfo::renderer() << std::endl;

    // std::string filename_poi("/home/fusion_4d/Desktop/data/synthetic_dataset/model-triangulated.obj");
    // std::string filename_poi("/home/fusion_4d/Desktop/data/synthetic_dataset/breakers.obj");
    // std::string filename_poi("/home/fusion_4d/Desktop/data/synthetic_dataset/lion-simple.obj");
    // std::string filename_poi("/home/fusion_4d/Desktop/data/synthetic_dataset/cube.obj");
    // std::string filename_poi("/home/fusion_4d/Desktop/data/synthetic_dataset/sphere.obj");
    std::string filename_poi("/home/fusion_4d/Desktop/data/synthetic_dataset/uv_unwrapped_egg.obj");
    // std::string filename_environment("/home/fusion_4d/Desktop/data/synthetic_dataset/emptyworldground.dae");
    std::string filename_environment("/home/fusion_4d/Desktop/data/synthetic_dataset/environment.dae");

    Controller controller(filename_poi, filename_environment);
    FrameSequencer sequencer(FrameSequencer::Type::INCREASING_STEP, 0, 1);
    Choreographer choreographer(&sequencer);
    choreographer.set_translation({0.05, 0., 0.05});
    // choreographer.set_rotation(glm::radians(90.), {0., 1., 0.});
    Renderer renderer(&controller, &choreographer, &sequencer);

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    renderer.resize(width, height);

    glfwSetWindowSizeCallback(window, size_callback);

    WindowPointer *ptr = new WindowPointer();
    ptr->_renderer = &renderer;

    glfwSetWindowUserPointer(window, ptr);

    glEnable(GL_MULTISAMPLE);
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    // glEnable(GL_CULL_FACE);
    // glFrontFace(GL_CCW);
    glDisable(GL_BLEND);

    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        if(!sequencer.is_finished())
        {
            renderer.draw();
        }
        else
        {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
        glfwSwapBuffers(window);
    }

    globjects::detachAllObjects();

    glfwTerminate();
    return 0;
}
