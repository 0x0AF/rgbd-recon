#ifndef RGBD_RECON_CHOREOGRAPHER_H
#define RGBD_RECON_CHOREOGRAPHER_H

#include <glm/vec3.hpp>

class Choreographer
{
  public:
    Choreographer() = default;
    ~Choreographer() = default;

    glm::dvec3 get_translation(int frame);

    /// angle + axis
    glm::dvec4 get_rotation(int frame);

    void set_translation(glm::dvec3 translation);
    void set_rotation(double angle, glm::dvec3 axis);

  private:
    int _frame_cycle = 100;
    glm::dvec3 _translation;
    double _angle;
    glm::dvec3 _axis;
};

#endif // RGBD_RECON_CHOREOGRAPHER_H