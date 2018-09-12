#include "Choreographer.h"
#include <glm/gtc/quaternion.hpp>
void Choreographer::set_translation(glm::dvec3 translation) { _translation = translation; }
void Choreographer::set_rotation(double angle, glm::dvec3 axis)
{
    _angle = angle;
    _axis = axis;
}

glm::dvec3 Choreographer::get_translation(int frame)
{
    double step = (double)frame / (double)_frame_cycle;
    return _translation * step;
}
glm::dvec4 Choreographer::get_rotation(int frame)
{
    double step = (double)frame / (double)_frame_cycle;
    glm::dquat rot_min = glm::angleAxis(-_angle / 2., _axis);
    glm::dquat rot_max = glm::angleAxis(_angle / 2., _axis);

    glm::dquat rot_result = glm::lerp(rot_min, rot_max, step);
    glm::vec3 axis = glm::axis(rot_result);

    glm::dvec4 result;
    result.x = glm::angle(rot_result);
    result.y = axis.x;
    result.z = axis.y;
    result.w = axis.z;

    return result;
}
