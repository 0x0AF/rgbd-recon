#ifndef PADDED_CALIB_VOLUME_HPP
#define PADDED_CALIB_VOLUME_HPP

#include "calibration_volume.hpp"

namespace kinect
{
class PaddedCalibrationVolume : public CalibrationVolume<glm::fvec4>
{
  public:
    PaddedCalibrationVolume(std::string const &filename) : CalibrationVolume(filename, true) { read(filename); };

  protected:
    void read(std::string const &filename)
    {
        FILE *file_input = fopen(filename.c_str(), "rb");
        std::size_t res = 0;
        res = fread(&m_resolution.x, sizeof(unsigned), 1, file_input);
        assert(res == 1);
        res = fread(&m_resolution.y, sizeof(unsigned), 1, file_input);
        assert(res == 1);
        res = fread(&m_resolution.z, sizeof(unsigned), 1, file_input);
        assert(res == 1);
        res = fread(&m_depth_limits.x, sizeof(float), 1, file_input);
        assert(res == 1);
        res = fread(&m_depth_limits.y, sizeof(float), 1, file_input);
        assert(res == 1);

        m_volume.resize(m_resolution.x * m_resolution.y * m_resolution.z);

        for(size_t i = 0; i < m_resolution.x * m_resolution.y * m_resolution.z; i++)
        {
            glm::fvec3 data;
            res = fread(&data, sizeof(glm::fvec3), 1, file_input);

            m_volume.at(i).r = data.r;
            m_volume.at(i).g = data.g;
            m_volume.at(i).b = data.b;
            m_volume.at(i).a = 0.f;

            // printf("\nm_volume.data()[%lu]: (%f,%f,%f,%f)\n", i, m_volume.data()[i].r, m_volume.data()[i].g, m_volume.data()[i].b, m_volume.data()[i].a);

            assert(res == 4);
        }

        fclose(file_input);
    }
};
} // namespace kinect
#endif