#ifndef KINECT_CALIBVOLUME_H
#define KINECT_CALIBVOLUME_H

#include <DataTypes.h>
#include "volume_sampler.hpp"

#include <string>
#include <vector>

#include <globjects/Buffer.h>
#include <globjects/Texture.h>
#include <globjects/Program.h>

#include "gloost/BoundingBox.h"

namespace kinect{

class CalibVolume{

public:
  CalibVolume(std::vector<std::string> const& calib_volume_files, gloost::BoundingBox const& bbox);
  virtual ~CalibVolume();

  void reload();
  
  void setStartTextureUnit(unsigned start_texture_unit);
  void setStartTextureUnitInv(unsigned start_texture_unit);
  // unsigned getStartTextureUnit() const;

  std::vector<int> getXYZVolumeUnits() const;
  std::vector<int> getUVVolumeUnits() const;

  std::vector<int> getXYZVolumeUnitsInv() const;
  std::vector<int> getUVVolumeUnitsInv() const;

  void calculateInverseVolumes();

  glm::uvec3 getVolumeRes() const;

private:
  void bindToTextureUnits();
  void bindToTextureUnitsInv();

  std::vector<std::string> m_cv_xyz_filenames;
  std::vector<std::string> m_cv_uv_filenames;

  std::vector<globjects::Texture*> m_volumes_xyz;
  std::vector<globjects::Texture*> m_volumes_uv;

  std::vector<globjects::Texture*> m_volumes_xyz_inv;
  std::vector<globjects::Texture*> m_volumes_uv_inv;

  globjects::Program* m_program;
  VolumeSampler       m_sampler;


  gloost::BoundingBox m_bbox;

public:
  std::vector<float> m_cv_min_ds;
  std::vector<float> m_cv_max_ds;
  std::vector<unsigned> m_cv_widths;
  std::vector<unsigned> m_cv_heights;
  std::vector<unsigned> m_cv_depths;

 protected:
  int m_start_texture_unit;
  int m_start_texture_unit_inv;

  void addVolume(std::string const& filename_xyz, std::string filename_uv);
};

}

#endif // #ifndef KINECT_CALIBVOLUME_H