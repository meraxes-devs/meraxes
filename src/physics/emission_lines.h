#ifndef EMISSION_LINES_H
#define EMISSION_LINES_H

#include "meraxes.h"

#define COLD_GAS_SOUND_SPEED_CMS_SQ 1e12 // [cm^2/s^2]
#define LOIII_A31 4.57e-6
#define LOIII_A32 3.52e-5
#define LOIII_A41 2.15e-1
#define LOIII_A43 1.7
#define nu32 SPEED_OF_LIGHT * 1e8 / 5008.0 // Hz

#ifdef __cplusplus
extern "C"
{
#endif

  void set_OIII_coeffs(double T);
  void compute_LOIII(struct galaxy_t* gal, int snapshot);

#ifdef __cplusplus
}
#endif

#endif
