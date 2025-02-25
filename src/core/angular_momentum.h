#ifndef ANGULAR_MOMENTUM_H
#define ANGULAR_MOMENTUM_H

#include <stdbool.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C"
{
#endif

  void increment_angular_momentum(double *reservoir, double *angmom,
                                double mass);
  void total_to_specific_angmom(double *total, double mass, double *specific);
  void specific_to_total_angmom(double *specific, double mass, double *total);
  void add_disks(galaxy_t *gal, int gas, double new_mass, double new_rad,
               double new_vel, double *new_am);                              

#ifdef __cplusplus
}
#endif

#endif
