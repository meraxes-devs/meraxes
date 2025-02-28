#include "meraxes.h"
#include <math.h>
#include "misc_tools.h"
#include "angular_momentum.h"

// If incrementing total angular momentum then set `mass=1`.
// Otherwise `angmom` is interpreted as specific angular momentum.
void increment_angular_momentum(double *reservoir, double *angmom,
                                double mass) {
  for (int ii = 0; ii < 3; ii++)
    reservoir[ii] += angmom[ii] * mass;
}

void total_to_specific_angmom(double *total, double mass, double *specific) {
  if (mass <= 0) {
    for (int ii = 0; ii < 3; ii++)
      specific[ii] = 0;
  } else {
    for (int ii = 0; ii < 3; ii++)
      specific[ii] = total[ii] / mass;
  }
}

void specific_to_total_angmom(double *specific, double mass, double *total) {
  for (int ii = 0; ii < 3; ii++)
    total[ii] = specific[ii] * mass;
}

#if USE_ANG_MOM
void add_disks(galaxy_t *gal, int gas, double new_mass, double new_rad,
               double new_vel, double *new_am) {
  if ((gas == 0) && (gal->StellarDiskScaleLength < 1e-10)) {
    // First time you form stars (see star_formation.c)
    gal->VStellarDisk = new_vel;
    gal->StellarDiskScaleLength = new_rad;
  } else if ((gas == 1) && (gal->DiskScaleLength < 1e-10)) {
    // First time you cool down the gas (see cooling.c)!
    gal->VGasDisk = new_vel;
    gal->DiskScaleLength = new_rad;
  } else if (((gas == 0) && (gal->StellarDiskScaleLength == new_rad)) ||
             ((gas == 1) && (gal->DiskScaleLength == new_rad)) ||
             (new_mass == 0))
      return;
  else if (new_mass != 0) {
    double AMcombined[3];
    if ((gas == 0) & (new_mass + gal->StellarMass > 1e-10)) {
      for (int ii = 0; ii < 3; ii++) {
        AMcombined[ii] =
            gal->AMstars[ii] + ((new_mass > 0) - (new_mass < 0)) * new_am[ii];
      }
      double disk_mass = gal->StellarMass;
      double mplusm = disk_mass + new_mass;
      // v_final is Eq. 12 in Maddie's paper for Vnew
      double v_final = pow(
          (disk_mass * pow(gal->VStellarDisk, 2) + new_mass * pow(new_vel, 2)) /
              mplusm,
          0.5);
      // the one below is Eq. 13 in Maddie's paper for Rnew
      gal->StellarDiskScaleLength =
          vector_magnitude(AMcombined) / (2 * mplusm * v_final);
      gal->VStellarDisk = v_final;
    } else if ((gas == 1) && (new_mass + gal->ColdGas != 0)) {
      // You have already cool down gas once (see cooling.c)
      // You enter here also when you form stars (see star_formation.c)
      // and you are removing angular momentum from gas disk
      // You enter here also in Sn feedback (new gas)
      for (int ii = 0; ii < 3; ii++) {
       // sum each vector component of am
        AMcombined[ii] =
            gal->AMcold[ii] + ((new_mass > 0) - (new_mass < 0)) * new_am[ii];
      }
      double mplusm = gal->ColdGas + new_mass;
      //Eq. 12
      double v_final = pow(
          (gal->ColdGas * pow(gal->VGasDisk, 2) + new_mass * pow(new_vel, 2)) /
              mplusm,
          0.5);
      //Eq. 13    
      gal->DiskScaleLength = vector_magnitude(AMcombined) /
                                (2 * mplusm * v_final);
      gal->VGasDisk = v_final;
    }
  }
}
#endif         
