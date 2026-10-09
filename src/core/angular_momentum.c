#include "angular_momentum.h"
#include "meraxes.h"
#include "misc_tools.h"
#include <math.h>

// "If incrementing total angular momentum then set `mass=1`.
// Otherwise `angmom` is interpreted as specific angular momentum."
void increment_angular_momentum(double* reservoir, double* angmom, double mass)
{
  for (int ii = 0; ii < 3; ii++)
    reservoir[ii] += angmom[ii] * mass;
}

void total_to_specific_angmom(double* total, double mass, double* specific)
{
  if (mass <= 0) {
    for (int ii = 0; ii < 3; ii++)
      specific[ii] = 0;
  } else {
    for (int ii = 0; ii < 3; ii++)
      specific[ii] = total[ii] / mass;
  }
}

void specific_to_total_angmom(double* specific, double mass, double* total)
{
  for (int ii = 0; ii < 3; ii++)
    total[ii] = specific[ii] * mass;
}

double calculate_spin_param(halo_t* halo)
{
#if USE_ANG_MOM
  float spin;
  spin =
    sqrt(halo->AngMom[0] * halo->AngMom[0] + halo->AngMom[1] * halo->AngMom[1] + halo->AngMom[2] * halo->AngMom[2]);
  spin = spin / (1.414213562 * halo->Vvir * halo->Rvir);
  return (double)spin;
#else
  return halo->AngMom / (1.414213562 * halo->Vvir * halo->Rvir);
#endif
}

#if USE_ANG_MOM
void add_disks(galaxy_t* gal, int gas, double new_mass, double new_rad, double new_vel, double* new_am)
{
  // new_am is total AM!
  if ((gas == 0) && (gal->StellarDiskScaleLength < 1e-10)) {
    gal->VStellarDisk = new_vel;
    gal->StellarDiskScaleLength = new_rad;
  } else if ((gas == 1) && (gal->DiskScaleLength < 1e-10)) {
    gal->VGasDisk = new_vel;
    gal->DiskScaleLength = new_rad;
  } else if (((gas == 0) && (gal->StellarDiskScaleLength == new_rad)) ||
             ((gas == 1) && (gal->DiskScaleLength == new_rad)) || (fabs(new_mass) < 1e-10))
    return;
  else if (fabs(new_mass) >= 1e-10) {
    double AMcombined[3];
    if ((gas == 0) && (fabs(new_mass + gal->StellarMass) >= 1e-10)) {
      for (int ii = 0; ii < 3; ii++) {
        AMcombined[ii] = gal->AMstars[ii] + ((new_mass > 0) - (new_mass < 0)) * new_am[ii];
      }
      double disk_mass = gal->StellarMass;
      double mplusm = disk_mass + new_mass;
      // v_final is Eq. 12 in Maddie's paper for Vnew
      double v_final = pow((disk_mass * pow(gal->VStellarDisk, 2) + new_mass * pow(new_vel, 2)) / mplusm, 0.5);
      // the one below is Eq. 13 in Maddie's paper for Rnew
      gal->StellarDiskScaleLength = vector_magnitude(AMcombined) / (2 * mplusm * v_final);
      gal->VStellarDisk = v_final;
    } else if ((gas == 1) && (fabs(new_mass + gal->ColdGas) >= 1e-10)) {
      for (int ii = 0; ii < 3; ii++) {
        // sum each vector component of am
        AMcombined[ii] = gal->AMcold[ii] + ((new_mass > 0) - (new_mass < 0)) * new_am[ii];
      }
      double mplusm = gal->ColdGas + new_mass;
      // Eq. 12
      double v_final = pow((gal->ColdGas * pow(gal->VGasDisk, 2) + new_mass * pow(new_vel, 2)) / mplusm, 0.5);
      // Eq. 13
      gal->DiskScaleLength = vector_magnitude(AMcombined) / (2 * mplusm * v_final);
      gal->VGasDisk = v_final;
    }
  }
}
#endif
