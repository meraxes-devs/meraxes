#ifndef EMISSION_LINES_H
#define EMISSION_LINES_H

#include "meraxes.h"

#define COLD_GAS_SOUND_SPEED_CMS_SQ 1e12 // [cm^2/s^2]
// [O III] Einstein coefficients [1/s] from the NIST Atomic Spectra Database, with
// levels 1 = 3P1, 2 = 3P2, 3 = 1D2 and 4 = 1S0.
#define LOIII_A31 6.21e-3                  // 1D2 -> 3P1, [O III] 4959
#define LOIII_A32 1.81e-2                  // 1D2 -> 3P2, [O III] 5007
#define LOIII_A41 2.15e-1                  // 1S0 -> 3P1, [O III] 2321
#define LOIII_A43 1.71                     // 1S0 -> 1D2, [O III] 4363
#define nu32 SPEED_OF_LIGHT * 1e8 / 5008.0 // Hz

#ifdef __cplusplus
extern "C"
{
#endif

  void set_OIII_coeffs(void);
  double oiii_electron_temperature(double metallicity);
  double case_b_recombination_coefficient(double temperature);
  double oiii_excitation_rate(double temperature);
  void compute_LOIII(struct galaxy_t* gal, double sfr, double metallicity, int snapshot);

#ifdef __cplusplus
}
#endif

#endif
