#ifndef BLACKHOLE_FEEDBACK_H
#define BLACKHOLE_FEEDBACK_H

#include "meraxes.h"

#define ETA 0.06
// standard efficiency, 6% accreted mass is radiated

#define BONDI_HOYLE_COEFFICIENT 3.4754
// 15/8*pi*mu=3.4754, with mu=0.59; x=k*m_p*t/lambda
#define VELOCITY_SCALE 280.0

#define EMISSIVITY_CONVERTOR 8.40925088e-8
//=1e60 * PROTONMASS / 1e10 / SOLAR_MASS
// Unit_convertor is to convert emissivity from 1e60 photons to photons per atom

#define LUMINOSITY_CONVERTOR 32886.5934
// = (1 solar mass *(speed of light)^2/450e6year) /3.828e26watt
// where 450e6year is eddington time scale

#define LUV2EMISSIVITY 2.79387171897e-6
// firstly UV 1450 band luminosity is:   LUV = Lbol/kuv 1e10Lsun
// then Lnu(1450) = LUV * 1e10Lsun / nu_1450 = 1.851239e28 erg/s/Hz LUV
// then Lnu(912) = Lnu(1450) * (912/1450)**SpecIndexUVAGNSoft
// then dot Nion =  Lnu(912) / planck constant (6.62607015e-27 erg/Hz) / SpecIndexUVAGNHard
// so LUV2EMISSIVITY = 1.851239e28 / 6.62607015e-27 / SpecIndexUVAGNHard * (912/1450)**SpecIndexUVAGNSoft
//  = 2.79387171897e54 / SpecIndexUVAGNHard * (912/1450)**SpecIndexUVAGNSoft 
// finally divided by 1e60 to convert to 1e60 photons

/* Morrison & McCammon (1983) Table 2: photoelectric cross-section coefficients.
 * σ(E) = (C0 + C1·E + C2·E²) × E^{-3} × 10^{-24} cm²,  E in keV.
 * Columns: E_low [keV], E_high [keV], C0, C1, C2 */
static const double mm83[14][5] = {
  { 0.030,  0.100,  17.3,    608.1,  -2150.0 },
  { 0.100,  0.284,  34.6,    267.9,   -476.1 },
  { 0.284,  0.400,  78.1,     18.8,      4.3 },
  { 0.400,  0.532,  71.4,     66.8,    -51.4 },
  { 0.532,  0.707,  95.5,    145.8,    -61.1 },
  { 0.707,  0.867, 308.9,   -380.6,    294.0 },
  { 0.867,  1.303, 120.6,    169.3,    -47.7 },
  { 1.303,  1.840, 141.3,    146.8,    -31.5 },
  { 1.840,  2.471, 202.7,    104.7,    -17.0 },
  { 2.471,  3.210, 342.7,     18.7,      0.0 },
  { 3.210,  4.038, 352.2,     18.7,      0.0 },
  { 4.038,  7.111, 433.9,     -2.4,      0.75},
  { 7.111,  8.331, 629.0,     30.9,      0.0 },
  { 8.331, 10.000, 701.2,     25.2,      0.0 },
};

#define OBS_EPSILON   1.7     /* ratio logNH=23-24 to logNH=22-23 quasars  */
/* ε = 1.7 means there are 1.7× more AGN
in the logNH=23–24 bin than the logNH=22–23 bin (the absorbed bins are not equal).*/
#define OBS_FCTK      1.0     /* CTK fraction relative to absorbed CTN      */
/*fCTK = 1.0 means the total CTK population equals the total absorbed CTN population.*/
#define OBS_PSI_MIN   0.20    /* minimum absorbed fraction                  */
#define OBS_PSI_MAX   0.84    /* maximum absorbed fraction                  */
#define OBS_LX_REF    43.75   /* reference log10(LX/erg/s) for psi(LX,z)   */



#ifdef __cplusplus
extern "C"
{
#endif
  double radio_mode_BH_heating(struct galaxy_t* gal, double cooling_mass, double x);

  void calculate_BHemissivity(double BlackHoleMass, double accreted_mass,
                              double *emissivity,     double *accretion_time,
                              double *quasar_luv,     double *quasar_lx,
                              double *quasar_lx_soft);

  void merger_driven_BH_growth(struct galaxy_t* gal, double merger_ratio, int snapshot);
  void previous_merger_driven_BH_growth(struct galaxy_t* gal, int snapshot);
  void get_nh_transmission(double T_out[5]);
  void get_nh_fracs(double LX_1e10Lsun, double redshift, double f_out[5]);
  /* Builds the s_T_hard/s_T_soft NH-transmission tables once. Must be called during startup (init_meraxes(), after params are loaded) */
  void init_xray_obscuration_tables(void);


#ifdef __cplusplus
}
#endif

#endif
