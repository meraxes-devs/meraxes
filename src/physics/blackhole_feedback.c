#include "blackhole_feedback.h"
#include "core/misc_tools.h"
#include "meraxes.h"
#include <assert.h>
#include <gsl/gsl_cdf.h>
#include <math.h>

/* Morrison & McCammon (1983) photoelectric absorption cross-section, cm^2/H atom. */
static double morrison_mccammon_sigma(double E_keV)
{
  double C0;
  double C1;
  double C2;
  int i;

  assert(E_keV >= 0.0);

  /* MM83 is only tabulated over 0.03-10 keV. */
  if (E_keV > 10.0)
    return 0.0;
  if (E_keV < 0.03)
    return 0.0;

  for (i = 0; i < 14; i++) {
    if (E_keV >= mm83[i][0] && E_keV < mm83[i][1]) {
      C0 = mm83[i][2];
      C1 = mm83[i][3];
      C2 = mm83[i][4];
      return (C0 + C1 * E_keV + C2 * E_keV * E_keV) * 1.0e-24 / pow(E_keV, 3);
    }
  }

  return 0.0;
}

#define N_E 200 /* energy samples for the integral below */

/* Band- and NH-averaged X-ray transmission, SED-weighted with photon index gamma. */
static double xray_transmission_at_NH(double log_NH, double E_min_keV, double E_max_keV, double gamma)
{
  double NH = pow(10.0, log_NH);
  double dE = (E_max_keV - E_min_keV) / N_E;
  double num = 0.0;
  double den = 0.0;
  double E, weight, sigma, tau, T;
  int i;

  for (i = 0; i < N_E; i++) {
    E = E_min_keV + (i + 0.5) * dE;
    weight = pow(E, -gamma);
    sigma = morrison_mccammon_sigma(E);
    tau = NH * sigma;
    T = exp(-tau);
    if (log_NH >= 24.0)
      T *= exp(-NH * 1.21 * 6.6524e-25);
    num += weight * T;
    den += weight;
  }
  return (den > 0.0) ? num / den : 0.0;
}

/* log-midpoint NH for each of the 5 bins */
static const double LOG_NH_MID[5] = { 20.5, 21.5, 22.5, 23.5, 25.0 };
/* edges of the 5 bins in log10(N_H / cm^-2); the last (Compton-thick) bin spans 2 dex */
static const double LOG_NH_LO[5] = { 20.0, 21.0, 22.0, 23.0, 24.0 };
static const double LOG_NH_HI[5] = { 21.0, 22.0, 23.0, 24.0, 26.0 };
static double s_T_hard[5];
static double s_T_soft[5];
static int s_T_init = 0;

/* Builds s_T_hard/s_T_soft. Called once from init_meraxes(), not per-AGN. */
void init_xray_obscuration_tables(void)
{
  int k;
  if (s_T_init)
    return;

  physics_params_t* physics = &(run_globals.params.physics);

  for (k = 0; k < 5; k++) {
    s_T_hard[k] = xray_transmission_at_NH(
      LOG_NH_MID[k], physics->NuXraySoftCut / 1000.0, physics->NuXrayMax / 1000.0, physics->SpecIndexXrayAGNHard);
    s_T_soft[k] = xray_transmission_at_NH(
      LOG_NH_MID[k], physics->NuXrayThreshold / 1000.0, physics->NuXraySoftCut / 1000.0, physics->SpecIndexXrayAGNSoft);
  }
  s_T_init = 1;
}

static double _psi_ref(double z)
{
  double zeff = (z < 2.0) ? z : 2.0;
  return 0.43 * pow(1.0 + zeff, 0.48);
}

static double _psi(double LX_log, double z) /* LX_log: log10(LX / erg s^-1) */
{
  double val = _psi_ref(z) - 0.24 * (LX_log - OBS_LX_REF);
  if (val < OBS_PSI_MIN)
    val = OBS_PSI_MIN;
  if (val > OBS_PSI_MAX)
    val = OBS_PSI_MAX;
  return val;
}

static void _NH_distribution(double LX_log, double z, double f[5]) /* LX_log: log10(LX / erg s^-1) */
{
  double p = _psi(LX_log, z);
  double eps = OBS_EPSILON;
  double thr = (1.0 + eps) / (3.0 + eps);
  double total_f;
  int i;

  if (p < thr) {
    f[0] = 1.0 - (2.0 + eps) / (1.0 + eps) * p;
    f[1] = 1.0 / (1.0 + eps) * p;
  } else {
    f[0] = 2.0 / 3.0 - (3.0 + 2.0 * eps) / (3.0 + 3.0 * eps) * p;
    f[1] = 1.0 / 3.0 - eps / (3.0 + 3.0 * eps) * p;
  }
  f[2] = 1.0 / (1.0 + eps) * p;
  f[3] = eps / (1.0 + eps) * p;
  f[4] = (OBS_FCTK / 2.0) * p;

  total_f = f[0] + f[1] + f[2] + f[3] + 2.0 * f[4];
  if (total_f <= 0.0) {
    for (i = 0; i < 5; i++)
      f[i] = 0.0;
    return;
  }
  for (i = 0; i < 5; i++)
    f[i] /= total_f;
}

static void apply_xray_obscuration(double LX_1e10Lsun,
                                   double redshift,
                                   double* obs_fraction_hard,
                                   double* obs_fraction_soft,
                                   int* NH_bin)
{
  double LX_log_ergs; /* log10(LX / erg s^-1) */
  double f[5];
  int k;

  *obs_fraction_hard = 0.0;
  *obs_fraction_soft = 0.0;
  *NH_bin = -1;

  if (LX_1e10Lsun <= 0.0)
    return;

  LX_log_ergs = log10(LX_1e10Lsun) + 10.0 + LOG_10_SOLAR_LUM; /* 1e10 Lsun -> log10 erg/s */
  _NH_distribution(LX_log_ergs, redshift, f);

  /* CDF; f[0..3] are 1 dex bins, 2*f[4] covers the 2-dex CTK bin. */
  double cdf[5];
  cdf[0] = f[0];
  cdf[1] = cdf[0] + f[1];
  cdf[2] = cdf[1] + f[2];
  cdf[3] = cdf[2] + f[3];
  cdf[4] = cdf[3] + 2.0 * f[4]; /* = 1.0 */

  double u = gsl_rng_uniform(run_globals.random_generator);
  int k_random = 4; /* default to CTK */
  for (k = 0; k < 5; k++) {
    if (u < cdf[k]) {
      k_random = k;
      break;
    }
  }

  /* only the drawn bin is nonzero, so just record which one */
  *obs_fraction_hard = s_T_hard[k_random];
  *obs_fraction_soft = s_T_soft[k_random];
  *NH_bin = k_random;
}

void get_nh_transmission(double T_out[5])
{
  int k;
  for (k = 0; k < 5; k++)
    T_out[k] = s_T_hard[k];
}

/* These are the probabilities from the NH distribution model — independent
 * of the stochastic draw made per galaxy. */
void get_nh_fracs(double LX_1e10Lsun, double redshift, double f_out[5])
{
  int k;
  double LX_log_ergs;
  double f[5];
  for (k = 0; k < 5; k++)
    f_out[k] = 0.0;
  if (LX_1e10Lsun <= 0.0)
    return;
  LX_log_ergs = log10(LX_1e10Lsun) + 10.0 + LOG_10_SOLAR_LUM;
  _NH_distribution(LX_log_ergs, redshift, f);
  for (k = 0; k < 5; k++)
    f_out[k] = f[k];
}

void get_nh_bin_edges(double lo[5], double hi[5])
{
  int k;
  for (k = 0; k < 5; k++) {
    lo[k] = LOG_NH_LO[k];
    hi[k] = LOG_NH_HI[k];
  }
}

/* ------------------------------------------------------------------------
 * Shen et al. (2020) Sec. 3.2: dust extinction of the AGN light, used instead
 * of the quasar_open_angle cone when Flag_AGNDustExtinction = 1 (UV LF in
 * save.c, ionising photons below) and for the QuasarLF_dust output. Each
 * AGN's N_H is converted to a dust
 * column with a dust-to-gas ratio: locally A_B / N_H = 8.47e-22 mag cm^2
 * (following Hopkins et al. 2007), scaled with redshift like the gas-phase
 * metallicity fit of Ma et al. (2016) at fixed stellar mass,
 *   DGR(z) / DGR(0) = 10^(0.93 [exp(-0.43 z) - 1])   (0.15 at z = 5, 0.13 at z = 7),
 * and a Milky-Way extinction curve (Pei 1992) for the wavelength dependence:
 *   A_1450 = (A_B / N_H) * N_H * (A_1450 / A_B) * DGR(z) / DGR(0),   A_1450 / A_B = 2.09.
 * The ionising photons are dimmed at the Lyman limit, A_912 / A_B = 4.04.
 * ------------------------------------------------------------------------ */
#define SHEN20_AB_PER_NH 8.47e-22   /* A_B / N_H [mag cm^2], local value */
#define SHEN20_LAMBDA_ION_UM 0.0912 /* Lyman limit [micron]: where the ionising photons are dimmed */

/* Pei (1992) ApJ 395, 130, Table 4, Milky Way: xi(lambda) = A_lambda / A_B, lambda in micron. */
static double pei92_mw_xi(double lambda_um)
{
  static const double a[6] = { 165.0, 14.0, 0.045, 0.002, 0.002, 0.012 };
  static const double l[6] = { 0.047, 0.08, 0.22, 9.7, 18.0, 25.0 };
  static const double b[6] = { 90.0, 4.0, -1.95, -1.95, -1.80, 0.0 };
  static const double n[6] = { 2.0, 6.5, 2.0, 2.0, 2.0, 2.0 };
  double xi;
  int i;

  xi = 0.0;
  for (i = 0; i < 6; i++)
    xi += a[i] / (pow(lambda_um / l[i], n[i]) + pow(l[i] / lambda_um, n[i]) + b[i]);
  return xi;
}

/* Ma et al. (2016) gas-metallicity scaling of the dust-to-gas ratio at fixed stellar mass, DGR(z) / DGR(0). */
static double shen20_dgr_ratio(double redshift)
{
  return pow(10.0, 0.93 * (exp(-0.43 * redshift) - 1.0));
}

double shen20_a1450_per_nh(double redshift, double* ab_per_nh, double* a1450_per_ab, double* dgr_ratio)
{
  double r_curve;
  double r_dgr;

  r_curve = pei92_mw_xi(0.145) / pei92_mw_xi(0.44); /* 1450 A relative to the B band (4400 A) */
  r_dgr = shen20_dgr_ratio(redshift);
  if (ab_per_nh != NULL)
    *ab_per_nh = SHEN20_AB_PER_NH;
  if (a1450_per_ab != NULL)
    *a1450_per_ab = r_curve;
  if (dgr_ratio != NULL)
    *dgr_ratio = r_dgr;
  return SHEN20_AB_PER_NH * r_curve * r_dgr;
}

/* Fraction of an AGN's ionising photons that escape when Flag_AGNDustExtinction = 1 (instead of the cone's
 * quasar_fobs): the Shen+20 dust transmission at the Lyman limit, 10^(-0.4 A_912) with A_912 / A_B = 4.04 (Pei92 MW
 * curve), averaged evenly in log N_H over the AGN's N_H bin (the bin drawn for its X-ray obscuration). Absorption by
 * the gas itself is not included. Tabulated once per redshift. */
static double agn_dust_escape_ionising(int nh_bin, double redshift)
{
  static double table_z = -1.0;
  static double table[5];
  double a_per_nh;
  double sum;
  int k;
  int j;

  if ((nh_bin < 0) || (nh_bin > 4))
    return 1.0; /* no N_H bin: this episode emitted nothing (L_X <= 0) */
  if (redshift != table_z) {
    a_per_nh = SHEN20_AB_PER_NH * pei92_mw_xi(SHEN20_LAMBDA_ION_UM) / pei92_mw_xi(0.44) * shen20_dgr_ratio(redshift);
    for (k = 0; k < 5; k++) {
      sum = 0.0;
      for (j = 0; j < AGN_DUST_NH_SAMPLES; j++)
        sum += pow(10.0,
                   -0.4 * a_per_nh *
                     pow(10.0, LOG_NH_LO[k] + (j + 0.5) * (LOG_NH_HI[k] - LOG_NH_LO[k]) / AGN_DUST_NH_SAMPLES));
      table[k] = sum / AGN_DUST_NH_SAMPLES;
    }
    table_z = redshift;
  }
  return table[nh_bin];
}

/* ------------------------------------------------------------------------
 * Shen et al. (2020) 1-sigma uncertainty of the bolometric corrections.
 *
 * Their Table 1 gives, per band, (sigma1, sigma2, logL0, sigma3) describing the
 * 1-sigma uncertainty of log10(Lbol/Lband) via their Eq. 4:
 *
 *   sigma(logLbol) = sigma2 + sigma1 * [ 1/2 + 1/2 erf( (logLbol - logL0)
 *                                                       / (sqrt(2) sigma3) ) ]
 *
 * with logLbol in erg/s. The shape is a double plateau: sigma -> sigma2 at low
 * Lbol and sigma -> sigma2 + sigma1 at high Lbol. For the hard band that is
 * 0.066 -> 0.259 dex (sigma GROWS toward bright); for the UV it is
 * 0.405 -> 0.033 dex (sigma SHRINKS toward bright, because at high luminosity
 * the UV carries most of Lbol and its correction is tightly pinned).
 *
 * Two uses: Flag_ShenBCScatter applies it per accretion episode as a random
 * lognormal factor on the correction; QuasarLFBCNSigma shifts every UV correction
 * by -/+ n sigma to give the QuasarLF_BC band written next to QuasarLF.
 * ------------------------------------------------------------------------ */
typedef struct shen_disp_t
{
  double sigma1;
  double sigma2;
  double logL0;
  double sigma3;
} shen_disp_t;

/* Shen et al. (2020), Table 1, last column. */
static const shen_disp_t SHEN_DISP_UV = { -0.372, 0.405, 42.31, 2.310 };
static const shen_disp_t SHEN_DISP_HARD = { 0.193, 0.066, 42.99, 1.883 };
static const shen_disp_t SHEN_DISP_SOFT = { 0.080, 0.180, 44.16, 1.496 };

/* Shen et al. (2020) Eq. 4, in dex. */
static double shen20_bc_sigma(const shen_disp_t* d, double logLbol_ergs)
{
  return d->sigma2 + d->sigma1 * (0.5 + 0.5 * erf((logLbol_ergs - d->logL0) / (sqrt(2.0) * d->sigma3)));
}

/* One standard-normal deviate from the run's GSL generator. */
static double shen20_draw_gaussian(void)
{
  return gsl_cdf_ugaussian_Pinv(gsl_rng_uniform(run_globals.random_generator));
}

/* Lognormal factor to MULTIPLY a bolometric correction by, given a standard
 * normal deviate g. Returns 1.0 (no change) if the scatter is switched off. */
static double shen20_bc_scatter_factor(const shen_disp_t* d, double logLbol_ergs, double g)
{
  double sigma_dex;

  sigma_dex = run_globals.params.physics.ShenBCScatterScale * shen20_bc_sigma(d, logLbol_ergs);
  if (sigma_dex <= 0.0)
    return 1.0;

  return pow(10.0, sigma_dex * g);
}

/* Bolometric luminosity [1e10 Lsun] of an accretion episode, at the MIDDLE of its accretion time. Used by
 * calculate_BHemissivity and by the QuasarLF_BC band, so both see the same Lbol. */
static double quasar_lbol(double BlackHoleMass, double accreted_mass)
{
  return sqrt(1.0 + accreted_mass / BlackHoleMass) * run_globals.params.physics.EddingtonRatio * BlackHoleMass /
         run_globals.params.Hubble_h * LUMINOSITY_CONVERTOR;
}

/* QuasarLF_BC band (output only; nothing in the model uses it). Returns 10^(QuasarLFBCNSigma * sigma_UV), with
 * sigma_UV the Shen+20 1-sigma uncertainty of the UV bolometric correction log10(Lbol/L_UV) (their Eq. 4 with the
 * Table 1 parameters sigma1, sigma2, logL0, sigma3) at this episode's Lbol. Since
 * L_UV = Lbol / kb_uv, L_UV times this factor is the episode with kb_uv lowered by QuasarLFBCNSigma sigma (brighter)
 * and L_UV divided by it the episode with kb_uv raised by the same amount (fainter). 1.0 when the band is off. */
static double quasar_uv_bc_band_factor(double BlackHoleMass, double accreted_mass)
{
  double Lbol;
  double nsigma;

  nsigma = run_globals.params.QuasarLFBCNSigma;
  if ((!run_globals.params.Flag_OutputQuasarLF) || (nsigma <= 0.0) || (BlackHoleMass <= 0.0))
    return 1.0;
  Lbol = quasar_lbol(BlackHoleMass, accreted_mass);
  if (Lbol <= 0.0)
    return 1.0;
  return pow(10.0, nsigma * shen20_bc_sigma(&SHEN_DISP_UV, log10(Lbol) + 10.0 + LOG_10_SOLAR_LUM));
}

/* XrayLF_BC band (output only), the hard X-ray counterpart of quasar_uv_bc_band_factor: 10^(XrayLFBCNSigma *
 * sigma_hard), with sigma_hard the Shen+20 1-sigma uncertainty of the 2-10 keV bolometric correction log10(Lbol/L_X)
 * (Eq. 4, Table 1) at this episode's Lbol. L_X times / divided by it is the episode with kb_hard lowered / raised by
 * XrayLFBCNSigma sigma. 1.0 when the band is off. */
static double quasar_hard_bc_band_factor(double BlackHoleMass, double accreted_mass)
{
  double Lbol;
  double nsigma;

  nsigma = run_globals.params.XrayLFBCNSigma;
  if ((!run_globals.params.Flag_OutputXrayLF) || (nsigma <= 0.0) || (BlackHoleMass <= 0.0))
    return 1.0;
  Lbol = quasar_lbol(BlackHoleMass, accreted_mass);
  if (Lbol <= 0.0)
    return 1.0;
  return pow(10.0, nsigma * shen20_bc_sigma(&SHEN_DISP_HARD, log10(Lbol) + 10.0 + LOG_10_SOLAR_LUM));
}

void calculate_BHemissivity(double BlackHoleMass,
                            double accreted_mass,
                            double* emissivity,
                            double* accretion_time,
                            double* quasar_luv,
                            double* quasar_lx,
                            double* quasar_lx_soft)
{
  double Lbol;
  double kb_uv;
  double kb_hard;
  double kb_soft;
  double logLbol_ergs;
  double g_uv;
  double g_hard;
  double g_soft;
  int shared_deviate;
  physics_params_t* physics = &(run_globals.params.physics);

  // Compute accretion_time directly in internal units using pre-computed EddingtonTimescale
  *accretion_time =
    log1p(accreted_mass / BlackHoleMass) * run_globals.EddingtonTimescale * ETA / physics->EddingtonRatio;
  // Bolometric luminosity in 1e10 Lsun at the MIDDLE of accretion time
  Lbol = quasar_lbol(BlackHoleMass, accreted_mass);
  // bolometric correction from Shen et al. 2020
  kb_uv = 1.862 * pow(Lbol, -0.361) + 4.870 * pow(Lbol, -0.0063);
  kb_hard = 4.073 * pow(Lbol, -0.026) + 12.60 * pow(Lbol, 0.278);
  kb_soft = 5.712 * pow(Lbol, -0.026) + 17.67 * pow(Lbol, 0.278);

  /* Optional Shen+20 Table 1 1-sigma uncertainty applied as random scatter. Off by default,
   * so the fiducial model (and the unit tests) are bit-for-bit unchanged. */
  if ((physics->Flag_ShenBCScatter != 0) && (Lbol > 0.0)) {
    /* Shen's sigma is a function of Lbol in erg/s; Lbol here is in 1e10 Lsun.
     * Same conversion as _NH_distribution uses for LX above. */
    logLbol_ergs = log10(Lbol) + 10.0 + LOG_10_SOLAR_LUM;
    shared_deviate = (physics->Flag_ShenBCScatter == 2);
    g_uv = shen20_draw_gaussian();
    g_hard = shared_deviate ? g_uv : shen20_draw_gaussian();
    g_soft = shared_deviate ? g_uv : shen20_draw_gaussian();
    kb_uv *= shen20_bc_scatter_factor(&SHEN_DISP_UV, logLbol_ergs, g_uv);
    kb_hard *= shen20_bc_scatter_factor(&SHEN_DISP_HARD, logLbol_ergs, g_hard);
    kb_soft *= shen20_bc_scatter_factor(&SHEN_DISP_SOFT, logLbol_ergs, g_soft);
  }

  // Return UV luminosity LUV = Lbol/kb in 1e10 Lsun
  *quasar_luv = Lbol / kb_uv;
  *quasar_lx = Lbol / kb_hard;
  *quasar_lx_soft = Lbol / kb_soft;

  double break_factor = pow(NUIONIZATION / NU_1450, -physics->SpecIndexUVAGNSoft);

  // Approximation using the emissivity at the MIDDLE of accretion time
  *emissivity = physics->quasar_fobs * *quasar_luv * LUV2EMISSIVITY * break_factor / physics->SpecIndexUVAGNHard *
                *accretion_time * run_globals.units.UnitTime_in_s / run_globals.params.Hubble_h;
}

static double get_vvir(galaxy_t* gal)
{
  // If this galaxy is the central of it's FOF group then use the FOF Halo properties
  // TODO: This needs closer thought as to if this is the best thing to do...
  return ((gal->Type == 0) && (!gal->ghost_flag)) ? gal->Halo->FOFGroup->Vvir : gal->Vvir;
}

// quasar feedback suggested by Croton et al. 2016
static void update_reservoirs_from_quasar_mode_bh_feedback(galaxy_t* gal, double m_reheat)
{
  double metallicity;
  galaxy_t* central;

  if (gal->ghost_flag)
    central = gal;
  else
    central = gal->Halo->FOFGroup->FirstOccupiedHalo->Galaxy;

  if (m_reheat < gal->ColdGas) {
    metallicity = calc_metallicity(gal->ColdGas, gal->MetalsColdGas);
    gal->ColdGas -= m_reheat;
    gal->MetalsColdGas -= m_reheat * metallicity;
    central->MetalsHotGas += m_reheat * metallicity;
    central->HotGas += m_reheat;
  } else {
    metallicity = calc_metallicity(central->HotGas, central->MetalsHotGas);
    gal->ColdGas = 0.0;
    gal->MetalsColdGas = 0.0;
    central->HotGas -= m_reheat;
    central->MetalsHotGas -= m_reheat * metallicity;
    central->EjectedGas += m_reheat;
    central->MetalsEjectedGas += m_reheat * metallicity;
  }

  // Check the validity of the modified reservoir values (HotGas can be negative for too strong quasar feedback)
  CLAMP_NEGATIVE(central->HotGas);
  CLAMP_NEGATIVE(central->MetalsHotGas);
  CLAMP_NEGATIVE(gal->ColdGas);
  CLAMP_NEGATIVE(gal->MetalsColdGas);
  CLAMP_NEGATIVE(gal->StellarMass);
  CLAMP_NEGATIVE(central->EjectedGas);
  CLAMP_NEGATIVE(central->MetalsEjectedGas);
}

double radio_mode_BH_heating(galaxy_t* gal, double cooling_mass, double x)
{
  double heated_mass = 0.0;
  // if there is any hot gas
  if (gal->HotGas > 0.0) {
    double Vvir = get_vvir(gal);
    run_units_t* units = &(run_globals.units);
    double accreted_mass;
    double eddington_mass;

    // bondi-hoyle accretion model
    accreted_mass = run_globals.params.physics.RadioModeEff * run_globals.G * BONDI_HOYLE_COEFFICIENT * x *
                    gal->BlackHoleMass * gal->dt;

    // eddington rate
    eddington_mass = exp(gal->dt / run_globals.EddingtonTimescale / ETA * run_globals.params.physics.EddingtonRatio) *
                     gal->BlackHoleMass;

    // limit accretion by the eddington rate
    if (accreted_mass > eddington_mass)
      accreted_mass = eddington_mass;

    // limit accretion by amount of hot gas available
    if (accreted_mass > gal->HotGas)
      accreted_mass = gal->HotGas;

    // mass heated by AGN following Croton et al. 2006
    heated_mass = 2. * ETA * run_globals.Csquare / Vvir / Vvir * accreted_mass;

    // limit the amount of heating to the amount of cooling
    if (heated_mass > cooling_mass) {
      accreted_mass = cooling_mass / heated_mass * accreted_mass;
      heated_mass = cooling_mass;
    }

    gal->BlackHoleAccretedHotMass = accreted_mass;

    // add the accreted mass to the black hole from hot gas
    double metallicity = calc_metallicity(gal->HotGas, gal->MetalsHotGas);

    // Assuming all energy from radio mode is going to heat the cooling flow
    // So no emissivity from radio mode!
    // TODO: we could add heating effienciency to split the energy into
    // heating and reionization.

    gal->BlackHoleMass += accreted_mass * (1. - ETA);
    gal->HotGas -= accreted_mass;
    gal->MetalsHotGas -= accreted_mass * metallicity;
  }
  return heated_mass;
}

void merger_driven_BH_growth(galaxy_t* gal, double merger_ratio, int snapshot)
{
  if (gal->ColdGas > 0) {
    // If there is any cold gas to feed the black hole...
    double Vvir = get_vvir(gal);

    // Suggested by Bonoli et al. 2009 and Wyithe et al. 2003
    double z_scaling = pow((1 + run_globals.ZZ[snapshot]), run_globals.params.physics.quasar_mode_scaling);

    double accreting_mass = run_globals.params.physics.BlackHoleGrowthRate * merger_ratio /
                            (1.0 + pow(VELOCITY_SCALE / Vvir, 2)) * gal->ColdGas * z_scaling;

    // limit accretion to what is available
    if (accreting_mass > gal->ColdGas)
      accreting_mass = gal->ColdGas;
    // add the accreting mass to the black hole from coldgas
    double metallicity = calc_metallicity(gal->ColdGas, gal->MetalsColdGas);

    // put the mass onto the accretion disk and let the black hole accrete it in the next snapshot
    // TODO: since the merger is put in the end of galaxy evolution, this is following the
    // inconsistence consistently
    gal->BlackHoleAccretingColdMass += accreting_mass;
    gal->ColdGas -= accreting_mass;
    gal->MetalsColdGas -= accreting_mass * metallicity;
  }
}

void previous_merger_driven_BH_growth(galaxy_t* gal, int snapshot)
{
  // If there is any cold gas to feed the black hole...
  double Vvir = get_vvir(gal);
  run_units_t* units = &(run_globals.units);
  double factor = EMISSIVITY_CONVERTOR * gal->FescBH / run_globals.params.physics.ReionNionPhotPerBary;

  // Use snapshot cadence timestep instead of gal->dt to ensure proper accretion
  // for ghost galaxies that have been in ghost state for multiple snapshots.
  // snapshot is the current snapshot, snapshot+1 is the next snapshot in the past.
  double dt = (snapshot > 0) ? (run_globals.LTTime[snapshot - 1] - run_globals.LTTime[snapshot]) : 0.0;

  double m_reheat;
  double accreted_mass;
  double BHemissivity, accretion_time, quasar_luv;
  double quasar_lx, quasar_lx_soft;
  double bc_band_factor; /* QuasarLF_BC band, output only */
  double xr_band_factor; /* XrayLF_BC band, output only */
  double obs_fraction_hard;
  double obs_fraction_soft;
  int NH_bin;
  double t_off, t_resp;

  // Determine the accretion on-time within this snapshot
  if (dt <= 0.0) {
    return; // No time passed, exit function
  }

  // Determine the accretion on-time within this snapshot
  if (run_globals.params.physics.Flag_BHARExponentialCut) {
    if (gal->BHAccretionOnTime < 0.0) {
      // First snapshot of accretion after merger - assign random on-time
      gal->BHAccretionOnTime = gsl_rng_uniform(run_globals.random_generator);
    } else {
      // Accretion was already happening in previous snapshot - start immediately
      gal->BHAccretionOnTime = 0.0;
    }
    // Adjust effective timestep based on when accretion starts
    dt *= (1.0 - gal->BHAccretionOnTime);
  } else {
    // No random on-time when using duty-cycle weighting
    gal->BHAccretionOnTime = 0.0;
  }

  if (dt > 0.0) {
    // Eddington rate (using effective timestep)
    accreted_mass =
      expm1(dt / run_globals.EddingtonTimescale / ETA * run_globals.params.physics.EddingtonRatio) * gal->BlackHoleMass;

    // limit accretion to what is need
    if (accreted_mass > gal->BlackHoleAccretingColdMass)
      accreted_mass = gal->BlackHoleAccretingColdMass;

    gal->BlackHoleAccretedColdMass += accreted_mass;
    gal->BlackHoleAccretingColdMass -= accreted_mass;

    // Reset on-time if accretion is complete
    if (gal->BlackHoleAccretingColdMass <= 0.0)
      gal->BHAccretionOnTime = -1.0;

    // N_gamma,q * N_bh; later 1e60*BHemissivity * PROTONMASS/1e10/SOLAR_MASS will be N_gamma,q * M_bh
    calculate_BHemissivity(
      gal->BlackHoleMass, accreted_mass, &BHemissivity, &accretion_time, &quasar_luv, &quasar_lx, &quasar_lx_soft);
    /* same BlackHoleMass as above, i.e. before this episode's growth is added */
    bc_band_factor = quasar_uv_bc_band_factor(gal->BlackHoleMass, accreted_mass);
    xr_band_factor = quasar_hard_bc_band_factor(gal->BlackHoleMass, accreted_mass);
    apply_xray_obscuration(quasar_lx, run_globals.ZZ[snapshot], &obs_fraction_hard, &obs_fraction_soft, &NH_bin);
    /* Flag_AGNDustExtinction = 1: the AGN's own Shen+20 dust, not the quasar_open_angle cone, sets how many of its
     * ionising photons escape (quasar_fobs = 1 then, so BHemissivity above is the unobscured value) */
    if (run_globals.params.physics.Flag_AGNDustExtinction)
      BHemissivity *= agn_dust_escape_ionising(NH_bin, run_globals.ZZ[snapshot]);

    gal->DutyCycleAGN = (accretion_time > 1e-30) ? (accretion_time / dt) : 0.0;
    CLAMP_0_1(gal->DutyCycleAGN);

    gal->BlackHoleMass += (1. - ETA) * accreted_mass;
    gal->EffectiveBHM += BHemissivity * factor;

    // historical reason for us to store nion rather than the emissivity in BHemissivity...
    BHemissivity *= factor / accretion_time;

    if (run_globals.params.physics.Flag_BHARExponentialCut) {
      t_off = (1.0 - gal->DutyCycleAGN) * dt;
      if (t_off > 0.0) {
        t_resp = gal->t_resp * run_globals.params.Hubble_h / units->UnitTime_in_Megayears;
        if (t_resp > 0.0)
          BHemissivity *= exp(-t_off / t_resp);
      }
    } else
      BHemissivity *= gal->DutyCycleAGN;

    gal->BHemissivity += BHemissivity;
    gal->QuasarLuv += quasar_luv;
    gal->QuasarLuvBCBright += quasar_luv * bc_band_factor;
    gal->QuasarLuvBCFaint += quasar_luv / bc_band_factor;
    gal->QuasarLX += quasar_lx;
    gal->QuasarLXBCBright += quasar_lx * xr_band_factor;
    gal->QuasarLXBCFaint += quasar_lx / xr_band_factor;
    gal->NHbin = NH_bin; /* -1 if no AGN activity this step */
    gal->BHXrayEmissivity_hard += quasar_lx * obs_fraction_hard;
    gal->BHXrayEmissivity_soft += quasar_lx_soft * obs_fraction_soft;
    gal->EffectiveBHAR += BHemissivity;
    // quasar mode feedback
    m_reheat = run_globals.params.physics.QuasarModeEff * 2. * ETA * run_globals.Csquare * accreted_mass / Vvir / Vvir;
    update_reservoirs_from_quasar_mode_bh_feedback(gal, m_reheat);
  }
}
