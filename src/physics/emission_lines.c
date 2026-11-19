#include "emission_lines.h"
#include <math.h>

// Disc state needed by the ionization parameter (not by the luminosity).
static bool has_valid_disc_state(const galaxy_t* gal)
{
  return gal->ColdGas > 0.0 && gal->Mvir > 0.0 && gal->DiskScaleLength > 0.0 && gal->Rvir > 0.0 && gal->dt > 0.0 &&
         gal->Vmax > 0.0 && gal->StellarMass >= 0.0 && gal->Mcool >= 0.0 && gal->Spin > 0.0;
}

static double clamp_non_finite(double value)
{
  if (!isfinite(value) || value < 0.0)
    return 0.0;

  return value;
}

// Metallicity range [Z_sun] of the Yang et al. (2023) electron-temperature fit.
// Outside it the temperature stays at its value at the nearest end.
#define OIII_TEMPERATURE_FIT_Z_MIN 0.016
#define OIII_TEMPERATURE_FIT_Z_MAX 1.6

void set_OIII_coeffs(void)
{
  // sqrt(2 pi hbar^4 / (k m_e^3)) = 8.63e-6 cm^3 s^-1 K^1/2
  run_globals.loiii_params.excitation_prefactor =
    sqrt(2.0 * M_PI * pow(HBAR_SI, 4) / (BOLTZMANN_SI * pow(ELECTRON_MASS_SI, 3))) * 1e6;
  run_globals.loiii_params.oxygen_abundance_over_z_sun = (pow(10.0, 8.69) / 1e12) / Z_SUN;
  run_globals.loiii_params.branching_ratio = LOIII_A32 / (LOIII_A32 + LOIII_A31);
}

// Electron temperature [K] of HII regions in gas of this metallicity, from the
// Yang et al. (2023) fit T4 = 0.824 x^2 + 0.101 x + 1.08 with x = log10(Z / Z_sun).
double oiii_electron_temperature(double metallicity)
{
  // fmax also takes a zero, negative or NaN metallicity to the low end of the fit.
  const double z_ratio = fmin(fmax(metallicity / Z_SUN, OIII_TEMPERATURE_FIT_Z_MIN), OIII_TEMPERATURE_FIT_Z_MAX);
  const double x = log10(z_ratio);
  return 1e4 * (0.824 * x * x + 0.101 * x + 1.08);
}

// Case-B hydrogen recombination coefficient [cm^3/s] (Osterbrock 1989).
double case_b_recombination_coefficient(double temperature)
{
  return 2.59e-13 * pow(temperature / 1e4, -0.7);
}

// Rate coefficient [cm^3/s] for electrons exciting O++ into 1D2, directly from the
// ground term and through 1S0 -> 1D2 cascades; collision strengths from Draine (2011).
double oiii_excitation_rate(double temperature)
{
  const double t4 = temperature / 1e4;
  const double prefactor = run_globals.loiii_params.excitation_prefactor / sqrt(temperature);
  const double k03 = prefactor * 0.243 * pow(t4, 0.120 + 0.031 * log(t4)) * exp(-29169.0 / temperature);
  const double k04 = prefactor * 0.0321 * pow(t4, 0.118 + 0.057 * log(t4)) * exp(-62134.0 / temperature);
  return k03 + k04 * (LOIII_A43 / (LOIII_A43 + LOIII_A41));
}

// Ionization parameter of the HII regions of one star-formation event with rate
// sfr_gs [g/s], total ionizing photon rate photon_rate [1/s] and case-B
// recombination coefficient alpha_b, from the disc state at that event; 0 if the
// disc state is invalid.
static double event_ionization_param(const galaxy_t* gal,
                                     double sfr_gs,
                                     double photon_rate,
                                     double alpha_b,
                                     int snapshot)
{
  if (!has_valid_disc_state(gal))
    return 0.0;

  double density, bubble_count, ionizing_photon_rate, ionization_param;

  const run_units_t* units = &run_globals.units;
  const double rate_unit_gs = units->UnitMass_in_g / units->UnitTime_in_s;
  const double mass_unit_g = units->UnitMass_in_g / run_globals.params.Hubble_h;
  const double length_unit_cm = units->UnitLength_in_cm / run_globals.params.Hubble_h;

  const double disk_mass_fraction = (gal->StellarMass + gal->ColdGas) / gal->Mvir;
  const double starburst_radius_cm_cb =
    pow(8e-5 * MPC, 3) * pow(gal->Spin / 0.05, 4) * pow(disk_mass_fraction / 0.17, -2) *
    pow(gal->Mvir * 1e2 / run_globals.params.Hubble_h, -2.0 / 3.0) * pow((1.0 + run_globals.ZZ[snapshot]) / 10.0, -4);

  const double disk_radius_cm_sq = pow(gal->DiskScaleLength * length_unit_cm, 2);
  const double total_mass_g = (gal->StellarMass + gal->ColdGas) * mass_unit_g;
  const double cold_gas_mass_g = gal->ColdGas * mass_unit_g;
  density = (GRAVITY * cold_gas_mass_g * total_mass_g) /
            (58.7528 * M_PI * COLD_GAS_SOUND_SPEED_CMS_SQ * disk_radius_cm_sq * disk_radius_cm_sq);

  const double sn_rate_surface_density =
    (0.156 * sfr_gs) / (SN_PROGENITOR_MASS_MSUN * SOLAR_MASS * M_PI * disk_radius_cm_sq);
  const double sn_velocity_cms_sq_norm =
    pow((2.0 * sn_rate_surface_density * 0.03 * EnergySN) / density, 2.0 / 3.0) / COLD_GAS_SOUND_SPEED_CMS_SQ;

  const double disk_to_virial_ratio = pow(gal->DiskScaleLength / gal->Rvir, 3);
  const double delta_inv = (gal->Mvir * disk_to_virial_ratio + gal->ColdGas + gal->StellarMass) / gal->ColdGas;
  const double toomre_q_sq = 0.98 * pow(delta_inv, 2) *
                             (COLD_GAS_SOUND_SPEED_CMS_SQ / pow(1.4 * gal->Vmax * units->UnitVelocity_in_cm_per_s, 2));
  const double cooling_rate_gs = gal->Mcool / gal->dt * rate_unit_gs;
  const double accretion_velocity_cms_sq_norm =
    pow((0.6 * GRAVITY * cooling_rate_gs * toomre_q_sq) / 2.94, 2.0 / 3.0) / COLD_GAS_SOUND_SPEED_CMS_SQ;

  density /= (1 + sn_velocity_cms_sq_norm + accretion_velocity_cms_sq_norm);
  // The superbubbles hold the disc's cold gas (ISM porosity 1).
  bubble_count = cold_gas_mass_g / density / 4.0 * 3.0 / M_PI / starburst_radius_cm_cb;
  ionizing_photon_rate = photon_rate / bubble_count;

  // density is a mass density (g/cm^3); the Stromgren radius and the
  // ionization parameter (cm/s) need the hydrogen number density (cm^-3).
  const double n_H = (1.0 - run_globals.params.physics.Y_He) * density / PROTONMASS;
  const double stromgren_radius_sq = pow(3.0 * ionizing_photon_rate / (4.0 * M_PI * alpha_b * n_H * n_H), 2.0 / 3.0);
  ionization_param = 1.5874 * ionizing_photon_rate / (4.0 * M_PI * stromgren_radius_sq * n_H);

  return clamp_non_finite(ionization_param);
}

// Adds one star-formation event (in situ or merger-driven) to the galaxy's snapshot
// LOIII and folds its ionization parameter into the LOIII-weighted mean. Call from
// update_reservoirs_from_sf with the event's SFR and the metallicity of the gas that
// forms the stars, before ColdGas, MetalsColdGas and StellarMass are updated. Star
// formation leaves the gas Z unchanged, so this is the Z after star formation and
// before the event's SN feedback.
void compute_LOIII(galaxy_t* gal, double sfr, double metallicity, int snapshot)
{
  if (!(sfr > 0.0))
    return;

  const double sfr_gs = sfr * run_globals.units.UnitMass_in_g / run_globals.units.UnitTime_in_s;

  // Ionizing photons per stellar baryon as in reionization (Pop. II). The photon
  // rate per bubble times the number of bubbles is this total, so the luminosity
  // does not depend on the disc state.
  const double photon_rate = sfr_gs / PROTONMASS * run_globals.params.physics.ReionNionPhotPerBary;
  // The HII regions' temperature, and with it the excitation and recombination
  // rates, follows the metallicity of the gas the stars form from.
  const double temperature = oiii_electron_temperature(metallicity);
  const double alpha_b = case_b_recombination_coefficient(temperature);
  const double oiii_volume_fraction = 0.8;
  // Scaled by 1e-40 so gal->LOIII is stored in units of 1e40 erg/s (matches
  // galaxy_output_t.LOIII).
  const double loiii = clamp_non_finite(run_globals.loiii_params.oxygen_abundance_over_z_sun * metallicity *
                                        oiii_excitation_rate(temperature) * run_globals.loiii_params.branching_ratio *
                                        (photon_rate / alpha_b) * PLANCK * nu32 * oiii_volume_fraction * 1e-40);
  if (loiii <= 0.0)
    return;

  // The ionization parameter is intensive, so the galaxy's value is the mean over its
  // events weighted by their [O III] luminosities; mergers combine galaxies the same
  // way (merge_with_target). Each event's HII regions are a population of their own:
  // they form from gas of their own metallicity, which sets their temperature and
  // recombination rate. LOIII is the weight because q is measured from [O III]-based
  // line ratios of the integrated spectrum, which the brightest [O III] regions
  // dominate. Since the weight is additive, the result does not depend on the order
  // of events or mergers, and a galaxy has q > 0 exactly when LOIII > 0 (events in
  // metal-free gas carry no weight).
  // Caveat: at a fixed disc state q scales as SFR^(1/3), so for events sharing a disc
  // in one snapshot (an in-situ event and a merger burst) the mean lies below the q
  // of their combined SFR, by up to 2^(1/3) (0.1 dex) for equal SFRs.
  const double total_loiii = gal->LOIII + loiii;
  gal->ionization_param =
    (gal->LOIII * gal->ionization_param + loiii * event_ionization_param(gal, sfr_gs, photon_rate, alpha_b, snapshot)) /
    total_loiii;
  gal->LOIII = total_loiii;
}
