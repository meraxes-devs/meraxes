#include "emission_lines.h"
#include <math.h>

static bool has_valid_loiii_inputs(const galaxy_t* gal)
{
  return gal->Sfr > 0.0 && gal->MetalsColdGas > 0.0 && gal->ColdGas > 0.0 && gal->Mvir > 0.0 &&
         gal->DiskScaleLength > 0.0 && gal->Rvir > 0.0 && gal->dt > 0.0 && gal->Vmax > 0.0 && gal->StellarMass >= 0.0 &&
         gal->Mcool >= 0.0 && gal->Spin > 0.0;
}

static double clamp_non_finite(double value)
{
  if (!isfinite(value) || value < 0.0)
    return 0.0;

  return value;
}

void set_OIII_coeffs(double T)
{
  if (T <= 0.0)
    T = 1e4;

  const double beta_q = sqrt(2.0 * M_PI * pow(HBAR_SI, 4) / (BOLTZMANN_SI * pow(ELECTRON_MASS_SI, 3))) * 1e6;
  const double o30 = 0.243 * pow(T / 1e4, 0.120 + 0.031 * log(T / 1e4));
  const double o40 = 0.0321 * pow(T / 1e4, 0.118 + 0.057 * log(T / 1e4));
  const double k03 = (beta_q / sqrt(T)) * o30 * exp(-29169.0 / T);
  const double k04 = (beta_q / sqrt(T)) * o40 * exp(-61207.0 / T);

  run_globals.loiii_params.oxygen_abundance_over_z_sun = (pow(10.0, 8.69) / 1e12) / Z_SUN;
  run_globals.loiii_params.excitation_rate = k03 + k04 * (LOIII_A43 / (LOIII_A43 + LOIII_A41));
  run_globals.loiii_params.branching_ratio = LOIII_A32 / (LOIII_A32 + LOIII_A31);
}

// Sets (does not accumulate) the galaxy's snapshot LOIII and ionization_param;
// call once per snapshot after all star formation and mergers.
void compute_LOIII(galaxy_t* gal, int snapshot)
{
  if (!has_valid_loiii_inputs(gal)) {
    gal->LOIII = 0.0;
    gal->ionization_param = 0.0;
    return;
  }

  double density, bubble_count, ionizing_photon_rate, ionization_param;

  const run_units_t* units = &run_globals.units;

  const double mass_unit_g = units->UnitMass_in_g / run_globals.params.Hubble_h;
  const double rate_unit_gs = units->UnitMass_in_g / units->UnitTime_in_s;
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

  const double sfr_gs = gal->Sfr * rate_unit_gs;
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
  bubble_count = total_mass_g / density / 4.0 * 3.0 / M_PI / starburst_radius_cm_cb;
  ionizing_photon_rate = (sfr_gs / bubble_count / PROTONMASS) * 4000.0;

  // density is a mass density (g/cm^3); the Stromgren radius and the
  // ionization parameter (cm/s) need the hydrogen number density (cm^-3).
  const double n_H = (1.0 - run_globals.params.physics.Y_He) * density / PROTONMASS;
  const double stromgren_radius_sq = pow(3.0 * ionizing_photon_rate / (4.0 * M_PI * ALPHA_HII * n_H * n_H), 2.0 / 3.0);
  ionization_param = 1.5874 * ionizing_photon_rate / (4.0 * M_PI * stromgren_radius_sq * n_H);

  gal->ionization_param = clamp_non_finite(ionization_param);

  double metallicity = gal->MetalsColdGas / gal->ColdGas;
  double oiii_volume_fraction = 0.8;
  // Scaled by 1e-40 so gal->LOIII is stored in units of 1e40 erg/s (matches
  // galaxy_output_t.LOIII).
  double loiii = run_globals.loiii_params.oxygen_abundance_over_z_sun * metallicity *
                 run_globals.loiii_params.excitation_rate * run_globals.loiii_params.branching_ratio *
                 (ionizing_photon_rate * bubble_count / ALPHA_HII) * PLANCK * nu32 * oiii_volume_fraction * 1e-40;

  gal->LOIII = clamp_non_finite(loiii);
}
