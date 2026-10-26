#define _MAIN
#include <criterion/criterion.h>
#include <math.h>
#include <string.h>

#include "meraxes.h"

// Exercise the actual static normalization helper, following test_xray_luminosity.c.
#include "core/ComputeTs.c"

static void setup(void)
{
  memset(&run_globals, 0, sizeof(run_globals));
  physics_params_t* params = &run_globals.params.physics;
  params->NuXrayThreshold = 500.0;
  params->NuXraySoftCut = 2000.0;
  params->NuXrayMax = 10000.0;
  params->SpecIndexXrayAGNSoft = 2.2;
  params->SpecIndexXrayAGNHard = 1.7;
}

TestSuite(agn_xray_heating, .init = setup);

Test(agn_xray_heating, disabled_agn_ignores_empty_hard_band)
{
  run_globals.params.physics.Flag_IncludeAGNXray = 0;
  run_globals.params.physics.NuXrayMax = run_globals.params.physics.NuXraySoftCut;
  cr_expect_eq(agn_xray_heating_prefactor(false, 10.0), 0.0);
  cr_expect_eq(agn_xray_heating_prefactor(true, 10.0), 0.0);
}

Test(agn_xray_heating, soft_only_ignores_empty_hard_band)
{
  run_globals.params.physics.Flag_IncludeAGNXray = 3;
  run_globals.params.physics.NuXrayMax = run_globals.params.physics.NuXraySoftCut;
  double soft = agn_xray_heating_prefactor(false, 10.0);
  cr_assert(isfinite(soft));
  cr_expect_gt(soft, 0.0);
  cr_expect_eq(agn_xray_heating_prefactor(true, 10.0), 0.0);
}

Test(agn_xray_heating, hard_only_ignores_disabled_soft_spectral_index)
{
  run_globals.params.physics.Flag_IncludeAGNXray = 2;
  run_globals.params.physics.SpecIndexXrayAGNSoft = NAN;
  cr_expect_eq(agn_xray_heating_prefactor(false, 10.0), 0.0);
  double hard = agn_xray_heating_prefactor(true, 10.0);
  cr_assert(isfinite(hard));
  cr_expect_gt(hard, 0.0);
}

Test(agn_xray_heating, enabled_hard_band_rejects_empty_or_reversed_interval)
{
  for (int flag = 1; flag <= 2; flag++) {
    run_globals.params.physics.Flag_IncludeAGNXray = flag;
    run_globals.params.physics.NuXrayMax = 2000.0;
    cr_expect(isnan(agn_xray_heating_prefactor(true, 10.0)));
    run_globals.params.physics.NuXrayMax = 1000.0;
    cr_expect(isnan(agn_xray_heating_prefactor(true, 10.0)));
  }
}

Test(agn_xray_heating, enabled_bands_reject_nonfinite_parameters)
{
  run_globals.params.physics.Flag_IncludeAGNXray = 1;
  run_globals.params.physics.SpecIndexXrayAGNSoft = NAN;
  cr_expect(isnan(agn_xray_heating_prefactor(false, 10.0)));
  run_globals.params.physics.SpecIndexXrayAGNHard = INFINITY;
  cr_expect(isnan(agn_xray_heating_prefactor(true, 10.0)));
  run_globals.params.physics.SpecIndexXrayAGNHard = 1.7;
  run_globals.params.physics.NuXrayMax = INFINITY;
  cr_expect(isnan(agn_xray_heating_prefactor(true, 10.0)));
}

// Independently integrate the adopted spectrum with Simpson's rule in energy space.
static double spectral_integral(double energy_lo, double energy_hi, double alpha)
{
  const int intervals = 2048;
  double step = (energy_hi - energy_lo) / intervals;
  double sum = 0.0;
  for (int i = 0; i <= intervals; i++) {
    double energy = energy_lo + i * step;
    int weight = (i == 0 || i == intervals) ? 1 : (i % 2 == 0 ? 2 : 4);
    sum += weight * pow(energy / run_globals.params.physics.NuXrayThreshold, -alpha);
  }
  return sum * step / 3.0 * NU_over_EV;
}

Test(agn_xray_heating, enabled_prefactors_normalize_the_integrated_spectrum)
{
  run_globals.params.physics.Flag_IncludeAGNXray = 1;
  const double alphas[] = { 1.0, 1.7, 2.2 };
  const double redshifts[] = { 8.0, 12.0 };
  for (int band = 0; band <= 1; band++) {
    double energy_lo = band ? 2000.0 : 500.0;
    double energy_hi = band ? 10000.0 : 2000.0;
    for (int i = 0; i < 3; i++) {
      run_globals.params.physics.SpecIndexXrayAGNSoft = alphas[i];
      run_globals.params.physics.SpecIndexXrayAGNHard = alphas[i];
      double integral = spectral_integral(energy_lo, energy_hi, alphas[i]);
      for (int j = 0; j < 2; j++) {
        double prefactor = agn_xray_heating_prefactor((bool)band, redshifts[j]);
        cr_assert(isfinite(prefactor));
        cr_assert_gt(prefactor, 0.0);
        double normalization = prefactor * PLANCK * (500.0 * NU_over_EV) * integral /
                               (SPEED_OF_LIGHT * pow(1.0 + redshifts[j], alphas[i] + 3.0));
        cr_expect_float_eq(normalization, 1.0, 1e-9);
      }
    }
  }
}
