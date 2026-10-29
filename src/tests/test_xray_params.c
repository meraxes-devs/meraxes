#define _MAIN
#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>

#include "meraxes.h"

// Retain production validation; replace MPI termination in this isolated test.
#undef ABORT
#define ABORT(status) exit(status)
#include "core/read_params.c"

static void setup(void)
{
  memset(&run_globals, 0, sizeof(run_globals));
  run_globals.params.NSteps = 1;

  physics_params_t* params = &run_globals.params.physics;
  params->LXrayGal = 3.16e40;
  params->XrayMetallicityExponent = 0.3;
  params->XrayMetallicityTurnover = 0.008;
  params->SpecIndexXrayGal = 1.0;
  params->NuXrayThreshold = 500.0;
  params->NuXraySoftCut = 2000.0;
}

TestSuite(xray_params, .init = setup);

Test(xray_params, combined_models_are_rejected, .exit_code = EXIT_FAILURE)
{
  run_globals.params.physics.Flag_UseXrayMetallicity = 1;
  run_globals.params.physics.XrayScatterDex = 0.3;
  check_problem_params(&run_globals.params);
  cr_assert_fail("The metallicity-plus-scatter configuration was accepted.");
}

Test(xray_params, metallicity_without_scatter_is_accepted)
{
  run_globals.params.physics.Flag_UseXrayMetallicity = 1;
  check_problem_params(&run_globals.params);
}

#if USE_STOCHASTICITY
Test(xray_params, empirical_scatter_without_metallicity_is_accepted)
{
  run_globals.params.physics.XrayScatterDex = 0.3;
  check_problem_params(&run_globals.params);
}
#endif