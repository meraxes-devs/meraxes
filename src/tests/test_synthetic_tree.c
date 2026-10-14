#define _MAIN
#include <math.h>
#include <stdlib.h>

#include "fixtures/synthetic_tree.h"
#include "physics/cooling.h"
#include "physics/mergers.h"
#include "physics/star_formation.h"

// Keep checks active in Release builds, where assert() may be compiled out.
#define CHECK(condition)                                                                                               \
  do {                                                                                                                 \
    if (!(condition)) {                                                                                                \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition);                                                  \
      goto cleanup;                                                                                                    \
    }                                                                                                                  \
  } while (0)

static int close_enough(double actual, double expected)
{
  if (isfinite(actual) && fabs(actual - expected) <= 1e-12 + 1e-10 * fabs(expected))
    return 1;
  fprintf(stderr, "actual=%.17g expected=%.17g\n", actual, expected);
  return 0;
}

static double total_mass(const synthetic_tree_t* tree)
{
  double total = 0.0;
  for (int i = 0; i < SYNTHETIC_TREE_NGALAXIES; i++) {
    const galaxy_t* gal = tree->galaxies[i];
    if (gal->Type >= 3)
      continue;
    total += gal->HotGas + gal->ColdGas + gal->EjectedGas + gal->StellarMass + gal->BlackHoleMass +
             gal->BlackHoleAccretingColdMass;
#if USE_MINI_HALOS
    total += gal->Remnant_Mass;
#endif
  }
  return total;
}

static double total_metals(const synthetic_tree_t* tree)
{
  double total = 0.0;
  for (int i = 0; i < SYNTHETIC_TREE_NGALAXIES; i++) {
    const galaxy_t* gal = tree->galaxies[i];
    if (gal->Type < 3)
      total += gal->MetalsHotGas + gal->MetalsColdGas + gal->MetalsEjectedGas + gal->MetalsStellarMass;
  }
  return total;
}

int main(void)
{
  synthetic_tree_t tree;
  int result = EXIT_FAILURE;
  int dead_galaxies = 0;
  CHECK(synthetic_tree_init(&tree) == 0);
  galaxy_t* a = tree.galaxies[0];
  galaxy_t* b = tree.galaxies[1];
  const unsigned long id_a = a->ID;
  const unsigned long id_b = b->ID;

  CHECK(tree.n_galaxies == 2 && tree.n_created == 2 && tree.n_halo_mergers == 0);
  CHECK(run_globals.FirstGal == a && a->Next == b && run_globals.LastGal == b && b->Next == NULL);
  CHECK(a->Type == 0 && b->Type == 0 && a->Halo->FOFGroup != b->Halo->FOFGroup);
  CHECK(a->HaloDescIndex == 0 && b->HaloDescIndex == 1);
  CHECK(a->Halo->FOFGroup->FirstOccupiedHalo == a->Halo);
  CHECK(b->Halo->FOFGroup->FirstOccupiedHalo == b->Halo);
  CHECK(close_enough(total_mass(&tree), 0.08202));
  CHECK(close_enough(total_metals(&tree), 0.000735));
  for (int s = 1; s < SYNTHETIC_TREE_NSNAPS; s++) {
    CHECK(tree.scale_factors[s] > tree.scale_factors[s - 1]);
    CHECK(tree.redshifts[s] < tree.redshifts[s - 1]);
    CHECK(tree.lookback_times[s] < tree.lookback_times[s - 1]);
  }
  puts("snapshot 0: two central galaxies in two FOF groups");
  synthetic_tree_finish_snapshot(&tree);

  CHECK(synthetic_tree_advance(&tree) == 0);
  CHECK(a->Type == 0 && b->Type == 1);
  CHECK(a->Halo->FOFGroup == b->Halo->FOFGroup);
  CHECK(a->Halo->NextHaloInFOFGroup == b->Halo);
  CHECK(a->Halo->FOFGroup->TotalSubhaloLen == 176);
  CHECK(a->Halo->FOFGroup->FirstOccupiedHalo == a->Halo);
  CHECK(a->HaloDescIndex == 0 && b->HaloDescIndex == 0);
  CHECK(a->dt > 0.0 && close_enough(a->dt, b->dt));

  // Prescribed transfers isolate bookkeeping from cooling/SF rate prescriptions and table files.
  cool_gas_onto_galaxy(a, 0.004);
  cool_gas_onto_galaxy(b, 0.002);
  CHECK(close_enough(a->ColdGas, 0.044) && close_enough(b->ColdGas, 0.012));
  CHECK(close_enough(total_mass(&tree), 0.08202));
  CHECK(close_enough(total_metals(&tree), 0.000735));
  update_reservoirs_from_sf(a, 0.003, 1, INSITU);
  update_reservoirs_from_sf(b, 0.001, 1, INSITU);
  CHECK(close_enough(a->StellarMass, 0.003) && close_enough(b->StellarMass, 0.001));
  CHECK(close_enough(a->Sfr * a->dt, 0.003) && close_enough(b->Sfr * b->dt, 0.001));
  CHECK(close_enough(a->NewStars[0], 0.003) && close_enough(b->NewStars[0], 0.001));
  CHECK(close_enough(a->NewMetals[0], 0.00003) && close_enough(b->NewMetals[0], 0.000005));
  CHECK(close_enough(total_mass(&tree), 0.08202));
  CHECK(close_enough(total_metals(&tree), 0.000735));
  double satellite_mvir = b->Mvir;
  puts("snapshot 1: central + satellite; cooling and star formation conserve mass and metals");
  synthetic_tree_finish_snapshot(&tree);

  CHECK(synthetic_tree_advance(&tree) == 0);
  CHECK(tree.n_halo_mergers == 1);
  CHECK(a->Type == 0 && b->Type == 2 && b->OldType == 1);
  CHECK(a->Halo == b->Halo && a->Halo->Galaxy == a);
  CHECK(a->FirstGalInHalo == a && b->FirstGalInHalo == a);
  CHECK(a->NextGalInHalo == b && b->NextGalInHalo == NULL && b->MergerTarget == a);
  CHECK(close_enough(b->Mvir, satellite_mvir));
  CHECK(close_enough(a->Sfr, 0.0) && close_enough(b->Sfr, 0.0));
  CHECK(close_enough(a->NewStars[0], 0.0) && close_enough(b->NewStars[0], 0.0));
  CHECK(close_enough(a->NewStars[1], 0.003) && close_enough(b->NewStars[1], 0.001));
  CHECK(close_enough(a->StellarMass, 0.003) && close_enough(b->StellarMass, 0.001));
  CHECK(close_enough(b->dt, tree.lookback_times[1] - tree.lookback_times[2]));
  update_reservoirs_from_sf(a, 0.002, 2, INSITU);
  update_reservoirs_from_sf(b, 0.0005, 2, INSITU);
  // Match the NSteps=1 decrement in evolve.c: a new orphan must survive this frame.
  b->MergTime -= b->dt;
  CHECK(isfinite(b->MergTime) && b->MergTime > 0.0);
  CHECK(b->MergTime < tree.lookback_times[2] - tree.lookback_times[3]);
  double remaining_merger_time = b->MergTime;
  CHECK(close_enough(total_mass(&tree), 0.08202));
  CHECK(close_enough(total_metals(&tree), 0.000735));
  puts("snapshot 2: one halo with a central + orphan; merger clock remains positive");
  synthetic_tree_finish_snapshot(&tree);

  CHECK(synthetic_tree_advance(&tree) == 0);
  CHECK(tree.n_halo_mergers == 1 && tree.n_galaxies == 2);
  CHECK(a->Type == 0 && b->Type == 2 && b->OldType == 2);
  CHECK(a->Halo == &tree.snapshots[3].halos[0] && b->Halo == a->Halo);
  CHECK(a->NextGalInHalo == b && b->FirstGalInHalo == a && b->MergerTarget == a);
  CHECK(close_enough(b->Mvir, satellite_mvir));
  CHECK(close_enough(b->MergTime, remaining_merger_time));
  CHECK(close_enough(a->Sfr, 0.0) && close_enough(b->Sfr, 0.0));
  CHECK(close_enough(a->NewStars[2], 0.003) && close_enough(b->NewStars[2], 0.001));
  update_reservoirs_from_sf(a, 0.001, 3, INSITU);
  b->MergTime -= b->dt;
  CHECK(b->MergTime < 0.0);
  merge_with_target(b, &dead_galaxies, 3);

  CHECK(dead_galaxies == 1 && a->Type == 0 && b->Type == 3 && b->HaloDescIndex == -1);
  CHECK(a->ID == id_a && b->ID == id_b);
  CHECK(close_enough(a->HotGas, 0.019) && close_enough(a->ColdGas, 0.0485));
  CHECK(close_enough(a->EjectedGas, 0.007) && close_enough(a->BlackHoleMass, 0.00002));
  CHECK(close_enough(a->StellarMass, 0.0075) && close_enough(a->GrossStellarMass, 0.0075));
  CHECK(close_enough(a->MetalsHotGas, 0.000175) && close_enough(a->MetalsColdGas, 0.0004325));
  CHECK(close_enough(a->MetalsEjectedGas, 0.00006) && close_enough(a->MetalsStellarMass, 0.0000675));
  CHECK(close_enough(a->Sfr * a->dt, 0.001));
  CHECK(close_enough(a->FescWeightedGSM, 0.0015));
  CHECK(close_enough(a->FescWeightedSfr * a->dt, 0.0002));
  const double expected_stars[] = { 0.001, 0.0025, 0.004 };
  const double expected_metals[] = { 0.00001, 0.0000225, 0.000035 };
  for (int i = 0; i < N_HISTORY_SNAPS; i++) {
    double stars = i < 3 ? expected_stars[i] : 0.0;
    double metals = i < 3 ? expected_metals[i] : 0.0;
    CHECK(close_enough(a->NewStars[i], stars));
    CHECK(close_enough(a->NewMetals[i], metals));
#if USE_MINI_HALOS
    CHECK(close_enough(a->NewStars_II[i], stars));
    CHECK(close_enough(a->NewStars_III[i], 0.0));
#endif
  }
#if USE_MINI_HALOS
  CHECK(close_enough(a->StellarMass_II, 0.0075));
  CHECK(close_enough(a->StellarMass_III, 0.0));
#endif
  // merge_with_target marks B dead; its old reservoirs must not be counted a second time.
  CHECK(close_enough(total_mass(&tree), 0.08202));
  CHECK(close_enough(total_metals(&tree), 0.000735));
  synthetic_tree_finish_snapshot(&tree);
  CHECK(synthetic_tree_advance(&tree) == -1);
  puts("snapshot 3: merger complete; one live galaxy with the expected reservoirs and history");
  puts("PASS: synthetic merger tree (4 snapshots, 6 halo records, 2 galaxies)");
  result = EXIT_SUCCESS;

cleanup:
  synthetic_tree_free(&tree);
  return result;
}
