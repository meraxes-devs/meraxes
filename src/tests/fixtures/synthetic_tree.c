#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "core/galaxies.h"
#include "physics/mergers.h"
#include "synthetic_tree.h"
#include "tree_flags.h"

static void set_halo(halo_t* halo, int snapshot, int index, double mass, int len, float x)
{
  halo->ID = (unsigned long)(100 * snapshot + index + 1);
  halo->Mvir = mass;
  halo->Rvir = 0.005 * cbrt(mass);
  halo->Vvir = sqrt(run_globals.G * mass / halo->Rvir);
  halo->Vmax = (float)halo->Vvir;
  halo->AngMom = (float)(0.04 * 1.414213562 * halo->Vvir * halo->Rvir);
  halo->Len = len;
  halo->Pos[0] = x;
  halo->Pos[1] = 0.25f;
  halo->Pos[2] = 0.25f;
  halo->SnapOffset = 1;
  halo->DescIndex = snapshot == SYNTHETIC_TREE_NSNAPS - 1 ? -1 : 0;
  halo->ProgIndex = snapshot == 0 ? -1 : index;
}

static void set_fof(fof_group_t* fof, halo_t* first, halo_t* second)
{
  fof->FirstHalo = first;
  fof->Mvir = first->Mvir + (second == NULL ? 0.0 : second->Mvir);
  fof->Rvir = 0.005 * cbrt(fof->Mvir);
  fof->Vvir = sqrt(run_globals.G * fof->Mvir / fof->Rvir);
  fof->FOFMvirModifier = 1.0;
  fof->TotalSubhaloLen = first->Len + (second == NULL ? 0 : second->Len);
  first->FOFGroup = fof;
  first->NextHaloInFOFGroup = second;
  if (second != NULL) {
    second->FOFGroup = fof;
    second->Type = 1;
  }
}

static void update_occupied_halos(synthetic_snapshot_t* snapshot)
{
  for (int i = 0; i < snapshot->n_fof_groups; i++) {
    fof_group_t* fof = &snapshot->fof_groups[i];
    fof->FirstOccupiedHalo = fof->FirstHalo;
    while (fof->FirstOccupiedHalo != NULL && fof->FirstOccupiedHalo->Galaxy == NULL)
      fof->FirstOccupiedHalo = fof->FirstOccupiedHalo->NextHaloInFOFGroup;
  }
}

int synthetic_tree_init(synthetic_tree_t* tree)
{
  memset(tree, 0, sizeof(*tree));
  memset(&run_globals, 0, sizeof(run_globals));
  run_globals.params.TreesID = GBPTREES_TREES;
  run_globals.params.NSteps = 1;
  run_globals.params.SnaplistLength = SYNTHETIC_TREE_NSNAPS;
  run_globals.params.BoxSize = 1.0; // cMpc/h; only used for periodic distances here.
  run_globals.params.Hubble_h = 0.7;
  run_globals.params.OmegaM = 1.0;
  run_globals.units.UnitMass_in_g = 1e10 * SOLAR_MASS;
  run_globals.units.UnitLength_in_cm = MPC;
  run_globals.units.UnitVelocity_in_cm_per_s = 1e5;
  run_globals.units.UnitTime_in_s = MPC / 1e5;
  run_globals.units.UnitTime_in_Megayears = run_globals.units.UnitTime_in_s / SEC_PER_MEGAYEAR;
  run_globals.G = GRAVITY / pow(MPC, 3) * run_globals.units.UnitMass_in_g * pow(run_globals.units.UnitTime_in_s, 2);
  run_globals.Hubble = HUBBLE * run_globals.units.UnitTime_in_s;

  physics_params_t* params = &run_globals.params.physics;
  params->Flag_IRA = 1;
  params->BlackHoleSeed = 1e-5;
  params->EscapeFracNorm = 0.2;
  params->EscapeFracBHNorm = 1.0;
  params->MergerTimeFactor = 1.0;
  params->MinMergerRatioForBurst = 1.0;
  // BH feedback, delayed feedback and stochastic scatter are off in this fixture.

  run_globals.AA = tree->scale_factors;
  run_globals.ZZ = tree->redshifts;
  run_globals.LTTime = tree->lookback_times;
  for (int s = 0; s < SYNTHETIC_TREE_NSNAPS; s++) {
    double a = 0.05 + 0.01 * s;
    tree->scale_factors[s] = a;
    tree->redshifts[s] = 1.0 / a - 1.0;
    // Exact lookback time for the fixture's OmegaM=1, OmegaLambda=0 cosmology.
    tree->lookback_times[s] = 2.0 / (3.0 * run_globals.Hubble) * (1.0 - pow(a, 1.5));
    synthetic_snapshot_t* snap = &tree->snapshots[s];
    snap->n_halos = s < 2 ? 2 : 1;
    snap->n_fof_groups = s == 0 ? 2 : 1;
    set_halo(&snap->halos[0], s, 0, 1.0 + 0.1 * s, 128 + 16 * s, 0.25f);
    if (s < 2)
      set_halo(&snap->halos[1], s, 1, 0.25, 32, s == 0 ? 0.5f : 0.254f);
    if (s == 0) {
      snap->halos[1].DescIndex = 1;
      set_fof(&snap->fof_groups[0], &snap->halos[0], NULL);
      set_fof(&snap->fof_groups[1], &snap->halos[1], NULL);
    } else {
      set_fof(&snap->fof_groups[0], &snap->halos[0], s == 1 ? &snap->halos[1] : NULL);
    }
  }
  tree->snapshots[1].halos[1].TreeFlags = TREE_CASE_MERGER;

  // new_galaxy() draws an inclination even without USE_STOCHASTICITY.
  run_globals.random_generator = gsl_rng_alloc(gsl_rng_mt19937);
  if (run_globals.random_generator == NULL)
    return -1;
  gsl_rng_set(run_globals.random_generator, 1809);
  for (int i = 0; i < SYNTHETIC_TREE_NGALAXIES; i++) {
    halo_t* halo = &tree->snapshots[0].halos[i];
    create_new_galaxy(0, halo, &tree->n_galaxies, &tree->n_created, &tree->n_halo_mergers);
    galaxy_t* gal = tree->galaxies[i] = halo->Galaxy;
    copy_halo_props_to_galaxy(halo, gal);
    gal->HotGas = i == 0 ? 0.02 : 0.005;
    gal->ColdGas = i == 0 ? 0.04 : 0.01;
    gal->EjectedGas = i == 0 ? 0.005 : 0.002;
    double metallicity = i == 0 ? 0.01 : 0.005;
    gal->MetalsHotGas = gal->HotGas * metallicity;
    gal->MetalsColdGas = gal->ColdGas * metallicity;
    gal->MetalsEjectedGas = gal->EjectedGas * metallicity;
#if USE_MINI_HALOS
    gal->Galaxy_Population = 2;
#endif
  }
  update_occupied_halos(&tree->snapshots[0]);
  return 0;
}

int synthetic_tree_advance(synthetic_tree_t* tree)
{
  int snapshot = tree->snapshot + 1;
  if (snapshot >= SYNTHETIC_TREE_NSNAPS)
    return -1;
  synthetic_snapshot_t* snap = &tree->snapshots[snapshot];

  // This restricted fixture has no skipped snapshots or dead galaxies before its last frame.
  for (int i = 0; i < SYNTHETIC_TREE_NGALAXIES; i++) {
    galaxy_t* gal = tree->galaxies[i];
    if (gal->Type >= 3 || gal->LastIdentSnap != tree->snapshot || gal->HaloDescIndex < 0 ||
        gal->HaloDescIndex >= snap->n_halos)
      return -1;
  }
  for (int i = 0; i < SYNTHETIC_TREE_NGALAXIES; i++) {
    galaxy_t* gal = tree->galaxies[i];
    gal->Halo = NULL;
    gal->ghost_flag = false;
    gal->SnapSkipCounter--;
    reset_galaxy_properties(gal, snapshot);
  }
  for (int i = 0; i < SYNTHETIC_TREE_NGALAXIES; i++) {
    galaxy_t* gal = tree->galaxies[i];
    gal->OldType = gal->Type;
    gal->dt = tree->lookback_times[gal->LastIdentSnap] - tree->lookback_times[snapshot];
    // An existing orphan travels with its parent's NextGalInHalo chain.
    if (gal->Type < 2)
      connect_galaxy_and_halo(gal, &snap->halos[gal->HaloDescIndex], &tree->n_halo_mergers);
  }
  for (int i = 0; i < SYNTHETIC_TREE_NGALAXIES; i++) {
    galaxy_t* gal = tree->galaxies[i];
    if (gal->Type == 2 && gal->MergerTarget == NULL) {
      gal->MergerTarget = gal->FirstGalInHalo;
      // Match dracarys.c: use the previous halo properties and compensate this step's decrement.
      gal->MergTime = calculate_merging_time(gal, snapshot) + gal->dt;
    }
  }
  update_occupied_halos(snap);
  for (int i = 0; i < SYNTHETIC_TREE_NGALAXIES; i++) {
    galaxy_t* gal = tree->galaxies[i];
    if (gal->Type < 2)
      copy_halo_props_to_galaxy(gal->Halo, gal);
  }
  tree->snapshot = snapshot;
  return 0;
}

void synthetic_tree_finish_snapshot(synthetic_tree_t* tree)
{
  for (int i = 0; i < SYNTHETIC_TREE_NGALAXIES; i++)
    tree->galaxies[i]->LastIdentSnap = tree->snapshot;
}

void synthetic_tree_free(synthetic_tree_t* tree)
{
  for (int i = 0; i < SYNTHETIC_TREE_NGALAXIES; i++)
    free(tree->galaxies[i]);
  if (run_globals.random_generator != NULL)
    gsl_rng_free(run_globals.random_generator);
  memset(&run_globals, 0, sizeof(run_globals));
  memset(tree, 0, sizeof(*tree));
}
