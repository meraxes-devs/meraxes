#include <assert.h>
#include <hdf5_hl.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "XRayHeatingFunctions.h"
#include "dist_func.h"
#include "magnitudes.h"
#include "meraxes.h"
#include "parse_paramfile.h"
#include "physics/blackhole_feedback.h"
#include "reionization.h"
#include "save.h"
#if USE_MINI_HALOS
#include "metal_evo.h"
#endif

static float current_mwmsa(galaxy_t* gal, int i_snap)
{
  double* LTTime = run_globals.LTTime;
  double mwmsa_num = gal->mwmsa_num;
  double mwmsa_denom = gal->mwmsa_denom;
  int snapshot = run_globals.ListOutputSnaps[i_snap];

  for (int ii = 0, jj = snapshot; (ii < N_HISTORY_SNAPS) && (jj >= 0); ii++, jj--) {
    mwmsa_num += gal->NewStars[ii] * LTTime[jj];
    mwmsa_denom += gal->NewStars[ii];
  }
  if (mwmsa_denom > 0)
    return (float)((mwmsa_num / mwmsa_denom) - LTTime[snapshot]);
  else
    return 0;
}

void prepare_galaxy_for_output(galaxy_t gal, galaxy_output_t* galout, int i_snap)
{
  run_units_t* units = &(run_globals.units);

  galout->ID = gal.ID;
  galout->Type = gal.Type;
  if (!gal.ghost_flag) {
    galout->HaloID = (long long)gal.Halo->ID;
    galout->CentralGal = gal.Halo->FOFGroup->FirstOccupiedHalo->Galaxy->output_index;
    galout->FOFMvir = (float)(gal.Halo->FOFGroup->Mvir);
  } else {
    galout->HaloID = -1;
    galout->CentralGal = -1;
    galout->FOFMvir = (float)-1.0;
  }
  galout->GhostFlag = (int)gal.ghost_flag;

  for (int ii = 0; ii < 3; ii++) {
    galout->Pos[ii] = gal.Pos[ii];
    galout->Vel[ii] = gal.Vel[ii];
  }

  galout->Len = gal.Len;
  galout->MaxLen = gal.MaxLen;
  galout->Mvir = (float)(gal.Mvir);
  galout->Rvir = (float)(gal.Rvir);
  galout->Vvir = (float)(gal.Vvir);
  galout->Vmax = (float)(gal.Vmax);
  galout->Spin = (float)(gal.Spin);
  galout->HotGas = (float)(gal.HotGas);
  galout->MetalsHotGas = (float)(gal.MetalsHotGas);
  galout->ColdGas = (float)(gal.ColdGas);
  galout->MetalsColdGas = (float)(gal.MetalsColdGas);
  galout->H2Frac = (float)(gal.H2Frac);
  galout->H2Mass = (float)(gal.H2Mass);
  galout->HIMass = (float)(gal.HIMass);
  galout->Mcool = (float)(gal.Mcool);
  galout->StellarMass = (float)(gal.StellarMass);
  galout->GrossStellarMass = (float)(gal.GrossStellarMass);
  galout->Fesc = (float)(gal.Fesc);
  galout->FescWeightedGSM = (float)(gal.FescWeightedGSM);
  galout->BlackHoleMass = (float)(gal.BlackHoleMass);
  galout->FescBH = (float)(gal.FescBH);
  galout->BHemissivity = (float)(gal.BHemissivity);
  galout->QuasarMag = (gal.QuasarLuv > 0.0) ? (float)(-19.07395 - 2.5 * log10(gal.QuasarLuv)) : 999.9f;
  galout->QuasarLX = (float)gal.QuasarLX;
  galout->NHbin = gal.NHbin;
  galout->BHXrayEmissivity = (float)gal.BHXrayEmissivity_hard;
  galout->DutyCycleAGN = (float)(gal.DutyCycleAGN);
  galout->EffectiveBHM = (float)(gal.EffectiveBHM);
  galout->BlackHoleAccretedHotMass = (float)(gal.BlackHoleAccretedHotMass);
  galout->BlackHoleAccretedColdMass = (float)(gal.BlackHoleAccretedColdMass);
  galout->DiskScaleLength = (float)(gal.DiskScaleLength);
  galout->MetalsStellarMass = (float)(gal.MetalsStellarMass);
  galout->Sfr = (float)(gal.Sfr * units->UnitMass_in_g / units->UnitTime_in_s * SEC_PER_YEAR / SOLAR_MASS);
  galout->LOIII = (float)gal.LOIII; // gal.LOIII is already stored in units of 1e40 erg/s
  galout->ionization_param = (float)(gal.ionization_param);
  galout->FescWeightedSfr =
    (float)(gal.FescWeightedSfr * units->UnitMass_in_g / units->UnitTime_in_s * SEC_PER_YEAR / SOLAR_MASS);
  galout->EjectedGas = (float)(gal.EjectedGas);
  galout->MetalsEjectedGas = (float)(gal.MetalsEjectedGas);
  galout->Rcool = (float)(gal.Rcool);
  galout->Cos_Inc = (float)(gal.Cos_Inc);
  galout->BaryonFracModifier = (float)(gal.BaryonFracModifier);
  galout->FOFMvirModifier = (float)(gal.FOFMvirModifier);
  galout->MvirCrit = (float)(gal.MvirCrit);
  galout->tau_cgm = (float)(gal.tau_cgm);
  galout->dt = (float)(gal.dt * units->UnitTime_in_Megayears);
  galout->MergerBurstMass = (float)(gal.MergerBurstMass);
  galout->MergTime = (float)(gal.MergTime * units->UnitTime_in_Megayears);
  galout->MergerStartRadius = (float)(gal.MergerStartRadius);
  galout->MWMSA = current_mwmsa(&gal, i_snap);

#if USE_MINI_HALOS
  galout->GrossStellarMassIII = (float)(gal.GrossStellarMassIII);
  galout->FescIII = (float)(gal.FescIII);
  galout->FescIIIWeightedGSM = (float)(gal.FescIIIWeightedGSM);
  galout->FescIIIWeightedSfr =
    (float)(gal.FescIIIWeightedSfr * units->UnitMass_in_g / units->UnitTime_in_s * SEC_PER_YEAR / SOLAR_MASS);
  galout->SfrIII = (float)(gal.SfrIII * units->UnitMass_in_g / units->UnitTime_in_s * SEC_PER_YEAR / SOLAR_MASS);

  galout->MvirCrit_MC = (float)(gal.MvirCrit_MC);

  galout->RmetalBubble = (float)(gal.RmetalBubble); // new for MetalEvo
  galout->Galaxy_Population = (int)(gal.Galaxy_Population);
  galout->Flag_ExtMetEnr = (int)(gal.Flag_ExtMetEnr);
  galout->Metal_Probability = (float)(gal.Metal_Probability);
  galout->GalMetal_Probability = (float)(gal.GalMetal_Probability);
  galout->StellarMass_II = (float)(gal.StellarMass_II);
  galout->StellarMass_III = (float)(gal.StellarMass_III);
  galout->Remnant_Mass = (float)(gal.Remnant_Mass);
#endif

  for (int ii = 0; ii < N_HISTORY_SNAPS; ii++) {
    galout->NewStars[ii] = (float)(gal.NewStars[ii]);
#if USE_MINI_HALOS
    galout->NewStars_III[ii] = (float)(gal.NewStars_III[ii]);
    galout->NewStars_II[ii] = (float)(gal.NewStars_II[ii]);
#endif
  }

#ifdef CALC_MAGS
  galout->LOIII_dusty = galout->LOIII;
  get_output_magnitudes(galout->Mags, galout->DustyMags, &gal, run_globals.ListOutputSnaps[i_snap]);

  const int loiii_band_idx = run_globals.loiii_rest_band_mag_index;
  if (loiii_band_idx >= 0) {
    const float mag = galout->Mags[loiii_band_idx];
    const float dusty_mag = galout->DustyMags[loiii_band_idx];

    if (isfinite(mag) && isfinite(dusty_mag) && mag < 900.0f && dusty_mag < 900.0f) {
      const double attenuation_mag = (double)mag - (double)dusty_mag;
      const double attenuation_factor = pow(10.0, 0.4 * attenuation_mag);
      const double loiii_dusty = (double)galout->LOIII * attenuation_factor;

      if (isfinite(loiii_dusty) && loiii_dusty >= 0.0)
        galout->LOIII_dusty = (float)loiii_dusty;
      else
        galout->LOIII_dusty = 0.0f;
    }
  }

#if USE_MINI_HALOS
  get_output_magnitudesIII(galout->MagsIII, &gal, run_globals.ListOutputSnaps[i_snap]);
#endif
#endif
}

void calc_hdf5_props()
{
  /*
   * Prepare an HDF5 to receive the output galaxy data.
   * Here we store the data in an hdf5 table for easily appending new data.
   */

  if (!run_globals.params.FlagMCMC) {
    hdf5_output_t* h5props = &(run_globals.hdf5props);
    galaxy_output_t galout;
    int i; // dummy

    h5props->n_props = 58; /* 57 base + 1: NHbin */
#if USE_MINI_HALOS
    h5props->n_props += 16;
#endif

#ifdef CALC_MAGS
    h5props->n_props += 3;
    h5props->array_nmag_f_tid = H5Tarray_create(H5T_NATIVE_FLOAT, 1, (hsize_t[]){ MAGS_N_BANDS });
#if USE_MINI_HALOS
    h5props->n_props += 1;
#endif
#endif

    // Size of a single galaxy entry.
    h5props->dst_size = sizeof(galaxy_output_t);

    // Create datatypes for different size arrays
    h5props->array3f_tid = H5Tarray_create(H5T_NATIVE_FLOAT, 1, (hsize_t[]){ 3 });
    h5props->array_nhist_f_tid = H5Tarray_create(H5T_NATIVE_FLOAT, 1, (hsize_t[]){ N_HISTORY_SNAPS });

    // Calculate the offsets of our struct members in memory
    h5props->dst_offsets = malloc(sizeof(size_t) * h5props->n_props);
    // Calculate the sizes of our struct members in memory.
    h5props->dst_field_sizes = malloc(sizeof(size_t) * h5props->n_props);
    // Give each galaxy property a field name in the table
    h5props->field_names = malloc(sizeof(const char*) * h5props->n_props);
    // Assign a type to each galaxy property field in the table.
    h5props->field_types = malloc(sizeof(hid_t) * h5props->n_props);
    // Store the **output** units of each property for writing to the master file.
    // Units should be compatible with the python astropy.units module.
    h5props->field_units = malloc(sizeof(const char*) * h5props->n_props);
    // Store the **output** h conversion for each property.  The string will be
    // parsed by python eval(), substituting h for the appropriate value at read
    // time and v for the property value.
    h5props->field_h_conv = malloc(sizeof(const char*) * h5props->n_props);

    i = 0;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, HaloID);
    h5props->dst_field_sizes[i] = sizeof(galout.HaloID);
    h5props->field_names[i] = "HaloID";
    h5props->field_units[i] = "None";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_LLONG;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, ID);
    h5props->dst_field_sizes[i] = sizeof(galout.ID);
    h5props->field_names[i] = "ID";
    h5props->field_units[i] = "None";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_LLONG;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, Type);
    h5props->dst_field_sizes[i] = sizeof(galout.Type);
    h5props->field_names[i] = "Type";
    h5props->field_units[i] = "None";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_INT;

#if USE_MINI_HALOS
    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, Galaxy_Population);
    h5props->dst_field_sizes[i] = sizeof(galout.Galaxy_Population);
    h5props->field_names[i] = "Galaxy_Population";
    h5props->field_units[i] = "None";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_INT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, Flag_ExtMetEnr);
    h5props->dst_field_sizes[i] = sizeof(galout.Flag_ExtMetEnr);
    h5props->field_names[i] = "Flag_ExtMetEnr";
    h5props->field_units[i] = "None";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_INT;
#endif

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, CentralGal);
    h5props->dst_field_sizes[i] = sizeof(galout.CentralGal);
    h5props->field_names[i] = "CentralGal";
    h5props->field_units[i] = "None";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_INT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, GhostFlag);
    h5props->dst_field_sizes[i] = sizeof(galout.GhostFlag);
    h5props->field_names[i] = "GhostFlag";
    h5props->field_units[i] = "None";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_INT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, Len);
    h5props->dst_field_sizes[i] = sizeof(galout.Len);
    h5props->field_names[i] = "Len";
    h5props->field_units[i] = "None";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_INT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, MaxLen);
    h5props->dst_field_sizes[i] = sizeof(galout.MaxLen);
    h5props->field_names[i] = "MaxLen";
    h5props->field_units[i] = "None";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_INT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, Pos);
    h5props->dst_field_sizes[i] = sizeof(galout.Pos);
    h5props->field_names[i] = "Pos";
    h5props->field_units[i] = "Mpc"; // comoving
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = h5props->array3f_tid;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, Vel);
    h5props->dst_field_sizes[i] = sizeof(galout.Vel);
    h5props->field_names[i] = "Vel";
    h5props->field_units[i] = "km/s"; // comoving
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = h5props->array3f_tid;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, Spin);
    h5props->dst_field_sizes[i] = sizeof(galout.Spin);
    h5props->field_names[i] = "Spin";
    h5props->field_units[i] = "None";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, Mvir);
    h5props->dst_field_sizes[i] = sizeof(galout.Mvir);
    h5props->field_names[i] = "Mvir";
    h5props->field_units[i] = "1e10 solMass";
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, Rvir);
    h5props->dst_field_sizes[i] = sizeof(galout.Rvir);
    h5props->field_names[i] = "Rvir";
    h5props->field_units[i] = "Mpc"; // physical
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, Vvir);
    h5props->dst_field_sizes[i] = sizeof(galout.Vvir);
    h5props->field_names[i] = "Vvir";
    h5props->field_units[i] = "km/s"; // physical
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, Vmax);
    h5props->dst_field_sizes[i] = sizeof(galout.Vmax);
    h5props->field_names[i] = "Vmax";
    h5props->field_units[i] = "km/s"; // physical
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, FOFMvir);
    h5props->dst_field_sizes[i] = sizeof(galout.FOFMvir);
    h5props->field_names[i] = "FOFMvir";
    h5props->field_units[i] = "1e10 solMass";
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, HotGas);
    h5props->dst_field_sizes[i] = sizeof(galout.HotGas);
    h5props->field_names[i] = "HotGas";
    h5props->field_units[i] = "1e10 solMass";
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, MetalsHotGas);
    h5props->dst_field_sizes[i] = sizeof(galout.MetalsHotGas);
    h5props->field_names[i] = "MetalsHotGas";
    h5props->field_units[i] = "1e10 solMass";
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, ColdGas);
    h5props->dst_field_sizes[i] = sizeof(galout.ColdGas);
    h5props->field_names[i] = "ColdGas";
    h5props->field_units[i] = "1e10 solMass";
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, MetalsColdGas);
    h5props->dst_field_sizes[i] = sizeof(galout.MetalsColdGas);
    h5props->field_names[i] = "MetalsColdGas";
    h5props->field_units[i] = "1e10 solMass";
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, H2Frac);
    h5props->dst_field_sizes[i] = sizeof(galout.H2Frac);
    h5props->field_names[i] = "H2Frac";
    h5props->field_units[i] = "None";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, H2Mass);
    h5props->dst_field_sizes[i] = sizeof(galout.H2Mass);
    h5props->field_names[i] = "H2Mass";
    h5props->field_units[i] = "1e10 solMass";
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, HIMass);
    h5props->dst_field_sizes[i] = sizeof(galout.HIMass);
    h5props->field_names[i] = "HIMass";
    h5props->field_units[i] = "1e10 solMass";
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, Mcool);
    h5props->dst_field_sizes[i] = sizeof(galout.Mcool);
    h5props->field_names[i] = "Mcool";
    h5props->field_units[i] = "1e10 solMass";
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, DiskScaleLength);
    h5props->dst_field_sizes[i] = sizeof(galout.DiskScaleLength);
    h5props->field_names[i] = "DiskScaleLength";
    h5props->field_units[i] = "Mpc"; // physical
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, StellarMass);
    h5props->dst_field_sizes[i] = sizeof(galout.StellarMass);
    h5props->field_names[i] = "StellarMass";
    h5props->field_units[i] = "1e10 solMass";
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

#if USE_MINI_HALOS
    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, StellarMass_II);
    h5props->dst_field_sizes[i] = sizeof(galout.StellarMass_II);
    h5props->field_names[i] = "Pop2StellarMass";
    h5props->field_units[i] = "1e10 solMass";
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, StellarMass_III);
    h5props->dst_field_sizes[i] = sizeof(galout.StellarMass_III);
    h5props->field_names[i] = "Pop3StellarMass";
    h5props->field_units[i] = "1e10 solMass";
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, Remnant_Mass);
    h5props->dst_field_sizes[i] = sizeof(galout.Remnant_Mass);
    h5props->field_names[i] = "RemnantMass";
    h5props->field_units[i] = "1e10 solMass";
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, GrossStellarMassIII);
    h5props->dst_field_sizes[i] = sizeof(galout.GrossStellarMassIII);
    h5props->field_names[i] = "GrossStellarMassIII";
    h5props->field_units[i] = "1e10 solMass";
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, FescIIIWeightedSfr);
    h5props->dst_field_sizes[i] = sizeof(galout.FescIIIWeightedSfr);
    h5props->field_names[i] = "FescIIIWeightedSfr";
    h5props->field_units[i] = "solMass/yr";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, SfrIII);
    h5props->dst_field_sizes[i] = sizeof(galout.SfrIII);
    h5props->field_names[i] = "SfrIII";
    h5props->field_units[i] = "solMass/yr";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;
#endif

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, GrossStellarMass);
    h5props->dst_field_sizes[i] = sizeof(galout.GrossStellarMass);
    h5props->field_names[i] = "GrossStellarMass";
    h5props->field_units[i] = "1e10 solMass";
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, MetalsStellarMass);
    h5props->dst_field_sizes[i] = sizeof(galout.MetalsStellarMass);
    h5props->field_names[i] = "MetalsStellarMass";
    h5props->field_units[i] = "1e10 solMass";
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, Sfr);
    h5props->dst_field_sizes[i] = sizeof(galout.Sfr);
    h5props->field_names[i] = "Sfr";
    h5props->field_units[i] = "solMass/yr";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, LOIII);
    h5props->dst_field_sizes[i] = sizeof(galout.LOIII);
    h5props->field_names[i] = "LOIII";
    h5props->field_units[i] = "1e40 erg/s";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

#ifdef CALC_MAGS
    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, LOIII_dusty);
    h5props->dst_field_sizes[i] = sizeof(galout.LOIII_dusty);
    h5props->field_names[i] = "LOIII_dusty";
    h5props->field_units[i] = "1e40 erg/s";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;
#endif

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, ionization_param);
    h5props->dst_field_sizes[i] = sizeof(galout.ionization_param);
    h5props->field_names[i] = "ionization_param";
    h5props->field_units[i] = "cm/s";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, FescWeightedSfr);
    h5props->dst_field_sizes[i] = sizeof(galout.FescWeightedSfr);
    h5props->field_names[i] = "FescWeightedSfr";
    h5props->field_units[i] = "solMass/yr";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

#ifdef CALC_MAGS
    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, Mags);
    h5props->dst_field_sizes[i] = sizeof(galout.Mags);
    h5props->field_names[i] = "Mags";
    h5props->field_units[i] = "mag";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = h5props->array_nmag_f_tid;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, DustyMags);
    h5props->dst_field_sizes[i] = sizeof(galout.DustyMags);
    h5props->field_names[i] = "DustyMags";
    h5props->field_units[i] = "mag";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = h5props->array_nmag_f_tid;

#if USE_MINI_HALOS
    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, MagsIII);
    h5props->dst_field_sizes[i] = sizeof(galout.MagsIII);
    h5props->field_names[i] = "MagsIII";
    h5props->field_units[i] = "mag";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = h5props->array_nmag_f_tid;
#endif
#endif

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, EjectedGas);
    h5props->dst_field_sizes[i] = sizeof(galout.EjectedGas);
    h5props->field_names[i] = "EjectedGas";
    h5props->field_units[i] = "1e10 solMass";
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, MetalsEjectedGas);
    h5props->dst_field_sizes[i] = sizeof(galout.MetalsEjectedGas);
    h5props->field_names[i] = "MetalsEjectedGas";
    h5props->field_units[i] = "1e10 solMass";
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

#if USE_MINI_HALOS
    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, RmetalBubble);
    h5props->dst_field_sizes[i] = sizeof(galout.RmetalBubble);
    h5props->field_names[i] = "RmetalBubble";
    h5props->field_units[i] = "Mpc"; // physical
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, Metal_Probability);
    h5props->dst_field_sizes[i] = sizeof(galout.Metal_Probability);
    h5props->field_names[i] = "MetalProbability";
    h5props->field_units[i] = "None";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, GalMetal_Probability);
    h5props->dst_field_sizes[i] = sizeof(galout.GalMetal_Probability);
    h5props->field_names[i] = "GalMetalProbability";
    h5props->field_units[i] = "None";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;
#endif

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, BlackHoleMass);
    h5props->dst_field_sizes[i] = sizeof(galout.BlackHoleMass);
    h5props->field_names[i] = "BlackHoleMass";
    h5props->field_units[i] = "1e10 solMass";
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, Rcool);
    h5props->dst_field_sizes[i] = sizeof(galout.Rcool);
    h5props->field_names[i] = "Rcool";
    h5props->field_units[i] = "Mpc"; // physical
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, Cos_Inc);
    h5props->dst_field_sizes[i] = sizeof(galout.Cos_Inc);
    h5props->field_names[i] = "Cos_Inc";
    h5props->field_units[i] = "None";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, MergTime);
    h5props->dst_field_sizes[i] = sizeof(galout.MergTime);
    h5props->field_names[i] = "MergTime";
    h5props->field_units[i] = "Myr";
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, MergerStartRadius);
    h5props->dst_field_sizes[i] = sizeof(galout.MergerStartRadius);
    h5props->field_names[i] = "MergerStartRadius";
    h5props->field_units[i] = "Mpc";
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, BaryonFracModifier);
    h5props->dst_field_sizes[i] = sizeof(galout.BaryonFracModifier);
    h5props->field_names[i] = "BaryonFracModifier";
    h5props->field_units[i] = "None";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, FOFMvirModifier);
    h5props->dst_field_sizes[i] = sizeof(galout.FOFMvirModifier);
    h5props->field_names[i] = "FOFMvirModifier";
    h5props->field_units[i] = "None";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, MvirCrit);
    h5props->dst_field_sizes[i] = sizeof(galout.MvirCrit);
    h5props->field_names[i] = "MvirCrit";
    h5props->field_units[i] = "1e10 solMass";
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

#if USE_MINI_HALOS
    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, MvirCrit_MC);
    h5props->dst_field_sizes[i] = sizeof(galout.MvirCrit_MC);
    h5props->field_names[i] = "MvirCrit_MC";
    h5props->field_units[i] = "1e10 solMass";
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;
#endif

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, tau_cgm);
    h5props->dst_field_sizes[i] = sizeof(galout.tau_cgm);
    h5props->field_names[i] = "tau_cgm";
    h5props->field_units[i] = "None";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, MergerBurstMass);
    h5props->dst_field_sizes[i] = sizeof(galout.MergerBurstMass);
    h5props->field_names[i] = "MergerBurstMass";
    h5props->field_units[i] = "1e10 solMass";
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, MWMSA);
    h5props->dst_field_sizes[i] = sizeof(galout.MWMSA);
    h5props->field_names[i] = "MWMSA";
    h5props->field_units[i] = "Myr";
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, NewStars);
    h5props->dst_field_sizes[i] = sizeof(galout.NewStars);
    h5props->field_names[i] = "NewStars";
    h5props->field_units[i] = "1e10 solMass";
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = h5props->array_nhist_f_tid;

#if USE_MINI_HALOS
    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, NewStars_II);
    h5props->dst_field_sizes[i] = sizeof(galout.NewStars_II);
    h5props->field_names[i] = "NewStarsPop2";
    h5props->field_units[i] = "1e10 solMass";
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = h5props->array_nhist_f_tid;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, NewStars_III);
    h5props->dst_field_sizes[i] = sizeof(galout.NewStars_III);
    h5props->field_names[i] = "NewStarsPop3";
    h5props->field_units[i] = "1e10 solMass";
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = h5props->array_nhist_f_tid;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, FescIII);
    h5props->dst_field_sizes[i] = sizeof(galout.FescIII);
    h5props->field_names[i] = "FescIII";
    h5props->field_units[i] = "None";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, FescIIIWeightedGSM);
    h5props->dst_field_sizes[i] = sizeof(galout.FescIIIWeightedGSM);
    h5props->field_names[i] = "FescIIIWeightedGSM";
    h5props->field_units[i] = "1e10 solMass";
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;
#endif

    // Blackhole or Emissivity related
    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, Fesc);
    h5props->dst_field_sizes[i] = sizeof(galout.Fesc);
    h5props->field_names[i] = "Fesc";
    h5props->field_units[i] = "None";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, FescWeightedGSM);
    h5props->dst_field_sizes[i] = sizeof(galout.FescWeightedGSM);
    h5props->field_names[i] = "FescWeightedGSM";
    h5props->field_units[i] = "1e10 solMass";
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, FescBH);
    h5props->dst_field_sizes[i] = sizeof(galout.FescBH);
    h5props->field_names[i] = "FescBH";
    h5props->field_units[i] = "None";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, BHemissivity);
    h5props->dst_field_sizes[i] = sizeof(galout.BHemissivity);
    h5props->field_names[i] = "BHemissivity";
    h5props->field_units[i] = "1e60 photons";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, QuasarMag);
    h5props->dst_field_sizes[i] = sizeof(galout.QuasarMag);
    h5props->field_names[i] = "QuasarMag";
    h5props->field_units[i] = "AB mag (M1450)";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, QuasarLX);
    h5props->dst_field_sizes[i] = sizeof(galout.QuasarLX);
    h5props->field_names[i] = "QuasarLX";
    h5props->field_units[i] = "LX [1e10 L_sun, 2-10 keV, intrinsic]";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, NHbin);
    h5props->dst_field_sizes[i] = sizeof(galout.NHbin);
    h5props->field_names[i] = "NHbin";
    h5props->field_units[i] = "0-4 = drawn logNH bin (20-21/21-22/22-23/23-24/24-26 CTK); -1 = no AGN";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_INT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, BHXrayEmissivity);
    h5props->dst_field_sizes[i] = sizeof(galout.BHXrayEmissivity);
    h5props->field_names[i] = "BHXrayEmissivity";
    h5props->field_units[i] = "LX [1e10 L_sun, 2-10 keV, observed/obscured]";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, DutyCycleAGN);
    h5props->dst_field_sizes[i] = sizeof(galout.DutyCycleAGN);
    h5props->field_names[i] = "DutyCycleAGN";
    h5props->field_units[i] = "None";
    h5props->field_h_conv[i] = "None";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, EffectiveBHM);
    h5props->dst_field_sizes[i] = sizeof(galout.EffectiveBHM);
    h5props->field_names[i] = "EffectiveBHM";
    h5props->field_units[i] = "1e10 solMass";
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, BlackHoleAccretedHotMass);
    h5props->dst_field_sizes[i] = sizeof(galout.BlackHoleAccretedHotMass);
    h5props->field_names[i] = "BlackHoleAccretedHotMass";
    h5props->field_units[i] = "1e10 solMass";
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, BlackHoleAccretedColdMass);
    h5props->dst_field_sizes[i] = sizeof(galout.BlackHoleAccretedColdMass);
    h5props->field_names[i] = "BlackHoleAccretedColdMass";
    h5props->field_units[i] = "1e10 solMass";
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    h5props->dst_offsets[i] = HOFFSET(galaxy_output_t, dt);
    h5props->dst_field_sizes[i] = sizeof(galout.dt);
    h5props->field_names[i] = "dt";
    h5props->field_units[i] = "Myr";
    h5props->field_h_conv[i] = "v/h";
    h5props->field_types[i++] = H5T_NATIVE_FLOAT;

    // DEBUG
    if (i != h5props->n_props) {
      mlog_error("Incorrect number of galaxy properties in HDF5 file. Should be %d, but is %d", h5props->n_props, i);
      ABORT(EXIT_FAILURE);
    }
  }
}

void prep_hdf5_file()
{
  /*
   * Create the per-rank galaxy output file and write the two root attributes
   * (iCore, NCores).  The file is stored in run_globals.output_file_id and
   * kept open for the lifetime of the run so that HDF5 1.10.x's free-space
   * manager never accumulates the open-close-cycle fragmentation that causes
   * the "duplicate entry in cache" / H5Acreate2 failure when many snapshots
   * have been written to the same file.  The caller (dracarys) must call
   * close_hdf5_file() after the snapshot loop.
   */
  if (access(run_globals.FNameOut, F_OK) != -1)
    remove(run_globals.FNameOut);
  run_globals.output_file_id = H5Fcreate(run_globals.FNameOut, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  // store the file number and total number of cores
  H5LTset_attribute_int(run_globals.output_file_id, "/", "iCore", &(run_globals.mpi_rank), 1);
  H5LTset_attribute_int(run_globals.output_file_id, "/", "NCores", &(run_globals.mpi_size), 1);

  // Do NOT close the file here; write_snapshot() will reuse output_file_id.
}

void close_hdf5_file()
{
  /*
   * Flush and close the per-rank galaxy HDF5 file that was kept open since
   * prep_hdf5_file().  Call this once at the end of the dracarys snapshot
   * loop (or after FlagInteractive/FlagMCMC mode is done).
   */
  if (H5Iis_valid(run_globals.output_file_id)) {
    H5Fflush(run_globals.output_file_id, H5F_SCOPE_GLOBAL);
    H5Fclose(run_globals.output_file_id);
    run_globals.output_file_id = H5I_INVALID_HID;
  }
}

void create_master_file()
{
  hid_t file_id, group_id;
  char fname[STRLEN * 2 + 7];
  hdf5_output_t* h5props = &(run_globals.hdf5props);
  char** params_tag = h5props->params_tag;
  void** params_addr = h5props->params_addr;
  int* params_type = h5props->params_type;
  int params_count = h5props->params_count;

  mlog("Creating master file...", MLOG_OPEN | MLOG_TIMERSTART);

  // Create a new file
  sprintf(fname, "%s/%s.hdf5", run_globals.params.OutputDir, run_globals.params.FileNameGalaxies);
  if (access(fname, F_OK) != -1)
    remove(fname);
  file_id = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  // Open the group
  {
    const char* group_name = { "InputParams" };
    group_id = H5Gcreate(file_id, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Save all of the input params
    for (int ii = 0; (ii < params_count) && (params_type[ii] != PARAM_TYPE_UNUSED); ii++)
      switch (params_type[ii]) {
        case PARAM_TYPE_STRING:
          H5LTset_attribute_string(file_id, group_name, params_tag[ii], params_addr[ii]);
          break;
        case PARAM_TYPE_INT:
          H5LTset_attribute_int(file_id, group_name, params_tag[ii], params_addr[ii], 1);
          break;
        case PARAM_TYPE_DOUBLE:
          H5LTset_attribute_double(file_id, group_name, params_tag[ii], params_addr[ii], 1);
          break;
        case PARAM_TYPE_FLOAT:
          H5LTset_attribute_float(file_id, group_name, params_tag[ii], params_addr[ii], 1);
          break;
        case PARAM_TYPE_LONGLONG:
          H5LTset_attribute_long_long(file_id, group_name, params_tag[ii], params_addr[ii], 1);
          break;
        default:
          ABORT(EXIT_FAILURE);
          break;
      }

    // extra params which may be set internally should be written here
    if (run_globals.params.Flag_ConstructLightcone) {
      H5LTset_attribute_int(file_id, group_name, "EndSnapshotLightcone", &(run_globals.params.EndSnapshotLightcone), 1);
    }

    // Close the group
    H5Gclose(group_id);
  }

  // save the units of each galaxy property and grid
  {
    const char* group_name[2] = { "Units", "HubbleConversions" };
    for (int ii = 0; ii < 2; ii++) {
      group_id = H5Gcreate(file_id, group_name[ii], H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Gclose(group_id);
    }

    for (int ii = 0; ii < h5props->n_props; ii++) {
      H5LTset_attribute_string(file_id, group_name[0], h5props->field_names[ii], h5props->field_units[ii]);
      H5LTset_attribute_string(file_id, group_name[1], h5props->field_names[ii], h5props->field_h_conv[ii]);
    }
  }

#if USE_MINI_HALOS
  if (run_globals.params.Flag_IncludeMetalEvo) {
    const char* group_name = { "Units/MetalGrids" };
    group_id = H5Gcreate(file_id, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5LTset_attribute_string(file_id, group_name, "mass_metals", "solMass");
    H5LTset_attribute_string(file_id, group_name, "mass_gas", "solMass");
    H5LTset_attribute_string(file_id, group_name, "Zigm_box", "Zsol");
    H5LTset_attribute_string(file_id, group_name, "Probability_metals", "none");
    H5LTset_attribute_string(file_id, group_name, "N_bubbles", "none");
    H5LTset_attribute_string(file_id, group_name, "R_ave", "cMpc");
    H5LTset_attribute_string(file_id, group_name, "R_max", "cMpc");
    H5LTset_attribute_string(file_id, group_name, "mass_IGM", "solMass");

    H5Gclose(group_id);
  }
#endif

  if (run_globals.params.Flag_PatchyReion) {
    {
      const char* group_name = { "Units/Grids" };
      group_id = H5Gcreate(file_id, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5LTset_attribute_string(file_id, group_name, "xH", "None");
      H5LTset_attribute_string(file_id, group_name, "Gamma12", "1e-12 /s");
      H5LTset_attribute_string(file_id, group_name, "J_21_at_ionization", "10e-21 erg/s/Hz/cm/cm/sr");
      H5LTset_attribute_string(file_id, group_name, "z_at_ionization", "None");
      H5LTset_attribute_string(file_id, group_name, "r_bubble", "Mpc");
      H5LTset_attribute_string(file_id, group_name, "Mvir_crit", "1e10 solMass");
      H5LTset_attribute_string(file_id, group_name, "effective_bhar", "solMass/yr");
      H5LTset_attribute_string(file_id, group_name, "stars", "1e10 solMass");
      H5LTset_attribute_string(file_id, group_name, "weighted_sfr", "solMass/yr");
      H5LTset_attribute_string(file_id, group_name, "deltax", "None");
      H5LTset_attribute_string(file_id, group_name, "residual_xH", "1e4");
      H5LTset_attribute_string(file_id, group_name, "clumping_factor", "None");
      H5LTset_attribute_string(file_id, group_name, "temp_kinetic_all_gas", "K");
      H5LTset_attribute_string(file_id, group_name, "N_rec", "None");
      H5LTset_attribute_string(file_id, group_name, "t_resp", "Myr");

#if USE_MINI_HALOS
      if (run_globals.params.Flag_IncludeLymanWerner) {
        H5LTset_attribute_string(file_id, group_name, "starsIII", "1e10 solMass");
        H5LTset_attribute_string(file_id, group_name, "weighted_sfrIII", "solMass/yr");
        H5LTset_attribute_string(file_id, group_name, "JLW_box", "1e-21 erg/s/Hz/cm/cm/sr");
        H5LTset_attribute_string(file_id, group_name, "JLW_box_II", "1e-21 erg/s/Hz/cm/cm/sr");
        H5LTset_attribute_string(file_id, group_name, "Mvir_crit_MC", "1e10 solMass");
      }
#endif

      if (run_globals.params.Flag_ConstructLightcone) {
        H5LTset_attribute_string(file_id, group_name, "LightconeBox", "mK");
      }

      if (run_globals.params.Flag_IncludeSpinTemp) {
        H5LTset_attribute_string(file_id, group_name, "Ts_box", "K");
        H5LTset_attribute_string(file_id, group_name, "Tk_box", "K");
        H5LTset_attribute_string(file_id, group_name, "Ts_boxII", "K");
        H5LTset_attribute_string(file_id, group_name, "Tk_boxII", "K");
        H5LTset_attribute_string(file_id, group_name, "x_e_box", "None");
      }

      if (run_globals.params.Flag_Compute21cmBrightTemp) {
        H5LTset_attribute_string(file_id, group_name, "delta_T", "mK");
        H5LTset_attribute_string(file_id, group_name, "delta_TII", "mK");
      }

      if (run_globals.params.Flag_ComputePS) {
        H5LTset_attribute_string(file_id, group_name, "PS_data", "mK2");
        H5LTset_attribute_string(file_id, group_name, "PS_error", "mK2");
        H5LTset_attribute_string(file_id, group_name, "PSII_data", "mK2");
        H5LTset_attribute_string(file_id, group_name, "PSII_error", "mK2");
      }

      H5Gclose(group_id);
    }

    {
      const char* group_name = { "HubbleConversions/Grids" };
      group_id = H5Gcreate(file_id, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5LTset_attribute_string(file_id, group_name, "xH", "None");
      H5LTset_attribute_string(file_id, group_name, "Gamma12", "v*(h**2)");
      H5LTset_attribute_string(file_id, group_name, "J_21_at_ionization", "v*(h**2)");
      H5LTset_attribute_string(file_id, group_name, "z_at_ionization", "None");
      H5LTset_attribute_string(file_id, group_name, "r_bubble", "v/h");
      H5LTset_attribute_string(file_id, group_name, "Mvir_crit", "v/h");
      H5LTset_attribute_string(file_id, group_name, "stars", "v/h");
      H5LTset_attribute_string(file_id, group_name, "weighted_sfr", "None");
      H5LTset_attribute_string(file_id, group_name, "effective_bhar", "None");
      H5LTset_attribute_string(file_id, group_name, "deltax", "None");
      H5LTset_attribute_string(file_id, group_name, "residual_xH", "None");
      H5LTset_attribute_string(file_id, group_name, "clumping_factor", "None");
      H5LTset_attribute_string(file_id, group_name, "temp_kinetic_all_gas", "None");
      H5LTset_attribute_string(file_id, group_name, "N_rec", "None");
      H5LTset_attribute_string(file_id, group_name, "t_resp", "None");

#if USE_MINI_HALOS
      if (run_globals.params.Flag_IncludeLymanWerner) {
        H5LTset_attribute_string(file_id, group_name, "starsIII", "v/h");
        H5LTset_attribute_string(file_id, group_name, "weighted_sfrIII", "None");
        H5LTset_attribute_string(file_id, group_name, "JLW_box", "v");
        H5LTset_attribute_string(file_id, group_name, "JLW_box_II", "v");
        H5LTset_attribute_string(file_id, group_name, "Mvir_crit_MC", "v/h");
      }
#endif

      if (run_globals.params.Flag_ConstructLightcone) {
        H5LTset_attribute_string(file_id, group_name, "LightconeBox", "None");
      }

      if (run_globals.params.Flag_IncludeSpinTemp) {
        H5LTset_attribute_string(file_id, group_name, "Ts_box", "None");
        H5LTset_attribute_string(file_id, group_name, "Tk_box", "None");
        H5LTset_attribute_string(file_id, group_name, "Ts_boxII", "None");
        H5LTset_attribute_string(file_id, group_name, "Tk_boxII", "None");
        H5LTset_attribute_string(file_id, group_name, "x_e_box", "None");
      }

      if (run_globals.params.Flag_Compute21cmBrightTemp) {
        H5LTset_attribute_string(file_id, group_name, "delta_T", "None");
        H5LTset_attribute_string(file_id, group_name, "delta_TII", "None");
      }

      if (run_globals.params.Flag_ComputePS) {
        H5LTset_attribute_string(file_id, group_name, "PS_data", "None");
        H5LTset_attribute_string(file_id, group_name, "PS_error", "None");
        H5LTset_attribute_string(file_id, group_name, "PSII_data", "None");
        H5LTset_attribute_string(file_id, group_name, "PSII_error", "None");
      }

      H5Gclose(group_id);
    }
  }

#ifdef MERAXES_GITREF_STR
  // Save the git ref and diff if requested
  H5LTmake_dataset_string(file_id, "gitdiff", MERAXES_GITDIFF_STR);
  H5LTset_attribute_string(file_id, "gitdiff", "gitref", MERAXES_GITREF_STR);
#endif

  // save the number of cores used in this run
  H5LTset_attribute_int(file_id, "/", "NCores", &(run_globals.mpi_size), 1);

  if (run_globals.params.Flag_OutputXrayLF) {
    double s_T_vals[5];
    get_nh_transmission(s_T_vals);
    hsize_t nhtrans_dim = 5;
    H5LTmake_dataset_double(file_id, "NHTrans", 1, &nhtrans_dim, s_T_vals);

    int n_lx_bins_nhfrac = (int)((run_globals.params.XrayLF_MaxLogL - run_globals.params.XrayLF_MinLogL) *
                                 run_globals.params.XrayLF_BinsPerDex);
    if (n_lx_bins_nhfrac < 1)
      n_lx_bins_nhfrac = 1;
    double lx_bin_width_nhfrac =
      (run_globals.params.XrayLF_MaxLogL - run_globals.params.XrayLF_MinLogL) / n_lx_bins_nhfrac;

    double f_det[5];
    double lx_log_center, lx_lin_1e10Lsun;
    double* nhfrac_table = malloc((size_t)n_lx_bins_nhfrac * 5 * sizeof(double));
    for (int ilx = 0; ilx < n_lx_bins_nhfrac; ilx++) {
      lx_log_center = run_globals.params.XrayLF_MinLogL + (ilx + 0.5) * lx_bin_width_nhfrac;
      lx_lin_1e10Lsun = pow(10.0, lx_log_center - 10.0 - LOG_10_SOLAR_LUM);
      get_nh_fracs(lx_lin_1e10Lsun, 2.0, f_det); // this model assume the z-dependency plateaus at z>=2.0
      for (int ib = 0; ib < 5; ib++)
        nhfrac_table[ilx * 5 + ib] = f_det[ib];
    }
    hsize_t nhfrac_dims[2] = { (hsize_t)n_lx_bins_nhfrac, 5 };
    H5LTmake_dataset_double(file_id, "NHfrac", 2, nhfrac_dims, nhfrac_table);
    free(nhfrac_table);

    hsize_t n_xray_snaps = (hsize_t)run_globals.params.SnaplistLength;
    H5LTmake_dataset_double(file_id, "XrayEmissivity_hard", 1, &n_xray_snaps, stored_XrayEmissivity_hard);
    H5LTmake_dataset_double(file_id, "XrayEmissivity_soft", 1, &n_xray_snaps, stored_XrayEmissivity_soft);
    H5LTmake_dataset_double(file_id, "XrayEmissivity_HMXB", 1, &n_xray_snaps, stored_XrayEmissivity_HMXB);
  }

  char target_group[50];
  char source_ds[50];
  char source_group[50];
  char source_file[STRLEN];
  char relative_source_file[50];
  hid_t snap_group_id;
  hid_t source_file_id;
  hid_t source_group_id;
  hsize_t core_n_gals;
  double temp;

  // Now create soft links to all of the files and datasets that make up this run
  for (int i_out = 0, snap_n_gals = 0; i_out < run_globals.NOutputSnaps; i_out++, snap_n_gals = 0) {
    sprintf(target_group, "Snap%03d", run_globals.ListOutputSnaps[i_out]);
    snap_group_id = H5Gcreate(file_id, target_group, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    for (int i_core = 0; i_core < run_globals.mpi_size; i_core++) {
      sprintf(target_group, "Core%d", i_core);
      group_id = H5Gcreate(snap_group_id, target_group, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

      sprintf(source_file, "%s/%s_%d.hdf5", run_globals.params.OutputDir, run_globals.params.FileNameGalaxies, i_core);
      sprintf(relative_source_file, "%s_%d.hdf5", run_globals.params.FileNameGalaxies, i_core);

      source_file_id = H5Fopen(source_file, H5F_ACC_RDONLY, H5P_DEFAULT);

      // if they exist, create links to galaxies and walk indices
      sprintf(source_group, "Snap%03d", run_globals.ListOutputSnaps[i_out]);
      source_group_id = H5Gopen(source_file_id, source_group, H5P_DEFAULT);
      if (H5LTfind_dataset(source_group_id, "Galaxies")) {
        sprintf(source_ds, "Snap%03d/Galaxies", run_globals.ListOutputSnaps[i_out]);
        H5Lcreate_external(relative_source_file, source_ds, group_id, "Galaxies", H5P_DEFAULT, H5P_DEFAULT);
        H5TBget_table_info(source_file_id, source_ds, NULL, &core_n_gals);
        snap_n_gals += (int)core_n_gals;
      }

      if (H5LTfind_dataset(source_group_id, "FirstProgenitorIndices")) {
        sprintf(source_ds, "Snap%03d/FirstProgenitorIndices", run_globals.ListOutputSnaps[i_out]);
        H5Lcreate_external(
          relative_source_file, source_ds, group_id, "FirstProgenitorIndices", H5P_DEFAULT, H5P_DEFAULT);
      }
      if (H5LTfind_dataset(source_group_id, "NextProgenitorIndices")) {
        sprintf(source_ds, "Snap%03d/NextProgenitorIndices", run_globals.ListOutputSnaps[i_out]);
        H5Lcreate_external(
          relative_source_file, source_ds, group_id, "NextProgenitorIndices", H5P_DEFAULT, H5P_DEFAULT);
      }
      if (H5LTfind_dataset(source_group_id, "DescendantIndices")) {
        sprintf(source_ds, "Snap%03d/DescendantIndices", run_globals.ListOutputSnaps[i_out]);
        H5Lcreate_external(relative_source_file, source_ds, group_id, "DescendantIndices", H5P_DEFAULT, H5P_DEFAULT);
      }

      H5Gclose(source_group_id);
      H5Gclose(group_id);
      H5Fclose(source_file_id);

      if ((i_core == 0) && (run_globals.params.Flag_PatchyReion)) {
        // create links to the 21cmFAST grids that exist
        gen_grids_fname(run_globals.ListOutputSnaps[i_out], relative_source_file, true);
        gen_grids_fname(run_globals.ListOutputSnaps[i_out], source_file, false);
        if (access(source_file, F_OK) != -1) {
          source_file_id = H5Fopen(source_file, H5F_ACC_RDONLY, H5P_DEFAULT);
          H5Lcreate_external(relative_source_file, "/", snap_group_id, "Grids", H5P_DEFAULT, H5P_DEFAULT);
          H5Fclose(source_file_id);
        }
      }

#if USE_MINI_HALOS
      if ((i_core == 0) && (run_globals.params.Flag_IncludeMetalEvo)) {
        // create links to the 21cmFAST grids that exist
        gen_metal_grids_fname(run_globals.ListOutputSnaps[i_out], relative_source_file, true);
        gen_metal_grids_fname(run_globals.ListOutputSnaps[i_out], source_file, false);
        if (access(source_file, F_OK) != -1) {
          source_file_id = H5Fopen(source_file, H5F_ACC_RDONLY, H5P_DEFAULT);
          H5Lcreate_external(relative_source_file, "/", snap_group_id, "MetalGrids", H5P_DEFAULT, H5P_DEFAULT);
          H5Fclose(source_file_id);
        }
      }
#endif
    }

    // Create links to distribution functions from rank 0's core file
    // Distribution functions are only computed by rank 0 after MPI reduction
    sprintf(source_file, "%s/%s_0.hdf5", run_globals.params.OutputDir, run_globals.params.FileNameGalaxies);
    sprintf(relative_source_file, "%s_0.hdf5", run_globals.params.FileNameGalaxies);
    sprintf(source_group, "Snap%03d", run_globals.ListOutputSnaps[i_out]);

    source_file_id = H5Fopen(source_file, H5F_ACC_RDONLY, H5P_DEFAULT);
    source_group_id = H5Gopen(source_file_id, source_group, H5P_DEFAULT);

    if (run_globals.params.Flag_OutputHMF && H5LTfind_dataset(source_group_id, "HMF")) {
      sprintf(source_ds, "Snap%03d/HMF", run_globals.ListOutputSnaps[i_out]);
      H5Lcreate_external(relative_source_file, source_ds, snap_group_id, "HMF", H5P_DEFAULT, H5P_DEFAULT);
    }

    if (run_globals.params.Flag_OutputSMF && H5LTfind_dataset(source_group_id, "SMF")) {
      sprintf(source_ds, "Snap%03d/SMF", run_globals.ListOutputSnaps[i_out]);
      H5Lcreate_external(relative_source_file, source_ds, snap_group_id, "SMF", H5P_DEFAULT, H5P_DEFAULT);
    }

#ifdef CALC_MAGS
    if (run_globals.params.Flag_OutputUVLF && H5LTfind_dataset(source_group_id, "UVLF")) {
      sprintf(source_ds, "Snap%03d/UVLF", run_globals.ListOutputSnaps[i_out]);
      H5Lcreate_external(relative_source_file, source_ds, snap_group_id, "UVLF", H5P_DEFAULT, H5P_DEFAULT);
    }

    if (run_globals.params.Flag_OutputDustyLF && H5LTfind_dataset(source_group_id, "DustyLF")) {
      sprintf(source_ds, "Snap%03d/DustyLF", run_globals.ListOutputSnaps[i_out]);
      H5Lcreate_external(relative_source_file, source_ds, snap_group_id, "DustyLF", H5P_DEFAULT, H5P_DEFAULT);
    }

    if (run_globals.params.Flag_OutputOIIILF && H5LTfind_dataset(source_group_id, "OIIIDustyLF")) {
      sprintf(source_ds, "Snap%03d/OIIIDustyLF", run_globals.ListOutputSnaps[i_out]);
      H5Lcreate_external(relative_source_file, source_ds, snap_group_id, "OIIIDustyLF", H5P_DEFAULT, H5P_DEFAULT);
    }
#endif

    // QuasarLF external link (not dependent on CALC_MAGS)
    if (run_globals.params.Flag_OutputQuasarLF && H5LTfind_dataset(source_group_id, "QuasarLF")) {
      sprintf(source_ds, "Snap%03d/QuasarLF", run_globals.ListOutputSnaps[i_out]);
      H5Lcreate_external(relative_source_file, source_ds, snap_group_id, "QuasarLF", H5P_DEFAULT, H5P_DEFAULT);
    }

    if (run_globals.params.Flag_OutputOIIILF && H5LTfind_dataset(source_group_id, "OIIILF")) {
      sprintf(source_ds, "Snap%03d/OIIILF", run_globals.ListOutputSnaps[i_out]);
      H5Lcreate_external(relative_source_file, source_ds, snap_group_id, "OIIILF", H5P_DEFAULT, H5P_DEFAULT);
    }

    if (run_globals.params.Flag_OutputXrayLF && H5LTfind_dataset(source_group_id, "XrayLF")) {
      sprintf(source_ds, "Snap%03d/XrayLF", run_globals.ListOutputSnaps[i_out]);
      H5Lcreate_external(relative_source_file, source_ds, snap_group_id, "XrayLF", H5P_DEFAULT, H5P_DEFAULT);
    }
    if (run_globals.params.Flag_OutputXrayLF && H5LTfind_dataset(source_group_id, "XrayLF_obs")) {
      sprintf(source_ds, "Snap%03d/XrayLF_obs", run_globals.ListOutputSnaps[i_out]);
      H5Lcreate_external(relative_source_file, source_ds, snap_group_id, "XrayLF_obs", H5P_DEFAULT, H5P_DEFAULT);
    }
    H5Gclose(source_group_id);
    H5Fclose(source_file_id);

    // Save a few useful attributes
    sprintf(target_group, "Snap%03d", run_globals.ListOutputSnaps[i_out]);

    // save the total number of galaxies at this snapshot
    H5LTset_attribute_int(file_id, target_group, "NGalaxies", &snap_n_gals, 1);

    H5LTset_attribute_double(
      file_id, target_group, "Redshift", &(run_globals.ZZ[run_globals.ListOutputSnaps[i_out]]), 1);

    temp = run_globals.LTTime[run_globals.ListOutputSnaps[i_out]] * run_globals.units.UnitLength_in_cm /
           run_globals.units.UnitVelocity_in_cm_per_s / SEC_PER_MEGAYEAR;
    H5LTset_attribute_double(file_id, target_group, "LTTime", &temp, 1);

    H5Gclose(snap_group_id);
  }

  // Close the HDF5 file.
  H5Fclose(file_id);

  mlog(" ...done", MLOG_CLOSE | MLOG_TIMERSTOP);
}

static inline void save_walk_indices(hid_t file_id,
                                     int i_out,
                                     int prev_i_out,
                                     int* descendant_index,
                                     int* first_progenitor_index,
                                     int* next_progenitor_index,
                                     int old_count,
                                     int n_write)
{
  hid_t dset_id;
  char target[50];
  int chunk_size = 1000;

  if (old_count > 0) {
    hsize_t dim[1] = { (hsize_t)old_count };
    hsize_t chunks[1] = { chunk_size > old_count ? (hsize_t)old_count : (hsize_t)chunk_size };

    hid_t plist_id = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(plist_id, 1, chunks);
    H5Pset_deflate(plist_id, 6);

    hid_t dspace_id = H5Screate_simple(1, dim, NULL);

    sprintf(target, "Snap%03d/DescendantIndices", (run_globals.ListOutputSnaps)[prev_i_out]);
    dset_id = H5Dcreate(file_id, target, H5T_NATIVE_INT, dspace_id, H5P_DEFAULT, plist_id, H5P_DEFAULT);
    H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, descendant_index);
    H5Dclose(dset_id);

    sprintf(target, "Snap%03d/NextProgenitorIndices", (run_globals.ListOutputSnaps)[prev_i_out]);
    dset_id = H5Dcreate(file_id, target, H5T_NATIVE_INT, dspace_id, H5P_DEFAULT, plist_id, H5P_DEFAULT);
    H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, next_progenitor_index);
    H5Dclose(dset_id);

    H5Sclose(dspace_id);
    H5Pclose(plist_id);
  } else {
    // Here we create empty datasets.  This is purely to maintain backward
    // compatibility for other codes which read the output (e.g. Yisheng
    // Qiu's magcalc code).
    hsize_t dim[1] = { 0 };

    sprintf(target, "Snap%03d/DescendantIndices", (run_globals.ListOutputSnaps)[prev_i_out]);
    H5LTmake_dataset(file_id, target, 1, dim, H5T_NATIVE_INT, descendant_index);

    sprintf(target, "Snap%03d/NextProgenitorIndices", (run_globals.ListOutputSnaps)[prev_i_out]);
    H5LTmake_dataset(file_id, target, 1, dim, H5T_NATIVE_INT, next_progenitor_index);
  }

  if (n_write > 0) {
    hsize_t dim[1] = { (hsize_t)n_write };
    hsize_t chunks[1] = { chunk_size > n_write ? n_write : chunk_size };

    hid_t plist_id = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(plist_id, 1, chunks);
    H5Pset_deflate(plist_id, 6);

    H5Pset_chunk(plist_id, 1, chunks);
    hid_t dspace_id = H5Screate_simple(1, dim, NULL);

    sprintf(target, "Snap%03d/FirstProgenitorIndices", (run_globals.ListOutputSnaps)[i_out]);
    dset_id = H5Dcreate(file_id, target, H5T_NATIVE_INT, dspace_id, H5P_DEFAULT, plist_id, H5P_DEFAULT);
    H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, first_progenitor_index);
    H5Dclose(dset_id);

    H5Sclose(dspace_id);
    H5Pclose(plist_id);
  } else {
    // Here we create empty datasets.  This is purely to maintain backward
    // compatibility for other codes which read the output (e.g. Yisheng
    // Qiu's magcalc code).
    hsize_t dim[1] = { 0 };

    sprintf(target, "Snap%03d/FirstProgenitorIndices", (run_globals.ListOutputSnaps)[i_out]);
    H5LTmake_dataset(file_id, target, 1, dim, H5T_NATIVE_INT, first_progenitor_index);
  }
}

static inline bool pass_write_check(galaxy_t* gal, bool flag_merger)
{
  if (
    // Test for non-merger galaxy to be written in the current snap
    (!flag_merger && (gal->Type < 3)) // && ((gal->output_index > -1) || (gal->StellarMass >= 1e-10)))
    // and this is the test for a merger to be accounted for in descendant / progenitor arrays
    || (flag_merger && (gal->Type == 3) && (gal->output_index > -1)))
    return true;
  else
    return false;
}

void write_snapshot(int n_write, int i_out, int* last_n_write)
{
  /*
   * Write a batch of galaxies to the output HDF5 table.
   */

  hid_t file_id;
  hid_t group_id;
  hsize_t chunk_size = 5000;
  galaxy_output_t* output_buffer = NULL;
  int* fill_data = NULL;
  char target_group[20];
  galaxy_t* gal = NULL;
  hdf5_output_t h5props = run_globals.hdf5props;
  int gal_count = 0;
  int old_count = 0;
  int* first_progenitor_index = NULL;
  int* next_progenitor_index = NULL;
  int calc_descendants_i_out = -1;
  int prev_snapshot = -1;
  int write_count = 0;
  int index;

  // Distribution function structures
  distribution_function_t hmf, smf;
  distribution_function_t quasarlf;
  distribution_function_t oiiilf;
  distribution_function_t xraylf;
  distribution_function_t xraylf_obs;
#ifdef CALC_MAGS
  distribution_function_t uvlf, dustylf, oiiidustylf;
#endif

  // Volume will be set during initialization
  double df_volume = (run_globals.params.BoxSize / run_globals.params.Hubble_h);
  df_volume = df_volume * df_volume * df_volume;

  // n_write is this rank's local count; report the global total across all
  // ranks in the log instead, since that's what's actually meaningful here.
  int n_write_global = 0;
  MPI_Reduce(&n_write, &n_write_global, 1, MPI_INT, MPI_SUM, 0, run_globals.mpi_comm);

  mlog("Writing output file (n_write = %d)...", MLOG_OPEN | MLOG_TIMERSTART, n_write_global);

  // We aren't going to write any galaxies that have zero stellar mass, so
  // modify n_write appropriately...
  gal = run_globals.FirstGal;
  while (gal != NULL) {
    if (pass_write_check(gal, false))
      write_count++;
    gal = gal->Next;
  }

  if (n_write != write_count) {
    mlog("Excluding %d ~zero mass galaxies...", MLOG_MESG, n_write - write_count);
    mlog("New write count = %d", MLOG_MESG, write_count);
    n_write = write_count;
  }

  // Use the persistently-open per-rank file (avoids HDF5 1.10.x open-close
  // fragmentation that causes "duplicate entry in cache" after ~17 snapshots).
  file_id = run_globals.output_file_id;

  // Create the relevant group.
  sprintf(target_group, "Snap%03d", (run_globals.ListOutputSnaps)[i_out]);
  group_id = H5Gcreate(file_id, target_group, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // reset the chunk size if required
  if ((int)chunk_size < n_write)
    chunk_size = (hsize_t)n_write;

  // Make the table (skip for FlagInteractive==2 which only outputs distribution functions)
  if (run_globals.params.FlagInteractive != 2) {
    H5TBmake_table("Galaxies",
                   group_id,
                   "Galaxies",
                   (hsize_t)h5props.n_props,
                   (hsize_t)n_write,
                   h5props.dst_size,
                   h5props.field_names,
                   h5props.dst_offsets,
                   h5props.field_types,
                   chunk_size,
                   fill_data,
                   1,
                   NULL);
  }

  // Initialize distribution functions
  if (run_globals.params.Flag_OutputHMF) {
    if (run_globals.params.HMF_MaxMass <= run_globals.params.HMF_MinMass) {
      mlog_error("HMF_MaxMass must be greater than HMF_MinMass.");
      ABORT(EXIT_FAILURE);
    }
    if (run_globals.params.HMF_BinsPerDex <= 0) {
      mlog_error("HMF_BinsPerDex must be > 0.");
      ABORT(EXIT_FAILURE);
    }
    df_init(&hmf,
            run_globals.params.HMF_MinMass,
            run_globals.params.HMF_MaxMass,
            run_globals.params.HMF_BinsPerDex,
            "Halo Mass Function");
    hmf.volume = df_volume;
  }

  if (run_globals.params.Flag_OutputSMF) {
    if (run_globals.params.SMF_MaxMass <= run_globals.params.SMF_MinMass) {
      mlog_error("SMF_MaxMass must be greater than SMF_MinMass.");
      ABORT(EXIT_FAILURE);
    }
    if (run_globals.params.SMF_BinsPerDex <= 0) {
      mlog_error("SMF_BinsPerDex must be > 0.");
      ABORT(EXIT_FAILURE);
    }
    df_init(&smf,
            run_globals.params.SMF_MinMass,
            run_globals.params.SMF_MaxMass,
            run_globals.params.SMF_BinsPerDex,
            "Stellar Mass Function");
    smf.volume = df_volume;
  }

#ifdef CALC_MAGS
  // Check if this snapshot is a target snapshot for magnitude calculations
  int is_target_snap = 0;
  for (int iS = 0; iS < MAGS_N_SNAPS; ++iS) {
    if (run_globals.ListOutputSnaps[i_out] == run_globals.mag_params.targetSnap[iS]) {
      is_target_snap = 1;
      break;
    }
  }

  const int do_oiii_dusty_lf =
    is_target_snap && run_globals.params.Flag_OutputOIIILF && (run_globals.loiii_rest_band_mag_index >= 0);

  if (is_target_snap && run_globals.params.Flag_OutputUVLF) {
    if (run_globals.params.UVLF_MaxMag <= run_globals.params.UVLF_MinMag) {
      mlog_error("UVLF_MaxMag must be greater than UVLF_MinMag.");
      ABORT(EXIT_FAILURE);
    }
    if (run_globals.params.UVLF_BinsPerMag <= 0) {
      mlog_error("UVLF_BinsPerMag must be > 0.");
      ABORT(EXIT_FAILURE);
    }
    df_init(&uvlf,
            run_globals.params.UVLF_MinMag,
            run_globals.params.UVLF_MaxMag,
            run_globals.params.UVLF_BinsPerMag,
            "UV Luminosity Function");
    uvlf.volume = df_volume;
  }

  if (is_target_snap && run_globals.params.Flag_OutputDustyLF) {
    if (run_globals.params.UVLF_MaxMag <= run_globals.params.UVLF_MinMag) {
      mlog_error("UVLF_MaxMag must be greater than UVLF_MinMag for DustyLF.");
      ABORT(EXIT_FAILURE);
    }
    if (run_globals.params.UVLF_BinsPerMag <= 0) {
      mlog_error("UVLF_BinsPerMag must be > 0 for DustyLF.");
      ABORT(EXIT_FAILURE);
    }
    df_init(&dustylf,
            run_globals.params.UVLF_MinMag,
            run_globals.params.UVLF_MaxMag,
            run_globals.params.UVLF_BinsPerMag,
            "Dusty UV Luminosity Function");
    dustylf.volume = df_volume;
  }
#endif

  // QuasarLF uses same mag bins as UVLF but weighted by duty cycle
  if (run_globals.params.Flag_OutputQuasarLF) {
    if (run_globals.params.UVLF_MaxMag <= run_globals.params.UVLF_MinMag) {
      mlog_error("UVLF_MaxMag must be greater than UVLF_MinMag for QuasarLF.");
      ABORT(EXIT_FAILURE);
    }
    if (run_globals.params.UVLF_BinsPerMag <= 0) {
      mlog_error("UVLF_BinsPerMag must be > 0 for QuasarLF.");
      ABORT(EXIT_FAILURE);
    }
    df_init(&quasarlf,
            run_globals.params.UVLF_MinMag,
            run_globals.params.UVLF_MaxMag,
            run_globals.params.UVLF_BinsPerMag,
            "Quasar UV Luminosity Function");
    quasarlf.volume = df_volume;
  }

  if (run_globals.params.Flag_OutputOIIILF) {
    if (run_globals.params.OIIILF_MaxLogL <= run_globals.params.OIIILF_MinLogL) {
      mlog_error("OIIILF_MaxLogL must be greater than OIIILF_MinLogL.");
      ABORT(EXIT_FAILURE);
    }
    if (run_globals.params.OIIILF_BinsPerDex <= 0) {
      mlog_error("OIIILF_BinsPerDex must be > 0.");
      ABORT(EXIT_FAILURE);
    }
    df_init(&oiiilf,
            run_globals.params.OIIILF_MinLogL,
            run_globals.params.OIIILF_MaxLogL,
            run_globals.params.OIIILF_BinsPerDex,
            "OIII Luminosity Function");
    oiiilf.volume = df_volume;

#ifdef CALC_MAGS
    if (do_oiii_dusty_lf) {
      df_init(&oiiidustylf,
              run_globals.params.OIIILF_MinLogL,
              run_globals.params.OIIILF_MaxLogL,
              run_globals.params.OIIILF_BinsPerDex,
              "Dusty OIII Luminosity Function");
      oiiidustylf.volume = df_volume;
    }
#endif
  }

  if (run_globals.params.Flag_OutputXrayLF) {
    if (run_globals.params.XrayLF_MaxLogL <= run_globals.params.XrayLF_MinLogL) {
      mlog_error("XrayLF_MaxLogL must be greater than XrayLF_MinLogL.");
      ABORT(EXIT_FAILURE);
    }
    if (run_globals.params.XrayLF_BinsPerDex <= 0) {
      mlog_error("XrayLF_BinsPerDex must be > 0.");
      ABORT(EXIT_FAILURE);
    }
    df_init(&xraylf,
            run_globals.params.XrayLF_MinLogL,
            run_globals.params.XrayLF_MaxLogL,
            run_globals.params.XrayLF_BinsPerDex,
            "X-ray Luminosity Function (2-10 keV, intrinsic)");
    xraylf.volume = df_volume;
    df_init(&xraylf_obs,
            run_globals.params.XrayLF_MinLogL,
            run_globals.params.XrayLF_MaxLogL,
            run_globals.params.XrayLF_BinsPerDex,
            "X-ray Luminosity Function (2-10 keV, observed after obscuration)");
    xraylf_obs.volume = df_volume;
  }

  // If the immediately preceding snapshot was also written, then save the
  // descendent indices (skip for FlagInteractive==2)
  prev_snapshot = run_globals.ListOutputSnaps[i_out] - 1;
  if (i_out > 0 && run_globals.params.FlagInteractive != 2) {
    for (int ii = 0; ii < run_globals.NOutputSnaps; ii++)
      if (run_globals.ListOutputSnaps[ii] == prev_snapshot) {
        calc_descendants_i_out = ii;
        break;
      }
  }

  // Assign the write order indices to each galaxy and store the old indices if required
  gal_count = 0;
  old_count = 0;
  if (calc_descendants_i_out > -1) {
    // malloc the arrays
    int* descendant_index = malloc(sizeof(int) * (*last_n_write));
    next_progenitor_index = malloc(sizeof(int) * (*last_n_write));
    first_progenitor_index = malloc(sizeof(int) * n_write);

    // initialise all entries to -1
    for (int ii = 0; ii < *last_n_write; ii++) {
      descendant_index[ii] = -1;
      next_progenitor_index[ii] = -1;
    }
    for (int ii = 0; ii < n_write; ii++)
      first_progenitor_index[ii] = -1;

    // loop through the current galaxies and save their first progenitor
    // indices as their previous output_index, and the descendent indices of
    // the last snapshot to what will be the output index when the current
    // galaxy is written.
    gal = run_globals.FirstGal;
    while (gal != NULL) {
      if (pass_write_check(gal, false)) {
        if (gal->output_index > -1) {
          first_progenitor_index[gal_count] = gal->output_index;

          assert(gal->output_index < *last_n_write);

          descendant_index[gal->output_index] = gal_count;
          old_count++;
        }
        gal->output_index = gal_count++;
      }
      gal = gal->Next;
    }

    // Here we want to walk the progenitor indices to tag on galaxies which
    // have merged in this timestep and also set their descendant_index.
    gal = run_globals.FirstGal;
    while (gal != NULL) {
      if (pass_write_check(gal, true)) {
        assert((gal->output_index < *last_n_write) && (gal->output_index >= 0));
        descendant_index[gal->output_index] = gal->MergerTarget->output_index;
        old_count++;

        assert(gal->MergerTarget->output_index < n_write);
        if (gal->MergerTarget->output_index >= 0) {
          index = first_progenitor_index[gal->MergerTarget->output_index];
          if (index > -1) {
            while (next_progenitor_index[index] > -1)
              index = next_progenitor_index[index];
            next_progenitor_index[index] = gal->output_index;
          }
        }
      }
      gal = gal->Next;
    }

    save_walk_indices(file_id,
                      i_out,
                      calc_descendants_i_out,
                      descendant_index,
                      first_progenitor_index,
                      next_progenitor_index,
                      *last_n_write,
                      n_write);

    // Free the allocated arrays
    free(first_progenitor_index);
    free(next_progenitor_index);
    free(descendant_index);

  } else {

    gal = run_globals.FirstGal;
    while (gal != NULL) {
      if (pass_write_check(gal, false))
        gal->output_index = gal_count++;
      gal = gal->Next;
    }
  }

  // Skip galaxy count validation for FlagInteractive==2
  if (run_globals.params.FlagInteractive != 2) {
    if (n_write != gal_count) {
      fprintf(stderr, "We don't have the expected number of galaxies in save...");
      fprintf(stderr, "gal_count=%d, n_write=%d", gal_count, n_write);
      ABORT(EXIT_FAILURE);
    }
  }

  // Write the galaxies.
  // In order to speed things up, we will chunk our write.
  // This can cause significant memory overhead if `chunk_size` is large.
  gal_count = 0;
  gal = run_globals.FirstGal;
  output_buffer = calloc((int)chunk_size, sizeof(galaxy_output_t));
  // Accumulate into distribution functions from output buffer
  double val, weight;
  int bin_idx;
  double lx_int_lin, lx_obs_lin;

  int buffer_count = 0;
  while (gal != NULL) {
    // Don't output galaxies which merged at this timestep
    if (pass_write_check(gal, false)) {
      prepare_galaxy_for_output(*gal, &(output_buffer[buffer_count]), i_out);

      if (run_globals.params.Flag_OutputHMF && !output_buffer[buffer_count].GhostFlag) {
        // Extract HMF value (log10 halo mass in solar masses/h)
        val = log10(output_buffer[buffer_count].Mvir * 1e10 / run_globals.params.Hubble_h);
        bin_idx = df_bin_index(&hmf, val);
        if (bin_idx >= 0) {
          hmf.bin_counts[bin_idx] += 1.0;
        }
      }

      if (run_globals.params.Flag_OutputSMF && !output_buffer[buffer_count].GhostFlag) {
        // Extract SMF value (log10 stellar mass in solar masses)
        val = output_buffer[buffer_count].StellarMass * 1e10 / run_globals.params.Hubble_h;
        if (val > 0.0) {
          val = log10(val);
          bin_idx = df_bin_index(&smf, val);
          if (bin_idx >= 0) {
            smf.bin_counts[bin_idx] += 1.0;
          }
        }
      }

#ifdef CALC_MAGS
      if (is_target_snap && run_globals.params.Flag_OutputUVLF && !output_buffer[buffer_count].GhostFlag) {
        // Extract UVLF value (UV magnitude)
        if (isfinite(output_buffer[buffer_count].Mags[0])) {
          val = output_buffer[buffer_count].Mags[0];
          bin_idx = df_bin_index(&uvlf, val);
          if (bin_idx >= 0) {
            uvlf.bin_counts[bin_idx] += 1.0;
          }
        }
      }

      if (is_target_snap && run_globals.params.Flag_OutputDustyLF && !output_buffer[buffer_count].GhostFlag) {
        // Extract DustyLF value (dusty UV magnitude)
        if (isfinite(output_buffer[buffer_count].DustyMags[0])) {
          val = output_buffer[buffer_count].DustyMags[0];
          bin_idx = df_bin_index(&dustylf, val);
          if (bin_idx >= 0) {
            dustylf.bin_counts[bin_idx] += 1.0;
          }
        }
      }
#endif

      // QuasarLF: bin QuasarMag weighted by DutyCycleAGN * quasar_fobs (opening angle)
      if (run_globals.params.Flag_OutputQuasarLF && !output_buffer[buffer_count].GhostFlag) {
        val = output_buffer[buffer_count].QuasarMag;
        weight = output_buffer[buffer_count].DutyCycleAGN * run_globals.params.physics.quasar_fobs;
        // Only include quasars that are "on" (QuasarMag < 999) and have positive duty cycle
        if (val < 900.0 && weight > 0.0 && isfinite(val)) {
          bin_idx = df_bin_index(&quasarlf, val);
          if (bin_idx >= 0) {
            quasarlf.bin_counts[bin_idx] += weight;                    // Weight by duty cycle
            quasarlf.bin_variance[bin_idx] += weight * (1.0 - weight); // Bernoulli variance
          }
        }
      }

      if (run_globals.params.Flag_OutputOIIILF && !output_buffer[buffer_count].GhostFlag) {
        val = output_buffer[buffer_count].LOIII;
        if (val > 0.0 && isfinite(val)) {
          val = log10(val) + 40;
          bin_idx = df_bin_index(&oiiilf, val);
          if (bin_idx >= 0) {
            oiiilf.bin_counts[bin_idx] += 1.0;
          }
        }
      }

#ifdef CALC_MAGS
      if (do_oiii_dusty_lf && !output_buffer[buffer_count].GhostFlag) {
        val = output_buffer[buffer_count].LOIII_dusty;
        if (val > 0.0 && isfinite(val)) {
          val = log10(val) + 40;
          bin_idx = df_bin_index(&oiiidustylf, val);
          if (bin_idx >= 0) {
            oiiidustylf.bin_counts[bin_idx] += 1.0;
          }
        }
      }
#endif

      if (run_globals.params.Flag_OutputXrayLF) {
        lx_int_lin = (double)output_buffer[buffer_count].QuasarLX;
        lx_obs_lin = (double)output_buffer[buffer_count].BHXrayEmissivity;
        weight = output_buffer[buffer_count].DutyCycleAGN;
        if (weight > 0.0) {
          if (lx_int_lin > 0.0) {
            val = log10(lx_int_lin) + LOG_10_SOLAR_LUM + 10;
            bin_idx = df_bin_index(&xraylf, val);
            if (bin_idx >= 0) {
              xraylf.bin_counts[bin_idx] += weight;
              xraylf.bin_variance[bin_idx] += weight * (1.0 - weight); /* Bernoulli */
            }
          }

          if (lx_obs_lin > 0.0) {
            val = log10(lx_obs_lin) + LOG_10_SOLAR_LUM + 10;
            bin_idx = df_bin_index(&xraylf_obs, val);
            if (bin_idx >= 0) {
              xraylf_obs.bin_counts[bin_idx] += weight;
              xraylf_obs.bin_variance[bin_idx] += weight * (1.0 - weight);
            }
          }
        }
      }

      buffer_count++;
      ;
    }
    if (buffer_count == (int)chunk_size) {
      // Write galaxies to HDF5 (skip for FlagInteractive==2)
      if (run_globals.params.FlagInteractive != 2) {
        H5TBwrite_records(group_id,
                          "Galaxies",
                          (hsize_t)gal_count,
                          (hsize_t)buffer_count,
                          h5props.dst_size,
                          h5props.dst_offsets,
                          h5props.dst_field_sizes,
                          output_buffer);
      }
      gal_count += buffer_count;
      buffer_count = 0;
    }
    gal = gal->Next;
  }

  // Write any remaining galaxies in the buffer (skip for FlagInteractive==2)
  if (buffer_count > 0) {
    if (run_globals.params.FlagInteractive != 2) {
      H5TBwrite_records(group_id,
                        "Galaxies",
                        (hsize_t)gal_count,
                        (hsize_t)buffer_count,
                        h5props.dst_size,
                        h5props.dst_offsets,
                        h5props.dst_field_sizes,
                        output_buffer);
    }
    gal_count += buffer_count;
  }

  // Skip galaxy count validation for FlagInteractive==2
  if (run_globals.params.FlagInteractive != 2) {
    if (n_write != gal_count) {
      mlog("We don't have the expected number of galaxies in save...", MLOG_MESG);
      mlog("gal_count=%d, n_write=%d", MLOG_MESG, gal_count, n_write);
      ABORT(EXIT_FAILURE);
    }
  }

  // Free the output buffer
  free(output_buffer);

  if ((run_globals.params.Flag_PatchyReion) && check_if_reionization_ongoing(run_globals.ListOutputSnaps[i_out]) &&
      (run_globals.params.Flag_OutputGrids))
    save_reion_output_grids(run_globals.ListOutputSnaps[i_out]);

  // MPI reduction and output for all distribution functions
  if (run_globals.params.Flag_OutputHMF) {
    df_mpi_reduce(&hmf, run_globals.mpi_rank, run_globals.mpi_size);
    if (run_globals.mpi_rank == 0) {
      df_write_hdf5(file_id, target_group, &hmf, "HMF", "per Mpc^3 per dex");
    }
    df_free(&hmf);
  }

  if (run_globals.params.Flag_OutputSMF) {
    df_mpi_reduce(&smf, run_globals.mpi_rank, run_globals.mpi_size);
    if (run_globals.mpi_rank == 0) {
      df_write_hdf5(file_id, target_group, &smf, "SMF", "per Mpc^3 per dex");
    }
    df_free(&smf);
  }

#ifdef CALC_MAGS
  if (is_target_snap && run_globals.params.Flag_OutputUVLF) {
    df_mpi_reduce(&uvlf, run_globals.mpi_rank, run_globals.mpi_size);
    if (run_globals.mpi_rank == 0) {
      df_write_hdf5(file_id, target_group, &uvlf, "UVLF", "per Mpc^3 per mag");
    }
    df_free(&uvlf);
  }

  if (is_target_snap && run_globals.params.Flag_OutputDustyLF) {
    df_mpi_reduce(&dustylf, run_globals.mpi_rank, run_globals.mpi_size);
    if (run_globals.mpi_rank == 0) {
      df_write_hdf5(file_id, target_group, &dustylf, "DustyLF", "per Mpc^3 per mag");
    }
    df_free(&dustylf);
  }
#endif

  // QuasarLF - uses weighted MPI reduce and write
  if (run_globals.params.Flag_OutputQuasarLF) {
    df_mpi_reduce(&quasarlf, run_globals.mpi_rank, run_globals.mpi_size);
    if (run_globals.mpi_rank == 0) {
      df_write_hdf5(file_id, target_group, &quasarlf, "QuasarLF", "per Mpc^3 per mag");
    }
    df_free(&quasarlf);
  }

  if (run_globals.params.Flag_OutputOIIILF) {
    df_mpi_reduce(&oiiilf, run_globals.mpi_rank, run_globals.mpi_size);
    if (run_globals.mpi_rank == 0) {
      df_write_hdf5(file_id, target_group, &oiiilf, "OIIILF", "per Mpc^3 per dex");
    }
    df_free(&oiiilf);

#ifdef CALC_MAGS
    if (do_oiii_dusty_lf) {
      df_mpi_reduce(&oiiidustylf, run_globals.mpi_rank, run_globals.mpi_size);
      if (run_globals.mpi_rank == 0) {
        df_write_hdf5(file_id, target_group, &oiiidustylf, "OIIIDustyLF", "per Mpc^3 per dex");
      }
      df_free(&oiiidustylf);
    }
#endif
  }

  if (run_globals.params.Flag_OutputXrayLF) {
    df_mpi_reduce(&xraylf, run_globals.mpi_rank, run_globals.mpi_size);
    df_mpi_reduce(&xraylf_obs, run_globals.mpi_rank, run_globals.mpi_size);
    if (run_globals.mpi_rank == 0) {
      df_write_hdf5(file_id, target_group, &xraylf, "XrayLF", "per Mpc^3 per dex");
      df_write_hdf5(file_id, target_group, &xraylf_obs, "XrayLF_obs", "per Mpc^3 per dex");
    }
    df_free(&xraylf);
    df_free(&xraylf_obs);
  }

  // Close the group.
  H5Gclose(group_id);

  // Flush (but do not close) this snapshot's data to disk now, so it survives a crash later in the run rather than
  // being lost with everything that was never flushed.
  H5Fflush(run_globals.output_file_id, H5F_SCOPE_GLOBAL);

  // Update the value of last_n_write
  *last_n_write = n_write;

  mlog("...done", MLOG_CLOSE | MLOG_TIMERSTOP);
}
