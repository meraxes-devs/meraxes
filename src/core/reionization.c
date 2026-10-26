#include <assert.h>
#include <complex.h>
#include <fenv.h>
#include <fftw3-mpi.h>
#include <gsl/gsl_integration.h>
#include <hdf5_hl.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include "ComputeTs.h"
#include "XRayHeatingFunctions.h"
#include "cpu_grids.h"
#include "cpu_utils.h"
#include "find_HII_bubbles.h"
#include "meraxes.h"
#include "misc_tools.h"
#include "read_grids.h"
#include "reionization.h"
#include "virial_properties.h"
#if USE_STOCHASTICITY
#include "Stochasticity.h"
#endif

static hid_t create_reion_grid(const int snapshot, const bool parallel);

static inline double tau_e_prefactor_at_z(double z)
{
  // c * sigma_T * (1+z)^2 / H(z), with H evaluated continuously in redshift.
  double zplus1 = 1.0 + z;
  double Hz = hubble_at_z(z) * run_globals.params.Hubble_h / run_globals.units.UnitTime_in_s; // [s^-1]
  return SPEED_OF_LIGHT * SIGMA_T_CGS * zplus1 * zplus1 / Hz;
} // cm^3

static inline double tau_e_sim_integrand_at_snapshot(int snapshot, double xHII)
{
  // Follow post-processing convention: n_e ~ xHII * (n_H + n_He)
  return tau_e_prefactor_at_z(run_globals.ZZ[snapshot]) * N_b0 * xHII;
} // unitless

static inline double tau_e_postsim_integrand(double z, void* params)
{
  (void)params;
  // Mirror run.py piecewise helium treatment:
  // z <= 4: fully double-ionized helium (n_H + 2 n_He)
  // z  > 4: singly-ionized helium (n_H + n_He)
  double ne = (z <= 4.0) ? (No + 2.0 * He_No) : N_b0;
  return tau_e_prefactor_at_z(z) * ne;
} // unitless

double integrate_tau_e_postEoR(double zmax)
{
  if (zmax <= 0.0)
    return 0.0;

  gsl_function F;
  F.function = &tau_e_postsim_integrand;
  F.params = NULL;

  gsl_integration_workspace* w = gsl_integration_workspace_alloc(1000);
  double result = 0.0;
  double error = 0.0;

  gsl_integration_qag(&F, 0.0, zmax, 0.0, 1e-7, 1000, GSL_INTEG_GAUSS61, w, &result, &error);

  gsl_integration_workspace_free(w);
  return result;
}

static void update_mass_weighted_tau_e(const int snapshot)
{
  reion_grids_t* grids = &(run_globals.reion_grids);

  // Protect against accidental duplicate updates for the same snapshot.
  if (grids->tau_e_prev_snapshot == snapshot)
    return;

  double xHII = fmax(0.0, fmin(1.0, 1.0 - grids->mass_weighted_global_xH));

  if (grids->tau_e_prev_snapshot >= 0) {
    int prev_snapshot = grids->tau_e_prev_snapshot;
    double prev_z = run_globals.ZZ[prev_snapshot];
    double curr_z = run_globals.ZZ[snapshot];
    double dz = fabs(prev_z - curr_z);

    double f_prev = tau_e_sim_integrand_at_snapshot(prev_snapshot, grids->tau_e_prev_mass_weighted_xHII);
    double f_curr = tau_e_sim_integrand_at_snapshot(snapshot, xHII);
    grids->mass_weighted_global_tau_e_sim += 0.5 * (f_prev + f_curr) * dz;
  }

  grids->mass_weighted_global_tau_e = grids->mass_weighted_global_tau_e_sim + run_globals.tau_e_postEoR;

  grids->tau_e_prev_snapshot = snapshot;
  grids->tau_e_prev_mass_weighted_xHII = xHII;
}

void update_galaxy_fesc_vals(galaxy_t* gal, double new_stars, int snapshot)
{
  physics_params_t* params = &(run_globals.params.physics);

  float fesc_bh = (float)(params->EscapeFracBHNorm *
                          (powf((float)((1.0 + run_globals.ZZ[snapshot]) / 6.0), (float)params->EscapeFracBHScaling)));

  double fesc = params->EscapeFracNorm;
#if USE_MINI_HALOS
  double fescIII = params->EscapeFracNormIII;
#endif

  // redshift
  if ((params->EscapeFracDependency > 0) && (params->EscapeFracDependency <= 6))
    if (params->EscapeFracRedshiftScaling != 0.0) {
      fesc *=
        pow((1.0 + run_globals.ZZ[snapshot]) / params->EscapeFracRedshiftOffset, params->EscapeFracRedshiftScaling);
#if USE_MINI_HALOS
      fescIII *=
        pow((1.0 + run_globals.ZZ[snapshot]) / params->EscapeFracRedshiftOffset, params->EscapeFracRedshiftScaling);
#endif
    }

  // galaxy properties
  switch (params->EscapeFracDependency) {
    case 0:
    case 1:
      break;
    case 2: // stellar mass (Msun)
      if (gal->StellarMass > 0.0) {
        fesc *= pow((gal->StellarMass / run_globals.params.Hubble_h), params->EscapeFracPropScaling);
#if USE_MINI_HALOS
        // TODO: why is this not III
        fescIII *= pow((gal->StellarMass / run_globals.params.Hubble_h), params->EscapeFracPropScaling);
#endif
      } else {
        fesc = 1.0;
#if USE_MINI_HALOS
        fescIII = 1.0;
#endif
      }

      break;
    case 3: // star formation rate (Msun / yr)
      if (gal->Sfr > 0.0)
        fesc *=
          pow(gal->Sfr * run_globals.units.UnitMass_in_g / run_globals.units.UnitTime_in_s * SEC_PER_YEAR / SOLAR_MASS,
              params->EscapeFracPropScaling);
      else
        fesc = 0.0;
#if USE_MINI_HALOS
      if (gal->SfrIII > 0.0)
        fescIII *= pow(gal->SfrIII * run_globals.units.UnitMass_in_g / run_globals.units.UnitTime_in_s * SEC_PER_YEAR /
                         SOLAR_MASS,
                       params->EscapeFracPropScaling);
      else
        fescIII = 0.0;
#endif

      break;
    case 4: // cold gas density (Msun / pc^2)
      if ((gal->ColdGas > 0.0) && (gal->DiskScaleLength > 0.0)) {
        fesc *=
          pow((gal->ColdGas / gal->DiskScaleLength / gal->DiskScaleLength * 0.01 * run_globals.params.Hubble_h) / 10.,
              params->EscapeFracPropScaling);
#if USE_MINI_HALOS
        fescIII *=
          pow((gal->ColdGas / gal->DiskScaleLength / gal->DiskScaleLength * 0.01 * run_globals.params.Hubble_h) / 10.,
              params->EscapeFracPropScaling);
#endif
      } else {
        fesc = 1.0;
#if USE_MINI_HALOS
        fescIII = 1.0;
#endif
      }
      break;
    case 5: // halo mass (1e10 Msun)
      if (gal->Mvir > 0.0) {
        fesc *= pow(gal->Mvir / run_globals.params.Hubble_h, params->EscapeFracPropScaling);
#if USE_MINI_HALOS
        fescIII *= pow(gal->Mvir / run_globals.params.Hubble_h, params->EscapeFracPropScaling);
#endif
      } else {
        fesc = 1.0;
#if USE_MINI_HALOS
        fescIII = 1.0;
#endif
      }
      break;
    case 6: // specific star formation rate (10/ Gyr)
      if ((gal->Sfr > 0.0) && (gal->StellarMass > 0.0))
        fesc *= pow(gal->Sfr / gal->StellarMass / run_globals.units.UnitTime_in_s * SEC_PER_MEGAYEAR * 100,
                    params->EscapeFracPropScaling);
      else
        fesc = 0.0;
#if USE_MINI_HALOS
      // TODO: why are stellar masses not III
      if ((gal->SfrIII > 0.0) && (gal->StellarMass > 0.0))
        fescIII *= pow(gal->SfrIII / gal->StellarMass / run_globals.units.UnitTime_in_s * SEC_PER_MEGAYEAR * 100,
                       params->EscapeFracPropScaling);
      else
        fescIII = 0.0;
#endif
      break;
    default:
      mlog_error("Unrecognised EscapeFracDependency parameter value.");
  }
  // CGM suppression of fesc based on pre-computed tau_cgm (optical depth formulation)
  // Flag_FescCGMSuppression modes: 1 = instantaneous Gamma12, 2 = cumulative Gamma12, 3 = clumping factor
  if ((params->Flag_FescCGMSuppression > 0) && (gal->tau_cgm > 0.0)) {
    // Suppression through optical depth: fesc_suppressed = fesc * exp(-tau_CGM)
    // tau_cgm is computed during reionization grid processing and stored per galaxy
    double suppression = exp(-gal->tau_cgm);
    fesc *= suppression;
#if USE_MINI_HALOS
    fescIII *= suppression;
#endif
  }

  CLAMP_0_1(fesc);
#if USE_MINI_HALOS
  CLAMP_0_1(fescIII);
#endif
  CLAMP_0_1(fesc_bh);

#if USE_STOCHASTICITY
  double scattered_fesc;

#if USE_MINI_HALOS
  double scattered_fescIII;
#endif

  if (params->EscapeFracScatterDex > ABS_TOL) {
#if USE_MINI_HALOS
    if (gal->Galaxy_Population == 2) {
      scattered_fesc = apply_lognormal_scatter(fesc, params->EscapeFracScatterDex);
      CLAMP_0_1(scattered_fesc);
    } else if (gal->Galaxy_Population == 3) {
      scattered_fescIII = apply_lognormal_scatter(fescIII, params->EscapeFracScatterDex);
      CLAMP_0_1(scattered_fescIII);
    }
#else
    scattered_fesc = apply_lognormal_scatter(fesc, params->EscapeFracScatterDex);
    CLAMP_0_1(scattered_fesc);
#endif
  }
#endif

#if USE_MINI_HALOS
  if (gal->Galaxy_Population == 2) {
    gal->Fesc = fesc;
    gal->FescWeightedGSM += new_stars * fesc;
    gal->FescWeightedSfr += gal->Sfr * fesc;

#if USE_STOCHASTICITY
    if (params->EscapeFracScatterDex > ABS_TOL) {
      gal->StochasticityTreatedFescWeightedGSM += new_stars * scattered_fesc;
      gal->StochasticityTreatedFescWeightedSfr += gal->Sfr * scattered_fesc;
    };
#endif
  }

  if (gal->Galaxy_Population == 3) {
    gal->FescIII = fescIII;
    gal->FescIIIWeightedGSM += new_stars * fescIII;
    gal->FescIIIWeightedSfr += gal->SfrIII * fescIII;

#if USE_STOCHASTICITY
    if (params->EscapeFracScatterDex > ABS_TOL) {
      gal->StochasticityTreatedFescIIIWeightedGSM += new_stars * scattered_fescIII;
      gal->StochasticityTreatedFescIIIWeightedSfr += gal->SfrIII * scattered_fescIII;
    };
#endif
  }
#else
  gal->Fesc = fesc;
  gal->FescWeightedGSM += new_stars * fesc;
  gal->FescWeightedSfr += gal->Sfr * fesc;

#if USE_STOCHASTICITY
  if (params->EscapeFracScatterDex > ABS_TOL) {
    gal->StochasticityTreatedFescWeightedGSM += new_stars * scattered_fesc;
    gal->StochasticityTreatedFescWeightedSfr += gal->Sfr * scattered_fesc;
  };
#endif
#endif

  gal->FescBH = fesc_bh;
}

void set_quasar_fobs()
{
  physics_params_t* params = &(run_globals.params.physics);

  params->quasar_fobs = 1. - cos(params->quasar_open_angle / 180. * M_PI / 2.);
  mlog("Quasar radiation open angle is set to be %g, corresponding to an obscure fraction of %g",
       MLOG_MESG | MLOG_FLUSH,
       params->quasar_open_angle,
       params->quasar_fobs);
}

void set_ReionEfficiency() //  You need to update this function! You need to build a function that allows you to compute
                           //  the ReionNionPhotPerBary for PopIII stars depending on the IMF!
{
  // Use the params passed to Meraxes via the input file to set the HII ionising efficiency factor
  physics_params_t* params = &(run_globals.params.physics);

  // The following is based on Sobacchi & Messinger (2013) eqn 7
  // with f_* removed and f_b added since we define f_coll as M_*/M_tot rather than M_vir/M_tot,
  // and also with the inclusion of the effects of the Helium fraction.
  params->ReionEfficiency =
    1.0 / run_globals.params.BaryonFrac * params->ReionNionPhotPerBary / (1.0 - 0.75 * params->Y_He);

#if USE_MINI_HALOS
  params->ReionEfficiencyIII =
    1.0 / run_globals.params.BaryonFrac * params->ReionNionPhotPerBaryIII / (1.0 - 0.75 * params->Y_He);
#endif

  // Account for instantaneous recycling factor so that stellar mass is cumulative
  if (params->Flag_IRA) {
    params->ReionEfficiency /= params->SfRecycleFraction;
#if USE_MINI_HALOS
    params->ReionEfficiencyIII /= params->SfRecycleFraction_III;
#endif
  }

  mlog("Set value of run_globals.params.ReionEfficiency = %g", MLOG_MESG, params->ReionEfficiency);
#if USE_MINI_HALOS
  mlog("Set value of run_globals.params.ReionEfficiencyIII = %g", MLOG_MESG, params->ReionEfficiencyIII);
#endif
}

void assign_slabs()
{
  mlog("Assigning slabs to MPI cores...", MLOG_OPEN);

  // Assign the slab size
  int n_rank = run_globals.mpi_size;
  int dim = run_globals.params.ReionGridDim;

  // Use fftw to find out what slab each rank should get
  ptrdiff_t local_nix, local_ix_start;
  ptrdiff_t local_n_complex =
    fftwf_mpi_local_size_3d(dim, dim, dim / 2 + 1, run_globals.mpi_comm, &local_nix, &local_ix_start);

  // let every core know...
  ptrdiff_t** slab_nix = &run_globals.reion_grids.slab_nix;
  *slab_nix = malloc(sizeof(ptrdiff_t) * n_rank); ///< array of number of x cells of every rank
  MPI_Allgather(&local_nix, sizeof(ptrdiff_t), MPI_BYTE, *slab_nix, sizeof(ptrdiff_t), MPI_BYTE, run_globals.mpi_comm);

  ptrdiff_t** slab_ix_start = &run_globals.reion_grids.slab_ix_start;
  *slab_ix_start = malloc(sizeof(ptrdiff_t) * n_rank); ///< array first x cell of every rank
  (*slab_ix_start)[0] = 0;
  for (int ii = 1; ii < n_rank; ii++)
    (*slab_ix_start)[ii] = (*slab_ix_start)[ii - 1] + (*slab_nix)[ii - 1];

  ptrdiff_t** slab_n_complex = &run_globals.reion_grids.slab_n_complex; ///< array of allocation counts for every rank
  *slab_n_complex = malloc(sizeof(ptrdiff_t) * n_rank);                 ///< array of allocation counts for every rank
  MPI_Allgather(
    &local_n_complex, sizeof(ptrdiff_t), MPI_BYTE, *slab_n_complex, sizeof(ptrdiff_t), MPI_BYTE, run_globals.mpi_comm);

  mlog("...done", MLOG_CLOSE);
}

void call_find_HII_bubbles(int snapshot, int nout_gals, timer_info* timer)
{
  // Thin wrapper round find_HII_bubbles

  int total_n_out_gals = 0;
  int flag_output = 0;

  reion_grids_t* grids = &(run_globals.reion_grids);

  mlog("Getting ready to call find_HII_bubbles...", MLOG_OPEN);

  // Check to see if there are actually any galaxies at this snapshot
  MPI_Allreduce(&nout_gals, &total_n_out_gals, 1, MPI_INT, MPI_SUM, run_globals.mpi_comm);
  //    if (total_n_out_gals == 0) {
  //        mlog("No galaxies in the simulation - skipping...", MLOG_CLOSE);
  //        return;
  //    }

  // Logic statement to avoid gridding the density field twice: skip grid construction and
  // grid saving if Flag_IncludeSpinTemp is true, since these operations have already been
  // performed by the preceding call_find_HII_bubbles
  if (!run_globals.params.Flag_IncludeSpinTemp || !run_globals.params.ReionUVBFlag) {

    // Construct the baryon grids
    construct_baryon_grids(snapshot, nout_gals);

    // Read in the dark matter density grid
    read_grid(DENSITY, snapshot, grids->deltax);

    // save the grids prior to doing FFTs to avoid precision loss and aliasing etc.
    // NOTE: I dont think we are reusing them...
    if (!run_globals.params.FlagMCMC) {
      for (int i_out = 0; i_out < run_globals.NOutputSnaps; i_out++)
        if (snapshot == run_globals.ListOutputSnaps[i_out] && run_globals.params.Flag_OutputGrids) {
          save_reion_input_grids(snapshot);
          flag_output = 1;
        }
      if ((!flag_output) && (run_globals.mpi_rank == 0)) {
        hid_t file_id = create_reion_grid(snapshot, false);
        H5Fclose(file_id);
      }
    }
  }

  mlog("...done", MLOG_CLOSE);

  // Call find_HII_bubbles
  mlog("Calling find_HII_bubbles", MLOG_OPEN | MLOG_TIMERSTART);

  // Call find_HII_bubbles
  find_HII_bubbles(snapshot, timer);

  mlog("global quantities = volume-weighted VS mass-weighted", MLOG_MESG);
  mlog("xH = %g VS %g", MLOG_MESG, grids->volume_weighted_global_xH, grids->mass_weighted_global_xH);
  mlog("r_bubble = %g VS %g (h**-1 Mpc)",
       MLOG_MESG,
       grids->volume_weighted_global_r_bubble,
       grids->mass_weighted_global_r_bubble);
  mlog("temp_kinetic_all_gas = %g VS %g (K)",
       MLOG_MESG,
       grids->volume_weighted_global_temp_kinetic_all_gas,
       grids->mass_weighted_global_temp_kinetic_all_gas);
  if (run_globals.params.Flag_IncludeRecombinations) {
    mlog("Gamma12 = %g VS %g (h**2 1e-12 /s)",
         MLOG_MESG,
         grids->volume_weighted_global_Gamma12,
         grids->mass_weighted_global_Gamma12);
    mlog("N_rec = %g VS %g (/N_b)", MLOG_MESG, grids->volume_weighted_global_N_rec, grids->mass_weighted_global_N_rec);
    mlog("residual_xH = %g VS %g (x10**-4)",
         MLOG_MESG,
         grids->volume_weighted_global_residual_xH,
         grids->mass_weighted_global_residual_xH);
    mlog("clumping_factor = %g VS %g",
         MLOG_MESG,
         grids->volume_weighted_global_clumping_factor,
         grids->mass_weighted_global_clumping_factor);
  }

  mlog("sfr = %g (Msun/yr)", MLOG_MESG, grids->volume_weighted_global_weighted_sfr);
#if USE_MINI_HALOS
  mlog("sfrIII = %g (Msun/yr)", MLOG_MESG, grids->volume_weighted_global_weighted_sfrIII);
#endif
  mlog("bhar = %g (equivlently Msun/yr)", MLOG_MESG, grids->volume_weighted_global_effective_bhar);

  update_mass_weighted_tau_e(snapshot);
  mlog("tau_e(mass-weighted) = %.6f [sim=%.6f, postsim=%.6f]",
       MLOG_MESG,
       grids->mass_weighted_global_tau_e,
       grids->mass_weighted_global_tau_e_sim,
       run_globals.tau_e_postEoR);

  mlog("...done", MLOG_CLOSE | MLOG_TIMERSTOP);
}

void call_ComputeTs(int snapshot, int nout_gals, timer_info* timer)
{
  // Thin wrapper round ComputeTs

  int total_n_out_gals = 0;
  int flag_output = 0;

  reion_grids_t* grids = &(run_globals.reion_grids);

  mlog("Getting ready to call ComputeTs...", MLOG_OPEN);

  // Check to see if there are actually any galaxies at this snapshot
  MPI_Allreduce(&nout_gals, &total_n_out_gals, 1, MPI_INT, MPI_SUM, run_globals.mpi_comm);

  // Construct the baryon grids
  construct_baryon_grids(snapshot, nout_gals);

  // Read in the dark matter density grid
  read_grid(DENSITY, snapshot, grids->deltax);

  // read in the velocity grids
  if (run_globals.params.Flag_IncludePecVelsFor21cm > 0) {
    read_grid(run_globals.params.TsVelocityComponent, snapshot, grids->vel);
  }

  // save the grids prior to doing FFTs to avoid precision loss and aliasing etc.
  if (!run_globals.params.FlagMCMC) {
    for (int i_out = 0; i_out < run_globals.NOutputSnaps; i_out++)
      if (snapshot == run_globals.ListOutputSnaps[i_out]) {
        save_reion_input_grids(snapshot);
        flag_output = 1;
      }

    if ((!flag_output) && (run_globals.mpi_rank == 0)) {
      hid_t file_id = create_reion_grid(snapshot, false);
      H5Fclose(file_id);
    }
  }

  mlog("...done", MLOG_CLOSE);

  // Call Compute Ts
  mlog("Calling ComputeTs", MLOG_OPEN | MLOG_TIMERSTART);

  ComputeTs(snapshot, timer);
  mlog("...done", MLOG_CLOSE | MLOG_TIMERSTOP);
}

void init_reion_grids()
{
  run_globals.reion_grids.heating_history_head = 0;

  reion_grids_t* grids = &(run_globals.reion_grids);
  int ReionGridDim = run_globals.params.ReionGridDim;
  ptrdiff_t* slab_nix = run_globals.reion_grids.slab_nix;
  ptrdiff_t slab_n_real = slab_nix[run_globals.mpi_rank] * ReionGridDim * ReionGridDim;
  ptrdiff_t slab_n_complex = run_globals.reion_grids.slab_n_complex[run_globals.mpi_rank];
  ptrdiff_t slab_n_real_smoothedHeating;
  if (run_globals.params.Flag_IncludeSpinTemp) {
    slab_n_real_smoothedHeating =
      slab_nix[run_globals.mpi_rank] * run_globals.params.TsNumFilterSteps * ReionGridDim * ReionGridDim;
  }
  ptrdiff_t slab_n_real_LC;
  if (run_globals.params.Flag_ConstructLightcone) {
    slab_n_real_LC = slab_nix[run_globals.mpi_rank] * ReionGridDim * run_globals.params.LightconeLength;
  }

  mlog("Initialising grids...", MLOG_MESG);

  grids->volume_weighted_global_xH = 1.0;
  grids->volume_weighted_global_Gamma12 = 0.0;
  grids->volume_weighted_global_r_bubble = 0.0;
  grids->mass_weighted_global_xH = 1.0;
  grids->mass_weighted_global_tau_e = run_globals.tau_e_postEoR;
  grids->mass_weighted_global_tau_e_sim = 0.0;
  grids->tau_e_prev_snapshot = -1;
  grids->tau_e_prev_mass_weighted_xHII = 0.0;
  grids->started = 0;
  grids->finished = 0;

  grids->volume_ave_J_alpha = 0.0;
  grids->volume_ave_xalpha = 0.0;
  grids->volume_ave_Xheat = 0.0;
  grids->volume_ave_Xion = 0.0;
  grids->volume_ave_Xheat_AGN_soft = 0.0;
  grids->volume_ave_Xheat_AGN_hard = 0.0;
  grids->volume_ave_TS = 0.0;
  grids->volume_ave_TK = 0.0;
  grids->volume_ave_xe = 0.0;
  grids->volume_ave_Tb = 0.0;
#if USE_MINI_HALOS
  grids->volume_ave_J_alphaII = 0.0;
  grids->volume_ave_XheatII = 0.0;
  grids->volume_ave_XionII = 0.0;
  grids->volume_ave_J_LW = 0.0;
  grids->volume_ave_J_LWII = 0.0;
  grids->volume_ave_J_LW_AGN = 0.0;
  grids->volume_ave_TKII = 0.0;
  grids->volume_ave_TSII = 0.0;
  grids->volume_ave_TbII = 0.0;
#endif

  for (int ii = 0; ii < slab_n_real; ii++) {
    grids->xH[ii] = 1.0;
    grids->z_at_ionization[ii] = -1;
    grids->r_bubble[ii] = 0.0;
    grids->temp_kinetic_all_gas[ii] = 0.0;
#if USE_MINI_HALOS
    if (run_globals.params.Flag_IncludeLymanWerner) {
      grids->JLW_box[ii] = 0.0;
      grids->JLW_boxII[ii] = 0.0;
    }
#endif
    if (run_globals.params.Flag_IncludeSpinTemp) {
      grids->Tk_box[ii] = 0.0;
      grids->TS_box[ii] = 0.0;
#if USE_MINI_HALOS
      grids->Tk_boxII[ii] = 0.0;
      grids->TS_boxII[ii] = 0.0;
#endif
    }
    if (run_globals.params.Flag_IncludeRecombinations) {
      grids->z_re[ii] = 0.0;
      grids->Gamma12[ii] = 0.0;
      grids->residual_xH[ii] = 0;
      grids->clumping_factor[ii] = 0;
      grids->t_resp[ii] = 1e30;
    }
    if (run_globals.params.Flag_Compute21cmBrightTemp) {
      grids->delta_T[ii] = 0.0;
#if USE_MINI_HALOS
      grids->delta_TII[ii] = 0.0;
#endif
      if (run_globals.params.Flag_ConstructLightcone) {
        grids->delta_T_prev[ii] = 0.0;
#if USE_MINI_HALOS
        grids->delta_TII_prev[ii] = 0.0;
#endif
      }
    }
  }

  if (run_globals.params.Flag_IncludeSpinTemp) {

    for (int ii = 0; ii < slab_n_real_smoothedHeating; ii++) {
      grids->SMOOTHED_SFR_GAL[ii] = 0.0;
#if USE_STOCHASTICITY
      grids->SMOOTHED_XRAY_LUMINOSITY_GAL[ii] = 0.0;
#endif
#if USE_MINI_HALOS
      grids->SMOOTHED_SFR_III[ii] = 0.0;
#endif
    }
  }

  if (run_globals.params.Flag_ConstructLightcone) {
    for (int ii = 0; ii < slab_n_real_LC; ii++) {
      grids->LightconeBox[ii] = 0.0;
    }

    for (int ii = 0; ii < run_globals.params.LightconeLength; ii++) {
      grids->Lightcone_redshifts[ii] = 0.0;
    }
  }

  for (int ii = 0; ii < slab_n_real; ii++)
    if (run_globals.params.ReionUVBFlag) {
      grids->J_21_at_ionization[ii] = (float)0.;
      grids->Mvir_crit[ii] = (float)0.;
#if USE_MINI_HALOS
      if (run_globals.params.Flag_IncludeLymanWerner)
        grids->Mvir_crit_MC[ii] = (float)0.;
#endif
    }

  for (int ii = 0; ii < slab_n_complex; ii++) {
    grids->stars_filtered[ii] = 0 + 0I;
    grids->stars_unfiltered[ii] = 0 + 0I;
    if (run_globals.params.physics.Flag_BHFeedback) {
      grids->effective_bhm_filtered[ii] = 0 + 0I;
      grids->effective_bhm_unfiltered[ii] = 0 + 0I;
      grids->effective_bhar_filtered[ii] = 0 + 0I;
      grids->effective_bhar_unfiltered[ii] = 0 + 0I;
    }
    grids->deltax_filtered[ii] = 0 + 0I;
    grids->deltax_unfiltered[ii] = 0 + 0I;
    grids->weighted_sfr_filtered[ii] = 0 + 0I;
    grids->weighted_sfr_unfiltered[ii] = 0 + 0I;
#if USE_MINI_HALOS
    grids->starsIII_filtered[ii] = 0 + 0I;
    grids->starsIII_unfiltered[ii] = 0 + 0I;
    grids->weighted_sfrIII_filtered[ii] = 0 + 0I;
    grids->weighted_sfrIII_unfiltered[ii] = 0 + 0I;
#endif
    if (run_globals.params.Flag_IncludeSpinTemp) {
      grids->sfr_filtered[ii] = 0 + 0I;
      grids->sfr_unfiltered[ii] = 0 + 0I;
#if USE_STOCHASTICITY
      grids->xray_luminosity_filtered[ii] = 0 + 0I;
      grids->xray_luminosity_unfiltered[ii] = 0 + 0I;
#endif
      grids->x_e_filtered[ii] = 0 + 0I;
      grids->x_e_unfiltered[ii] = 0 + 0I;
#if USE_MINI_HALOS
      grids->sfrIII_filtered[ii] = 0 + 0I;
      grids->sfrIII_unfiltered[ii] = 0 + 0I;
#endif
    }
    if (run_globals.params.Flag_IncludeRecombinations) {
      grids->N_rec_filtered[ii] = 0 + 0I;
      grids->N_rec_unfiltered[ii] = 0 + 0I;
    }
    if (run_globals.params.Flag_Compute21cmBrightTemp && (run_globals.params.Flag_IncludePecVelsFor21cm > 0)) {
      grids->vel_gradient[ii] = 0 + 0I;
    }
  }

  for (int ii = 0; ii < slab_n_complex * 2; ii++) {
    grids->deltax[ii] = 0;
    grids->stars[ii] = 0;
    if (run_globals.params.physics.Flag_BHFeedback) {
      grids->effective_bhm[ii] = 0;
      grids->effective_bhar[ii] = 0;
    }
    grids->weighted_sfr[ii] = 0;
#if USE_MINI_HALOS
    grids->starsIII[ii] = 0;
    grids->weighted_sfrIII[ii] = 0;
#endif

    if (run_globals.params.Flag_IncludeSpinTemp) {
      grids->sfr[ii] = 0;
#if USE_STOCHASTICITY
      grids->xray_luminosity[ii] = 0;
#endif
      grids->x_e_box_prev[ii] = 0;
      grids->x_e_box[ii] = 0;
#if USE_MINI_HALOS
      grids->sfrIII[ii] = 0;
#endif
      for (int jj = 0; jj < run_globals.NstoreSnapshots_Heating; jj++) {
        grids->sfr_histories[jj * slab_n_complex * 2 + ii] = 0;
        size_t history_index = (size_t)jj * slab_n_complex * 2 + ii;
        if (grids->bh_xray_histories_hard)
          grids->bh_xray_histories_hard[history_index] = 0;
        if (grids->bh_xray_histories_soft)
          grids->bh_xray_histories_soft[history_index] = 0;
#if USE_MINI_HALOS
        if (grids->bh_uv_histories)
          grids->bh_uv_histories[history_index] = 0;
#endif
#if USE_STOCHASTICITY
        grids->xray_luminosity_histories[jj * slab_n_complex * 2 + ii] = 0;
#endif
#if USE_MINI_HALOS
        grids->sfrIII_histories[jj * slab_n_complex * 2 + ii] = 0;
#endif
      }
    }
    if (run_globals.params.Flag_IncludeRecombinations) {
      grids->N_rec[ii] = 0;
    }
    if (run_globals.params.Flag_Compute21cmBrightTemp && (run_globals.params.Flag_IncludePecVelsFor21cm > 0)) {
      grids->vel[ii] = 0;
    }
  }

  if (run_globals.params.Flag_ComputePS) {
    for (int ii = 0; ii < run_globals.params.PS_Length; ii++) {
      grids->PS_k[ii] = (float)0.;
      grids->PS_data[ii] = (float)0.;
      grids->PS_error[ii] = (float)0.;
#if USE_MINI_HALOS
      grids->PSII_data[ii] = (float)0.;
      grids->PSII_error[ii] = (float)0.;
#endif
    }
  }
}

void malloc_reionization_grids()
{
  mlog("Allocating reionization grids...", MLOG_OPEN);

  reion_grids_t* grids = &(run_globals.reion_grids);

  fftwf_mpi_init();

  // Load wisdom if requested
  run_globals.reion_grids.flag_wisdom = strlen(run_globals.params.FFTW3WisdomDir) > 0;
  bool save_wisdom = false;
  char wisdom_fname[STRLEN + 32];
  const int ReionGridDim = run_globals.params.ReionGridDim;
  unsigned plan_flags = FFTW_ESTIMATE;

  if (run_globals.reion_grids.flag_wisdom) {
    plan_flags = FFTW_PATIENT;
    sprintf(wisdom_fname,
            "%s/fftw3f-meraxes-N_%d-ranks_%d.wisdom",
            run_globals.params.FFTW3WisdomDir,
            ReionGridDim,
            run_globals.mpi_size);
    if (run_globals.mpi_rank == 0) {
      if (fftwf_import_wisdom_from_filename(wisdom_fname)) {
        mlog("Successfully loaded FFTW3 wisdom from %s", MLOG_MESG, wisdom_fname);
      } else {
        mlog("FFTW3 wisdom directory provided, but no suitable wisdom exists. New wisdom will be created (this make "
             "take a while).",
             MLOG_MESG | MLOG_FLUSH);
        save_wisdom = true;
        // Check to see if the wisdom directory exists and if not, create it
        struct stat filestatus;
        if (stat(run_globals.params.FFTW3WisdomDir, &filestatus) != 0)
          mkdir(run_globals.params.FFTW3WisdomDir, 02755);
      }
    }
    MPI_Bcast(&save_wisdom, 1, MPI_C_BOOL, 0, run_globals.mpi_comm);
    if (!save_wisdom) {
      fftwf_mpi_broadcast_wisdom(run_globals.mpi_comm);
    }
  }
  // run_globals.NStoreSnapshots is set in `initialize_halo_storage`
  run_globals.SnapshotDeltax = (float**)calloc((size_t)run_globals.NStoreSnapshots, sizeof(float*));
  run_globals.SnapshotVel = (float**)calloc((size_t)run_globals.NStoreSnapshots, sizeof(float*));

  grids->galaxy_to_slab_map = NULL;

  grids->xH = NULL;
  grids->stars = NULL;
  grids->stars_unfiltered = NULL;
  grids->stars_filtered = NULL;
  grids->effective_bhm = NULL;
  grids->effective_bhm_unfiltered = NULL;
  grids->effective_bhm_filtered = NULL;
  grids->effective_bhar = NULL;
  grids->effective_bhar_unfiltered = NULL;
  grids->effective_bhar_filtered = NULL;
  grids->deltax = NULL;
  grids->deltax_unfiltered = NULL;
  grids->deltax_filtered = NULL;
  grids->sfr = NULL;
  grids->sfr_histories = NULL;
  grids->sfr_unfiltered = NULL;
  grids->sfr_filtered = NULL;
#if USE_STOCHASTICITY
  grids->xray_luminosity = NULL;
  grids->xray_luminosity_histories = NULL;
  grids->xray_luminosity_unfiltered = NULL;
  grids->xray_luminosity_filtered = NULL;
#endif
  grids->weighted_sfr = NULL;
  grids->weighted_sfr_unfiltered = NULL;
  grids->weighted_sfr_filtered = NULL;
  grids->z_at_ionization = NULL;
  grids->J_21_at_ionization = NULL;
  grids->temp_kinetic_all_gas = NULL;

#if USE_MINI_HALOS
  grids->JLW_box = NULL;
  grids->JLW_boxII = NULL;
  grids->starsIII = NULL;
  grids->starsIII_unfiltered = NULL;
  grids->starsIII_filtered = NULL;
  grids->sfrIII = NULL;
  grids->sfrIII_histories = NULL;
  grids->sfrIII_unfiltered = NULL;
  grids->sfrIII_filtered = NULL;
  grids->weighted_sfrIII = NULL;
  grids->weighted_sfrIII_unfiltered = NULL;
  grids->weighted_sfrIII_filtered = NULL;
#endif

  // Grids required for the spin temperature calculation
  grids->x_e_box = NULL;
  grids->x_e_box_prev = NULL;
  grids->Tk_box = NULL;
  grids->TS_box = NULL;
  grids->x_e_unfiltered = NULL;
  grids->x_e_filtered = NULL;

  grids->SMOOTHED_SFR_GAL = NULL;
#if USE_STOCHASTICITY
  grids->SMOOTHED_XRAY_LUMINOSITY_GAL = NULL;
#endif

  grids->BHXrayEmissivity_hard = NULL;
  grids->bh_xray_histories_hard = NULL;
  grids->SMOOTHED_AGN_hard = NULL;
  grids->BHXrayEmissivity_soft = NULL;
  grids->bh_xray_histories_soft = NULL;
  grids->SMOOTHED_AGN_soft = NULL;

#if USE_MINI_HALOS
  grids->BHUVEmissivity = NULL;
  grids->bh_uv_histories = NULL;
  grids->SMOOTHED_AGN_UV = NULL;

  grids->Tk_boxII = NULL;
  grids->TS_boxII = NULL;

  grids->SMOOTHED_SFR_III = NULL;
#endif

  // Grids required for inhomogeneous recombinations
  grids->z_re = NULL;
  grids->Gamma12 = NULL;
  grids->residual_xH = NULL;
  grids->clumping_factor = NULL;
  grids->t_resp = NULL;
  grids->N_rec = NULL;
  grids->N_rec_filtered = NULL;
  grids->N_rec_unfiltered = NULL;

  // Grids required for 21cm brightness temperature
  grids->delta_T = NULL;
  grids->delta_T_prev = NULL;
#if USE_MINI_HALOS
  grids->delta_TII = NULL;
  grids->delta_TII_prev = NULL;
#endif

  // A grid for the lightcone (cuboid) box
  grids->LightconeBox = NULL;
  grids->Lightcone_redshifts = NULL;

  // Grids required for addining in peculiar velocity effects
  grids->vel = NULL;
  grids->vel_gradient = NULL;

  grids->PS_k = NULL;
  grids->PS_data = NULL;
  grids->PS_error = NULL;

#if USE_MINI_HALOS
  grids->PSII_data = NULL;
  grids->PSII_error = NULL;
#endif

  if (run_globals.params.Flag_PatchyReion) {
    assign_slabs();

    int ReionGridDim = run_globals.params.ReionGridDim;
    ptrdiff_t* slab_nix = run_globals.reion_grids.slab_nix;
    ptrdiff_t slab_n_real = slab_nix[run_globals.mpi_rank] * ReionGridDim * ReionGridDim;
    ptrdiff_t slab_n_complex = run_globals.reion_grids.slab_n_complex[run_globals.mpi_rank];

    ptrdiff_t slab_n_real_smoothedHeating;
    if (run_globals.params.Flag_IncludeSpinTemp) {
      slab_n_real_smoothedHeating =
        slab_nix[run_globals.mpi_rank] * run_globals.params.TsNumFilterSteps * ReionGridDim * ReionGridDim;
    }

    ptrdiff_t slab_n_real_LC;
    if (run_globals.params.Flag_ConstructLightcone) {
      slab_n_real_LC = slab_nix[run_globals.mpi_rank] * ReionGridDim * run_globals.params.LightconeLength;
    }

    // create a buffer on each rank which is as large as the largest LOGICAL allocation on any single rank
    int max_cells = 0;

    for (int ii = 0; ii < run_globals.mpi_size; ii++)
      if (slab_nix[ii] > max_cells)
        max_cells = (int)slab_nix[ii];

    max_cells *= ReionGridDim * ReionGridDim;
    grids->buffer_size = max_cells;

    grids->buffer = fftwf_alloc_real((size_t)max_cells);

    grids->stars = fftwf_alloc_real((size_t)slab_n_complex * 2);
    grids->stars_unfiltered = fftwf_alloc_complex((size_t)slab_n_complex);
    grids->stars_filtered = fftwf_alloc_complex((size_t)slab_n_complex);

    grids->stars_forward_plan = fftwf_mpi_plan_dft_r2c_3d(ReionGridDim,
                                                          ReionGridDim,
                                                          ReionGridDim,
                                                          grids->stars,
                                                          grids->stars_unfiltered,
                                                          run_globals.mpi_comm,
                                                          plan_flags);
    grids->stars_filtered_reverse_plan = fftwf_mpi_plan_dft_c2r_3d(ReionGridDim,
                                                                   ReionGridDim,
                                                                   ReionGridDim,
                                                                   grids->stars_filtered,
                                                                   (float*)grids->stars_filtered,
                                                                   run_globals.mpi_comm,
                                                                   plan_flags);

    if (run_globals.params.physics.Flag_BHFeedback) {
      grids->effective_bhm = fftwf_alloc_real((size_t)slab_n_complex * 2);
      grids->effective_bhm_unfiltered = fftwf_alloc_complex((size_t)slab_n_complex);
      grids->effective_bhm_filtered = fftwf_alloc_complex((size_t)slab_n_complex);

      grids->effective_bhm_forward_plan = fftwf_mpi_plan_dft_r2c_3d(ReionGridDim,
                                                                    ReionGridDim,
                                                                    ReionGridDim,
                                                                    grids->effective_bhm,
                                                                    grids->effective_bhm_unfiltered,
                                                                    run_globals.mpi_comm,
                                                                    plan_flags);
      grids->effective_bhm_filtered_reverse_plan = fftwf_mpi_plan_dft_c2r_3d(ReionGridDim,
                                                                             ReionGridDim,
                                                                             ReionGridDim,
                                                                             grids->effective_bhm_filtered,
                                                                             (float*)grids->effective_bhm_filtered,
                                                                             run_globals.mpi_comm,
                                                                             plan_flags);

      grids->effective_bhar = fftwf_alloc_real((size_t)slab_n_complex * 2);
      grids->effective_bhar_unfiltered = fftwf_alloc_complex((size_t)slab_n_complex);
      grids->effective_bhar_filtered = fftwf_alloc_complex((size_t)slab_n_complex);

      grids->effective_bhar_forward_plan = fftwf_mpi_plan_dft_r2c_3d(ReionGridDim,
                                                                     ReionGridDim,
                                                                     ReionGridDim,
                                                                     grids->effective_bhar,
                                                                     grids->effective_bhar_unfiltered,
                                                                     run_globals.mpi_comm,
                                                                     plan_flags);
      grids->effective_bhar_filtered_reverse_plan = fftwf_mpi_plan_dft_c2r_3d(ReionGridDim,
                                                                              ReionGridDim,
                                                                              ReionGridDim,
                                                                              grids->effective_bhar_filtered,
                                                                              (float*)grids->effective_bhar_filtered,
                                                                              run_globals.mpi_comm,
                                                                              plan_flags);
    }
    grids->deltax = fftwf_alloc_real((size_t)slab_n_complex * 2);
    grids->deltax_unfiltered = fftwf_alloc_complex((size_t)slab_n_complex);
    grids->deltax_filtered = fftwf_alloc_complex((size_t)slab_n_complex);

    grids->deltax_forward_plan = fftwf_mpi_plan_dft_r2c_3d(ReionGridDim,
                                                           ReionGridDim,
                                                           ReionGridDim,
                                                           grids->deltax,
                                                           grids->deltax_unfiltered,
                                                           run_globals.mpi_comm,
                                                           plan_flags);
    grids->deltax_filtered_reverse_plan = fftwf_mpi_plan_dft_c2r_3d(ReionGridDim,
                                                                    ReionGridDim,
                                                                    ReionGridDim,
                                                                    grids->deltax_filtered,
                                                                    (float*)grids->deltax_filtered,
                                                                    run_globals.mpi_comm,
                                                                    plan_flags);

    grids->weighted_sfr = fftwf_alloc_real((size_t)slab_n_complex * 2);
    grids->weighted_sfr_unfiltered = fftwf_alloc_complex((size_t)slab_n_complex);
    grids->weighted_sfr_filtered = fftwf_alloc_complex((size_t)slab_n_complex);

    grids->weighted_sfr_forward_plan = fftwf_mpi_plan_dft_r2c_3d(ReionGridDim,
                                                                 ReionGridDim,
                                                                 ReionGridDim,
                                                                 grids->weighted_sfr,
                                                                 grids->weighted_sfr_unfiltered,
                                                                 run_globals.mpi_comm,
                                                                 plan_flags);
    grids->weighted_sfr_filtered_reverse_plan = fftwf_mpi_plan_dft_c2r_3d(ReionGridDim,
                                                                          ReionGridDim,
                                                                          ReionGridDim,
                                                                          grids->weighted_sfr_filtered,
                                                                          (float*)grids->weighted_sfr_filtered,
                                                                          run_globals.mpi_comm,
                                                                          plan_flags);
#if USE_MINI_HALOS
    grids->starsIII = fftwf_alloc_real((size_t)slab_n_complex * 2);
    grids->starsIII_unfiltered = fftwf_alloc_complex((size_t)slab_n_complex);
    grids->starsIII_filtered = fftwf_alloc_complex((size_t)slab_n_complex);

    grids->starsIII_forward_plan = fftwf_mpi_plan_dft_r2c_3d(ReionGridDim,
                                                             ReionGridDim,
                                                             ReionGridDim,
                                                             grids->starsIII,
                                                             grids->starsIII_unfiltered,
                                                             run_globals.mpi_comm,
                                                             plan_flags);
    grids->starsIII_filtered_reverse_plan = fftwf_mpi_plan_dft_c2r_3d(ReionGridDim,
                                                                      ReionGridDim,
                                                                      ReionGridDim,
                                                                      grids->starsIII_filtered,
                                                                      (float*)grids->starsIII_filtered,
                                                                      run_globals.mpi_comm,
                                                                      plan_flags);

    grids->weighted_sfrIII = fftwf_alloc_real((size_t)slab_n_complex * 2);
    grids->weighted_sfrIII_unfiltered = fftwf_alloc_complex((size_t)slab_n_complex);
    grids->weighted_sfrIII_filtered = fftwf_alloc_complex((size_t)slab_n_complex);

    grids->weighted_sfrIII_forward_plan = fftwf_mpi_plan_dft_r2c_3d(ReionGridDim,
                                                                    ReionGridDim,
                                                                    ReionGridDim,
                                                                    grids->weighted_sfrIII,
                                                                    grids->weighted_sfrIII_unfiltered,
                                                                    run_globals.mpi_comm,
                                                                    plan_flags);
    grids->weighted_sfrIII_filtered_reverse_plan = fftwf_mpi_plan_dft_c2r_3d(ReionGridDim,
                                                                             ReionGridDim,
                                                                             ReionGridDim,
                                                                             grids->weighted_sfrIII_filtered,
                                                                             (float*)grids->weighted_sfrIII_filtered,
                                                                             run_globals.mpi_comm,
                                                                             plan_flags);
#endif

    grids->xH = fftwf_alloc_real((size_t)slab_n_real);
    grids->z_at_ionization = fftwf_alloc_real((size_t)slab_n_real);
    grids->r_bubble = fftwf_alloc_real((size_t)slab_n_real);
    grids->temp_kinetic_all_gas = fftwf_alloc_real((size_t)slab_n_real);

#if USE_MINI_HALOS
    if (run_globals.params.Flag_IncludeLymanWerner) {
      grids->JLW_box = fftwf_alloc_real((size_t)slab_n_real);
      grids->JLW_boxII = fftwf_alloc_real((size_t)slab_n_real);
    }
#endif

    if (run_globals.params.Flag_IncludeSpinTemp) {

      grids->sfr = fftwf_alloc_real((size_t)slab_n_complex * 2);
      grids->sfr_histories = fftwf_alloc_real((size_t)slab_n_complex * 2 * run_globals.NstoreSnapshots_Heating);
      grids->sfr_unfiltered = fftwf_alloc_complex((size_t)slab_n_complex);
      grids->sfr_filtered = fftwf_alloc_complex((size_t)slab_n_complex);

      grids->sfr_forward_plan = fftwf_mpi_plan_dft_r2c_3d(
        ReionGridDim, ReionGridDim, ReionGridDim, grids->sfr, grids->sfr_unfiltered, run_globals.mpi_comm, plan_flags);
      grids->sfr_filtered_reverse_plan = fftwf_mpi_plan_dft_c2r_3d(ReionGridDim,
                                                                   ReionGridDim,
                                                                   ReionGridDim,
                                                                   grids->sfr_filtered,
                                                                   (float*)grids->sfr_filtered,
                                                                   run_globals.mpi_comm,
                                                                   plan_flags);

#if USE_STOCHASTICITY
      grids->xray_luminosity = fftwf_alloc_real((size_t)slab_n_complex * 2);
      grids->xray_luminosity_histories =
        fftwf_alloc_real((size_t)slab_n_complex * 2 * run_globals.NstoreSnapshots_Heating);
      grids->xray_luminosity_unfiltered = fftwf_alloc_complex((size_t)slab_n_complex);
      grids->xray_luminosity_filtered = fftwf_alloc_complex((size_t)slab_n_complex);

      grids->xray_luminosity_forward_plan = fftwf_mpi_plan_dft_r2c_3d(ReionGridDim,
                                                                      ReionGridDim,
                                                                      ReionGridDim,
                                                                      grids->xray_luminosity,
                                                                      grids->xray_luminosity_unfiltered,
                                                                      run_globals.mpi_comm,
                                                                      plan_flags);
      grids->xray_luminosity_filtered_reverse_plan = fftwf_mpi_plan_dft_c2r_3d(ReionGridDim,
                                                                               ReionGridDim,
                                                                               ReionGridDim,
                                                                               grids->xray_luminosity_filtered,
                                                                               (float*)grids->xray_luminosity_filtered,
                                                                               run_globals.mpi_comm,
                                                                               plan_flags);
#endif

      {
        /* Flag_IncludeAGNXray: 0=off, 1=soft+hard, 2=hard only, 3=soft only.
         * Allocate each band's buffers only when it's actually selected. */
        int flag_agn = run_globals.params.physics.Flag_IncludeAGNXray;
        bool agn_hard_needed = (flag_agn == 1 || flag_agn == 2);
        bool agn_soft_needed = (flag_agn == 1 || flag_agn == 3);

        if (agn_hard_needed) {
          grids->BHXrayEmissivity_hard = fftwf_alloc_real((size_t)slab_n_complex * 2);
          grids->bh_xray_histories_hard =
            fftwf_alloc_real((size_t)slab_n_complex * 2 * run_globals.NstoreSnapshots_Heating);

          grids->BHXrayEmissivity_hard_unfiltered = fftwf_alloc_complex((size_t)slab_n_complex);
          grids->BHXrayEmissivity_hard_filtered = fftwf_alloc_complex((size_t)slab_n_complex);
          grids->BHXrayEmissivity_hard_forward_plan = fftwf_mpi_plan_dft_r2c_3d(ReionGridDim,
                                                                                ReionGridDim,
                                                                                ReionGridDim,
                                                                                grids->BHXrayEmissivity_hard,
                                                                                grids->BHXrayEmissivity_hard_unfiltered,
                                                                                run_globals.mpi_comm,
                                                                                plan_flags);
          grids->BHXrayEmissivity_hard_filtered_reverse_plan =
            fftwf_mpi_plan_dft_c2r_3d(ReionGridDim,
                                      ReionGridDim,
                                      ReionGridDim,
                                      grids->BHXrayEmissivity_hard_filtered,
                                      (float*)grids->BHXrayEmissivity_hard_filtered,
                                      run_globals.mpi_comm,
                                      plan_flags);

          /*
           * Fail loudly here if allocation ever fails, rather than leaving a
           * NULL pointer for downstream code to silently work around. Nothing
           * else in this function checks its fftwf_alloc_* calls either, but
           * these are the only ones that are conditionally allocated, so
           * a NULL here is the one case worth telling apart from "feature off".
           */
          if (grids->BHXrayEmissivity_hard == NULL || grids->bh_xray_histories_hard == NULL ||
              grids->BHXrayEmissivity_hard_unfiltered == NULL || grids->BHXrayEmissivity_hard_filtered == NULL) {
            mlog_error("Failed to allocate AGN X-ray (hard band) emissivity grids.");
            ABORT(EXIT_FAILURE);
          }
          for (size_t ii = 0; ii < (size_t)slab_n_complex * 2; ii++)
            grids->BHXrayEmissivity_hard[ii] = 0.0f;
          for (size_t ii = 0; ii < (size_t)slab_n_complex * 2 * run_globals.NstoreSnapshots_Heating; ii++)
            grids->bh_xray_histories_hard[ii] = 0.0f;

          // Same size as SMOOTHED_SFR_GAL; calloc ensures zeroed on allocation.
          grids->SMOOTHED_AGN_hard = calloc((size_t)slab_n_real_smoothedHeating, sizeof(double));
        }

        if (agn_soft_needed) {
          grids->BHXrayEmissivity_soft = fftwf_alloc_real((size_t)slab_n_complex * 2);
          grids->bh_xray_histories_soft =
            fftwf_alloc_real((size_t)slab_n_complex * 2 * run_globals.NstoreSnapshots_Heating);

          grids->BHXrayEmissivity_soft_unfiltered = fftwf_alloc_complex((size_t)slab_n_complex);
          grids->BHXrayEmissivity_soft_filtered = fftwf_alloc_complex((size_t)slab_n_complex);
          grids->BHXrayEmissivity_soft_forward_plan = fftwf_mpi_plan_dft_r2c_3d(ReionGridDim,
                                                                                ReionGridDim,
                                                                                ReionGridDim,
                                                                                grids->BHXrayEmissivity_soft,
                                                                                grids->BHXrayEmissivity_soft_unfiltered,
                                                                                run_globals.mpi_comm,
                                                                                plan_flags);
          grids->BHXrayEmissivity_soft_filtered_reverse_plan =
            fftwf_mpi_plan_dft_c2r_3d(ReionGridDim,
                                      ReionGridDim,
                                      ReionGridDim,
                                      grids->BHXrayEmissivity_soft_filtered,
                                      (float*)grids->BHXrayEmissivity_soft_filtered,
                                      run_globals.mpi_comm,
                                      plan_flags);

          if (grids->BHXrayEmissivity_soft == NULL || grids->bh_xray_histories_soft == NULL ||
              grids->BHXrayEmissivity_soft_unfiltered == NULL || grids->BHXrayEmissivity_soft_filtered == NULL) {
            mlog_error("Failed to allocate AGN X-ray (soft band) emissivity grids.");
            ABORT(EXIT_FAILURE);
          }
          for (size_t ii = 0; ii < (size_t)slab_n_complex * 2; ii++)
            grids->BHXrayEmissivity_soft[ii] = 0.0f;
          for (size_t ii = 0; ii < (size_t)slab_n_complex * 2 * run_globals.NstoreSnapshots_Heating; ii++)
            grids->bh_xray_histories_soft[ii] = 0.0f;

          grids->SMOOTHED_AGN_soft = calloc((size_t)slab_n_real_smoothedHeating, sizeof(double));
        }
      }

#if USE_MINI_HALOS
      if (run_globals.params.Flag_IncludeLymanWerner) {
        grids->BHUVEmissivity = fftwf_alloc_real((size_t)slab_n_complex * 2);
        grids->bh_uv_histories = fftwf_alloc_real((size_t)slab_n_complex * 2 * run_globals.NstoreSnapshots_Heating);

        grids->BHUVEmissivity_unfiltered = fftwf_alloc_complex((size_t)slab_n_complex);
        grids->BHUVEmissivity_filtered = fftwf_alloc_complex((size_t)slab_n_complex);
        grids->BHUVEmissivity_forward_plan = fftwf_mpi_plan_dft_r2c_3d(ReionGridDim,
                                                                       ReionGridDim,
                                                                       ReionGridDim,
                                                                       grids->BHUVEmissivity,
                                                                       grids->BHUVEmissivity_unfiltered,
                                                                       run_globals.mpi_comm,
                                                                       plan_flags);
        grids->BHUVEmissivity_filtered_reverse_plan = fftwf_mpi_plan_dft_c2r_3d(ReionGridDim,
                                                                                ReionGridDim,
                                                                                ReionGridDim,
                                                                                grids->BHUVEmissivity_filtered,
                                                                                (float*)grids->BHUVEmissivity_filtered,
                                                                                run_globals.mpi_comm,
                                                                                plan_flags);

        if (grids->BHUVEmissivity == NULL || grids->bh_uv_histories == NULL ||
            grids->BHUVEmissivity_unfiltered == NULL || grids->BHUVEmissivity_filtered == NULL) {
          mlog_error("Failed to allocate AGN Lyman-Werner emissivity grids.");
          ABORT(EXIT_FAILURE);
        }
        for (size_t ii = 0; ii < (size_t)slab_n_complex * 2; ii++)
          grids->BHUVEmissivity[ii] = 0.0f;
        for (size_t ii = 0; ii < (size_t)slab_n_complex * 2 * run_globals.NstoreSnapshots_Heating; ii++)
          grids->bh_uv_histories[ii] = 0.0f;

        grids->SMOOTHED_AGN_UV = calloc((size_t)slab_n_real_smoothedHeating, sizeof(double));
      }
#endif

      grids->x_e_box = fftwf_alloc_real((size_t)slab_n_complex * 2);
      grids->x_e_unfiltered = fftwf_alloc_complex((size_t)slab_n_complex);
      grids->x_e_filtered = fftwf_alloc_complex((size_t)slab_n_complex);
      grids->x_e_box_prev = fftwf_alloc_real((size_t)slab_n_complex * 2);

      grids->x_e_box_forward_plan = fftwf_mpi_plan_dft_r2c_3d(ReionGridDim,
                                                              ReionGridDim,
                                                              ReionGridDim,
                                                              grids->x_e_box,
                                                              grids->x_e_unfiltered,
                                                              run_globals.mpi_comm,
                                                              plan_flags);
      grids->x_e_filtered_reverse_plan = fftwf_mpi_plan_dft_c2r_3d(ReionGridDim,
                                                                   ReionGridDim,
                                                                   ReionGridDim,
                                                                   grids->x_e_filtered,
                                                                   (float*)grids->x_e_filtered,
                                                                   run_globals.mpi_comm,
                                                                   plan_flags);

#if USE_MINI_HALOS
      grids->sfrIII = fftwf_alloc_real((size_t)slab_n_complex * 2);
      grids->sfrIII_histories = fftwf_alloc_real((size_t)slab_n_complex * 2 * run_globals.NstoreSnapshots_Heating);
      grids->sfrIII_unfiltered = fftwf_alloc_complex((size_t)slab_n_complex);
      grids->sfrIII_filtered = fftwf_alloc_complex((size_t)slab_n_complex);

      grids->sfrIII_forward_plan = fftwf_mpi_plan_dft_r2c_3d(ReionGridDim,
                                                             ReionGridDim,
                                                             ReionGridDim,
                                                             grids->sfrIII,
                                                             grids->sfrIII_unfiltered,
                                                             run_globals.mpi_comm,
                                                             plan_flags);
      grids->sfrIII_filtered_reverse_plan = fftwf_mpi_plan_dft_c2r_3d(ReionGridDim,
                                                                      ReionGridDim,
                                                                      ReionGridDim,
                                                                      grids->sfrIII_filtered,
                                                                      (float*)grids->sfrIII_filtered,
                                                                      run_globals.mpi_comm,
                                                                      plan_flags);

#endif
      grids->Tk_box = fftwf_alloc_real((size_t)slab_n_real);
      grids->TS_box = fftwf_alloc_real((size_t)slab_n_real);

      grids->SMOOTHED_SFR_GAL = calloc((size_t)slab_n_real_smoothedHeating, sizeof(double));
#if USE_STOCHASTICITY
      grids->SMOOTHED_XRAY_LUMINOSITY_GAL = calloc((size_t)slab_n_real_smoothedHeating, sizeof(double));
#endif
#if USE_MINI_HALOS
      grids->Tk_boxII = fftwf_alloc_real((size_t)slab_n_real);
      grids->TS_boxII = fftwf_alloc_real((size_t)slab_n_real);

      grids->SMOOTHED_SFR_III = calloc((size_t)slab_n_real_smoothedHeating, sizeof(double));
#endif
    }

    if (run_globals.params.Flag_IncludeRecombinations) {
      grids->N_rec = fftwf_alloc_real((size_t)slab_n_complex * 2);
      grids->N_rec_unfiltered = fftwf_alloc_complex((size_t)slab_n_complex);
      grids->N_rec_filtered = fftwf_alloc_complex((size_t)slab_n_complex);

      grids->N_rec_forward_plan = fftwf_mpi_plan_dft_r2c_3d(ReionGridDim,
                                                            ReionGridDim,
                                                            ReionGridDim,
                                                            grids->N_rec,
                                                            grids->N_rec_unfiltered,
                                                            run_globals.mpi_comm,
                                                            plan_flags);
      grids->N_rec_filtered_reverse_plan = fftwf_mpi_plan_dft_c2r_3d(ReionGridDim,
                                                                     ReionGridDim,
                                                                     ReionGridDim,
                                                                     grids->N_rec_filtered,
                                                                     (float*)grids->N_rec_filtered,
                                                                     run_globals.mpi_comm,
                                                                     plan_flags);

      grids->z_re = fftwf_alloc_real((size_t)slab_n_real);
      grids->Gamma12 = fftwf_alloc_real((size_t)slab_n_real);
      grids->residual_xH = fftwf_alloc_real((size_t)slab_n_real);
      grids->clumping_factor = fftwf_alloc_real((size_t)slab_n_real);
      grids->t_resp = fftwf_alloc_real((size_t)slab_n_real);
    }

    if (run_globals.params.Flag_Compute21cmBrightTemp) {
      grids->delta_T = fftwf_alloc_real((size_t)slab_n_real);
#if USE_MINI_HALOS
      grids->delta_TII = fftwf_alloc_real((size_t)slab_n_real);
#endif

      if (run_globals.params.Flag_IncludePecVelsFor21cm > 0) {
        grids->vel = fftwf_alloc_real((size_t)slab_n_complex * 2);
        grids->vel_gradient = fftwf_alloc_complex((size_t)slab_n_complex);

        grids->vel_forward_plan = fftwf_mpi_plan_dft_r2c_3d(
          ReionGridDim, ReionGridDim, ReionGridDim, grids->vel, grids->vel_gradient, run_globals.mpi_comm, plan_flags);
        grids->vel_gradient_reverse_plan = fftwf_mpi_plan_dft_c2r_3d(ReionGridDim,
                                                                     ReionGridDim,
                                                                     ReionGridDim,
                                                                     grids->vel_gradient,
                                                                     (float*)grids->vel_gradient,
                                                                     run_globals.mpi_comm,
                                                                     plan_flags);
      }

      if (run_globals.params.Flag_ConstructLightcone) {
        grids->delta_T_prev = fftwf_alloc_real((size_t)slab_n_real);
#if USE_MINI_HALOS
        grids->delta_TII_prev = fftwf_alloc_real((size_t)slab_n_real);
#endif
      }
    }

    if (run_globals.params.ReionUVBFlag) {
      grids->J_21_at_ionization = fftwf_alloc_real((size_t)slab_n_real);
      grids->Mvir_crit = fftwf_alloc_real((size_t)slab_n_real);

#if USE_MINI_HALOS
      if (run_globals.params.Flag_IncludeLymanWerner)
        grids->Mvir_crit_MC = fftwf_alloc_real((size_t)slab_n_real);
#endif
    }

    if (run_globals.params.Flag_ConstructLightcone) {
      grids->LightconeBox = fftwf_alloc_real((size_t)slab_n_real_LC);
      grids->Lightcone_redshifts = fftwf_alloc_real((size_t)run_globals.params.LightconeLength);
    }

    if (run_globals.params.Flag_ComputePS) {
      grids->PS_k = fftwf_alloc_real((size_t)run_globals.params.PS_Length);
      grids->PS_data = fftwf_alloc_real((size_t)run_globals.params.PS_Length);
      grids->PS_error = fftwf_alloc_real((size_t)run_globals.params.PS_Length);
#if USE_MINI_HALOS
      grids->PSII_data = fftwf_alloc_real((size_t)run_globals.params.PS_Length);
      grids->PSII_error = fftwf_alloc_real((size_t)run_globals.params.PS_Length);
#endif
    }

    init_reion_grids();

    if (run_globals.reion_grids.flag_wisdom && save_wisdom) {
      fftwf_mpi_gather_wisdom(run_globals.mpi_comm);
      if (run_globals.mpi_rank == 0) {
        if (fftwf_export_wisdom_to_filename(wisdom_fname)) {
          mlog("Successfully saved FFTW3 wisdom to %s", MLOG_MESG, wisdom_fname);
        }
      }
    }

  } // if (run_globals.params.Flag_PatchyReion)

  mlog("...done", MLOG_CLOSE);
}

void free_reionization_grids()
{
  mlog("Freeing reionization grids...", MLOG_OPEN);
  free_filter_window_cache();

  reion_grids_t* grids = &(run_globals.reion_grids);

  free(run_globals.reion_grids.slab_n_complex);
  free(run_globals.reion_grids.slab_ix_start);
  free(run_globals.reion_grids.slab_nix);

  if (run_globals.params.Flag_ComputePS) {
    fftwf_free(grids->PS_error);
    fftwf_free(grids->PS_data);
    fftwf_free(grids->PS_k);
#if USE_MINI_HALOS
    fftwf_free(grids->PSII_error);
    fftwf_free(grids->PSII_data);
#endif
  }

  if (run_globals.params.Flag_ConstructLightcone) {
    fftwf_free(grids->LightconeBox);
  }

  if (run_globals.params.ReionUVBFlag) {
    fftwf_free(grids->Mvir_crit);
    fftwf_free(grids->J_21_at_ionization);

#if USE_MINI_HALOS
    if (run_globals.params.Flag_IncludeLymanWerner)
      fftwf_free(grids->Mvir_crit_MC);
#endif
  }

  if (run_globals.params.Flag_Compute21cmBrightTemp) {

    if (run_globals.params.Flag_ConstructLightcone) {
      fftwf_free(grids->delta_T_prev);
#if USE_MINI_HALOS
      fftwf_free(grids->delta_TII_prev);
#endif
      fftwf_free(grids->Lightcone_redshifts);
    }

    if (run_globals.params.Flag_IncludePecVelsFor21cm > 0) {
      fftwf_destroy_plan(grids->vel_gradient_reverse_plan);
      fftwf_destroy_plan(grids->vel_forward_plan);
      fftwf_free(grids->vel_gradient);
      fftwf_free(grids->vel);
    }

    fftwf_free(grids->delta_T);
#if USE_MINI_HALOS
    fftwf_free(grids->delta_TII);
#endif
  }

  if (run_globals.params.Flag_IncludeRecombinations) {

    fftwf_free(grids->Gamma12);
    fftwf_free(grids->z_re);
    fftwf_free(grids->residual_xH);
    fftwf_free(grids->clumping_factor);
    fftwf_free(grids->t_resp);

    fftwf_destroy_plan(grids->N_rec_filtered_reverse_plan);
    fftwf_destroy_plan(grids->N_rec_forward_plan);
    fftwf_free(grids->N_rec_filtered);
    fftwf_free(grids->N_rec_unfiltered);
    fftwf_free(grids->N_rec);
  }

  if (run_globals.params.Flag_IncludeSpinTemp) {
    free(grids->SMOOTHED_SFR_GAL);
#if USE_STOCHASTICITY
    free(grids->SMOOTHED_XRAY_LUMINOSITY_GAL);
#endif

    free(grids->SMOOTHED_AGN_hard);
    free(grids->SMOOTHED_AGN_soft);
    {
      int flag_agn = run_globals.params.physics.Flag_IncludeAGNXray;
      if (flag_agn == 1 || flag_agn == 2) {
        fftwf_destroy_plan(grids->BHXrayEmissivity_hard_filtered_reverse_plan);
        fftwf_destroy_plan(grids->BHXrayEmissivity_hard_forward_plan);
        fftwf_free(grids->BHXrayEmissivity_hard_filtered);
        fftwf_free(grids->BHXrayEmissivity_hard_unfiltered);
        fftwf_free(grids->BHXrayEmissivity_hard);
        fftwf_free(grids->bh_xray_histories_hard);
      }
      if (flag_agn == 1 || flag_agn == 3) {
        fftwf_destroy_plan(grids->BHXrayEmissivity_soft_filtered_reverse_plan);
        fftwf_destroy_plan(grids->BHXrayEmissivity_soft_forward_plan);
        fftwf_free(grids->BHXrayEmissivity_soft_filtered);
        fftwf_free(grids->BHXrayEmissivity_soft_unfiltered);
        fftwf_free(grids->BHXrayEmissivity_soft);
        fftwf_free(grids->bh_xray_histories_soft);
      }
    }

#if USE_MINI_HALOS
    free(grids->SMOOTHED_SFR_III);

    if (run_globals.params.Flag_IncludeLymanWerner) {
      free(grids->SMOOTHED_AGN_UV);
      fftwf_destroy_plan(grids->BHUVEmissivity_filtered_reverse_plan);
      fftwf_destroy_plan(grids->BHUVEmissivity_forward_plan);
      fftwf_free(grids->BHUVEmissivity_filtered);
      fftwf_free(grids->BHUVEmissivity_unfiltered);
      fftwf_free(grids->BHUVEmissivity);
      fftwf_free(grids->bh_uv_histories);
    }
#endif

    fftwf_free(grids->Tk_box);
    fftwf_free(grids->TS_box);
#if USE_MINI_HALOS
    fftwf_free(grids->Tk_boxII);
    fftwf_free(grids->TS_boxII);
#endif

    fftwf_destroy_plan(grids->x_e_filtered_reverse_plan);
    fftwf_destroy_plan(grids->x_e_box_forward_plan);
    fftwf_free(grids->x_e_filtered);
    fftwf_free(grids->x_e_unfiltered);
    fftwf_free(grids->x_e_box_prev);
    fftwf_free(grids->x_e_box);

    fftwf_destroy_plan(grids->sfr_filtered_reverse_plan);
    fftwf_destroy_plan(grids->sfr_forward_plan);
    fftwf_free(grids->sfr_filtered);
    fftwf_free(grids->sfr_unfiltered);
    fftwf_free(grids->sfr);
    fftwf_free(grids->sfr_histories);

#if USE_STOCHASTICITY
    fftwf_destroy_plan(grids->xray_luminosity_filtered_reverse_plan);
    fftwf_destroy_plan(grids->xray_luminosity_forward_plan);
    fftwf_free(grids->xray_luminosity_filtered);
    fftwf_free(grids->xray_luminosity_unfiltered);
    fftwf_free(grids->xray_luminosity);
    fftwf_free(grids->xray_luminosity_histories);
#endif

#if USE_MINI_HALOS
    fftwf_destroy_plan(grids->sfrIII_filtered_reverse_plan);
    fftwf_destroy_plan(grids->sfrIII_forward_plan);
    fftwf_free(grids->sfrIII_filtered);
    fftwf_free(grids->sfrIII_unfiltered);
    fftwf_free(grids->sfrIII);
    fftwf_free(grids->sfrIII_histories);
#endif
  }

#if USE_MINI_HALOS
  if (run_globals.params.Flag_IncludeLymanWerner) {
    free(grids->JLW_box);
    free(grids->JLW_boxII);
  }
#endif

  fftwf_free(grids->r_bubble);
  fftwf_free(grids->z_at_ionization);
  fftwf_free(grids->xH);
  fftwf_free(grids->temp_kinetic_all_gas);

  fftwf_destroy_plan(grids->weighted_sfr_filtered_reverse_plan);
  fftwf_destroy_plan(grids->weighted_sfr_forward_plan);
  fftwf_free(grids->weighted_sfr_filtered);
  fftwf_free(grids->weighted_sfr_unfiltered);
  fftwf_free(grids->weighted_sfr);

  fftwf_destroy_plan(grids->deltax_filtered_reverse_plan);
  fftwf_destroy_plan(grids->deltax_forward_plan);
  fftwf_free(grids->deltax_filtered);
  fftwf_free(grids->deltax_unfiltered);
  fftwf_free(grids->deltax);

  fftwf_destroy_plan(grids->stars_filtered_reverse_plan);
  fftwf_destroy_plan(grids->stars_forward_plan);
  fftwf_free(grids->stars_filtered);
  fftwf_free(grids->stars_unfiltered);
  fftwf_free(grids->stars);

  if (run_globals.params.physics.Flag_BHFeedback) {
    fftwf_destroy_plan(grids->effective_bhm_filtered_reverse_plan);
    fftwf_destroy_plan(grids->effective_bhm_forward_plan);
    fftwf_free(grids->effective_bhm_filtered);
    fftwf_free(grids->effective_bhm_unfiltered);
    fftwf_free(grids->effective_bhm);

    fftwf_destroy_plan(grids->effective_bhar_filtered_reverse_plan);
    fftwf_destroy_plan(grids->effective_bhar_forward_plan);
    fftwf_free(grids->effective_bhar_filtered);
    fftwf_free(grids->effective_bhar_unfiltered);
    fftwf_free(grids->effective_bhar);
  }

#if USE_MINI_HALOS
  fftwf_destroy_plan(grids->weighted_sfrIII_filtered_reverse_plan);
  fftwf_destroy_plan(grids->weighted_sfrIII_forward_plan);
  fftwf_free(grids->weighted_sfrIII_filtered);
  fftwf_free(grids->weighted_sfrIII_unfiltered);
  fftwf_free(grids->weighted_sfrIII);

  fftwf_destroy_plan(grids->starsIII_filtered_reverse_plan);
  fftwf_destroy_plan(grids->starsIII_forward_plan);
  fftwf_free(grids->starsIII_filtered);
  fftwf_free(grids->starsIII_unfiltered);
  fftwf_free(grids->starsIII);
#endif

  fftwf_free(grids->buffer);
  mlog(" ...done", MLOG_CLOSE);
}

int map_galaxies_to_slabs(int ngals)
{
  double box_size = run_globals.params.BoxSize;
  int ReionGridDim = run_globals.params.ReionGridDim;

  mlog("Mapping galaxies to slabs...", MLOG_OPEN);

  // Loop through each valid galaxy and find what slab it sits in
  if (ngals > 0)
    run_globals.reion_grids.galaxy_to_slab_map = malloc(sizeof(gal_to_slab_t) * ngals);
  else
    run_globals.reion_grids.galaxy_to_slab_map = NULL;

  gal_to_slab_t* galaxy_to_slab_map = run_globals.reion_grids.galaxy_to_slab_map;
  ptrdiff_t* slab_ix_start = run_globals.reion_grids.slab_ix_start;

  galaxy_t* gal = run_globals.FirstGal;
  int gal_counter = 0;
  while (gal != NULL) {
    // TODO: Note that I am including ghosts here.  We will need to check the
    // validity of this.  By definition, if they are ghosts then their host
    // halo hasn't been identified at this time step and hence they haven't
    // been evolved.  Their properties (Sfr, StellarMass, etc.) will all have
    // been set when they were last identified.
    if (gal->Type < 3) {
      // TODO: for type 2 galaxies these positions will be set from the last
      // time they were identified.  If their host halo has moved significantly
      // since then, these positions won't reflect that and the satellites will
      // be spatially disconnected from their hosts.  We will need to fix this
      // at some point.

      ptrdiff_t ix = pos_to_ngp(gal->Pos[0], box_size, ReionGridDim);

      assert((ix >= 0) && (ix < ReionGridDim));

      galaxy_to_slab_map[gal_counter].index = gal_counter;
      galaxy_to_slab_map[gal_counter].slab_ind =
        searchsorted(&ix, slab_ix_start, run_globals.mpi_size, sizeof(ptrdiff_t), compare_ptrdiff, -1, -1);
      galaxy_to_slab_map[gal_counter++].galaxy = gal;
    }

    gal = gal->Next;
  }

  // sort the slab indices IN PLACE (n.b. compare_slab_assign is a stable comparison)
  if (galaxy_to_slab_map != NULL)
    qsort(galaxy_to_slab_map, (size_t)gal_counter, sizeof(gal_to_slab_t), compare_slab_assign);

  assert(gal_counter == ngals);

  mlog("...done.", MLOG_CLOSE);

  return gal_counter;
}

void assign_Mvir_crit_to_galaxies(int ngals_in_slabs, int flag_feed)
{
  reion_grids_t* grids = &run_globals.reion_grids;
  gal_to_slab_t* map = grids->galaxy_to_slab_map;
  int cgm_mode = run_globals.params.physics.Flag_FescCGMSuppression;
  const float* source = NULL;
  switch (flag_feed) {
    case 1:
      source = grids->Mvir_crit;
      break;
#if USE_MINI_HALOS
    case 2:
      source = grids->Mvir_crit_MC;
      break;
#endif
    case 3:
      source = grids->t_resp;
      break;
    case 4:
      if (cgm_mode <= 0)
        return;
      source = cgm_mode == 3 ? grids->clumping_factor : grids->Gamma12;
      break;
    default:
      mlog_error("Unknown galaxy feedback grid %d.", flag_feed);
      ABORT(EXIT_FAILURE);
  }
  mlog("Assigning feedback grid %d to galaxies...", MLOG_OPEN, flag_feed);
  float* values = sample_grid_for_galaxies(source, map, ngals_in_slabs);
  for (int i = 0; i < ngals_in_slabs; ++i) {
    galaxy_t* gal = map[i].galaxy;
    double gamma12_local;
    if (flag_feed == 1)
      gal->MvirCrit = (double)values[i];
#if USE_MINI_HALOS
    if (flag_feed == 2)
      gal->MvirCrit_MC = (double)values[i];
#endif
    if (flag_feed == 3)
      gal->t_resp = (double)values[i];
    // Compute tau_cgm based on CGM suppression mode
    // Mode 1: instantaneous Gamma12, Mode 2: cumulative Gamma12, Mode 3: clumping factor
    if (flag_feed == 4) {
      physics_params_t* params = &(run_globals.params.physics);
      if (gal->HotGas > 0.0 && gal->Rvir > 0.0) {

        double grid_value = (double)values[i];
        double suppression_factor = 1.0; // Will be raised to FescCGMGamma12Scaling power

        switch (cgm_mode) {
          case 1:
            // Mode 1: Instantaneous Gamma12
            // Normalize to 1e-12 s^-1 and scale (reference: Gamma12 = 0.1)
            gamma12_local = grid_value * run_globals.params.Hubble_h * run_globals.params.Hubble_h;
            CLAMP_NEGATIVE(gamma12_local);
            suppression_factor = gamma12_local * 10.0; // Normalized at Gamma12 = 0.1
            break;

          case 2: {
            // Mode 2: Gamma12 * dt accumulated over the galaxy's history
            gamma12_local = grid_value * run_globals.params.Hubble_h * run_globals.params.Hubble_h;
            CLAMP_NEGATIVE(gamma12_local);
            double dt_myr = gal->dt * run_globals.units.UnitTime_in_s / SEC_PER_MEGAYEAR;
            gal->cumulative_ionization += gamma12_local * dt_myr;
            // Normalize at cumulative value of 1.0 (e.g., Gamma12=0.1 for 10 Myr)
            suppression_factor = gal->cumulative_ionization;
            break;
          }

          case 3:
            // Mode 3: Local clumping factor
            CLAMP_NEGATIVE(grid_value);
            suppression_factor = grid_value;
            break;

          default:
            suppression_factor = 1.0;
        }

        // Calculate optical depth from hot gas column density (HotGas/Rvir^2)
        // Normalized at 1e8 Msun / (10 kpc)^2 for gas, and suppression_factor = 1
        gal->tau_cgm = params->FescCGMSuppressionNorm *
                       pow(gal->HotGas * 1.0e2 / run_globals.params.Hubble_h, params->FescCGMSuppressionScaling) *
                       pow(0.01 * run_globals.params.Hubble_h / gal->Rvir, 2.0 * params->FescCGMSuppressionScaling) *
                       pow(suppression_factor, params->FescCGMGamma12Scaling);
      }
    }
  }
  free(values);
  mlog("...done.", MLOG_CLOSE);
}

#if USE_STOCHASTICITY
static double calculate_galaxy_xray_luminosity(const galaxy_t* source_view, double sfr_timescale)
{
  double source_sfr;

  source_sfr =
    run_globals.params.Flag_InstantaneousSFR ? source_view->Sfr : source_view->GrossStellarMass / sfr_timescale;

  source_sfr *= run_globals.units.UnitMass_in_g / SOLAR_MASS;
  source_sfr *= SEC_PER_YEAR / run_globals.units.UnitTime_in_s;

  return run_globals.params.physics.LXrayGal * source_sfr / XRAY_LUMINOSITY_UNIT;
}
#endif

static void cache_source_spectrum(float* history, fftwf_complex* spectrum, fftwf_plan plan)
{
  size_t count = (size_t)run_globals.reion_grids.slab_n_complex[run_globals.mpi_rank];
  float dim = run_globals.params.ReionGridDim;
  float n_cells = (float)((double)dim * dim * dim);
  fftwf_execute(plan);
  for (size_t i = 0; i < count; ++i)
    spectrum[i] /= n_cells;
  memcpy(history, spectrum, count * sizeof(fftwf_complex));
}

void construct_baryon_grids(int snapshot, int local_ngals)
{
  float* stellar_grid = run_globals.reion_grids.stars;
  float* effective_bhm_grid = run_globals.reion_grids.effective_bhm;
  float* effective_bhar_grid = run_globals.reion_grids.effective_bhar;
  float* sfr_grid = run_globals.reion_grids.sfr;
  float* sfr_histories_grid = run_globals.reion_grids.sfr_histories;
#if USE_STOCHASTICITY
  float* xray_luminosity_grid = run_globals.reion_grids.xray_luminosity;
  float* xray_luminosity_histories_grid = run_globals.reion_grids.xray_luminosity_histories;
  // Recalibrate the X-ray field only when source recalibration is enabled and either noSFR treatment or X-ray scatter
  // modifies the source luminosities.
  int recalibrate_xray_sources =
    run_globals.params.physics.Flag_SourceRecalibration &&
    (run_globals.params.physics.Flag_RemoveSFRScatter == 1 || run_globals.params.physics.XrayScatterDex > 0.0);
  int recalibrate_sfr_sources =
    run_globals.params.physics.Flag_SourceRecalibration && run_globals.params.physics.Flag_RemoveSFRScatter == 1;
#endif
  float* weighted_sfr_grid = run_globals.reion_grids.weighted_sfr;
  int ReionGridDim = run_globals.params.ReionGridDim;
  double sfr_timescale = run_globals.params.ReionSfrTimescale * hubble_time(snapshot);
#if USE_MINI_HALOS
  float* stellarIII_grid = run_globals.reion_grids.starsIII;
  float* sfrIII_grid = run_globals.reion_grids.sfrIII;
  float* sfrIII_histories_grid = run_globals.reion_grids.sfrIII_histories;
  float* weighted_sfrIII_grid = run_globals.reion_grids.weighted_sfrIII;
#endif

  float* bh_xray_grid_hard = run_globals.reion_grids.BHXrayEmissivity_hard;
  float* bh_xray_hist_grid_hard = run_globals.reion_grids.bh_xray_histories_hard;
  float* bh_xray_grid_soft = run_globals.reion_grids.BHXrayEmissivity_soft;
  float* bh_xray_hist_grid_soft = run_globals.reion_grids.bh_xray_histories_soft;
#if USE_MINI_HALOS
  float* bh_uv_grid = run_globals.reion_grids.BHUVEmissivity;
  float* bh_uv_hist_grid = run_globals.reion_grids.bh_uv_histories;
#endif

  gal_to_slab_t* galaxy_to_slab_map = run_globals.reion_grids.galaxy_to_slab_map;
  size_t local_n_complex = (size_t)run_globals.reion_grids.slab_n_complex[run_globals.mpi_rank];
  size_t stride = local_n_complex * 2;

#if USE_STOCHASTICITY
  if (run_globals.params.physics.Flag_RemoveSFRScatter == 1) {
    build_no_sfr_tables(2);
#if USE_MINI_HALOS
    build_no_sfr_tables(3);
#endif
    apply_no_sfr_treatment(snapshot);
  }

  if (run_globals.params.physics.Flag_SourceRecalibration) {
    if (run_globals.params.physics.Flag_RemoveSFRScatter == 1) {
      compute_no_sfr_recalibration_factors(2);
#if USE_MINI_HALOS
      compute_no_sfr_recalibration_factors(3);
#endif
    } else if (run_globals.params.physics.EscapeFracScatterDex > ABS_TOL) {
      compute_fesc_recalibration_factors();
    }
  }
#endif

  mlog("Constructing stellar mass and sfr grids...", MLOG_OPEN | MLOG_TIMERSTART);

  enum property
  {
    prop_stellar,
    prop_effective_bhm,
    prop_effective_bhar,
    prop_weighted_sfr,
#if USE_MINI_HALOS
    prop_stellarIII,
    prop_weighted_sfrIII,
    prop_sfrIII,
#endif
#if USE_STOCHASTICITY
    prop_xray_luminosity,
#endif
    prop_sfr,
    prop_bh_xray_emissivity_hard,
    prop_bh_xray_emissivity_soft,
#if USE_MINI_HALOS
    prop_bh_uv_emissivity,
#endif
    prop_count
  };

  bool agn_hard_needed =
    run_globals.params.physics.Flag_IncludeAGNXray == 1 || run_globals.params.physics.Flag_IncludeAGNXray == 2;
  bool agn_soft_needed =
    run_globals.params.physics.Flag_IncludeAGNXray == 1 || run_globals.params.physics.Flag_IncludeAGNXray == 3;
  float* destination[prop_count];
  int properties[prop_count], n_properties = 0;
#define ADD_SOURCE(prop, grid)                                                                                         \
  do {                                                                                                                 \
    properties[n_properties] = prop;                                                                                   \
    destination[n_properties++] = grid;                                                                                \
  } while (0)
  ADD_SOURCE(prop_stellar, stellar_grid);
  if (run_globals.params.physics.Flag_BHFeedback) {
    ADD_SOURCE(prop_effective_bhm, effective_bhm_grid);
    ADD_SOURCE(prop_effective_bhar, effective_bhar_grid);
  }
  ADD_SOURCE(prop_weighted_sfr, weighted_sfr_grid);
#if USE_MINI_HALOS
  ADD_SOURCE(prop_stellarIII, stellarIII_grid);
  ADD_SOURCE(prop_weighted_sfrIII, weighted_sfrIII_grid);
  if (run_globals.params.Flag_IncludeSpinTemp) {
    ADD_SOURCE(prop_sfrIII, sfrIII_grid);
  }
#endif
  if (run_globals.params.Flag_IncludeSpinTemp) {
#if USE_STOCHASTICITY
    ADD_SOURCE(prop_xray_luminosity, xray_luminosity_grid);
#endif
    ADD_SOURCE(prop_sfr, sfr_grid);
    if (agn_hard_needed) {
      ADD_SOURCE(prop_bh_xray_emissivity_hard, bh_xray_grid_hard);
    }
    if (agn_soft_needed) {
      ADD_SOURCE(prop_bh_xray_emissivity_soft, bh_xray_grid_soft);
    }
#if USE_MINI_HALOS
    if (run_globals.params.Flag_IncludeLymanWerner) {
      ADD_SOURCE(prop_bh_uv_emissivity, bh_uv_grid);
    }
#endif
  }
#undef ADD_SOURCE

  // The map was built before evolution; dead galaxies remain in it but not in local_ngals.
  gal_to_slab_t* active_map = cpu_checked_alloc(local_ngals, sizeof(*active_map));
  for (int i = 0, live = 0; live < local_ngals; ++i)
    if (galaxy_to_slab_map[i].galaxy->Type <= 2)
      active_map[live++] = galaxy_to_slab_map[i];
  cell_layout_t layout = cells_for_galaxies(active_map, local_ngals);
  // Each occupied cell carries all active source fields; align successive cell IDs.
  size_t record_size = (sizeof(uint64_t) + n_properties * sizeof(float) + 7) & ~(size_t)7;
  char* packets = cpu_checked_alloc(layout.count, record_size);
  for (size_t i = 0; i < layout.count; ++i)
    memcpy(packets + i * record_size, &layout.cell[i], sizeof(uint64_t));
  long N_BlackHoleMassLimitReion = 0;
#if USE_STOCHASTICITY
  double local_xray_raw = 0.0, local_xray_target = 0.0;
  double local_sfr_raw = 0.0, local_sfr_target = 0.0;
#endif
  for (int field = 0; field < n_properties; ++field) {
    int prop = properties[field];
    for (int i = 0; i < local_ngals; ++i) {
      galaxy_t* gal = active_map[i].galaxy;
      float* buffer = (float*)(packets + layout.slot[i] * record_size + sizeof(uint64_t)) + field;
      const int ind = 0;
      double stochasticity_calibration_factor = 1.0;
      switch (prop) {
        case prop_stellar:
#if USE_STOCHASTICITY
          if (run_globals.params.physics.EscapeFracScatterDex > ABS_TOL ||
              run_globals.params.physics.Flag_RemoveSFRScatter == 1) {
            if (run_globals.params.physics.Flag_SourceRecalibration)
              stochasticity_calibration_factor = extract_recalibration_factors(gal, 2, true);
            buffer[ind] += gal->StochasticityTreatedFescWeightedGSM * stochasticity_calibration_factor;
          } else
            buffer[ind] += gal->FescWeightedGSM;
#else
          buffer[ind] += gal->FescWeightedGSM;
#endif
          break;

        case prop_effective_bhm:
          if (gal->BlackHoleMass >= run_globals.params.physics.BlackHoleMassLimitReion) {
            buffer[ind] += gal->EffectiveBHM;
          } else {
            N_BlackHoleMassLimitReion++;
          }
          break;

#if USE_MINI_HALOS
        case prop_stellarIII:
#if USE_STOCHASTICITY
          if (run_globals.params.physics.EscapeFracScatterDex > ABS_TOL ||
              run_globals.params.physics.Flag_RemoveSFRScatter == 1) {
            if (run_globals.params.physics.Flag_SourceRecalibration)
              stochasticity_calibration_factor = extract_recalibration_factors(gal, 3, true);
            buffer[ind] += gal->StochasticityTreatedFescIIIWeightedGSM * stochasticity_calibration_factor;
          } else
            buffer[ind] += gal->FescIIIWeightedGSM;
#else
          buffer[ind] += gal->FescIIIWeightedGSM;
#endif
          break;

        case prop_weighted_sfrIII:
#if USE_STOCHASTICITY
          if (run_globals.params.physics.EscapeFracScatterDex > ABS_TOL ||
              run_globals.params.physics.Flag_RemoveSFRScatter == 1) {
            if (run_globals.params.physics.Flag_SourceRecalibration)
              stochasticity_calibration_factor = extract_recalibration_factors(gal, 3, false);
            buffer[ind] += gal->StochasticityTreatedFescIIIWeightedSfr * stochasticity_calibration_factor;
          } else
            buffer[ind] += gal->FescIIIWeightedSfr;
#else
          buffer[ind] += gal->FescIIIWeightedSfr;
#endif
          break;

        case prop_sfrIII:
#if USE_STOCHASTICITY
          buffer[ind] +=
            run_globals.params.Flag_InstantaneousSFR
              ? (run_globals.params.physics.Flag_RemoveSFRScatter == 1 ? gal->SfrIIINoScatter : gal->SfrIII)
              : (run_globals.params.physics.Flag_RemoveSFRScatter == 1 ? gal->GrossStellarMassIIINoScatter
                                                                       : gal->GrossStellarMassIII);
#else
          buffer[ind] += run_globals.params.Flag_InstantaneousSFR ? gal->SfrIII : gal->GrossStellarMassIII;
#endif
          break;
#endif

        case prop_weighted_sfr:
#if USE_STOCHASTICITY
          if (run_globals.params.physics.EscapeFracScatterDex > ABS_TOL ||
              run_globals.params.physics.Flag_RemoveSFRScatter == 1) {
            if (run_globals.params.physics.Flag_SourceRecalibration)
              stochasticity_calibration_factor = extract_recalibration_factors(gal, 2, false);
            buffer[ind] += gal->StochasticityTreatedFescWeightedSfr * stochasticity_calibration_factor;
          } else
            buffer[ind] += gal->FescWeightedSfr;
#else
          buffer[ind] += gal->FescWeightedSfr;
#endif
          break;

        case prop_effective_bhar:
          if (gal->BlackHoleMass >= run_globals.params.physics.BlackHoleMassLimitReion) {
            buffer[ind] += gal->EffectiveBHAR;
          }
          break;

#if USE_STOCHASTICITY
        case prop_xray_luminosity: {
          double xray_target = calculate_galaxy_xray_luminosity(gal, sfr_timescale);
          double xray_raw = xray_target;

          if (run_globals.params.physics.Flag_RemoveSFRScatter == 1) {
            galaxy_t source_view = *gal;
            source_view.GrossStellarMass = source_view.GrossStellarMassNoScatter;
            source_view.Sfr = source_view.SfrNoScatter;

            xray_raw = calculate_galaxy_xray_luminosity(&source_view, sfr_timescale);
          }

          if (run_globals.params.physics.XrayScatterDex > 0.0)
            xray_raw = apply_lognormal_scatter(xray_raw, run_globals.params.physics.XrayScatterDex);

          if (recalibrate_xray_sources) {
            local_xray_raw += xray_raw;
            local_xray_target += xray_target;
          }

          buffer[ind] += (float)xray_raw;
          break;
        }
#endif

        case prop_sfr: {
#if USE_STOCHASTICITY
          double sfr_target = run_globals.params.Flag_InstantaneousSFR ? gal->Sfr : gal->GrossStellarMass;
          double sfr_raw = sfr_target;

          if (run_globals.params.physics.Flag_RemoveSFRScatter == 1)
            sfr_raw = run_globals.params.Flag_InstantaneousSFR ? gal->SfrNoScatter : gal->GrossStellarMassNoScatter;

          if (recalibrate_sfr_sources) {
            local_sfr_raw += sfr_raw;
            local_sfr_target += sfr_target;
          }

          buffer[ind] += sfr_raw;
#else
          buffer[ind] += run_globals.params.Flag_InstantaneousSFR ? gal->Sfr : gal->GrossStellarMass;
#endif
          break;
        }

        /*
         *   We accumulate each per cell exactly as gal->Sfr is accumulated for prop_sfr. The owner-directed exchange
         * below sums contributions from all galaxies on all ranks. The BlackHoleMassLimitReion guard is applied
         * consistency with prop_effective_bhar.
         */
        case prop_bh_xray_emissivity_hard:
          if (gal->BlackHoleMass >= run_globals.params.physics.BlackHoleMassLimitReion)
            buffer[ind] += gal->BHXrayEmissivity_hard;
          break;

        case prop_bh_xray_emissivity_soft:
          if (gal->BlackHoleMass >= run_globals.params.physics.BlackHoleMassLimitReion)
            buffer[ind] += gal->BHXrayEmissivity_soft;
          break;

#if USE_MINI_HALOS
        case prop_bh_uv_emissivity:
          if (gal->BlackHoleMass >= run_globals.params.physics.BlackHoleMassLimitReion)
            buffer[ind] += gal->QuasarLuv;
          break;
#endif

        default:
          mlog_error("Unrecognised property in slab creation.");
          ABORT(EXIT_FAILURE);
          break;
      }
    }
  }
  free(active_map);
  size_t *counts, *offsets;
  char* received = exchange_records(packets, layout.send_count, record_size, &counts, &offsets, run_globals.mpi_comm);
  free(packets);
  free_cell_layout(&layout);
  for (int field = 0; field < n_properties; ++field)
    memset(destination[field], 0, stride * sizeof(float));
  const uint64_t local_cells =
    (uint64_t)run_globals.reion_grids.slab_nix[run_globals.mpi_rank] * ReionGridDim * ReionGridDim;
  const size_t padded_z = 2 * (ReionGridDim / 2 + 1);
  // exchange_records groups by sender, so accumulation has a stable rank order.
  for (size_t i = 0; i < offsets[run_globals.mpi_size]; ++i) {
    uint64_t cell;
    memcpy(&cell, received + i * record_size, sizeof(cell));
    if (cell >= local_cells)
      ABORT(EXIT_FAILURE);
    size_t index = cell % ReionGridDim + padded_z * (cell / ReionGridDim);
    const float* values = (const float*)(received + i * record_size + sizeof(cell));
    for (int field = 0; field < n_properties; ++field)
      destination[field][index] += values[field];
  }
  free(received);
  free(counts);
  free(offsets);
  for (int field = 0; field < n_properties; ++field) {
    int prop = properties[field];
    int convert_to_sfr = prop == prop_sfr;
#if USE_MINI_HALOS
    convert_to_sfr |= prop == prop_sfrIII;
#endif
    for (size_t i = 0; i < stride; ++i) {
      double val = destination[field][i];
#if USE_STOCHASTICITY
      if (prop != prop_xray_luminosity)
#endif
        CLAMP_NEGATIVE(val);
      if (convert_to_sfr && !run_globals.params.Flag_InstantaneousSFR)
        val /= sfr_timescale;
      destination[field][i] = (float)val;
    }
  }
#if USE_STOCHASTICITY
  if (run_globals.params.Flag_IncludeSpinTemp && recalibrate_xray_sources) {
    double factor = compute_xray_recalibration_factor(local_xray_raw, local_xray_target);
    for (size_t i = 0; i < stride; ++i)
      xray_luminosity_grid[i] *= (float)factor;
  }
  if (run_globals.params.Flag_IncludeSpinTemp && recalibrate_sfr_sources) {
    double factor = compute_xray_recalibration_factor(local_sfr_raw, local_sfr_target);
    for (size_t i = 0; i < stride; ++i)
      sfr_grid[i] = (float)(sfr_grid[i] * factor);
  }
#endif
  MPI_Allreduce(MPI_IN_PLACE, &N_BlackHoleMassLimitReion, 1, MPI_LONG, MPI_SUM, run_globals.mpi_comm);
  if (run_globals.params.physics.Flag_BHFeedback)
    mlog("%ld quasars are smaller than %g",
         MLOG_MESG,
         N_BlackHoleMassLimitReion,
         run_globals.params.physics.BlackHoleMassLimitReion);

  if (run_globals.params.Flag_IncludeSpinTemp) {
    reion_grids_t* grids = &run_globals.reion_grids;
    const int n_history = run_globals.NstoreSnapshots_Heating;
    if (n_history <= 0)
      ABORT(EXIT_FAILURE);
    grids->heating_history_head = (grids->heating_history_head + n_history - 1) % n_history;
    size_t offset = (size_t)grids->heating_history_head * stride;
    // FFT each new source snapshot once. History storage holds normalized spectra,
    // which can be linearly combined before filtering each light-cone shell.
    cache_source_spectrum(sfr_histories_grid + offset, grids->sfr_unfiltered, grids->sfr_forward_plan);
#if USE_STOCHASTICITY
    cache_source_spectrum(
      xray_luminosity_histories_grid + offset, grids->xray_luminosity_unfiltered, grids->xray_luminosity_forward_plan);
#endif
#if USE_MINI_HALOS
    cache_source_spectrum(sfrIII_histories_grid + offset, grids->sfrIII_unfiltered, grids->sfrIII_forward_plan);
#endif
    if (agn_hard_needed)
      cache_source_spectrum(bh_xray_hist_grid_hard + offset,
                            grids->BHXrayEmissivity_hard_unfiltered,
                            grids->BHXrayEmissivity_hard_forward_plan);
    if (agn_soft_needed)
      cache_source_spectrum(bh_xray_hist_grid_soft + offset,
                            grids->BHXrayEmissivity_soft_unfiltered,
                            grids->BHXrayEmissivity_soft_forward_plan);
#if USE_MINI_HALOS
    if (run_globals.params.Flag_IncludeLymanWerner)
      cache_source_spectrum(
        bh_uv_hist_grid + offset, grids->BHUVEmissivity_unfiltered, grids->BHUVEmissivity_forward_plan);
#endif
  }
  mlog("done", MLOG_CLOSE | MLOG_TIMERSTOP);
}

void write_grid_float(const char* name, float* data, hid_t file_id, hid_t fspace_id, hid_t memspace_id, hid_t dcpl_id)
{
  // create the dataset
  hid_t dset_id = H5Dcreate(file_id, name, H5T_NATIVE_FLOAT, fspace_id, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);

  // create the property list
  hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);

  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

  // write the dataset
  H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace_id, fspace_id, plist_id, data);

  // cleanup
  H5Pclose(plist_id);
  H5Dclose(dset_id);
}

void gen_grids_fname(const int snapshot, char* name, const bool relative)
{
  if (!relative)
    sprintf(name, "%s/%s_grids_%d.hdf5", run_globals.params.OutputDir, run_globals.params.FileNameGalaxies, snapshot);
  else
    sprintf(name, "%s_grids_%d.hdf5", run_globals.params.FileNameGalaxies, snapshot);
}

static hid_t create_reion_grid(const int snapshot, const bool parallel)
{
  char name[STRLEN];
  gen_grids_fname(snapshot, name, false);

  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  if (parallel)
    H5Pset_fapl_mpio(plist_id, run_globals.mpi_comm, MPI_INFO_NULL);

  hid_t file_id = H5Fcreate(name, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
  H5Pclose(plist_id);

  return file_id;
}

void save_reion_input_grids(int snapshot)
{
  reion_grids_t* grids = &(run_globals.reion_grids);
  int ReionGridDim = run_globals.params.ReionGridDim;
  int local_nix = (int)(run_globals.reion_grids.slab_nix[run_globals.mpi_rank]);
  double UnitTime_in_s = run_globals.units.UnitTime_in_s;
  double UnitMass_in_g = run_globals.units.UnitMass_in_g;

  mlog("Saving tocf input grids...", MLOG_OPEN);
  hid_t file_id = create_reion_grid(snapshot, true);

  // create the filespace
  hsize_t dims[3] = { (hsize_t)ReionGridDim, (hsize_t)ReionGridDim, (hsize_t)ReionGridDim };
  hid_t fspace_id = H5Screate_simple(3, dims, NULL);

  // create the memspace
  hsize_t mem_dims[3] = { (hsize_t)local_nix, (hsize_t)ReionGridDim, (hsize_t)ReionGridDim };
  hid_t memspace_id = H5Screate_simple(3, mem_dims, NULL);

  // select a hyperslab in the filespace
  hsize_t start[3] = { (hsize_t)run_globals.reion_grids.slab_ix_start[run_globals.mpi_rank], 0, 0 };
  hsize_t count[3] = { (hsize_t)local_nix, (hsize_t)ReionGridDim, (hsize_t)ReionGridDim };
  H5Sselect_hyperslab(fspace_id, H5S_SELECT_SET, start, NULL, count, NULL);

  // set the dataset creation property list to use chunking along x-axis
  hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
  H5Pset_chunk(dcpl_id, 3, (hsize_t[3]){ 1, (hsize_t)ReionGridDim, (hsize_t)ReionGridDim });

  // fftw padded grids
  float* grid = (float*)calloc((size_t)local_nix * (size_t)ReionGridDim * (size_t)ReionGridDim, sizeof(float));

  if (run_globals.params.Flag_IncludeSpinTemp) {
    for (int ii = 0; ii < local_nix; ii++)
      for (int jj = 0; jj < ReionGridDim; jj++)
        for (int kk = 0; kk < ReionGridDim; kk++)
          grid[grid_index(ii, jj, kk, ReionGridDim, INDEX_REAL)] =
            (float)((grids->sfr)[grid_index(ii, jj, kk, ReionGridDim, INDEX_PADDED)] * UnitMass_in_g / UnitTime_in_s *
                    SEC_PER_YEAR / SOLAR_MASS);
    write_grid_float("sfr", grid, file_id, fspace_id, memspace_id, dcpl_id);
#if USE_MINI_HALOS
    for (int ii = 0; ii < local_nix; ii++)
      for (int jj = 0; jj < ReionGridDim; jj++)
        for (int kk = 0; kk < ReionGridDim; kk++)
          grid[grid_index(ii, jj, kk, ReionGridDim, INDEX_REAL)] =
            (float)((grids->sfrIII)[grid_index(ii, jj, kk, ReionGridDim, INDEX_PADDED)] * UnitMass_in_g /
                    UnitTime_in_s * SEC_PER_YEAR / SOLAR_MASS);
    write_grid_float("sfrIII", grid, file_id, fspace_id, memspace_id, dcpl_id);
#endif
  }

  for (int ii = 0; ii < local_nix; ii++)
    for (int jj = 0; jj < ReionGridDim; jj++)
      for (int kk = 0; kk < ReionGridDim; kk++)
        grid[grid_index(ii, jj, kk, ReionGridDim, INDEX_REAL)] =
          (grids->deltax)[grid_index(ii, jj, kk, ReionGridDim, INDEX_PADDED)];
  write_grid_float("deltax", grid, file_id, fspace_id, memspace_id, dcpl_id);

  for (int ii = 0; ii < local_nix; ii++)
    for (int jj = 0; jj < ReionGridDim; jj++)
      for (int kk = 0; kk < ReionGridDim; kk++)
        grid[grid_index(ii, jj, kk, ReionGridDim, INDEX_REAL)] =
          (grids->stars)[grid_index(ii, jj, kk, ReionGridDim, INDEX_PADDED)];
  write_grid_float("stars", grid, file_id, fspace_id, memspace_id, dcpl_id);

  if (run_globals.params.physics.Flag_BHFeedback) {
    for (int ii = 0; ii < local_nix; ii++)
      for (int jj = 0; jj < ReionGridDim; jj++)
        for (int kk = 0; kk < ReionGridDim; kk++)
          grid[grid_index(ii, jj, kk, ReionGridDim, INDEX_REAL)] =
            (grids->effective_bhm)[grid_index(ii, jj, kk, ReionGridDim, INDEX_PADDED)];
    write_grid_float("effective_bhm", grid, file_id, fspace_id, memspace_id, dcpl_id);

    for (int ii = 0; ii < local_nix; ii++)
      for (int jj = 0; jj < ReionGridDim; jj++)
        for (int kk = 0; kk < ReionGridDim; kk++)
          grid[grid_index(ii, jj, kk, ReionGridDim, INDEX_REAL)] =
            (float)((grids->effective_bhar)[grid_index(ii, jj, kk, ReionGridDim, INDEX_PADDED)] * UnitMass_in_g /
                    UnitTime_in_s * SEC_PER_YEAR / SOLAR_MASS);
    write_grid_float("effective_bhar", grid, file_id, fspace_id, memspace_id, dcpl_id);
  }

  for (int ii = 0; ii < local_nix; ii++)
    for (int jj = 0; jj < ReionGridDim; jj++)
      for (int kk = 0; kk < ReionGridDim; kk++)
        grid[grid_index(ii, jj, kk, ReionGridDim, INDEX_REAL)] =
          (float)((grids->weighted_sfr)[grid_index(ii, jj, kk, ReionGridDim, INDEX_PADDED)] * UnitMass_in_g /
                  UnitTime_in_s * SEC_PER_YEAR / SOLAR_MASS);
  write_grid_float("weighted_sfr", grid, file_id, fspace_id, memspace_id, dcpl_id);

#if USE_MINI_HALOS
  for (int ii = 0; ii < local_nix; ii++)
    for (int jj = 0; jj < ReionGridDim; jj++)
      for (int kk = 0; kk < ReionGridDim; kk++)
        grid[grid_index(ii, jj, kk, ReionGridDim, INDEX_REAL)] =
          (grids->starsIII)[grid_index(ii, jj, kk, ReionGridDim, INDEX_PADDED)];
  write_grid_float("starsIII", grid, file_id, fspace_id, memspace_id, dcpl_id);

  for (int ii = 0; ii < local_nix; ii++)
    for (int jj = 0; jj < ReionGridDim; jj++)
      for (int kk = 0; kk < ReionGridDim; kk++)
        grid[grid_index(ii, jj, kk, ReionGridDim, INDEX_REAL)] =
          (float)((grids->weighted_sfrIII)[grid_index(ii, jj, kk, ReionGridDim, INDEX_PADDED)] * UnitMass_in_g /
                  UnitTime_in_s * SEC_PER_YEAR / SOLAR_MASS);
  write_grid_float("weighted_sfrIII", grid, file_id, fspace_id, memspace_id, dcpl_id);
#endif

  // tidy up
  free(grid);
  H5Pclose(dcpl_id);
  H5Sclose(memspace_id);
  H5Sclose(fspace_id);
  H5Fclose(file_id);

  mlog("...done", MLOG_CLOSE);
}

static void load_source_spectrum(fftwf_complex* output, const float* history, int age, float weight, int replace)
{
  reion_grids_t* grids = &run_globals.reion_grids;
  int n_history = run_globals.NstoreSnapshots_Heating;
  if (age < 0 || age >= n_history) {
    mlog_error("Heating history age %d is outside the %d stored snapshots.", age, n_history);
    ABORT(EXIT_FAILURE);
  }
  size_t stride = (size_t)grids->slab_n_complex[run_globals.mpi_rank] * 2;
  size_t offset = (size_t)((grids->heating_history_head + age) % n_history) * stride;
  float* dest = (float*)output;
  for (size_t i = 0; i < stride; ++i) {
    float value = history[offset + i] * weight;
    if (replace)
      dest[i] = value;
    else
      dest[i] += value;
  }
}

void load_reion_sfr_grids(int snapshot_counter_backwards, float weight, const int new_load)
{
  reion_grids_t* grids = &run_globals.reion_grids;
  load_source_spectrum(grids->sfr_unfiltered, grids->sfr_histories, snapshot_counter_backwards, weight, new_load);
#if USE_STOCHASTICITY
  load_source_spectrum(
    grids->xray_luminosity_unfiltered, grids->xray_luminosity_histories, snapshot_counter_backwards, weight, new_load);
#endif
#if USE_MINI_HALOS
  load_source_spectrum(grids->sfrIII_unfiltered, grids->sfrIII_histories, snapshot_counter_backwards, weight, new_load);
#endif
}

void load_reion_bh_grids(int snapshot_counter_backwards, float weight, const int new_load)
{
  reion_grids_t* grids = &run_globals.reion_grids;
  int mode = run_globals.params.physics.Flag_IncludeAGNXray;
  if (mode == 1 || mode == 2)
    load_source_spectrum(grids->BHXrayEmissivity_hard_unfiltered,
                         grids->bh_xray_histories_hard,
                         snapshot_counter_backwards,
                         weight,
                         new_load);
  if (mode == 1 || mode == 3)
    load_source_spectrum(grids->BHXrayEmissivity_soft_unfiltered,
                         grids->bh_xray_histories_soft,
                         snapshot_counter_backwards,
                         weight,
                         new_load);
#if USE_MINI_HALOS
  if (run_globals.params.Flag_IncludeLymanWerner)
    load_source_spectrum(
      grids->BHUVEmissivity_unfiltered, grids->bh_uv_histories, snapshot_counter_backwards, weight, new_load);
#endif
}

void save_reion_output_grids(int snapshot)
{

  reion_grids_t* grids = &(run_globals.reion_grids);
  int ReionGridDim = run_globals.params.ReionGridDim;
  int local_nix = (int)(run_globals.reion_grids.slab_nix[run_globals.mpi_rank]);
  // fftw padded grids
  float* grid = (float*)calloc((size_t)local_nix * (size_t)ReionGridDim * (size_t)ReionGridDim, sizeof(float));

  mlog("Saving tocf output grids...", MLOG_OPEN);

  char name[STRLEN];
  gen_grids_fname(snapshot, name, false);

  // open the file (in parallel)
  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, run_globals.mpi_comm, MPI_INFO_NULL);
  hid_t file_id = H5Fopen(name, H5F_ACC_RDWR, plist_id);
  H5Pclose(plist_id);

  // create the filespace
  hsize_t dims[3] = { (hsize_t)ReionGridDim, (hsize_t)ReionGridDim, (hsize_t)ReionGridDim };
  hid_t fspace_id = H5Screate_simple(3, dims, NULL);

  // create the memspace
  hsize_t mem_dims[3] = { (hsize_t)local_nix, (hsize_t)ReionGridDim, (hsize_t)ReionGridDim };
  hid_t memspace_id = H5Screate_simple(3, mem_dims, NULL);

  // select a hyperslab in the filespace
  hsize_t start[3] = { (hsize_t)run_globals.reion_grids.slab_ix_start[run_globals.mpi_rank], 0, 0 };
  hsize_t count[3] = { (hsize_t)local_nix, (hsize_t)ReionGridDim, (hsize_t)ReionGridDim };
  H5Sselect_hyperslab(fspace_id, H5S_SELECT_SET, start, NULL, count, NULL);

  // set the dataset creation property list to use chunking along x-axis
  hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
  H5Pset_chunk(dcpl_id, 3, (hsize_t[3]){ 1, (hsize_t)ReionGridDim, (hsize_t)ReionGridDim });

  // create and write the datasets
  write_grid_float("xH", grids->xH, file_id, fspace_id, memspace_id, dcpl_id);
  write_grid_float("r_bubble", grids->r_bubble, file_id, fspace_id, memspace_id, dcpl_id);
  write_grid_float("temp_kinetic_all_gas", grids->temp_kinetic_all_gas, file_id, fspace_id, memspace_id, dcpl_id);
  if (run_globals.params.Flag_IncludeRecombinations) {
    write_grid_float("z_at_ionization", grids->z_at_ionization, file_id, fspace_id, memspace_id, dcpl_id);
    write_grid_float("residual_xH", grids->residual_xH, file_id, fspace_id, memspace_id, dcpl_id);
    write_grid_float("clumping_factor", grids->clumping_factor, file_id, fspace_id, memspace_id, dcpl_id);
    write_grid_float("Gamma12", grids->Gamma12, file_id, fspace_id, memspace_id, dcpl_id);
    write_grid_float("t_resp", grids->t_resp, file_id, fspace_id, memspace_id, dcpl_id);

    for (int ii = 0; ii < local_nix; ii++)
      for (int jj = 0; jj < ReionGridDim; jj++)
        for (int kk = 0; kk < ReionGridDim; kk++)
          grid[grid_index(ii, jj, kk, ReionGridDim, INDEX_REAL)] =
            (grids->N_rec)[grid_index(ii, jj, kk, ReionGridDim, INDEX_PADDED)];
    write_grid_float("N_rec", grid, file_id, fspace_id, memspace_id, dcpl_id);
  }

  if (run_globals.params.ReionUVBFlag) {
    write_grid_float("J_21_at_ionization", grids->J_21_at_ionization, file_id, fspace_id, memspace_id, dcpl_id);
    write_grid_float("Mvir_crit", grids->Mvir_crit, file_id, fspace_id, memspace_id, dcpl_id);

#if USE_MINI_HALOS
    if (run_globals.params.Flag_IncludeLymanWerner)
      write_grid_float("Mvir_crit_MC", grids->Mvir_crit_MC, file_id, fspace_id, memspace_id, dcpl_id);
#endif
  }

#if USE_MINI_HALOS
  if (run_globals.params.Flag_IncludeLymanWerner) {
    write_grid_float("JLW_box", grids->JLW_box, file_id, fspace_id, memspace_id, dcpl_id);
    write_grid_float("JLW_boxII", grids->JLW_boxII, file_id, fspace_id, memspace_id, dcpl_id);
  }
#endif

  if (run_globals.params.Flag_IncludeSpinTemp) {
    write_grid_float("TS_box", grids->TS_box, file_id, fspace_id, memspace_id, dcpl_id);
    write_grid_float("Tk_box", grids->Tk_box, file_id, fspace_id, memspace_id, dcpl_id);
#if USE_MINI_HALOS
    write_grid_float("TS_boxII", grids->TS_boxII, file_id, fspace_id, memspace_id, dcpl_id);
    write_grid_float("Tk_boxII", grids->Tk_boxII, file_id, fspace_id, memspace_id, dcpl_id);
#endif

    for (int ii = 0; ii < local_nix; ii++)
      for (int jj = 0; jj < ReionGridDim; jj++)
        for (int kk = 0; kk < ReionGridDim; kk++)
          grid[grid_index(ii, jj, kk, ReionGridDim, INDEX_REAL)] =
            (grids->x_e_box_prev)[grid_index(ii, jj, kk, ReionGridDim, INDEX_PADDED)];
    write_grid_float("x_e_box", grid, file_id, fspace_id, memspace_id, dcpl_id);
  }

  if (run_globals.params.Flag_Compute21cmBrightTemp) {
    write_grid_float("delta_T", grids->delta_T, file_id, fspace_id, memspace_id, dcpl_id);
#if USE_MINI_HALOS
    write_grid_float("delta_TII", grids->delta_TII, file_id, fspace_id, memspace_id, dcpl_id);
#endif
  }

  if (run_globals.params.Flag_ConstructLightcone && run_globals.params.EndSnapshotLightcone == snapshot &&
      snapshot != 0) {

    // create the filespace
    hsize_t dims_LC[3] = { (hsize_t)ReionGridDim, (hsize_t)ReionGridDim, (hsize_t)run_globals.params.LightconeLength };
    hid_t fspace_id_LC = H5Screate_simple(3, dims_LC, NULL);

    // create the memspace
    hsize_t mem_dims_LC[3] = { (hsize_t)local_nix, (hsize_t)ReionGridDim, (hsize_t)run_globals.params.LightconeLength };
    hid_t memspace_id_LC = H5Screate_simple(3, mem_dims_LC, NULL);

    // select a hyperslab in the filespace
    hsize_t start_LC[3] = { (hsize_t)run_globals.reion_grids.slab_ix_start[run_globals.mpi_rank], 0, 0 };
    hsize_t count_LC[3] = { (hsize_t)local_nix, (hsize_t)ReionGridDim, (hsize_t)run_globals.params.LightconeLength };
    H5Sselect_hyperslab(fspace_id_LC, H5S_SELECT_SET, start_LC, NULL, count_LC, NULL);

    // set the dataset creation property list to use chunking along x-axis
    hid_t dcpl_id_LC = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl_id_LC, 3, (hsize_t[3]){ 1, (hsize_t)ReionGridDim, (hsize_t)run_globals.params.LightconeLength });

    mlog("Outputting light-cone", MLOG_MESG);
    write_grid_float("LightconeBox", grids->LightconeBox, file_id, fspace_id_LC, memspace_id_LC, dcpl_id_LC);

    // create the filespace
    hsize_t dims_LCz[1] = { (hsize_t)run_globals.params.LightconeLength };
    hid_t fspace_id_LCz = H5Screate_simple(1, dims_LCz, NULL);

    // create the memspace
    hsize_t mem_dims_LCz[1] = { (hsize_t)run_globals.params.LightconeLength };
    hid_t memspace_id_LCz = H5Screate_simple(1, mem_dims_LCz, NULL);

    hid_t dcpl_id_LCz = H5Pcreate(H5P_DATASET_CREATE);
    hid_t dset_id =
      H5Dcreate(file_id, "lightcone-z", H5T_NATIVE_FLOAT, fspace_id_LCz, H5P_DEFAULT, dcpl_id_LCz, H5P_DEFAULT);

    plist_id = H5Pcreate(H5P_DATASET_XFER);

    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    // write the dataset
    H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace_id_LCz, fspace_id_LCz, plist_id, grids->Lightcone_redshifts);

    // cleanup
    H5Pclose(plist_id);
    H5Dclose(dset_id);
  }

  if (run_globals.params.Flag_ComputePS) {

    // create the filespace
    hsize_t dims_PS[1] = { (hsize_t)run_globals.params.PS_Length };
    hid_t fspace_id_PS = H5Screate_simple(1, dims_PS, NULL);

    // create the memspace
    hsize_t mem_dims_PS[1] = { (hsize_t)run_globals.params.PS_Length };
    hid_t memspace_id_PS = H5Screate_simple(1, mem_dims_PS, NULL);

    hid_t dcpl_id_PS = H5Pcreate(H5P_DATASET_CREATE);
    hid_t dset_id = H5Dcreate(file_id, "k_bins", H5T_NATIVE_FLOAT, fspace_id_PS, H5P_DEFAULT, dcpl_id_PS, H5P_DEFAULT);

    plist_id = H5Pcreate(H5P_DATASET_XFER);

    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    // write the dataset
    H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace_id_PS, fspace_id_PS, plist_id, grids->PS_k);

    // cleanup
    H5Pclose(plist_id);
    H5Dclose(dset_id);

    dset_id = H5Dcreate(file_id, "PS_data", H5T_NATIVE_FLOAT, fspace_id_PS, H5P_DEFAULT, dcpl_id_PS, H5P_DEFAULT);

    plist_id = H5Pcreate(H5P_DATASET_XFER);

    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace_id_PS, fspace_id_PS, plist_id, grids->PS_data);

    // cleanup
    H5Pclose(plist_id);
    H5Dclose(dset_id);

#if USE_MINI_HALOS
    dset_id = H5Dcreate(file_id, "PSII_data", H5T_NATIVE_FLOAT, fspace_id_PS, H5P_DEFAULT, dcpl_id_PS, H5P_DEFAULT);

    plist_id = H5Pcreate(H5P_DATASET_XFER);

    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace_id_PS, fspace_id_PS, plist_id, grids->PSII_data);

    // cleanup
    H5Pclose(plist_id);
    H5Dclose(dset_id);
#endif

    dset_id = H5Dcreate(file_id, "PS_error", H5T_NATIVE_FLOAT, fspace_id_PS, H5P_DEFAULT, dcpl_id_PS, H5P_DEFAULT);

    plist_id = H5Pcreate(H5P_DATASET_XFER);

    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace_id_PS, fspace_id_PS, plist_id, grids->PS_error);

    // cleanup
    H5Pclose(plist_id);
    H5Dclose(dset_id);

#if USE_MINI_HALOS
    dset_id = H5Dcreate(file_id, "PSII_error", H5T_NATIVE_FLOAT, fspace_id_PS, H5P_DEFAULT, dcpl_id_PS, H5P_DEFAULT);

    plist_id = H5Pcreate(H5P_DATASET_XFER);

    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace_id_PS, fspace_id_PS, plist_id, grids->PSII_error);

    // cleanup
    H5Pclose(plist_id);
    H5Dclose(dset_id);
#endif
  }

#if USE_MINI_HALOS
  // Flag_IncludeSpinTemp must match init.c's allocation guard, or these pointers are NULL.
  if (run_globals.params.Flag_IncludeSpinTemp && run_globals.params.Flag_IncludeLymanWerner) {
    // All ranks hold identical copies, so rank 0 writes and the others select none (avoids an overlapping-write race).
    hsize_t dims_LW[1] = { (hsize_t)run_globals.params.TsNumFilterSteps };
    hsize_t dims_LWspec[2] = { (hsize_t)run_globals.params.TsNumFilterSteps, (hsize_t)LW_NLEV };
    hid_t dcpl_id_LW = H5Pcreate(H5P_DATASET_CREATE);

    struct
    {
      const char* name;
      const void* buf;
      int rank;
    } lw_sets[] = {
      { "LW_shape_stellar", sum_lyn_LW, 1 },
      { "LW_shape_III", sum_lyn_LW_III, 1 },
      { "LW_shape_AGN", sum_lyn_LW_AGN, 1 },
      { "LW_zpp", LW_zpp, 1 },
      { "LW_emissivity_stellar", LW_emissivity_stellar, 1 },
      { "LW_emissivity_III", LW_emissivity_III, 1 },
      { "LW_emissivity_AGN", LW_emissivity_AGN, 1 },
      { "LW_spectral_stellar", LW_spectral_stellar, 2 },
      { "LW_spectral_III", LW_spectral_III, 2 },
      { "LW_spectral_AGN", LW_spectral_AGN, 2 },
    };

    for (int i_set = 0; i_set < (int)(sizeof(lw_sets) / sizeof(lw_sets[0])); i_set++) {
      hid_t fspace_id_LW =
        (lw_sets[i_set].rank == 1) ? H5Screate_simple(1, dims_LW, NULL) : H5Screate_simple(2, dims_LWspec, NULL);
      hid_t memspace_id_LW =
        (lw_sets[i_set].rank == 1) ? H5Screate_simple(1, dims_LW, NULL) : H5Screate_simple(2, dims_LWspec, NULL);

      hid_t dset_id_LW =
        H5Dcreate(file_id, lw_sets[i_set].name, H5T_NATIVE_DOUBLE, fspace_id_LW, H5P_DEFAULT, dcpl_id_LW, H5P_DEFAULT);

      if (run_globals.mpi_rank != 0) {
        H5Sselect_none(fspace_id_LW);
        H5Sselect_none(memspace_id_LW);
      }

      hid_t plist_id_LW = H5Pcreate(H5P_DATASET_XFER);
      H5Pset_dxpl_mpio(plist_id_LW, H5FD_MPIO_COLLECTIVE);
      H5Dwrite(dset_id_LW, H5T_NATIVE_DOUBLE, memspace_id_LW, fspace_id_LW, plist_id_LW, lw_sets[i_set].buf);
      H5Pclose(plist_id_LW);
      H5Dclose(dset_id_LW);
      H5Sclose(memspace_id_LW);
      H5Sclose(fspace_id_LW);
    }

    H5Pclose(dcpl_id_LW);
  }
#endif

  // tidy up
  free(grid);
  H5Pclose(dcpl_id);
  H5Sclose(memspace_id);
  H5Sclose(fspace_id);
  H5Fclose(file_id);

  if (run_globals.mpi_rank == 0)
    save_reion_output_attributes(snapshot);

  mlog("...done", MLOG_CLOSE); // Saving tocf grids
}

void save_reion_output_attributes(int snapshot)
{
  reion_grids_t* grids = &(run_globals.reion_grids);

  mlog("Saving tocf output attributes...", MLOG_OPEN);
  char name[STRLEN];
  gen_grids_fname(snapshot, name, false);

  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  hid_t file_id = H5Fopen(name, H5F_ACC_RDWR, plist_id);
  H5Pclose(plist_id);

  // Create a scalar dataspace with 0-sized dimension (empty)
  hsize_t dims[1] = { 0 }; // zero-length dataset
  hid_t fspace_id = H5Screate_simple(1, dims, NULL);

// Ensure datasets exist so attribute writes succeed in both code paths:
// (1) after full grid output, (2) when only attribute placeholders are needed.
#define ENSURE_DATASET(name)                                                                                           \
  do {                                                                                                                 \
    if (H5Lexists(file_id, name, H5P_DEFAULT) <= 0) {                                                                  \
      hid_t dset_id = H5Dcreate(file_id, name, H5T_NATIVE_FLOAT, fspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);    \
      H5Dclose(dset_id);                                                                                               \
    }                                                                                                                  \
  } while (0)

  ENSURE_DATASET("xH");
  ENSURE_DATASET("r_bubble");
  ENSURE_DATASET("temp_kinetic_all_gas");

  H5LTset_attribute_double(file_id, "xH", "volume_weighted_global_xH", &(grids->volume_weighted_global_xH), 1);
  H5LTset_attribute_double(file_id, "xH", "mass_weighted_global_xH", &(grids->mass_weighted_global_xH), 1);
  H5LTset_attribute_double(file_id, "xH", "mass_weighted_global_tau_e", &(grids->mass_weighted_global_tau_e), 1);
  H5LTset_attribute_double(
    file_id, "xH", "mass_weighted_global_tau_e_sim", &(grids->mass_weighted_global_tau_e_sim), 1);
  H5LTset_attribute_double(
    file_id, "r_bubble", "volume_weighted_global_r_bubble", &(grids->volume_weighted_global_r_bubble), 1);
  H5LTset_attribute_double(
    file_id, "r_bubble", "mass_weighted_global_r_bubble", &(grids->mass_weighted_global_r_bubble), 1);
  H5LTset_attribute_double(file_id,
                           "temp_kinetic_all_gas",
                           "volume_weighted_global_temp_kinetic_all_gas",
                           &(grids->volume_weighted_global_temp_kinetic_all_gas),
                           1);
  H5LTset_attribute_double(file_id,
                           "temp_kinetic_all_gas",
                           "mass_weighted_global_temp_kinetic_all_gas",
                           &(grids->mass_weighted_global_temp_kinetic_all_gas),
                           1);

  if (run_globals.params.Flag_IncludeRecombinations) {
    ENSURE_DATASET("Gamma12");
    ENSURE_DATASET("N_rec");
    ENSURE_DATASET("residual_xH");
    ENSURE_DATASET("clumping_factor");
    H5LTset_attribute_double(
      file_id, "Gamma12", "volume_weighted_global_Gamma12", &(grids->volume_weighted_global_Gamma12), 1);
    H5LTset_attribute_double(
      file_id, "Gamma12", "mass_weighted_global_Gamma12", &(grids->mass_weighted_global_Gamma12), 1);
    H5LTset_attribute_double(
      file_id, "N_rec", "volume_weighted_global_N_rec", &(grids->volume_weighted_global_N_rec), 1);
    H5LTset_attribute_double(file_id, "N_rec", "mass_weighted_global_N_rec", &(grids->mass_weighted_global_N_rec), 1);
    H5LTset_attribute_double(
      file_id, "residual_xH", "volume_weighted_global_residual_xH", &(grids->volume_weighted_global_residual_xH), 1);
    H5LTset_attribute_double(
      file_id, "residual_xH", "mass_weighted_global_residual_xH", &(grids->mass_weighted_global_residual_xH), 1);
    H5LTset_attribute_double(file_id,
                             "clumping_factor",
                             "volume_weighted_global_clumping_factor",
                             &(grids->volume_weighted_global_clumping_factor),
                             1);
    H5LTset_attribute_double(file_id,
                             "clumping_factor",
                             "mass_weighted_global_clumping_factor",
                             &(grids->mass_weighted_global_clumping_factor),
                             1);
  }

  ENSURE_DATASET("weighted_sfr");
  H5LTset_attribute_double(
    file_id, "weighted_sfr", "volume_weighted_global_weighted_sfr", &(grids->volume_weighted_global_weighted_sfr), 1);
#if USE_MINI_HALOS
  ENSURE_DATASET("weighted_sfrIII");
  H5LTset_attribute_double(file_id,
                           "weighted_sfrIII",
                           "volume_weighted_global_weighted_sfrIII",
                           &(grids->volume_weighted_global_weighted_sfrIII),
                           1);
#endif

  if (run_globals.params.physics.Flag_BHFeedback) {
    ENSURE_DATASET("effective_bhar");
    H5LTset_attribute_double(file_id,
                             "effective_bhar",
                             "volume_weighted_global_effective_bhar",
                             &(grids->volume_weighted_global_effective_bhar),
                             1);
  }

  if (run_globals.params.Flag_IncludeSpinTemp) {
    ENSURE_DATASET("TS_box");
    ENSURE_DATASET("Tk_box");
    ENSURE_DATASET("x_e_box");
    H5LTset_attribute_double(file_id, "TS_box", "volume_ave_TS", &(grids->volume_ave_TS), 1);
    H5LTset_attribute_double(file_id, "Tk_box", "volume_ave_TK", &(grids->volume_ave_TK), 1);
    H5LTset_attribute_double(file_id, "x_e_box", "volume_ave_xe", &(grids->volume_ave_xe), 1);

    H5LTset_attribute_double(file_id, "TS_box", "volume_ave_J_alpha", &(grids->volume_ave_J_alpha), 1);
    H5LTset_attribute_double(file_id, "TS_box", "volume_ave_xalpha", &(grids->volume_ave_xalpha), 1);
    H5LTset_attribute_double(file_id, "TS_box", "volume_ave_Xheat", &(grids->volume_ave_Xheat), 1);
    H5LTset_attribute_double(file_id, "TS_box", "volume_ave_Xion", &(grids->volume_ave_Xion), 1);
    H5LTset_attribute_double(file_id, "TS_box", "volume_ave_Xheat_AGN_soft", &(grids->volume_ave_Xheat_AGN_soft), 1);
    H5LTset_attribute_double(file_id, "TS_box", "volume_ave_Xheat_AGN_hard", &(grids->volume_ave_Xheat_AGN_hard), 1);

#if USE_MINI_HALOS
    ENSURE_DATASET("TS_boxII");
    ENSURE_DATASET("Tk_boxII");
    H5LTset_attribute_double(file_id, "TS_boxII", "volume_ave_TSII", &(grids->volume_ave_TSII), 1);
    H5LTset_attribute_double(file_id, "Tk_boxII", "volume_ave_TKII", &(grids->volume_ave_TKII), 1);
    H5LTset_attribute_double(file_id, "TS_boxII", "volume_ave_J_alphaII", &(grids->volume_ave_J_alphaII), 1);
    H5LTset_attribute_double(file_id, "TS_boxII", "volume_ave_XheatII", &(grids->volume_ave_XheatII), 1);
    H5LTset_attribute_double(file_id, "TS_boxII", "volume_ave_XionII", &(grids->volume_ave_XionII), 1);
#endif
  }

#if USE_MINI_HALOS
  if (run_globals.params.Flag_IncludeLymanWerner) {
    ENSURE_DATASET("JLW_box");
    ENSURE_DATASET("JLW_boxII");
    H5LTset_attribute_double(file_id, "JLW_box", "volume_ave_JLW", &(grids->volume_ave_J_LW), 1);
    H5LTset_attribute_double(file_id, "JLW_boxII", "volume_ave_JLW_II", &(grids->volume_ave_J_LWII), 1);
    H5LTset_attribute_double(file_id, "JLW_box", "volume_ave_JLW_AGN", &(grids->volume_ave_J_LW_AGN), 1);
  }
#endif

  if (run_globals.params.Flag_Compute21cmBrightTemp) {
    ENSURE_DATASET("delta_T");
    H5LTset_attribute_double(file_id, "delta_T", "volume_ave_Tb", &(grids->volume_ave_Tb), 1);
#if USE_MINI_HALOS
    ENSURE_DATASET("delta_TII");
    H5LTset_attribute_double(file_id, "delta_TII", "volume_ave_TbII", &(grids->volume_ave_TbII), 1);
#endif
  }

#undef ENSURE_DATASET
  H5Sclose(fspace_id);
  H5Fclose(file_id);
  mlog("...done", MLOG_CLOSE); // Saving tocf grids
}

bool check_if_reionization_ongoing(int snapshot)
{
  int started = run_globals.reion_grids.started;
  int finished = run_globals.reion_grids.finished;

  // First check if we've already finished on all cores.
  if (finished)
    return false;

  // Ok, so we haven't finished.  Have we started then?
  if (started) {
    // whether we want to continue even when reionization is finished
    // In order to keep outputting meraxes_grids_%d.hdf5
    if (run_globals.params.Flag_OutputGridsPostReion)
      return true;

    if (run_globals.params.Flag_ConstructLightcone && snapshot <= run_globals.params.EndSnapshotLightcone) {
      return true;
    }

    // So we have started, but have not previously found to be finished.  Have
    // we now finished though?
    float* xH = run_globals.reion_grids.xH;
    int ReionGridDim = run_globals.params.ReionGridDim;
    int slab_n_real = (int)(run_globals.reion_grids.slab_nix[run_globals.mpi_rank]) * ReionGridDim * ReionGridDim;

    // If not all cells are ionised then reionization is still progressing...
    finished = 1;
    for (int ii = 0; ii < slab_n_real; ii++)
      if (xH[ii] != 0.0) {
        finished = 0;
        break;
      }
  } else {

    // Here we haven't finished or previously started.  Should we start then?
    if (run_globals.params.Flag_IncludeSpinTemp || run_globals.params.Flag_ConstructLightcone) {
      started = 1;
    } else {
      if (run_globals.FirstGal != NULL) {
        started = 1;
      }
    }
  }

  // At this stage, `started` and `finished` should be set accordingly for each
  // individual core.  Now we need to combine them on all cores.
  MPI_Allreduce(MPI_IN_PLACE, &started, 1, MPI_INT, MPI_LOR, run_globals.mpi_comm);
  run_globals.reion_grids.started = started;
  MPI_Allreduce(MPI_IN_PLACE, &finished, 1, MPI_INT, MPI_LAND, run_globals.mpi_comm);
  run_globals.reion_grids.finished = finished;

  if (started && (!finished))
    return true;
  else
    return false;
}

void filter(fftwf_complex* box, int local_ix_start, int slab_nx, int grid_dim, float R, int filter_type)
{
  // Keep the single-grid API on the same kernels as heating and ionization.
  const fftwf_complex* inputs[] = { box };
  fftwf_complex* outputs[] = { box };
  filter_fields(inputs, outputs, 1, local_ix_start, slab_nx, grid_dim, R, filter_type, 1);
}

void velocity_gradient(fftwf_complex* box, int slab_nx, int grid_dim)
{
  int middle = grid_dim / 2;
  float box_size = (float)run_globals.params.BoxSize;
  float delta_k = (float)(2.0 * M_PI / box_size);

  // Loop through k-box
  for (int n_x = 0; n_x < slab_nx; n_x++) {
    for (int n_y = 0; n_y < grid_dim; n_y++) {
      for (int n_z = 0; n_z <= middle; n_z++) {
        float k_z = n_z * delta_k;
        box[grid_index(n_x, n_y, n_z, grid_dim, INDEX_COMPLEX_HERM)] *= (fftwf_complex)(k_z * 1I);
      }
    }
  } // End looping through k box
}
