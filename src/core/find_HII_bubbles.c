#include <complex.h>
#include <fftw3-mpi.h>
#include <math.h>
#include <string.h>

#include "XRayHeatingFunctions.h"
#include "meraxes.h"
#include "meraxes_gpu.h"
#include "misc_tools.h"
#include "recombinations.h"
#include "reionization.h"
#include "utils.h"
#include "virial_properties.h"

/*
 * This code is a re-write of the modified version of 21cmFAST used in Mutch et
 * al. (2016; Meraxes paper).  The original code was written by Andrei Mesinger
 * with additions as detailed in Sobacchi & Mesinger (2013abc).  Updates were
 * subsequently made by Simon Mutch & Paul Geil.
 *
 * Inclusion of electron fraction (X-ray heating) and inhomogeneous recombinations
 * by Bradley Greig. Relevant functions taken from public version of 21cmFAST.
 *
 * Inclusion od Pop. III galaxies when accounting for MINIHALOS by Manu Ventura.
 */

double RtoM(double R)
{
  // All in internal units
  int filter = run_globals.params.ReionRtoMFilterType;
  double OmegaM = run_globals.params.OmegaM;
  double RhoCrit = run_globals.RhoCrit;

  switch (filter) {
    case 0: // top hat M = (4/3) PI <rho> R^3
      return (4.0 / 3.0) * M_PI * pow(R, 3) * (OmegaM * RhoCrit);
    case 1: // gaussian: M = (2PI)^1.5 <rho> R^3
      return pow(2 * M_PI, 1.5) * OmegaM * RhoCrit * pow(R, 3);
    default: // filter not defined
      mlog_error("Unrecognised filter (%d). Aborting...", filter);
      ABORT(EXIT_FAILURE);
      break;
  }

  return -1;
}

float ComputeFullyIoinizedTemperature(float z_re, float z, float delta)
{
  // z_re: the redshift of reionization
  // z:    the current redshift
  // delta:the density contrast
  float result, delta_re;
  // just be fully ionized
  if (fabs(z - z_re) < 1e-4)
    result = 1;
  else {
    // linearly extrapolate to get density at reionization
    delta_re = delta * (1. + z) / (1. + z_re);
    if (delta_re <= -1)
      delta_re = -1. + ABS_TOL;
    // evolving ionized box eq. 6 of McQuinn 2015, ignored the dependency of density at ionization
    if (delta <= -1)
      delta = -1. + ABS_TOL;
    result = pow((1. + delta) / (1. + delta_re), 1.1333);
    result *= pow((1. + z) / (1. + z_re), 3.4);
    result *= expf(pow((1. + z) / 7.1, 2.5) - pow((1. + z_re) / 7.1, 2.5));
  }
  result *= pow(T_RE, 1.7);
  // 1e4 before helium reionization; double it after
  result += pow(1e4 * ((1. + z) / 4.), 1.7) * (1 + delta);
  result = pow(result, 0.5882);
  return result;
}

float ComputePartiallyIoinizedTemperature(float T_HI, float res_xH)
{
  if (res_xH <= 0.)
    return T_RE;
  if (res_xH >= 1)
    return T_HI;

  return T_HI * res_xH + T_RE * (1. - res_xH);
}

void _find_HII_bubbles(const int snapshot)
{
  // TODO: TAKE A VERY VERY CLOSE LOOK AT UNITS!!!!

  const double box_size = run_globals.params.BoxSize; // Mpc/h
  const int ReionGridDim = run_globals.params.ReionGridDim;
  const double pixel_volume = pow(box_size / (double)ReionGridDim, 3); // (Mpc/h)^3
  double cell_length_factor = L_FACTOR;
  const double total_n_cells = pow((double)ReionGridDim, 3);
  const int local_nix = (int)(run_globals.reion_grids.slab_nix[run_globals.mpi_rank]);
  const int slab_n_real = local_nix * ReionGridDim * ReionGridDim;
  const int slab_n_complex = (int)(run_globals.reion_grids.slab_n_complex[run_globals.mpi_rank]);
  const int flag_ReionUVBFlag = run_globals.params.ReionUVBFlag;
  const double ReionEfficiency = run_globals.params.physics.ReionEfficiency;
  const double ReionNionPhotPerBary = run_globals.params.physics.ReionNionPhotPerBary;
  run_units_t* units = &(run_globals.units);
  float J_21_aux_constant, J_21_aux_constant_BH;
  double density_over_mean;
  double f_coll_stars;
  double sfr_timescale = run_globals.params.ReionSfrTimescale * hubble_time(snapshot);
  double f_coll_effective_bhm = 0.0;
  double neutral_fraction;
  double Gamma_R_prefactor, f_coll_prefactor, Gamma_R_prefactor_BH;
  float thistk, TK;
  float cT_ad; // finding the adiabatic index at the initial redshift from 2302.08506 to fix adiabatic fluctuations.
#if USE_MINI_HALOS
  const double ReionEfficiencyIII = run_globals.params.physics.ReionEfficiencyIII;
  const double ReionNionPhotPerBaryIII = run_globals.params.physics.ReionNionPhotPerBaryIII;
  float J_21_auxIII_constant;
  double f_coll_starsIII;
  double Gamma_R_prefactorIII, f_coll_prefactorIII;
#endif

  double recombination_rate, cf, rec, z_eff, rnh;

  const double redshift = run_globals.ZZ[snapshot];
  double prev_redshift;
  if (snapshot == 0) {
    prev_redshift = run_globals.ZZ[snapshot];
  } else {
    prev_redshift = run_globals.ZZ[snapshot - 1];
  }

  float zstep = (float)(prev_redshift - redshift);
  float fabs_dtdz = (float)fabs(dtdz((float)redshift) / run_globals.params.Hubble_h);
  if (T_RECFAST(100, 1) < 0) {
    mlog_error("Failed to init T_RECFAST. Aborting...");
    ABORT(EXIT_FAILURE);
  }
  TK = T_RECFAST(redshift, 0);
  if (T_RECFAST(100, 2) < 0) {
    mlog_error("Failed to free T_RECFAST. Aborting...");
    ABORT(EXIT_FAILURE);
  }
  cT_ad = cT_approx(redshift);

  int i_real;
  int i_padded;

  // This parameter choice is sensitive to noise on the cell size, at least for the typical
  // cell sizes in RT simulations. It probably doesn't matter for larger cell sizes.
  if ((box_size / (double)ReionGridDim) < 1.0) // Fairly arbitrary length based on 2 runs Sobacchi did
    cell_length_factor = 1.0;

  // Init xH
  float* xH = run_globals.reion_grids.xH;
  for (int ii = 0; ii < slab_n_real; ii++)
    xH[ii] = 1.0;

  // Init r_bubble
  float* r_bubble = run_globals.reion_grids.r_bubble;
  for (int ii = 0; ii < slab_n_real; ii++)
    r_bubble[ii] = 0.0;

  // Forward fourier transform to obtain k-space fields
  // TODO: Ensure that fftwf_mpi_init has been called and fftwf_mpi_cleanup will be called

  float* deltax = run_globals.reion_grids.deltax;
  fftwf_complex* deltax_unfiltered = run_globals.reion_grids.deltax_unfiltered;
  fftwf_complex* deltax_filtered = run_globals.reion_grids.deltax_filtered;
  fftwf_execute(run_globals.reion_grids.deltax_forward_plan);

  fftwf_complex* stars_unfiltered = run_globals.reion_grids.stars_unfiltered;
  fftwf_complex* stars_filtered = run_globals.reion_grids.stars_filtered;
  fftwf_execute(run_globals.reion_grids.stars_forward_plan);

  fftwf_complex* effective_bhm_unfiltered = NULL;
  fftwf_complex* effective_bhm_filtered = NULL;
  fftwf_complex* effective_bhar_unfiltered = NULL;
  fftwf_complex* effective_bhar_filtered = NULL;
  float* effective_bhar = NULL;
  if (run_globals.params.physics.Flag_BHFeedback) {
    effective_bhm_unfiltered = run_globals.reion_grids.effective_bhm_unfiltered;
    effective_bhm_filtered = run_globals.reion_grids.effective_bhm_filtered;
    fftwf_execute(run_globals.reion_grids.effective_bhm_forward_plan);

    effective_bhar = run_globals.reion_grids.effective_bhar;
    effective_bhar_unfiltered = run_globals.reion_grids.effective_bhar_unfiltered;
    effective_bhar_filtered = run_globals.reion_grids.effective_bhar_filtered;
    fftwf_execute(run_globals.reion_grids.effective_bhar_forward_plan);
  }

  float* weighted_sfr = run_globals.reion_grids.weighted_sfr;
  fftwf_complex* weighted_sfr_unfiltered = run_globals.reion_grids.weighted_sfr_unfiltered;
  fftwf_complex* weighted_sfr_filtered = run_globals.reion_grids.weighted_sfr_filtered;
  fftwf_execute(run_globals.reion_grids.weighted_sfr_forward_plan);

#if USE_MINI_HALOS
  fftwf_complex* starsIII_unfiltered = run_globals.reion_grids.starsIII_unfiltered;
  fftwf_complex* starsIII_filtered = run_globals.reion_grids.starsIII_filtered;
  fftwf_execute(run_globals.reion_grids.starsIII_forward_plan);

  float* weighted_sfrIII = run_globals.reion_grids.weighted_sfrIII;
  fftwf_complex* weighted_sfrIII_unfiltered = run_globals.reion_grids.weighted_sfrIII_unfiltered;
  fftwf_complex* weighted_sfrIII_filtered = run_globals.reion_grids.weighted_sfrIII_filtered;
  fftwf_execute(run_globals.reion_grids.weighted_sfrIII_forward_plan);
#endif

  // The free electron fraction from X-rays
  // TODO: Only necessary if we aren't using the GPU (not implemented there yet)
  fftwf_complex* x_e_unfiltered = NULL;
  fftwf_complex* x_e_filtered = NULL;
  if (run_globals.params.Flag_IncludeSpinTemp) {
    x_e_unfiltered = run_globals.reion_grids.x_e_unfiltered;
    x_e_filtered = run_globals.reion_grids.x_e_filtered;
    fftwf_execute(run_globals.reion_grids.x_e_box_forward_plan);
  }

  // Fields relevant for computing the inhomogeneous recombinations
  float* Gamma12 = run_globals.reion_grids.Gamma12;
  float* J_21_at_ionization = run_globals.reion_grids.J_21_at_ionization;
  float* Tk_box = run_globals.reion_grids.Tk_box;
  float* temp_kinetic_all_gas = run_globals.reion_grids.temp_kinetic_all_gas;
  float* z_in = run_globals.reion_grids.z_at_ionization;
  float* N_rec = run_globals.reion_grids.N_rec;
  float* residual_xH = run_globals.reion_grids.residual_xH;
  float* clumping_factor = run_globals.reion_grids.clumping_factor;
  float* t_resp = run_globals.reion_grids.t_resp;
  fftwf_complex* N_rec_unfiltered = NULL;
  fftwf_complex* N_rec_filtered = NULL;
  if (run_globals.params.Flag_IncludeRecombinations) {
    N_rec_unfiltered = run_globals.reion_grids.N_rec_unfiltered;
    N_rec_filtered = run_globals.reion_grids.N_rec_filtered;
    fftwf_execute(run_globals.reion_grids.N_rec_forward_plan);
  }

  // Remember to add the factor of VOLUME/TOT_NUM_PIXELS when converting from real space to k-space
  // Note: we will leave off factor of VOLUME, in anticipation of the inverse FFT below
  // TODO: Double check that looping over correct number of elements here
  for (int ii = 0; ii < slab_n_complex; ii++) {
    deltax_unfiltered[ii] /= total_n_cells;
    stars_unfiltered[ii] /= total_n_cells;
    weighted_sfr_unfiltered[ii] /= total_n_cells;
#if USE_MINI_HALOS
    starsIII_unfiltered[ii] /= total_n_cells;
    weighted_sfrIII_unfiltered[ii] /= total_n_cells;
#endif
    if (run_globals.params.Flag_IncludeRecombinations) {
      N_rec_unfiltered[ii] /= total_n_cells;
    }
    if (run_globals.params.Flag_IncludeSpinTemp) {
      x_e_unfiltered[ii] /= total_n_cells;
    }
    if (run_globals.params.physics.Flag_BHFeedback) {
      effective_bhm_unfiltered[ii] /= total_n_cells;
      effective_bhar_unfiltered[ii] /= total_n_cells;
    }
  }

  // Loop through filter radii
  double ReionRBubbleMax;
  if (run_globals.params.Flag_EvolvingReionRBubbleMax) {
    if (redshift > 6.)
      ReionRBubbleMax = 25.483241248322766; // Mpc/h
    else
      ReionRBubbleMax = 112. * pow((1. + redshift) / 5., -4.4);
  } else {
    if (run_globals.params.Flag_IncludeRecombinations)
      ReionRBubbleMax = run_globals.params.physics.ReionRBubbleMaxRecomb; // Mpc/h
    else
      ReionRBubbleMax = run_globals.params.physics.ReionRBubbleMax; // Mpc/h
  }

  double ReionRBubbleMin = run_globals.params.physics.ReionRBubbleMin; // Mpc/h
  double R = fmin(ReionRBubbleMax, L_FACTOR * box_size);               // Mpc/h
  double ReionDeltaRFactor = run_globals.params.ReionDeltaRFactor;
  double ReionGammaHaloBias = run_globals.params.physics.ReionGammaHaloBias;

  bool flag_last_filter_step = false;

  // set recombinations to zero (for case when recombinations are not used)
  rec = 0.0;

  while (!flag_last_filter_step) {
    // check to see if this is our last filtering step
    if (((R / ReionDeltaRFactor) <= (cell_length_factor * box_size / (double)ReionGridDim)) ||
        ((R / ReionDeltaRFactor) <= ReionRBubbleMin)) {
      flag_last_filter_step = true;
      R = cell_length_factor * box_size / (double)ReionGridDim;
    }

    // DEBUG
    // mlog("R = %.2e (h=0.678 -> %.2e)", MLOG_MESG, R, R/0.678);
    mlog(".", MLOG_CONT);

    // copy the k-space grids
    memcpy(deltax_filtered, deltax_unfiltered, sizeof(fftwf_complex) * slab_n_complex);
    memcpy(stars_filtered, stars_unfiltered, sizeof(fftwf_complex) * slab_n_complex);
    memcpy(weighted_sfr_filtered, weighted_sfr_unfiltered, sizeof(fftwf_complex) * slab_n_complex);
#if USE_MINI_HALOS
    memcpy(starsIII_filtered, starsIII_unfiltered, sizeof(fftwf_complex) * slab_n_complex);
    memcpy(weighted_sfrIII_filtered, weighted_sfrIII_unfiltered, sizeof(fftwf_complex) * slab_n_complex);
#endif

    if (run_globals.params.Flag_IncludeRecombinations) {
      memcpy(N_rec_filtered, N_rec_unfiltered, sizeof(fftwf_complex) * slab_n_complex);
    }
    if (run_globals.params.Flag_IncludeSpinTemp) {
      memcpy(x_e_filtered, x_e_unfiltered, sizeof(fftwf_complex) * slab_n_complex);
    }
    if (run_globals.params.physics.Flag_BHFeedback) {
      memcpy(effective_bhm_filtered, effective_bhm_unfiltered, sizeof(fftwf_complex) * slab_n_complex);
      memcpy(effective_bhar_filtered, effective_bhar_unfiltered, sizeof(fftwf_complex) * slab_n_complex);
    }

    // do the filtering unless this is the last filter step
    int local_ix_start = (int)(run_globals.reion_grids.slab_ix_start[run_globals.mpi_rank]);
    if (!flag_last_filter_step) {
      filter(deltax_filtered, local_ix_start, local_nix, ReionGridDim, (float)R, run_globals.params.ReionFilterType);
      filter(stars_filtered, local_ix_start, local_nix, ReionGridDim, (float)R, run_globals.params.ReionFilterType);
      filter(
        weighted_sfr_filtered, local_ix_start, local_nix, ReionGridDim, (float)R, run_globals.params.ReionFilterType);
#if USE_MINI_HALOS
      filter(starsIII_filtered, local_ix_start, local_nix, ReionGridDim, (float)R, run_globals.params.ReionFilterType);
      filter(weighted_sfrIII_filtered,
             local_ix_start,
             local_nix,
             ReionGridDim,
             (float)R,
             run_globals.params.ReionFilterType);
#endif

      if (run_globals.params.Flag_IncludeRecombinations) {
        filter(N_rec_filtered, local_ix_start, local_nix, ReionGridDim, (float)R, run_globals.params.ReionFilterType);
      }
      if (run_globals.params.Flag_IncludeSpinTemp) {
        filter(x_e_filtered, local_ix_start, local_nix, ReionGridDim, (float)R, run_globals.params.ReionFilterType);
      }
      if (run_globals.params.physics.Flag_BHFeedback) {
        filter(effective_bhm_filtered,
               local_ix_start,
               local_nix,
               ReionGridDim,
               (float)R,
               run_globals.params.ReionFilterType);
        filter(effective_bhar_filtered,
               local_ix_start,
               local_nix,
               ReionGridDim,
               (float)R,
               run_globals.params.ReionFilterType);
      }
    }

    // inverse fourier transform back to real space
    fftwf_execute(run_globals.reion_grids.deltax_filtered_reverse_plan);
    fftwf_execute(run_globals.reion_grids.stars_filtered_reverse_plan);
    fftwf_execute(run_globals.reion_grids.weighted_sfr_filtered_reverse_plan);
#if USE_MINI_HALOS
    fftwf_execute(run_globals.reion_grids.starsIII_filtered_reverse_plan);
    fftwf_execute(run_globals.reion_grids.weighted_sfrIII_filtered_reverse_plan);
#endif

    if (run_globals.params.Flag_IncludeRecombinations) {
      fftwf_execute(run_globals.reion_grids.N_rec_filtered_reverse_plan);
    }

    if (run_globals.params.Flag_IncludeSpinTemp) {
      fftwf_execute(run_globals.reion_grids.x_e_filtered_reverse_plan);
    }

    if (run_globals.params.physics.Flag_BHFeedback) {
      fftwf_execute(run_globals.reion_grids.effective_bhm_filtered_reverse_plan);
      fftwf_execute(run_globals.reion_grids.effective_bhar_filtered_reverse_plan);
    }

    // Perform sanity checks to account for aliasing effects
    for (int ix = 0; ix < local_nix; ix++)
      for (int iy = 0; iy < ReionGridDim; iy++)
        for (int iz = 0; iz < ReionGridDim; iz++) {
          i_padded = grid_index(ix, iy, iz, ReionGridDim, INDEX_PADDED);
          ((float*)deltax_filtered)[i_padded] = fmaxf(((float*)deltax_filtered)[i_padded], -1 + REL_TOL);
          ((float*)stars_filtered)[i_padded] = fmaxf(((float*)stars_filtered)[i_padded], 0.0);
          if (((float*)stars_filtered)[i_padded] < ABS_TOL) {
            ((float*)stars_filtered)[i_padded] = 0;
          }
          ((float*)weighted_sfr_filtered)[i_padded] = fmaxf(((float*)weighted_sfr_filtered)[i_padded], 0.0);
          if (((float*)weighted_sfr_filtered)[i_padded] < ABS_TOL) {
            ((float*)weighted_sfr_filtered)[i_padded] = 0;
          }

#if USE_MINI_HALOS
          ((float*)starsIII_filtered)[i_padded] = fmaxf(((float*)starsIII_filtered)[i_padded], 0.0);
          if (((float*)starsIII_filtered)[i_padded] < ABS_TOL) {
            ((float*)starsIII_filtered)[i_padded] = 0;
          }

          ((float*)weighted_sfrIII_filtered)[i_padded] = fmaxf(((float*)weighted_sfrIII_filtered)[i_padded], 0.0);
          if (((float*)weighted_sfrIII_filtered)[i_padded] < ABS_TOL) {
            ((float*)weighted_sfrIII_filtered)[i_padded] = 0;
          }
#endif

          if (run_globals.params.Flag_IncludeRecombinations) {
            ((float*)N_rec_filtered)[i_padded] = fmaxf(((float*)N_rec_filtered)[i_padded], 0.0);
            if (((float*)N_rec_filtered)[i_padded] < ABS_TOL) {
              ((float*)N_rec_filtered)[i_padded] = 0;
            }
          }
          if (run_globals.params.Flag_IncludeSpinTemp) {
            ((float*)x_e_filtered)[i_padded] = fmaxf(((float*)x_e_filtered)[i_padded], 0.0);
            if (((float*)x_e_filtered)[i_padded] < ABS_TOL) {
              ((float*)x_e_filtered)[i_padded] = 0;
            }
            ((float*)x_e_filtered)[i_padded] = fminf(((float*)x_e_filtered)[i_padded], 0.999);
          }
          if (run_globals.params.physics.Flag_BHFeedback) {
            ((float*)effective_bhm_filtered)[i_padded] = fmaxf(((float*)effective_bhm_filtered)[i_padded], 0.0);
            if (((float*)effective_bhm_filtered)[i_padded] < ABS_TOL) {
              ((float*)effective_bhm_filtered)[i_padded] = 0;
            }
            ((float*)effective_bhar_filtered)[i_padded] = fmaxf(((float*)effective_bhar_filtered)[i_padded], 0.0);
            if (((float*)effective_bhar_filtered)[i_padded] < ABS_TOL) {
              ((float*)effective_bhar_filtered)[i_padded] = 0;
            }
          }
        }

    /*
     * Main loop through the box...
     */
    double M_mean = RtoM(R);
    double R_cubed = R * R * R;

    Gamma_R_prefactor = (1.0 + redshift) * (1.0 + redshift) * R * units->UnitLength_in_cm / pixel_volume *
                        pow(units->UnitLength_in_cm, -3.);
    Gamma_R_prefactor *= (units->UnitMass_in_g / units->UnitTime_in_s) / PROTONMASS * ReionNionPhotPerBary;
    Gamma_R_prefactor_BH = Gamma_R_prefactor;
    J_21_aux_constant = (float)(Gamma_R_prefactor * PLANCK * 1e21 / (4.0 * M_PI) * ReionGammaHaloBias / sfr_timescale *
                                run_globals.params.physics.ReionAlphaUV);
    J_21_aux_constant_BH = (float)(Gamma_R_prefactor_BH * PLANCK * 1e21 / (4.0 * M_PI) * ReionGammaHaloBias /
                                   sfr_timescale * run_globals.params.physics.ReionAlphaUVBH);
    Gamma_R_prefactor *=
      SIGMA_HI * run_globals.params.physics.ReionAlphaUV / (run_globals.params.physics.ReionAlphaUV + 2.75) * 1e12;
    Gamma_R_prefactor_BH *=
      SIGMA_HI * run_globals.params.physics.ReionAlphaUVBH / (run_globals.params.physics.ReionAlphaUVBH + 2.75) * 1e12;
    f_coll_prefactor = ReionEfficiency * (4.0 / 3.0) * M_PI / pixel_volume / M_mean * R_cubed;
#if USE_MINI_HALOS
    J_21_auxIII_constant =
      J_21_aux_constant *
      (float)(ReionNionPhotPerBaryIII / ReionNionPhotPerBary); // Is HaloBias the same for PopIII / Pop II?
    Gamma_R_prefactorIII = Gamma_R_prefactor * ReionNionPhotPerBaryIII / ReionNionPhotPerBary;
    f_coll_prefactorIII = f_coll_prefactor / ReionEfficiency * ReionEfficiencyIII;
#endif

    for (int ix = 0; ix < local_nix; ix++)
      for (int iy = 0; iy < ReionGridDim; iy++)
        for (int iz = 0; iz < ReionGridDim; iz++) {
          i_real = grid_index(ix, iy, iz, ReionGridDim, INDEX_REAL);
          i_padded = grid_index(ix, iy, iz, ReionGridDim, INDEX_PADDED);

          density_over_mean = 1.0 + (double)((float*)deltax_filtered)[i_padded];

          f_coll_stars = (double)((float*)stars_filtered)[i_padded] / density_over_mean;
#if USE_MINI_HALOS
          f_coll_starsIII = (double)((float*)starsIII_filtered)[i_padded] / density_over_mean;
#endif
          if (run_globals.params.physics.Flag_BHFeedback)
            f_coll_effective_bhm = (double)((float*)effective_bhm_filtered)[i_padded] / density_over_mean;

          // Calculate the recombinations within the cell
          if (run_globals.params.Flag_IncludeRecombinations) {
            rec = (double)((float*)N_rec_filtered)[i_padded] / density_over_mean;
          }

          // Account for the partial ionisation of the cell from X-rays
          if (run_globals.params.Flag_IncludeSpinTemp) {
            neutral_fraction = 1.0 - (double)((float*)x_e_filtered)[i_padded];
          } else {
            neutral_fraction = 1.0;
          }

          // Modified reionisation condition, including recombinations and partial ionisations from X-rays
          // Check if ionised!

#if USE_MINI_HALOS
          if (((f_coll_effective_bhm + f_coll_stars) * f_coll_prefactor + f_coll_starsIII * f_coll_prefactorIII) >
              neutral_fraction * (1. + rec)) // IONISED!!!!
#else
          if ((f_coll_effective_bhm + f_coll_stars) * f_coll_prefactor > neutral_fraction * (1. + rec))
#endif
          {
            // If it is the first crossing of the ionisation barrier for this cell (largest R)
            // Store the ionisation background and the reionisation redshift for each cell
            if ((xH[i_real] > REL_TOL) && (run_globals.params.Flag_IncludeRecombinations)) {
              Gamma12[i_real] = ((float*)weighted_sfr_filtered)[i_padded] * (float)(Gamma_R_prefactor);
#if USE_MINI_HALOS
              Gamma12[i_real] += ((float*)weighted_sfrIII_filtered)[i_padded] * (float)(Gamma_R_prefactorIII);
#endif
              if (run_globals.params.physics.Flag_BHFeedback)
                Gamma12[i_real] += ((float*)effective_bhar_filtered)[i_padded] * (float)(Gamma_R_prefactor_BH);
              // Record radius
              r_bubble[i_real] = (float)R;
              // Update J_21 continuously if ReionUVBFlag = 2
              if (flag_ReionUVBFlag == 2) {
                J_21_at_ionization[i_real] = ((float*)stars_filtered)[i_padded] * J_21_aux_constant;
#if USE_MINI_HALOS
                J_21_at_ionization[i_real] += ((float*)starsIII_filtered)[i_padded] * J_21_auxIII_constant;
#endif
                if (run_globals.params.physics.Flag_BHFeedback)
                  J_21_at_ionization[i_real] += ((float*)effective_bhm_filtered)[i_padded] * J_21_aux_constant_BH;
              }
            }

            // Mark as ionised
            xH[i_real] = 0;

          }
          // Check if this is the last filtering step.
          // If so, assign partial ionisations to those cells which aren't fully ionised
          else if (flag_last_filter_step && (xH[i_real] > REL_TOL)) {
            if (run_globals.params.Flag_IncludeSpinTemp)
              temp_kinetic_all_gas[i_real] = ComputePartiallyIoinizedTemperature(Tk_box[i_real], xH[i_real]);
            else
              temp_kinetic_all_gas[i_real] =
                ComputePartiallyIoinizedTemperature(TK * (1. + cT_ad * deltax[i_padded]), xH[i_real]);

            xH[i_real] = (float)(neutral_fraction - (f_coll_effective_bhm + f_coll_stars) * f_coll_prefactor);
#if USE_MINI_HALOS
            xH[i_real] -= (float)(f_coll_starsIII * f_coll_prefactorIII);
#endif
            CLAMP_0_1(xH[i_real]);
          }

          // Check if new ionisation
          if ((xH[i_real] < REL_TOL) && (z_in[i_real] < 0)) // New ionisation!
          {
            z_in[i_real] = (float)redshift;
            // Set J_21 at new ionization if ReionUVBFlag = 1 (const J_21)
            if (flag_ReionUVBFlag == 1) {
              J_21_at_ionization[i_real] = ((float*)stars_filtered)[i_padded] * J_21_aux_constant;
#if USE_MINI_HALOS
              J_21_at_ionization[i_real] += ((float*)starsIII_filtered)[i_padded] * J_21_auxIII_constant;
#endif
              if (run_globals.params.physics.Flag_BHFeedback)
                J_21_at_ionization[i_real] += ((float*)effective_bhm_filtered)[i_padded] * J_21_aux_constant_BH;
            }
          }
        }
    // iz
    R /= ReionDeltaRFactor;
  }

  // Find the volume and mass weighted neutral fractions
  // TODO: The deltax grid will have rounding errors from forward and reverse
  //       FFT. Should cache deltax slabs prior to ffts and reuse here.
  double volume_weighted_global_xH = 0.0;
  double volume_weighted_global_Gamma12 = 0.0;
  double volume_weighted_global_r_bubble = 0.0;
  double volume_weighted_global_weighted_sfr = 0.0;
#if USE_MINI_HALOS
  double volume_weighted_global_weighted_sfrIII = 0.0;
#endif
  double volume_weighted_global_effective_bhar = 0.0;
  double volume_weighted_global_temp_kinetic_all_gas = 0.0;
  double volume_weighted_global_N_rec = 0.0;
  double volume_weighted_global_residual_xH = 0.0;
  double volume_weighted_global_clumping_factor = 0.0;

  double mass_weight = 0.0;
  double mass_weighted_global_xH = 0.0;
  double mass_weighted_global_Gamma12 = 0.0;
  double mass_weighted_global_r_bubble = 0.0;
  double mass_weighted_global_temp_kinetic_all_gas = 0.0;
  double mass_weighted_global_N_rec = 0.0;
  double mass_weighted_global_residual_xH = 0.0;
  double mass_weighted_global_clumping_factor = 0.0;

  double Hubble_h = run_globals.params.Hubble_h;
  double temp;

  for (int ix = 0; ix < local_nix; ix++)
    for (int iy = 0; iy < ReionGridDim; iy++)
      for (int iz = 0; iz < ReionGridDim; iz++) {
        i_real = grid_index(ix, iy, iz, ReionGridDim, INDEX_REAL);
        i_padded = grid_index(ix, iy, iz, ReionGridDim, INDEX_PADDED);
        double cell_xH = (double)(xH[i_real]);
        volume_weighted_global_xH += cell_xH;
        volume_weighted_global_r_bubble += (double)r_bubble[i_real];
        volume_weighted_global_weighted_sfr += (double)((float*)weighted_sfr)[i_padded];
#if USE_MINI_HALOS
        volume_weighted_global_weighted_sfrIII += (double)weighted_sfrIII[i_padded];
#endif
        if (run_globals.params.physics.Flag_BHFeedback)
          volume_weighted_global_effective_bhar += (double)((float*)effective_bhar)[i_padded];

        density_over_mean = 1.0 + (double)((float*)deltax)[i_padded];
        mass_weighted_global_xH += cell_xH * density_over_mean;
        mass_weighted_global_r_bubble += (double)r_bubble[i_real] * density_over_mean;
        mass_weight += density_over_mean;

        if ((z_in[i_real] > 0) && (xH[i_real] < REL_TOL))
          temp_kinetic_all_gas[i_real] =
            ComputeFullyIoinizedTemperature(z_in[i_real], (float)redshift, ((float*)deltax)[i_padded]);

        // Below sometimes (very rare though) can happen when the density drops too fast and to below T_HI
        if (run_globals.params.Flag_IncludeSpinTemp) {
          if (temp_kinetic_all_gas[i_real] < Tk_box[i_real])
            temp_kinetic_all_gas[i_real] = Tk_box[i_real];
        } else {
          thistk = TK * (1. + cT_ad * deltax[i_padded]);
          if (temp_kinetic_all_gas[i_real] < thistk)
            temp_kinetic_all_gas[i_real] = thistk;
        }
        volume_weighted_global_temp_kinetic_all_gas += (double)temp_kinetic_all_gas[i_real];
        mass_weighted_global_temp_kinetic_all_gas += (double)temp_kinetic_all_gas[i_real] * density_over_mean;

        if (run_globals.params.Flag_IncludeRecombinations) {
          // Store the resultant recombination cell
          z_eff = (1. + redshift) * pow(density_over_mean, 1.0 / 3.0) - 1;
          if (run_globals.params.Flag_TemperatureDependentRec)
            temp = (double)temp_kinetic_all_gas[i_real];
          else
            temp = 1e4;
          if (splined_recombination(
                z_eff, (double)Gamma12[i_real] * Hubble_h * Hubble_h, temp, &recombination_rate, &rnh, &cf) != 1) {
            mlog_error("splined_recombination failed. Aborting...");
            ABORT(EXIT_FAILURE);
          }
          N_rec[i_padded] += (float)recombination_rate * fabs_dtdz * zstep * (1. - (float)cell_xH);
          residual_xH[i_real] = (float)rnh;
          clumping_factor[i_real] = (float)cf;

          volume_weighted_global_N_rec += (double)N_rec[i_padded];
          // Compute relaxation timescale: t_resp = 1 / (Gamma + recombination_rate)
          t_resp[i_real] = Gamma12[i_real] * Hubble_h * Hubble_h * 1e-12 + recombination_rate;
          if (t_resp[i_real] > 0)
            t_resp[i_real] = 1.0 / t_resp[i_real] / SEC_PER_MEGAYEAR; // store in Myr
          else
            t_resp[i_real] = 1e30; // effectively infinite

          volume_weighted_global_residual_xH += (double)residual_xH[i_real];
          volume_weighted_global_clumping_factor += (double)clumping_factor[i_real];
          volume_weighted_global_Gamma12 += (double)Gamma12[i_real];
          mass_weighted_global_N_rec += (double)N_rec[i_padded] * density_over_mean;
          mass_weighted_global_residual_xH += (double)residual_xH[i_real] * density_over_mean;
          mass_weighted_global_clumping_factor += (double)clumping_factor[i_real] * density_over_mean;
          mass_weighted_global_Gamma12 += (double)Gamma12[i_real] * density_over_mean;
        }
      }

  MPI_Allreduce(MPI_IN_PLACE, &volume_weighted_global_xH, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
  MPI_Allreduce(MPI_IN_PLACE, &volume_weighted_global_Gamma12, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
  MPI_Allreduce(MPI_IN_PLACE, &volume_weighted_global_r_bubble, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
  MPI_Allreduce(MPI_IN_PLACE, &volume_weighted_global_weighted_sfr, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
  MPI_Allreduce(MPI_IN_PLACE, &volume_weighted_global_effective_bhar, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
  MPI_Allreduce(
    MPI_IN_PLACE, &volume_weighted_global_temp_kinetic_all_gas, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
  MPI_Allreduce(MPI_IN_PLACE, &volume_weighted_global_N_rec, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
  MPI_Allreduce(MPI_IN_PLACE, &volume_weighted_global_residual_xH, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
  MPI_Allreduce(MPI_IN_PLACE, &volume_weighted_global_clumping_factor, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
  MPI_Allreduce(MPI_IN_PLACE, &mass_weighted_global_xH, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
  MPI_Allreduce(MPI_IN_PLACE, &mass_weighted_global_Gamma12, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
  MPI_Allreduce(MPI_IN_PLACE, &mass_weighted_global_r_bubble, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
  MPI_Allreduce(MPI_IN_PLACE, &mass_weighted_global_temp_kinetic_all_gas, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
  MPI_Allreduce(MPI_IN_PLACE, &mass_weighted_global_N_rec, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
  MPI_Allreduce(MPI_IN_PLACE, &mass_weighted_global_residual_xH, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
  MPI_Allreduce(MPI_IN_PLACE, &mass_weighted_global_clumping_factor, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
  MPI_Allreduce(MPI_IN_PLACE, &mass_weight, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);

  volume_weighted_global_xH /= total_n_cells;
  volume_weighted_global_Gamma12 /= total_n_cells;
  volume_weighted_global_r_bubble /= total_n_cells;
  volume_weighted_global_weighted_sfr /= total_n_cells;
  volume_weighted_global_weighted_sfr *= units->UnitMass_in_g / units->UnitTime_in_s * SEC_PER_YEAR / SOLAR_MASS;
  volume_weighted_global_effective_bhar /= total_n_cells;
  volume_weighted_global_effective_bhar *= units->UnitMass_in_g / units->UnitTime_in_s * SEC_PER_YEAR / SOLAR_MASS;
  volume_weighted_global_temp_kinetic_all_gas /= total_n_cells;
  volume_weighted_global_N_rec /= total_n_cells;
  volume_weighted_global_residual_xH /= total_n_cells;
  volume_weighted_global_clumping_factor /= total_n_cells;

  mass_weighted_global_xH /= mass_weight;
  mass_weighted_global_Gamma12 /= mass_weight;
  mass_weighted_global_r_bubble /= mass_weight;
  mass_weighted_global_temp_kinetic_all_gas /= mass_weight;
  mass_weighted_global_N_rec /= mass_weight;
  mass_weighted_global_residual_xH /= mass_weight;
  mass_weighted_global_clumping_factor /= mass_weight;

  run_globals.reion_grids.volume_weighted_global_xH = volume_weighted_global_xH;
  run_globals.reion_grids.volume_weighted_global_Gamma12 = volume_weighted_global_Gamma12;
  run_globals.reion_grids.volume_weighted_global_r_bubble = volume_weighted_global_r_bubble;
  run_globals.reion_grids.volume_weighted_global_weighted_sfr = volume_weighted_global_weighted_sfr;
  run_globals.reion_grids.volume_weighted_global_effective_bhar = volume_weighted_global_effective_bhar;
  run_globals.reion_grids.volume_weighted_global_temp_kinetic_all_gas = volume_weighted_global_temp_kinetic_all_gas;
  run_globals.reion_grids.volume_weighted_global_N_rec = volume_weighted_global_N_rec;
  run_globals.reion_grids.volume_weighted_global_residual_xH = volume_weighted_global_residual_xH;
  run_globals.reion_grids.volume_weighted_global_clumping_factor = volume_weighted_global_clumping_factor;

  run_globals.reion_grids.mass_weighted_global_xH = mass_weighted_global_xH;
  run_globals.reion_grids.mass_weighted_global_Gamma12 = mass_weighted_global_Gamma12;
  run_globals.reion_grids.mass_weighted_global_r_bubble = mass_weighted_global_r_bubble;
  run_globals.reion_grids.mass_weighted_global_temp_kinetic_all_gas = mass_weighted_global_temp_kinetic_all_gas;
  run_globals.reion_grids.mass_weighted_global_N_rec = mass_weighted_global_N_rec;
  run_globals.reion_grids.mass_weighted_global_residual_xH = mass_weighted_global_residual_xH;
  run_globals.reion_grids.mass_weighted_global_clumping_factor = mass_weighted_global_clumping_factor;

#if USE_MINI_HALOS
  MPI_Allreduce(MPI_IN_PLACE, &volume_weighted_global_weighted_sfrIII, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
  volume_weighted_global_weighted_sfrIII /= total_n_cells;
  volume_weighted_global_weighted_sfrIII *= units->UnitMass_in_g / units->UnitTime_in_s * SEC_PER_YEAR / SOLAR_MASS;
  run_globals.reion_grids.volume_weighted_global_weighted_sfrIII = volume_weighted_global_weighted_sfrIII;
#endif
}

// This function makes sure that the right version of find_HII_bubbles() gets called.
void find_HII_bubbles(int snapshot, timer_info* timer_total)
{
  // Call the version of find_HII_bubbles we've been passed (and time it)
  double redshift = run_globals.ZZ[snapshot];
  timer_info timer;
#ifdef USE_CUDA
  mlog("Calling hybrid-GPU/FFTW version of find_HII_bubbles() for snap=%d/z=%.2lf...",
       MLOG_OPEN | MLOG_TIMERSTART,
       snapshot,
       redshift);

  // Run the GPU version of _find_HII_bubbles()
  timer_start(&timer);

  int flag_write_validation_data = false;
  _find_HII_bubbles_gpu(snapshot, flag_write_validation_data);
#else
  // Run the Meraxes version of _find_HII_bubbles()
  mlog("Calling pure-CPU version of find_HII_bubbles() for snap=%d/z=%.2lf...",
       MLOG_OPEN | MLOG_TIMERSTART,
       snapshot,
       redshift);
  timer_start(&timer);
  _find_HII_bubbles(snapshot);
#endif
  timer_stop(&timer);
  timer_stop(timer_total);
  timer_gpu += timer_delta(timer);
  mlog("...done", MLOG_CLOSE | MLOG_TIMERSTOP);
  mlog("Total time spent in find_HII_bubbles vs. total run time (snapshot %d ): %.2f of %.2f s",
       MLOG_MESG,
       snapshot,
       timer_gpu,
       timer_delta(*timer_total));
}
