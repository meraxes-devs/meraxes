#include <complex.h>
#include <fftw3-mpi.h>
#include <math.h>
#include <string.h>

#include "ComputeTs.h"
#include "XRayHeatingFunctions.h"
#include "find_HII_bubbles.h"
#include "meraxes.h"
#include "misc_tools.h"
#include "reionization.h"
#include "utils.h"

/*
 * This code is a re-write of the spin temperature calculation (Ts.c) within 21cmFAST.
 * Modified for usage within Meraxes by Bradley Greig.
 *
 * Note: 21cmFAST has little_h included, therefore below I explicitly convert (using little_h) where appropriate. Be
 * careful with units!
 */

/*
 * The minihalo feature was written by Manu Ventura, which includes an
 * amalgamation of the requisite functions for LW background using formulation of Qin2020a
 *
 * The output with II are computed accounting only for Pop II galaxies. In particular we compute Lyman-alpha, Xray, LW
 * only for Pop II. Notice that the reionization still accounts for Pop III galaxies so in this "only Pop II scenario"
 * we are still considering for the UV ionizing feedback from Pop III (xHI, and xe are computed using both PopII and Pop
 * III)
 */

int set_sfr_history()
{
  double box_size = run_globals.params.BoxSize / run_globals.params.Hubble_h; // Mpc
  int ReionGridDim = run_globals.params.ReionGridDim;
  int TsNumFilterSteps = run_globals.params.TsNumFilterSteps;
  int snapshot_counter_backwards, R_ct;
  double R, R_factor, zp, prev_zpp, prev_R, zpp_edge = 0;
  int NOutputSnaps = run_globals.NOutputSnaps;
  int last_snap = 0;

  for (int ii = 0; ii < NOutputSnaps; ii++)
    if (run_globals.ListOutputSnaps[ii] > last_snap)
      last_snap = run_globals.ListOutputSnaps[ii];

  int max_snaps = 0;
  for (int snapshot = 1; snapshot <= last_snap; snapshot++) {
    zp = run_globals.ZZ[snapshot];

    R = L_FACTOR * box_size / (float)ReionGridDim;
    R_factor = pow(R_XLy_MAX / R, 1 / (float)TsNumFilterSteps);

    snapshot_counter_backwards = 1;
    prev_R = 0;
    prev_zpp = zp;
    for (R_ct = 0; R_ct < TsNumFilterSteps; R_ct++) {
      zpp_edge = prev_zpp - (R - prev_R) * MPC / (drdz((float)prev_zpp)); // cell size

      while (zpp_edge > run_globals.ZZ[snapshot - snapshot_counter_backwards]) {
        if (snapshot - snapshot_counter_backwards == 0)
          break;
        snapshot_counter_backwards += 1;
      }
#ifdef DEBUG
      mlog("snapshot=%d, z=%.1f, R_ct = %d, R=%.1f, zpp_edge=%.1f, snapshot_counter_backwards=%d",
           MLOG_MESG,
           snapshot,
           zp,
           R_ct,
           R,
           zpp_edge,
           snapshot_counter_backwards);
#endif

      prev_R = R;
      prev_zpp = zpp_edge;
      R *= R_factor;
    }
    if (max_snaps < snapshot_counter_backwards)
      max_snaps = snapshot_counter_backwards;
  }
  return max_snaps;
}

void _ComputeTs(int snapshot)
{
  double box_size = run_globals.params.BoxSize / run_globals.params.Hubble_h; // Mpc
  int ReionGridDim = run_globals.params.ReionGridDim;
  double pixel_volume = pow(box_size / (double)ReionGridDim, 3); // (Mpc)^3
  double total_n_cells = pow((double)ReionGridDim, 3);
  int local_nix = (int)(run_globals.reion_grids.slab_nix[run_globals.mpi_rank]);
  int slab_n_complex = (int)(run_globals.reion_grids.slab_n_complex[run_globals.mpi_rank]);
  double ReionEfficiency = run_globals.params.physics.ReionEfficiency;
  run_units_t* units = &(run_globals.units);
#if USE_MINI_HALOS
  double ReionEfficiencyIII = run_globals.params.physics.ReionEfficiencyIII;
#endif

  double redshift = run_globals.ZZ[snapshot];
  double prev_redshift;
  if (snapshot == 0) {
    prev_redshift = run_globals.ZZ[snapshot];
  } else {
    prev_redshift = run_globals.ZZ[snapshot - 1];
  }

  int i_real, i_padded, i_smoothed_heating, R_ct, x_e_ct, n_ct, m_xHII_low, m_xHII_high, NO_LIGHT;

  double prev_zpp, prev_R, zpp, zp, lower_int_limit_GAL, filling_factor_of_HI_zp, R_factor, R, nuprime, dzp,
    Luminosity_converstion_factor_GAL;
  double collapse_fraction, density_over_mean, collapse_fraction_in_cell;
  double agn_xray_hard_ave = 0.0, agn_xray_soft_ave = 0.0; /* volume-averaged emissivity density, this snapshot */
  double hmxb_xray_ave = 0.0; /* volume-averaged HMXB (galaxy) X-ray emissivity density, this snapshot */
  double lower_int_limit_AGN_soft;
  double lower_int_limit_AGN_hard;
  double upper_int_limit_GAL;
  double upper_int_limit_AGN_soft;
  double upper_int_limit_AGN_hard;
  double nu_tau_one_zpp;
  double Luminosity_converstion_factor_AGN_soft; /* soft band: nu_thresh -> nu_break   */
  double Luminosity_converstion_factor_AGN_hard; /* hard band: nu_break  -> nu_hard_cut */
  float bh, bh_soft; /* per-cell BHXrayEmissivity(_soft), read while building SMOOTHED_AGN_hard(_soft) */
#if USE_MINI_HALOS
  float bh_uv; /* per-cell BHUVEmissivity, read while building SMOOTHED_AGN_UV */
#endif

#if USE_MINI_HALOS
  double Luminosity_converstion_factor_III, collapse_fractionIII, collapse_fractionIII_in_cell;
#endif

  // TODO: Can we reduce the scope of these variables and, if not, improve the names?
  double weight, total_weight;
  bool first_radii = true;
  bool first_zero = true;
  int n_pts_radii = 1000;

  float curr_xalpha;
  int TsNumFilterSteps = run_globals.params.TsNumFilterSteps;

  double freq_int_heat_AGN_soft[TsNumFilterSteps], freq_int_ion_AGN_soft[TsNumFilterSteps],
    freq_int_lya_AGN_soft[TsNumFilterSteps], freq_int_heat_AGN_hard[TsNumFilterSteps],
    freq_int_ion_AGN_hard[TsNumFilterSteps], freq_int_lya_AGN_hard[TsNumFilterSteps];
  double freq_int_heat_GAL[TsNumFilterSteps], freq_int_ion_GAL[TsNumFilterSteps], freq_int_lya_GAL[TsNumFilterSteps];
#if USE_MINI_HALOS
  double freq_int_heat_III[TsNumFilterSteps], freq_int_ion_III[TsNumFilterSteps], freq_int_lya_III[TsNumFilterSteps];
#endif

  double freq_int_heat_tbl_AGN_soft[x_int_NXHII][TsNumFilterSteps],
    freq_int_ion_tbl_AGN_soft[x_int_NXHII][TsNumFilterSteps], freq_int_lya_tbl_AGN_soft[x_int_NXHII][TsNumFilterSteps],
    freq_int_heat_tbl_AGN_hard[x_int_NXHII][TsNumFilterSteps], freq_int_ion_tbl_AGN_hard[x_int_NXHII][TsNumFilterSteps],
    freq_int_lya_tbl_AGN_hard[x_int_NXHII][TsNumFilterSteps];
  double freq_int_heat_tbl_GAL[x_int_NXHII][TsNumFilterSteps], freq_int_ion_tbl_GAL[x_int_NXHII][TsNumFilterSteps],
    freq_int_lya_tbl_GAL[x_int_NXHII][TsNumFilterSteps];
#if USE_MINI_HALOS
  double freq_int_heat_tbl_III[x_int_NXHII][TsNumFilterSteps], freq_int_ion_tbl_III[x_int_NXHII][TsNumFilterSteps],
    freq_int_lya_tbl_III[x_int_NXHII][TsNumFilterSteps];
#endif

  bool agn_soft_needed =
    (run_globals.params.physics.Flag_IncludeAGNXray == 1 || run_globals.params.physics.Flag_IncludeAGNXray == 3);
  bool agn_hard_needed =
    (run_globals.params.physics.Flag_IncludeAGNXray == 1 || run_globals.params.physics.Flag_IncludeAGNXray == 2);

  double R_values[TsNumFilterSteps];
  int snapshot_counter_backwards[TsNumFilterSteps];
  double zedge;
#if USE_MINI_HALOS
  double ans[3], dansdz[14];
#else
  double ans[2], dansdz[7];
#endif
  double xHII_call;
  double SFR_GAL[TsNumFilterSteps];
#if USE_STOCHASTICITY
  double XRAY_LUMINOSITY_GAL[TsNumFilterSteps];
#endif
  double XAGN_soft[TsNumFilterSteps];
  double XAGN_hard[TsNumFilterSteps];
#if USE_MINI_HALOS
  double AGN_LW[TsNumFilterSteps];
  double lw_term_stellar, lw_term_III, lw_term_AGN;
  int i_spec;
#endif

#if USE_MINI_HALOS
  double SFR_III[TsNumFilterSteps];
#endif

  float* x_e_box = run_globals.reion_grids.x_e_box;
  float* x_e_box_prev = run_globals.reion_grids.x_e_box_prev;
  float* Tk_box = run_globals.reion_grids.Tk_box;
  float* TS_box = run_globals.reion_grids.TS_box;

#if USE_MINI_HALOS
  float* Tk_boxII = run_globals.reion_grids.Tk_boxII;
  float* TS_boxII = run_globals.reion_grids.TS_boxII;
  float* JLW_box = run_globals.reion_grids.JLW_box;
  float* JLW_boxII = run_globals.reion_grids.JLW_boxII;
#endif

  fftwf_complex* sfr_unfiltered = run_globals.reion_grids.sfr_unfiltered;
  fftwf_complex* sfr_filtered = run_globals.reion_grids.sfr_filtered;
#if USE_STOCHASTICITY
  fftwf_complex* xray_luminosity_unfiltered = run_globals.reion_grids.xray_luminosity_unfiltered;
  fftwf_complex* xray_luminosity_filtered = run_globals.reion_grids.xray_luminosity_filtered;
#endif

  fftwf_complex* BHXrayEmissivity_hard_unfiltered = run_globals.reion_grids.BHXrayEmissivity_hard_unfiltered;
  fftwf_complex* BHXrayEmissivity_hard_filtered = run_globals.reion_grids.BHXrayEmissivity_hard_filtered;
  fftwf_complex* BHXrayEmissivity_soft_unfiltered = run_globals.reion_grids.BHXrayEmissivity_soft_unfiltered;
  fftwf_complex* BHXrayEmissivity_soft_filtered = run_globals.reion_grids.BHXrayEmissivity_soft_filtered;
#if USE_MINI_HALOS
  fftwf_complex* BHUVEmissivity_unfiltered = run_globals.reion_grids.BHUVEmissivity_unfiltered;
  fftwf_complex* BHUVEmissivity_filtered = run_globals.reion_grids.BHUVEmissivity_filtered;
#endif

#if USE_MINI_HALOS
  fftwf_complex* sfrIII_unfiltered = run_globals.reion_grids.sfrIII_unfiltered;
  fftwf_complex* sfrIII_filtered = run_globals.reion_grids.sfrIII_filtered;
#endif

  double* SMOOTHED_SFR_GAL = run_globals.reion_grids.SMOOTHED_SFR_GAL;
#if USE_STOCHASTICITY
  double* SMOOTHED_XRAY_LUMINOSITY_GAL = run_globals.reion_grids.SMOOTHED_XRAY_LUMINOSITY_GAL;
#endif
  double* SMOOTHED_AGN_hard = run_globals.reion_grids.SMOOTHED_AGN_hard;
  double* SMOOTHED_AGN_soft = run_globals.reion_grids.SMOOTHED_AGN_soft;
#if USE_MINI_HALOS
  double* SMOOTHED_AGN_UV = run_globals.reion_grids.SMOOTHED_AGN_UV;
#endif
#if USE_MINI_HALOS
  double* SMOOTHED_SFR_III = run_globals.reion_grids.SMOOTHED_SFR_III;
#endif

  // Initialise the RECFAST, electron rate tables
  init_heat();

  x_e_ave = 0.0;

  double J_alpha_ave, xalpha_ave, Xheat_ave, Xion_ave, J_LW_ave;
  J_alpha_ave = xalpha_ave = Xheat_ave = Xion_ave = J_LW_ave = 0.0;

  double Xheat_ave_AGN_soft = 0.0;
  double Xheat_ave_AGN_hard = 0.0;

#if USE_MINI_HALOS
  double J_alpha_aveII, xalpha_aveII, Xheat_aveII, Xion_aveII, J_LW_aveII, J_LW_ave_AGN;
  J_alpha_aveII = xalpha_aveII = Xheat_aveII = Xion_aveII = J_LW_aveII = J_LW_ave_AGN = 0.0;
#endif

  // Place current redshift in 21cmFAST nomenclature (zp), delta zp (dzp) and delta z in seconds (dt_dzp)
  zp = redshift;
  dzp = zp - prev_redshift;
  dt_dzp = dtdz((float)(float)zp);

  // Check redshift against ReionMaxHeatingRedshift. If zp > ReionMaxHeatingRedshift assume the x_e (electron fraction)
  // and gas temperatures are homogenous Equivalent to the default setup of 21cmFAST.
  if ((zp - run_globals.params.physics.ReionMaxHeatingRedshift) >= -0.0001) {

    float xe = (float)xion_RECFAST((float)zp, 0);
    float TK = (float)T_RECFAST((float)zp, 0);
    float cT_ad = cT_approx(
      (float)zp); // finding the adiabatic index at the initial redshift from 2302.08506 to fix adiabatic fluctuations.

    for (int ix = 0; ix < local_nix; ix++)
      for (int iy = 0; iy < ReionGridDim; iy++)
        for (int iz = 0; iz < ReionGridDim; iz++) {
          i_real = grid_index(ix, iy, iz, ReionGridDim, INDEX_REAL);
          i_padded = grid_index(ix, iy, iz, ReionGridDim, INDEX_PADDED);

          x_e_box_prev[i_padded] = xe;
          Tk_box[i_real] = TK * (1.0 + cT_ad * run_globals.reion_grids.deltax[i_padded]);

          TS_box[i_real] = get_Ts((float)zp,
                                  run_globals.reion_grids.deltax[i_padded],
                                  Tk_box[i_real],
                                  x_e_box_prev[i_padded],
                                  0,
                                  &curr_xalpha);
#if USE_MINI_HALOS
          Tk_boxII[i_real] = Tk_box[i_real];
          TS_boxII[i_real] = TS_box[i_real]; // This is true because Jalpha = 0 so curr_xalpha is the same.
#endif
        }

    // Below I calculate the collapse fraction for all sources.
    // This should be zero (especially for the default high redshift ReionMaxHeatingRedshift = 35). However, I compute
    // it anyway in case it is non-zero. In principle I think this should probably be used instead of
    // ReionMaxHeatingRedshift to switch between homogeneous/inhomogeneous. However, I do not think it'll matter too
    // much. Will look into this later.

    collapse_fraction = 0.;
#if USE_MINI_HALOS
    collapse_fractionIII = 0.;
#endif

    R = L_FACTOR * box_size / (float)ReionGridDim; // Mpc

    for (int ix = 0; ix < local_nix; ix++)
      for (int iy = 0; iy < ReionGridDim; iy++)
        for (int iz = 0; iz < ReionGridDim; iz++) {
          i_padded = grid_index(ix, iy, iz, ReionGridDim, INDEX_PADDED);

          density_over_mean = 1.0 + run_globals.reion_grids.deltax[i_padded];

          // Multiplied by h^2 as RtoM(R) uses RhoCrit which doesn't include h factors, Geil2016
          collapse_fraction_in_cell =
            run_globals.reion_grids.stars[i_padded] /
            (RtoM(R) * run_globals.params.Hubble_h * run_globals.params.Hubble_h * density_over_mean) * (4.0 / 3.0) *
            M_PI * pow(R, 3.0) / pixel_volume;
#if USE_MINI_HALOS
          collapse_fractionIII_in_cell =
            run_globals.reion_grids.starsIII[i_padded] /
            (RtoM(R) * run_globals.params.Hubble_h * run_globals.params.Hubble_h * density_over_mean) * (4.0 / 3.0) *
            M_PI * pow(R, 3.0) / pixel_volume;
#endif

          if (collapse_fraction_in_cell > 1.0) {
            collapse_fraction_in_cell = 1.0;
          }
          collapse_fraction += collapse_fraction_in_cell;

#if USE_MINI_HALOS
          if (collapse_fractionIII_in_cell > 1.0) {
            collapse_fractionIII_in_cell = 1.0;
          }
          collapse_fractionIII += collapse_fractionIII_in_cell;
#endif
        }

    MPI_Allreduce(MPI_IN_PLACE, &collapse_fraction, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
    collapse_fraction = collapse_fraction / total_n_cells;
#if USE_MINI_HALOS
    MPI_Allreduce(MPI_IN_PLACE, &collapse_fractionIII, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
    collapse_fractionIII = collapse_fractionIII / total_n_cells;
#endif

  } else {

    collapse_fraction = 0.;
#if USE_MINI_HALOS
    collapse_fractionIII = 0.;
#endif

    // Setup starting radius (minimum) and scaling to obtaining the maximum filtering radius for the X-ray background
    R = L_FACTOR * box_size / (float)ReionGridDim;
    R_factor = pow(R_XLy_MAX / R, 1 / (float)TsNumFilterSteps);

    // Smooth the density, stars and SFR fields over increasingly larger filtering radii (for evaluating the
    // heating/ionisation integrals)
    snapshot_counter_backwards[0] = 1;
    reion_grids_t* grids = &(run_globals.reion_grids);

    weight = 1;
    for (R_ct = 0; R_ct < TsNumFilterSteps; R_ct++) {

      R_values[R_ct] = R;

      if (R_ct == 0) {
        prev_zpp = zp;
        prev_R = 0;
      } else {
        prev_zpp = zpp_edge[R_ct - 1];
        prev_R = R_values[R_ct - 1];
        snapshot_counter_backwards[R_ct] = snapshot_counter_backwards[R_ct - 1];
      }

      zpp_edge[R_ct] = prev_zpp - (R - prev_R) * MPC / (drdz((float)prev_zpp)); // cell size
      zpp = (zpp_edge[R_ct] + prev_zpp) * 0.5;                // average redshift value of shell: z'' + 0.5 * dz''
      total_weight = (prev_zpp - zpp_edge[R_ct]) * dtdz(zpp); // this is the total weight

      if (snapshot - snapshot_counter_backwards[R_ct] >= 0) {

        if (snapshot - snapshot_counter_backwards[R_ct] == 0 && weight < 0) {
#ifdef DEBUG
          mlog("R_ct = %d, went beyong snapshot 0, clear tocf sfr grids.", MLOG_MESG, R_ct);
#endif
          for (int ii = 0; ii < slab_n_complex * 2; ii++) {
            grids->sfr[ii] = 0;
#if USE_STOCHASTICITY
            grids->xray_luminosity[ii] = 0;
#endif
          }
          snapshot_counter_backwards[R_ct] += 1;
        } else if (zpp_edge[R_ct] > run_globals.ZZ[snapshot - snapshot_counter_backwards[R_ct]]) {
          if (R_ct == 0) {
            weight = dt_dzp * dzp / total_weight; // YQ:I would do  dt_dzp using the mid redshift just like dzpp...
#ifdef DEBUG
            mlog("R_ct = %d, reaching %d snapshots earlier, reweighting tocf sfr grids at snapshot %d with a weight of "
                 "%.2f...",
                 MLOG_OPEN,
                 R_ct,
                 snapshot_counter_backwards[R_ct] - 1,
                 snapshot - snapshot_counter_backwards[R_ct] + 1,
                 weight);
#endif
            for (int ii = 0; ii < slab_n_complex * 2; ii++) {
              grids->sfr[ii] *= weight;
#if USE_STOCHASTICITY
              grids->xray_luminosity[ii] *= weight;
#endif
            }
          } else {
            if (run_globals.ZZ[snapshot - snapshot_counter_backwards[R_ct]] > zpp_edge[R_ct - 1]) {
              weight = -dtdz(0.5 * (run_globals.ZZ[snapshot - snapshot_counter_backwards[R_ct]] + zpp_edge[R_ct - 1]));
              weight *= (run_globals.ZZ[snapshot - snapshot_counter_backwards[R_ct]] - zpp_edge[R_ct - 1]);
              weight /= total_weight;
#ifdef DEBUG
              mlog("R_ct = %d, reaching %d snapshots earlier, reloading tocf sfr grids at snapshot %d with a weight of "
                   "%.2f...",
                   MLOG_OPEN,
                   R_ct,
                   snapshot_counter_backwards[R_ct] - 1,
                   snapshot - snapshot_counter_backwards[R_ct] + 1,
                   weight);
#endif
              load_reion_sfr_grids(snapshot_counter_backwards[R_ct] - 1, weight, 1);
              load_reion_bh_grids(snapshot_counter_backwards[R_ct] - 1, weight, 1);
            } else {
              // shouldn't really happen!
#ifdef DEBUG
              mlog("R_ct = %d, reaching %d snapshots earlier, same as R_ct = %d, clear tocf sfr grids...",
                   MLOG_OPEN,
                   R_ct,
                   snapshot_counter_backwards[R_ct],
                   R_ct - 1);
#endif
              for (int ii = 0; ii < slab_n_complex * 2; ii++) {
                grids->sfr[ii] = 0;
                ;
#if USE_STOCHASTICITY
                grids->xray_luminosity[ii] = 0;
#endif
              }
            }
          }
          while (zpp_edge[R_ct] > run_globals.ZZ[snapshot - snapshot_counter_backwards[R_ct]]) {
            // nothing should happen at snapshot = 0 and this happens at snapshot<~20 for GENESIS
            if (snapshot - snapshot_counter_backwards[R_ct] == 0) {
#ifdef DEBUG
              mlog("going beyong snapshot 0!", MLOG_MESG);
#endif
              weight = -1;
              break;
            }
            if (zpp_edge[R_ct] < run_globals.ZZ[snapshot - snapshot_counter_backwards[R_ct] - 1]) {
              zedge = prev_zpp > run_globals.ZZ[snapshot - snapshot_counter_backwards[R_ct]]
                        ? prev_zpp
                        : run_globals.ZZ[snapshot - snapshot_counter_backwards[R_ct]];
              weight = zpp_edge[R_ct] - zedge;
              weight *= -dtdz(0.5 * (zpp_edge[R_ct] + zedge));
            } else {
              weight = run_globals.ZZ[snapshot - snapshot_counter_backwards[R_ct] - 1] -
                       run_globals.ZZ[snapshot - snapshot_counter_backwards[R_ct]];
              weight *= -dtdz(0.5 * (run_globals.ZZ[snapshot - snapshot_counter_backwards[R_ct] - 1] +
                                     run_globals.ZZ[snapshot - snapshot_counter_backwards[R_ct]]));
            }
            weight /= total_weight;
#ifdef DEBUG
            mlog("       reaching %d snapshots earlier, loading tocf sfr grids at snapshot %d with a weight of %.2f.",
                 MLOG_MESG,
                 snapshot_counter_backwards[R_ct],
                 snapshot - snapshot_counter_backwards[R_ct],
                 weight);
#endif
            load_reion_sfr_grids(snapshot_counter_backwards[R_ct], weight, 0);
            load_reion_bh_grids(snapshot_counter_backwards[R_ct], weight, 0);
            snapshot_counter_backwards[R_ct] += 1;
          }
#ifdef DEBUG
          mlog("...done", MLOG_CLOSE);
#endif
        } else {
          if (R_ct > 0) {
            // This means that we are at higher snapshots than previous filtering radius
            if (weight != 1) {
              weight = 1;
#ifdef DEBUG
              mlog("R_ct = %d, reached %d snapshots earlier, reloading tocf sfr grids at snapshot %d",
                   MLOG_MESG,
                   R_ct,
                   snapshot_counter_backwards[R_ct] - 1,
                   snapshot - snapshot_counter_backwards[R_ct] + 1);
#endif
              load_reion_sfr_grids(snapshot_counter_backwards[R_ct] - 1, weight, 1);
              load_reion_bh_grids(snapshot_counter_backwards[R_ct] - 1, weight, 1);
            }
          }
        }
      }

      fftwf_execute(run_globals.reion_grids.sfr_forward_plan);
      // Remember to add the factor of VOLUME/TOT_NUM_PIXELS when converting from real space to k-space
      // Note: we will leave off factor of VOLUME, in anticipation of the inverse FFT below
      // TODO: Double check that looping over correct number of elements here
      for (int ii = 0; ii < slab_n_complex; ii++)
        sfr_unfiltered[ii] /= (float)total_n_cells;
#if USE_STOCHASTICITY
      fftwf_execute(run_globals.reion_grids.xray_luminosity_forward_plan);
      for (int ii = 0; ii < slab_n_complex; ii++)
        xray_luminosity_unfiltered[ii] /= (float)total_n_cells;
#endif
#if USE_MINI_HALOS
      fftwf_execute(run_globals.reion_grids.sfrIII_forward_plan);
      for (int ii = 0; ii < slab_n_complex; ii++)
        sfrIII_unfiltered[ii] /= (float)total_n_cells;
#endif

      if (agn_hard_needed) {
        fftwf_execute(run_globals.reion_grids.BHXrayEmissivity_hard_forward_plan);
        for (int ii = 0; ii < slab_n_complex; ii++)
          BHXrayEmissivity_hard_unfiltered[ii] /= (float)total_n_cells;
      }
      if (agn_soft_needed) {
        fftwf_execute(run_globals.reion_grids.BHXrayEmissivity_soft_forward_plan);
        for (int ii = 0; ii < slab_n_complex; ii++)
          BHXrayEmissivity_soft_unfiltered[ii] /= (float)total_n_cells;
      }
#if USE_MINI_HALOS
      if (run_globals.params.Flag_IncludeLymanWerner) {
        fftwf_execute(run_globals.reion_grids.BHUVEmissivity_forward_plan);
        for (int ii = 0; ii < slab_n_complex; ii++)
          BHUVEmissivity_unfiltered[ii] /= (float)total_n_cells;
      }
#endif

      memcpy(sfr_filtered, sfr_unfiltered, sizeof(fftwf_complex) * slab_n_complex);
#if USE_STOCHASTICITY
      memcpy(xray_luminosity_filtered, xray_luminosity_unfiltered, sizeof(fftwf_complex) * slab_n_complex);
#endif
#if USE_MINI_HALOS
      memcpy(sfrIII_filtered, sfrIII_unfiltered, sizeof(fftwf_complex) * slab_n_complex);
#endif
      if (agn_hard_needed)
        memcpy(
          BHXrayEmissivity_hard_filtered, BHXrayEmissivity_hard_unfiltered, sizeof(fftwf_complex) * slab_n_complex);
      if (agn_soft_needed)
        memcpy(
          BHXrayEmissivity_soft_filtered, BHXrayEmissivity_soft_unfiltered, sizeof(fftwf_complex) * slab_n_complex);
#if USE_MINI_HALOS
      if (run_globals.params.Flag_IncludeLymanWerner)
        memcpy(BHUVEmissivity_filtered, BHUVEmissivity_unfiltered, sizeof(fftwf_complex) * slab_n_complex);
#endif

      if (R_ct > 0) {
        int local_ix_start = (int)(run_globals.reion_grids.slab_ix_start[run_globals.mpi_rank]);

        filter(sfr_filtered, local_ix_start, local_nix, ReionGridDim, (float)R, run_globals.params.TsHeatingFilterType);
#if USE_STOCHASTICITY
        filter(xray_luminosity_filtered,
               local_ix_start,
               local_nix,
               ReionGridDim,
               (float)R,
               run_globals.params.TsHeatingFilterType);
#endif
#if USE_MINI_HALOS
        filter(
          sfrIII_filtered, local_ix_start, local_nix, ReionGridDim, (float)R, run_globals.params.TsHeatingFilterType);
#endif
        if (agn_hard_needed)
          filter(BHXrayEmissivity_hard_filtered,
                 local_ix_start,
                 local_nix,
                 ReionGridDim,
                 (float)R,
                 run_globals.params.TsHeatingFilterType);
        if (agn_soft_needed)
          filter(BHXrayEmissivity_soft_filtered,
                 local_ix_start,
                 local_nix,
                 ReionGridDim,
                 (float)R,
                 run_globals.params.TsHeatingFilterType);
#if USE_MINI_HALOS
        if (run_globals.params.Flag_IncludeLymanWerner)
          filter(BHUVEmissivity_filtered,
                 local_ix_start,
                 local_nix,
                 ReionGridDim,
                 (float)R,
                 run_globals.params.TsHeatingFilterType);
#endif
      }

      // inverse fourier transform back to real space
      fftwf_execute(run_globals.reion_grids.sfr_filtered_reverse_plan);
#if USE_STOCHASTICITY
      fftwf_execute(run_globals.reion_grids.xray_luminosity_filtered_reverse_plan);
#endif
#if USE_MINI_HALOS
      fftwf_execute(run_globals.reion_grids.sfrIII_filtered_reverse_plan);
#endif
      if (agn_hard_needed)
        fftwf_execute(run_globals.reion_grids.BHXrayEmissivity_hard_filtered_reverse_plan);
      if (agn_soft_needed)
        fftwf_execute(run_globals.reion_grids.BHXrayEmissivity_soft_filtered_reverse_plan);
#if USE_MINI_HALOS
      if (run_globals.params.Flag_IncludeLymanWerner)
        fftwf_execute(run_globals.reion_grids.BHUVEmissivity_filtered_reverse_plan);
#endif

      // Compute and store the collapse fraction and average electron fraction. Necessary for evaluating the integrals
      // back along the light-cone. Need the non-smoothed version, hence this is only done for R_ct == 0.
      if (R_ct == 0) {

        for (int ix = 0; ix < local_nix; ix++)
          for (int iy = 0; iy < ReionGridDim; iy++)
            for (int iz = 0; iz < ReionGridDim; iz++) {
              i_padded = grid_index(ix, iy, iz, ReionGridDim, INDEX_PADDED);
              i_real = grid_index(ix, iy, iz, ReionGridDim, INDEX_REAL);
              i_smoothed_heating = grid_index_smoothedSFR(R_ct, ix, iy, iz, TsNumFilterSteps, ReionGridDim);

              ((float*)sfr_filtered)[i_padded] = fmaxf(((float*)sfr_filtered)[i_padded], 0.0);

              SMOOTHED_SFR_GAL[i_smoothed_heating] = (((float*)sfr_filtered)[i_padded] / pixel_volume) *
                                                     (units->UnitMass_in_g / units->UnitTime_in_s) *
                                                     pow(units->UnitLength_in_cm, -3.) / SOLAR_MASS;
#if USE_STOCHASTICITY
              SMOOTHED_XRAY_LUMINOSITY_GAL[i_smoothed_heating] =
                (((float*)xray_luminosity_filtered)[i_padded] / pixel_volume) * XRAY_LUMINOSITY_UNIT *
                pow(units->UnitLength_in_cm, -3.);

              hmxb_xray_ave += SMOOTHED_XRAY_LUMINOSITY_GAL[i_smoothed_heating];
#else
              // Should include a hmxb_xray_ave branch in USE_STOCHASTICITY? I've added a line but correct it if I am
              // wrong
              hmxb_xray_ave +=
                run_globals.params.physics.LXrayGal * SEC_PER_YEAR * SMOOTHED_SFR_GAL[i_smoothed_heating];
#endif
#if USE_MINI_HALOS
              ((float*)sfrIII_filtered)[i_padded] = fmaxf(((float*)sfrIII_filtered)[i_padded], 0.0);

              SMOOTHED_SFR_III[i_smoothed_heating] = (((float*)sfrIII_filtered)[i_padded] / pixel_volume) *
                                                     (units->UnitMass_in_g / units->UnitTime_in_s) *
                                                     pow(units->UnitLength_in_cm, -3.) / SOLAR_MASS;
#endif

              if (agn_hard_needed) {
                ((float*)BHXrayEmissivity_hard_filtered)[i_padded] =
                  fmaxf(((float*)BHXrayEmissivity_hard_filtered)[i_padded], 0.0);

                bh = ((float*)BHXrayEmissivity_hard_filtered)[i_padded];
                SMOOTHED_AGN_hard[i_smoothed_heating] =
                  (double)bh * 1e10 * SOLAR_LUM / pixel_volume * pow(units->UnitLength_in_cm, -3.0);
                agn_xray_hard_ave += SMOOTHED_AGN_hard[i_smoothed_heating];
              }
              if (agn_soft_needed) {
                ((float*)BHXrayEmissivity_soft_filtered)[i_padded] =
                  fmaxf(((float*)BHXrayEmissivity_soft_filtered)[i_padded], 0.0);

                bh_soft = ((float*)BHXrayEmissivity_soft_filtered)[i_padded];
                SMOOTHED_AGN_soft[i_smoothed_heating] =
                  (double)bh_soft * 1e10 * SOLAR_LUM / pixel_volume * pow(units->UnitLength_in_cm, -3.0);
                agn_xray_soft_ave += SMOOTHED_AGN_soft[i_smoothed_heating];
              }
#if USE_MINI_HALOS
              if (run_globals.params.Flag_IncludeLymanWerner) {
                ((float*)BHUVEmissivity_filtered)[i_padded] = fmaxf(((float*)BHUVEmissivity_filtered)[i_padded], 0.0);

                bh_uv = ((float*)BHUVEmissivity_filtered)[i_padded];
                SMOOTHED_AGN_UV[i_smoothed_heating] = (double)bh_uv * 1e-11 * SOLAR_LUM / NU_1450 / pixel_volume *
                                                      pow(units->UnitLength_in_cm, -3.0); // 1e21 erg/s/Hz/cm^3
              }
#endif

              density_over_mean = 1.0 + run_globals.reion_grids.deltax[i_padded];

              collapse_fraction_in_cell =
                run_globals.reion_grids.stars[i_padded] /
                (RtoM(R) * run_globals.params.Hubble_h * run_globals.params.Hubble_h * density_over_mean) *
                (4.0 / 3.0) * M_PI * pow(R, 3.0) / pixel_volume;
#if USE_MINI_HALOS
              collapse_fractionIII_in_cell =
                run_globals.reion_grids.starsIII[i_padded] /
                (RtoM(R) * run_globals.params.Hubble_h * run_globals.params.Hubble_h * density_over_mean) *
                (4.0 / 3.0) * M_PI * pow(R, 3.0) / pixel_volume;
#endif

              if (collapse_fraction_in_cell > 1.0) {
                collapse_fraction_in_cell = 1.0;
              }
              collapse_fraction += collapse_fraction_in_cell;

#if USE_MINI_HALOS
              if (collapse_fractionIII_in_cell > 1.0) {
                collapse_fractionIII_in_cell = 1.0;
              }
              collapse_fractionIII += collapse_fractionIII_in_cell;
#endif
              x_e_ave += x_e_box_prev[i_padded];
            }

        MPI_Allreduce(MPI_IN_PLACE, &collapse_fraction, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
        MPI_Allreduce(MPI_IN_PLACE, &x_e_ave, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
        MPI_Allreduce(MPI_IN_PLACE, &agn_xray_hard_ave, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
        MPI_Allreduce(MPI_IN_PLACE, &agn_xray_soft_ave, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
        MPI_Allreduce(MPI_IN_PLACE, &hmxb_xray_ave, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);

        collapse_fraction = collapse_fraction / total_n_cells;
        x_e_ave = x_e_ave / total_n_cells;
        agn_xray_hard_ave = agn_xray_hard_ave / total_n_cells;
        agn_xray_soft_ave = agn_xray_soft_ave / total_n_cells;
        hmxb_xray_ave = hmxb_xray_ave / total_n_cells;

        stored_fcoll[snapshot] = collapse_fraction;
        stored_XrayEmissivity_hard[snapshot] = agn_xray_hard_ave;
        stored_XrayEmissivity_soft[snapshot] = agn_xray_soft_ave;
        stored_XrayEmissivity_HMXB[snapshot] = hmxb_xray_ave;

#if USE_MINI_HALOS
        MPI_Allreduce(MPI_IN_PLACE, &collapse_fractionIII, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
        collapse_fractionIII = collapse_fractionIII / total_n_cells;
        stored_fcollIII[snapshot] = collapse_fractionIII;
#endif

      } else {

        // Perform sanity checks to account for aliasing effects
        for (int ix = 0; ix < local_nix; ix++)
          for (int iy = 0; iy < ReionGridDim; iy++)
            for (int iz = 0; iz < ReionGridDim; iz++) {
              i_real = grid_index(ix, iy, iz, ReionGridDim, INDEX_REAL);
              i_padded = grid_index(ix, iy, iz, ReionGridDim, INDEX_PADDED);
              i_smoothed_heating = grid_index_smoothedSFR(R_ct, ix, iy, iz, TsNumFilterSteps, ReionGridDim);

              ((float*)sfr_filtered)[i_padded] = fmaxf(((float*)sfr_filtered)[i_padded], 0.0);
#if USE_MINI_HALOS
              ((float*)sfrIII_filtered)[i_padded] = fmaxf(((float*)sfrIII_filtered)[i_padded], 0.0);
#endif

              SMOOTHED_SFR_GAL[i_smoothed_heating] = (((float*)sfr_filtered)[i_padded] / pixel_volume) *
                                                     (units->UnitMass_in_g / units->UnitTime_in_s) *
                                                     pow(units->UnitLength_in_cm, -3.) / SOLAR_MASS;
#if USE_STOCHASTICITY
              SMOOTHED_XRAY_LUMINOSITY_GAL[i_smoothed_heating] =
                (((float*)xray_luminosity_filtered)[i_padded] / pixel_volume) * XRAY_LUMINOSITY_UNIT *
                pow(units->UnitLength_in_cm, -3.);
#endif
#if USE_MINI_HALOS
              SMOOTHED_SFR_III[i_smoothed_heating] = (((float*)sfrIII_filtered)[i_padded] / pixel_volume) *
                                                     (units->UnitMass_in_g / units->UnitTime_in_s) *
                                                     pow(units->UnitLength_in_cm, -3.) / SOLAR_MASS;
#endif

              if (agn_hard_needed) {
                ((float*)BHXrayEmissivity_hard_filtered)[i_padded] =
                  fmaxf(((float*)BHXrayEmissivity_hard_filtered)[i_padded], 0.0);

                bh = ((float*)BHXrayEmissivity_hard_filtered)[i_padded];
                SMOOTHED_AGN_hard[i_smoothed_heating] =
                  (double)bh * 1e10 * SOLAR_LUM / pixel_volume * pow(units->UnitLength_in_cm, -3.0);
              }
              if (agn_soft_needed) {
                ((float*)BHXrayEmissivity_soft_filtered)[i_padded] =
                  fmaxf(((float*)BHXrayEmissivity_soft_filtered)[i_padded], 0.0);

                bh_soft = ((float*)BHXrayEmissivity_soft_filtered)[i_padded];
                SMOOTHED_AGN_soft[i_smoothed_heating] =
                  (double)bh_soft * 1e10 * SOLAR_LUM / pixel_volume * pow(units->UnitLength_in_cm, -3.0);
              }
#if USE_MINI_HALOS
              if (run_globals.params.Flag_IncludeLymanWerner) {
                ((float*)BHUVEmissivity_filtered)[i_padded] = fmaxf(((float*)BHUVEmissivity_filtered)[i_padded], 0.0);

                bh_uv = ((float*)BHUVEmissivity_filtered)[i_padded];
                SMOOTHED_AGN_UV[i_smoothed_heating] = (double)bh_uv * 1e-11 * SOLAR_LUM / NU_1450 / pixel_volume *
                                                      pow(units->UnitLength_in_cm, -3.0); // 1e21 erg/s/Hz/cm^3
              }
#endif
            }
      }

      R *= R_factor;
    }

    // A condition (defined by whether or not there are stars) for evaluating the heating/ionisation integrals
#if USE_MINI_HALOS
    if ((collapse_fraction + collapse_fractionIII) > 0.0)
#else
    if (collapse_fraction > 0.0)
#endif
    {
      NO_LIGHT = 0;
    } else {
      NO_LIGHT = 1;
    }

    // Populate the initial ionisation/heating tables
    for (R_ct = 0; R_ct < TsNumFilterSteps; R_ct++) {

      if (R_ct == 0) {
        prev_zpp = zp;
        prev_R = 0;
      } else {
        prev_zpp = zpp_edge[R_ct - 1];
        prev_R = R_values[R_ct - 1];
      }
      zpp = (zpp_edge[R_ct] + prev_zpp) * 0.5; // average redshift value of shell: z'' + 0.5 * dz''

#if USE_MINI_HALOS
      filling_factor_of_HI_zp =
        1. - (ReionEfficiency * collapse_fraction + ReionEfficiencyIII * collapse_fractionIII) / (1.0 - x_e_ave);
#else
      filling_factor_of_HI_zp = 1. - ReionEfficiency * collapse_fraction / (1.0 - x_e_ave);
#endif

      nu_tau_one_zpp = nu_tau_one(zp, zpp, x_e_ave, filling_factor_of_HI_zp, snapshot);
      lower_int_limit_GAL = fmax(nu_tau_one_zpp, run_globals.params.physics.NuXrayThreshold * NU_over_EV);
      upper_int_limit_GAL = run_globals.params.physics.NuXrayMax * NU_over_EV;

      if (filling_factor_of_HI_zp < 0)
        filling_factor_of_HI_zp =
          0; // for global evol; nu_tau_one above treats negative (post_reionization) inferred filling factors properly

      for (x_e_ct = 0; x_e_ct < x_int_NXHII; x_e_ct++) {
        freq_int_heat_tbl_GAL[x_e_ct][R_ct] = integrate_over_nu(zp,
                                                                x_int_XHII[x_e_ct],
                                                                lower_int_limit_GAL,
                                                                upper_int_limit_GAL,
                                                                run_globals.params.physics.NuXrayThreshold,
                                                                run_globals.params.physics.SpecIndexXrayGal,
                                                                0);
        freq_int_ion_tbl_GAL[x_e_ct][R_ct] = integrate_over_nu(zp,
                                                               x_int_XHII[x_e_ct],
                                                               lower_int_limit_GAL,
                                                               upper_int_limit_GAL,
                                                               run_globals.params.physics.NuXrayThreshold,
                                                               run_globals.params.physics.SpecIndexXrayGal,
                                                               1);
        freq_int_lya_tbl_GAL[x_e_ct][R_ct] = integrate_over_nu(zp,
                                                               x_int_XHII[x_e_ct],
                                                               lower_int_limit_GAL,
                                                               upper_int_limit_GAL,
                                                               run_globals.params.physics.NuXrayThreshold,
                                                               run_globals.params.physics.SpecIndexXrayGal,
                                                               2);

#if USE_MINI_HALOS
        freq_int_heat_tbl_III[x_e_ct][R_ct] = integrate_over_nu(zp,
                                                                x_int_XHII[x_e_ct],
                                                                lower_int_limit_GAL,
                                                                upper_int_limit_GAL,
                                                                run_globals.params.physics.NuXrayThreshold,
                                                                run_globals.params.physics.SpecIndexXrayIII,
                                                                0);
        freq_int_ion_tbl_III[x_e_ct][R_ct] = integrate_over_nu(zp,
                                                               x_int_XHII[x_e_ct],
                                                               lower_int_limit_GAL,
                                                               upper_int_limit_GAL,
                                                               run_globals.params.physics.NuXrayThreshold,
                                                               run_globals.params.physics.SpecIndexXrayIII,
                                                               1);
        freq_int_lya_tbl_III[x_e_ct][R_ct] = integrate_over_nu(zp,
                                                               x_int_XHII[x_e_ct],
                                                               lower_int_limit_GAL,
                                                               upper_int_limit_GAL,
                                                               run_globals.params.physics.NuXrayThreshold,
                                                               run_globals.params.physics.SpecIndexXrayIII,
                                                               2);
#endif
      }

      /* K_b(zp,zpp,x_e) = Int_{nu_lo(zp,zpp)}^{nu_hi(zp,zpp)} f_dep(nu,x_e) * (nu/nu_th)^(-alpha_b-1) dnu */
      if (agn_soft_needed) {
        lower_int_limit_AGN_soft =
          fmax(nu_tau_one_zpp, run_globals.params.physics.NuXrayThreshold * NU_over_EV * (1. + zp) / (1. + zpp));
        upper_int_limit_AGN_soft = run_globals.params.physics.NuXraySoftCut * NU_over_EV * (1. + zp) / (1. + zpp);

        for (x_e_ct = 0; x_e_ct < x_int_NXHII; x_e_ct++) {
          freq_int_heat_tbl_AGN_soft[x_e_ct][R_ct] = integrate_over_nu(zp,
                                                                       x_int_XHII[x_e_ct],
                                                                       lower_int_limit_AGN_soft,
                                                                       upper_int_limit_AGN_soft,
                                                                       run_globals.params.physics.NuXrayThreshold,
                                                                       run_globals.params.physics.SpecIndexXrayAGNSoft,
                                                                       0);

          freq_int_ion_tbl_AGN_soft[x_e_ct][R_ct] = integrate_over_nu(zp,
                                                                      x_int_XHII[x_e_ct],
                                                                      lower_int_limit_AGN_soft,
                                                                      upper_int_limit_AGN_soft,
                                                                      run_globals.params.physics.NuXrayThreshold,
                                                                      run_globals.params.physics.SpecIndexXrayAGNSoft,
                                                                      1);

          freq_int_lya_tbl_AGN_soft[x_e_ct][R_ct] = integrate_over_nu(zp,
                                                                      x_int_XHII[x_e_ct],
                                                                      lower_int_limit_AGN_soft,
                                                                      upper_int_limit_AGN_soft,
                                                                      run_globals.params.physics.NuXrayThreshold,
                                                                      run_globals.params.physics.SpecIndexXrayAGNSoft,
                                                                      2);
        }
      }

      if (agn_hard_needed) {
        lower_int_limit_AGN_hard =
          fmax(nu_tau_one_zpp, run_globals.params.physics.NuXraySoftCut * NU_over_EV * (1. + zp) / (1. + zpp));
        upper_int_limit_AGN_hard = run_globals.params.physics.NuXrayMax * NU_over_EV * (1. + zp) / (1. + zpp);

        for (x_e_ct = 0; x_e_ct < x_int_NXHII; x_e_ct++) {
          freq_int_heat_tbl_AGN_hard[x_e_ct][R_ct] = integrate_over_nu(zp,
                                                                       x_int_XHII[x_e_ct],
                                                                       lower_int_limit_AGN_hard,
                                                                       upper_int_limit_AGN_hard,
                                                                       run_globals.params.physics.NuXrayThreshold,
                                                                       run_globals.params.physics.SpecIndexXrayAGNHard,
                                                                       0);

          freq_int_ion_tbl_AGN_hard[x_e_ct][R_ct] = integrate_over_nu(zp,
                                                                      x_int_XHII[x_e_ct],
                                                                      lower_int_limit_AGN_hard,
                                                                      upper_int_limit_AGN_hard,
                                                                      run_globals.params.physics.NuXrayThreshold,
                                                                      run_globals.params.physics.SpecIndexXrayAGNHard,
                                                                      1);

          freq_int_lya_tbl_AGN_hard[x_e_ct][R_ct] = integrate_over_nu(zp,
                                                                      x_int_XHII[x_e_ct],
                                                                      lower_int_limit_AGN_hard,
                                                                      upper_int_limit_AGN_hard,
                                                                      run_globals.params.physics.NuXrayThreshold,
                                                                      run_globals.params.physics.SpecIndexXrayAGNHard,
                                                                      2);
        }
      }

      // and create the sum over Lya transitions from direct Lyn flux
      sum_lyn[R_ct] = 0;
#if USE_MINI_HALOS
      sum_lyn_III[R_ct] = 0;
      if (run_globals.params.Flag_IncludeLymanWerner) {
        sum_lyn_LW[R_ct] = 0;
        sum_lyn_LW_III[R_ct] = 0;
        sum_lyn_LW_AGN[R_ct] = 0;
        // per-Lyman-level breakdown, so the LW spectral shape is recoverable from the output
        for (i_spec = 0; i_spec < LW_NLEV; i_spec++) {
          LW_spectral_stellar[R_ct * LW_NLEV + i_spec] = 0;
          LW_spectral_III[R_ct * LW_NLEV + i_spec] = 0;
          LW_spectral_AGN[R_ct * LW_NLEV + i_spec] = 0;
        }
        LW_zpp[R_ct] = zpp;
      }
#endif

      for (n_ct = NSPEC_MAX; n_ct >= 2; n_ct--) {
        if (zpp > zmax((float)zp, n_ct))
          continue;

        nuprime = nu_n(n_ct) * (1 + zpp) / (1.0 + zp);
        sum_lyn[R_ct] += frecycle(n_ct) * spectral_emissivity(nuprime, 0, 2);
#if USE_MINI_HALOS
        sum_lyn_III[R_ct] += frecycle(n_ct) * spectral_emissivity(nuprime, 0, 3);
        if (run_globals.params.Flag_IncludeLymanWerner) {
          if (nuprime < NU_LW / NU_LA)
            nuprime = NU_LW / NU_LA;
          if (nuprime > nu_n(n_ct + 1))
            continue;
          // photons per stellar baryon
          lw_term_stellar = spectral_emissivity(nuprime, 2, 2);
          lw_term_III = spectral_emissivity(nuprime, 2, 3);

          if (fabs(run_globals.params.physics.SpecIndexUVAGNSoft - 1.0) < REL_TOL) {
            lw_term_AGN = log(nu_n(n_ct + 1) / nuprime); // unitless
          } else {
            lw_term_AGN = (pow(nu_n(n_ct + 1) * NU_LA / NU_1450, 1 - run_globals.params.physics.SpecIndexUVAGNSoft) -
                           pow(nuprime * NU_LA / NU_1450, 1 - run_globals.params.physics.SpecIndexUVAGNSoft)) /
                          (1 - run_globals.params.physics.SpecIndexUVAGNSoft);
          }

          sum_lyn_LW[R_ct] += lw_term_stellar;
          sum_lyn_LW_III[R_ct] += lw_term_III;
          sum_lyn_LW_AGN[R_ct] += lw_term_AGN;

          // keep the individual level so the sawtooth can be rebuilt; summing over n_ct reproduces sum_lyn_LW*
          LW_spectral_stellar[R_ct * LW_NLEV + n_ct] = lw_term_stellar;
          LW_spectral_III[R_ct * LW_NLEV + n_ct] = lw_term_III;
          LW_spectral_AGN[R_ct * LW_NLEV + n_ct] = lw_term_AGN;
        }

#endif
      }

      // Find if we need to add a partial contribution to a radii to avoid kinks in the Lyman-alpha flux
      // As we look at discrete radii (light-cone redshift, zpp) we can have two radii where one has a
      // contribution and the next (larger) radii has no contribution. However, if the number of filtering
      // steps were infinitely large, we would have contributions between these two discrete radii
      // Thus, this aims to add a weighted contribution to the first radii where this occurs to smooth out
      // kinks in the average Lyman-alpha flux.

      // Note: We do not apply this correction to the LW background as it is unaffected by this. It is only
      // the Lyn contribution that experiences the kink. Applying this correction to LW introduces kinks
      // into the otherwise smooth quantity
      if (R_ct > 2 && sum_lyn[R_ct] == 0.0 && sum_lyn[R_ct - 1] > 0. && first_radii) {

        // The current zpp for which we are getting zero contribution
        double trial_zpp_max = (prev_zpp - (R_values[R_ct] - prev_R) * MPC / drdz((float)prev_zpp) + prev_zpp) * 0.5;
        // The zpp for the previous radius for which we had a non-zero contribution
        double trial_zpp_min =
          (zpp_edge[R_ct - 2] - (R_values[R_ct - 1] - R_values[R_ct - 2]) * MPC / drdz((float)zpp_edge[R_ct - 2]) +
           zpp_edge[R_ct - 2]) *
          0.5;

        // Split the previous radii and current radii into n_pts_radii smaller radii (redshift) to have fine control of
        // where it transitions from zero to non-zero This is a coarse approximation as it assumes that the linear
        // sampling is a good representation of the different volumes of the shells (from different radii).
        for (int ii = 0; ii < n_pts_radii; ii++) {
          double trial_zpp = trial_zpp_min + (trial_zpp_max - trial_zpp_min) * (float)ii / ((float)n_pts_radii - 1.);

          int counter = 0;
          for (n_ct = NSPEC_MAX; n_ct >= 2; n_ct--) {
            if (trial_zpp > zmax(zp, n_ct))
              continue;

            counter += 1;
          }
          if (counter == 0 && first_zero) {
            first_zero = false;
            weight = (float)ii / (float)n_pts_radii;
          }
        }

        // Now add a non-zero contribution to the previously zero contribution
        // The amount is the weight, multplied by the contribution from the previous radii
        sum_lyn[R_ct] = weight * sum_lyn[R_ct - 1];
#if USE_MINI_HALOS
        sum_lyn_III[R_ct] = weight * sum_lyn_III[R_ct - 1]; // I am not really sure about this line!
        if (run_globals.params.Flag_IncludeLymanWerner) {
          sum_lyn_LW[R_ct] = weight * sum_lyn_LW[R_ct - 1];
          sum_lyn_LW_AGN[R_ct] = weight * sum_lyn_LW_AGN[R_ct - 1];
          // weight the per-level breakdown too, so it still sums to sum_lyn_LW* for this shell
          for (i_spec = 0; i_spec < LW_NLEV; i_spec++) {
            LW_spectral_stellar[R_ct * LW_NLEV + i_spec] = weight * LW_spectral_stellar[(R_ct - 1) * LW_NLEV + i_spec];
            LW_spectral_AGN[R_ct * LW_NLEV + i_spec] = weight * LW_spectral_AGN[(R_ct - 1) * LW_NLEV + i_spec];
          }
        }
#endif
        first_radii = false;
      }
    }

    growth_factor_zp = dicke(zp);
    dgrowth_factor_dzp = ddicke_dz(zp);
    dt_dzp = dtdz((float)zp);

    // Below is the converstion of the soft-band X_ray luminosity into number of X-ray photons produced. This is the
    // code taken from 21CMMC, which somewhat uses the 21cmFAST nomenclature (to ease flipping between old/new
    // parameterisation), so isn't necessarily the most intuitive way to express this.

    // Conversion of the input bolometric luminosity (new) to a ZETA_X (old) to be consistent with Ts.c from 21cmFAST
    // Conversion here means the code otherwise remains the same as the original Ts.c

    /*
     * L(nu) = L0*(nu/nu_th)^(-alpha) on [nu_lo,nu_hi], nu_th = NuXrayThreshold.
     * C = L0/L_band:
     *   alpha != 1:  C = (1-alpha) * nu_th^-alpha / [nu_hi^(1-alpha) - nu_lo^(1-alpha)]
     *   alpha == 1:  C = 1 / (nu_th * ln(nu_hi/nu_lo))
     * [nu_lo,nu_hi]: GAL/III/AGN-soft -> [NuXrayThreshold,NuXraySoftCut], AGN-hard -> [NuXraySoftCut,NuXrayMax].
     */
    if (fabs(run_globals.params.physics.SpecIndexXrayGal - 1.0) < REL_TOL) {
      Luminosity_converstion_factor_GAL =
        (run_globals.params.physics.NuXrayThreshold * NU_over_EV) *
        log(run_globals.params.physics.NuXraySoftCut / run_globals.params.physics.NuXrayThreshold);
      Luminosity_converstion_factor_GAL = 1. / Luminosity_converstion_factor_GAL;
    } else {
      Luminosity_converstion_factor_GAL =
        pow(run_globals.params.physics.NuXraySoftCut * NU_over_EV, 1. - run_globals.params.physics.SpecIndexXrayGal) -
        pow(run_globals.params.physics.NuXrayThreshold * NU_over_EV, 1. - run_globals.params.physics.SpecIndexXrayGal);
      Luminosity_converstion_factor_GAL = 1. / Luminosity_converstion_factor_GAL;
      Luminosity_converstion_factor_GAL *=
        pow(run_globals.params.physics.NuXrayThreshold * NU_over_EV, -run_globals.params.physics.SpecIndexXrayGal) *
        (1 - run_globals.params.physics.SpecIndexXrayGal);
    }
    Luminosity_converstion_factor_GAL *= 1.0 / PLANCK;
    // Finally, convert to the correct units. NU_over_EV*hplank as only want to divide by eV -> erg (owing to the
    // definition of Luminosity)
    if (fabs(run_globals.params.physics.SpecIndexXrayAGNSoft - 1.0) < REL_TOL) {
      Luminosity_converstion_factor_AGN_soft =
        (run_globals.params.physics.NuXrayThreshold * NU_over_EV) *
        log(run_globals.params.physics.NuXraySoftCut / run_globals.params.physics.NuXrayThreshold);
      Luminosity_converstion_factor_AGN_soft = 1.0 / Luminosity_converstion_factor_AGN_soft;
    } else {
      Luminosity_converstion_factor_AGN_soft = pow(run_globals.params.physics.NuXraySoftCut * NU_over_EV,
                                                   1.0 - run_globals.params.physics.SpecIndexXrayAGNSoft) -
                                               pow(run_globals.params.physics.NuXrayThreshold * NU_over_EV,
                                                   1.0 - run_globals.params.physics.SpecIndexXrayAGNSoft);
      Luminosity_converstion_factor_AGN_soft = 1.0 / Luminosity_converstion_factor_AGN_soft;
      Luminosity_converstion_factor_AGN_soft *=
        pow(run_globals.params.physics.NuXrayThreshold * NU_over_EV, -run_globals.params.physics.SpecIndexXrayAGNSoft) *
        (1.0 - run_globals.params.physics.SpecIndexXrayAGNSoft);
    }
    Luminosity_converstion_factor_AGN_soft /= (PLANCK);

    if (fabs(run_globals.params.physics.SpecIndexXrayAGNHard - 1.0) < REL_TOL) {
      Luminosity_converstion_factor_AGN_hard =
        (run_globals.params.physics.NuXrayThreshold * NU_over_EV) *
        log(run_globals.params.physics.NuXrayMax / run_globals.params.physics.NuXraySoftCut);
      Luminosity_converstion_factor_AGN_hard = 1.0 / Luminosity_converstion_factor_AGN_hard;
    } else {
      Luminosity_converstion_factor_AGN_hard =
        pow(run_globals.params.physics.NuXrayMax * NU_over_EV, 1.0 - run_globals.params.physics.SpecIndexXrayAGNHard) -
        pow(run_globals.params.physics.NuXraySoftCut * NU_over_EV,
            1.0 - run_globals.params.physics.SpecIndexXrayAGNHard);
      Luminosity_converstion_factor_AGN_hard = 1.0 / Luminosity_converstion_factor_AGN_hard;
      Luminosity_converstion_factor_AGN_hard *=
        pow(run_globals.params.physics.NuXrayThreshold * NU_over_EV, -run_globals.params.physics.SpecIndexXrayAGNHard) *
        (1.0 - run_globals.params.physics.SpecIndexXrayAGNHard);
    }
    Luminosity_converstion_factor_AGN_hard /= (PLANCK);

    // Do the same for Pop III.

#if USE_MINI_HALOS
    if (fabs(run_globals.params.physics.SpecIndexXrayIII - 1.0) < REL_TOL) {
      Luminosity_converstion_factor_III =
        (run_globals.params.physics.NuXrayThreshold * NU_over_EV) *
        log(run_globals.params.physics.NuXraySoftCut / run_globals.params.physics.NuXrayThreshold);
      Luminosity_converstion_factor_III = 1. / Luminosity_converstion_factor_III;
    } else {
      Luminosity_converstion_factor_III =
        pow(run_globals.params.physics.NuXraySoftCut * NU_over_EV, 1. - run_globals.params.physics.SpecIndexXrayIII) -
        pow(run_globals.params.physics.NuXrayThreshold * NU_over_EV, 1. - run_globals.params.physics.SpecIndexXrayIII);
      Luminosity_converstion_factor_III = 1. / Luminosity_converstion_factor_III;
      Luminosity_converstion_factor_III *=
        pow(run_globals.params.physics.NuXrayThreshold * NU_over_EV, -run_globals.params.physics.SpecIndexXrayIII) *
        (1 - run_globals.params.physics.SpecIndexXrayIII);
    }
    Luminosity_converstion_factor_III *= (SEC_PER_YEAR) / (PLANCK);
#endif

    // Leave the original 21cmFAST code for reference. Refer to Greig & Mesinger (2017) for the new parameterisation.
    //        const_zp_prefactor_GAL = (1.0/0.59)*( run_globals.params.physics.LXrayGal *
    //        Luminosity_converstion_factor_GAL ) / (run_globals.params.physics.NuXrayThreshold*NU_over_EV) *
    //        SPEED_OF_LIGHT * pow(1+zp, run_globals.params.physics.SpecIndexXrayGal+3);
#if USE_STOCHASTICITY
    const_zp_prefactor_GAL = Luminosity_converstion_factor_GAL /
                             (run_globals.params.physics.NuXrayThreshold * NU_over_EV) * SPEED_OF_LIGHT *
                             pow(1 + zp, run_globals.params.physics.SpecIndexXrayGal + 3);
#else
    const_zp_prefactor_GAL = (run_globals.params.physics.LXrayGal * SEC_PER_YEAR * Luminosity_converstion_factor_GAL) /
                             (run_globals.params.physics.NuXrayThreshold * NU_over_EV) * SPEED_OF_LIGHT *
                             pow(1 + zp, run_globals.params.physics.SpecIndexXrayGal + 3);
#endif
#if USE_MINI_HALOS
    const_zp_prefactor_III = (run_globals.params.physics.LXrayGalIII * Luminosity_converstion_factor_III) /
                             (run_globals.params.physics.NuXrayThreshold * NU_over_EV) * SPEED_OF_LIGHT *
                             pow(1 + zp, run_globals.params.physics.SpecIndexXrayIII + 3);
#endif

    /* A_b(zp) = (C_b / nu_th) * c * (1+zp)^(alpha_b + 3) */
    const_zp_prefactor_AGN_soft = Luminosity_converstion_factor_AGN_soft /
                                  (run_globals.params.physics.NuXrayThreshold * NU_over_EV) * SPEED_OF_LIGHT *
                                  pow(1.0 + zp, run_globals.params.physics.SpecIndexXrayAGNSoft + 3.0);
    if (!isfinite(const_zp_prefactor_AGN_soft)) {
      mlog_error(
        "const_zp_prefactor_AGN_soft is not finite — check NuXrayThreshold/NuXraySoftCut/SpecIndexXrayAGNSoft.");
      ABORT(EXIT_FAILURE);
    }

    const_zp_prefactor_AGN_hard = Luminosity_converstion_factor_AGN_hard /
                                  (run_globals.params.physics.NuXrayThreshold * NU_over_EV) * SPEED_OF_LIGHT *
                                  pow(1.0 + zp, run_globals.params.physics.SpecIndexXrayAGNHard + 3.0);
    if (!isfinite(const_zp_prefactor_AGN_hard)) {
      mlog_error("const_zp_prefactor_AGN_hard is not finite — check NuXrayThreshold/NuXrayMax/SpecIndexXrayAGNHard.");
      ABORT(EXIT_FAILURE);
    }
    // Note the factor of 0.59 appears to be required to match 21cmFAST

    // I believe it arises from differing definitions of a stellar baryon mass
    // 21cmFAST appears to define a stellar baryon as 0.59*m_p, whereas Meraxes defines it as m_p
    // Had issues with normalisation factors comparing the codes, adding 0.59 here rectified the normalisation somewhat
    // The Lya background appeared to be a factor of ~ 2 higher than 21cmFAST for the same luminosity. Spent days
    // searching for the difference, this seems to explain it. Can either boost the luminosity conversion, or lower the
    // lya background by the same factor in XRayHeatingFunctions.c (evolveInt). Note: When comparing to 21cmFAST it is
    // important that this factor is included! Will mean the normalisation within Meraxes is (1/0.59) higher than
    // 21cmFAST, which can be trivially compensated for by reducing L_X. Ultimately the backgrounds in Meraxes will be
    // this same factor higher than 21cmFAST, but at least it is understood why and trivially accounted for.

#if USE_MINI_HALOS
    // Box-average the LW source emissivity per shell: LW_emissivity_X * LW_spectral_X is an absolute flux, not just a
    // survival fraction.
    if (run_globals.params.Flag_IncludeLymanWerner) {
      for (R_ct = 0; R_ct < TsNumFilterSteps; R_ct++) {
        double sum_gal = 0.0, sum_III = 0.0, sum_agn = 0.0;
        for (int ix = 0; ix < local_nix; ix++)
          for (int iy = 0; iy < ReionGridDim; iy++)
            for (int iz = 0; iz < ReionGridDim; iz++) {
              int i_sm = grid_index_smoothedSFR(R_ct, ix, iy, iz, TsNumFilterSteps, ReionGridDim);
              sum_gal += SMOOTHED_SFR_GAL[i_sm];
              sum_III += SMOOTHED_SFR_III[i_sm];
              sum_agn += SMOOTHED_AGN_UV[i_sm];
            }
        MPI_Allreduce(MPI_IN_PLACE, &sum_gal, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
        MPI_Allreduce(MPI_IN_PLACE, &sum_III, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
        MPI_Allreduce(MPI_IN_PLACE, &sum_agn, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
        LW_emissivity_stellar[R_ct] = sum_gal / total_n_cells;
        LW_emissivity_III[R_ct] = sum_III / total_n_cells;
        // carry AGNLWEfficiency so 0 zeroes the AGN LW output, matching AGN_LW[] below
        LW_emissivity_AGN[R_ct] = run_globals.params.physics.AGNLWEfficiency * sum_agn / total_n_cells;
      }
    }
#endif

    // interpolate to correct nu integral value based on the cell's ionization state
    for (int ix = 0; ix < local_nix; ix++)
      for (int iy = 0; iy < ReionGridDim; iy++)
        for (int iz = 0; iz < ReionGridDim; iz++) {
          i_real = grid_index(ix, iy, iz, ReionGridDim, INDEX_REAL);
          i_padded = grid_index(ix, iy, iz, ReionGridDim, INDEX_PADDED);

          ans[0] = x_e_box_prev[i_padded];
          ans[1] = Tk_box[i_real];
#if USE_MINI_HALOS
          ans[2] = Tk_boxII[i_real];
#endif

          for (R_ct = 0; R_ct < TsNumFilterSteps; R_ct++) {
            i_smoothed_heating = grid_index_smoothedSFR(R_ct, ix, iy, iz, TsNumFilterSteps, ReionGridDim);

            SFR_GAL[R_ct] = SMOOTHED_SFR_GAL[i_smoothed_heating];
#if USE_STOCHASTICITY
            XRAY_LUMINOSITY_GAL[R_ct] = SMOOTHED_XRAY_LUMINOSITY_GAL[i_smoothed_heating];
#endif
            {
              XAGN_soft[R_ct] = agn_soft_needed ? SMOOTHED_AGN_soft[i_smoothed_heating] : 0.0;
              XAGN_hard[R_ct] = agn_hard_needed ? SMOOTHED_AGN_hard[i_smoothed_heating] : 0.0;
            }
#if USE_MINI_HALOS
            SFR_III[R_ct] = SMOOTHED_SFR_III[i_smoothed_heating];
            AGN_LW[R_ct] = run_globals.params.Flag_IncludeLymanWerner
                             ? run_globals.params.physics.AGNLWEfficiency * SMOOTHED_AGN_UV[i_smoothed_heating]
                             : 0.0;
#endif
            xHII_call = x_e_box_prev[i_padded];

            // Check if ionized fraction is within boundaries; if not, adjust to be within
            if (xHII_call > x_int_XHII[x_int_NXHII - 1] * 0.999) {
              xHII_call = x_int_XHII[x_int_NXHII - 1] * 0.999;
            } else if (xHII_call < x_int_XHII[0]) {
              xHII_call = 1.001 * x_int_XHII[0];
            }

            m_xHII_low = locate_xHII_index((float)xHII_call);
            m_xHII_high = m_xHII_low + 1;

            // heat
            freq_int_heat_GAL[R_ct] =
              (freq_int_heat_tbl_GAL[m_xHII_high][R_ct] - freq_int_heat_tbl_GAL[m_xHII_low][R_ct]) /
              (x_int_XHII[m_xHII_high] - x_int_XHII[m_xHII_low]);
            freq_int_heat_GAL[R_ct] *= (xHII_call - x_int_XHII[m_xHII_low]);
            freq_int_heat_GAL[R_ct] += freq_int_heat_tbl_GAL[m_xHII_low][R_ct];

            // ionization
            freq_int_ion_GAL[R_ct] =
              (freq_int_ion_tbl_GAL[m_xHII_high][R_ct] - freq_int_ion_tbl_GAL[m_xHII_low][R_ct]) /
              (x_int_XHII[m_xHII_high] - x_int_XHII[m_xHII_low]);
            freq_int_ion_GAL[R_ct] *= (xHII_call - x_int_XHII[m_xHII_low]);
            freq_int_ion_GAL[R_ct] += freq_int_ion_tbl_GAL[m_xHII_low][R_ct];

            // lya
            freq_int_lya_GAL[R_ct] =
              (freq_int_lya_tbl_GAL[m_xHII_high][R_ct] - freq_int_lya_tbl_GAL[m_xHII_low][R_ct]) /
              (x_int_XHII[m_xHII_high] - x_int_XHII[m_xHII_low]);
            freq_int_lya_GAL[R_ct] *= (xHII_call - x_int_XHII[m_xHII_low]);
            freq_int_lya_GAL[R_ct] += freq_int_lya_tbl_GAL[m_xHII_low][R_ct];

#if USE_MINI_HALOS
            freq_int_heat_III[R_ct] =
              (freq_int_heat_tbl_III[m_xHII_high][R_ct] - freq_int_heat_tbl_III[m_xHII_low][R_ct]) /
              (x_int_XHII[m_xHII_high] - x_int_XHII[m_xHII_low]);
            freq_int_heat_III[R_ct] *= (xHII_call - x_int_XHII[m_xHII_low]);
            freq_int_heat_III[R_ct] += freq_int_heat_tbl_III[m_xHII_low][R_ct];

            // ionization
            freq_int_ion_III[R_ct] =
              (freq_int_ion_tbl_III[m_xHII_high][R_ct] - freq_int_ion_tbl_III[m_xHII_low][R_ct]) /
              (x_int_XHII[m_xHII_high] - x_int_XHII[m_xHII_low]);
            freq_int_ion_III[R_ct] *= (xHII_call - x_int_XHII[m_xHII_low]);
            freq_int_ion_III[R_ct] += freq_int_ion_tbl_III[m_xHII_low][R_ct];

            // lya
            freq_int_lya_III[R_ct] =
              (freq_int_lya_tbl_III[m_xHII_high][R_ct] - freq_int_lya_tbl_III[m_xHII_low][R_ct]) /
              (x_int_XHII[m_xHII_high] - x_int_XHII[m_xHII_low]);
            freq_int_lya_III[R_ct] *= (xHII_call - x_int_XHII[m_xHII_low]);
            freq_int_lya_III[R_ct] += freq_int_lya_tbl_III[m_xHII_low][R_ct];
#endif

            if (agn_soft_needed) {
              freq_int_heat_AGN_soft[R_ct] =
                (freq_int_heat_tbl_AGN_soft[m_xHII_high][R_ct] - freq_int_heat_tbl_AGN_soft[m_xHII_low][R_ct]) /
                (x_int_XHII[m_xHII_high] - x_int_XHII[m_xHII_low]);
              freq_int_heat_AGN_soft[R_ct] *= (xHII_call - x_int_XHII[m_xHII_low]);
              freq_int_heat_AGN_soft[R_ct] += freq_int_heat_tbl_AGN_soft[m_xHII_low][R_ct];

              freq_int_ion_AGN_soft[R_ct] =
                (freq_int_ion_tbl_AGN_soft[m_xHII_high][R_ct] - freq_int_ion_tbl_AGN_soft[m_xHII_low][R_ct]) /
                (x_int_XHII[m_xHII_high] - x_int_XHII[m_xHII_low]);
              freq_int_ion_AGN_soft[R_ct] *= (xHII_call - x_int_XHII[m_xHII_low]);
              freq_int_ion_AGN_soft[R_ct] += freq_int_ion_tbl_AGN_soft[m_xHII_low][R_ct];

              freq_int_lya_AGN_soft[R_ct] =
                (freq_int_lya_tbl_AGN_soft[m_xHII_high][R_ct] - freq_int_lya_tbl_AGN_soft[m_xHII_low][R_ct]) /
                (x_int_XHII[m_xHII_high] - x_int_XHII[m_xHII_low]);
              freq_int_lya_AGN_soft[R_ct] *= (xHII_call - x_int_XHII[m_xHII_low]);
              freq_int_lya_AGN_soft[R_ct] += freq_int_lya_tbl_AGN_soft[m_xHII_low][R_ct];
            } else {
              freq_int_heat_AGN_soft[R_ct] = 0.0;
              freq_int_ion_AGN_soft[R_ct] = 0.0;
              freq_int_lya_AGN_soft[R_ct] = 0.0;
            }

            if (agn_hard_needed) {
              freq_int_heat_AGN_hard[R_ct] =
                (freq_int_heat_tbl_AGN_hard[m_xHII_high][R_ct] - freq_int_heat_tbl_AGN_hard[m_xHII_low][R_ct]) /
                (x_int_XHII[m_xHII_high] - x_int_XHII[m_xHII_low]);
              freq_int_heat_AGN_hard[R_ct] *= (xHII_call - x_int_XHII[m_xHII_low]);
              freq_int_heat_AGN_hard[R_ct] += freq_int_heat_tbl_AGN_hard[m_xHII_low][R_ct];

              freq_int_ion_AGN_hard[R_ct] =
                (freq_int_ion_tbl_AGN_hard[m_xHII_high][R_ct] - freq_int_ion_tbl_AGN_hard[m_xHII_low][R_ct]) /
                (x_int_XHII[m_xHII_high] - x_int_XHII[m_xHII_low]);
              freq_int_ion_AGN_hard[R_ct] *= (xHII_call - x_int_XHII[m_xHII_low]);
              freq_int_ion_AGN_hard[R_ct] += freq_int_ion_tbl_AGN_hard[m_xHII_low][R_ct];

              freq_int_lya_AGN_hard[R_ct] =
                (freq_int_lya_tbl_AGN_hard[m_xHII_high][R_ct] - freq_int_lya_tbl_AGN_hard[m_xHII_low][R_ct]) /
                (x_int_XHII[m_xHII_high] - x_int_XHII[m_xHII_low]);
              freq_int_lya_AGN_hard[R_ct] *= (xHII_call - x_int_XHII[m_xHII_low]);
              freq_int_lya_AGN_hard[R_ct] += freq_int_lya_tbl_AGN_hard[m_xHII_low][R_ct];
            } else {
              freq_int_heat_AGN_hard[R_ct] = 0.0;
              freq_int_ion_AGN_hard[R_ct] = 0.0;
              freq_int_lya_AGN_hard[R_ct] = 0.0;
            }
          }

          // Perform the calculation of the heating/ionisation integrals, updating relevant quantities etc.
#if USE_MINI_HALOS
          evolveInt((float)zp,
                    run_globals.reion_grids.deltax[i_padded],
                    SFR_GAL,
#if USE_STOCHASTICITY
                    XRAY_LUMINOSITY_GAL,
#endif
                    SFR_III,
                    XAGN_soft,
                    XAGN_hard,
                    AGN_LW,
                    freq_int_heat_GAL,
                    freq_int_ion_GAL,
                    freq_int_lya_GAL,
                    freq_int_heat_III,
                    freq_int_ion_III,
                    freq_int_lya_III,
                    freq_int_heat_AGN_soft,
                    freq_int_ion_AGN_soft,
                    freq_int_lya_AGN_soft,
                    freq_int_heat_AGN_hard,
                    freq_int_ion_AGN_hard,
                    freq_int_lya_AGN_hard,
                    NO_LIGHT,
                    ans,
                    dansdz);
#else
          evolveInt((float)zp,
                    run_globals.reion_grids.deltax[i_padded],
                    SFR_GAL,
#if USE_STOCHASTICITY
                    XRAY_LUMINOSITY_GAL,
#endif
                    XAGN_soft,
                    XAGN_hard,
                    freq_int_heat_GAL,
                    freq_int_ion_GAL,
                    freq_int_lya_GAL,
                    freq_int_heat_AGN_soft,
                    freq_int_ion_AGN_soft,
                    freq_int_lya_AGN_soft,
                    freq_int_heat_AGN_hard,
                    freq_int_ion_AGN_hard,
                    freq_int_lya_AGN_hard,
                    NO_LIGHT,
                    ans,
                    dansdz);
#endif
          Xheat_ave_AGN_soft += dansdz[5];
          Xheat_ave_AGN_hard += dansdz[6];

          x_e_box_prev[i_padded] += dansdz[0] * dzp; // remember dzp is negative
          if (x_e_box_prev[i_padded] > 1)            // can do this late in evolution if dzp is too large
            x_e_box_prev[i_padded] = (float)(1 - FRACT_FLOAT_ERR);
          else if (x_e_box_prev[i_padded] < 0)
            x_e_box_prev[i_padded] = 0;
          if (Tk_box[i_real] < MAX_TK)
            Tk_box[i_real] += dansdz[1] * dzp;

#if USE_MINI_HALOS
          if (Tk_boxII[i_real] < MAX_TK)
            Tk_boxII[i_real] += dansdz[9] * dzp;
          if (run_globals.params.Flag_IncludeLymanWerner) {
            JLW_box[i_real] = dansdz[8];
            JLW_boxII[i_real] = dansdz[13];
          }
#endif
          // spurious bahaviour of the trapazoidalintegrator. generally overcooling in underdensities.
          // Floored at MIN_TK (not just < 0) because 1/TK and 1/TK^2 in Salpha_tilde()/Tc_eff()
          // (get_Ts(), spin-temperature coupling) blow up to Inf/NaN as TK -> 0, even while still
          // nominally positive — most easily triggered when AGN heating (sparse) can't fill in for
          // suppressed HMXB heating (normally spatially-widespread) in underdense cells.
          if (Tk_box[i_real] < MIN_TK)
            Tk_box[i_real] = (float)(TCMB * (1 + zp));

#if USE_MINI_HALOS
          if (Tk_boxII[i_real] < MIN_TK)
            Tk_boxII[i_real] = (float)(TCMB * (1 + zp));
#endif

          TS_box[i_real] = get_Ts((float)zp,
                                  run_globals.reion_grids.deltax[i_padded],
                                  Tk_box[i_real],
                                  x_e_box_prev[i_padded],
                                  (float)dansdz[2],
                                  &curr_xalpha);
#if USE_MINI_HALOS
          TS_boxII[i_real] = get_Ts((float)zp,
                                    run_globals.reion_grids.deltax[i_padded],
                                    Tk_boxII[i_real],
                                    x_e_box_prev[i_padded],
                                    (float)dansdz[10],
                                    &curr_xalpha); // It should be correct, probably I don't need a new curr_xalphaII
#endif
          J_alpha_ave += dansdz[2];
          xalpha_ave += curr_xalpha; // Double check this! It might be saving the one from PopIII!
          Xheat_ave += dansdz[3];
          Xion_ave += dansdz[4];
#if USE_MINI_HALOS
          J_alpha_aveII += dansdz[10];
          Xheat_aveII += dansdz[11];
          Xion_aveII += dansdz[12];
          if (run_globals.params.Flag_IncludeLymanWerner) {
            J_LW_ave += dansdz[8];
            J_LW_aveII += dansdz[13];
            J_LW_ave_AGN += dansdz[7];
          }
#endif
        }

    MPI_Allreduce(MPI_IN_PLACE, &J_alpha_ave, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
    MPI_Allreduce(MPI_IN_PLACE, &xalpha_ave, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
    MPI_Allreduce(MPI_IN_PLACE, &Xheat_ave, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
    MPI_Allreduce(MPI_IN_PLACE, &Xion_ave, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
    MPI_Allreduce(MPI_IN_PLACE, &Xheat_ave_AGN_soft, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
    MPI_Allreduce(MPI_IN_PLACE, &Xheat_ave_AGN_hard, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
#if USE_MINI_HALOS
    MPI_Allreduce(MPI_IN_PLACE, &J_alpha_aveII, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
    MPI_Allreduce(MPI_IN_PLACE, &Xheat_aveII, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
    MPI_Allreduce(MPI_IN_PLACE, &Xion_aveII, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
    MPI_Allreduce(MPI_IN_PLACE, &J_LW_ave, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
    MPI_Allreduce(MPI_IN_PLACE, &J_LW_aveII, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
    MPI_Allreduce(MPI_IN_PLACE, &J_LW_ave_AGN, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
#endif

    J_alpha_ave /= total_n_cells;
    xalpha_ave /= total_n_cells;
    Xheat_ave /= total_n_cells;
    Xion_ave /= total_n_cells;
    Xheat_ave_AGN_soft /= total_n_cells;
    Xheat_ave_AGN_hard /= total_n_cells;
#if USE_MINI_HALOS
    J_alpha_aveII /= total_n_cells;
    Xheat_aveII /= total_n_cells;
    Xion_aveII /= total_n_cells;
    if (run_globals.params.Flag_IncludeLymanWerner) {
      J_LW_ave /= total_n_cells;
      J_LW_aveII /= total_n_cells;
      J_LW_ave_AGN /= total_n_cells;
    }
#endif

    run_globals.reion_grids.volume_ave_J_alpha = J_alpha_ave;
    run_globals.reion_grids.volume_ave_xalpha = xalpha_ave;
    run_globals.reion_grids.volume_ave_Xheat = Xheat_ave;
    run_globals.reion_grids.volume_ave_Xion = Xion_ave;
    run_globals.reion_grids.volume_ave_Xheat_AGN_soft = Xheat_ave_AGN_soft;
    run_globals.reion_grids.volume_ave_Xheat_AGN_hard = Xheat_ave_AGN_hard;
#if USE_MINI_HALOS
    run_globals.reion_grids.volume_ave_J_alphaII = J_alpha_aveII;
    run_globals.reion_grids.volume_ave_XheatII = Xheat_aveII;
    run_globals.reion_grids.volume_ave_XionII = Xion_aveII;
    if (run_globals.params.Flag_IncludeLymanWerner) {
      run_globals.reion_grids.volume_ave_J_LW = J_LW_ave;
      run_globals.reion_grids.volume_ave_J_LWII = J_LW_aveII;
      run_globals.reion_grids.volume_ave_J_LW_AGN = J_LW_ave_AGN;
    }
#endif
  }

  memcpy(x_e_box, x_e_box_prev, sizeof(fftwf_complex) * slab_n_complex);

  double Ave_Ts = 0.0;
  double Ave_x_e = 0.0;
  double Ave_Tk = 0.0;
#if USE_MINI_HALOS
  double Ave_TkII = 0.0;
  double Ave_TsII = 0.0;
#endif

  for (int ix = 0; ix < local_nix; ix++)
    for (int iy = 0; iy < ReionGridDim; iy++)
      for (int iz = 0; iz < ReionGridDim; iz++) {
        i_real = grid_index(ix, iy, iz, ReionGridDim, INDEX_REAL);
        i_padded = grid_index(ix, iy, iz, ReionGridDim, INDEX_PADDED);

        Ave_Ts += (double)TS_box[i_real];
        Ave_Tk += (double)Tk_box[i_real];
        Ave_x_e += (double)x_e_box_prev[i_padded];
#if USE_MINI_HALOS
        Ave_TsII += (double)TS_boxII[i_real];
        Ave_TkII += (double)Tk_boxII[i_real];
#endif
      }

  MPI_Allreduce(MPI_IN_PLACE, &Ave_Ts, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
  MPI_Allreduce(MPI_IN_PLACE, &Ave_Tk, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
  MPI_Allreduce(MPI_IN_PLACE, &Ave_x_e, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);

  Ave_Ts /= total_n_cells;
  Ave_Tk /= total_n_cells;
  Ave_x_e /= total_n_cells;

  run_globals.reion_grids.volume_ave_TS = Ave_Ts;
  run_globals.reion_grids.volume_ave_TK = Ave_Tk;
  run_globals.reion_grids.volume_ave_xe = Ave_x_e;

#if USE_MINI_HALOS
  MPI_Allreduce(MPI_IN_PLACE, &Ave_TsII, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);
  MPI_Allreduce(MPI_IN_PLACE, &Ave_TkII, 1, MPI_DOUBLE, MPI_SUM, run_globals.mpi_comm);

  Ave_TsII /= total_n_cells;
  Ave_TkII /= total_n_cells;

  run_globals.reion_grids.volume_ave_TSII = Ave_TsII;
  run_globals.reion_grids.volume_ave_TKII = Ave_TkII;
#endif

  destruct_heat();

#if USE_MINI_HALOS
  mlog("zp = %e Ts_ave = %e Ts_ave (PopII) = %e Tk_ave = %e Tk_ave (PopII) = %e x_e_ave = %e",
       MLOG_MESG,
       zp,
       Ave_Ts,
       Ave_TsII,
       Ave_Tk,
       Ave_TkII,
       Ave_x_e);
  mlog("zp = %e J_alpha_ave = %e J_alpha_ave (PopII) = %e xalpha_ave = %e Xheat_ave = %e Xheat_ave (PopII) = %e "
       "Xion_ave = %e Xion_ave (PopII) = %e J_LW_ave = %e J_LW_ave (PopII) = %e J_LW_ave (AGN) = %e",
       MLOG_MESG,
       zp,
       J_alpha_ave,
       J_alpha_aveII,
       xalpha_ave,
       Xheat_ave,
       Xheat_aveII,
       Xion_ave,
       Xion_aveII,
       J_LW_ave,
       J_LW_aveII,
       J_LW_ave_AGN);
  mlog("zp = %e  AGN_Xheat_soft = %e  AGN_Xheat_hard = %e  (Flag_IncludeAGNXray=%d)",
       MLOG_MESG,
       zp,
       Xheat_ave_AGN_soft,
       Xheat_ave_AGN_hard,
       run_globals.params.physics.Flag_IncludeAGNXray);
#else
  mlog("zp = %e Ts_ave = %e Tk_ave = %e x_e_ave = %e", MLOG_MESG, zp, Ave_Ts, Ave_Tk, Ave_x_e);
  mlog("zp = %e J_alpha_ave = %e xalpha_ave = %e Xheat_ave = %e Xion_ave = %e",
       MLOG_MESG,
       zp,
       J_alpha_ave,
       xalpha_ave,
       Xheat_ave,
       Xion_ave);
  mlog("zp = %e  AGN_Xheat_soft = %e  AGN_Xheat_hard = %e  (Flag_IncludeAGNXray=%d)",
       MLOG_MESG,
       zp,
       Xheat_ave_AGN_soft,
       Xheat_ave_AGN_hard,
       run_globals.params.physics.Flag_IncludeAGNXray);
#endif
}

// This function makes sure that the right version of ComputeTs() gets called.
// Note: Only the CPU version works for now
void ComputeTs(int snapshot, timer_info* timer_total)
{
  // Call the version of ComputeTs we've been passed (and time it)
  timer_info timer;
  // double redshift = run_globals.ZZ[snapshot];
  // Run the Meraxes version of _ComputeTs()
  mlog("Calling pure-CPU version of ComputeTs() for snap=%d/z=%.2lf...",
       MLOG_OPEN | MLOG_TIMERSTART,
       snapshot,
       run_globals.ZZ[snapshot]);
  timer_start(&timer);
  _ComputeTs(snapshot);
  timer_stop(&timer);
  timer_stop(timer_total);
  timer_gpu += timer_delta(timer);
  mlog("...done", MLOG_CLOSE | MLOG_TIMERSTOP);
  mlog("Total time spent in ComputeTs vs. total run time (snapshot %d ): %.2f of %.2f s",
       MLOG_MESG,
       snapshot,
       timer_gpu,
       timer_delta(*timer_total));
}
