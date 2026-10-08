/*
 * These are the relevant functions taken from the public version of 21cmFAST code
 * to compute inhomogeneous recombinations. Taken from "recombinations.c" written
 * by Andrei Mesinger and Emanuele Sobacchi (2013abc).
 *
 * Inclusion of this for Meraxes was written by Bradley Greig.
 */

#include "recombinations.h"
#include "XRayHeatingFunctions.h"
#include "meraxes.h"
#include "reionization.h"

static double A_table[A_NPTS], A_params[A_NPTS];
static gsl_interp_accel* A_acc;
static gsl_spline* A_spline;

static double C_table[C_NPTS], C_params[C_NPTS];
static gsl_interp_accel* C_acc;
static gsl_spline* C_spline;

static double beta_table[beta_NPTS], beta_params[beta_NPTS];
static gsl_interp_accel* beta_acc;
static gsl_spline* beta_spline;

double *lnGamma_values, *RR_table, *RNH_table, *CF_table;
gsl_interp_accel **RR_acc, **RNH_acc, **CF_acc;
gsl_spline **RR_spline, **RNH_spline, **CF_spline;

int splined_recombination(double z_eff, double gamma12_bg, double temp, double *recombination_rate, double *residual_xH, double *clumping_factor)
{
  int z_ct = (int)((z_eff-RR_Z_END) / RR_DEL_Z + 0.5); // round to nearest int
  int t_ct = (int)((log10(temp)-RR_T_STA) / RR_DEL_T + 0.5); // round to nearest int
  double lnGamma = log(gamma12_bg);

  // check out of bounds
  if (z_ct < 0) { // out of array bounds
    mlog("WARNING: splined_recombination_rate: effective redshift %g is outside of array left bound", MLOG_MESG, z_eff);
    z_ct = 0;
  } else if (z_ct >= RR_Z_NPTS) {
    mlog("WARNING: splined_recombination_rate: effective redshift %g is outside of array right bound", MLOG_MESG, z_eff);
    z_ct = RR_Z_NPTS - 1;
  }

  if (t_ct < 0) { // out of array bounds
    //mlog("WARNING: splined_recombination_rate: temperature %g is outside of array left bound", MLOG_MESG, temp);
    //t_ct = 0;
    *recombination_rate = 0.;
    *residual_xH = 1e4;
	*clumping_factor = 1.0;
    return 1;
  } else if (t_ct >= RR_T_NPTS) {
    //mlog("WARNING: splined_recombination_rate: temperature %g is outside of array right bound", MLOG_MESG, temp);
    t_ct = RR_T_NPTS - 1;
  }

  if (lnGamma < RR_lnGamma_min) {
    *recombination_rate = 0.;
    *residual_xH = 1e4;
	*clumping_factor = 1.0;
    return 1;
  } else if (lnGamma >= (RR_lnGamma_min + RR_DEL_lnGamma * (RR_lnGamma_NPTS-1))) {
    mlog("WARNING: splined_recombination_rate: Gamma12 of %g is outside of interpolation array", MLOG_MESG, gamma12_bg);
    lnGamma = RR_lnGamma_min + RR_DEL_lnGamma * (RR_lnGamma_NPTS-1) - FRACT_FLOAT_ERR;
  }

  int idx = z_ct * RR_T_NPTS + t_ct;
  *recombination_rate = exp(gsl_spline_eval(RR_spline[idx], lnGamma, RR_acc[idx]));
  *residual_xH = exp(gsl_spline_eval(RNH_spline[idx], lnGamma, RNH_acc[idx]));
  *clumping_factor = gsl_spline_eval(CF_spline[idx], lnGamma, CF_acc[idx]);

  return 1;
}

void init_MHR()
{
  int z_ct, gamma_ct, t_ct, idx, flag_recalc;
  double z, gamma, temp;
  int RR_ZT_NPTS = RR_Z_NPTS*RR_T_NPTS;
  int TOT_NPTS = RR_ZT_NPTS*RR_lnGamma_NPTS;
  
  FILE *gamma_fp, *rr_fp, *cf_fp, *rnh_fp;
  char GAMMA_FILENAME[2 * STRLEN];
  char RR_FILENAME[2 * STRLEN];
  char CF_FILENAME[2 * STRLEN];
  char RNH_FILENAME[2 * STRLEN];

  mlog("Initialising MHR parameter and recombination interpolation tables...", MLOG_OPEN | MLOG_TIMERSTART);

  // first initialize the MHR parameter look up tables
  init_C_MHR();    /*initializes the lookup table for the C paremeter in MHR00 model*/
  init_beta_MHR(); /*initializes the lookup table for the beta paremeter in MHR00 model*/
  init_A_MHR();    /*initializes the lookup table for the A paremeter in MHR00 model*/

  RR_table = malloc(TOT_NPTS*sizeof(double));
  CF_table = malloc(TOT_NPTS*sizeof(double));
  RNH_table = malloc(TOT_NPTS*sizeof(double));
  lnGamma_values = malloc(RR_lnGamma_NPTS * sizeof(double));
  if (!RR_table || !CF_table || !RNH_table || !lnGamma_values){
    mlog_error("Failed to allocate memory for the tables. Aborting...");
    ABORT(EXIT_FAILURE);
  }

  if (run_globals.mpi_rank == 0) {
    sprintf(GAMMA_FILENAME, "%s/lnGamma_table_%g-%d-%g.bin", run_globals.params.RecombinationDir, RR_lnGamma_min, RR_lnGamma_NPTS, RR_DEL_lnGamma);
    sprintf(RR_FILENAME, "%s/RR_table_%g-%d-%g_%g-%d-%g_%g-%d-%g.bin", run_globals.params.RecombinationDir, RR_lnGamma_min, RR_lnGamma_NPTS, RR_DEL_lnGamma, RR_Z_END, RR_Z_NPTS, RR_DEL_Z, RR_T_STA, RR_T_NPTS, RR_DEL_T);
    sprintf(CF_FILENAME, "%s/CF_table_%g-%d-%g_%g-%d-%g_%g-%d-%g.bin", run_globals.params.RecombinationDir, RR_lnGamma_min, RR_lnGamma_NPTS, RR_DEL_lnGamma, RR_Z_END, RR_Z_NPTS, RR_DEL_Z, RR_T_STA, RR_T_NPTS, RR_DEL_T);
    sprintf(RNH_FILENAME, "%s/RNH_table_%g-%d-%g_%g-%d-%g_%g-%d-%g.bin", run_globals.params.RecombinationDir, RR_lnGamma_min, RR_lnGamma_NPTS, RR_DEL_lnGamma, RR_Z_END, RR_Z_NPTS, RR_DEL_Z, RR_T_STA, RR_T_NPTS, RR_DEL_T);
    gamma_fp = fopen(GAMMA_FILENAME, "rb");
    rr_fp = fopen(RR_FILENAME, "rb");
    cf_fp = fopen(CF_FILENAME, "rb");
    rnh_fp = fopen(RNH_FILENAME, "rb");

    if (gamma_fp && rr_fp && cf_fp && rnh_fp) {
      fread(lnGamma_values, sizeof(double), RR_lnGamma_NPTS, gamma_fp);
      fread(RR_table, sizeof(double), TOT_NPTS, rr_fp);
      fread(CF_table, sizeof(double), TOT_NPTS, cf_fp);
      fread(RNH_table, sizeof(double), TOT_NPTS, rnh_fp);
      fclose(gamma_fp);
      fclose(rr_fp);
      fclose(cf_fp);
      fclose(rnh_fp);
      flag_recalc = 0;
      mlog("Loaded recombination tables from disk.", MLOG_MESG);
    }
    else{
      if (gamma_fp) fclose(gamma_fp);
      if (rr_fp)    fclose(rr_fp);
      if (cf_fp)    fclose(cf_fp);
      if (rnh_fp)   fclose(rnh_fp);
      flag_recalc = 1;
      mlog("Recomputing recombination tables in parallel.", MLOG_MESG | MLOG_TIMERSTART);
      for (gamma_ct = 0; gamma_ct < RR_lnGamma_NPTS; gamma_ct++)
        lnGamma_values[gamma_ct] = RR_lnGamma_min + gamma_ct * RR_DEL_lnGamma; // ln of Gamma12
    }
  }
  MPI_Bcast(&flag_recalc, 1, MPI_INT, 0, run_globals.mpi_comm);
  MPI_Bcast(lnGamma_values, RR_lnGamma_NPTS, MPI_DOUBLE, 0, run_globals.mpi_comm);

  if (flag_recalc){
      int *recvcounts = malloc(run_globals.mpi_size * sizeof(int));
      int *displs     = malloc(run_globals.mpi_size * sizeof(int));

      int local_start = (RR_ZT_NPTS * run_globals.mpi_rank) / run_globals.mpi_size;
      int local_end   = (RR_ZT_NPTS * (run_globals.mpi_rank + 1)) / run_globals.mpi_size;
      int local_count = local_end - local_start;
      int local_idx;

      for (int r = 0; r < run_globals.mpi_size; r++) {
        recvcounts[r] = ((int)((RR_ZT_NPTS * (r + 1)) / run_globals.mpi_size) -
                         (int)((RR_ZT_NPTS * r) / run_globals.mpi_size)) * RR_lnGamma_NPTS ;
        displs[r]     = (r == 0) ? 0 : displs[r - 1] + recvcounts[r - 1];
      }

      double* local_RR  = malloc(local_count * RR_lnGamma_NPTS * sizeof(double));
      double* local_CF  = malloc(local_count * RR_lnGamma_NPTS * sizeof(double));
      double* local_RNH = malloc(local_count * RR_lnGamma_NPTS * sizeof(double));

      for (idx = local_start; idx < local_end; idx++) {
        z_ct = idx / RR_T_NPTS;
        t_ct = idx % RR_T_NPTS;

        z = z_ct * RR_DEL_Z + RR_Z_END; // redshift corresponding to index z_ct of the array

        temp = pow(10, (t_ct * RR_DEL_T + RR_T_STA) - 4.0);

        for (gamma_ct = 0; gamma_ct < RR_lnGamma_NPTS; gamma_ct++) {
          gamma = exp(lnGamma_values[gamma_ct]);

          local_idx = (idx - local_start) * RR_lnGamma_NPTS + gamma_ct;
          local_RR[local_idx]  = log(recombination_rate(z, gamma, temp, 1));
          local_CF[local_idx]  = clumping_factor(z, gamma, temp, 1);
          local_RNH[local_idx] = log(residual_neutral_hydrogen(z, gamma, temp, 1));
          // NOTE: although the table is more linear when taken log, it's faster otherwise have to do exp() 
        }
      }
      MPI_Allgatherv(local_RR, local_count * RR_lnGamma_NPTS, MPI_DOUBLE,
                     RR_table, recvcounts, displs, MPI_DOUBLE, run_globals.mpi_comm);
      MPI_Allgatherv(local_CF, local_count * RR_lnGamma_NPTS, MPI_DOUBLE,
                     CF_table, recvcounts, displs, MPI_DOUBLE, run_globals.mpi_comm);
      MPI_Allgatherv(local_RNH, local_count * RR_lnGamma_NPTS, MPI_DOUBLE,
                     RNH_table, recvcounts, displs, MPI_DOUBLE, run_globals.mpi_comm);
      free(local_RR);
      free(local_CF);
      free(local_RNH);

      if (run_globals.mpi_rank == 0){

        // Save to disk
        gamma_fp = fopen(GAMMA_FILENAME, "wb");
        rr_fp = fopen(RR_FILENAME, "wb");
        cf_fp = fopen(CF_FILENAME, "wb");
        rnh_fp = fopen(RNH_FILENAME, "wb");
        if (gamma_fp && rr_fp && cf_fp && rnh_fp) {
          fwrite(lnGamma_values, sizeof(double), RR_lnGamma_NPTS, gamma_fp);
          fwrite(RR_table, sizeof(double), TOT_NPTS, rr_fp);
          fwrite(CF_table, sizeof(double), TOT_NPTS, cf_fp);
          fwrite(RNH_table, sizeof(double), TOT_NPTS, rnh_fp);
          fclose(gamma_fp);
          fclose(rr_fp);
          fclose(cf_fp);
          fclose(rnh_fp);
          mlog("Saved RR_table and RNH_table to disk.", MLOG_MESG);
        } 
        else{
          if (gamma_fp) fclose(gamma_fp);
          if (rr_fp)    fclose(rr_fp);
          if (cf_fp)    fclose(cf_fp);
          if (rnh_fp)   fclose(rnh_fp);
          mlog("Warning: Failed to save lookup tables to disk.", MLOG_MESG);
        }
      }
      mlog("...done.", MLOG_CONT | MLOG_TIMERSTOP);
  }
  else{
      MPI_Bcast(RR_table, TOT_NPTS, MPI_DOUBLE, 0, run_globals.mpi_comm);
      MPI_Bcast(CF_table, TOT_NPTS, MPI_DOUBLE, 0, run_globals.mpi_comm);
      MPI_Bcast(RNH_table, TOT_NPTS, MPI_DOUBLE, 0, run_globals.mpi_comm);
  }
  RR_acc  = malloc(RR_ZT_NPTS * sizeof(gsl_interp_accel *));
  CF_acc  = malloc(RR_ZT_NPTS * sizeof(gsl_interp_accel *));
  RNH_acc = malloc(RR_ZT_NPTS * sizeof(gsl_interp_accel *));
  RR_spline = malloc(RR_ZT_NPTS * sizeof(gsl_spline *));
  CF_spline = malloc(RR_ZT_NPTS * sizeof(gsl_spline *));
  RNH_spline = malloc(RR_ZT_NPTS * sizeof(gsl_spline *));

  // now the recombination rate look up tables
  for (idx = 0; idx < RR_ZT_NPTS; idx++) {
       //z_ct = idx / RR_T_NPTS;
       //t_ct = idx % RR_T_NPTS;
       //z = z_ct * RR_DEL_Z + RR_Z_END; // redshift corresponding to index z_ct of the array
       //temp = pow(10, (t_ct * RR_DEL_T + RR_T_STA));
      //for (gamma_ct = 0; gamma_ct < RR_lnGamma_NPTS; gamma_ct++) {
      //  gamma = exp(lnGamma_values[gamma_ct]);
      //  mlog("z=%.2f, temp = %.2f x 1e4 K, Gamma12=%.2f, recomibiation rate=%g, clumping factor=%g, residual xH=%g", MLOG_MESG, z, temp, gamma, RR_table[idx][gamma_ct], CF_table[idx][gamma_ct], RNH_table[idx][gamma_ct]);
      //}

      // set up the spline in gamma
      RR_acc[idx] = gsl_interp_accel_alloc();
      RR_spline[idx] = gsl_spline_alloc(gsl_interp_cspline, RR_lnGamma_NPTS);
      gsl_spline_init(RR_spline[idx], lnGamma_values, &RR_table[idx * RR_lnGamma_NPTS], RR_lnGamma_NPTS);

      CF_acc[idx] = gsl_interp_accel_alloc();
      CF_spline[idx] = gsl_spline_alloc(gsl_interp_cspline, RR_lnGamma_NPTS);
      gsl_spline_init(CF_spline[idx], lnGamma_values, &CF_table[idx * RR_lnGamma_NPTS], RR_lnGamma_NPTS);
    
      RNH_acc[idx] = gsl_interp_accel_alloc();
      RNH_spline[idx] = gsl_spline_alloc(gsl_interp_cspline, RR_lnGamma_NPTS);
      gsl_spline_init(RNH_spline[idx], lnGamma_values, &RNH_table[idx * RR_lnGamma_NPTS], RR_lnGamma_NPTS);

  }

  mlog("...done.", MLOG_CLOSE | MLOG_TIMERSTOP);

}

void free_MHR()
{
  int idx;

  if (!RR_spline) return;  /* init_MHR was never called — nothing to free */

  free_A_MHR();
  free_C_MHR();
  free_beta_MHR();

  // now the recombination rate look up tables
  for (idx = 0; idx < RR_Z_NPTS * RR_T_NPTS; idx++) {
      gsl_spline_free(RR_spline[idx]);
      gsl_interp_accel_free(RR_acc[idx]);
      gsl_spline_free(CF_spline[idx]);
      gsl_interp_accel_free(CF_acc[idx]);
      gsl_spline_free(RNH_spline[idx]);
      gsl_interp_accel_free(RNH_acc[idx]);
  }
  free(RR_spline);
  free(RR_acc);
  free(RNH_spline);
  free(RNH_acc);
  free(CF_spline);
  free(CF_acc);
  free(lnGamma_values);
  free(RR_table);
  free(RNH_table);
  free(CF_table);
}

// calculates the attenuated photoionization rate due to self-shielding (in units of 1e-12 s^-1)
// input parameters are the background ionization rate, overdensity, temperature (in 10^4k), redshift, respectively
//  Uses the fitting formula from Rahmati et al, assuming a UVB power law index of alpha=5
double Gamma_SS(double Gamma_bg, double Delta, double T_4, double z)
{
  double D_ss = 26.7 * pow(T_4, 0.17) * pow((1 + z) / 10.0, -3) * pow(Gamma_bg, 2.0 / 3.0);
  return Gamma_bg * (0.98 * pow((1.0 + pow(Delta / D_ss, 1.64)), -2.28) + 0.02 * pow(1.0 + Delta / D_ss, -0.84));
}

typedef struct
{
  double z, gamma12_bg, T4, A, C_0, beta, avenH;
  int usecaseB;
} RR_par;

double MHR_rr(double lnD, void* params)
{
  double D = exp(lnD);
  double alpha;
  RR_par p = *(RR_par*)params;
  double z = p.z;
  double gamma = Gamma_SS(p.gamma12_bg, D, p.T4, z);
  double n_H = p.avenH * D;
  double x_e = 1.0 - neutral_fraction(n_H, p.T4, gamma, p.usecaseB);
  double PDelta;

  PDelta = p.A * exp(-0.5 * pow((pow(D, -2.0 / 3.0) - p.C_0) / ((2.0 * 7.61 / (3.0 * (1.0 + z)))), 2)) * pow(D, p.beta);

  if (p.usecaseB)
    alpha = alpha_B(p.T4 * 1e4);
  else
    alpha = alpha_A(p.T4 * 1e4);

  return n_H * PDelta * alpha * x_e * x_e * D * D; // note extra D since we are integrating over lnD
}

double MHR_cf_numerator(double lnD, void* params)
{
  double D = exp(lnD);
  RR_par p = *(RR_par*)params;
  double z = p.z;
  double gamma = Gamma_SS(p.gamma12_bg, D, p.T4, z);
  double n_H = p.avenH * D;
  double x_e = 1.0 - neutral_fraction(n_H, p.T4, gamma, p.usecaseB);
  double PDelta;

  PDelta = p.A * exp(-0.5 * pow((pow(D, -2.0 / 3.0) - p.C_0) / ((2.0 * 7.61 / (3.0 * (1.0 + z)))), 2)) * pow(D, p.beta);

  return PDelta * x_e * x_e * D * D * D; // note extra D since we are integrating over lnD
}

double MHR_cf_denominator(double lnD, void* params)
{
  double D = exp(lnD);
  RR_par p = *(RR_par*)params;
  double z = p.z;
  double gamma = Gamma_SS(p.gamma12_bg, D, p.T4, z);
  double n_H = p.avenH * D;
  double x_e = 1.0 - neutral_fraction(n_H, p.T4, gamma, p.usecaseB);
  double PDelta;

  PDelta = p.A * exp(-0.5 * pow((pow(D, -2.0 / 3.0) - p.C_0) / ((2.0 * 7.61 / (3.0 * (1.0 + z)))), 2)) * pow(D, p.beta);

  return PDelta * x_e * D * D; // note extra D since we are integrating over lnD
}

double MHR_rnh(double lnD, void* params)
{
  double D = exp(lnD);
  RR_par p = *(RR_par*)params;
  double z = p.z;
  double gamma = Gamma_SS(p.gamma12_bg, D, p.T4, z);
  double n_H = p.avenH * D;
  double x_HI = neutral_fraction(n_H, p.T4, gamma, p.usecaseB);
  double PDelta;

  PDelta = p.A * exp(-0.5 * pow((pow(D, -2.0 / 3.0) - p.C_0) / ((2.0 * 7.61 / (3.0 * (1.0 + z)))), 2)) * pow(D, p.beta);

  return 1e4 * PDelta * x_HI * D;
}

// returns the recombination rate per baryon (1/s), integrated over the MHR density PDF,
// given an ionizing background of gamma12_bg
// temeperature T4 (in 1e4 K), and usecaseB rate coefficient
// Assumes self-shielding according to Rahmati+ 2013
double recombination_rate(double z_eff, double gamma12_bg, double T4, int usecaseB)
{
  double result, error, lower_limit, upper_limit;
  gsl_function F;
  double rel_tol = 0.01; //<- relative tolerance
  gsl_integration_workspace* w = gsl_integration_workspace_alloc(1000);
  RR_par p = { z_eff, gamma12_bg, T4, A_MHR(z_eff), C_MHR(z_eff), beta_MHR(z_eff), No * pow(1 + z_eff, 3), usecaseB };

  F.function = &MHR_rr;
  F.params = &p;
  lower_limit = log(0.01);
  upper_limit = log(200);

  gsl_integration_qag(&F, lower_limit, upper_limit, 0, rel_tol, 1000, GSL_INTEG_GAUSS61, w, &result, &error);
  gsl_integration_workspace_free(w);

  return result;
}

double clumping_factor(double z_eff, double gamma12_bg, double T4, int usecaseB)
{
  double result, error, lower_limit, upper_limit;
  double denominator, numerator;
  gsl_function F;
  double rel_tol = 0.01; //<- relative tolerance
  gsl_integration_workspace* w = gsl_integration_workspace_alloc(1000);
  RR_par p = { z_eff, gamma12_bg, T4, A_MHR(z_eff), C_MHR(z_eff), beta_MHR(z_eff), No * pow(1 + z_eff, 3), usecaseB };

  F.params = &p;
  lower_limit = log(0.01);
  upper_limit = log(200);

  F.function = &MHR_cf_numerator;
  gsl_integration_qag(&F, lower_limit, upper_limit, 0, rel_tol, 1000, GSL_INTEG_GAUSS61, w, &numerator, &error);

  F.function = &MHR_cf_denominator;
  gsl_integration_qag(&F, lower_limit, upper_limit, 0, rel_tol, 1000, GSL_INTEG_GAUSS61, w, &denominator, &error);

  gsl_integration_workspace_free(w);

  result = numerator / denominator / denominator;

  return result;
}

double residual_neutral_hydrogen(double z_eff, double gamma12_bg, double T4, int usecaseB)
{
  double result, error, lower_limit, upper_limit;
  gsl_function F;
  double rel_tol = 0.01; //<- relative tolerance
  gsl_integration_workspace* w = gsl_integration_workspace_alloc(1000);
  RR_par p = { z_eff, gamma12_bg, T4, A_MHR(z_eff), C_MHR(z_eff), beta_MHR(z_eff), No * pow(1 + z_eff, 3), usecaseB };

  F.function = &MHR_rnh;
  F.params = &p;
  lower_limit = log(0.01);
  upper_limit = log(200);

  gsl_integration_qag(&F, lower_limit, upper_limit, 0, rel_tol, 1000, GSL_INTEG_GAUSS61, w, &result, &error);
  gsl_integration_workspace_free(w);

  return result;
}

double aux_function(double D, void* params)
{
  double result;
  double z = *(double*)params;

  result = exp(-(pow(D, -2.0 / 3.0) - C_MHR(z)) * (pow(D, -2.0 / 3.0) - C_MHR(z)) /
               (2.0 * (2.0 * 7.61 / (3.0 * (1.0 + z))) * (2.0 * 7.61 / (3.0 * (1.0 + z))))) *
           pow(D, beta_MHR(z));

  return result;
}

double A_aux_integral(double z)
{
  double result, error, lower_limit, upper_limit;
  gsl_function F;
  double rel_tol = 0.001; //<- relative tolerance
  gsl_integration_workspace* w = gsl_integration_workspace_alloc(1000);

  F.function = &aux_function;
  F.params = &z;
  lower_limit = 1e-25;
  upper_limit = 1e25;

  gsl_integration_qag(&F, lower_limit, upper_limit, 0, rel_tol, 1000, GSL_INTEG_GAUSS61, w, &result, &error);
  gsl_integration_workspace_free(w);

  return result;
}

double A_MHR(double z)
{
  double result;
  if (z >= 2.0 + (float)A_NPTS)
    result = splined_A_MHR(2.0 + (float)A_NPTS);
  else if (z <= 2.0)
    result = splined_A_MHR(2.0);
  else
    result = splined_A_MHR(z);
  return result;
}

void init_A_MHR()
{
  /* initialize the lookup table for the parameter A in the MHR00 model */
  int i;

  for (i = 0; i < A_NPTS; i++) {
    A_params[i] = 2.0 + (float)i;
    A_table[i] = 1.0 / A_aux_integral(2.0 + (float)i);
  }

  // Set up spline table
  A_acc = gsl_interp_accel_alloc();
  A_spline = gsl_spline_alloc(gsl_interp_cspline, A_NPTS);
  gsl_spline_init(A_spline, A_params, A_table, A_NPTS);
}

double splined_A_MHR(double z)
{
  return gsl_spline_eval(A_spline, z, A_acc);
}

void free_A_MHR()
{

  gsl_spline_free(A_spline);
  gsl_interp_accel_free(A_acc);
}

double C_MHR(double z)
{
  double result;
  if (z >= 13.0)
    result = 1.0;
  else if (z <= 2.0)
    result = 0.558;
  else
    result = splined_C_MHR(z);
  return result;
}

void init_C_MHR()
{
  /* initialize the lookup table for the parameter C in the MHR00 model */
  int i;

  for (i = 0; i < C_NPTS; i++)
    C_params[i] = (float)i + 2.0;

  C_table[0] = 0.558;
  C_table[1] = 0.599;
  C_table[2] = 0.611;
  C_table[3] = 0.769;
  C_table[4] = 0.868;
  C_table[5] = 0.930;
  C_table[6] = 0.964;
  C_table[7] = 0.983;
  C_table[8] = 0.993;
  C_table[9] = 0.998;
  C_table[10] = 0.999;
  C_table[11] = 1.00;

  // Set up spline table
  C_acc = gsl_interp_accel_alloc();
  C_spline = gsl_spline_alloc(gsl_interp_cspline, C_NPTS);
  gsl_spline_init(C_spline, C_params, C_table, C_NPTS);
}

double splined_C_MHR(double z)
{
  return gsl_spline_eval(C_spline, z, C_acc);
}

void free_C_MHR()
{

  gsl_spline_free(C_spline);
  gsl_interp_accel_free(C_acc);
}

double beta_MHR(double z)
{
  double result;
  if (z >= 6.0)
    result = -2.50;
  else if (z <= 2.0)
    result = -2.23;
  else
    result = splined_beta_MHR(z);
  return result;
}

void init_beta_MHR()
{
  /* initialize the lookup table for the parameter C in the MHR00 model */
  int i;

  for (i = 0; i < beta_NPTS; i++)
    beta_params[i] = (float)i + 2.0;

  beta_table[0] = -2.23;
  beta_table[1] = -2.35;
  beta_table[2] = -2.48;
  beta_table[3] = -2.49;
  beta_table[4] = -2.50;

  // Set up spline table
  beta_acc = gsl_interp_accel_alloc();
  beta_spline = gsl_spline_alloc(gsl_interp_cspline, beta_NPTS);
  gsl_spline_init(beta_spline, beta_params, beta_table, beta_NPTS);
}

double splined_beta_MHR(double z)
{
  return gsl_spline_eval(beta_spline, z, beta_acc);
}

void free_beta_MHR()
{

  gsl_spline_free(beta_spline);
  gsl_interp_accel_free(beta_acc);
}

/***********  END NEW FUNCTIONS for v1.3 (recombinations) *********/
/*
   Function NEUTRAL_FRACTION returns the hydrogen neutral fraction, chi, given:
   hydrogen density (pcm^-3)
   gas temperature (10^4 K)
   ionization rate (1e-12 s^-1)
   */
double neutral_fraction(double density, double T4, double gamma12, int usecaseB)
{
  double chi, b, alpha, corr_He = 1.0 / (4.0 / run_globals.params.physics.Y_He - 3);

  if (usecaseB)
    alpha = alpha_B(T4 * 1e4);
  else
    alpha = alpha_A(T4 * 1e4);

  gamma12 *= 1e-12;

  // approximation chi << 1
  chi = (1 + corr_He) * density * alpha / gamma12;
  if (chi < TINY) {
    return 0;
  }
  if (chi < 1e-5)
    return chi;

  //  this code, while mathematically accurate, is numerically buggy for very small x_HI, so i will use valid
  //  approximation x_HI <<1 above when x_HI < 1e-5, and this otherwise... the two converge seemlessly
  // get solutions of quadratic of chi (neutral fraction)
  b = -2 - gamma12 / (density * (1 + corr_He) * alpha);
  chi = (-b - sqrt(b * b - 4)) / 2.0; // correct root
  return chi;
}

/* returns the case B hydrogen recombination coefficient (Spitzer 1978) in cm^3 s^-1*/
double alpha_B(double T)
{
  return alphaB_10k * pow(T / 1.0e4, -0.75);
}
