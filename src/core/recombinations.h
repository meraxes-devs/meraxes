/*
 * These are the relevant functions taken from the public version of 21cmFAST code
 * to compute inhomogeneous recombinations. Taken from "recombinations.c" written
 * by Andrei Mesinger and Emanuele Sobacchi (2013abc).
 *
 * Inclusion of this for Meraxes was written by Bradley Greig.
 */

#ifndef RECOMBINATIONS_H
#define RECOMBINATIONS_H

#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>

#include "meraxes.h"

// Warning: the calculation of the MHR model parameters is valid only from redshift 2 to A_NPTS+2
#define A_NPTS (int)(60)
#define C_NPTS (int)(12)
#define beta_NPTS (int)(5)

// number of points in redshift axis;  we will only interpolate over gamma, and just index sample in redshift
#define RR_Z_NPTS (int)(200)
#define RR_DEL_Z (float)(0.2)
#define RR_Z_END (float)(0.0) // added zend so that the 400 points span z=0, 0.2, ... 39.8.

// same, just index sample in temperature
#define RR_T_NPTS (int)(100)
#define RR_DEL_T (float)(0.03)
#define RR_T_STA (float)(2.0) // added zend so that the 300 points span T=1e2, 1e2.3, ... 1e4.97

// number of samples of gamma for the interpolation tables
#define RR_lnGamma_NPTS (int)(150)
#define RR_lnGamma_min (float)(-10.0)
#define RR_DEL_lnGamma (float)(0.1)

#ifdef __cplusplus
extern "C"
{
#endif

  double alpha_A(double T);
  double alpha_B(double T); // case B hydrogen recombination coefficient (Spitzer 1978) T in K
  double neutral_fraction(double density,
                          double T4,
                          double gamma12,
                          int usecaseB); // neutral fraction given H density (cm^-3), gas temperature (in 1e4 K), and
                                         // gamma12  (in 1e-12 s^-1). if usecase B, then use case B, otherwise case A
  int splined_recombination(double z_eff,
                            double gamma12_bg,
                            double temp,
                            double* recombination_rate,
                            double* residual_xH,
                            double* clumping_factor); // assumes case B
  double recombination_rate(double z_eff, double gamma12_bg, double T4, int usecaseB);
  double clumping_factor(double z_eff, double gamma12_bg, double T4, int usecaseB);
  double residual_neutral_hydrogen(double z_eff, double gamma12_bg, double T4, int usecaseB);
  void init_MHR(); /*initializes the lookup table for the PDF density integral in MHR00 model at redshift z*/
  void free_MHR(); /* deallocates the gsl structures from init_MHR */
  double Gamma_SS(double Gamma_bg, double Delta, double T_4, double z); // ionization rate w. self shielding
  double MHR_rr(double lnD, void* params);
  double A_MHR(double z);            /*returns the A parameter in MHR00model*/
  double C_MHR(double z);            /*returns the C parameter in MHR00model*/
  double beta_MHR(double z);         /*returns the beta parameter in MHR00model*/
  double splined_A_MHR(double z);    /*returns the splined A parameter in MHR00model*/
  double splined_C_MHR(double z);    /*returns the splined C parameter in MHR00model*/
  double splined_beta_MHR(double z); /*returns the splined beta parameter in MHR00*/
  void free_A_MHR();                 /* deallocates the gsl structures from init_A */
  void free_C_MHR();                 /* deallocates the gsl structures from init_C */
  void free_beta_MHR();              /* deallocates the gsl structures from init_beta */
  void init_A_MHR();                 /*initializes the lookup table for the A paremeter in MHR00 model*/
  void init_C_MHR();                 /*initializes the lookup table for the C paremeter in MHR00 model*/
  void init_beta_MHR();              /*initializes the lookup table for the beta paremeter in MHR00 model*/

#ifdef __cplusplus
}
#endif

#endif
