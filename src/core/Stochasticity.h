#ifndef STOCHASTICITY_H
#define STOCHASTICITY_H

#include <stddef.h>

#include "meraxes.h"
#define NO_SFR_SFR_MIN_COUNT 1
#define NO_SHMR_LOG10_SFR_FLOOR (-30.0)

// Parameters for removing SFR--halo scatter.
// Source SFR tables use this halo-mass grid; GSM is accumulated per galaxy.
#define SFR_NTYPES 3
#define SFR_NX     376
#define SFR_XMIN   (-3.50)
#define SFR_XMAX   (4.00)
#define SFR_DX     ((SFR_XMAX - SFR_XMIN) / ((double)(SFR_NX - 1)))

#define SFR_INDEX(t,i) ((size_t)(t) * (size_t)SFR_NX + (size_t)(i))
void compute_fesc_recalibration_factors(void);
double compute_xray_recalibration_factor(double local_raw, double local_target);

void build_no_sfr_tables(int population);
void apply_no_sfr_treatment(int snapshot);
void compute_no_sfr_recalibration_factors(int population);

double extract_recalibration_factors(galaxy_t* gal, int population, bool use_gsm);

void no_sfr_sources_init(void);
void no_sfr_sources_free(void);

#endif