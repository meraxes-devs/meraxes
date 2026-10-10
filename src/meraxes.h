#ifndef _INIT_MERAXES
#define _INIT_MERAXES

#include <complex.h>
#include <fftw3.h>
#include <gsl/gsl_rng.h>
#include <hdf5.h>
#include <mlog.h>
#include <stdio.h>

/*
 * Definitions
 */

#include "meraxes_conf.h"

/*
 * Units (cgs)
 */

#define GRAVITY 6.672e-8
#define SOLAR_MASS 1.989e33
#define SOLAR_LUM 3.826e33
#define LOG_10_SOLAR_LUM 33.5827449657
#define RAD_CONST 7.565e-15
#define AVOGADRO 6.0222e23
#define BOLTZMANN 1.3806e-16
#define GAS_CONST 8.31425e7
#define SPEED_OF_LIGHT 2.9979e10 // [cm/s]
#define PLANCK 6.6262e-27        // [erg/s]
#define PROTONMASS 1.6726e-24    //[g]
#define HUBBLE 3.2407789e-18     // [h/sec]
#define SEC_PER_MEGAYEAR 3.155e13
#define SEC_PER_YEAR 3.155e7
#define MPC 3.086e24
#if USE_STOCHASTICITY
#define XRAY_LUMINOSITY_UNIT 1.0e40 // [erg/s]
#endif
#define TCMB 2.728
#define NU_over_EV (double)(1.60217646e-12 / PLANCK)
#define NUIONIZATION (double)(13.60 * NU_over_EV)     /* ionization frequency of H */
#define HeI_NUIONIZATION (double)(24.59 * NU_over_EV) /* ionization frequency of HeI */
#define HeII_NUIONIZATION (double)(NUIONIZATION * 4)  /* ionization frequency of HeII */
#define NU_LA (double)(10.2 * NU_over_EV)             /* frequency of Lyalpha */
#define NU_LW (double)(11.2 * NU_over_EV)             /* lower frequency of Lyman-Werner band */
#define NU_1450 (double)(SPEED_OF_LIGHT / 1450e-8)    /* frequency of 1450 Angstroms */
#define PLANCK_EV (double)(4.1357e-15)
#define T_RE (double)(1e4)
#define EDDINGTON_TIME_SCALE (double)(450.514890) // Eddington timescale in Megayears
#define SIGMA_T_CGS (double)(6.652e-25)
#define EnergySN (double)(1e51) // Energy of a single supernova in ergs
// Pop III stuff (Atm ENOVA_CC and ENOVA_PISN are the same but you could change)
#define ENOVA_CC (double)(1e51)
#define ENOVA_PISN (double)(1e51)

// Extra constants for emission-line calculations
#define HBAR_SI (PLANCK * 1e-7 / 2 / M_PI) // [J s]
#define BOLTZMANN_SI (BOLTZMANN * 1e-7)    // [J/K]
#define ELECTRON_MASS_SI 9.1093837015e-31  // [kg]
#define PLANCK_SI (PLANCK * 1e-7)          // [J s], converted from PLANCK [erg s]
#define ALPHA_HII 2.6e-13                  // [cm^3/s]
#define Z_SUN 0.0134                       // solar metallicity fraction
#define SN_PROGENITOR_MASS_MSUN 12.26      // [Msun]

// ======================================================
// Don't change these unless you know what you are doing!
#define STRLEN 256 //!< Default string length
#define REL_TOL (float)1e-5
#define ABS_TOL (float)1e-8
// ======================================================

// Define things used for aborting exceptions
#ifdef __cplusplus
extern "C"
{
#endif
  void myexit(int signum);
#ifdef __cplusplus
}
#endif
#define ABORT(sigterm)                                                                                                 \
  do {                                                                                                                 \
    fprintf(stderr, "\nIn file: %s\tfunc: %s\tline: %i\n", __FILE__, __FUNCTION__, __LINE__);                          \
    myexit(sigterm);                                                                                                   \
  } while (0)

//! Physics parameter values: duplicate these for Pop. III / Pop. II SF
typedef struct physics_params_t
{
  double SfEfficiency;
  double SfEfficiencyScaling;
  double SfCriticalSDNorm;
  double SfRecycleFraction;
  int SnModel;
  double SnReheatRedshiftDep;
  double SnReheatEff;
  double SnReheatLimit;
  double SnReheatScaling;
  double SnReheatScaling2;
  double SnReheatNorm;
  double SnMetalRetentionFraction;
  double SnEjectionRedshiftDep;
  double SnEjectionEff;
  double SnEjectionScaling;
  double SnEjectionScaling2;
  double SnEjectionNorm;
  double MaxCoolingMassFactor;
  int ReincorporationModel;
  double ReincorporationEff;
  double Yield;
  double RadioModeEff;
  double QuasarModeEff;
  double BlackHoleGrowthRate;
  double EddingtonRatio;
  double quasar_mode_scaling;
  double quasar_open_angle;
  double quasar_fobs;

  double SfCriticalSDNorm_III;
  double SfEfficiency_III;
  double SfEfficiencyScaling_III;
  double SfRecycleFraction_III;
  double SnReheatRedshiftDep_III;
  double SnReheatEff_III;
  double SnReheatLimit_III;
  double SnReheatScaling_III;
  double SnReheatScaling2_III;
  double SnReheatNorm_III;
  double SnEjectionRedshiftDep_III;
  double SnEjectionEff_III;
  double SnEjectionScaling_III;
  double SnEjectionScaling2_III;
  double SnEjectionNorm_III;
  double Yield_III;

  double ThreshMajorMerger;
  double MinMergerStellarMass;
  double MinMergerRatioForBurst;
  double MergerBurstScaling;
  double MergerBurstFactor;
  double MergerTimeFactor;

  // TODO: These parameters should be used to set the TOCF HII_EFF_FACTOR value
  double ReionEfficiency;
  double ReionEfficiencyIII;
  double ReionNionPhotPerBary;
  double ReionNionPhotPerBaryIII;
  double BlackHoleSeed;
  double BlackHoleMassLimitReion;
  double ReionTcool;
  double Y_He;

  // Pop III IMF parameters
  double MminIMF;
  double MmaxIMF;
  double AlphaIMF;
  double McharIMF; // for LogNorm
  double SigmaIMF;
  int PopIII_IMF;
  int PopIIIAgePrescription;

  // Parameters for metallicity evolution. These include the parameters for the fitting function of clustering and the
  // critical metallicity
  double AlphaCluster;
  double BetaCluster;
  double GammaCluster;
  double NormCluster;

  double ZCrit;

  // Parameters to describe the X-ray properties of the sources
  double LXrayGal;
  double XrayScatterDex;
  double NuXrayThreshold;
  double SpecIndexXrayGal;
  double LXrayGalIII;
  double SpecIndexXrayIII;
  double NuXraySoftCut;
  double NuXrayMax;

  int Flag_IncludeAGNXray; /* 0=no AGN, 1=soft+hard, 2=hard only, 3=soft only */
  double SpecIndexXrayAGNSoft;
  double SpecIndexXrayAGNHard;
  double SpecIndexUVAGNSoft; /* lambda > 912A (redward of/at the break) - LW band amplitude/shape */
  double SpecIndexUVAGNHard; /* lambda <= 912A (shortward of the break) - ionizing photon rate only */
  double AGNLWEfficiency;    /* scale factor on the AGN LW amplitude */

  double ReionMaxHeatingRedshift;

  double ReionGammaHaloBias;
  double ReionAlphaUV;
  double ReionAlphaUVBH;
  double ReionRBubbleMin;
  double ReionRBubbleMax;
  double ReionRBubbleMaxRecomb;

  double EscapeFracNorm;
  double EscapeFracNormIII;
  double EscapeFracRedshiftOffset;
  double EscapeFracRedshiftScaling;
  double EscapeFracPropScaling;
  double EscapeFracBHNorm;
  double EscapeFracBHScaling;
  double EscapeFracScatterDex;

  // CGM suppression of fesc
  double FescCGMSuppressionNorm;
  double FescCGMSuppressionScaling;
  double FescCGMGamma12Scaling;

  // global reionization prescription
  double ReionSobacchi_Zre;
  double ReionSobacchi_DeltaZre;
  double ReionSobacchi_DeltaZsc;
  double ReionSobacchi_T0;

  // global reionization prescription
  double ReionGnedin_z0;
  double ReionGnedin_zr;

  // filtering mass fit
  double ReionSMParam_m0;
  double ReionSMParam_a;
  double ReionSMParam_b;
  double ReionSMParam_c;
  double ReionSMParam_d;

  // options
  int EscapeFracDependency;
  int SfDiskVelOpt;
  int SfPrescription;
  bool InstantSfIII;

  // Flags
  int Flag_ReionizationModifier;
  int Flag_BHFeedback;
  int Flag_BHARExponentialCut;
  int Flag_IRA;
  int Flag_FixDiskRadiusOnInfall;
  int Flag_FixVmaxOnInfall;
  int Flag_ReheatToFOFGroupTemp;
  int Flag_FescCGMSuppression;
  int Flag_RemoveSFRScatter;
  int Flag_SourceRecalibration;
} physics_params_t;

enum tree_ids
{
  VELOCIRAPTOR_TREES,
  GBPTREES_TREES,
  VELOCIRAPTOR_TREES_AUG
};

//! Run params
//! Everything in this structure is supplied by the user...
typedef struct run_params_t
{
  char DefaultsFile[STRLEN];
  char SimParamsFile[STRLEN];
  char OutputDir[STRLEN];
  char FileNameGalaxies[STRLEN];
  char SimName[STRLEN];
  char SimulationDir[STRLEN];
  char CatalogFilePrefix[STRLEN];
  char OutputSnapsString[STRLEN];
  char PhotometricTablesDir[STRLEN];
  char TargetSnaps[STRLEN];
  char BetaBands[STRLEN];
  char RestBands[STRLEN];
  double BirthCloudLifetime;
  double DustMetallicityScale;
  double DustTauUVISM;
  double DustNISM;
  double DustTauUVBC;
  double DustNBC;
  double DustAZ;
  double DeltaT; // New Parameter added to consider different time of observation! Very important for Pop. III
  char CoolingFuncsDir[STRLEN];
  char RecombinationDir[STRLEN];
  char StellarFeedbackDir[STRLEN];
  char TablesForXHeatingDir[STRLEN];
  char IMF[STRLEN];
  char MagSystem[STRLEN];
  char MagBands[STRLEN];
  char ForestIDFile[STRLEN];
  char MvirCritFile[STRLEN];
  char MvirCritMCFile[STRLEN];
  char MassRatioModifier[STRLEN];
  char BaryonFracModifier[STRLEN];
  char FFTW3WisdomDir[STRLEN];
  char SFRTableFile[STRLEN]; // sfr atable
  physics_params_t physics;

  double BoxSize;
  double VolumeFactor;
  double Hubble_h;
  double BaryonFrac;
  double OmegaM;
  double OmegaK;
  double OmegaR;
  double OmegaLambda;
  double Sigma8;
  double wLambda;
  double SpectralIndex;
  double PartMass;
  long long NPart;

  double* MvirCrit;
  double* MvirCrit_MC;

  double ReionDeltaRFactor;
  double ReionPowerSpecDeltaK;
  int ReionGridDim;
  int ReionFilterType;
  int TsHeatingFilterType;
  int ReionRtoMFilterType;
  int ReionUVBFlag;
  int MetalGridDim;

  enum tree_ids TreesID;
  int FirstFile;
  int LastFile;
  int NSteps;
  int SnaplistLength;
  int RandomSeed;
  int FlagSubhaloVirialProps;
  int FlagInteractive;
  int FlagMCMC;
  int Flag_PatchyReion;
  int Flag_IncludeSpinTemp;
  int Flag_IncludeLymanWerner;
  int Flag_IncludeStreamVel;
  int Flag_IncludeMetalEvo; // New for Metallicity
  int Flag_IncludeRecombinations;
  int Flag_EvolvingReionRBubbleMax;
  int Flag_InstantaneousSFR;
  int Flag_TemperatureDependentRec;
  int Flag_Compute21cmBrightTemp;
  int Flag_ComputePS;
  int Flag_IncludePecVelsFor21cm;
  int Flag_ConstructLightcone;

  int TsVelocityComponent;
  int TsNumFilterSteps;

  double ReionSfrTimescale;

  double EndRedshiftLightcone;
  int EndSnapshotLightcone;
  long long LightconeLength;
  long long CurrentLCPos;
  int PS_Length;
  int Flag_OutputGrids;
  int Flag_OutputGridsPostReion;
  int Flag_OutputHMF;      //!< Flag to enable/disable HMF output
  double HMF_MinMass;      //!< Minimum log10(M) for HMF bins (solar masses/h)
  double HMF_MaxMass;      //!< Maximum log10(M) for HMF bins
  int HMF_BinsPerDex;      //!< Number of HMF bins per dex
  int Flag_OutputSMF;      //!< Flag to enable/disable Stellar Mass Function output
  double SMF_MinMass;      //!< Minimum log10(M*) for SMF bins (solar masses)
  double SMF_MaxMass;      //!< Maximum log10(M*) for SMF bins
  int SMF_BinsPerDex;      //!< Number of SMF bins per dex
  int Flag_OutputUVLF;     //!< Flag to enable/disable UV Luminosity Function output
  double UVLF_MinMag;      //!< Minimum magnitude for UVLF bins
  double UVLF_MaxMag;      //!< Maximum magnitude for UVLF bins
  int UVLF_BinsPerMag;     //!< Number of UVLF bins per magnitude (linear bins)
  int Flag_OutputDustyLF;  //!< Flag to enable/disable Dusty UV Luminosity Function output
  int Flag_OutputQuasarLF; //!< Flag to enable/disable Quasar UV Luminosity Function output
  int Flag_OutputOIIILF;   //!< Flag to enable/disable OIII Luminosity Function output
  double OIIILF_MinLogL;   //!< Minimum log10(LOIII [erg/s]) for OIIILF bins
  double OIIILF_MaxLogL;   //!< Maximum log10(LOIII [erg/s]) for OIIILF bins
  int OIIILF_BinsPerDex;   //!< Number of OIIILF bins per dex
  int Flag_OutputXrayLF;   //!< Flag to enable/disable X-ray Luminosity Function output
  double XrayLF_MinLogL;   //!< Minimum log10(LX [erg/s]) for XrayLF bins (e.g. 40.0)
  double XrayLF_MaxLogL;   //!< Maximum log10(LX [erg/s]) for XrayLF bins (e.g. 48.0)
  int XrayLF_BinsPerDex;   //!< Number of XrayLF bins per dex
  int FlagIgnoreProgIndex;
} run_params_t;

typedef struct run_units_t
{
  double UnitTime_in_s;
  double UnitLength_in_cm;
  double UnitVelocity_in_cm_per_s;
  double UnitTime_in_Megayears;
  double UnitMass_in_g;
  double UnitDensity_in_cgs;
  double UnitPressure_in_cgs;
  double UnitCoolingRate_in_cgs;
  double UnitEnergy_in_cgs;
  // TOTAL : 72  (must be multiple of 8)
} run_units_t;

typedef struct hdf5_output_t
{
  char** params_tag;
  void** params_addr;
  int* params_type;
  size_t* dst_offsets;
  size_t* dst_field_sizes;
  const char** field_names;
  const char** field_units;
  const char** field_h_conv;
  hid_t* field_types;
  size_t dst_size;
  hid_t array3f_tid; // sizeof(hid_t) = 4
  hid_t array_nmag_f_tid;
  hid_t array_nhist_f_tid;
  int n_props;
  int params_count;

  // TOTAL : 52 + 4 padding (must be multiple of 8)
} hdf5_output_t;

#if USE_MINI_HALOS
typedef struct metal_grids_t // New stuff for MetalEvo, probably you will need to add new stuff
{
  ptrdiff_t* slab_nix_metals;
  ptrdiff_t* slab_ix_start_metals;

  float* buffer_metals;

  struct gal_to_slab_t* galaxy_to_slab_map_metals;

  double volume_ave_ZIGM;        // Is it necessary? Maybe this one could be used as a log message
  double volume_ave_mass_metals; // Is it necessary?

  int buffer_size_metals;

  float* N_bubbles;
  float* mass_IGM; // It's the total! Computed from the overdensity and you sum the galaxy net contribution (mass_gas)
  float* mass_metals;
  float* mass_gas; // The one coming from galaxies (Ejected - Hot - Cold)
  float* Zigm_box;
  float* Probability_metals;
  float* R_ave; // Average radius in the bubble
  float* R_max; // Max radius in the bubble

} metal_grids_t;
#endif

typedef struct reion_grids_t
{
  ptrdiff_t* slab_nix;
  ptrdiff_t* slab_ix_start;
  ptrdiff_t* slab_n_complex;

  float* buffer;

  float* stars;
  fftwf_complex* stars_unfiltered;
  fftwf_complex* stars_filtered;
  fftwf_plan stars_forward_plan;
  fftwf_plan stars_filtered_reverse_plan;

  float* effective_bhm;
  fftwf_complex* effective_bhm_unfiltered;
  fftwf_complex* effective_bhm_filtered;
  fftwf_plan effective_bhm_forward_plan;
  fftwf_plan effective_bhm_filtered_reverse_plan;
  float* effective_bhar;
  fftwf_complex* effective_bhar_unfiltered;
  fftwf_complex* effective_bhar_filtered;
  fftwf_plan effective_bhar_forward_plan;
  fftwf_plan effective_bhar_filtered_reverse_plan;

  float* deltax;
  fftwf_complex* deltax_unfiltered;
  fftwf_complex* deltax_filtered;
  fftwf_plan deltax_forward_plan;
  fftwf_plan deltax_filtered_reverse_plan;

  float* sfr;
  float* sfr_histories;
  float* weighted_sfr;
  fftwf_complex* sfr_unfiltered;
  fftwf_complex* sfr_filtered;
  fftwf_complex* weighted_sfr_unfiltered;
  fftwf_complex* weighted_sfr_filtered;
  fftwf_plan sfr_forward_plan;
  fftwf_plan weighted_sfr_forward_plan;
  fftwf_plan sfr_filtered_reverse_plan;
  fftwf_plan weighted_sfr_filtered_reverse_plan;

#if USE_STOCHASTICITY
  float* xray_luminosity;           // Cell luminosity in units of 1e40 erg/s
  float* xray_luminosity_histories; // Stored luminosity grids in units of 1e40 erg/s
  fftwf_complex* xray_luminosity_unfiltered;
  fftwf_complex* xray_luminosity_filtered;
  fftwf_plan xray_luminosity_forward_plan;
  fftwf_plan xray_luminosity_filtered_reverse_plan;
#endif

#if USE_MINI_HALOS
  float* starsIII;
  fftwf_complex* starsIII_unfiltered;
  fftwf_complex* starsIII_filtered;
  fftwf_plan starsIII_forward_plan;
  fftwf_plan starsIII_filtered_reverse_plan;

  float* sfrIII;
  float* sfrIII_histories;
  float* weighted_sfrIII;
  fftwf_complex* sfrIII_unfiltered;
  fftwf_complex* sfrIII_filtered;
  fftwf_complex* weighted_sfrIII_unfiltered;
  fftwf_complex* weighted_sfrIII_filtered;
  fftwf_plan sfrIII_forward_plan;
  fftwf_plan weighted_sfrIII_forward_plan;
  fftwf_plan sfrIII_filtered_reverse_plan;
  fftwf_plan weighted_sfrIII_filtered_reverse_plan;
#endif

  float* xH;
  float* z_at_ionization;
  float* J_21_at_ionization;
  float* Mvir_crit;
#if USE_MINI_HALOS
  float* Mvir_crit_MC;
#endif
  float* r_bubble;

  // Grids necessary for the IGM spin temperature
  float* x_e_box;
  fftwf_complex* x_e_unfiltered;
  fftwf_complex* x_e_filtered;
  fftwf_plan x_e_box_forward_plan;
  fftwf_plan x_e_filtered_reverse_plan;

  float* x_e_box_prev;
  float* Tk_box;
  float* Tk_box_prev;
  float* TS_box;

#if USE_MINI_HALOS
  float* Tk_boxII;
  float* Tk_box_prevII;
  float* TS_boxII;
#endif

  double* SMOOTHED_SFR_GAL;
#if USE_STOCHASTICITY
  double* SMOOTHED_XRAY_LUMINOSITY_GAL; // Per-cell luminosity density per shell [erg/s/cm^3]
#endif
  double* SMOOTHED_AGN_hard; //!< per-cell AGN X-ray luminosity density per shell [erg/s/cm^3] (hard band)
  double* SMOOTHED_AGN_soft; //!< per-cell AGN X-ray luminosity density per shell [erg/s/cm^3] (soft band)
#if USE_MINI_HALOS
  double* SMOOTHED_SFR_III;
  double* SMOOTHED_AGN_UV; //!< per-cell AGN UV luminosity density per shell [1e21 erg/s/Hz/cm^3]
#endif

  float* BHXrayEmissivity_hard; //!< Per-cell AGN X-ray emissivity grid (current snapshot, hard band) [slab_n_complex*2]
  float* bh_xray_histories_hard; //!< Ring-buffer of NstoreSnapshots_Heating past BHXrayEmissivity_hard snapshots
  float* BHXrayEmissivity_soft; //!< Per-cell AGN X-ray emissivity grid (current snapshot, soft band) [slab_n_complex*2]
  float* bh_xray_histories_soft; //!< Ring-buffer of NstoreSnapshots_Heating past BHXrayEmissivity_soft snapshots
#if USE_MINI_HALOS
  float* BHUVEmissivity;  //!< Per-cell AGN UV emissivity grid (current snapshot) [slab_n_complex*2]
  float* bh_uv_histories; //!< Ring-buffer of NstoreSnapshots_Heating past BHUVEmissivity snapshots
#endif

  fftwf_complex* BHXrayEmissivity_hard_unfiltered;
  fftwf_complex* BHXrayEmissivity_hard_filtered;
  fftwf_plan BHXrayEmissivity_hard_forward_plan;
  fftwf_plan BHXrayEmissivity_hard_filtered_reverse_plan;
  fftwf_complex* BHXrayEmissivity_soft_unfiltered;
  fftwf_complex* BHXrayEmissivity_soft_filtered;
  fftwf_plan BHXrayEmissivity_soft_forward_plan;
  fftwf_plan BHXrayEmissivity_soft_filtered_reverse_plan;
#if USE_MINI_HALOS
  fftwf_complex* BHUVEmissivity_unfiltered;
  fftwf_complex* BHUVEmissivity_filtered;
  fftwf_plan BHUVEmissivity_forward_plan;
  fftwf_plan BHUVEmissivity_filtered_reverse_plan;
#endif

  // Grids necessary for LW background and future disentangling between MC/AC Pop3/Pop2 stuff

#if USE_MINI_HALOS
  float* JLW_box;
  float* JLW_boxII;
#endif

  // Grids necessary for inhomogeneous recombinations
  float* z_re;
  float* temp_kinetic_all_gas;

  float* N_rec;
  fftwf_complex* N_rec_unfiltered;
  fftwf_complex* N_rec_filtered;
  fftwf_plan N_rec_forward_plan;
  fftwf_plan N_rec_filtered_reverse_plan;

  float* Gamma12;
  float* residual_xH;
  float* clumping_factor;
  float* t_resp; // Relaxation timescale grid: 1/(Gamma + alpha_B * n_e)

  // Grids necessary for the 21cm brightness temperature
  float* delta_T;
  float* delta_T_prev;
#if USE_MINI_HALOS
  float* delta_TII;
  float* delta_TII_prev;
#endif
  float* vel;
  fftwf_complex* vel_gradient;
  fftwf_plan vel_forward_plan;
  fftwf_plan vel_gradient_reverse_plan;

  // Grid for the lightcone (cuboid) box
  float* LightconeBox;
  float* Lightcone_redshifts;

  // Data for the power spectrum
  float* PS_k;
  float* PS_data;
  float* PS_error;

#if USE_MINI_HALOS
  float* PSII_data;
  float* PSII_error;
#endif

  struct gal_to_slab_t* galaxy_to_slab_map;

  double volume_weighted_global_xH;
  double volume_weighted_global_Gamma12;
  double volume_weighted_global_r_bubble;
  double volume_weighted_global_temp_kinetic_all_gas;
  double volume_weighted_global_N_rec;
  double volume_weighted_global_residual_xH;
  double volume_weighted_global_clumping_factor;
  double volume_weighted_global_t_resp;
  double volume_weighted_global_weighted_sfr;
  double volume_weighted_global_effective_bhar;

  double mass_weighted_global_xH;
  double mass_weighted_global_Gamma12;
  double mass_weighted_global_r_bubble;
  double mass_weighted_global_temp_kinetic_all_gas;
  double mass_weighted_global_N_rec;
  double mass_weighted_global_residual_xH;
  double mass_weighted_global_clumping_factor;
  double mass_weighted_global_t_resp;

  // Thomson optical depth tracking based on mass-weighted ionization history.
  double mass_weighted_global_tau_e;
  double mass_weighted_global_tau_e_sim;
  int tau_e_prev_snapshot;
  double tau_e_prev_mass_weighted_xHII;

  double volume_ave_J_alpha;
  double volume_ave_xalpha;
  double volume_ave_Xheat;
  double volume_ave_Xion;
  double volume_ave_Xheat_AGN_soft;
  double volume_ave_Xheat_AGN_hard;
  double volume_ave_TS;
  double volume_ave_TK;
  double volume_ave_xe;
  double volume_ave_Tb;
#if USE_MINI_HALOS
  double volume_weighted_global_weighted_sfrIII;
  double volume_ave_J_alphaII;
  double volume_ave_J_LW;
  double volume_ave_J_LWII;
  double volume_ave_J_LW_AGN;
  double volume_ave_XheatII;
  double volume_ave_XionII;
  double volume_ave_TSII;
  double volume_ave_TKII;
  double volume_ave_TbII;
#endif

  int started;
  int finished;
  int buffer_size;
  bool flag_wisdom;
} reion_grids_t;

typedef struct galaxy_t
{
  double NewStars[N_HISTORY_SNAPS];
#if USE_MINI_HALOS
  double NewStars_II[N_HISTORY_SNAPS]; // New
  double NewStars_III[N_HISTORY_SNAPS];
#endif
  double NewMetals[N_HISTORY_SNAPS];

#ifdef CALC_MAGS
  double inBCFlux[MAGS_N];
  double outBCFlux[MAGS_N];
#if USE_MINI_HALOS
  double inBCFluxIII[MAGS_N];
  double outBCFluxIII[MAGS_N];
#endif
#endif

  // Unique ID for the galaxy
  unsigned long ID;

  // properties of subhalo at the last time this galaxy was a central galaxy
  float Pos[3];
  float Vel[3];
  double Mvir;
  double Rvir;
  double Vvir;
  double Vmax;
  double Spin;

  double dt; //!< Time between current snapshot and last identification

  struct halo_t* Halo;
  struct galaxy_t* FirstGalInHalo;
  struct galaxy_t* NextGalInHalo;
  struct galaxy_t* Next;
  struct galaxy_t* MergerTarget;

  // baryonic reservoirs
  double HotGas;
  double MetalsHotGas;
  double ColdGas;
  double MetalsColdGas;
  double H2Frac;
  double H2Mass;
  double HIMass;
  double Mcool;
  double StellarMass;
  double GrossStellarMass;
  double Fesc;
  double FescWeightedGSM;
  double FescWeightedSfr;
  double MetalsStellarMass;
  double DiskScaleLength;
  double Sfr;
  double LOIII; //!< [O III] luminosity [1e40 erg/s]
  double ionization_param;
  double EjectedGas;
  double MetalsEjectedGas;
  double BlackHoleMass;
  double FescBH;
  double BHemissivity;
  double QuasarLuv; //!< UV luminosity LUV of quasar (1e10 Lsun, summable for mergers)
  double QuasarLX;  //!< Intrinsic hard X-ray luminosity [1e10 Lsun]; 0 if inactive
  int NHbin;        //!< Which of the 5 NH bins this snapshot's stochastic draw landed in (0-4; logNH
                    //!< 20-21/21-22/22-23/23-24/24-26 CTK), or -1 if no AGN.
  //!< This (bin, luminosity) pair replaces the old QuasarLX_obs0-4/NHfrac0-4 fields (5 mostly-zero doubles
  //!< each, every galaxy).
  double BHXrayEmissivity_hard; //!< Observed hard X-ray emissivity [1e10 Lsun], obscuration-weighted
  double BHXrayEmissivity_soft; //!< Observed soft X-ray emissivity [1e10 Lsun], obscuration-weighted
  double EffectiveBHM;
  double EffectiveBHAR;
  double DutyCycleAGN;
  double BlackHoleAccretedHotMass;
  double BlackHoleAccretedColdMass;
  double BlackHoleAccretingColdMass;
  double BHAccretionOnTime; //!< Random on-time fraction [0, 1] for when accretion starts within snapshot; -1 indicates
                            //!< no prior accretion
  double t_resp;            //!< Local relaxation timescale (in Myr)
#if USE_STOCHASTICITY
  // Alternative stellar sources with the SFR--halo scatter removed.
  // Source GSM accumulates SfrNoScatter * dt; SfrNoScatter is snapshot-local.
  double GrossStellarMassNoScatter;
  double SfrNoScatter;
  // StochasticityTreated means adding scatter to Fesc or using the treated source properties
  double StochasticityTreatedFescWeightedGSM;
  double StochasticityTreatedFescWeightedSfr;
#endif
#if USE_MINI_HALOS
  // Differentiation Pop III / Pop II
  double SfrIII;
  double StellarMass_II;
  double StellarMass_III;
  double GrossStellarMassIII;
  double FescIII;
  double FescIIIWeightedGSM;
  double FescIIIWeightedSfr;
#if USE_STOCHASTICITY
  double GrossStellarMassIIINoScatter;
  double FescIIIWeightedGSMNoScatter;
  double SfrIIINoScatter;
  // Pop III prepared/reference sources mirror the Pop II Target fields.
  double StochasticityTreatedFescIIIWeightedGSM;
  double StochasticityTreatedFescIIIWeightedSfr;
#endif

  double Remnant_Mass; // Coming from Pop III with M between 40 and 140 and larger than 260 Msol and remnant of CCSN
                       // [8,40]Msun. Atm those are silent.

  int
    Galaxy_Population; // Disentangles Pop III/Pop II; not needed when Mini_halos is off, since all galaxies are Pop II

  double MvirCrit_MC;

  // Metallicity stuff
  double RmetalBubble; // New for MetalEvo
  double PrefactorBubble;
  double TimeBubble;
  double Metal_Probability;    // Probability to be polluted by other metal bubbles
  double GalMetal_Probability; // Random number between 0 and 1 associated to the galaxy.
  double Metals_IGM;
  double Gas_IGM;
  double Metallicity_IGM; //
  double MaxBubble;       // Need this for Boost probability
  double AveBubble;   // Same (you will likely use only one of the two). You could actually save only the boost factor
  int Flag_ExtMetEnr; // 0 if not enriched from a bubble, 1 yes.

  double Prefactor[N_HISTORY_SNAPS]; // here you store the prefactors of the metal bubbles
  double Times[N_HISTORY_SNAPS];     // Time at which the SN explode!
  double Radii[N_HISTORY_SNAPS];
#endif

  // baryonic hostories
  double mwmsa_num;
  double mwmsa_denom;

  // misc
  double Rcool;
  double Cos_Inc;
  double MergTime;
  double MergerStartRadius;
  double BaryonFracModifier;
  double FOFMvirModifier;
  double MvirCrit;
  double tau_cgm;
  double cumulative_ionization; //!< Integrated Gamma12 * dt for CGM suppression mode 2
  double MergerBurstMass;

  int Type;
  int OldType;
  int Len;
  int MaxLen;
  int SnapSkipCounter;
  int HaloDescIndex;
  int TreeFlags;
  int LastIdentSnap; //!< Last snapshot at which the halo in which this galaxy resides was identified
  int output_index;  //!< write index

  bool ghost_flag;

  // N.B. There will be padding present at the end of this struct, but amount
  // is dependent on CALC_MAGS, MAX_PHOTO_NBANDS, NOUT and N_HISTORY_SNAPS.
} galaxy_t;

//! The meraxes halo structure
typedef struct halo_t
{
  struct fof_group_t* FOFGroup;
  struct halo_t* NextHaloInFOFGroup;
  galaxy_t* Galaxy;

  float Pos[3]; //!< Most bound particle position [Mpc/h]
  float Vel[3]; //!< Centre of mass velocity [Mpc/h]
  float AngMom; //!< Specific angular momentum length [Mpc/h *km/s]

  double Mvir; //!< virial mass [M_sol/h]
  double Rvir; //!< Virial radius [Mpc/h]
  double Vvir; //!< Virial velocity [km/s]

  float Vmax;       //!< Maximum circular velocity [km/s]
  unsigned long ID; //!< Halo ID
  int Type;         //!< Type (0 for central, 1 for satellite)
  int SnapOffset;   //!< Number of snapshots this halo skips before reappearing
  int DescIndex;    //!< Index of descendant in next relevant snapshot
  int ProgIndex;    //!< Index of progenitor in previous relevant snapshot
  int TreeFlags;    //!< Bitwise flag indicating the type of match in the trees
  int Len;          //!< Number of particles in the structure
} halo_t;

typedef struct fof_group_t
{
  halo_t* FirstHalo;
  halo_t* FirstOccupiedHalo;
  double Mvir;
  double Rvir;
  double Vvir;
  double FOFMvirModifier;
  int TotalSubhaloLen;
} fof_group_t;

//! Tree info struct
typedef struct trees_info_t
{
  int n_halos;
  int n_halos_max;
  int max_tree_id;
  int n_fof_groups;
  int n_fof_groups_max;
} trees_info_t;

// This structure carries the information about
//   the GPU allocated to this CPU's scope. It
//   needs to be declared by all compilers since
//   it is used by run_globals.
#ifdef USE_CUDA
#include <cuda_runtime.h>
typedef struct gpu_info
{
  int device;                       // the ordinal of the current context's device
  struct cudaDeviceProp properties; // Properties of this context's assigned device
  int n_threads;                    // No. of threads to use in kernal calls
  int n_contexts;                   // No. of ranks with successfully allocated GPU contexts
} gpu_info;
#else
typedef char gpu_info;
#endif

#ifdef CALC_MAGS
typedef struct mag_params_t
{
  int targetSnap[MAGS_N_SNAPS];
  int nBeta;
  int nRest;
  int minZ;
  int maxZ;
  int nMaxZ;
  double tBC;
  int iAgeBC[MAGS_N_SNAPS];
  size_t totalSize;
  double* working;
  double* inBC;
  double* outBC;
  double* centreWaves;
  double allcentreWaves[MAGS_N_SNAPS][MAGS_N_BANDS];
  double* logWaves;
#ifdef USE_MINI_HALOS
  size_t totalSizeIII;
  double* workingIII;
#endif
} mag_params_t;
#endif

typedef struct loiii_params_t
{
  double oxygen_abundance_over_z_sun;
  double excitation_rate;
  double branching_ratio;
} loiii_params_t;

//! Global variables which will will be passed around
typedef struct run_globals_t
{
  struct run_params_t params;
  char FNameOut[STRLEN];
  hid_t output_file_id; /*!< per-rank galaxy HDF5 file, kept open across all snapshots to avoid
                             HDF5 1.10.x free-space fragmentation bug (duplicate entry in cache) */
  reion_grids_t reion_grids;
#if USE_MINI_HALOS
  metal_grids_t metal_grids;
#endif
  struct run_units_t units;
  hdf5_output_t hdf5props;

  MPI_Comm mpi_comm;
  int mpi_rank;
  int mpi_size;
  gpu_info* gpu;

  double* AA;
  double* ZZ;
  double* LTTime;
  double* rhocrit;
  long* RequestedForestId;
  int RequestedMassRatioModifier;
  int RequestedBaryonFracModifier;
  int* ListOutputSnaps;
  halo_t** SnapshotHalo;
  fof_group_t** SnapshotFOFGroup;
  int** SnapshotIndexLookup;
  float** SnapshotDeltax;
  float** SnapshotVel;
  trees_info_t* SnapshotTreesInfo;
  galaxy_t* FirstGal;
  galaxy_t* LastGal;
  gsl_rng* random_generator;
  void* mhysa_self;
  double Hubble;
  double tau_e_postEoR;
  double RhoCrit;
  double G;
  double Csquare;
  double EddingtonTimescale;
  loiii_params_t loiii_params;
  // PopIII stuff

  double IMFnorm;
  double NumberPISN;
  double MassPISN;
  double NumberSNII;
  double MassSNII;
  double MassBHs;

  float* Mass_Values;
  float* Time_Values;

#if USE_STOCHASTICITY
  // The SFR source table holds only the current snapshot: size SFR_NTYPES * SFR_NX.
  float* SFRs;
  double* no_sfr_gsm_stochasticity_calibrations;
  double* fesc_stochasticity_calibrations;
  double* no_sfr_sfr_stochasticity_calibrations;
#if USE_MINI_HALOS
  // Independent Pop III SFR table with the same halo-mass layout.
  float* SFRsIII;
  double* no_sfr_gsm_stochasticity_calibrations_iii;
  double* no_sfr_sfr_stochasticity_calibrations_iii;
#endif
#endif
#ifdef CALC_MAGS
  struct mag_params_t mag_params;
  int loiii_rest_band_mag_index;
#endif

  int NOutputSnaps;
  int LastOutputSnap;
  int NGhosts;
  int NHalosMax;
  int NFOFGroupsMax;
  int NRequestedForests;
  int NStoreSnapshots;
  int NstoreSnapshots_Heating;

  bool SelectForestsSwitch;
  struct Modifier* mass_ratio_modifier;
  struct Modifier* baryon_frac_modifier;
} run_globals_t;
#ifdef _MAIN
run_globals_t run_globals;
#else
extern run_globals_t run_globals;
#endif

/*
 * Functions
 */

#ifdef __cplusplus
extern "C"
{
#endif

  // core/dracarys.c
  void dracarys(void);

  // core/read_parameter_file.c
  void read_parameter_file(char* fname, int mode);

  // core/init.c
  void init_storage(void);
  void init_meraxes(void);

  int set_sfr_history(void);
  // core/cleanup.c
  void cleanup(void);

  // core/magnitudes.c
  void get_output_magnitudes(float* mags, float* dusty_mags, galaxy_t* gal, int snapshot);
  void get_output_magnitudesIII(float* mags, galaxy_t* gal, int snapshot);

// MCMC related
// meraxes_mhysa_hook must be implemented by the calling code (Mhysa)!
#ifdef _MAIN
  int (*meraxes_mhysa_hook)(void* self, int snapshot, int ngals);
#else
extern int (*meraxes_mhysa_hook)(void* self, int snapshot, int ngals);
#endif

#ifdef USE_CUDA
// This stuff is needed by the GPU routines.  This
//    needs to be included after mlog.h is included
//    and after ABORT(), myexit() & run_globals are
//    defined, since they are used within.
#include "meraxes_gpu.h"
#endif

#ifdef __cplusplus
}
#endif

#endif // _INIT_MERAXES
