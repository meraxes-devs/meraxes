# Inputs and configuration

Meraxes requires halo merger trees, a snapshot list, physical tables and run
parameters. Merger trees are supplied by the N-body simulation. Spatial
radiation calculations also require simulation density grids.

## Input files

| Input | Location or contents |
|---|---|
| Snapshot times | `SimulationDir/a_list.txt`: expansion factors in snapshot order. |
| Cooling | `CoolingFuncsDir/SD93.hdf5`: metallicity-dependent cooling curves. |
| Stellar feedback | `StellarFeedbackDir/stellar_feedback_tables.hdf5`: ages, mass return, metal yields and energy. |
| Thermal evolution | `TablesForXHeatingDir`: recombination history, stellar spectra, collision rates and secondary-ionization tables. |
| Photometry | `PhotometricTablesDir/sed_library.hdf5`, enabled filters and Pop. III SEDs when applicable. |
| Recombination cache | `RecombinationDir`; missing interpolation tables are generated. |


## Parameter files

Use one case-sensitive `Name : value` entry per line and `#` for comments.
The priority is **run parameter file → `SimParamsFile` → `DefaultsFile`**;
the first occurrence of each parameter wins. Relative paths start from the
run directory.

```text
OutputDir       : ./output
FileNameGalaxies : meraxes
OutputSnapshots : :
```

`OutputSnapshots` accepts individual indices, comma-separated lists and
`start:stop` ranges with an exclusive upper bound. `:` selects all snapshots;
`-1` selects the last. Intermediate snapshots are always evolved. Select a
final snapshot greater than 1 for stellar-feedback initialization.

### Files and outputs

| Parameter | Default | Meaning / units |
|---|---|---|
| `DefaultsFile` | Required | Default parameter file. |
| `SimParamsFile` | Empty / optional | Simulation parameters, data locations and cosmology. |
| `SimulationDir` | Required | Root of the external simulation data. |
| `CatalogFilePrefix` | Required | Catalogue basename interpreted by the selected reader. |
| `CoolingFuncsDir` | Required | Directory containing `SD93.hdf5`. |
| `StellarFeedbackDir` | Required | Directory containing `stellar_feedback_tables.hdf5`. |
| `TablesForXHeatingDir` | Required | Directory of thermal and radiative lookup tables. |
| `RecombinationDir` | Empty / optional | Recombination interpolation cache directory. |
| `PhotometricTablesDir` | Required with `CALC_MAGS` | Stellar SEDs and observed filters. |
| `FileNameGalaxies` | Required | Output basename; the template uses `meraxes`. |
| `OutputDir` | Required | Output directory; create parent directories first. |
| `OutputSnapshots` | Required | Saved snapshot indices or ranges. |
| `FFTW3WisdomDir` | Empty / optional | Directory for FFTW planning wisdom. |

### Simulation and units

Use the cosmology and units of the input simulation.

| Parameter | Meaning / units |
|---|---|
| `SimName` | Descriptive simulation name. |
| `TreesID` | `0`: VELOCIraptor; `1`: gbpTrees; `2`: augmented VELOCIraptor. |
| `BoxSize` | Comoving box side, cMpc/h. |
| `PartMass` | Dark-matter particle mass in internal mass units. |
| `NPart` | Total particle count (64-bit integer). |
| `Hubble_h` | Hubble constant in units of 100 km s⁻¹ Mpc⁻¹. |
| `BaryonFrac` | Universal baryon-to-total-matter mass fraction. |
| `OmegaM` | Present matter density parameter. |
| `OmegaK` | Curvature density parameter for galaxy virial and time calculations. |
| `OmegaR` | Radiation density parameter for IGM Hubble and growth calculations. |
| `OmegaLambda` | Cosmological-constant density parameter. |
| `Sigma8` | Linear fluctuation normalization. |
| `SpectralIndex` | Primordial spectral index. |
| `wLambda` | Dark-energy equation-of-state parameter for the IGM growth helper. |
| `UnitLength_in_cm` | Internal length scale in cm. |
| `UnitMass_in_g` | Internal mass scale in g. |
| `UnitVelocity_in_cm_per_s` | Internal velocity scale in cm s⁻¹. |


## Model parameters

The tables below give `defaults.par` values. Pop. III and minihalo settings
require `USE_MINI_HALOS`; photometry requires `CALC_MAGS`; source scatter,
noSFR and recalibration require `USE_STOCHASTICITY`.

Parentheses abbreviate optional Pop. III counterparts: `SfEfficiency(_III)`
means `SfEfficiency` and `SfEfficiency_III`; `EscapeFracNorm(III)` means
`EscapeFracNorm` and `EscapeFracNormIII`. The III quantity requires
`USE_MINI_HALOS`. A single default applies to both; differing III defaults
are given in parentheses. Omit the parentheses in parameter files.

<details>
<summary>Execution controls</summary>

| Parameter | Default | Meaning / units |
|---|---|---|
| `NSteps` | `1` | Galaxy substeps; must be 1. |
| `FlagInteractive` | `0` | 0: standard; 1: interactive; 2: distribution functions without galaxy records. |
| `FlagSubhaloVirialProps` | `0` | Use catalogue virial properties for central subhalos in gbpTrees. |
| `FlagMCMC` | `0` | External MCMC mode; suppress ordinary outputs. |
| `FlagIgnoreProgIndex` | `0` | Skip progenitor-index tracking. |
| `RandomSeed` | `1809` | Random-generator seed. |
| `VolumeFactor` | `1.0` | Effective-volume correction for subsampled trees. |

</details>

<details>
<summary>Radiation execution flags</summary>

| Parameter | Default | Meaning / units |
|---|---|---|
| `Flag_IncludeRecombinations` | `0` | Enable inhomogeneous hydrogen recombinations. |
| `Flag_EvolvingReionRBubbleMax` | `1` | Use the evolving maximum bubble radius. |
| `Flag_TemperatureDependentRec` | `1` | Use temperature-dependent recombination rates. |
| `Flag_Compute21cmBrightTemp` | `0` | Compute coeval 21-cm brightness temperature. |
| `Flag_ComputePS` | `0` | Calculate the 21-cm power spectrum; requires brightness output. |
| `Flag_IncludeSpinTemp` | `0` | Calculate thermal and spin temperatures; 0 assumes saturated spin temperature. |
| `Flag_InstantaneousSFR` | `1` | Use instantaneous rather than smoothed SFR for X-rays. |
| `Flag_IncludePecVelsFor21cm` | `0` | 0: off; 1: capped gradient; 2: uncapped gradient; 3: uncapped gradient + RSD. Requires spin-temperature calculations. |
| `Flag_ConstructLightcone` | `0` | Construct a brightness lightcone from the coupled snapshot history. |
| `Flag_IncludeLymanWerner` | `0` | Minihalo Lyman–Werner radiation feedback. |
| `Flag_IncludeMetalEvo` | `0` | Minihalo external IGM metal-enrichment calculation. |
| `Flag_IncludeStreamVel` | `0` | Minihalo baryon–dark-matter streaming-velocity treatment. |

</details>

<details>
<summary>Minihalo and external metal-enrichment controls</summary>

| Parameter | Default | Meaning / units |
|---|---|---|
| `MetalGridDim` | `128` | Cells per side of the external-metal grid. |
| `AlphaCluster` | `-1.4` | Inactive nonlinear-clustering fit coefficient. |
| `BetaCluster` | `0.8` | Inactive nonlinear-clustering fit coefficient. |
| `GammaCluster` | `2.8` | Inactive nonlinear-clustering fit coefficient. |
| `NormCluster` | `200.0` | Inactive nonlinear-clustering fit normalization. |
| `ZCrit` | `0.0001` | Pop. III/II transition metallicity in units of a metal mass fraction of 0.01. |

</details>

<details>
<summary>X-ray luminosity and AGN spectral controls</summary>

| Parameter | Default | Meaning / units |
|---|---|---|
| `LXrayGal(III)` | `3.16e40` | Galaxy soft-band luminosity per SFR, (erg s⁻¹)/(M☉ yr⁻¹). |
| `SpecIndexXrayGal` / `SpecIndexXrayIII` | `1.` | Pop. II / optional Pop. III X-ray spectral index. |
| `NuXrayThreshold` | `500.` | Lower escaping X-ray photon energy, eV. |
| `NuXraySoftCut` | `2000.` | Soft/hard X-ray break or upper soft-band energy, eV. |
| `NuXrayMax` | `10000.` | Upper X-ray integration energy, eV. |
| `Flag_IncludeAGNXray` | `0` | 0: no AGN X-rays; 1: soft and hard; 2: hard only; 3: soft only. |
| `SpecIndexXrayAGNSoft` | `2.2` | AGN soft-band X-ray spectral index. |
| `SpecIndexXrayAGNHard` | `1.7` | AGN hard-band X-ray spectral index. |
| `SpecIndexUVAGNSoft` | `0.61` | AGN nonionizing UV spectral index, wavelengths above 912 Å. |
| `SpecIndexUVAGNHard` | `1.70` | AGN ionizing/EUV spectral index, wavelengths at or below 912 Å. |
| `AGNLWEfficiency` | `1.0` | AGN Lyman–Werner normalization. |
| `Flag_BHARExponentialCut` | `0` | Randomized exponential accretion timing; 0 uses duty-cycle weighting. |
| `Flag_OutputXrayLF` | `0` | Write X-ray luminosity functions and emissivity histories; requires `Flag_IncludeSpinTemp=1`. |
| `XrayLF_MinLogL` | `38.0` | Lower log10 X-ray luminosity bound, luminosity in erg s⁻¹. |
| `XrayLF_MaxLogL` | `46.0` | Upper log10 X-ray luminosity bound, luminosity in erg s⁻¹. |
| `XrayLF_BinsPerDex` | `5` | X-ray histogram bins per luminosity dex. |

</details>

<details>
<summary>Star formation and Pop. III controls</summary>

| Parameter | Default | Meaning / units |
|---|---|---|
| `SfDiskVelOpt` | `1` | 1: Vmax; 2: Vvir in the star-formation disk prescription. |
| `SfPrescription` | `1` | 1: critical surface density; 2: pressure-based molecular gas; 3: GALFORM cold-gas law. |
| `SfEfficiency(_III)` | `0.08` (III: `0.008`) | Star-formation efficiency normalization. |
| `SfEfficiencyScaling(_III)` | `0.0` | Redshift scaling of the star-formation efficiency. |
| `SfCriticalSDNorm(_III)` | `0.2` | Critical surface-density normalization in internal units. |
| `PopIII_IMF` | `1` | 1: Sal500_001; 2: Sal500_050; 3: logA500_001; 4: logE500_001. |
| `PopIIIAgePrescription` | `2` | 1: Schaerer strong-mass-loss lifetimes; 2: no-mass-loss lifetimes. |

</details>

<details>
<summary>Stellar feedback</summary>

| Parameter | Default | Meaning / units |
|---|---|---|
| `Flag_IRA` | `0` | Use instantaneous recycling instead of the delayed-feedback tables. |
| `Flag_ReheatToFOFGroupTemp` | `0` | Reheat using the FoF virial temperature instead of the subhalo temperature. |
| `SfRecycleFraction(_III)` | `0.25` | Instantaneous recycled mass fraction used by IRA. |
| `Yield(_III)` | `0.03` | IRA metal yield per unit formed stellar mass. |
| `SnModel` | `1` | 1: Guo-style velocity factors; 2: broken power-law velocity factors. |
| `SnEjectionRedshiftDep(_III)` | `0.0` | SN energy redshift exponent. |
| `SnEjectionEff(_III)` | `0.5` | SN energy efficiency normalization. |
| `SnEjectionScaling(_III)` | `2.0` | SN energy high-velocity exponent. |
| `SnEjectionScaling2(_III)` | `2.0` | SN energy low-velocity exponent. |
| `SnEjectionNorm(_III)` | `70.0` | SN energy velocity pivot, km s⁻¹. |
| `SnReheatRedshiftDep(_III)` | `0.0` | Reheating redshift exponent. |
| `SnReheatEff(_III)` | `10.0` | Reheating efficiency normalization. |
| `SnReheatLimit(_III)` | `10.0` | Reheating maximum mass-loading factor. |
| `SnReheatScaling(_III)` | `0.0` | Reheating high-velocity exponent. |
| `SnReheatScaling2(_III)` | `0.0` | Reheating low-velocity exponent. |
| `SnReheatNorm(_III)` | `70.0` | Reheating velocity pivot, km s⁻¹. |
| `SnMetalRetentionFraction` | `0.0` | Fraction of reheated metals retained in cold gas; range 0–1. |

</details>

<details>
<summary>Cooling and reincorporation</summary>

| Parameter | Default | Meaning / units |
|---|---|---|
| `MaxCoolingMassFactor` | `1.0` | Maximum cooling/free-fall mass factor. |
| `ReincorporationModel` | `1` | 1: halo-dynamical-time prescription; 2: mass-dependent prescription. |
| `ReincorporationEff` | `0.0` | Model-dependent reincorporation efficiency; model 2 uses Myr normalization. |

</details>

<details>
<summary>Mergers and infall</summary>

| Parameter | Default | Meaning / units |
|---|---|---|
| `Flag_FixVmaxOnInfall` | `0` | Preserve the infall Vmax for satellites. |
| `Flag_FixDiskRadiusOnInfall` | `0` | Preserve the infall disk radius for satellites. |
| `ThreshMajorMerger` | `0.3` | Legacy major-merger threshold; unused. |
| `MergerTimeFactor` | `0.5` | Multiplicative dynamical-friction timescale factor. |
| `MinMergerStellarMass` | `1e-9` | Minimum stellar mass for merger bursts/friction, internal units. |
| `MinMergerRatioForBurst` | `0.1` | Minimum merger ratio for a starburst. |
| `MergerBurstFactor` | `0.57` | Merger-burst mass-fraction normalization. |
| `MergerBurstScaling` | `0.7` | Merger-burst mass-ratio exponent. |

</details>

<details>
<summary>Black-hole growth and feedback</summary>

| Parameter | Default | Meaning / units |
|---|---|---|
| `Flag_BHFeedback` | `1` | Enable the BH feedback prescription. |
| `RadioModeEff` | `0.3` | Radio-mode accretion efficiency parameter. |
| `QuasarModeEff` | `0.0005` | Quasar-mode feedback coupling parameter. |
| `BlackHoleGrowthRate` | `0.05` | Merger-driven BH cold-gas accretion normalization. |
| `EddingtonRatio` | `1.0` | Accretion Eddington-ratio parameter. |
| `BlackHoleSeed` | `1e-7` | BH seed mass in internal mass units. |
| `BlackHoleMassLimitReion` | `-1` | BH mass cutoff for ionizing sources; negative disables the cutoff. |
| `quasar_mode_scaling` | `0.0` | Redshift scaling of quasar-mode BH growth. |
| `quasar_open_angle` | `80.0` | Quasar opening angle, degrees. |

</details>

<details>
<summary>Reionization, escape fraction and filtering</summary>

| Parameter | Default | Meaning / units |
|---|---|---|
| `Flag_ReionizationModifier` | `1` | 0: no infall suppression; 1: Sobacchi-style; 2: Gnedin-style; 3: precomputed critical-mass history. |
| `ReionSobacchi_Zre` | `9.3` | Global reionization redshift in the Sobacchi prescription. |
| `ReionSobacchi_DeltaZre` | `1.0` | Global reionization-history width. |
| `ReionSobacchi_DeltaZsc` | `2.0` | Global reionization redshift-transition parameter. |
| `ReionSobacchi_T0` | `5.0e4` | Global reionization temperature normalization, K. |
| `ReionGnedin_z0` | `8` | Initial reionization redshift in the Gnedin prescription. |
| `ReionGnedin_zr` | `7` | Final reionization redshift in the Gnedin prescription. |
| `Flag_PatchyReion` | `1` | Enable the spatial radiation/ionization calculation. |
| `Flag_OutputGrids` | `1` | Save radiation grids; the thermal path also writes its source inputs. |
| `Flag_OutputGridsPostReion` | `1` | Continue grid calculations and outputs after reionization. |
| `Flag_FescCGMSuppression` | `0` | 0: off; 1: instantaneous Gamma12; 2: accumulated Gamma12; 3: clumping-factor modulation. |
| `ReionUVBFlag` | `1` | 0: decoupled; 1: store UVB at ionization; 2: update UVB after ionization. |
| `ReionGridDim` | `128` | Cells per side of the radiation grid. |
| `ReionDeltaRFactor` | `1.1` | Ratio between successive excursion-set filtering radii. |
| `ReionFilterType` | `0` | Radiation filter selector: 0 real-space top-hat; 1 sharp-k; 2 Gaussian. |
| `ReionPowerSpecDeltaK` | `0.1` | Legacy bin control; unused. Power-spectrum bins grow by a fixed factor of 1.35. |
| `ReionRtoMFilterType` | `0` | Radius-to-mass volume convention: 0 top-hat, 1 Gaussian. |
| `Y_He` | `0.24` | Primordial helium mass fraction. |
| `ReionRBubbleMin` | `0.4068` | Minimum excursion-set filter radius, cMpc/h. |
| `ReionRBubbleMax` | `20.34` | Fixed maximum bubble radius without recombinations, cMpc/h. |
| `ReionGammaHaloBias` | `2.0` | UVB halo-bias factor. |
| `ReionAlphaUV` | `2.0` | Stellar UV spectral index used in the UVB normalization. |
| `ReionAlphaUVBH` | `2.0` | BH UV spectral index for UVB conversion. |
| `EscapeFracDependency` | `1` | 0: constant; 1: redshift; 2: stellar mass; 3: SFR; 4: cold-gas surface density; 5: halo mass; 6: specific SFR. |
| `EscapeFracNorm(III)` | `0.06` | Stellar escape-fraction normalization. |
| `EscapeFracRedshiftOffset` | `6.0` | Escape-fraction redshift pivot. |
| `EscapeFracRedshiftScaling` | `0.5` | Escape-fraction redshift exponent. |
| `EscapeFracPropScaling` | `0.5` | Galaxy-property exponent for property-dependent escape fractions. |
| `EscapeFracBHNorm` | `1` | BH escape-fraction normalization. |
| `EscapeFracBHScaling` | `0` | BH escape-fraction redshift exponent. |
| `FescCGMSuppressionNorm` | `0.00008` | CGM suppression normalization. |
| `FescCGMSuppressionScaling` | `0.2` | CGM column-density exponent. |
| `FescCGMGamma12Scaling` | `6.0` | CGM UVB/clumping modulation exponent. |
| `ReionSMParam_m0` | `0.18984` | Sobacchi critical-mass normalization, internal mass units. |
| `ReionSMParam_a` | `0.17` | Sobacchi UVB-intensity exponent. |
| `ReionSMParam_b` | `-2.1` | Sobacchi redshift exponent. |
| `ReionSMParam_c` | `2.0` | Sobacchi ionization-history exponent. |
| `ReionSMParam_d` | `2.5` | Sobacchi ionization-history exponent. |
| `ReionTcool` | `1.0e4` | Atomic-cooling virial-temperature threshold, K. |
| `ReionNionPhotPerBary` | `4000` | Ionizing photons per stellar baryon. |
| `ReionSfrTimescale` | `0.5` | SFR-averaging interval in Hubble-time units when `Flag_InstantaneousSFR=0`. |
| `TsHeatingFilterType` | `1` | Thermal-history filter selector: 0 real-space top-hat; 1 sharp-k; 2 Gaussian. |
| `TsNumFilterSteps` | `40` | Number of thermal-history filtering steps. |
| `TsVelocityComponent` | `3` | 1: x velocity; 2: y; 3: z. |
| `EndRedshiftLightcone` | `5.0` | Low-redshift endpoint of the requested lightcone. |
| `ReionRBubbleMaxRecomb` | `33.9` | Fixed maximum radius with recombinations, cMpc/h; superseded by the evolving-radius flag. |
| `ReionMaxHeatingRedshift` | `30.` | Maximum heating redshift; no higher than the first snapshot redshift. |

</details>

<details>
<summary>Stochasticity</summary>

Enable `USE_STOCHASTICITY` at compilation to modify stellar radiation sources.

| Parameter | Default | Meaning |
|---|---|---|
| `EscapeFracScatterDex` | `0.0` | Scatter in log10 stellar escape fraction, dex. |
| `Flag_RemoveSFRScatter` | `0` | Remove SFR scatter from radiation sources at fixed halo mass. |
| `XrayScatterDex` | `0.0` | Scatter in log10 galaxy X-ray luminosity at fixed SFR, dex. |
| `Flag_SourceRecalibration` | `0` | Match modified source budgets to the untreated galaxy population in the same run. |

Escape-fraction scatter and median-SFR sources are mutually exclusive.
X-ray scatter can accompany either and requires `Flag_IncludeSpinTemp=1`.
Recalibration requires an active treatment and restores the corresponding
untreated source budgets. `RandomSeed` sets the random generator.

See the [escape-fraction and scatter formulas](formulas/galaxies.md#stellar-radiation)
and [source normalization](formulas/igm.md#source-normalization).

</details>

<details>
<summary>Photometry and dust</summary>

| Parameter | Default | Meaning / units |
|---|---|---|
| `BirthCloudLifetime` | `10e6` | Birth-cloud stellar-age boundary, yr. |
| `DustMetallicityScale` | `1.2` | Metallicity exponent in dust optical depth. |
| `DustTauUVISM` | `13.5` | ISM UV optical-depth normalization. |
| `DustNISM` | `-1.6` | ISM wavelength attenuation exponent. |
| `DustTauUVBC` | `381.3` | Birth-cloud UV optical-depth normalization. |
| `DustNBC` | `-1.6` | Birth-cloud wavelength attenuation exponent. |
| `DustAZ` | `-0.35` | Redshift coefficient in the dust attenuation factor. |
| `TargetSnaps` | `-1` | Magnitude snapshots; match `MAGS_N_SNAPS` and saved outputs. |
| `RestBands` | `1550,1650` | Rest-frame top-hat wavelength boundaries, Å; each pair defines one band. |
| `BetaBands` | Empty | UV-slope band boundaries, Å. |
| `InstantSfIII` | `0` | 0: continuous Pop. III SF templates; 1: instantaneous burst templates. |
| `DeltaT` | `1.0` | Pop. III instantaneous-burst time within the snapshot, Myr. |

</details>

<details>
<summary>Distribution functions</summary>

| Parameter | Default | Meaning / units |
|---|---|---|
| `Flag_OutputHMF` | `1` | Write the halo mass function. |
| `HMF_MinMass` | `8.0` | Lower log10 halo-mass bound, mass in M☉/h. |
| `HMF_MaxMass` | `15.0` | Upper log10 halo-mass bound, mass in M☉/h. |
| `HMF_BinsPerDex` | `2` | Halo-mass bins per dex. |
| `Flag_OutputSMF` | `1` | Write the stellar mass function. |
| `SMF_MinMass` | `6.0` | Lower log10 stellar-mass bound, mass in M☉. |
| `SMF_MaxMass` | `13.0` | Upper log10 stellar-mass bound, mass in M☉. |
| `SMF_BinsPerDex` | `2` | Stellar-mass bins per dex. |
| `Flag_OutputUVLF` | `1` | Write the intrinsic UV luminosity function; requires CALC_MAGS. |
| `UVLF_MinMag` | `-28.0` | Bright absolute-magnitude limit. |
| `UVLF_MaxMag` | `-8.0` | Faint absolute-magnitude limit. |
| `UVLF_BinsPerMag` | `2` | UV magnitude bins per magnitude. |
| `Flag_OutputDustyLF` | `1` | Write the dust-attenuated UV luminosity function; requires CALC_MAGS. |
| `Flag_OutputQuasarLF` | `1` | Write the quasar UV luminosity function with AGN duty-cycle weighting. |
| `Flag_OutputOIIILF` | `1` | Write the [O III] luminosity function. |
| `OIIILF_MinLogL` | `38.0` | Lower log10 [O III] luminosity bound, luminosity in erg s⁻¹. |
| `OIIILF_MaxLogL` | `45.0` | Upper log10 [O III] luminosity bound, luminosity in erg s⁻¹. |
| `OIIILF_BinsPerDex` | `2` | [O III] luminosity bins per dex. |

</details>
