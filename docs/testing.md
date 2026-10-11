# Testing

Meraxes has two kinds of tests:

- **Unit tests** check individual physics modules (cooling, black-hole feedback, emission lines, parameter parsing).
- **Regression tests** run the whole model on a small simulation and compare the results with stored references, so any change to the outputs is caught.

GitHub Actions runs both on every push and pull request (see [Continuous integration](#continuous-integration)).

## Running the tests

Configure a Debug build with the tests enabled. `MERAXES_TESTDATA_DIR` is only needed for the regression tests:

```sh
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON -DMERAXES_TESTDATA_DIR=/path/to/testdata
make
ctest -LE regression    # unit tests
ctest -L regression     # regression tests
```

Debug builds compile with `-Werror=implicit` and run under AddressSanitizer and UBSan. The tests need:

| Tests | Requirement |
|---|---|
| Unit tests | [Criterion](https://github.com/Snaipe/Criterion) |
| Regression runs | MPI; each run uses 2 ranks |
| Regression comparisons | Python 3 with `h5py` and `numpy` |
| Regression plots (optional) | `matplotlib` |

Without Criterion, only the two standalone drivers are built. Without Python, the regression runs still happen but aren't compared.

| Unit test | Checks |
|---|---|
| `test_parse_snaplist` | Parsing of snapshot lists such as `OutputSnapshots`. |
| `test_cooling` | Cooling tables, cooling rates and the streaming-velocity cooling mass. |
| `test_blackhole_feedback` | Black-hole emissivity, accretion time and luminosities. |
| `test_emission_lines` | [O III] coefficients and luminosities. |
| `main_cooling`, `main_blackhole_feedback` | Standalone drivers that print the same quantities for given inputs. |

## Regression tests

### Test data

The regression tests use **Genesis L10_N512**, published on Zenodo ([doi:10.5281/zenodo.23281409](https://doi.org/10.5281/zenodo.23281409)): a dark-matter-only box of 10 *h*⁻¹ cMpc with 512³ particles and 76 snapshots from *z* ≈ 30 to 10. Download `L10_N512.zip` and unzip it:

```sh
unzip L10_N512.zip -d /path/to/testdata
```

This creates `/path/to/testdata/L10_N512/` with `a_list.txt`, the 32³ density grids in `grids/` and the trees in `trees/`. Point `MERAXES_TESTDATA_DIR` at `/path/to/testdata`. The simulation parameters are in `input/params/simulations/Genesis_L10_N512.par`.

### Setups

Each setup is an `input.par` template in `src/tests/regression/setups/`. All of them use:

| Setting | Value |
|---|---|
| Simulation | `Genesis_L10_N512.par` (`ReionGridDim` 32) |
| MPI ranks | 2 |
| `RandomSeed` | 1809 |
| `OutputSnapshots` | 48, 63, 75 (*z* = 15, 12, 10) |
| Everything else | `input/params/defaults.par` |

With a fixed seed and rank count, repeated runs give bitwise-identical output. Each setup changes only the parameters below, and runs only in the build configuration its reference comes from:

| Setup | Build | Changes from the defaults | Covers |
|---|---|---|---|
| `default` | default | none | Galaxy formation, black-hole feedback, patchy reionization, HMF, SMF, QuasarLF, OIIILF |
| `igm` | default | `Flag_IncludeRecombinations 1`, `ReionUVBFlag 2`, `Flag_FescCGMSuppression 2` | Temperature-dependent recombinations, residual neutral fraction, clumping, CGM suppression of the escape fraction |
| `xray` | default | `Flag_IncludeSpinTemp 1`, `Flag_Compute21cmBrightTemp 1`, `Flag_ComputePS 1`, `Flag_IncludeAGNXray 1`, `Flag_OutputXrayLF 1`, `Flag_BHARExponentialCut 1` | Spin temperature, 21-cm brightness temperature and power spectrum, HMXB and AGN X-ray heating, obscuration, XrayLF |
| `mini_halos` | `-DUSE_MINI_HALOS=ON` | as `xray`, plus `Flag_IncludeLymanWerner 1`, `Flag_IncludeMetalEvo 1`, `MetalGridDim 32` | Pop. III stars, Lyman–Werner feedback (including AGN), IGM metal enrichment |
| `calc_mags` | `-DCALC_MAGS=ON -DMAGS_N_SNAPS=3 -DMAGS_N_BANDS=2` | `TargetSnaps 48, 63, 75`, `RestBands 1550, 1650, 4958, 5058` | Photometry, UVLF, DustyLF, OIIIDustyLF |
| `stochasticity` | `-DUSE_STOCHASTICITY=ON` | spin temperature and power spectrum, `EscapeFracScatterDex 0.3`, `XrayScatterDex 0.3`, `Flag_SourceRecalibration 1` | Escape-fraction and X-ray scatter, source recalibration |

In Debug builds the runs take from about 10 s (`default`, `igm`, `calc_mags`) to 2–3 minutes (`xray`, `mini_halos`).

### What is compared

After each run, `src/tests/regression/compare.py` summarises the output in a small HDF5 file and compares it with `src/tests/regression/reference/<setup>.h5`. For each output snapshot, the summary holds:

- every distribution function the run writes (bin centre, φ and its uncertainty);
- galaxy counts by type, and box totals of the main mass, metal and emission fields;
- the 5, 16, 50, 84 and 95% quantiles of log₁₀ of key galaxy properties (`Mvir`, `StellarMass`, `ColdGas`, `Sfr`, `BlackHoleMass`, `LOIII`, `ionization_param`, `Fesc`, `tau_cgm`);
- volume- and mass-weighted means of the reionization and 21-cm grids, the median log₁₀ `t_resp` and the 21-cm power spectrum;

plus the X-ray emissivity histories. The CMake option `REGRESSION_COMPARE` chooses how strictly they're compared:

| Mode | Tolerances | Used by |
|---|---|---|
| `strict` | Counts exact; everything else to 10⁻⁶ relative. | CI, whose references come from the same build environment. |
| `loose` (default) | Distribution functions within 3σ of their uncertainties; counts within 3√N or 1%; log₁₀ medians and 16/50/84% quantiles within 0.05 dex; other quantities within 1%. | Other machines, where compiler and library differences shift results slightly. |

In both modes a quantity that appears or disappears also fails the test. The references pass loose comparisons with runs built on Gadi with GCC 11.

### Plots

Configure with `-DREGRESSION_PLOTS=ON` to have each comparison plot the run over its reference in `plots/` next to its `summary.h5`, or plot any summary by hand:

```sh
python3 src/tests/regression/compare.py plot src/tests/regression/reference/<setup>.h5 plots/ --new <summary.h5>
```

Each setup gets three figures. The reference is drawn as lines (with ±σ or quantile bands) and the run as points; panels holding a quantity that differs beyond tolerance get a red title. The reference figures themselves are in `src/tests/regression/reference/plots/` and are shown in
[Reference figures](#reference-figures).

### Reference figures

The current references, plotted by CI from the runs they come from. Each figure shows a reference
on its own; plotting a comparison adds the new run as points. The figures are updated together with
the references.

#### `default`

Galaxy formation, black-hole feedback and patchy reionization with the default parameters.

```{figure} ../src/tests/regression/reference/plots/default_distributions.png
:alt: default reference: distribution functions
:width: 100%

`default`: Distribution functions at *z* = 15, 12 and 10.
```

```{figure} ../src/tests/regression/reference/plots/default_galaxies.png
:alt: default reference: galaxy properties
:width: 100%

`default`: Quantiles of galaxy properties, galaxy counts and box totals against redshift.
```

```{figure} ../src/tests/regression/reference/plots/default_igm.png
:alt: default reference: IGM quantities
:width: 100%

`default`: Averages of the reionization grids against redshift.
```

#### `igm`

Temperature-dependent recombinations, residual neutral fraction, clumping and CGM suppression of the escape fraction.

```{figure} ../src/tests/regression/reference/plots/igm_distributions.png
:alt: igm reference: distribution functions
:width: 100%

`igm`: Distribution functions at *z* = 15, 12 and 10.
```

```{figure} ../src/tests/regression/reference/plots/igm_galaxies.png
:alt: igm reference: galaxy properties
:width: 100%

`igm`: Quantiles of galaxy properties, galaxy counts and box totals against redshift.
```

```{figure} ../src/tests/regression/reference/plots/igm_igm.png
:alt: igm reference: IGM quantities
:width: 100%

`igm`: Averages of the reionization grids against redshift.
```

#### `xray`

Spin temperature, 21-cm brightness temperature and power spectrum, with HMXB and AGN X-ray heating.

```{figure} ../src/tests/regression/reference/plots/xray_distributions.png
:alt: xray reference: distribution functions
:width: 100%

`xray`: Distribution functions at *z* = 15, 12 and 10.
```

```{figure} ../src/tests/regression/reference/plots/xray_galaxies.png
:alt: xray reference: galaxy properties
:width: 100%

`xray`: Quantiles of galaxy properties, galaxy counts and box totals against redshift.
```

```{figure} ../src/tests/regression/reference/plots/xray_igm.png
:alt: xray reference: IGM quantities
:width: 100%

`xray`: Averages of the reionization and 21-cm grids against redshift, the 21-cm power spectrum and the X-ray emissivity histories.
```

#### `mini_halos`

Pop. III stars, Lyman–Werner feedback and IGM metal enrichment, on top of `xray`.

```{figure} ../src/tests/regression/reference/plots/mini_halos_distributions.png
:alt: mini_halos reference: distribution functions
:width: 100%

`mini_halos`: Distribution functions at *z* = 15, 12 and 10.
```

```{figure} ../src/tests/regression/reference/plots/mini_halos_galaxies.png
:alt: mini_halos reference: galaxy properties
:width: 100%

`mini_halos`: Quantiles of galaxy properties, galaxy counts and box totals against redshift.
```

```{figure} ../src/tests/regression/reference/plots/mini_halos_igm.png
:alt: mini_halos reference: IGM quantities
:width: 100%

`mini_halos`: Averages of the reionization and 21-cm grids against redshift, the 21-cm power spectrum and the X-ray emissivity histories.
```

#### `calc_mags`

Photometry and the UV and dust-attenuated luminosity functions.

```{figure} ../src/tests/regression/reference/plots/calc_mags_distributions.png
:alt: calc_mags reference: distribution functions
:width: 100%

`calc_mags`: Distribution functions at *z* = 15, 12 and 10.
```

```{figure} ../src/tests/regression/reference/plots/calc_mags_galaxies.png
:alt: calc_mags reference: galaxy properties
:width: 100%

`calc_mags`: Quantiles of galaxy properties, galaxy counts and box totals against redshift.
```

```{figure} ../src/tests/regression/reference/plots/calc_mags_igm.png
:alt: calc_mags reference: IGM quantities
:width: 100%

`calc_mags`: Averages of the reionization grids against redshift.
```

#### `stochasticity`

Escape-fraction and X-ray scatter with source recalibration, with spin temperature on.

```{figure} ../src/tests/regression/reference/plots/stochasticity_distributions.png
:alt: stochasticity reference: distribution functions
:width: 100%

`stochasticity`: Distribution functions at *z* = 15, 12 and 10.
```

```{figure} ../src/tests/regression/reference/plots/stochasticity_galaxies.png
:alt: stochasticity reference: galaxy properties
:width: 100%

`stochasticity`: Quantiles of galaxy properties, galaxy counts and box totals against redshift.
```

```{figure} ../src/tests/regression/reference/plots/stochasticity_igm.png
:alt: stochasticity reference: IGM quantities
:width: 100%

`stochasticity`: Averages of the reionization and 21-cm grids against redshift, the 21-cm power spectrum and the X-ray emissivity histories.
```

### Updating the references

When a change alters the results on purpose, the regression comparisons fail. Each CI regression job uploads an artifact, `regression-summaries-<config>`, holding every setup's `summary.h5` and its plots over the reference. To accept the new results:

1. Download the artifacts of the failing run.
2. Copy each `<setup>/summary.h5` to `src/tests/regression/reference/<setup>.h5`, and each `<setup>/plots/*.png` to `src/tests/regression/reference/plots/`.
3. Commit them in the same pull request, and paste the output of `python3 compare.py diff <old reference> <new summary>` into its description so reviewers can see what moved.

The references must come from CI, not from a local build, because CI compares them strictly.

## Continuous integration

`.github/workflows/tests.yml` runs on every push and pull request, and can be started by hand from the Actions tab. Each job builds Meraxes in Debug, runs the unit tests and drivers, and, where a setup exists, the regression tests on the L10_N512 data, downloaded from Zenodo and cached. To save time, only the configurations a change can affect are built:

| Configuration | Regression setups | Runs when |
|---|---|---|
| default | `default`, `igm`, `xray` | always |
| mini-halos | `mini_halos` | a changed C source or header mentions `USE_MINI_HALOS` |
| calc-mags | `calc_mags` | a changed C source or header mentions `CALC_MAGS`, `USE_JWST` or `USE_HST` |
| stochasticity | `stochasticity` | a changed C source or header mentions `USE_STOCHASTICITY` |
| stochasticity-mini-halos | none (unit tests only) | as stochasticity |

Every configuration runs when the build system, the workflow, `src/tests/regression/` or `input/` changes, on manual runs, and when the changes can't be determined.

A final job, `All tests passed`, succeeds only if every configuration did. Branch protection requires this one check, since the names of the configuration jobs change from run to run.

## Formatting

C, C++ and CUDA sources under `src/` follow the repository's `.clang-format`, checked with clang-format 14 by `.github/workflows/clang-format-check.yml`. The check fails with the diff to apply; it doesn't fix the code for you. Format your changes before pushing:

```bash
pipx install clang-format==14.0.6  # once; other versions format differently
tools/format.sh                    # format every source under src/
tools/format.sh src/core/foo.c     # or just the files you changed
tools/format.sh --check            # change nothing, show what would change
```

To check every commit automatically, enable the pre-commit hook once per clone:

```bash
git config core.hooksPath .githooks
```

It refuses a commit whose staged sources aren't formatted. Run `tools/format.sh` on the files it lists, stage them again, and commit. If clang-format 14 isn't installed, the hook skips the check and lets the commit through.
