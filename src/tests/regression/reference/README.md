# Regression references

One summary file per setup (`<setup>.h5`), written by `../compare.py` from a
run of `../setups/<setup>.par.in` on the L10_N512 test set. The CI regression
job compares against these in strict mode. Prefer summaries from CI, which
uploads them as artifacts. For local regeneration, use the same build settings
and MPI rank count, reproduce the unchanged controls in strict mode, and record
the source state and toolchain. When a change alters results on purpose, replace
the affected files in the same PR, and include the output of

    python3 compare.py diff reference/<setup>.h5 <new summary.h5>

in the PR description so reviewers can see what moved.

The same CI artifact has each run's plots over its reference (`<setup>/plots/`);
panels holding a quantity that differs beyond tolerance get a red title. You can
also make them yourself from a downloaded summary (needs matplotlib):

    python3 compare.py plot reference/<setup>.h5 plots/ --new <downloaded summary.h5>

When developing locally, configure with -DREGRESSION_PLOTS=ON to have every
regression comparison write these plots next to its summary.h5.

`plots/` holds figures of each reference on its own, for browsing on GitHub.
Regenerate them whenever a reference changes:

    python3 compare.py plot reference/<setup>.h5 reference/plots

## Metallicity-dependent X-ray references

The two references below were regenerated locally on 2026-10-19 with the
Fragos/Kaur metallicity prescription. They are byte-for-byte copies of the
corresponding complete MPI runs' `summary.h5` files.

| Reference | Build | Run summary | Quantities changed from the previous reference |
| --- | --- | --- | --- |
| `xray_metallicity.h5` | default | `xray_metallicity/summary.h5` | 101 |
| `xray_metallicity_scatter.h5` | `USE_STOCHASTICITY=ON` | `xray_metallicity_scatter/summary.h5` | 105 |

Both runs use L10_N512 (archive MD5 `afe3719ce9144869c9f0b8629c62e9d5`),
two MPI ranks, random seed 1809, and output snapshots 48, 63, and 75. The
parameters are `XrayMetallicityNorm=3.16e40`, `XrayMetallicityExponent=0.3`,
`XrayMetallicityTurnover=0.008`, and a 500--2000 eV calibration band.
The scatter setup uses `XrayScatterDex=0.3` and `Flag_SourceRecalibration=1`.

The source checkout is based on commit
`6565b5d475f8e3062642a513a9b735fe92d60233`, with the uncommitted Fragos/Kaur
replacement and build-metadata escaping fix. The summary metadata therefore
records that commit and `git_dirty=1` for these local runs.
The build environment was Ubuntu 24.04, GCC 13.3.0, CMake 3.28.3,
Open MPI 4.1.6, parallel HDF5 1.10.10, GSL 2.7.1 and FFTW 3.3.10.
Both builds used Debug, AddressSanitizer and UBSan, with leak detection disabled
as in CI. Local MPI communication used shared memory, separate temporary
directories for each build, and the PMIx hash data store.

The existing `xray.h5` and `stochasticity.h5` remain byte-for-byte unchanged
from upstream master `e0a73833d180264711c40ab70abaec2320e3c2bd`.
Their full MPI runs both passed strict comparisons in this local environment.
The two new summaries retain all 230 datasets, their shapes and finite-value
masks; their differences and plots were inspected before replacement.
No comparison tolerances or CMake comparison registrations were changed.

The previous metallicity references came from
[CI run 38486985381](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/actions/runs/38486985381),
at commit `f6492ebab351f5f965a7a74782a51d706d6f5221`.
The old-to-new numerical differences and local test details are recorded in
[local_validation_20261019.txt](local_validation_20261019.txt).

Reference-only plots were regenerated from the new HDF5 summaries:

| Setup | Distribution functions | Galaxy properties | IGM and 21-cm |
| --- | --- | --- | --- |
| `xray_metallicity` | [plot](plots/xray_metallicity_distributions.png) | [plot](plots/xray_metallicity_galaxies.png) | [plot](plots/xray_metallicity_igm.png) |
| `xray_metallicity_scatter` | [plot](plots/xray_metallicity_scatter_distributions.png) | [plot](plots/xray_metallicity_scatter_galaxies.png) | [plot](plots/xray_metallicity_scatter_igm.png) |
