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

### Review validation on 2026-10-23

The current local source is based on
`f81101401e69110801efb13fa6c2ad9dccd77b07` plus the uncommitted PR review
changes: shared `LXrayGal` normalization, explicit input-band normalization,
recorded birth metallicity and the equivalent heating-source unit conversion.
The `xray` and `stochasticity` flag-off setups have completed their native
two-rank MPI runs. Both pass strict comparison against the unchanged checked-in
references, whose metadata identifies source `72800527`.

The builds use Ubuntu 24.04.3, GCC 13.3.0, CMake 3.28.3, Open MPI 4.1.6,
parallel HDF5 1.10.10, GSL 2.7.1 and FFTW 3.3.10. The two non-mini-halo builds
used for the full MPI runs are Debug builds with `-O2 -g`. The two mini-halo
builds used for unit checks retain `-g` without added optimization. All four
use AddressSanitizer and UBSan, without fast-math; leak detection is disabled
for the test runs. The input is the official L10_N512 archive from
Zenodo record 23281409 (MD5 `afe3719ce9144869c9f0b8629c62e9d5`), with two MPI
ranks, seed 1809 and output snapshots 48, 63 and 75.

All four native model runs completed successfully. The two metallicity-enabled
summaries differ from the previous references because the abundance now tracks
the gas that formed the stars. Their numerical differences, source trends and
IGM comparison plots were inspected before replacing only these two HDF5
references with the corresponding run summaries. This was an accepted physics
update, not a strict-comparison pass against the old metallicity references.

| Reference | Build | Run summary | Quantities differing from the 2026-10-19 reference in strict mode |
| --- | --- | --- | --- |
| `xray_metallicity.h5` | default | `xray_metallicity/summary.h5` | 63: 55 grid, 7 galaxy-total and 1 global-emissivity quantities |
| `xray_metallicity_scatter.h5` | `USE_STOCHASTICITY=ON` | `xray_metallicity_scatter/summary.h5` | 77: 58 grid, 15 galaxy, 1 global-emissivity and 3 luminosity-function quantities |

Both summaries retain the same 230 dataset keys, shapes, and NaN and positive/
negative-infinity masks; galaxy counts are unchanged. At snapshot 75, the
volume-mean `Tk_box` changes from 403.9478 to 429.51494 K without scatter and
from 405.40097 to 431.09932 K with scatter. The unchanged flag-off controls
isolate these changes from the unit and parameter refactoring.

The current parameters are `LXrayGal=3.16e40`,
`XrayMetallicityExponent=0.3`, `XrayMetallicityTurnover=0.008`,
`NuXrayThreshold=500` eV and `NuXraySoftCut=2000` eV. The scatter setup uses
`XrayScatterDex=0.3` and `Flag_SourceRecalibration=1`. Both reference files
record `gitref=f81101401e69110801efb13fa6c2ad9dccd77b07`, `git_dirty=1`
and `ncores=2`. No comparison tolerances or CMake comparison registrations
were changed. The standard strict checks of the completed outputs against the
new references both pass. These checks reuse the generating runs; independent
repeat-run reproducibility was not tested.
The hashes, old-to-new differences and unit-test execution details are in
[local_validation_20261023.txt](local_validation_20261023.txt).

### Current reference plots

The six reference-only plots below were generated from the 2026-10-23 HDF5
summaries and added with this update:

| Setup | Distribution functions | Galaxy properties | IGM and 21-cm |
| --- | --- | --- | --- |
| `xray_metallicity` | [plot](plots/xray_metallicity_distributions.png) | [plot](plots/xray_metallicity_galaxies.png) | [plot](plots/xray_metallicity_igm.png) |
| `xray_metallicity_scatter` | [plot](plots/xray_metallicity_scatter_distributions.png) | [plot](plots/xray_metallicity_scatter_galaxies.png) | [plot](plots/xray_metallicity_scatter_igm.png) |

### Previous references from 2026-10-19

The replaced references were generated from
`6565b5d475f8e3062642a513a9b735fe92d60233` plus the uncommitted Fragos/Kaur
replacement and build-metadata escaping fix (`git_dirty=1`). They used the
same L10_N512 data, two ranks, seed and output snapshots, with the former
`XrayMetallicityNorm=3.16e40`, exponent 0.3, turnover 0.008 and a 500–2000 eV
calibration band. That update changed 101 and 105 quantities, respectively,
relative to [CI run 38486985381](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/actions/runs/38486985381)
at `f6492ebab351f5f965a7a74782a51d706d6f5221`. Both unchanged flag-off controls
passed strict comparison at that time. These details describe the superseded
references, not the current files.
