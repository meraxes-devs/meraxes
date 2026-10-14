# Regression references

One summary file per setup (`<setup>.h5`), written by `../compare.py` from a
run of `../setups/<setup>.par.in` on the L10_N512 test set. The CI regression
job compares against these in strict mode, so they must come from CI: each CI
run uploads its summaries as an artifact. When a change alters results on
purpose, replace the affected files with that artifact's summaries in the same
PR, and include the output of

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

The following references are unchanged copies of `summary.h5` from the successful
[CI run 38486985381](https://github.com/ChangqIngovo/meraxes-devs_qyxnew/actions/runs/38486985381),
at commit `f6492ebab351f5f965a7a74782a51d706d6f5221`:

| Reference | CI artifact | File within artifact |
| --- | --- | --- |
| `xray_metallicity.h5` | `regression-summaries-default` | `xray_metallicity/summary.h5` |
| `xray_metallicity_scatter.h5` | `regression-summaries-stochasticity` | `xray_metallicity_scatter/summary.h5` |

Both runs use L10_N512, two MPI ranks, random seed 1809, and output snapshots
48, 63, and 75. Their summary metadata records the commit above and `git_dirty=0`.
With these reference files present, CMake registers `compare.py check` for both
setups instead of the missing-reference `compare.py write` fallback.

The existing `xray.h5` and `stochasticity.h5` are unchanged from upstream master
`e0a73833d180264711c40ab70abaec2320e3c2bd`. Their comparisons both passed in strict
mode in the same CI run, checking the original `Flag_UseXrayMetallicity=0` behavior.
