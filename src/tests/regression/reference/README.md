# Regression references

One summary file per setup (`<setup>.h5`), written by `../compare.py` from a
run of `../setups/<setup>.par.in` on the L10_N512 test set. The CI regression
job compares against these in strict mode, so they must come from CI: each CI
run uploads its summaries as an artifact. When a change alters results on
purpose, replace the affected files with that artifact's summaries in the same
PR, and include the output of

    python3 compare.py diff reference/<setup>.h5 <new summary.h5>

in the PR description so reviewers can see what moved.

CI doesn't make plots. To see what changed in a failed CI run, download its
summaries and plot them over the references (needs matplotlib):

    python3 compare.py plot reference/<setup>.h5 plots/ --new <downloaded summary.h5>

Panels holding a quantity that differs beyond tolerance get a red title. When
developing locally, configure with -DREGRESSION_PLOTS=ON to have every
regression comparison write these plots next to its summary.h5.

`plots/` holds figures of each reference on its own, for browsing on GitHub.
Regenerate them whenever a reference changes:

    python3 compare.py plot reference/<setup>.h5 reference/plots
