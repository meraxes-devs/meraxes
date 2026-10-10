# Regression references

One summary file per setup (`<setup>.h5`), written by `../compare.py` from a
run of `../setups/<setup>.par.in` on the L10_N512 test set. The CI regression
job compares against these in strict mode, so they must come from CI: each CI
run uploads its summaries as an artifact. When a change alters results on
purpose, replace the affected files with that artifact's summaries in the same
PR, and include the output of

    python3 compare.py diff reference/<setup>.h5 <new summary.h5>

in the PR description so reviewers can see what moved.
