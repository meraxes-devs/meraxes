#!/usr/bin/env python3
"""Summarise a Meraxes regression run and compare it with a reference summary.

The summary is a small HDF5 file holding, for each output snapshot, the
distribution functions (HMF, SMF, luminosity functions), galaxy counts, box
totals and quantiles of key galaxy properties, global averages of the
reionization / 21-cm grids and the 21-cm power spectrum, plus the X-ray
emissivity histories. The same summarize() builds both the reference and the
summary of the run being tested, so the two are always comparable.

Usage:
  compare.py write <output_dir> <summary.h5> [--plot DIR]
      Summarise a run (e.g. to create or update a reference).
  compare.py check <output_dir> <reference.h5> [--mode strict|loose] [--save FILE] [--plot DIR]
      Summarise a run and compare it with the reference; exit 1 on any
      difference beyond tolerance. --save also writes the run's summary.
  compare.py diff <old.h5> <new.h5> [--mode strict|loose] [--plot DIR]
      Compare two summary files (e.g. to review a reference update).
  compare.py plot <reference.h5> <outdir> [--new RUN] [--mode strict|loose]
      Plot a reference, optionally overlaid with RUN (an output directory or a
      summary file); quantities that differ beyond tolerance are flagged.

--plot DIR writes the same figures (see plots.py) for that command: the run
over its reference for check/diff, the run alone for write. Plotting needs
matplotlib; without it the plots are skipped with a note.

Modes:
  strict  counts exact; everything else to 1e-6 relative. For CI, whose
          references come from the same build environment.
  loose   distribution functions within 3 sigma of their stored errors;
          counts within 3 sqrt(N) or 1%; log10 quantiles and medians within
          0.05 dex;
          totals, grid means and spectra within 1%. For other platforms,
          where compiler and library differences shift results.
"""
import argparse
import os
import sys

import h5py
import numpy as np

SUMMARY_VERSION = 1

# Galaxy fields summed over the box (mass and metal bookkeeping, emission)
TOTAL_FIELDS = [
    "StellarMass", "GrossStellarMass", "MetalsStellarMass", "ColdGas", "MetalsColdGas", "HotGas",
    "MetalsHotGas", "EjectedGas", "MetalsEjectedGas", "BlackHoleMass", "BlackHoleAccretedHotMass",
    "BlackHoleAccretedColdMass", "Sfr", "SfrIII", "FescWeightedSfr", "LOIII", "BHemissivity",
    "QuasarLX", "BHXrayEmissivity", "Pop3StellarMass", "Pop2StellarMass",
]
# Galaxy fields whose distribution (of log10 of positive values) is summarised
QUANTILE_FIELDS = [
    "Mvir", "StellarMass", "ColdGas", "Sfr", "BlackHoleMass", "LOIII", "ionization_param", "Fesc", "tau_cgm",
]
QUANTILES = [5, 16, 50, 84, 95]
# Grids whose box averages are summarised (each only if the run writes it)
GRID_FIELDS = [
    "xH", "x_e_box", "Tk_box", "TS_box", "delta_T", "J_21_at_ionization", "r_bubble", "z_at_ionization",
    "Gamma12", "residual_xH", "clumping_factor", "N_rec", "t_resp", "temp_kinetic_all_gas", "Mvir_crit",
    "weighted_sfr", "effective_bhar",
]
# ... of which these also get a mass-weighted mean
MASS_WEIGHTED_GRID_FIELDS = ["xH", "x_e_box", "Tk_box", "residual_xH", "temp_kinetic_all_gas"]
# Placeholder values meaning "not set" (t_resp is 1e30 where nothing ionizes the
# cell): excluded from the averages, and the fraction of such cells recorded
GRID_UNSET_THRESHOLD = {"t_resp": 1e29}
# Heavy-tailed grids summarised by the median of log10 instead of mean and spread.
# The finite t_resp values span decades and early on only ~100 cells have one, so
# a few cells changing between finite and unset (e.g. with a different libm) swing
# the mean by tens of percent and the tail quantiles by up to 2.5 dex; the median
# moves by < 0.03 dex.
GRID_LOG_MEDIAN_FIELDS = ["t_resp"]
# Loose mode compares log10 values (quantiles, medians) to this many dex
# (relative tolerances misbehave for log values near zero)
LOOSE_DEX = 0.05


def _snapshot_groups(f):
    return sorted(k for k in f.keys() if k.startswith("Snap") and k[4:].isdigit())


def _galaxies(snap):
    """Concatenate the galaxy catalogues of all cores for one snapshot."""
    cores = sorted(k for k in snap.keys() if k.startswith("Core"))
    tables = [snap[c]["Galaxies"][()] for c in cores if "Galaxies" in snap[c]]
    return np.concatenate(tables) if tables else None


def summarize(output_dir):
    """Return {path: array} summarising the run in output_dir."""
    s = {}
    with h5py.File(f"{output_dir}/meraxes.hdf5", "r") as f:
        for name in ("NHTrans", "NHfrac", "XrayEmissivity_hard", "XrayEmissivity_soft", "XrayEmissivity_HMXB"):
            if name in f:
                s[f"global/{name}"] = f[name][()]

        for snap_name in _snapshot_groups(f):
            snap = f[snap_name]
            base = snap_name
            s[f"{base}/Redshift"] = np.asarray(snap.attrs["Redshift"], dtype=float).reshape(-1)[:1]
            s[f"{base}/NGalaxies"] = np.asarray(snap.attrs["NGalaxies"], dtype=np.int64).reshape(-1)[:1]

            # Distribution functions: (n_bins, 3) datasets of bin centre, phi, sigma
            for name, obj in snap.items():
                if isinstance(obj, h5py.Dataset) and obj.ndim == 2 and obj.shape[1] == 3:
                    s[f"{base}/lf/{name}"] = obj[()]

            gals = _galaxies(snap)
            if gals is not None and len(gals) > 0:
                types = gals["Type"]
                s[f"{base}/galaxies/count_by_type"] = np.array([np.sum(types == t) for t in (0, 1, 2, 3)], dtype=np.int64)
                if "GhostFlag" in gals.dtype.names:
                    s[f"{base}/galaxies/count_ghosts"] = np.array([np.sum(gals["GhostFlag"] != 0)], dtype=np.int64)
                for field in TOTAL_FIELDS:
                    if field in gals.dtype.names:
                        s[f"{base}/galaxies/total/{field}"] = np.array([np.sum(gals[field], dtype=np.float64)])
                for field in QUANTILE_FIELDS:
                    if field in gals.dtype.names:
                        v = np.asarray(gals[field], dtype=np.float64)
                        v = v[np.isfinite(v) & (v > 0)]
                        s[f"{base}/galaxies/n_positive/{field}"] = np.array([v.size], dtype=np.int64)
                        q = np.percentile(np.log10(v), QUANTILES) if v.size else np.full(len(QUANTILES), np.nan)
                        s[f"{base}/galaxies/log10_quantiles/{field}"] = q

            if "Grids" in snap:
                grids = snap["Grids"]
                density = 1.0 + grids["deltax"][()].astype(np.float64) if "deltax" in grids else None
                for name in GRID_FIELDS:
                    if name in grids:
                        # Averages over finite cells, in case a grid marks cells with inf/NaN
                        g = grids[name][()].astype(np.float64)
                        ok = np.isfinite(g)
                        if name in GRID_UNSET_THRESHOLD:
                            unset = g >= GRID_UNSET_THRESHOLD[name]
                            s[f"{base}/grids/fraction_unset/{name}"] = np.array([np.mean(unset)])
                            ok &= ~unset
                        if name in GRID_LOG_MEDIAN_FIELDS:
                            v = g[ok & (g > 0)]
                            s[f"{base}/grids/log10_median/{name}"] = np.array([np.median(np.log10(v)) if v.size else np.nan])
                            continue
                        s[f"{base}/grids/volume_mean/{name}"] = np.array([np.mean(g[ok]) if ok.any() else np.nan])
                        s[f"{base}/grids/std/{name}"] = np.array([np.std(g[ok]) if ok.any() else np.nan])
                        if density is not None and name in MASS_WEIGHTED_GRID_FIELDS and ok.any():
                            w = density[ok]
                            s[f"{base}/grids/mass_mean/{name}"] = np.array([np.sum(g[ok] * w) / np.sum(w)])
                # 21-cm power spectrum and its k bins
                for name, obj in grids.items():
                    if (name.startswith("PS") or name == "k_bins") and isinstance(obj, h5py.Dataset):
                        s[f"{base}/grids/{name}"] = obj[()].astype(np.float64)
    return s


def run_metadata(output_dir):
    meta = {"summary_version": SUMMARY_VERSION}
    with h5py.File(f"{output_dir}/meraxes.hdf5", "r") as f:
        meta["ncores"] = int(np.asarray(f.attrs.get("NCores", [0])).reshape(-1)[0])
        if "gitdiff" in f:
            meta["gitref"] = f["gitdiff"].attrs.get("gitref", b"").decode()
            meta["git_dirty"] = int(len(f["gitdiff"][()]) > 0)
    return meta


def write_summary(summary, meta, path):
    with h5py.File(path, "w") as f:
        for key, value in meta.items():
            f.attrs[key] = value
        for key, value in summary.items():
            f.create_dataset(key, data=value)


def read_summary(path):
    s = {}
    with h5py.File(path, "r") as f:
        f.visititems(lambda name, obj: s.__setitem__(name, obj[()]) if isinstance(obj, h5py.Dataset) else None)
        meta = dict(f.attrs)
    return s, meta


def _rel(a, b):
    scale = np.maximum(np.abs(a), np.abs(b))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(scale > 0, np.abs(a - b) / scale, 0.0)
    return np.nan_to_num(r, nan=0.0)


def compare(ref, new, mode):
    """Return a list of (path, message, size) for every disagreement."""
    problems = []
    for key in sorted(set(ref) - set(new)):
        problems.append((key, "missing from the new run", np.inf))
    for key in sorted(set(new) - set(ref)):
        problems.append((key, "not in the reference", np.inf))

    for key in sorted(set(ref) & set(new)):
        a, b = np.asarray(ref[key]), np.asarray(new[key])
        if a.shape != b.shape:
            problems.append((key, f"shape {a.shape} -> {b.shape}", np.inf))
            continue
        nan_a, nan_b = np.isnan(a) if a.dtype.kind == "f" else False, np.isnan(b) if b.dtype.kind == "f" else False
        if np.any(nan_a != nan_b):
            problems.append((key, "NaN pattern changed", np.inf))
            continue

        if a.dtype.kind in "iu" or b.dtype.kind in "iu":
            d = np.abs(a.astype(np.int64) - b.astype(np.int64))
            if mode == "strict":
                allowed = np.zeros_like(d)
            else:  # Poisson-like scatter: 3 sqrt(N) or 1%, whichever is larger
                n = np.maximum(a, 1).astype(np.float64)
                allowed = np.maximum(3.0 * np.sqrt(n), 0.01 * n)
            if np.any(d > allowed):
                problems.append((key, f"counts differ (max {d.max()}): {a.tolist()} -> {b.tolist()}", float(d.max())))
            continue

        finite = np.isfinite(a) & np.isfinite(b)
        if mode == "loose" and "/lf/" in key:
            # columns: bin centre, phi, sigma
            sigma = np.sqrt(a[:, 2] ** 2 + b[:, 2] ** 2)
            dphi = np.abs(a[:, 1] - b[:, 1])
            tol = 3.0 * sigma + 1e-12 * np.maximum(np.abs(a[:, 1]), np.abs(b[:, 1]))
            bad = dphi > tol
            if np.any(bad):
                i = int(np.argmax(np.where(bad, dphi / np.where(sigma > 0, sigma, np.inf), 0)))
                problems.append((key, f"{bad.sum()} bins beyond 3 sigma (e.g. x={a[i, 0]:g}: "
                                      f"{a[i, 1]:.6g} -> {b[i, 1]:.6g} +- {sigma[i]:.3g})", float(bad.sum())))
            continue

        if mode == "loose" and ("/log10_quantiles/" in key or "/log10_median/" in key):
            d = np.abs(np.where(finite, a - b, 0.0))
            if np.any(d > LOOSE_DEX) or np.any(np.isfinite(a) != np.isfinite(b)):
                i = int(np.argmax(d))
                problems.append((key, f"{a.reshape(-1)[i]:.6g} -> {b.reshape(-1)[i]:.6g} "
                                      f"({d.reshape(-1)[i]:.3g} dex)", float(d.max())))
            continue

        rtol = 1e-6 if mode == "strict" else 1e-2
        r = _rel(np.where(finite, a, 0.0), np.where(finite, b, 0.0))
        if np.any(r > rtol) or np.any(np.isfinite(a) != np.isfinite(b)):
            i = int(np.argmax(r)) if r.size else 0
            av, bv = a.reshape(-1)[i], b.reshape(-1)[i]
            problems.append((key, f"{av:.8g} -> {bv:.8g} (rel {r.reshape(-1)[i]:.2e})", float(r.max())))
    return problems


def _load(path):
    """A summary dict and metadata from a run's output directory or a summary file."""
    if os.path.isdir(path):
        return summarize(path), run_metadata(path)
    return read_summary(path)


def _setup_name(path):
    """Name for the figures: a summary file's stem, or the run directory's name."""
    path = os.path.abspath(path)
    if os.path.isdir(path):
        path = os.path.dirname(path) if os.path.basename(path) == "output" else path
        return os.path.basename(path)
    return os.path.splitext(os.path.basename(path))[0]


def _plot(ref, new, outdir, name, problems=(), ref_meta=None, new_meta=None):
    try:
        from plots import make_plots
    except ImportError as err:
        print(f"Skipping plots ({err})")
        return
    ref_label = f"reference {str((ref_meta or {}).get('gitref', ''))[:8]}".strip()
    new_label = f"this run {str((new_meta or {}).get('gitref', ''))[:8]}".strip()
    try:
        for path in make_plots(ref, new, outdir, name, {p[0] for p in problems}, ref_label, new_label):
            print(f"Wrote {path}")
    except Exception as err:  # plots are a convenience: never change the result
        print(f"Plotting failed, continuing without plots: {err!r}")


def report(problems, mode, label):
    if not problems:
        print(f"{label}: all quantities agree ({mode} mode)")
        return 0
    print(f"{label}: {len(problems)} quantities differ ({mode} mode):")
    width = max(len(p[0]) for p in problems)
    for path, message, _ in sorted(problems, key=lambda p: -p[2]):
        print(f"  {path:<{width}}  {message}")
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    w = sub.add_parser("write")
    w.add_argument("output_dir")
    w.add_argument("summary")
    w.add_argument("--plot", metavar="DIR", help="also plot the run")
    c = sub.add_parser("check")
    c.add_argument("output_dir")
    c.add_argument("reference")
    c.add_argument("--mode", choices=["strict", "loose"], default="strict")
    c.add_argument("--save", help="also write this run's summary here")
    c.add_argument("--plot", metavar="DIR", help="also plot the run over the reference")
    d = sub.add_parser("diff")
    d.add_argument("old")
    d.add_argument("new")
    d.add_argument("--mode", choices=["strict", "loose"], default="strict")
    d.add_argument("--plot", metavar="DIR", help="also plot new over old")
    p = sub.add_parser("plot")
    p.add_argument("reference")
    p.add_argument("outdir")
    p.add_argument("--new", help="run output directory or summary file to overlay")
    p.add_argument("--mode", choices=["strict", "loose"], default="strict")
    p.add_argument("--name", help="figure name prefix (default: the reference's file name)")
    args = parser.parse_args()

    if args.command == "write":
        summary, meta = summarize(args.output_dir), run_metadata(args.output_dir)
        write_summary(summary, meta, args.summary)
        if args.plot:
            _plot(summary, None, args.plot, _setup_name(args.output_dir), ref_meta=meta)
        return 0
    if args.command == "check":
        new, new_meta = summarize(args.output_dir), run_metadata(args.output_dir)
        if args.save:
            write_summary(new, new_meta, args.save)
        ref, ref_meta = read_summary(args.reference)
        problems = compare(ref, new, args.mode)
        if args.plot:
            _plot(ref, new, args.plot, _setup_name(args.reference), problems, ref_meta, new_meta)
        label = f"vs reference from {str(ref_meta.get('gitref', '?'))[:8]}"
        return report(problems, args.mode, label)
    if args.command == "diff":
        ref, ref_meta = read_summary(args.old)
        new, new_meta = read_summary(args.new)
        problems = compare(ref, new, args.mode)
        if args.plot:
            _plot(ref, new, args.plot, _setup_name(args.old), problems, ref_meta, new_meta)
        return report(problems, args.mode, f"{args.old} -> {args.new}")
    ref, ref_meta = read_summary(args.reference)
    new, new_meta = _load(args.new) if args.new else (None, None)
    problems = compare(ref, new, args.mode) if new is not None else []
    _plot(ref, new, args.outdir, args.name or _setup_name(args.reference), problems, ref_meta, new_meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
