"""Plots for compare.py: a regression run overlaid on its reference summary.

Three figures per setup, written as PNGs:
  <name>_distributions.png  each distribution function (HMF, SMF, luminosity
                            functions) at every output snapshot
  <name>_galaxies.png       quantiles of key galaxy properties, galaxy counts
                            and box totals against redshift
  <name>_igm.png            grid means against redshift, the 21-cm power
                            spectrum and the X-ray emissivity histories
The reference is drawn as lines (with +-sigma or quantile bands), the run as
markers. Panels with a quantity that failed the comparison get a red title.
With no run, only the reference is drawn (for reference/plots/).
"""
import os

import numpy as np

# x-axis label and phi units of each distribution function
LF_AXES = {
    "HMF": (r"$\log_{10}(M_{\rm vir}/h^{-1}{\rm M_\odot})$", "dex"),
    "SMF": (r"$\log_{10}(M_*/{\rm M_\odot})$", "dex"),
    "UVLF": (r"$M_{\rm UV}$", "mag"),
    "DustyLF": (r"$M_{\rm UV}$ (dust attenuated)", "mag"),
    "QuasarLF": (r"$M_{\rm UV}$ (quasars)", "mag"),
    "OIIILF": (r"$\log_{10}(L_{\rm [OIII]}/{\rm erg\,s^{-1}})$", "dex"),
    "OIIIDustyLF": (r"$\log_{10}(L_{\rm [OIII]}/{\rm erg\,s^{-1}})$ (dust attenuated)", "dex"),
    "XrayLF": (r"$\log_{10}(L_{\rm 2-10\,keV}/{\rm erg\,s^{-1}})$ (intrinsic)", "dex"),
    "XrayLF_obs": (r"$\log_{10}(L_{\rm 2-10\,keV}/{\rm erg\,s^{-1}})$ (obscured)", "dex"),
}
GRID_LABELS = {
    "xH": r"$\langle x_{\rm HI} \rangle$",
    "x_e_box": r"$\langle x_e \rangle$",
    "Tk_box": r"$\langle T_K \rangle$ [K]",
    "TS_box": r"$\langle T_S \rangle$ [K]",
    "delta_T": r"$\langle \delta T_b \rangle$ [mK]",
    "J_21_at_ionization": r"$\langle J_{21} \rangle$ at ionization",
    "r_bubble": r"$\langle r_{\rm bubble} \rangle$ [$h^{-1}$cMpc]",
    "z_at_ionization": r"$\langle z_{\rm ion} \rangle$",
    "Gamma12": r"$\langle \Gamma_{12} \rangle$",
    "residual_xH": r"$\langle x_{\rm HI,res} \rangle \times 10^4$",
    "clumping_factor": r"$\langle C \rangle$",
    "N_rec": r"$\langle N_{\rm rec} \rangle$",
    "t_resp": r"$t_{\rm resp}$ [Myr] (cells with finite $t_{\rm resp}$)",
    "temp_kinetic_all_gas": r"$\langle T_{\rm gas} \rangle$ [K]",
    "Mvir_crit": r"$\langle M_{\rm vir,crit} \rangle$",
    "weighted_sfr": r"$\langle$ $f_{\rm esc}$-weighted SFR $\rangle$",
    "effective_bhar": r"$\langle$ effective BHAR $\rangle$",
}
TYPE_NAMES = ["centrals", "satellites", "orphans", "type 3"]


def _snaps(s):
    return sorted(k.split("/")[0] for k in s if k.endswith("/Redshift"))


def _redshifts(s, snaps):
    return np.array([float(s[f"{snap}/Redshift"][0]) for snap in snaps])


def _colours(n):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("viridis")
    return [cmap(x) for x in np.linspace(0.0, 0.85, max(n, 1))]


def _title(ax, text, differs):
    ax.set_title(text + ("  (differs)" if differs else ""), fontsize=9, color="tab:red" if differs else "black")


def _grid(n, ncols=3, size=(4.2, 3.2)):
    import matplotlib.pyplot as plt

    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size[0] * ncols, size[1] * nrows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, list(axes.flat[:n])


def _style_legend(fig, snaps, z, colours, have_new, ref_label, new_label):
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=c, lw=2, label=f"z = {zz:.1f}") for c, zz in zip(colours, z)]
    handles.append(Line2D([], [], color="k", lw=1.5, label=ref_label))
    if have_new:
        handles.append(Line2D([], [], color="k", ls="none", marker="o", ms=4, label=new_label))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=len(handles), fontsize=8,
               frameon=False)


def _save(fig, outdir, name, kind, title):
    fig.suptitle(title, y=0.995, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(outdir, f"{name}_{kind}.png")
    fig.savefig(path, dpi=110)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def _plot_distributions(ref, new, snaps, z, colours, differing, labels):
    names = sorted({k.split("/")[2] for k in ref if "/lf/" in k} | ({k.split("/")[2] for k in new if "/lf/" in k} if new else set()))
    if not names:
        return None
    fig, axes = _grid(len(names))
    for ax, name in zip(axes, names):
        xlabel, unit = LF_AXES.get(name, (name, "bin"))
        differs = False
        for snap, colour in zip(snaps, colours):
            key = f"{snap}/lf/{name}"
            differs |= key in differing
            if key in ref:
                x, phi, sig = ref[key].T
                ok = phi > 0
                ax.plot(x[ok], phi[ok], color=colour, lw=1.5)
                ax.fill_between(x[ok], np.clip(phi - sig, phi * 0.1, None)[ok], (phi + sig)[ok], color=colour, alpha=0.2, lw=0)
            if new and key in new:
                x, phi, sig = new[key].T
                ok = phi > 0
                ax.errorbar(x[ok], phi[ok], yerr=sig[ok], color=colour, ls="none", marker="o", ms=3, elinewidth=0.8)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(rf"$\phi$ [cMpc$^{{-3}}$ {unit}$^{{-1}}$]", fontsize=8)
        _title(ax, name, differs)
    _style_legend(fig, snaps, z, colours, bool(new), *labels)
    return fig


def _series(s, snaps, path_fmt, index=None):
    """Values of one quantity across snapshots (NaN where missing)."""
    out = []
    for snap in snaps:
        v = s.get(path_fmt.format(snap=snap))
        out.append(np.nan if v is None else float(np.asarray(v).reshape(-1)[0 if index is None else index]))
    return np.array(out)


def _plot_galaxies(ref, new, snaps, z, differing, labels):
    qfields = sorted({k.split("/")[3] for k in ref if "/log10_quantiles/" in k})
    tfields = sorted({k.split("/")[3] for k in ref if "/galaxies/total/" in k})
    n_panels = len(qfields) + 2
    fig, axes = _grid(n_panels)
    for ax, field in zip(axes, qfields):
        q_ref = np.array([ref.get(f"{s}/galaxies/log10_quantiles/{field}", np.full(5, np.nan)) for s in snaps])
        differs = any(f"{s}/galaxies/log10_quantiles/{field}" in differing for s in snaps)
        q_new = np.array([new.get(f"{s}/galaxies/log10_quantiles/{field}", np.full(5, np.nan)) for s in snaps]) if new else None
        if np.all(np.isnan(q_ref)) and (q_new is None or np.all(np.isnan(q_new))):
            ax.text(0.5, 0.5, "no galaxies with positive values", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            _title(ax, field, differs)
            continue
        ax.plot(z, q_ref[:, 2], color="k", lw=1.5)
        ax.fill_between(z, q_ref[:, 1], q_ref[:, 3], color="k", alpha=0.2, lw=0)
        ax.fill_between(z, q_ref[:, 0], q_ref[:, 4], color="k", alpha=0.08, lw=0)
        if new:
            ax.errorbar(z, q_new[:, 2], yerr=[q_new[:, 2] - q_new[:, 1], q_new[:, 3] - q_new[:, 2]],
                        color="tab:orange", ls="none", marker="o", ms=4)
            ax.plot(z, q_new[:, 0], color="tab:orange", ls="none", marker="v", ms=3)
            ax.plot(z, q_new[:, 4], color="tab:orange", ls="none", marker="^", ms=3)
        ax.set_xlabel("redshift", fontsize=8)
        ax.set_ylabel(rf"$\log_{{10}}$ {field}", fontsize=8)
        _title(ax, f"{field}: median, 16-84%, 5-95%", differs)

    ax = axes[len(qfields)]
    for t, colour in zip(range(4), _colours(4)):
        r = _series(ref, snaps, "{snap}/galaxies/count_by_type", t)
        if np.nansum(r) == 0 and not (new and np.nansum(_series(new, snaps, "{snap}/galaxies/count_by_type", t))):
            continue
        ax.plot(z, r, color=colour, lw=1.5, label=TYPE_NAMES[t])
        if new:
            ax.plot(z, _series(new, snaps, "{snap}/galaxies/count_by_type", t), color=colour, ls="none", marker="o", ms=4)
    ax.set_yscale("log")
    ax.set_xlabel("redshift", fontsize=8)
    ax.set_ylabel("number of galaxies", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    _title(ax, "galaxy counts", any(f"{s}/galaxies/count_by_type" in differing for s in snaps))

    ax = axes[len(qfields) + 1]
    colours = _colours(len(tfields))
    differs = False
    for field, colour in zip(tfields, colours):
        r = _series(ref, snaps, "{snap}/galaxies/total/" + field)
        differs |= any(f"{s}/galaxies/total/{field}" in differing for s in snaps)
        if new:
            n = _series(new, snaps, "{snap}/galaxies/total/" + field)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.where(r != 0, n / r - 1.0, np.where(n == 0, 0.0, np.nan))
            ax.plot(z, rel, color=colour, marker="o", ms=3, lw=1, label=field)
        elif np.any(r > 0) and r[np.argmin(z)] > 0:
            ax.plot(z, r / r[np.argmin(z)], color=colour, marker="o", ms=3, lw=1, label=field)
    if new:
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.set_ylabel("this run / reference - 1", fontsize=8)
        _title(ax, "box totals (relative difference)", differs)
    else:
        ax.set_yscale("log")
        ax.set_ylabel(f"total / total at z = {z.min():.1f}", fontsize=8)
        _title(ax, "box totals", differs)
    ax.set_xlabel("redshift", fontsize=8)
    ax.legend(fontsize=5.5, frameon=False, ncol=2)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [Line2D([], [], color="k", lw=1.5, label=f"{labels[0]} (median)"),
               Patch(color="k", alpha=0.2, label="16-84%"), Patch(color="k", alpha=0.08, label="5-95%")]
    if new:
        handles.append(Line2D([], [], color="tab:orange", ls="none", marker="o", ms=4, label=labels[1]))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=len(handles), fontsize=8,
               frameon=False)
    return fig


def _plot_igm(ref, new, snaps, z, colours, differing, labels):
    grid_names = [g for g in GRID_LABELS
                  if any(f"{s}/grids/volume_mean/{g}" in ref or f"{s}/grids/log10_median/{g}" in ref for s in snaps)]
    ps_snaps = [s for s in snaps if f"{s}/grids/PS_data" in ref]
    xray = sorted(k for k in ref if k.startswith("global/XrayEmissivity_"))
    n_panels = len(grid_names) + (1 if ps_snaps else 0) + (1 if xray else 0)
    if n_panels == 0:
        return None
    fig, axes = _grid(n_panels)
    i = 0
    for g in grid_names:
        ax = axes[i]
        i += 1
        differs = False
        mkey = "{snap}/grids/log10_median/" + g
        if any(mkey.format(snap=s) in ref for s in snaps):
            # heavy-tailed grids are summarised by the median of log10
            ax.plot(z, _series(ref, snaps, mkey), color="k", lw=1.5, label="median")
            if new:
                ax.plot(z, _series(new, snaps, mkey), color="tab:orange", ls="none", marker="o", ms=4)
            differs = any(mkey.format(snap=s) in differing for s in snaps)
            ax.set_xlabel("redshift", fontsize=8)
            ax.set_ylabel(r"$\log_{10}$ " + GRID_LABELS[g], fontsize=8)
            ax.legend(fontsize=7, frameon=False)
            _title(ax, f"{g}: median", differs)
            continue
        for kind, ls in (("volume_mean", "-"), ("mass_mean", "--")):
            path = "{snap}/grids/" + kind + "/" + g
            r = _series(ref, snaps, path)
            if np.all(np.isnan(r)):
                continue
            differs |= any(path.format(snap=s) in differing for s in snaps)
            ax.plot(z, r, color="k", ls=ls, lw=1.5, label=kind.replace("_", "-"))
            if new:
                ax.plot(z, _series(new, snaps, path), color="tab:orange", ls="none", marker="o" if ls == "-" else "s", ms=4)
        ax.set_xlabel("redshift", fontsize=8)
        ax.set_ylabel(GRID_LABELS[g], fontsize=8)
        ax.legend(fontsize=7, frameon=False)
        _title(ax, g, differs)
    if ps_snaps:
        ax = axes[i]
        i += 1
        differs = False
        for snap, colour in zip(snaps, colours):
            key = f"{snap}/grids/PS_data"
            if key not in ref:
                continue
            differs |= key in differing or f"{snap}/grids/PS_error" in differing
            p = ref[key]
            e = ref.get(f"{snap}/grids/PS_error", np.zeros_like(p))
            kb = ref.get(f"{snap}/grids/k_bins")
            k = kb if kb is not None and len(kb) == len(p) else np.arange(len(p))
            ok = p > 0
            ax.plot(k[ok], p[ok], color=colour, lw=1.5)
            ax.fill_between(k[ok], np.clip(p - e, p * 0.1, None)[ok], (p + e)[ok], color=colour, alpha=0.2, lw=0)
            if new and key in new:
                pn, en = new[key], new.get(f"{snap}/grids/PS_error", np.zeros_like(new[key]))
                okn = pn > 0
                ax.errorbar(k[okn], pn[okn], yerr=en[okn], color=colour, ls="none", marker="o", ms=3)
        ax.set_yscale("log")
        if any(f"{s}/grids/k_bins" in ref for s in ps_snaps):
            ax.set_xscale("log")
            ax.set_xlabel(r"$k$ [cMpc$^{-1}$]", fontsize=8)
        else:
            ax.set_xlabel("k bin", fontsize=8)
        ax.set_ylabel(r"$\Delta^2_{21}$ [mK$^2$]", fontsize=8)
        _title(ax, "21-cm power spectrum", differs)
    if xray:
        ax = axes[i]
        for key, colour in zip(xray, _colours(len(xray))):
            r = ref[key]
            label = key.split("_", 1)[1]
            ax.plot(np.arange(len(r)), r, color=colour, lw=1.5, label=label)
            if new and key in new:
                ax.plot(np.arange(len(new[key])), new[key], color=colour, ls="none", marker="o", ms=2)
        ax.set_yscale("log")
        top = max(float(np.nanmax(ref[key])) for key in xray)
        if top > 0:
            ax.set_ylim(top * 1e-4, top * 3)
        ax.set_xlabel("snapshot", fontsize=8)
        ax.set_ylabel(r"emissivity [erg s$^{-1}$ cm$^{-3}$]", fontsize=8)
        ax.legend(fontsize=7, frameon=False)
        _title(ax, "X-ray emissivity histories", any(k in differing for k in xray))
    _style_legend(fig, snaps, z, colours, bool(new), *labels)
    return fig


def make_plots(ref, new, outdir, name, differing=(), ref_label="reference", new_label="this run"):
    """Write the comparison figures for one setup; return their paths."""
    import matplotlib

    matplotlib.use("Agg")
    os.makedirs(outdir, exist_ok=True)
    differing = set(differing)
    snaps = _snaps(ref)
    z = _redshifts(ref, snaps)
    colours = _colours(len(snaps))
    labels = (ref_label, new_label)
    title = f"{name}: {new_label} vs {ref_label}" if new else f"{name}: {ref_label}"
    written = []
    for kind, fig in (
        ("distributions", _plot_distributions(ref, new, snaps, z, colours, differing, labels)),
        ("galaxies", _plot_galaxies(ref, new, snaps, z, differing, labels)),
        ("igm", _plot_igm(ref, new, snaps, z, colours, differing, labels)),
    ):
        if fig is not None:
            written.append(_save(fig, outdir, name, kind, title))
    return written
