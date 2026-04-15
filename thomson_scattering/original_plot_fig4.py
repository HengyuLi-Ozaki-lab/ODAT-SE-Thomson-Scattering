import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patheffects as pe
import sys
sys.path.insert(0, ".")
from thomson_model import make_objective_function
from run_lhd_test import generate_realistic_data

Te_true, ne_true = 1495.0, 1.577
_, obs, sig = generate_realistic_data(Te_true, ne_true, seed=44)
obj_fn = make_objective_function(obs, sig)

# ---- chi2 landscape ----
Te_grid = np.linspace(50, 4500, 300)
ne_grid = np.linspace(0.1, 4.5, 300)
Te_mesh, ne_mesh = np.meshgrid(Te_grid, ne_grid)
chi2_map = np.zeros_like(Te_mesh)
for i in range(len(ne_grid)):
    for j in range(len(Te_grid)):
        chi2_map[i, j] = obj_fn(np.array([Te_grid[j], ne_grid[i]]))
chi2_min = chi2_map.min()

# ---- Load samples ----
d_high = np.loadtxt("pamc/output/0/result_T0.txt")
d_low  = np.loadtxt("pamc/output/0/result_T13.txt")
d_high = d_high[len(d_high)//2:]
d_low  = d_low[len(d_low)//2:]

# ---- Poster-quality figure ----
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.facecolor": "white",
})

fig = plt.figure(figsize=(15, 6.5))
gs = fig.add_gridspec(1, 2, wspace=0.30, left=0.07, right=0.93, top=0.82, bottom=0.12)
axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]

levels = sorted(set([chi2_min + d for d in [1, 2.3, 5, 10, 20, 50, 100, 300, 1000]]))

# Log-spaced fill levels (guaranteed increasing)
fill_levels = np.logspace(np.log10(max(chi2_min, 1)), 3.5, 30)

panels = [
    (axes[0], d_high, r"Prior exploration  ($T = 100$)"),
    (axes[1], d_low,  r"Posterior concentration  ($T \approx 1$)"),
]

sc = None
for ax, data, subtitle in panels:
    Te_s = data[:, 4]
    ne_s = data[:, 5]
    fx_s = data[:, 3]

    # Smooth filled contour background
    cf = ax.contourf(Te_mesh, ne_mesh, chi2_map,
                     levels=fill_levels, cmap="YlGnBu_r", alpha=0.35, zorder=0,
                     extend="max")

    # Scatter samples
    sc = ax.scatter(Te_s, ne_s, c=fx_s, s=4, alpha=0.6,
                    cmap="inferno_r", norm=LogNorm(vmin=max(chi2_min, 10), vmax=800),
                    rasterized=True, zorder=2, edgecolors="none")

    # Contour lines
    cs = ax.contour(Te_mesh, ne_mesh, chi2_map, levels=levels,
                    colors="#444444", linewidths=0.6, zorder=5, linestyles="-")
    clabels = ax.clabel(cs, fmt="%.0f", fontsize=8, colors="#333333")
    for cl in clabels:
        cl.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])

    # True value star
    ax.plot(Te_true, ne_true, "*", color="#00FF00", markersize=20, zorder=10,
            markeredgecolor="black", markeredgewidth=1.0)

    # Annotation
    ax.annotate(
        f"True\n({Te_true:.0f} eV, {ne_true:.2f})",
        xy=(Te_true, ne_true), xytext=(Te_true + 700, ne_true + 0.85),
        fontsize=9, color="#222222", ha="center",
        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#aaaaaa", alpha=0.9),
        zorder=11
    )

    ax.set_xlabel(r"$T_e$ (eV)", fontsize=14)
    ax.set_ylabel(r"$n_e$ ($\times\,10^{19}$ m$^{-3}$)", fontsize=14)
    ax.set_title(subtitle, fontsize=13, pad=10)
    ax.set_xlim(50, 4500)
    ax.set_ylim(0.1, 4.5)
    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.tick_params(direction="in", which="both")

# Colorbar
cax = fig.add_axes([0.94, 0.12, 0.015, 0.70])
cbar = fig.colorbar(sc, cax=cax)
cbar.set_label(r"$\chi^2$ value", fontsize=12)
cbar.ax.tick_params(labelsize=10)

# Title
fig.text(0.50, 0.955,
         "Bayesian Inverse Inference of LHD Thomson Scattering Diagnostics",
         fontsize=16, fontweight="bold", ha="center", va="center",
         fontfamily="serif")
fig.text(0.50, 0.905,
         r"Population Annealing Monte Carlo (PAMC) via ODAT-SE  —  LHD Shot #175916, $R = 4119$ mm",
         fontsize=11, ha="center", va="center", color="#555555",
         fontfamily="serif")

fig.savefig("analysis/figures/pamc_search_process.png", dpi=250, bbox_inches="tight",
            facecolor="white", edgecolor="none")
# Also save PDF for poster
fig.savefig("analysis/figures/pamc_search_process.pdf", bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print("Saved: pamc_search_process.png + .pdf")
PYEOF