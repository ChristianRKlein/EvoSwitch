"""Create all figures of the manuscript and the Supplementary Information.

Run after the analysis scripts 01-08 (reads the CSV files in results/).
Outputs figures/fig1.png ... fig4.png and figures/figS1.png ... figS3.png.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.ticker import FuncFormatter
from _common import FIGURES, read_csv

from evoswitch import TIMING_EXAMPLE, max_compatible_fraction_formula, response_day, trajectory
from evoswitch.model import flow, initial_state
from evoswitch.nadir import time_to_criterion_off_treatment, _rising_through
from evoswitch.model import integrate

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans", "font.size": 8.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e8e8e8", "grid.linewidth": 0.6,
    "axes.edgecolor": "#555", "savefig.dpi": 300})
BLUE, ORANGE, GRAY, RED, PURPLE, DARK = "#2a5d8f", "#d4683a", "#8a8a8a", "#b23a3a", "#7b5ea7", "#333"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#9e2a2b", "#e8a68f", "#f2f2f2", "#8fb3d9", "#1f4e85"])
FIGURES.mkdir(exist_ok=True)


def panel(ax, letter, x=-0.12):
    ax.text(x, 1.06, letter, transform=ax.transAxes, fontweight="bold", fontsize=10)


def col(rows, key):
    return np.array([float(r[key]) for r in rows])


def edges_log(c):
    lc = np.log10(c)
    return 10 ** np.r_[lc[0] - (lc[1] - lc[0]) / 2, (lc[1:] + lc[:-1]) / 2, lc[-1] + (lc[-1] - lc[-2]) / 2]


def edges_lin(c):
    return np.r_[c[0] - (c[1] - c[0]) / 2, (c[1:] + c[:-1]) / 2, c[-1] + (c[-1] - c[-2]) / 2]


# ---------------------------------------------------------------- Fig. 1
def fig1():
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))

    def cells(a, n_sens, n_res, title, subtitle):
        a.set_xlim(0, 10); a.set_ylim(0, 10); a.set_aspect("equal"); a.axis("off")
        a.add_patch(Circle((5, 5), 4.6, fc="#f6f3ee", ec="#bbb", lw=1))
        pts = []
        while len(pts) < n_sens + n_res:
            q = rng.uniform(1, 9, 2)
            if np.hypot(*(q - 5)) < 4.1 and all(np.hypot(*(q - o)) > 0.62 for o in pts):
                pts.append(q)
        for i, q in enumerate(pts):
            a.add_patch(Circle(q, 0.28, fc=BLUE if i < n_sens else ORANGE, ec="white", lw=0.6))
        a.set_title(title, fontsize=9.5, fontweight="bold")
        a.text(5, -0.6, subtitle, ha="center", va="top", fontsize=8)

    cells(ax[0], 46, 3, "Before therapy", "Mostly drug-sensitive cells (blue);\na few pre-existing resistant cells (orange)")
    cells(ax[1], 10, 8, "On targeted therapy", "Sensitive cells die: the tumor shrinks,\nbut resistant cells lose their competitors")
    cells(ax[2], 30, 4, "Treatment interruption", "Sensitive cells regrow and again\nkeep resistant cells in check")
    for x in (0.345, 0.67):
        fig.patches.append(FancyArrowPatch((x - 0.012, 0.55), (x + 0.012, 0.55), transform=fig.transFigure,
                                           arrowstyle="-|>", mutation_scale=16, color=DARK))
    fig.subplots_adjust(bottom=0.2, wspace=0.25)
    fig.savefig(FIGURES / "fig1.png", bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- Fig. 2
def fig2():
    tr = read_csv("timing_trajectory.csv")
    gs = read_csv("timing_gain_by_start_day.csv")
    t, s, r, b = col(tr, "day"), col(tr, "sensitive"), col(tr, "resistant"), col(tr, "total")
    starts, d = col(gs, "start_day"), col(gs, "delta_T_days")
    t30, tpr = response_day(TIMING_EXAMPLE), response_day(TIMING_EXAMPLE, 0.7 ** 3)
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.2))
    ax[0].plot(t, s, c=BLUE, lw=1.6, label="Sensitive cells")
    ax[0].plot(t, r, c=ORANGE, lw=1.6, label="Resistant cells")
    ax[0].plot(t, b, c="k", lw=1.2, ls="--", label="Total tumor burden")
    ax[0].axvline(t30, c=GRAY, ls=":", lw=1); ax[0].text(t30 + 8, 0.255, "30%", fontsize=7.5, color="#444")
    ax[0].axvline(tpr, c=ORANGE, ls=":", lw=1.2)
    ax[0].text(tpr + 10, 0.232, "66%% reduction\n(≈ RECIST partial\nresponse, day %d)" % np.ceil(tpr),
               fontsize=7.5, color="#444", va="top")
    ax[0].set(xlabel="Days on continuous therapy", ylabel="Tumor burden (relative)", xlim=(0, 750), ylim=(0, 0.28))
    ax[0].legend(frameon=False, fontsize=7.5, loc="center", bbox_to_anchor=(0.62, 0.62))
    ax[1].fill_between(starts, 0, d, color=BLUE, alpha=0.08, lw=0); ax[1].plot(starts, d, c=BLUE, lw=1.8)
    ax[1].plot(starts[0], d[0], "o", mfc="white", mec=GRAY, ms=5, zorder=5)
    ax[1].annotate("30%% criterion met\n(day %d, +%d days)" % (starts[0], round(d[0])), (starts[0], d[0]), (120, 72),
                   fontsize=7, color="#555", arrowprops=dict(arrowstyle="-", color="#999", lw=0.6))
    i = int(np.ceil(tpr) - starts[0])
    ax[1].plot(starts[i], d[i], "o", c=ORANGE, ms=6, zorder=5)
    ax[1].annotate("Start as soon as a RECIST-equivalent\npartial response is reached\n(day %d, +%d days); every later\n"
                   "start yields less" % (starts[i], round(d[i])), (starts[i], d[i]), (250, 42), fontsize=7.5,
                   arrowprops=dict(arrowstyle="-", color="#666", lw=0.7))
    ax[1].axhline(0, c="k", lw=0.6)
    ax[1].set(xlabel="Start day of a 4-week interruption", ylabel="Time to progression gained (days)")
    panel(ax[0], "a"); panel(ax[1], "b"); fig.tight_layout()
    fig.savefig(FIGURES / "fig2.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig. 3
def map_grid(name, key):
    rows = read_csv(name)
    f = np.array(sorted({float(r["f0"]) for r in rows})); a = np.array(sorted({float(r["alpha"]) for r in rows}))
    fi, ai = {v: i for i, v in enumerate(f)}, {v: i for i, v in enumerate(a)}
    Z = np.full((len(a), len(f)), np.nan)
    for r in rows:
        Z[ai[float(r["alpha"])], fi[float(r["f0"])]] = float(r[key])
    return 100 * f, a, Z


def fig3():
    traj, summary = read_csv("tumors_AB_trajectories.csv"), {r["tumor"]: r for r in read_csv("tumors_AB.csv")}
    fig = plt.figure(figsize=(10, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.55, wspace=0.22)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    titles = {"A": "Tumor A: 0.0001% resistant cells, weak competition",
              "B": "Tumor B: 3% resistant cells, strong competition"}
    for a, k in zip(axs, "AB"):
        cont = [r for r in traj if r["tumor"] == k and r["schedule"] == "continuous"]
        intr = [r for r in traj if r["tumor"] == k and r["schedule"] == "interruption"]
        a.axvspan(56, 84, color="#eee", lw=0)
        a.plot(col(cont, "day"), col(cont, "total_burden"), c=BLUE, lw=1.6, label="Continuous therapy")
        a.plot(col(intr, "day"), col(intr, "total_burden"), c=ORANGE, lw=1.6, ls="--",
               label="4-week interruption from day 56")
        tc, th = float(summary[k]["progression_continuous"]), float(summary[k]["progression_interruption"])
        a.axhline(0.7, c=GRAY, ls=":", lw=1); a.text(20, 0.72, "progression threshold", fontsize=7, color="#555")
        a.plot([tc], [0.7], "o", c=BLUE, ms=5); a.plot([th], [0.7], "s", c=ORANGE, ms=5)
        a.text(0.03, 0.80, ("Same 40%% response at day 56\nEffect of interruption: %+.1f days" % (th - tc))
               .replace("-", "−"), transform=a.transAxes, fontsize=7.5)
        a.set(title=titles[k], xlabel="Days since start of therapy", xlim=(0, 1300), ylim=(0, 1.02))
        a.title.set_fontsize(8.5)
    axs[0].set_ylabel("Tumor burden (relative)"); axs[1].legend(frameon=False, fontsize=7.5, loc="center right")
    axs[1].set_yticklabels([])
    panel(axs[0], "a"); panel(axs[1], "b", -0.06)
    ac = fig.add_subplot(gs[1, :])
    f, al, Z = map_grid("response_matched_map.csv", "delta_T_days")
    m = ac.pcolormesh(edges_log(f), edges_lin(al), Z, cmap=DIVERGING, norm=SymLogNorm(linthresh=1, vmin=-40, vmax=40),
                      shading="flat", edgecolors="none", rasterized=True)
    ac.contour(f, al, Z, levels=[0], colors="k", linewidths=1); ac.set_xscale("log"); ac.grid(False)
    ac.set(xlabel="Pre-existing resistant cells (% of tumor)", ylabel="Competition by sensitive cells (α)",
           title="1,368 virtual tumors, all with exactly 40% response at day 56: effect of a 4-week interruption")
    ac.title.set_fontsize(8.5)
    ac.text(2e-4, 1.35, "interruption better", fontsize=8, color="white")
    ac.text(1.5e-1, 0.06, "continuous better", fontsize=7.5, color="#5a1a1a")
    cb = fig.colorbar(m, ax=ac, pad=0.01); cb.set_label("Time to progression gained (days)")
    cb.set_ticks([-40, -10, -1, 0, 1, 10, 40]); cb.set_ticklabels(["−40", "−10", "−1", "0", "1", "10", "40"])
    panel(ac, "c", -0.06)
    fig.savefig(FIGURES / "fig3.png", bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- Fig. 4
def fig4():
    curves, limits = read_csv("limit_curves.csv"), read_csv("upper_limits.csv")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    colors = {0.3: "#c2562a", 0.4: BLUE, 0.5: PURPLE}
    for a in (0.3, 0.4, 0.5):
        rows = [r for r in curves if float(r["shrinkage"]) == a]
        ax[0].plot(100 * col(rows, "f0"), col(rows, "delta_T_days"), c=colors[a], lw=1.6,
                   label="%d%% response" % round(100 * a))
    for a, xy in ((0.3, (0.06, -1.6)), (0.4, (4.2, 1.3))):
        u = float([r for r in limits if float(r["alpha"]) == 0.1 and abs(float(r["shrinkage"]) - a) < 1e-9][0]
                  ["upper_limit_percent"])
        ax[0].plot(u, 0, "o", c=colors[a], ms=5, zorder=5)
        ax[0].annotate(f"upper limit {u:.2f}%", (u, 0), xy, fontsize=7, color=colors[a],
                       arrowprops=dict(arrowstyle="-", color=colors[a], lw=0.6))
    ax[0].axhline(0, c="k", lw=0.6); ax[0].set_xscale("log"); ax[0].set_xlim(1e-4, 15); ax[0].set_ylim(-4, 2.6)
    ax[0].text(0.02, 0.95, "interruption better", transform=ax[0].transAxes, fontsize=7.5, color="#444")
    ax[0].text(0.02, 0.04, "continuous therapy better", transform=ax[0].transAxes, fontsize=7.5, color="#444")
    ax[0].set(xlabel="Pre-existing resistant cells (% of tumor, log scale)", ylabel="Time to progression gained (days)",
              title="Weak competition: effect of a 4-week interruption"); ax[0].title.set_fontsize(8.5)
    ax[0].legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0.0, 0.08))

    shrink = np.linspace(0.29, 0.51, 100)
    fmax = 100 * np.array([max_compatible_fraction_formula(a, 56, 0.012, 0.6) for a in shrink])
    ax[1].fill_between(100 * shrink, fmax, 100, color="#eeeeee", lw=0)
    ax[1].plot(100 * shrink, fmax, c=GRAY, lw=1.4, ls="--", label="maximum fraction compatible\nwith the response")
    ax[1].text(41, 52, "not compatible with the observed response", fontsize=7, color="#666", ha="center")
    for alpha, c, marker, label in ((0.1, BLUE, "-o", "upper limit, weak competition"),
                                    (0.2, ORANGE, "-s", "upper limit, moderate competition")):
        rows = [r for r in limits if float(r["alpha"]) == alpha and r["upper_limit_percent"] != "none"]
        ax[1].plot(100 * col(rows, "shrinkage"), col(rows, "upper_limit_percent"), marker, c=c, ms=4, lw=1.5,
                   label=label)
    ax[1].set_yscale("log"); ax[1].set_ylim(0.5, 100); ax[1].set_xlim(29, 51)
    ax[1].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax[1].text(47, 8, "no upper limit below\nthe maximum\ncompatible fraction", fontsize=7, color="#555", ha="center")
    ax[1].set(xlabel="Tumor shrinkage before the interruption (%)", ylabel="Pre-existing resistant cells (%)",
              title="Tolerable vs maximal possible pre-existing resistance"); ax[1].title.set_fontsize(8.5)
    ax[1].legend(frameon=False, fontsize=6.8, loc="lower right")
    panel(ax[0], "a"); panel(ax[1], "b"); fig.tight_layout()
    fig.savefig(FIGURES / "fig4.png"); plt.close(fig)


# ---------------------------------------------------------------- Supplementary figures
def figS1():
    gs, sens = read_csv("timing_gain_by_start_day.csv"), read_csv("start_time_sensitivity.csv")
    starts, d = col(gs, "start_day"), col(gs, "delta_T_days")
    fig, a = plt.subplots(figsize=(5.2, 3.0))
    a.plot(starts, np.gradient(d, starts), c=GRAY, lw=1.4, label="finite difference of ΔT")
    a.plot(col(sens, "start_day"), col(sens, "eq_S4"), "o", c=BLUE, ms=5, label="Eq. (S4), analytical")
    a.axhline(0, c="k", lw=0.6)
    a.set(xlabel="Start day of 28-day interruption, $t_1$", ylabel="∂ΔT/∂$t_1$")
    a.legend(frameon=False, fontsize=7.5, loc="lower right"); fig.tight_layout()
    fig.savefig(FIGURES / "figS1.png"); plt.close(fig)


def figS2():
    rows = read_csv("fixed_drug_effect_curve.csv")
    f = col(rows, "f0_percent")
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.9))
    ax[0].plot(f, col(rows, "shrinkage_percent"), c=BLUE, lw=1.6); ax[0].axhline(30, c=GRAY, ls=":", lw=1)
    ax[0].set(xlabel="Pre-existing resistant cells (%)", ylabel="Shrinkage at day 56 (%)", ylim=(25, 45))
    ax[1].plot(f, col(rows, "delta_T_days"), c=ORANGE, lw=1.6); ax[1].axhline(0, c="k", lw=0.6)
    ax[1].set(xlabel="Pre-existing resistant cells (%)", ylabel="Time to progression gained (days)")
    panel(ax[0], "a"); panel(ax[1], "b"); fig.tight_layout()
    fig.savefig(FIGURES / "figS2.png"); plt.close(fig)


def figS3():
    p, start = TIMING_EXAMPLE, 38
    x1 = flow(initial_state(p), start, p, 1); nadir = x1.sum()
    tau_q = time_to_criterion_off_treatment(p, start)
    off_marker = integrate(x1, start, start + 400, p, 0, events=_rising_through(1.1 * nadir)).t_events[0][0] - start
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.3))
    t, s, r = trajectory(p, [(520, 1)]); ax[0].plot(t, s + r, c=BLUE, lw=1.6, label="Continuous")
    for off, ls, c, label in ((28, "--", RED, "Fixed 4-week interruption"),
                              (0.8 * tau_q, "-", ORANGE, "Restart after 0.8 τ$_q$ (%.1f d)" % (0.8 * tau_q)),
                              (off_marker, "-.", PURPLE, "Restart at 1.10× nadir (%.1f d)" % off_marker)):
        t, s, r = trajectory(p, [(start, 1), (off, 0), (520 - start - off, 1)])
        ax[0].plot(t, s + r, c=c, ls=ls, lw=1.3, label=label)
    ax[0].axhline(1.2 * nadir, c=GRAY, ls=":", lw=1); ax[0].text(200, 1.2 * nadir + 0.004, "1.20 × nadir", fontsize=7, color="#444")
    ax[0].set(xlabel="Days since start of therapy", ylabel="Tumor burden (relative)", xlim=(0, 520), ylim=(0, 0.27))
    ax[0].legend(frameon=False, fontsize=7, loc="center right", bbox_to_anchor=(1, 0.6))
    rows = read_csv("restart_rules_map.csv"); bins = np.arange(0, 30, 1)
    ax[1].hist(col(rows, "delta_T_0.8tau_q"), bins=bins, color=ORANGE, histtype="step", lw=1.8, label="restart after 0.8 τ$_q$")
    ax[1].hist(col(rows, "delta_T_marker"), bins=bins, color=PURPLE, histtype="step", lw=1.8, label="restart at 1.10× nadir")
    ax[1].set(xlabel="Time to nadir-relative progression gained (days)", ylabel="Virtual tumors (of 1,368)")
    ax[1].legend(frameon=False, fontsize=7.5)
    panel(ax[0], "a"); panel(ax[1], "b"); fig.tight_layout()
    fig.savefig(FIGURES / "figS3.png"); plt.close(fig)


if __name__ == "__main__":
    for make in (fig1, fig2, fig3, fig4, figS1, figS2, figS3):
        make(); print(f"  {make.__name__} done")
