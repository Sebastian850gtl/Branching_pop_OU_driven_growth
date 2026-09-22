"""
Vanishing-stationary-noise / diverging-annealed-growth experiment
for the OU growth-fragmentation model.

Scaling:
    bar_sigma(theta) = bar_sigma0 * theta**alpha
with default
    m = 0.5, bar_sigma0 = 0.2, alpha = 1/4.

Then
    sigma(theta) = sqrt(2 theta) * bar_sigma(theta),
    lambda_theta = m + bar_sigma(theta)^2 / theta,
so bar_sigma(theta) -> 0 while lambda_theta -> infinity as theta -> 0.

Time is compared in OU correlation-time units s = theta t.

Diagnostics:
- mean and median individual rolling slopes of log total mass
- 5%-95% and 25%-75% quantile ribbons
- Monte Carlo annealed slope
- exact first-moment slope
- top-5% and top-1% mass shares
- effective number of contributing runs
- spine stationary mean and spine displacement in stationary std units

Run from repository root:
    python tests/test_ougf_vanishing_stationary_noise.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.ougf_model import Branching_OUGF


def division_rate(r, x):
    return np.maximum(r, 0.0) * x


def division_kernel(x):
    return x / 2.0


def stationary_std(theta, bar_sigma0, alpha):
    return bar_sigma0 * theta**alpha


def sigma_theta(theta, bar_sigma0, alpha):
    return np.sqrt(2.0 * theta) * stationary_std(theta, bar_sigma0, alpha)


def lambda_theta(m, theta, bar_sigma0, alpha):
    bs = stationary_std(theta, bar_sigma0, alpha)
    return m + bs**2 / theta


def spine_stationary_mean(m, theta, bar_sigma0, alpha):
    bs = stationary_std(theta, bar_sigma0, alpha)
    return m + 2.0 * bs**2 / theta


def spine_displacement_in_std(theta, bar_sigma0, alpha):
    bs = stationary_std(theta, bar_sigma0, alpha)
    return 2.0 * bs / theta


def exact_log_first_moment(t, m, theta, sigma, initial_mass=1.0):
    t = np.asarray(t, dtype=float)
    variance_integral = (
        sigma**2 / theta**2
        * (
            t
            - 2.0 * (1.0 - np.exp(-theta * t)) / theta
            + (1.0 - np.exp(-2.0 * theta * t)) / (2.0 * theta)
        )
    )
    return np.log(initial_mass) + m * t + 0.5 * variance_integral


def total_log_mass(snapshot):
    log_sizes = snapshot[:, 1]
    a = np.max(log_sizes)
    return float(a + np.log(np.sum(np.exp(log_sizes - a))))


def interpolate_log_mass(times, snapshots, t_grid):
    times = np.asarray(times, dtype=float)
    log_mass = np.asarray([total_log_mass(z) for z in snapshots], dtype=float)

    out = np.full_like(t_grid, np.nan, dtype=float)
    if len(times) < 2:
        return out

    valid = t_grid <= times[-1]
    out[valid] = np.interp(t_grid[valid], times, log_mass)
    return out


def rolling_slope(t_grid, y, physical_window, min_points=5):
    out = np.full_like(t_grid, np.nan, dtype=float)
    finite = np.isfinite(y)

    for j, tj in enumerate(t_grid):
        mask = finite & (t_grid >= tj - physical_window) & (t_grid <= tj)
        if np.count_nonzero(mask) < min_points:
            continue

        x = t_grid[mask]
        z = y[mask]
        xc = x - np.mean(x)
        denom = np.dot(xc, xc)

        if denom > 0.0:
            out[j] = np.dot(xc, z - np.mean(z)) / denom

    return out


def logmeanexp_columns(log_values):
    out = np.full(log_values.shape[1], np.nan, dtype=float)

    for j in range(log_values.shape[1]):
        vals = log_values[:, j]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue

        a = np.max(vals)
        out[j] = a + np.log(np.mean(np.exp(vals - a)))

    return out


def maximal_common_prefix(mask):
    if np.all(mask):
        return mask.copy()

    first_bad = np.where(~mask)[0][0]
    out = np.zeros_like(mask, dtype=bool)
    out[:first_bad] = True
    return out


def concentration_from_log_masses(log_mass_paths):
    n_times = log_mass_paths.shape[1]
    top5 = np.full(n_times, np.nan)
    top1 = np.full(n_times, np.nan)
    neff = np.full(n_times, np.nan)

    for j in range(n_times):
        logs = log_mass_paths[:, j]
        logs = logs[np.isfinite(logs)]
        n = len(logs)
        if n == 0:
            continue

        a = np.max(logs)
        w = np.exp(logs - a)
        w /= np.sum(w)
        w = np.sort(w)[::-1]

        k5 = max(1, int(np.ceil(0.05 * n)))
        k1 = max(1, int(np.ceil(0.01 * n)))

        top5[j] = np.sum(w[:k5])
        top1[j] = np.sum(w[:k1])
        neff[j] = 1.0 / np.sum(w**2)

    return top5, top1, neff


def simulate_theta(
    *,
    m,
    theta,
    bar_sigma0,
    alpha,
    n_runs,
    s_grid,
    s_window,
    init_birth_size,
    max_cells,
    dtmin,
    dtmax,
    dtobs,
    eps,
    seed,
):
    bs = stationary_std(theta, bar_sigma0, alpha)
    sigma = sigma_theta(theta, bar_sigma0, alpha)

    t_grid = s_grid / theta
    physical_window = s_window / theta

    np.random.seed(seed)

    model = Branching_OUGF(
        ou_growth_rate_parameters=(m, theta, sigma),
        division_rate=division_rate,
        division_kernel=division_kernel,
    )

    log_mass_paths = np.full((n_runs, len(t_grid)), np.nan)
    individual_slopes = np.full_like(log_mass_paths, np.nan)
    final_times = np.zeros(n_runs)

    for run in range(n_runs):
        times, snapshots = model.run(
            Tmax=t_grid[-1],
            init=(m, init_birth_size),
            max_number_of_cells=max_cells,
            dtmin=dtmin,
            dtmax=dtmax,
            dtobs=dtobs,
            eps=eps,
        )

        path = interpolate_log_mass(times, snapshots, t_grid)
        log_mass_paths[run] = path
        individual_slopes[run] = rolling_slope(
            t_grid, path, physical_window=physical_window
        )
        final_times[run] = times[-1]

        if (run + 1) % 10 == 0 or run + 1 == n_runs:
            print(
                f"    theta={theta:g} run {run+1:3d}/{n_runs} "
                f"| median final s={theta*np.median(final_times[:run+1]):.3f}"
            )

    mean_slope = np.nanmean(individual_slopes, axis=0)
    median_slope = np.nanmedian(individual_slopes, axis=0)
    q05 = np.nanquantile(individual_slopes, 0.05, axis=0)
    q25 = np.nanquantile(individual_slopes, 0.25, axis=0)
    q75 = np.nanquantile(individual_slopes, 0.75, axis=0)
    q95 = np.nanquantile(individual_slopes, 0.95, axis=0)

    log_mc_mean = logmeanexp_columns(log_mass_paths)
    annealed = rolling_slope(t_grid, log_mc_mean, physical_window)

    exact_log_mean = exact_log_first_moment(
        t_grid, m, theta, sigma, initial_mass=init_birth_size
    )
    exact = rolling_slope(t_grid, exact_log_mean, physical_window)

    top5, top1, neff = concentration_from_log_masses(log_mass_paths)

    safe = maximal_common_prefix(np.all(np.isfinite(log_mass_paths), axis=0))

    return {
        "theta": theta,
        "bar_sigma": bs,
        "sigma": sigma,
        "lambda": lambda_theta(m, theta, bar_sigma0, alpha),
        "spine_mean": spine_stationary_mean(m, theta, bar_sigma0, alpha),
        "spine_std_distance": spine_displacement_in_std(theta, bar_sigma0, alpha),
        "s_grid": s_grid,
        "t_grid": t_grid,
        "mean": mean_slope,
        "median": median_slope,
        "q05": q05,
        "q25": q25,
        "q75": q75,
        "q95": q95,
        "annealed": annealed,
        "exact": exact,
        "top5": top5,
        "top1": top1,
        "neff": neff,
        "safe": safe,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--s-max", type=float, default=6.0)
    parser.add_argument("--s-window", type=float, default=0.75)
    parser.add_argument("--max-cells", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=13579)
    args = parser.parse_args()

    m = 0.50
    bar_sigma0 = 0.20
    alpha = 0.25

    thetas = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.025, 0.01]

    ds = 0.05
    s_grid = np.arange(0.0, args.s_max + 0.5 * ds, ds)

    dtmin = 1e-6
    dtmax = 2e-2
    dtobs = 5e-2
    eps = 0.05

    print("\nVanishing-noise / diverging-annealed-growth regime")
    print(f"m={m}, bar_sigma0={bar_sigma0}, alpha={alpha}")
    print()

    results = {}

    for j, theta in enumerate(thetas):
        bs = stationary_std(theta, bar_sigma0, alpha)
        sigma = sigma_theta(theta, bar_sigma0, alpha)
        lam = lambda_theta(m, theta, bar_sigma0, alpha)
        sm = spine_stationary_mean(m, theta, bar_sigma0, alpha)
        sd = spine_displacement_in_std(theta, bar_sigma0, alpha)

        print(
            f"theta={theta:7.4f} | bar_sigma={bs:.5f} | "
            f"sigma={sigma:.5f} | lambda={lam:.5f} | "
            f"spine_mean={sm:.5f} | spine_shift/std={sd:.2f} | "
            f"Tmax={args.s_max/theta:.1f}"
        )

        results[theta] = simulate_theta(
            m=m,
            theta=theta,
            bar_sigma0=bar_sigma0,
            alpha=alpha,
            n_runs=args.n_runs,
            s_grid=s_grid,
            s_window=args.s_window,
            init_birth_size=1.0,
            max_cells=args.max_cells,
            dtmin=dtmin,
            dtmax=dtmax,
            dtobs=dtobs,
            eps=eps,
            seed=args.seed + j,
        )

    ncols = 3
    nrows = int(np.ceil(len(thetas) / ncols))

    # Growth figure
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(13.5, 3.9*nrows), dpi=180, sharex=True
    )
    axes = np.asarray(axes).ravel()

    for ax, theta in zip(axes, thetas):
        r = results[theta]
        safe = (
            r["safe"]
            & np.isfinite(r["mean"])
            & np.isfinite(r["annealed"])
            & np.isfinite(r["exact"])
        )

        ax.axhline(r["lambda"], linestyle="--", linewidth=1.2,
                   label=rf"$\lambda_\theta={r['lambda']:.3f}$")
        ax.plot(r["s_grid"][safe], r["mean"][safe], label="mean individual slope")
        ax.plot(r["s_grid"][safe], r["annealed"][safe], label="MC annealed")
        ax.plot(r["s_grid"][safe], r["exact"][safe], linestyle=":",
                linewidth=2, label="exact first moment")

        ax.set_title(
            rf"$\theta={theta:g}$, $\bar\sigma_\theta={r['bar_sigma']:.3f}$"
        )
        ax.set_xlabel(r"$s=\theta t$")
        ax.set_ylabel("physical-time local slope")
        ax.grid(alpha=0.2)

    for ax in axes[len(thetas):]:
        ax.axis("off")

    axes[0].legend(fontsize=8)
    fig.suptitle(
        rf"Vanishing OU stationary noise: "
        rf"$\bar\sigma_\theta={bar_sigma0}\theta^{{1/4}}$"
    )
    fig.tight_layout()

    out_growth = Path(__file__).with_name("ougf_vanishing_noise_growth.png")
    fig.savefig(out_growth, bbox_inches="tight")

    # Quantile figure
    fig2, axes2 = plt.subplots(
        nrows, ncols, figsize=(13.5, 3.9*nrows), dpi=180, sharex=True
    )
    axes2 = np.asarray(axes2).ravel()

    for ax, theta in zip(axes2, thetas):
        r = results[theta]
        safe = (
            r["safe"]
            & np.isfinite(r["mean"])
            & np.isfinite(r["median"])
            & np.isfinite(r["q05"])
            & np.isfinite(r["q25"])
            & np.isfinite(r["q75"])
            & np.isfinite(r["q95"])
            & np.isfinite(r["exact"])
        )
        s = r["s_grid"][safe]

        mean_line, = ax.plot(s, r["mean"][safe], linewidth=2, label="mean")
        c = mean_line.get_color()
        ax.fill_between(s, r["q05"][safe], r["q95"][safe],
                        alpha=0.15, color=c, label="5%-95%")
        ax.fill_between(s, r["q25"][safe], r["q75"][safe],
                        alpha=0.30, color=c, label="25%-75%")
        ax.plot(s, r["median"][safe], linestyle="--", color=c, label="median")
        ax.plot(s, r["exact"][safe], linestyle=":", linewidth=2,
                label="exact first moment")
        ax.axhline(r["lambda"], linestyle="-.", linewidth=1,
                   label=rf"$\lambda_\theta={r['lambda']:.3f}$")

        ax.set_title(rf"$\theta={theta:g}$")
        ax.set_xlabel(r"$s=\theta t$")
        ax.set_ylabel("physical-time local slope")
        ax.grid(alpha=0.2)

    for ax in axes2[len(thetas):]:
        ax.axis("off")

    axes2[0].legend(fontsize=7)
    fig2.suptitle("Quantiles of individual finite-time growth rates")
    fig2.tight_layout()

    out_quant = Path(__file__).with_name("ougf_vanishing_noise_quantiles.png")
    fig2.savefig(out_quant, bbox_inches="tight")

    # Concentration figure
    fig3, axes3 = plt.subplots(3, 1, figsize=(9.5, 10), dpi=180, sharex=True)

    for theta in thetas:
        r = results[theta]
        safe = r["safe"]
        axes3[0].plot(r["s_grid"][safe], r["top5"][safe], label=rf"$\theta={theta:g}$")
        axes3[1].plot(r["s_grid"][safe], r["top1"][safe], label=rf"$\theta={theta:g}$")
        axes3[2].plot(r["s_grid"][safe], r["neff"][safe]/args.n_runs,
                      label=rf"$\theta={theta:g}$")

    axes3[0].axhline(0.05, linestyle="--", linewidth=1)
    axes3[0].set_ylabel("mass share")
    axes3[0].set_title("Share of first moment carried by top 5% of runs")
    axes3[0].grid(alpha=0.2)
    axes3[0].legend(ncol=2)

    axes3[1].axhline(0.01, linestyle="--", linewidth=1)
    axes3[1].set_ylabel("mass share")
    axes3[1].set_title("Share carried by top 1% of runs")
    axes3[1].grid(alpha=0.2)

    axes3[2].axhline(1.0, linestyle="--", linewidth=1)
    axes3[2].set_xlabel(r"$s=\theta t$")
    axes3[2].set_ylabel(r"$N_{\rm eff}/N$")
    axes3[2].set_title("Effective fraction of runs contributing to first moment")
    axes3[2].grid(alpha=0.2)

    fig3.tight_layout()

    out_conc = Path(__file__).with_name("ougf_vanishing_noise_concentration.png")
    fig3.savefig(out_conc, bbox_inches="tight")

    # Scaling figure
    theta_array = np.asarray(thetas, dtype=float)
    bs_array = np.asarray([stationary_std(t, bar_sigma0, alpha) for t in thetas])
    lam_array = np.asarray([lambda_theta(m, t, bar_sigma0, alpha) for t in thetas])
    spine_std_array = np.asarray(
        [spine_displacement_in_std(t, bar_sigma0, alpha) for t in thetas]
    )

    fig4, axes4 = plt.subplots(3, 1, figsize=(8.5, 9), dpi=180, sharex=True)

    axes4[0].plot(theta_array, bs_array, marker="o")
    axes4[0].set_ylabel(r"$\bar\sigma_\theta$")
    axes4[0].set_title("Ordinary stationary fluctuations vanish")
    axes4[0].grid(alpha=0.2)

    axes4[1].plot(theta_array, lam_array, marker="o")
    axes4[1].set_ylabel(r"$\lambda_\theta$")
    axes4[1].set_title("Annealed Malthus exponent increases")
    axes4[1].grid(alpha=0.2)

    axes4[2].plot(theta_array, spine_std_array, marker="o")
    axes4[2].set_xlabel(r"$\theta$")
    axes4[2].set_ylabel(r"$(m_{\rm spine}-m)/\bar\sigma_\theta$")
    axes4[2].set_title("Mass-biased lineage moves farther into the OU tail")
    axes4[2].grid(alpha=0.2)

    for ax in axes4:
        ax.set_xscale("log")
        ax.invert_xaxis()

    fig4.tight_layout()

    out_scaling = Path(__file__).with_name("ougf_vanishing_noise_scaling.png")
    fig4.savefig(out_scaling, bbox_inches="tight")

    print("\nLast common uncensored point")
    print("----------------------------")

    for theta in thetas:
        r = results[theta]
        valid = np.where(
            r["safe"]
            & np.isfinite(r["mean"])
            & np.isfinite(r["median"])
            & np.isfinite(r["exact"])
        )[0]

        if len(valid) == 0:
            print(f"theta={theta:g}: insufficient data")
            continue

        j = valid[-1]
        print(
            f"theta={theta:7.4f} | s={r['s_grid'][j]:5.2f} | "
            f"t={r['t_grid'][j]:8.2f} | bar_sigma={r['bar_sigma']:.4f} | "
            f"lambda={r['lambda']:.4f} | mean={r['mean'][j]:.4f} | "
            f"median={r['median'][j]:.4f} | exact={r['exact'][j]:.4f} | "
            f"top5={r['top5'][j]:.3f} | top1={r['top1'][j]:.3f} | "
            f"Neff/N={r['neff'][j]/args.n_runs:.3f}"
        )

        if r["s_grid"][j] < args.s_max - ds:
            print("    WARNING: first-run censoring occurred before s_max.")

    print("\nSaved figures:")
    print(f"  {out_growth}")
    print(f"  {out_quant}")
    print(f"  {out_conc}")
    print(f"  {out_scaling}")

    plt.show()


if __name__ == "__main__":
    main()
