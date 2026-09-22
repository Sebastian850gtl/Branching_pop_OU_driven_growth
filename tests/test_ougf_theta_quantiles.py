"""
Quantile ribbons and rare-trajectory diagnostics for the OU growth-fragmentation model.

We keep

    bar_sigma = sigma / sqrt(2 theta)

fixed with bar_sigma / m = 0.4, and compare trajectories on the rescaled
correlation-time axis

    s = theta t.

For each theta we show:
    - mean individual local slope,
    - median individual local slope,
    - 25%-75% quantile ribbon,
    - 5%-95% quantile ribbon,
    - annealed slope = rolling slope of log(mean_i M_t^i),
    - exact first-moment slope.

Because quantiles alone do not reveal whether a tiny fraction of trajectories
dominates the Monte Carlo first moment, the script also computes:
    - mass share carried by the largest 5% of runs,
    - mass share carried by the largest 1% of runs,
    - effective number of contributing trajectories:
          N_eff = (sum M_i)^2 / sum M_i^2.

Run:
    python tests/test_ougf_theta_quantiles.py

Useful options:
    --n-runs 150
    --s-max 6
    --s-window 0.75
    --max-cells 300000
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


def sigma_from_stationary_std(theta, bar_sigma):
    return np.sqrt(2.0 * theta) * bar_sigma


def lambda_theta(m, theta, bar_sigma):
    return m + bar_sigma**2 / theta


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
        values = log_values[:, j]
        values = values[np.isfinite(values)]

        if len(values) == 0:
            continue

        a = np.max(values)
        out[j] = a + np.log(np.mean(np.exp(values - a)))

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

    top5_share = np.full(n_times, np.nan)
    top1_share = np.full(n_times, np.nan)
    effective_n = np.full(n_times, np.nan)

    for j in range(n_times):
        logs = log_mass_paths[:, j]
        logs = logs[np.isfinite(logs)]

        n = len(logs)
        if n == 0:
            continue

        a = np.max(logs)
        weights = np.exp(logs - a)
        weights /= np.sum(weights)

        ordered = np.sort(weights)[::-1]

        k5 = max(1, int(np.ceil(0.05 * n)))
        k1 = max(1, int(np.ceil(0.01 * n)))

        top5_share[j] = np.sum(ordered[:k5])
        top1_share[j] = np.sum(ordered[:k1])
        effective_n[j] = 1.0 / np.sum(weights**2)

    return top5_share, top1_share, effective_n


def simulate_theta(
    *,
    m,
    theta,
    bar_sigma,
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
    sigma = sigma_from_stationary_std(theta, bar_sigma)
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

        if (run + 1) % 25 == 0 or run + 1 == n_runs:
            print(f"    theta={theta:g} run {run+1:3d}/{n_runs}")

    mean_slope = np.nanmean(individual_slopes, axis=0)
    median_slope = np.nanmedian(individual_slopes, axis=0)

    q05 = np.nanquantile(individual_slopes, 0.05, axis=0)
    q25 = np.nanquantile(individual_slopes, 0.25, axis=0)
    q75 = np.nanquantile(individual_slopes, 0.75, axis=0)
    q95 = np.nanquantile(individual_slopes, 0.95, axis=0)

    log_mc_mean = logmeanexp_columns(log_mass_paths)
    annealed = rolling_slope(
        t_grid, log_mc_mean, physical_window=physical_window
    )

    exact_log_mean = exact_log_first_moment(
        t_grid, m, theta, sigma, initial_mass=init_birth_size
    )
    exact = rolling_slope(
        t_grid, exact_log_mean, physical_window=physical_window
    )

    top5_share, top1_share, effective_n = concentration_from_log_masses(
        log_mass_paths
    )

    safe = maximal_common_prefix(np.all(np.isfinite(log_mass_paths), axis=0))

    return {
        "theta": theta,
        "sigma": sigma,
        "lambda": lambda_theta(m, theta, bar_sigma),
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
        "top5_share": top5_share,
        "top1_share": top1_share,
        "effective_n": effective_n,
        "safe": safe,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=150)
    parser.add_argument("--s-max", type=float, default=6.0)
    parser.add_argument("--s-window", type=float, default=0.75)
    parser.add_argument("--max-cells", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=24680)
    args = parser.parse_args()

    m = 0.50
    bar_sigma = 0.4 * m

    thetas = [1.00, 0.75, 0.50, 0.35, 0.25, 0.15]

    ds = 0.05
    s_grid = np.arange(0.0, args.s_max + 0.5 * ds, ds)

    dtmin = 1e-6
    dtmax = 2e-2
    dtobs = 5e-2
    eps = 0.05

    print("\nQuantile and rare-trajectory diagnostic")
    print(f"m={m}, bar_sigma={bar_sigma}, bar_sigma/m={bar_sigma/m:.3f}")
    print(f"n_runs={args.n_runs}")
    print()

    results = {}

    for j, theta in enumerate(thetas):
        sigma = sigma_from_stationary_std(theta, bar_sigma)
        lam = lambda_theta(m, theta, bar_sigma)

        print(
            f"theta={theta:5.2f} | sigma={sigma:.5f} | "
            f"lambda_theta={lam:.5f} | Tmax={args.s_max/theta:.2f}"
        )

        results[theta] = simulate_theta(
            m=m,
            theta=theta,
            bar_sigma=bar_sigma,
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

    # Quantile-ribbon figure.
    fig, axes = plt.subplots(
        2, 3, figsize=(13, 8.5), dpi=180, sharex=True
    )
    axes = axes.ravel()

    for ax, theta in zip(axes, thetas):
        r = results[theta]

        safe = (
            r["safe"]
            & np.isfinite(r["mean"])
            & np.isfinite(r["median"])
            & np.isfinite(r["q05"])
            & np.isfinite(r["q25"])
            & np.isfinite(r["q75"])
            & np.isfinite(r["q95"])
            & np.isfinite(r["annealed"])
            & np.isfinite(r["exact"])
        )

        s = r["s_grid"][safe]

        mean_line, = ax.plot(
            s,
            r["mean"][safe],
            linewidth=2,
            label="mean individual slope",
        )
        ribbon_color = mean_line.get_color()

        ax.fill_between(
            s,
            r["q05"][safe],
            r["q95"][safe],
            color=ribbon_color,
            alpha=0.15,
            label="5%-95%",
        )
        ax.fill_between(
            s,
            r["q25"][safe],
            r["q75"][safe],
            color=ribbon_color,
            alpha=0.30,
            label="25%-75%",
        )
        ax.plot(
            s,
            r["median"][safe],
            color=ribbon_color,
            linestyle="--",
            linewidth=1.5,
            label="median",
        )
        ax.plot(
            s,
            r["annealed"][safe],
            linewidth=1.8,
            label="annealed slope",
        )
        ax.plot(
            s,
            r["exact"][safe],
            linestyle=":",
            linewidth=1.8,
            label="exact first moment",
        )
        ax.axhline(
            r["lambda"],
            linestyle="-.",
            linewidth=1,
            label=rf"$\lambda_\theta={r['lambda']:.3f}$",
        )

        ax.set_title(rf"$\theta={theta:g}$")
        ax.set_xlabel(r"$s=\theta t$")
        ax.set_ylabel("physical-time local slope")
        ax.grid(alpha=0.2)

    axes[0].legend(fontsize=7)

    fig.suptitle(
        "Quantiles of individual finite-time growth rates\n"
        rf"$\bar\sigma/m=0.4$, rolling window $\Delta s={args.s_window:g}$"
    )
    fig.tight_layout()

    output1 = Path(__file__).with_name("ougf_theta_quantile_ribbons.png")
    fig.savefig(output1, bbox_inches="tight")

    # Direct rare-trajectory contribution diagnostics.
    fig2, axes2 = plt.subplots(
        3, 1, figsize=(9, 10), dpi=180, sharex=True
    )

    for theta in thetas:
        r = results[theta]
        safe = r["safe"]

        axes2[0].plot(
            r["s_grid"][safe],
            r["top5_share"][safe],
            linewidth=2,
            label=rf"$\theta={theta:g}$",
        )
        axes2[1].plot(
            r["s_grid"][safe],
            r["top1_share"][safe],
            linewidth=2,
            label=rf"$\theta={theta:g}$",
        )
        axes2[2].plot(
            r["s_grid"][safe],
            r["effective_n"][safe] / args.n_runs,
            linewidth=2,
            label=rf"$\theta={theta:g}$",
        )

    axes2[0].axhline(0.05, linestyle="--", linewidth=1)
    axes2[0].set_ylabel("mass share")
    axes2[0].set_title(
        "Share of Monte Carlo mean mass carried by the largest 5% of runs"
    )
    axes2[0].grid(alpha=0.2)
    axes2[0].legend(ncol=2)

    axes2[1].axhline(0.01, linestyle="--", linewidth=1)
    axes2[1].set_ylabel("mass share")
    axes2[1].set_title(
        "Share carried by the largest 1% of runs"
    )
    axes2[1].grid(alpha=0.2)

    axes2[2].axhline(1.0, linestyle="--", linewidth=1)
    axes2[2].set_xlabel(r"rescaled time $s=\theta t$")
    axes2[2].set_ylabel(r"$N_{\rm eff}/N$")
    axes2[2].set_title(
        "Effective fraction of trajectories contributing to the first moment"
    )
    axes2[2].grid(alpha=0.2)

    fig2.tight_layout()

    output2 = Path(__file__).with_name("ougf_theta_mass_concentration.png")
    fig2.savefig(output2, bbox_inches="tight")

    print("\nLast common uncensored point")
    for theta in thetas:
        r = results[theta]

        valid = np.where(
            r["safe"]
            & np.isfinite(r["mean"])
            & np.isfinite(r["median"])
            & np.isfinite(r["q05"])
            & np.isfinite(r["q95"])
        )[0]

        if len(valid) == 0:
            print(f"theta={theta:g}: insufficient data")
            continue

        j = valid[-1]

        print(
            f"theta={theta:5.2f} | s={r['s_grid'][j]:5.2f} | "
            f"mean={r['mean'][j]:.4f} | median={r['median'][j]:.4f} | "
            f"q05={r['q05'][j]:.4f} | q95={r['q95'][j]:.4f} | "
            f"top5={r['top5_share'][j]:.3f} | "
            f"top1={r['top1_share'][j]:.3f} | "
            f"Neff/N={r['effective_n'][j]/args.n_runs:.3f}"
        )

    print(f"\nSaved: {output1}")
    print(f"Saved: {output2}")
    print(
        "\nIf, for example, the top-5% mass share is 0.7, then only 5% of "
        "the trajectories are carrying 70% of the Monte Carlo estimate of "
        "E[M_t]. That is a much more direct rare-lineage diagnostic than "
        "the mean curve alone."
    )

    plt.show()


if __name__ == "__main__":
    main()
