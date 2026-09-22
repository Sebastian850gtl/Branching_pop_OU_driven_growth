"""
Theta sweep for the OU growth-fragmentation model on the correlation-time scale.

We keep the stationary OU standard deviation fixed:

    bar_sigma = sigma / sqrt(2 theta),   bar_sigma / m = 0.1,

so that

    sigma(theta) = sqrt(2 theta) * bar_sigma

and

    lambda_theta = m + bar_sigma**2 / theta.

Time is compared through s = theta t, and the rolling regression window is
also fixed in s-units.

Run:
    python tests/test_oufg_large_theta_rescaled_gap.py

Useful options:
    --n-runs 100
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
    return x *np.random.rand(len(x))


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
    t_grid = s_grid / m
    physical_window = s_window / m

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
                f"| median final s={theta*np.median(final_times[:run+1]):.2f}"
            )

    typical = np.nanmean(individual_slopes, axis=0)

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

    safe = maximal_common_prefix(np.all(np.isfinite(log_mass_paths), axis=0))

    return {
        "theta": theta,
        "sigma": sigma,
        "lambda": lambda_theta(m, theta, bar_sigma),
        "s_grid": s_grid,
        "t_grid": t_grid,
        "typical": typical,
        "annealed": annealed,
        "exact": exact,
        "safe": safe,
        "log_mass_paths": log_mass_paths,
        "individual_slopes": individual_slopes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=60)
    parser.add_argument("--s-max", type=float, default=10.0)
    parser.add_argument("--s-window", type=float, default=0.75)
    parser.add_argument("--max-cells", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    m = 0.5
    bar_sigma = 0.1 * m

    thetas = [8.00, 6.00, 4.00, 3.00, 2.00 ,1.00]

    ds = 0.05
    s_grid = np.arange(0.0, args.s_max + 0.5 * ds, ds)

    dtmin = 1e-6
    dtmax = 2e-2
    dtobs = 5e-2
    eps = 0.05

    print("\nFixed stationary-variance theta sweep")
    print(f"m={m}, bar_sigma={bar_sigma}, bar_sigma/m={bar_sigma/m:.3f}")
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

    # One panel per theta.
    fig, axes = plt.subplots(
        2, 3, figsize=(13, 8), dpi=180, sharex=True
    )
    axes = axes.ravel()

    for ax, theta in zip(axes, thetas):
        r = results[theta]
        safe = (
            r["safe"]
            & np.isfinite(r["typical"])
            & np.isfinite(r["annealed"])
            & np.isfinite(r["exact"])
        )

        ax.axhline(
            r["lambda"],
            linestyle="--",
            linewidth=1.2,
            label=rf"$\lambda_\theta={r['lambda']:.3f}$",
        )
        ax.plot(r["s_grid"][safe], r["typical"][safe], label="typical")
        ax.plot(r["s_grid"][safe], r["annealed"][safe], label="annealed")
        ax.plot(
            r["s_grid"][safe],
            r["exact"][safe],
            linestyle=":",
            label="exact first moment",
        )
        ax.set_title(rf"$\theta={theta:g}$")
        ax.set_xlabel(r"$s=\theta t$")
        ax.set_ylabel("physical-time local slope")
        ax.grid(alpha=0.2)

    axes[0].legend(fontsize=8)

    fig.suptitle(
        "OU growth-fragmentation in correlation-time units\n"
        rf"$\bar\sigma/m=0.4$, rolling window $\Delta s={args.s_window:g}$"
    )
    fig.tight_layout()

    output1 = Path(__file__).with_name("oufg_large_theta_rescaled_growth.png")
    fig.savefig(output1, bbox_inches="tight")

    # Compare gaps across theta.
    fig2, axes2 = plt.subplots(
        2, 1, figsize=(8.5, 8), dpi=180, sharex=True
    )

    for theta in thetas:
        r = results[theta]
        gap = r["annealed"] - r["typical"]
        safe = r["safe"] & np.isfinite(gap)

        axes2[0].plot(
            r["s_grid"][safe],
            gap[safe],
            linewidth=2,
            label=rf"$\theta={theta:g}$",
        )

        correction = r["lambda"] - m
        axes2[1].plot(
            r["s_grid"][safe],
            gap[safe] / correction,
            linewidth=2,
            label=rf"$\theta={theta:g}$",
        )

    axes2[0].axhline(0.0, linestyle="--", linewidth=1)
    axes2[0].set_ylabel("annealed - typical slope")
    axes2[0].set_title("Annealed/typical gap")
    axes2[0].grid(alpha=0.2)
    axes2[0].legend(ncol=2)

    axes2[1].axhline(0.0, linestyle="--", linewidth=1)
    axes2[1].set_xlabel(r"rescaled time $s=\theta t$")
    axes2[1].set_ylabel(r"gap / $(\lambda_\theta-m)$")
    axes2[1].set_title("Gap normalized by the OU Malthus correction")
    axes2[1].grid(alpha=0.2)

    fig2.tight_layout()

    output2 = Path(__file__).with_name("oufg_large_theta_rescaled_gap.png")
    fig2.savefig(output2, bbox_inches="tight")

    print("\nLast common uncensored point")
    for theta in thetas:
        r = results[theta]
        valid = np.where(
            r["safe"]
            & np.isfinite(r["typical"])
            & np.isfinite(r["annealed"])
            & np.isfinite(r["exact"])
        )[0]

        if len(valid) == 0:
            print(f"theta={theta:g}: insufficient data")
            continue

        j = valid[-1]
        gap = r["annealed"][j] - r["typical"][j]

        print(
            f"theta={theta:5.2f} | s={r['s_grid'][j]:5.2f} | "
            f"t={r['t_grid'][j]:6.2f} | lambda={r['lambda']:.4f} | "
            f"typ={r['typical'][j]:.4f} | ann={r['annealed'][j]:.4f} | "
            f"exact={r['exact'][j]:.4f} | gap={gap:.4f}"
        )

        if r["s_grid"][j] < args.s_max - ds:
            print("    WARNING: increase --max-cells to extend the common horizon.")

    print(f"\nSaved: {output1}")
    print(f"Saved: {output2}")
    plt.show()


if __name__ == "__main__":
    main()
