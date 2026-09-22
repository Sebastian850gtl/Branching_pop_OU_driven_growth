"""
Small-theta OU growth-fragmentation diagnostics on the OU time scale s = theta t.

For each theta, this script studies four observables from the SAME simulations:

1. Total mass
       M_t = sum_u X_u(t)

2. Number of cells
       N_t = # V_t

3. h-mass / reproductive-value mass
       H_t = sum_u h(X_u(t), R_u(t)),
   with
       h(x,r) = x * exp(r/theta).

4. h-biased OU rate inside each population
       R_h(t) = sum_u h_u R_u / sum_u h_u.

For M_t, N_t, and H_t we compare:

    typical(t)
      = mean over runs of rolling_slope(log observable_i(t))

and

    annealed(t)
      = rolling_slope(log(mean over runs observable_i(t))).

For total mass M_t there is an exact finite-time first-moment curve.

For h-mass H_t, h is an eigenfunction, so

    E[H_t] = H_0 exp(lambda_theta t)

exactly, with

    lambda_theta = m + sigma^2/(2 theta^2)
                 = m + bar_sigma^2/theta.

Hence the exact h-mass annealed growth rate is lambda_theta from time 0.

For N_t we plot lambda_theta only as the asymptotic Malthus reference; there
is no explicit finite-time first-moment formula included here.

The comparison time is

    s = theta t,

and the rolling window is fixed in s-units:

    t_grid = s_grid / theta,
    physical_window = s_window / theta.

Run from the repository root:

    python tests/test_ougf_small_theta_mass_number_h.py

Useful options:

    --n-runs 100
    --s-max 6
    --s-window 0.75
    --max-cells 300000
    --bar-sigma-ratio 0.1

You can also override theta values, e.g.

    --thetas 1 0.75 0.5 0.35 0.25 0.15
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


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def division_rate(r, x):
    return np.maximum(r, 0.0) * x


def division_kernel(x):
    return x / 2.0


def sigma_from_stationary_std(theta, bar_sigma):
    return np.sqrt(2.0 * theta) * bar_sigma


def lambda_theta(m, theta, bar_sigma):
    return m + bar_sigma**2 / theta


def spine_stationary_mean(m, theta, bar_sigma):
    """
    Stationary mean of the OU rate under the h-transform associated with
        h(x,r) = x exp(r/theta).
    """
    return m + 2.0 * bar_sigma**2 / theta


# ---------------------------------------------------------------------
# Exact first moment for ordinary total mass
# ---------------------------------------------------------------------

def exact_log_first_moment_mass(t, m, theta, sigma, initial_mass=1.0):
    t = np.asarray(t, dtype=float)

    variance_integral = (
        sigma**2
        / theta**2
        * (
            t
            - 2.0 * (1.0 - np.exp(-theta * t)) / theta
            + (1.0 - np.exp(-2.0 * theta * t)) / (2.0 * theta)
        )
    )

    return (
        np.log(initial_mass)
        + m * t
        + 0.5 * variance_integral
    )


# ---------------------------------------------------------------------
# Stable snapshot observables
# ---------------------------------------------------------------------

def logsumexp(values):
    """
    Small local implementation to avoid requiring scipy.
    """
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return -np.inf

    a = np.max(values)
    return float(a + np.log(np.sum(np.exp(values - a))))


def snapshot_log_mass(snapshot):
    """
    log M_t where M_t = sum X_u.
    State convention is (R, log X).
    """
    return logsumexp(snapshot[:, 1])


def snapshot_log_number(snapshot):
    """
    log N_t.
    """
    return float(np.log(len(snapshot)))


def snapshot_log_hmass(snapshot, theta):
    """
    log H_t, with
        H_t = sum_u X_u exp(R_u/theta).

    Work in log-space because exp(R/theta) can be enormous at small theta.
    """
    rates = snapshot[:, 0]
    log_sizes = snapshot[:, 1]

    log_h = log_sizes + rates / theta
    return logsumexp(log_h)


def snapshot_h_biased_rate(snapshot, theta):
    """
    R_h = sum h_u R_u / sum h_u.

    This is computed with stabilized normalized weights.
    """
    rates = snapshot[:, 0]
    log_sizes = snapshot[:, 1]

    log_h = log_sizes + rates / theta
    a = np.max(log_h)

    weights = np.exp(log_h - a)
    weights /= np.sum(weights)

    return float(np.sum(weights * rates))


# ---------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------

def interpolate_snapshot_statistic(
    times,
    snapshots,
    t_grid,
    statistic,
):
    """
    Evaluate a scalar snapshot statistic at recorded times, then linearly
    interpolate it onto t_grid.

    For log N_t this smooths the step function slightly between observation
    times; with small dtobs this is adequate for rolling-slope diagnostics.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(
        [statistic(z) for z in snapshots],
        dtype=float,
    )

    out = np.full_like(t_grid, np.nan, dtype=float)

    if len(times) < 2:
        return out

    valid = t_grid <= times[-1]

    out[valid] = np.interp(
        t_grid[valid],
        times,
        values,
    )

    return out


# ---------------------------------------------------------------------
# Rolling regressions and Monte Carlo averages
# ---------------------------------------------------------------------

def rolling_slope(t_grid, y, physical_window, min_points=5):
    out = np.full_like(t_grid, np.nan, dtype=float)
    finite = np.isfinite(y)

    for j, tj in enumerate(t_grid):
        mask = (
            finite
            & (t_grid >= tj - physical_window)
            & (t_grid <= tj)
        )

        if np.count_nonzero(mask) < min_points:
            continue

        x = t_grid[mask]
        z = y[mask]

        xc = x - np.mean(x)
        denom = np.dot(xc, xc)

        if denom > 0.0:
            out[j] = (
                np.dot(xc, z - np.mean(z))
                / denom
            )

    return out


def logmeanexp_columns(log_values):
    """
    Column-wise log(mean(exp(log_values))).
    """
    out = np.full(
        log_values.shape[1],
        np.nan,
        dtype=float,
    )

    for j in range(log_values.shape[1]):
        values = log_values[:, j]
        values = values[np.isfinite(values)]

        if len(values) == 0:
            continue

        a = np.max(values)

        out[j] = (
            a
            + np.log(
                np.mean(np.exp(values - a))
            )
        )

    return out


def maximal_common_prefix(mask):
    """
    Keep the initial interval before the first trajectory becomes censored.
    """
    if np.all(mask):
        return mask.copy()

    first_bad = np.where(~mask)[0][0]

    out = np.zeros_like(mask, dtype=bool)
    out[:first_bad] = True

    return out


# ---------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------

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
    sigma = sigma_from_stationary_std(
        theta,
        bar_sigma,
    )

    # SMALL-THETA COMPARISON:
    # common OU correlation-time axis s = theta t.
    t_grid = s_grid / theta
    physical_window = s_window / theta

    np.random.seed(seed)

    model = Branching_OUGF(
        ou_growth_rate_parameters=(
            m,
            theta,
            sigma,
        ),
        division_rate=division_rate,
        division_kernel=division_kernel,
    )

    shape = (n_runs, len(t_grid))

    log_mass_paths = np.full(shape, np.nan)
    log_number_paths = np.full(shape, np.nan)
    log_hmass_paths = np.full(shape, np.nan)
    h_rate_paths = np.full(shape, np.nan)

    mass_individual_slopes = np.full(shape, np.nan)
    number_individual_slopes = np.full(shape, np.nan)
    hmass_individual_slopes = np.full(shape, np.nan)

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

        log_mass = interpolate_snapshot_statistic(
            times,
            snapshots,
            t_grid,
            snapshot_log_mass,
        )

        log_number = interpolate_snapshot_statistic(
            times,
            snapshots,
            t_grid,
            snapshot_log_number,
        )

        log_hmass = interpolate_snapshot_statistic(
            times,
            snapshots,
            t_grid,
            lambda z: snapshot_log_hmass(z, theta),
        )

        h_rate = interpolate_snapshot_statistic(
            times,
            snapshots,
            t_grid,
            lambda z: snapshot_h_biased_rate(z, theta),
        )

        log_mass_paths[run] = log_mass
        log_number_paths[run] = log_number
        log_hmass_paths[run] = log_hmass
        h_rate_paths[run] = h_rate

        mass_individual_slopes[run] = rolling_slope(
            t_grid,
            log_mass,
            physical_window=physical_window,
        )

        number_individual_slopes[run] = rolling_slope(
            t_grid,
            log_number,
            physical_window=physical_window,
        )

        hmass_individual_slopes[run] = rolling_slope(
            t_grid,
            log_hmass,
            physical_window=physical_window,
        )

        final_times[run] = times[-1]

        if (run + 1) % 10 == 0 or run + 1 == n_runs:
            print(
                f"    theta={theta:g} "
                f"run {run+1:3d}/{n_runs} "
                f"| median final s="
                f"{theta*np.median(final_times[:run+1]):.2f}"
            )

    # --------------------------------------------------------------
    # M_t
    # --------------------------------------------------------------

    mass_typical = np.nanmean(
        mass_individual_slopes,
        axis=0,
    )

    mass_log_mc_mean = logmeanexp_columns(
        log_mass_paths
    )

    mass_annealed = rolling_slope(
        t_grid,
        mass_log_mc_mean,
        physical_window=physical_window,
    )

    mass_exact_log_mean = exact_log_first_moment_mass(
        t_grid,
        m,
        theta,
        sigma,
        initial_mass=init_birth_size,
    )

    mass_exact = rolling_slope(
        t_grid,
        mass_exact_log_mean,
        physical_window=physical_window,
    )

    # --------------------------------------------------------------
    # N_t
    # --------------------------------------------------------------

    number_typical = np.nanmean(
        number_individual_slopes,
        axis=0,
    )

    number_log_mc_mean = logmeanexp_columns(
        log_number_paths
    )

    number_annealed = rolling_slope(
        t_grid,
        number_log_mc_mean,
        physical_window=physical_window,
    )

    # --------------------------------------------------------------
    # H_t = sum h
    # --------------------------------------------------------------

    hmass_typical = np.nanmean(
        hmass_individual_slopes,
        axis=0,
    )

    hmass_log_mc_mean = logmeanexp_columns(
        log_hmass_paths
    )

    hmass_annealed = rolling_slope(
        t_grid,
        hmass_log_mc_mean,
        physical_window=physical_window,
    )

    # Exact h-mass slope is lambda_theta at every time.
    lam = lambda_theta(
        m,
        theta,
        bar_sigma,
    )
    hmass_exact = np.full_like(
        t_grid,
        lam,
        dtype=float,
    )

    # --------------------------------------------------------------
    # h-biased rate R_h
    # --------------------------------------------------------------

    h_rate_mean = np.nanmean(
        h_rate_paths,
        axis=0,
    )

    h_rate_median = np.nanmedian(
        h_rate_paths,
        axis=0,
    )

    h_rate_q10 = np.nanquantile(
        h_rate_paths,
        0.10,
        axis=0,
    )

    h_rate_q90 = np.nanquantile(
        h_rate_paths,
        0.90,
        axis=0,
    )

    # --------------------------------------------------------------
    # Common censoring mask
    # --------------------------------------------------------------

    all_present = (
        np.all(np.isfinite(log_mass_paths), axis=0)
        & np.all(np.isfinite(log_number_paths), axis=0)
        & np.all(np.isfinite(log_hmass_paths), axis=0)
        & np.all(np.isfinite(h_rate_paths), axis=0)
    )

    safe = maximal_common_prefix(
        all_present
    )

    return {
        "theta": theta,
        "sigma": sigma,
        "lambda": lam,
        "spine_rate": spine_stationary_mean(
            m,
            theta,
            bar_sigma,
        ),
        "s_grid": s_grid,
        "t_grid": t_grid,
        "safe": safe,

        "mass_typical": mass_typical,
        "mass_annealed": mass_annealed,
        "mass_exact": mass_exact,

        "number_typical": number_typical,
        "number_annealed": number_annealed,

        "hmass_typical": hmass_typical,
        "hmass_annealed": hmass_annealed,
        "hmass_exact": hmass_exact,

        "h_rate_mean": h_rate_mean,
        "h_rate_median": h_rate_median,
        "h_rate_q10": h_rate_q10,
        "h_rate_q90": h_rate_q90,
    }


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------

def finite_safe_mask(result, *keys):
    mask = result["safe"].copy()

    for key in keys:
        mask &= np.isfinite(
            result[key]
        )

    return mask


def plot_growth_family(
    results,
    thetas,
    typical_key,
    annealed_key,
    exact_key,
    ylabel,
    title,
    filename,
    exact_label="exact",
):
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13, 8),
        dpi=180,
        sharex=True,
    )
    axes = axes.ravel()

    for ax, theta in zip(axes, thetas):
        r = results[theta]

        keys = [
            typical_key,
            annealed_key,
        ]
        if exact_key is not None:
            keys.append(exact_key)

        safe = finite_safe_mask(
            r,
            *keys,
        )

        ax.axhline(
            r["lambda"],
            linestyle="--",
            linewidth=1.2,
            label=rf"$\lambda_\theta={r['lambda']:.3f}$",
        )

        ax.plot(
            r["s_grid"][safe],
            r[typical_key][safe],
            label="typical",
        )

        ax.plot(
            r["s_grid"][safe],
            r[annealed_key][safe],
            label="annealed",
        )

        if exact_key is not None:
            ax.plot(
                r["s_grid"][safe],
                r[exact_key][safe],
                linestyle=":",
                linewidth=2.0,
                label=exact_label,
            )

        ax.set_title(
            rf"$\theta={theta:g}$"
        )
        ax.set_xlabel(
            r"rescaled time $s=\theta t$"
        )
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)

    axes[0].legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()

    output = Path(__file__).with_name(
        filename
    )
    fig.savefig(
        output,
        bbox_inches="tight",
    )

    return output


def plot_gap_family(
    results,
    thetas,
    typical_key,
    annealed_key,
    m,
    title,
    filename,
):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.5, 8),
        dpi=180,
        sharex=True,
    )

    for theta in thetas:
        r = results[theta]

        gap = (
            r[annealed_key]
            - r[typical_key]
        )

        safe = (
            r["safe"]
            & np.isfinite(gap)
        )

        axes[0].plot(
            r["s_grid"][safe],
            gap[safe],
            linewidth=2,
            label=rf"$\theta={theta:g}$",
        )

        correction = r["lambda"] - m

        axes[1].plot(
            r["s_grid"][safe],
            gap[safe] / correction,
            linewidth=2,
            label=rf"$\theta={theta:g}$",
        )

    axes[0].axhline(
        0.0,
        linestyle="--",
        linewidth=1,
    )
    axes[0].set_ylabel(
        "annealed - typical slope"
    )
    axes[0].set_title(title)
    axes[0].grid(alpha=0.2)
    axes[0].legend(ncol=2)

    axes[1].axhline(
        0.0,
        linestyle="--",
        linewidth=1,
    )
    axes[1].set_xlabel(
        r"rescaled time $s=\theta t$"
    )
    axes[1].set_ylabel(
        r"gap / $(\lambda_\theta-m)$"
    )
    axes[1].grid(alpha=0.2)

    fig.tight_layout()

    output = Path(__file__).with_name(
        filename
    )
    fig.savefig(
        output,
        bbox_inches="tight",
    )

    return output


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-runs",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--s-max",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--s-window",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=300_000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
    )
    parser.add_argument(
        "--bar-sigma-ratio",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--thetas",
        type=float,
        nargs="+",
        default=[
            1.00,
            0.75,
            0.50,
            0.35,
            0.25,
            0.15,
        ],
    )

    args = parser.parse_args()

    m = 0.5

    bar_sigma = (
        args.bar_sigma_ratio
        * m
    )

    thetas = args.thetas

    ds = 0.05
    s_grid = np.arange(
        0.0,
        args.s_max + 0.5 * ds,
        ds,
    )

    dtmin = 1e-6
    dtmax = 2e-2
    dtobs = 5e-2
    eps = 0.05

    print(
        "\nSmall-theta OU growth-fragmentation diagnostics"
    )
    print(
        "------------------------------------------------"
    )
    print(
        f"m={m}, "
        f"bar_sigma={bar_sigma}, "
        f"bar_sigma/m={bar_sigma/m:.3f}"
    )
    print(
        "time axis: s = theta t"
    )
    print()

    results = {}

    for j, theta in enumerate(thetas):
        sigma = sigma_from_stationary_std(
            theta,
            bar_sigma,
        )

        lam = lambda_theta(
            m,
            theta,
            bar_sigma,
        )

        spine_rate = spine_stationary_mean(
            m,
            theta,
            bar_sigma,
        )

        print(
            f"theta={theta:6.3f} | "
            f"sigma={sigma:.5f} | "
            f"lambda={lam:.5f} | "
            f"spine mean rate={spine_rate:.5f} | "
            f"Tmax={args.s_max/theta:.2f}"
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

    outputs = []

    outputs.append(
        plot_growth_family(
            results,
            thetas,
            typical_key="mass_typical",
            annealed_key="mass_annealed",
            exact_key="mass_exact",
            ylabel="local slope of log mass",
            title=(
                "Total mass growth in OU correlation-time units\n"
                rf"$\bar\sigma/m={bar_sigma/m:.3f}$, "
                rf"$\Delta s={args.s_window:g}$"
            ),
            filename="ougf_small_theta_mass_growth.png",
            exact_label="exact first moment",
        )
    )

    outputs.append(
        plot_gap_family(
            results,
            thetas,
            typical_key="mass_typical",
            annealed_key="mass_annealed",
            m=m,
            title="Total-mass annealed/typical gap",
            filename="ougf_small_theta_mass_gap.png",
        )
    )

    outputs.append(
        plot_growth_family(
            results,
            thetas,
            typical_key="number_typical",
            annealed_key="number_annealed",
            exact_key=None,
            ylabel="local slope of log cell number",
            title=(
                "Cell-number growth in OU correlation-time units\n"
                rf"$\bar\sigma/m={bar_sigma/m:.3f}$, "
                rf"$\Delta s={args.s_window:g}$"
            ),
            filename="ougf_small_theta_number_growth.png",
        )
    )

    outputs.append(
        plot_gap_family(
            results,
            thetas,
            typical_key="number_typical",
            annealed_key="number_annealed",
            m=m,
            title="Cell-number annealed/typical gap",
            filename="ougf_small_theta_number_gap.png",
        )
    )

    outputs.append(
        plot_growth_family(
            results,
            thetas,
            typical_key="hmass_typical",
            annealed_key="hmass_annealed",
            exact_key="hmass_exact",
            ylabel=r"local slope of $\log H_t$",
            title=(
                r"$h$-mass growth, "
                r"$H_t=\sum X_u e^{R_u/\theta}$"
                "\n"
                rf"$\bar\sigma/m={bar_sigma/m:.3f}$, "
                rf"$\Delta s={args.s_window:g}$"
            ),
            filename="ougf_small_theta_hmass_growth.png",
            exact_label=r"exact: $\lambda_\theta$",
        )
    )

    outputs.append(
        plot_gap_family(
            results,
            thetas,
            typical_key="hmass_typical",
            annealed_key="hmass_annealed",
            m=m,
            title=r"$h$-mass annealed/typical gap",
            filename="ougf_small_theta_hmass_gap.png",
        )
    )

    # h-biased OU rate inside each population.
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13, 8),
        dpi=180,
        sharex=True,
    )
    axes = axes.ravel()

    for ax, theta in zip(axes, thetas):
        r = results[theta]

        safe = finite_safe_mask(
            r,
            "h_rate_mean",
            "h_rate_median",
            "h_rate_q10",
            "h_rate_q90",
        )

        s = r["s_grid"][safe]

        ax.fill_between(
            s,
            r["h_rate_q10"][safe],
            r["h_rate_q90"][safe],
            alpha=0.2,
            label="10%-90%",
        )

        ax.plot(
            s,
            r["h_rate_mean"][safe],
            linewidth=2,
            label="mean",
        )

        ax.plot(
            s,
            r["h_rate_median"][safe],
            linestyle="--",
            linewidth=1.5,
            label="median",
        )

        ax.axhline(
            m,
            linestyle=":",
            linewidth=1.2,
            label=r"ordinary OU mean $m$",
        )

        ax.axhline(
            r["spine_rate"],
            linestyle="-.",
            linewidth=1.2,
            label=r"spine stationary mean",
        )

        ax.set_title(
            rf"$\theta={theta:g}$"
        )
        ax.set_xlabel(
            r"rescaled time $s=\theta t$"
        )
        ax.set_ylabel(
            r"$\sum h_uR_u/\sum h_u$"
        )
        ax.grid(alpha=0.2)

    axes[0].legend(fontsize=7)

    fig.suptitle(
        r"Within-population $h$-biased OU rate"
        "\n"
        rf"$h(x,r)=xe^{{r/\theta}}$, "
        rf"$\bar\sigma/m={bar_sigma/m:.3f}$"
    )

    fig.tight_layout()

    hrate_output = Path(__file__).with_name(
        "ougf_small_theta_h_biased_rate.png"
    )

    fig.savefig(
        hrate_output,
        bbox_inches="tight",
    )

    outputs.append(
        hrate_output
    )

    print(
        "\nLast common uncensored point"
    )
    print(
        "----------------------------"
    )

    for theta in thetas:
        r = results[theta]

        valid = np.where(
            r["safe"]
            & np.isfinite(r["mass_typical"])
            & np.isfinite(r["number_typical"])
            & np.isfinite(r["hmass_typical"])
            & np.isfinite(r["h_rate_mean"])
        )[0]

        if len(valid) == 0:
            print(
                f"theta={theta:g}: insufficient data"
            )
            continue

        j = valid[-1]

        print(
            f"theta={theta:6.3f} | "
            f"s={r['s_grid'][j]:5.2f} | "
            f"t={r['t_grid'][j]:7.2f} | "
            f"lambda={r['lambda']:.4f} | "
            f"M typ={r['mass_typical'][j]:.4f} | "
            f"N typ={r['number_typical'][j]:.4f} | "
            f"H typ={r['hmass_typical'][j]:.4f} | "
            f"R_h mean={r['h_rate_mean'][j]:.4f}"
        )

        if (
            r["s_grid"][j]
            < args.s_max - ds
        ):
            print(
                "    WARNING: first-run censoring "
                "occurred before s_max."
            )

    print(
        "\nSaved figures:"
    )

    for output in outputs:
        print(
            f"  {output}"
        )

    plt.show()


if __name__ == "__main__":
    main()
