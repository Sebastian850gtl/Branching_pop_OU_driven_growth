"""
Small-theta OU growth-fragmentation diagnostics in the vanishing-variance regime.

Scaling:
    bar_sigma(theta) = bar_sigma0 * theta**alpha
with default
    m = 0.5,
    bar_sigma0 = 0.2,
    alpha = 1/4.

Then
    sigma(theta) = sqrt(2 theta) * bar_sigma(theta),

and the annealed Malthus exponent is

    lambda_theta
      = m + sigma(theta)^2 / (2 theta^2)
      = m + bar_sigma(theta)^2 / theta
      = m + bar_sigma0^2 * theta**(2 alpha - 1).

Hence, for 0 < alpha < 1/2,

    bar_sigma(theta) -> 0
while
    lambda_theta -> infinity

as theta -> 0.

Time is compared in OU correlation-time units

    s = theta t,

with a rolling window fixed in s-units.

From the SAME simulated branching populations, the script studies:

1. Total mass
       M_t = sum_u X_u(t)

2. Number of cells
       N_t = # V_t

3. h-mass / reproductive-value mass
       H_t = sum_u h(X_u(t), R_u(t)),
   where
       h(x,r) = x exp(r/theta).

4. h-biased OU rate inside each population
       R_h(t) = sum_u h_u R_u / sum_u h_u.

For M_t, N_t, and H_t we compare:

    typical(t)
      = mean over runs of rolling_slope(log observable_i(t))

and

    annealed(t)
      = rolling_slope(log(mean over runs observable_i(t))).

For total mass M_t there is an explicit exact finite-time first-moment curve.

For h-mass H_t,
    E[H_t] = H_0 exp(lambda_theta t)
exactly, so the exact h-mass annealed growth rate is lambda_theta from time 0.

For N_t there is no explicit finite-time first-moment formula included here;
lambda_theta is shown as the asymptotic Malthus reference.

Run from repository root:

    python tests/test_ougf_vanishing_variance_mass_number_h.py

Useful options:

    --n-runs 100
    --s-max 6
    --s-window 0.75
    --max-cells 300000
    --bar-sigma0 0.2
    --alpha 0.25
    --thetas 1 0.5 0.25 0.125 0.0625 0.025 0.01

Outputs:
    ougf_vv_mass_growth.png
    ougf_vv_mass_gap.png
    ougf_vv_number_growth.png
    ougf_vv_number_gap.png
    ougf_vv_hmass_growth.png
    ougf_vv_hmass_gap.png
    ougf_vv_h_biased_rate.png
    ougf_vv_scaling.png
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
    return x * np.random.rand()


# ---------------------------------------------------------------------
# Vanishing-variance theta scaling
# ---------------------------------------------------------------------

def stationary_std(theta, bar_sigma0, alpha):
    return bar_sigma0 * theta**alpha


def sigma_theta(theta, bar_sigma0, alpha):
    return np.sqrt(2.0 * theta) * stationary_std(
        theta,
        bar_sigma0,
        alpha,
    )


def lambda_theta(m, theta, bar_sigma0, alpha):
    bar_sigma = stationary_std(
        theta,
        bar_sigma0,
        alpha,
    )
    return m + bar_sigma**2 / theta


def spine_stationary_mean(m, theta, bar_sigma0, alpha):
    """
    Stationary mean of the h-transformed OU rate.

    Since sigma^2 = 2 theta bar_sigma^2,

        m_spine = m + sigma^2/theta^2
                = m + 2 bar_sigma^2/theta.
    """
    bar_sigma = stationary_std(
        theta,
        bar_sigma0,
        alpha,
    )
    return m + 2.0 * bar_sigma**2 / theta


def spine_displacement_in_std(theta, bar_sigma0, alpha):
    bar_sigma = stationary_std(
        theta,
        bar_sigma0,
        alpha,
    )
    return 2.0 * bar_sigma / theta


# ---------------------------------------------------------------------
# Exact first moment for ordinary total mass
# ---------------------------------------------------------------------

def exact_log_first_moment_mass(
    t,
    m,
    theta,
    sigma,
    initial_mass=1.0,
):
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
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return -np.inf

    a = np.max(values)

    return float(
        a + np.log(
            np.sum(np.exp(values - a))
        )
    )


def snapshot_log_mass(snapshot):
    return logsumexp(snapshot[:, 1])


def snapshot_log_number(snapshot):
    return float(np.log(len(snapshot)))


def snapshot_log_hmass(snapshot, theta):
    rates = snapshot[:, 0]
    log_sizes = snapshot[:, 1]

    log_h = (
        log_sizes
        + rates / theta
    )

    return logsumexp(log_h)


def snapshot_h_biased_rate(snapshot, theta):
    rates = snapshot[:, 0]
    log_sizes = snapshot[:, 1]

    log_h = (
        log_sizes
        + rates / theta
    )

    a = np.max(log_h)

    weights = np.exp(
        log_h - a
    )

    weights /= np.sum(weights)

    return float(
        np.sum(weights * rates)
    )


# ---------------------------------------------------------------------
# Interpolation / rolling slope
# ---------------------------------------------------------------------

def interpolate_snapshot_statistic(
    times,
    snapshots,
    t_grid,
    statistic,
):
    times = np.asarray(
        times,
        dtype=float,
    )

    values = np.asarray(
        [
            statistic(snapshot)
            for snapshot in snapshots
        ],
        dtype=float,
    )

    out = np.full_like(
        t_grid,
        np.nan,
        dtype=float,
    )

    if len(times) < 2:
        return out

    valid = (
        t_grid <= times[-1]
    )

    out[valid] = np.interp(
        t_grid[valid],
        times,
        values,
    )

    return out


def rolling_slope(
    t_grid,
    y,
    physical_window,
    min_points=5,
):
    out = np.full_like(
        t_grid,
        np.nan,
        dtype=float,
    )

    finite = np.isfinite(y)

    for j, tj in enumerate(t_grid):
        mask = (
            finite
            & (t_grid >= tj - physical_window)
            & (t_grid <= tj)
        )

        if (
            np.count_nonzero(mask)
            < min_points
        ):
            continue

        x = t_grid[mask]
        z = y[mask]

        xc = (
            x - np.mean(x)
        )

        denom = np.dot(
            xc,
            xc,
        )

        if denom > 0.0:
            out[j] = (
                np.dot(
                    xc,
                    z - np.mean(z),
                )
                / denom
            )

    return out


def logmeanexp_columns(log_values):
    out = np.full(
        log_values.shape[1],
        np.nan,
        dtype=float,
    )

    for j in range(
        log_values.shape[1]
    ):
        values = (
            log_values[:, j]
        )

        values = values[
            np.isfinite(values)
        ]

        if len(values) == 0:
            continue

        a = np.max(values)

        out[j] = (
            a
            + np.log(
                np.mean(
                    np.exp(
                        values - a
                    )
                )
            )
        )

    return out


def maximal_common_prefix(mask):
    if np.all(mask):
        return mask.copy()

    first_bad = (
        np.where(~mask)[0][0]
    )

    out = np.zeros_like(
        mask,
        dtype=bool,
    )

    out[:first_bad] = True

    return out


# ---------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------

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
    bar_sigma = stationary_std(
        theta,
        bar_sigma0,
        alpha,
    )

    sigma = sigma_theta(
        theta,
        bar_sigma0,
        alpha,
    )

    # Small-theta OU clock.
    t_grid = (
        s_grid / theta
    )

    physical_window = (
        s_window / theta
    )

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

    shape = (
        n_runs,
        len(t_grid),
    )

    log_mass_paths = np.full(
        shape,
        np.nan,
    )

    log_number_paths = np.full(
        shape,
        np.nan,
    )

    log_hmass_paths = np.full(
        shape,
        np.nan,
    )

    h_rate_paths = np.full(
        shape,
        np.nan,
    )

    mass_individual_slopes = np.full(
        shape,
        np.nan,
    )

    number_individual_slopes = np.full(
        shape,
        np.nan,
    )

    hmass_individual_slopes = np.full(
        shape,
        np.nan,
    )

    final_times = np.zeros(
        n_runs
    )

    for run in range(n_runs):

        times, snapshots = model.run(
            Tmax=t_grid[-1],
            init=(
                m,
                init_birth_size,
            ),
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
            lambda z: snapshot_log_hmass(
                z,
                theta,
            ),
        )

        h_rate = interpolate_snapshot_statistic(
            times,
            snapshots,
            t_grid,
            lambda z: snapshot_h_biased_rate(
                z,
                theta,
            ),
        )

        log_mass_paths[run] = (
            log_mass
        )

        log_number_paths[run] = (
            log_number
        )

        log_hmass_paths[run] = (
            log_hmass
        )

        h_rate_paths[run] = (
            h_rate
        )

        mass_individual_slopes[run] = (
            rolling_slope(
                t_grid,
                log_mass,
                physical_window=physical_window,
            )
        )

        number_individual_slopes[run] = (
            rolling_slope(
                t_grid,
                log_number,
                physical_window=physical_window,
            )
        )

        hmass_individual_slopes[run] = (
            rolling_slope(
                t_grid,
                log_hmass,
                physical_window=physical_window,
            )
        )

        final_times[run] = (
            times[-1]
        )

        if (
            (run + 1) % 10 == 0
            or run + 1 == n_runs
        ):
            print(
                f"    theta={theta:g} "
                f"run {run+1:3d}/{n_runs} "
                f"| median final s="
                f"{theta*np.median(final_times[:run+1]):.3f}"
            )

    # --------------------------------------------------------------
    # Total mass
    # --------------------------------------------------------------

    mass_typical = np.nanmean(
        mass_individual_slopes,
        axis=0,
    )

    mass_log_mc_mean = (
        logmeanexp_columns(
            log_mass_paths
        )
    )

    mass_annealed = rolling_slope(
        t_grid,
        mass_log_mc_mean,
        physical_window=physical_window,
    )

    exact_mass_log_mean = (
        exact_log_first_moment_mass(
            t_grid,
            m,
            theta,
            sigma,
            initial_mass=init_birth_size,
        )
    )

    mass_exact = rolling_slope(
        t_grid,
        exact_mass_log_mean,
        physical_window=physical_window,
    )

    # --------------------------------------------------------------
    # Number of cells
    # --------------------------------------------------------------

    number_typical = np.nanmean(
        number_individual_slopes,
        axis=0,
    )

    number_log_mc_mean = (
        logmeanexp_columns(
            log_number_paths
        )
    )

    number_annealed = rolling_slope(
        t_grid,
        number_log_mc_mean,
        physical_window=physical_window,
    )

    # --------------------------------------------------------------
    # h-mass
    # --------------------------------------------------------------

    hmass_typical = np.nanmean(
        hmass_individual_slopes,
        axis=0,
    )

    hmass_log_mc_mean = (
        logmeanexp_columns(
            log_hmass_paths
        )
    )

    hmass_annealed = rolling_slope(
        t_grid,
        hmass_log_mc_mean,
        physical_window=physical_window,
    )

    lam = lambda_theta(
        m,
        theta,
        bar_sigma0,
        alpha,
    )

    hmass_exact = np.full_like(
        t_grid,
        lam,
        dtype=float,
    )

    # --------------------------------------------------------------
    # h-biased OU rate
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
    # Common uncensored interval
    # --------------------------------------------------------------

    all_present = (
        np.all(
            np.isfinite(log_mass_paths),
            axis=0,
        )
        & np.all(
            np.isfinite(log_number_paths),
            axis=0,
        )
        & np.all(
            np.isfinite(log_hmass_paths),
            axis=0,
        )
        & np.all(
            np.isfinite(h_rate_paths),
            axis=0,
        )
    )

    safe = maximal_common_prefix(
        all_present
    )

    return {
        "theta": theta,
        "bar_sigma": bar_sigma,
        "sigma": sigma,
        "lambda": lam,
        "spine_rate": spine_stationary_mean(
            m,
            theta,
            bar_sigma0,
            alpha,
        ),
        "spine_std_distance": (
            spine_displacement_in_std(
                theta,
                bar_sigma0,
                alpha,
            )
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

def finite_safe_mask(
    result,
    *keys,
):
    mask = result[
        "safe"
    ].copy()

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
    ncols = 3
    nrows = int(
        np.ceil(
            len(thetas) / ncols
        )
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            13.5,
            3.9 * nrows,
        ),
        dpi=180,
        sharex=True,
    )

    axes = np.asarray(
        axes
    ).ravel()

    for ax, theta in zip(
        axes,
        thetas,
    ):
        r = results[theta]

        keys = [
            typical_key,
            annealed_key,
        ]

        if exact_key is not None:
            keys.append(
                exact_key
            )

        safe = finite_safe_mask(
            r,
            *keys,
        )

        ax.axhline(
            r["lambda"],
            linestyle="--",
            linewidth=1.2,
            label=(
                rf"$\lambda_\theta="
                rf"{r['lambda']:.3f}$"
            ),
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
            rf"$\theta={theta:g}$, "
            rf"$\bar\sigma_\theta="
            rf"{r['bar_sigma']:.3f}$"
        )

        ax.set_xlabel(
            r"rescaled time $s=\theta t$"
        )

        ax.set_ylabel(
            ylabel
        )

        ax.grid(
            alpha=0.2
        )

    for ax in axes[
        len(thetas):
    ]:
        ax.axis(
            "off"
        )

    axes[0].legend(
        fontsize=8
    )

    fig.suptitle(
        title
    )

    fig.tight_layout()

    output = Path(
        __file__
    ).with_name(
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

        correction = (
            r["lambda"] - m
        )

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

    axes[0].set_title(
        title
    )

    axes[0].grid(
        alpha=0.2
    )

    axes[0].legend(
        ncol=2
    )

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

    axes[1].grid(
        alpha=0.2
    )

    fig.tight_layout()

    output = Path(
        __file__
    ).with_name(
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
        "--bar-sigma0",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--thetas",
        type=float,
        nargs="+",
        default=[
            1.0,
            0.5,
            0.25,
            0.125,
            0.0625,
            0.025,
            0.01,
        ],
    )

    args = parser.parse_args()

    m = 0.5
    bar_sigma0 = args.bar_sigma0
    alpha = args.alpha
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
        "\nVanishing-variance small-theta diagnostics"
    )

    print(
        "------------------------------------------"
    )

    print(
        f"m={m}, "
        f"bar_sigma0={bar_sigma0}, "
        f"alpha={alpha}"
    )

    print(
        "bar_sigma(theta) = "
        "bar_sigma0 * theta**alpha"
    )

    print(
        "time axis: s = theta t"
    )

    print()

    results = {}

    for j, theta in enumerate(
        thetas
    ):
        bar_sigma = stationary_std(
            theta,
            bar_sigma0,
            alpha,
        )

        sigma = sigma_theta(
            theta,
            bar_sigma0,
            alpha,
        )

        lam = lambda_theta(
            m,
            theta,
            bar_sigma0,
            alpha,
        )

        spine_rate = spine_stationary_mean(
            m,
            theta,
            bar_sigma0,
            alpha,
        )

        spine_distance = (
            spine_displacement_in_std(
                theta,
                bar_sigma0,
                alpha,
            )
        )

        print(
            f"theta={theta:8.5f} | "
            f"bar_sigma={bar_sigma:.5f} | "
            f"sigma={sigma:.5f} | "
            f"lambda={lam:.5f} | "
            f"spine mean={spine_rate:.5f} | "
            f"spine shift/std="
            f"{spine_distance:.2f} | "
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

    outputs = []

    # --------------------------------------------------------------
    # Mass
    # --------------------------------------------------------------

    outputs.append(
        plot_growth_family(
            results,
            thetas,
            typical_key="mass_typical",
            annealed_key="mass_annealed",
            exact_key="mass_exact",
            ylabel="local slope of log mass",
            title=(
                "Total-mass growth in the vanishing-variance regime\n"
                rf"$\bar\sigma_\theta="
                rf"{bar_sigma0}\theta^{{{alpha:g}}}$, "
                rf"$\Delta s={args.s_window:g}$"
            ),
            filename="ougf_vv_mass_growth.png",
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
            title=(
                "Total-mass annealed/typical gap"
            ),
            filename="ougf_vv_mass_gap.png",
        )
    )

    # --------------------------------------------------------------
    # Number
    # --------------------------------------------------------------

    outputs.append(
        plot_growth_family(
            results,
            thetas,
            typical_key="number_typical",
            annealed_key="number_annealed",
            exact_key=None,
            ylabel="local slope of log cell number",
            title=(
                "Cell-number growth in the vanishing-variance regime\n"
                rf"$\bar\sigma_\theta="
                rf"{bar_sigma0}\theta^{{{alpha:g}}}$, "
                rf"$\Delta s={args.s_window:g}$"
            ),
            filename="ougf_vv_number_growth.png",
        )
    )

    outputs.append(
        plot_gap_family(
            results,
            thetas,
            typical_key="number_typical",
            annealed_key="number_annealed",
            m=m,
            title=(
                "Cell-number annealed/typical gap"
            ),
            filename="ougf_vv_number_gap.png",
        )
    )

    # --------------------------------------------------------------
    # h-mass
    # --------------------------------------------------------------

    outputs.append(
        plot_growth_family(
            results,
            thetas,
            typical_key="hmass_typical",
            annealed_key="hmass_annealed",
            exact_key="hmass_exact",
            ylabel=r"local slope of $\log H_t$",
            title=(
                r"$h$-mass growth in the vanishing-variance regime"
                "\n"
                rf"$H_t=\sum X_u e^{{R_u/\theta}}$, "
                rf"$\Delta s={args.s_window:g}$"
            ),
            filename="ougf_vv_hmass_growth.png",
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
            title=(
                r"$h$-mass annealed/typical gap"
            ),
            filename="ougf_vv_hmass_gap.png",
        )
    )

    # --------------------------------------------------------------
    # h-biased rate
    # --------------------------------------------------------------

    ncols = 3
    nrows = int(
        np.ceil(
            len(thetas) / ncols
        )
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            13.5,
            3.9 * nrows,
        ),
        dpi=180,
        sharex=True,
    )

    axes = np.asarray(
        axes
    ).ravel()

    for ax, theta in zip(
        axes,
        thetas,
    ):
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
            rf"$\theta={theta:g}$, "
            rf"$\bar\sigma_\theta="
            rf"{r['bar_sigma']:.3f}$"
        )

        ax.set_xlabel(
            r"rescaled time $s=\theta t$"
        )

        ax.set_ylabel(
            r"$\sum h_uR_u/\sum h_u$"
        )

        ax.grid(
            alpha=0.2
        )

    for ax in axes[
        len(thetas):
    ]:
        ax.axis(
            "off"
        )

    axes[0].legend(
        fontsize=7
    )

    fig.suptitle(
        r"Within-population $h$-biased OU rate"
        "\n"
        rf"$\bar\sigma_\theta="
        rf"{bar_sigma0}\theta^{{{alpha:g}}}$"
    )

    fig.tight_layout()

    hrate_output = Path(
        __file__
    ).with_name(
        "ougf_vv_h_biased_rate.png"
    )

    fig.savefig(
        hrate_output,
        bbox_inches="tight",
    )

    outputs.append(
        hrate_output
    )

    # --------------------------------------------------------------
    # Scaling summary
    # --------------------------------------------------------------

    theta_array = np.asarray(
        thetas,
        dtype=float,
    )

    bar_sigma_array = np.asarray(
        [
            stationary_std(
                theta,
                bar_sigma0,
                alpha,
            )
            for theta in thetas
        ]
    )

    lambda_array = np.asarray(
        [
            lambda_theta(
                m,
                theta,
                bar_sigma0,
                alpha,
            )
            for theta in thetas
        ]
    )

    spine_distance_array = np.asarray(
        [
            spine_displacement_in_std(
                theta,
                bar_sigma0,
                alpha,
            )
            for theta in thetas
        ]
    )

    fig2, axes2 = plt.subplots(
        3,
        1,
        figsize=(8.5, 9),
        dpi=180,
        sharex=True,
    )

    axes2[0].plot(
        theta_array,
        bar_sigma_array,
        marker="o",
    )

    axes2[0].set_ylabel(
        r"$\bar\sigma_\theta$"
    )

    axes2[0].set_title(
        "Ordinary stationary OU fluctuations vanish"
    )

    axes2[0].grid(
        alpha=0.2
    )

    axes2[1].plot(
        theta_array,
        lambda_array,
        marker="o",
    )

    axes2[1].set_ylabel(
        r"$\lambda_\theta$"
    )

    axes2[1].set_title(
        "Annealed Malthus exponent increases"
    )

    axes2[1].grid(
        alpha=0.2
    )

    axes2[2].plot(
        theta_array,
        spine_distance_array,
        marker="o",
    )

    axes2[2].set_xlabel(
        r"$\theta$"
    )

    axes2[2].set_ylabel(
        r"$(m_{\rm spine}-m)/"
        r"\bar\sigma_\theta$"
    )

    axes2[2].set_title(
        "Spine moves farther into the ordinary OU tail"
    )

    axes2[2].grid(
        alpha=0.2
    )

    for ax in axes2:
        ax.set_xscale(
            "log"
        )
        ax.invert_xaxis()

    fig2.tight_layout()

    scaling_output = Path(
        __file__
    ).with_name(
        "ougf_vv_scaling.png"
    )

    fig2.savefig(
        scaling_output,
        bbox_inches="tight",
    )

    outputs.append(
        scaling_output
    )

    # --------------------------------------------------------------
    # Text summary
    # --------------------------------------------------------------

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
            & np.isfinite(
                r["mass_typical"]
            )
            & np.isfinite(
                r["number_typical"]
            )
            & np.isfinite(
                r["hmass_typical"]
            )
            & np.isfinite(
                r["h_rate_mean"]
            )
        )[0]

        if len(valid) == 0:
            print(
                f"theta={theta:g}: "
                "insufficient data"
            )
            continue

        j = valid[-1]

        print(
            f"theta={theta:8.5f} | "
            f"s={r['s_grid'][j]:5.2f} | "
            f"t={r['t_grid'][j]:8.2f} | "
            f"bar_sigma={r['bar_sigma']:.4f} | "
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
