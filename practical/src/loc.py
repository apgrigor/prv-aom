import numpy as np
import pandas as pd
from typing import Any
import matplotlib.pyplot as plt


def signal_noise_ratio(s, a, b, ns):
    s2 = s ** 2
    s4 = s ** 4
    a2 = a ** 2
    b2 = b ** 2
    ns2 = ns ** 2
    return np.sqrt(
        (s2 + a2 / 12) / ns + #
        (8 * np.pi * s4 * b2) / (a2 * ns2) #
    )

# def fit(xs, ys) -> tuple[float,float,Any]:
#     (a, b), cov = np.polyfit(xs, ys, 1, full = False, cov = True)
# 
#     return a, b, cov


# def fit(xs, ys, ns):
#     (a, b, c), cov = np.polyfit(xs, ys, 2, full = False, cov = True)
# 
#     return a * (ns ** 2) + b * ns + c

def fit(xs, ys, ns):
    (a, b), cov = np.polyfit(xs, ys, 1, full = False, cov = True)

    return a * ns + b


def background_0(df: pd.DataFrame):
    xname = "intensity [photon]"
    yname = "bkgstd [photon]"

    fig, ax = plt.subplots(figsize = (6, 5), layout = "tight")
    ax.grid(alpha = 0.5)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)

    xs = df[xname].to_numpy()
    ys = df[yname].to_numpy()
    ns = np.linspace(np.min(xs), np.max(xs), 200)
    fs = fit(xs, ys, ns)
    # a, b, cov = fit(xs, ys)
    ax.scatter(
        xs, ys,
        s = 2,
        color = "darkorange",
        label = "experiment"
    )
    ax.plot(
        ns, fs,
        linewidth = 1.5,
        color = "darkblue",
        label = "linear fit"
    )
    # ax.axline(
    #     xy1 = [0, b],
    #     slope = a,
    #     linewidth = 2,
    #     color = "darkorange",
    #     label = f"fit: $b={a*1e4:0.2f}\\cdot 10^{{-4}}N + {b:0.1f}$"
    # )

    ax.legend()
    fig.savefig("img/bkg_0.png")


def localization_0(df: pd.DataFrame):
    xname = "intensity [photon]"
    yname = "uncertainty [nm]"

    fig, ax = plt.subplots(figsize = (6, 5), layout = "tight")
    ax.grid(alpha = 0.5)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_xscale("log")
    ax.set_yscale("log")

    xs = df[xname].to_numpy()
    ys = df[yname].to_numpy()
    ax.plot(xs, ys, label = "experimental")

    s = df["sigma [nm]"].to_numpy().mean()
    ns = np.linspace(np.min(xs), np.max(xs), 200)

    ax.plot(ns, s / np.sqrt(ns), label = "shot noise")

    a = 160
    bs = df["bkgstd [photon]"].to_numpy()
    b = bs.mean()
    # bkg_slope, bkg_intercept, _ = fit(xs, bs)
    # theoretical_bs = bkg_slope * ns + bkg_intercept
    theoretical_bs = fit(xs, bs, ns)

    ax.plot(
        ns,
        signal_noise_ratio(s, a, b, ns),
        label = "theoretical"
    )
    ax.plot(
        ns,
        signal_noise_ratio(s, a, theoretical_bs, ns),
        label = "theoretical+"
    )

    ax.legend()
    fig.savefig("img/loc_0.png")


if __name__ == "__main__":
    df1 = pd.read_csv("out/origami_1.csv").sort_values(
        by = "intensity [photon]"
    )

    background_0(df1)
    localization_0(df1)
