import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def localization_0():
    xname = "intensity [photon]"
    yname = "uncertainty [nm]"
    df = pd.read_csv("out/origami_0.csv").sort_values(
        by = xname
    )
    # print(df)
    print(df.columns)

    fig, ax = plt.subplots(figsize = (6, 5), layout = "tight")
    ax.grid(alpha = 0.5)
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_yscale("log")
    ax.set_xscale("log")

    xs = df[xname].to_numpy()
    ys = df[yname].to_numpy()
    ax.plot(xs, ys, label = "experimental")

    s = df["sigma [nm]"].to_numpy().mean()
    ns = np.linspace(np.min(xs), np.max(xs), 200)

    ax.plot(ns, s / np.sqrt(ns), label = "shot noise")

    a = 160
    b = df["bkgstd [photon]"].to_numpy().mean()

    ax.plot(
        ns, np.sqrt(
            (s ** 2 + a ** 2 / 12) / ns +
            (8 * np.pi * (s ** 4) * (b ** 2)) / (
                (a ** 2) * (ns ** 2)
            )
        ),
        label = "theoretical"
    )

    ax.legend()
    fig.savefig("img/loc_0.png")


if __name__ == "__main__":
    localization_0()
