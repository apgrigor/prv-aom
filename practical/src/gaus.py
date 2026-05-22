import numpy as np
import pandas as pd
from typing import List
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


class Conf:
    Data = Path("data").absolute()
    Out = Path("out").absolute()
    Img = Path("img").absolute()

    ProfileXName = "Distance_(µm)"
    ProfileYName = "Gray_Value"
    ProfileXScale = 1000
    ProfileStep = ProfileXScale * 0.02
    ProfileBuffer = ProfileXScale * 0.002
    ProfileNumSteps = 4
    ProfileGausN = 400

    SubplotsKW = dict(
        figsize = (6, 5),
        layout = "tight"
    )
    GridKW = dict(alpha = 0.5)
    BarKW = dict(
        edgecolor = "white"
    )
    PlotKW = dict(
        linewidth = 3,
    )


def gaus(x, x0, A, sigma):
    return A * np.exp(-((x - x0) / sigma) ** 2)


def fit_gaussians(filenames: List[Path]) -> List[int]:
    result = []
    for filename in filenames:
        df = pd.read_csv(Conf.Out.joinpath(filename))

        xs = df[Conf.ProfileXName].to_numpy() * Conf.ProfileXScale
        ys = df[Conf.ProfileYName].to_numpy()
        gaus_xs = np.linspace(np.min(xs), np.max(xs), Conf.ProfileGausN)
        gaus_ys = np.zeros(Conf.ProfileGausN)

        fig, ax = plt.subplots(**Conf.SubplotsKW)
        ax.grid(**Conf.GridKW)
        ax.set_xlabel("Distance [nm]")
        ax.set_ylabel("Intensity [arbitrary units]")

        ax.bar(
            xs, ys,
            width = 0.9 * (xs[1] - xs[0]),
            color = "gray",
            alpha = 0.7,
            label = "experiment",
            **Conf.BarKW
        )

        partial = []
        for n in range(Conf.ProfileNumSteps):
            lower = n * Conf.ProfileStep - Conf.ProfileBuffer
            upper = (n + 1) * Conf.ProfileStep + Conf.ProfileBuffer
            mask = (xs >= lower) & (xs <= upper)
            mask_xs = xs[mask]
            mask_ys = ys[mask]
            vals, _ = curve_fit(gaus, mask_xs, mask_ys, p0 = [
                0.5 * (lower + upper),
                80,
                0.8 * Conf.ProfileStep,
            ])
            partial.append(vals[0])
            gaus_ys += gaus(gaus_xs, *list(vals))

        partial = np.array(partial)
        result.extend(list(partial[1:] - partial[:-1]))

        ax.plot(
            gaus_xs, gaus_ys,
            color = "dodgerblue",
            label = "fit"
        )

        ax.legend()
        fig.savefig(Conf.Img.joinpath(filename.with_suffix(".png")))

    return np.array(result)


if __name__ == "__main__":
    profiles_x = [Path(f"profile_x{i}.csv") for i in range(4)]
    profiles_y = [Path(f"profile_y{i}.csv") for i in range(4)]
    dxs0 = fit_gaussians(profiles_x)
    dys0 = fit_gaussians(profiles_y)

    # print(dxs0)
    # print(dys0)
    print(f"dx: {np.mean(dxs0):.2f}, {np.std(dxs0):.2f}")
    print(f"dy: {np.mean(dys0):.2f}, {np.std(dys0):.2f}")
