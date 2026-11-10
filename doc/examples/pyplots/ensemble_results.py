from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from pylab import get_cmap

forecast_names = ["forecast1", "forecast2"]
dir_template = "../../../examples/ensemble/reference_output/{}/timeseries_export.csv"

# Import Data
forecasts = {}
for forecast in forecast_names:
    data_path = dir_template.format(forecast)
    forecasts[forecast] = np.genfromtxt(
        data_path, delimiter=",", encoding=None, dtype=None, names=True, case_sensitive="lower"
    )

# Get times as datetime objects
times = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in forecasts[forecast_names[0]]["time"]]

n_subplots = 3
fig, axarr = plt.subplots(n_subplots, sharex=True, figsize=(8, 4 * n_subplots))
axarr[0].set_title("Water Volume and Discharge")
cmaps = ["Blues", "Greens"]
shades = [0.5, 0.8]

# Upper Subplot
axarr[0].set_ylabel("Water Volume in Storage [m³]")
axarr[0].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

# Lower Subplots
axarr[1].set_ylabel("Flow Rate [m³/s]")
axarr[2].set_ylabel("Flow Rate [m³/s]")

# Plot Ensemble Members
for idx, forecast in enumerate(forecast_names):
    # Upper Subplot
    results = forecasts[forecast]
    if idx == 0:
        axarr[0].plot(times, results["v_max"], label="Max", linewidth=2, color="r", linestyle="--")
        axarr[0].plot(times, results["v_min"], label="Min", linewidth=2, color="g", linestyle="--")
    axarr[0].plot(
        times,
        results["v_storage"],
        label=forecast + ":Volume",
        linewidth=2,
        color=get_cmap(cmaps[idx])(shades[1]),
    )

    # middle subplot
    # put dots to see at which times the decision variables are defined:
    axarr[1].scatter(times, results["q_in"], linewidth=1, color=get_cmap(cmaps[idx])(shades[0]))
    axarr[1].scatter(
        times, results["q_release"], linewidth=1, color=get_cmap(cmaps[idx])(shades[1])
    )

    # add horizontal lines to the left of these dots, to indicate that the value is attained over
    # an entire timestep:
    axarr[1].step(
        times,
        results["q_in"],
        linewidth=2,
        where="pre",
        label=f"{forecast}:Inflow",
        color=get_cmap(cmaps[idx])(shades[0]),
    )
    axarr[1].step(
        times,
        results["q_release"],
        linewidth=1,
        where="pre",
        label=f"{forecast}:Release",
        color=get_cmap(cmaps[idx])(shades[1]),
    )

    # Lower Subplot
    axarr[2].plot(
        times,
        results["q_in"],
        label=f"{forecast}:Inflow",
        linewidth=2,
        color=get_cmap(cmaps[idx])(shades[0]),
    )
    axarr[2].plot(
        times,
        results["q_release"],
        label=f"{forecast}:Release",
        linewidth=2,
        color=get_cmap(cmaps[idx])(shades[1]),
    )


# Format bottom axis labels
axarr[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

# Shrink margins
fig.tight_layout()

# Shrink each axis and put a legend to the right of the axis
for i in range(len(axarr)):
    box = axarr[i].get_position()
    axarr[i].set_position([box.x0, box.y0, box.width * 0.75, box.height])
    axarr[i].legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

plt.autoscale(enable=True, axis="x", tight=True)

# Output Plot
plt.show()
