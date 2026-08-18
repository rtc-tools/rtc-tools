import datetime as dt
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 2:
    png_name = "comparison_routing.png"
else:
    png_name = "comparison_routing_" + str(sys.argv[1]) + ".png"

# Map output_dir
output_dir = Path("../../../examples/channel_routings/output/").resolve()

# Import Data
rtc_tools_linear = np.genfromtxt(
    output_dir / "timeseries_export_linear.csv",
    delimiter=",",
    encoding=None,
    dtype=None,
    names=True,
    case_sensitive="lower",
)
rtc_tools_idz = np.genfromtxt(
    output_dir / "timeseries_export_IDZ.csv",
    delimiter=",",
    encoding=None,
    dtype=None,
    names=True,
    case_sensitive="lower",
)
rtc_tools_id = np.genfromtxt(
    output_dir / "timeseries_export_ID.csv",
    delimiter=",",
    encoding=None,
    dtype=None,
    names=True,
    case_sensitive="lower",
)
rtc_tools_homotopy = np.genfromtxt(
    output_dir / "timeseries_export_saint_venant_upwind.csv",
    delimiter=",",
    encoding=None,
    dtype=None,
    names=True,
    case_sensitive="lower",
)
rtc_tools_lin_sv = np.genfromtxt(
    output_dir / "timeseries_export_SV.csv",
    delimiter=",",
    encoding=None,
    dtype=None,
    names=True,
    case_sensitive="lower",
)

# Get times as datetime objects
times = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in rtc_tools_linear["time"]]

# Generate Plot
n_subplots = 2
fig, axarr = plt.subplots(n_subplots, sharex=True, figsize=(8, 4 * n_subplots))


start_time = dt.datetime(2013, 1, 1, 20, 0, 0)
end_time = dt.datetime(2013, 1, 2, 23, 0, 0)


axarr[0].set_title("Water Levels and Flow Rates")

# Upper subplot
axarr[0].set_ylabel("Flow Rate [m³/s]")
axarr[0].plot(
    times,
    rtc_tools_linear["channel_q_up"],
    label="Upstream input",
    color="black",
)
axarr[0].plot(
    times,
    rtc_tools_linear["channel_q_dn"],
    label="Downstream\n(RTC-Tools Inertial Wave)",
    linestyle="--",
    color="red",
)
axarr[0].plot(
    times,
    rtc_tools_idz["channel_q_dn"],
    label="Downstream\n(RTC-Tools Inertial Wave semi-impl.)",
    linestyle="--",
    color="pink",
)
axarr[0].plot(
    times,
    rtc_tools_homotopy["channel_q_dn"],
    label="Downstream\n(RTC-Tools Saint Venant central diff.)",
    linestyle="--",
    color="darkorange",
)
axarr[0].plot(
    times,
    rtc_tools_lin_sv["channel_q_dn"],
    label="Downstream\n(RTC-Tools Saint Venant upwind)",
    linestyle="--",
    color="purple",
)

axarr[0].plot(
    times,
    rtc_tools_id["channel_q_dn"],
    label="Downstream\n(RTC-Tools ID)",
    linestyle="--",
    color="green",
)
"""
axarr[0].plot(
    times,
    hec_ras_record["channel_q_dn"],
    label="Downstream\n(HEC-RAS)",
    linestyle="--",
    color="darkgreen",
)
"""

# axarr[0].set_xlim(left=start_time)
# axarr[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

# Lower subplot
axarr[1].set_ylabel("Water Level [m]")
axarr[1].plot(
    times,
    rtc_tools_linear["channel_h_up"],
    label="Upstream\n(RTC-Tools Linear)",
    linestyle="--",  # marker='o',
    color="red",
)
axarr[1].plot(
    times,
    rtc_tools_idz["channel_h_up"],
    label="Upstream\n(RTC-Tools IDZ)",
    linestyle="--",
    color="pink",
)
axarr[1].plot(
    times,
    rtc_tools_homotopy["channel_h_up"],
    label="Upstream\n(RTC-Tools Saint Venant homotopy)",
    linestyle="--",
    color="darkorange",
)
axarr[1].plot(
    times,
    rtc_tools_lin_sv["channel_h_up"],
    label="Upstream\n(RTC-Tools Linearized SV)",
    linestyle="--",
    color="purple",
)

axarr[1].plot(
    times,
    rtc_tools_id["channel_h_up"],
    label="Upstream\n(RTC-Tools ID)",
    linestyle="--",
    color="green",
)
"""
axarr[1].plot(
    times,
    hec_ras_record["channel_h_up"],
    label="Upstream\n(HEC-RAS)",
    linestyle="--",
    color="darkgreen",
)
"""
"""
axarr[1].plot(
    times,
    rtc_tools_linear["channel_h_dn"],
    label="Downstream",
    color="xkcd:dark sky blue",
)
"""
# axarr[1].set_xlim(left=start_time)
# axarr[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

# Format bottom axis label
axarr[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
axarr[-1].set_xlim(left=start_time)
axarr[-1].set_xlim(right=end_time)

# axarr[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

# Shrink margins
fig.tight_layout()

# Shrink each axis by 20% and put a legend to the right of the axis
for i in range(n_subplots):
    box = axarr[i].get_position()
    axarr[i].set_position([box.x0, box.y0, box.width * 0.65, box.height])
    axarr[i].legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, prop={"size": 8})

plt.autoscale(enable=True, axis="x", tight=True)


# Output Plot
plt.savefig(f"../output/figures/{png_name}")


# Second figure
n_subplots = 1
fig, axarr = plt.subplots(n_subplots, sharex=True, figsize=(8, 4 * n_subplots))


start_time = dt.datetime(2013, 1, 1, 20, 0, 0)
end_time = dt.datetime(2013, 1, 2, 23, 0, 0)


axarr.set_title("Water Levels and Flow Rates")

# Upper subplot
"""
axarr.set_ylabel("Flow Rate [m³/s]")
axarr.plot(
    times,
    rtc_tools_linear["channel_q_up"],
    label="Upstream input",
    color="black",
)
"""
axarr.plot(
    times,
    rtc_tools_homotopy["channel_h_up"],
    label="Upstream\n(RTC-Tools Saint Venant)",
    linestyle="-",
    color="black",
)


# Format bottom axis label
axarr.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
axarr.set_xlim(left=start_time)
axarr.set_xlim(right=end_time)

# axarr.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

# Shrink margins
fig.tight_layout()

"""
# Shrink each axis by 20% and put a legend to the right of the axis
for i in range(n_subplots):
    box = axarr.get_position()
    axarr.set_position([box.x0, box.y0, box.width * 0.65, box.height])
    axarr.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, prop={"size": 8})
"""
plt.autoscale(enable=True, axis="x", tight=True)


# Output Plot
plt.savefig(f"../output/figures/SV{png_name}")


# Third figure
n_subplots = 1
fig, axarr = plt.subplots(n_subplots, sharex=True, figsize=(8, 4 * n_subplots))


start_time = dt.datetime(2013, 1, 1, 15, 0, 0)
end_time = dt.datetime(2013, 1, 2, 23, 0, 0)


axarr.set_title("Downstream discharge")

# Upper subplot
"""
axarr.set_ylabel("Flow Rate [m³/s]")
axarr.plot(
    times,
    rtc_tools_linear["channel_q_up"],
    label="Upstream input",
    color="black",
)
"""
axarr.plot(
    times,
    rtc_tools_homotopy["channel_q_dn"],
    label="Upstream\n(RTC-Tools Saint Venant)",
    linestyle="-",
    color="black",
)

axarr.plot(
    times,
    rtc_tools_homotopy["channel_q_up"],
    label="Upstream\n(RTC-Tools Saint Venant)",
    linestyle="--",
    color="black",
)

# Format bottom axis label
axarr.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
axarr.set_xlim(left=start_time)
axarr.set_xlim(right=end_time)

# axarr.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

# Shrink margins
fig.tight_layout()


# plt.autoscale(enable=True, axis="x", tight=True)


# Output Plot
plt.savefig(f"../output/figures/SVFlow{png_name}")
