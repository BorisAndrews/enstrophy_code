"""Sec. 5.3.2: 3D flow past a sphere

    python tests/vortex_street_3d.py sweep <ratio>        Fig. 12
    python tests/vortex_street_3d.py no_delta             the delta comparison
    python tests/vortex_street_3d.py bdf2 <scheme>        Fig. 13
    python tests/vortex_street_3d.py midpoint <scheme>    Figs. 14 and 15

<ratio> is sigma_bar / sigma, one of 0, 0.03, 0.1, 0.3, 0.5, 1
<scheme> is "asf", "meevc" or "standard"
"no_delta" repeats the BDF2 run of the proposed scheme with the delta stabilisation removed, restoring the Lagrange multipliers
Append "restart" to resume a run from its checkpoint
"""
import sys

from enstrophy_stable_ns import vortex_street_3d

experiment = sys.argv[1]
restart = sys.argv[-1] == "restart"

if experiment == "bdf2":
    name = sys.argv[2]
    disc = {"scheme": name}
elif experiment == "midpoint":
    name = sys.argv[2]
    disc = {"scheme": name, "stages": 1, "theta": 0.5}
elif experiment == "sweep":
    ratio = float(sys.argv[2])
    name = f"sigma_bar_{ratio:g}"
    disc = {"sigma_vort_scale": ratio, "T": 10.0}
elif experiment == "no_delta":
    name = "asf"
    disc = {"epsilon": None}
else:
    raise ValueError(f"Unknown experiment {experiment!r}")

vortex_street_3d(
    disc_params=disc,
    save_params={"output_dir": f"output/vortex_street_3d/{experiment}/{name}",
                 "restart": restart},
)
