"""Sec. 5.2 (Figs. 7 and 8): Hill spherical vortex

    python tests/hill_vortex_3d.py <integrator> <scheme>

<integrator> is "euler" (Fig. 7) or "midpoint" (Fig. 8)
<scheme> is "asf", "rebholz" or "standard"
"""
import sys

from enstrophy_stable_ns import hill_vortex_3d

integrator, scheme = sys.argv[1], sys.argv[2]
theta = {"euler": 1.0, "midpoint": 0.5}[integrator]

hill_vortex_3d(
    disc_params={"scheme": scheme, "theta": theta},
    save_params={"output_dir": f"output/hill_vortex_3d/{integrator}/{scheme}"},
)
