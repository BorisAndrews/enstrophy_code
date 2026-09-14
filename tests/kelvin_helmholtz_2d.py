"""Sec. 5.1 (Figs. 5 and 6): Kelvin-Helmholtz instability

    python tests/kelvin_helmholtz_2d.py <BCs> <scheme>

<BCs> is "slip" or "dirichlet"
<scheme> is "asf" or "standard"
"""
import sys

from enstrophy_stable_ns import kelvin_helmholtz_2d

BCs, scheme = sys.argv[1], sys.argv[2]

kelvin_helmholtz_2d(prob_params={"BCs": BCs}, disc_params={"scheme": scheme})
