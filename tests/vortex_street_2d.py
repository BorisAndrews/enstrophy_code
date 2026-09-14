"""Sec. 5.3.1 (Figs. 10 and 11): 2D flow past a cylinder

    python tests/vortex_street_2d.py <scheme>

<scheme> is "asf", "meevc" or "standard"
"""
import sys

from enstrophy_stable_ns import vortex_street_2d

vortex_street_2d(disc_params={"scheme": sys.argv[1]})
