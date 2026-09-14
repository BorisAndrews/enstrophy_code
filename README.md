# enstrophy_stable_ns

Code for the numerical results in the manuscript

> Boris D. Andrews, Matin Shams and Patrick E. Farrell,
> *Strongly enstrophy-stable integrators for the incompressible Navier–Stokes equations.*

The simulations use [Firedrake](https://www.firedrakeproject.org/).

## Requirements

The results were computed with

| package | version |
| --- | --- |
| Firedrake | 2026.4.1 |
| PETSc / petsc4py | 3.25 |
| Python | 3.12 and 3.14 |
| gmsh | 4.15.2 |
| vtk | 9.6.2 |

Other versions of Firedrake may work, but the patches below are written
against 2026.4.1.

Run everything from this directory, inside a Firedrake virtual environment:

```bash
pip install -e .
pip install vtk            # VTK output; not part of a default Firedrake install
./patches/apply_patches.sh # two Firedrake patches, see patches/README.txt
```

The patches modify Firedrake's installed source and are **not** reapplied by a
Firedrake reinstall or upgrade. `apply_patches.sh` is idempotent (`--check`
for a dry run). Without them the 3D obstacle runs of the proposed scheme fail
at setup and the 2D obstacle runs segfault. The obstacle meshes are generated
with gmsh, which must also be available.

## Reproducing the manuscript

Each script in `tests/` runs **one** simulation each call. Run each in its
own process. The schemes are:

| name | scheme |
| --- | --- |
| `asf` | the proposed enstrophy-stable scheme |
| `meevc` | the MEEVC scheme (flow past an obstacle only) |
| `rebholz` | the helicity-preserving scheme (Hill vortex only) |
| `standard` | the unstabilised discretisation |

Every run writes `data.json` (diagnostic time series), `params.json` (every
resolved parameter) and `fields.pvd` (fields for ParaView) under `output/`.

### Sec. 5.1: Kelvin–Helmholtz instability (Figs. 5 and 6)

```bash
for bcs in slip dirichlet; do for s in asf standard; do
    python tests/kelvin_helmholtz_2d.py $bcs $s
done; done
```

### Sec. 5.2: Hill spherical vortex (Figs. 7 and 8)

```bash
for t in euler midpoint; do for s in asf rebholz standard; do
    python tests/hill_vortex_3d.py $t $s
done; done
```

### Sec. 5.3.1: 2D flow past a cylinder (Figs. 10 and 11)

```bash
for s in asf meevc standard; do python tests/vortex_street_2d.py $s; done
```

### Sec. 5.3.2: 3D flow past a sphere (Figs. 12 to 15)

```bash
for r in 0 0.03 0.1 0.3 0.5 1; do python tests/vortex_street_3d.py sweep $r; done  # Fig. 12
for s in asf meevc standard; do python tests/vortex_street_3d.py bdf2 $s; done       # Fig. 13
for s in asf meevc standard; do python tests/vortex_street_3d.py midpoint $s; done   # Figs. 14, 15
python tests/vortex_street_3d.py no_delta   # the delta comparison
```

These are expensive runs: several hours each on a single process. They
checkpoint every five steps; append `restart` to any command to resume. The
MEEVC run under BDF2 fails at t = 5.20, as reported in the manuscript.

Figs. 1 to 4 and 9 are illustrations and need no code. Figure numbers refer to
the submitted manuscript.

## Running other configurations

Each driver is a function taking `prob_params`, `mesh_params`, `disc_params`,
`save_params` and `solver_params` dictionaries, documented in its docstring.
The defaults are the manuscript's settings, so the scripts in `tests/` pass only
what varies between runs; `params.json` records what a run actually resolved.
For example:

```python
from enstrophy_stable_ns import vortex_street_3d
vortex_street_3d(prob_params={"Re": 1e3}, disc_params={"scheme": "asf"})
```

## Layout

```
enstrophy_stable_ns/
    common.py               shared forms, diagnostics, solver parameters
    meshes.py               gmsh mesh generators
    harmonic.py             discrete harmonic forms
    kelvin_helmholtz_2d.py  Sec. 5.1
    hill_vortex_3d.py       Sec. 5.2
    vortex_street_2d.py     Sec. 5.3.1
    vortex_street_3d.py     Sec. 5.3.2
tests/                      one script per manuscript experiment
patches/                    the two required Firedrake patches
```

## Licence

MIT (see `LICENSE`), except for the Firedrake patches in `patches/`, which
modify Firedrake's source and are licensed under the LGPL-3.0-or-later.
