"""Shared utilities for enstrophy-stable Navier-Stokes simulations."""

import json
from firedrake import *
from firedrake.petsc import PETSc


# ---------------------------------------------------------------------------
#   Parallel printing (MPI-safe)
# ---------------------------------------------------------------------------
def parprint(*args, **kwargs):
    """Print only on MPI rank 0."""
    if COMM_WORLD.rank == 0:
        PETSc.Sys.Print(*args, **kwargs)


# ---------------------------------------------------------------------------
#   Solver parameters
# ---------------------------------------------------------------------------
DEFAULT_SOLVER_PARAMS = {
    "snes_monitor": None,
    "snes_converged_reason": None,
    "ksp_monitor": None,
    "ksp_converged_reason": None,
}

LU_PARAMS = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "pc_factor_shift_type": "nonzero",
}

LU_PARAMS_UMFPACK = LU_PARAMS | {"pc_factor_mat_solver_type": "umfpack"}


class RegularisedBlockPC(AuxiliaryOperatorPC):
    """LU on the non-Real block, regularised so that the block is invertible.

    The non-Real block ``A`` is singular: the harmonic 1-form is a left null
    vector of the vorticity equation, and the pressure level is free. The
    multiplier column does not lie in the range of ``A``, so the Schur
    complement of :func:`real_block_params` is undefined without this.

    Mass terms in ``omega`` and ``p`` are added to the preconditioner only, so
    the solution is unchanged. ``omega_shift`` and ``pressure_shift`` come
    from the solve's ``appctx``.
    """

    def form(self, pc, test, trial):
        a, bcs = super().form(pc, test, trial)
        appctx = self.get_appctx(pc)
        w_, p_ = appctx["omega_index"], appctx["pressure_index"]
        return a + (
            appctx["omega_shift"] * inner(split(trial)[w_], split(test)[w_])
          + appctx["pressure_shift"] * inner(split(trial)[p_], split(test)[p_])
        ) * dx, bcs


def real_block_params(Z, params=None, assembled=None, regularised=False):
    """Solver parameters for a mixed space carrying "R" (Real) blocks.

    A monolithic AIJ matrix cannot hold a Real block, so the Real fields are
    eliminated with a matrix-free Schur complement and the remainder is
    factorised. Firedrake does this itself only when no solver parameters are
    supplied at all.

    ``params`` are merged on top, and ``assembled`` overrides the inner solver
    on the non-Real block. ``regularised`` uses :class:`RegularisedBlockPC` on
    that block, for when it is singular; it needs ``omega_shift``,
    ``pressure_shift``, ``omega_index`` and ``pressure_index`` in ``appctx``.
    """
    fields = [str(i) for i, V_ in enumerate(Z)
              if V_.ufl_element().family() != "Real"]
    reals = [str(i) for i, V_ in enumerate(Z)
             if V_.ufl_element().family() == "Real"]
    if not reals:
        raise ValueError("No Real blocks in Z; use the parameters directly.")

    out = {
        "mat_type": "matfree",
        "ksp_type": "fgmres",
        # Newton stalls at the default rtol; much tighter is unattainable, as a
        # constant-pressure mode can survive in the non-Real block.
        "ksp_rtol": 1e-9,
        "ksp_atol": 1e-14,
        "ksp_max_it": 200,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "full",
        "pc_fieldsplit_0_fields": ",".join(fields),
        "pc_fieldsplit_1_fields": ",".join(reals),
        "fieldsplit_0": {
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": (
                "enstrophy_stable_ns.common.RegularisedBlockPC" if regularised
                else "firedrake.AssembledPC"),
            ("aux" if regularised else "assembled"): assembled or {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                # The outer nullspace does not reach the split, so the
                # constant-pressure zero pivot has to be tolerated here.
                "pc_factor_shift_type": "nonzero",
            },
        },
        "fieldsplit_1": {
            "ksp_type": "gmres",
            "pc_type": "none",
            "ksp_rtol": 1e-12,
        },
    }
    return out | (params or {})


# ---------------------------------------------------------------------------
#   Data recording
# ---------------------------------------------------------------------------
class DataRecorder:
    """Records quantities of interest to JSON and manages VTK output."""

    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        self.data = []
        self.vtk_file = None
        self._fields = []

        import os
        os.makedirs(output_dir, exist_ok=True)

    def setup_vtk(self, filename, *fields):
        """Initialise a VTK file for output."""
        self.vtk_file = VTKFile(f"{self.output_dir}/{filename}.pvd")
        self._fields = fields

    def write_vtk(self, time=None):
        """Write current fields to VTK."""
        if self.vtk_file is None:
            raise AttributeError("VTK file not set up. Call setup_vtk() first.")
        if time is None:
            self.vtk_file.write(*self._fields)
        else:
            self.vtk_file.write(*self._fields, time=time)

    def record(self, **quantities):
        """Record a dictionary of scalar quantities."""
        self.data.append(quantities)

    def save_params(self, **groups):
        """Record the settings a run was made with, beside its data."""
        path = f"{self.output_dir}/params.json"
        with open(path, "w") as f:
            json.dump(groups, f, indent=4, default=str, sort_keys=True)

    def save_json(self, filename="data.json"):
        """Write all recorded data to a JSON file."""
        path = f"{self.output_dir}/{filename}"
        with open(path, "w") as f:
            json.dump(self.data, f, indent=4)
        parprint(RED % f"Data saved to {path}")

    def download_zip(self, zip_name=None):
        """Zip the output directory and download it (Colab only)."""
        download_zip_(zip_name=zip_name, output_dir=self.output_dir)


# ---------------------------------------------------------------------------
#   Download zip (Colab only)
# ---------------------------------------------------------------------------
def download_zip_(zip_name=None, output_dir="output"):
    """Zip the output directory and download it (Colab only)."""
    import zipfile
    import os
    from google.colab import files

    if zip_name is None:
        zip_name = f"output.zip"

    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, filenames in os.walk(output_dir):
            for filename in filenames:
                filepath = os.path.join(root, filename)
                zf.write(filepath)

    parprint(RED % f"Downloading {zip_name}...")
    if COMM_WORLD.rank == 0:
        files.download(zip_name)


# ---------------------------------------------------------------------------
#   Utility operators
# ---------------------------------------------------------------------------
def dev(w):
    """Symmetric gradient (deviator): sym(grad(w))."""
    return sym(grad(w))

def facet_jump(w, n):
    """Jump of a vector field across a facet: outer(w+ - w-, n+)."""
    return outer(w("+") - w("-"), n("+"))

def facet_jump_dev(w, n):
    """Symmetric part of the jump: sym(outer(w+ - w-, n+))."""
    return sym(facet_jump(w, n))

def rotate(w):
    """2D perpendicular rotation: (w0, w1) -> (-w1, w0)."""
    return as_vector([-w[1], w[0]])


# ---------------------------------------------------------------------------
#   Shared vortex street components
# ---------------------------------------------------------------------------
def boundary_submesh(mesh, ids, ds):
    """Submesh carrying the multiplier, with its cross-mesh measure.

    ``ids`` selects by label, so a boundary glued by periodicity is still
    picked up even though ``ds`` over it vanishes; omit such markers.
    """
    mesh_b = Submesh(mesh, mesh.topological_dimension - 1, ids)
    return mesh_b, Measure("dx", domain=mesh_b, intersect_measures=[ds])


def constant_nullspace(Z, mesh, component):
    """Constant nullspace in one component of ``Z``, for a free pressure."""
    return MixedVectorSpaceBasis(Z, [
        VectorSpaceBasis(constant=True, comm=mesh.comm)
        if i == component
        else Z.sub(i) for i in range(len(Z))
    ])


def tangent(w, n):
    """Tangential component w - (w.n) n."""
    return w - dot(w, n) * n


def h1_sym_build(n, h, sigma, dx, dS, mask=None, facet_degree=None):
    """Broken symmetric-gradient form D^h of sec:penalty.

    A ``mask`` restricts the integrals to a subregion. ``facet_degree``
    under-integrates the facet terms (``None`` integrates them fully).
    """
    m = 1 if mask is None else mask
    m_f = 1 if mask is None else avg(mask)
    dS_ = dS if facet_degree is None else dS(degree=facet_degree)
    def h1_sym(w1, w2):
        return m * inner(dev(w1), dev(w2)) * dx + m_f * (
          - inner(avg(dev(w1)), facet_jump_dev(w2, n))
          - inner(facet_jump_dev(w1, n), avg(dev(w2)))
          + sigma / avg(h) * inner(facet_jump_dev(w1, n), facet_jump_dev(w2, n))
        ) * dS_
    return h1_sym


def friction_build(gamma_arr, n, ds, degree=None):
    """Drag on the partial-slip boundaries, gamma <w1_||, w2_||>_S.

    ``degree`` under-integrates, as ``facet_degree`` in ``h1_sym_build``.
    """
    def friction(w1, w2):
        return sum([
            gamma * inner(tangent(w1, n), tangent(w2, n))
            * (ds(ind) if degree is None else ds(ind, degree=degree))
            for (ind, gamma) in gamma_arr
        ])
    return friction


# ---------------------------------------------------------------------------
#   Stage functions for IMEX in momentum equation
# ---------------------------------------------------------------------------
def midpoint_function(old, new, theta, implicit=True):
    """Midpoint value, implicit or extrapolated from the previous steps."""
    stages = len(old)
    if implicit:
        if stages == 1:
            return (
                theta * new
              + (1 - theta) * old[0]
            )
        else:
            return (
                theta * (1 + theta) / 2 * new
              + (1 - theta) * (1 + theta) * old[1]
              - (1 - theta) * theta / 2 * old[0]
            )
    else:
        if stages == 1:
            return old[0]
        else:
            return (
                (1 + theta) * old[1]
              - theta * old[0]
            )


def time_derivative(old, new, theta, ratio=None):
    """Time derivative approximation (BDF for multi-stage).

    ``ratio`` is the step ratio r = dt_n / dt_{n-1}. Passing it selects the
    variable-step BDF2 coefficients

        [ (1+2r)/(1+r) u^{n+1} - (1+r) u^n + r^2/(1+r) u^{n-1} ] / dt_n,

    which reduce to the constant-step formula at r = 1 (``theta = 1`` only).
    """
    stages = len(old)
    if stages == 1:
        return new - old[0]
    if ratio is not None:
        return (
            (1 + 2 * ratio) / (1 + ratio) * new
          - (1 + ratio) * old[1]
          + ratio * ratio / (1 + ratio) * old[0]
        )
    return (
        (1/2 + theta) * new
      - 2 * theta * old[1]
      - (1/2 - theta) * old[0]
    )
