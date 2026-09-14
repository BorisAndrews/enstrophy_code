"""2D Kelvin-Helmholtz instability.

The H^2-free ('charlie') reparametrisation over an Alfeld-split de Rham
complex, comparing the enstrophy-stable scheme ('asf') against a classical
unstabilised discretisation ('standard').
"""

from firedrake import *
from tqdm import tqdm
from enstrophy_stable_ns.common import (
    parprint,
    DataRecorder, DEFAULT_SOLVER_PARAMS, LU_PARAMS, real_block_params,
    dev, rotate,
)


# ---------------------------------------------------------------------------
#   Problem data
# ---------------------------------------------------------------------------
def _shear_profile(y, n, delta):
    """Tangential velocity of the shear layer, ``delta`` cells wide."""
    return tanh((y - 0.5) * n / delta)


def _initial_condition(U, P, u_target, u_BC, solver_params):
    """Project ``u_target`` onto the discretely divergence-free subspace."""
    Z = U * P
    z = Function(Z)
    u, p = split(z)
    v, q = TestFunctions(Z)

    F = (
        inner(dev(u - u_target), dev(v))
        - inner(p, div(v))
        - inner(div(u), q)
    ) * Measure("dx", domain=U.mesh())

    parprint(GREEN % "Setting up initial condition:")
    solve(F == 0, z, bcs=[DirichletBC(Z.sub(0), u_BC, "on_boundary")],
          solver_parameters=solver_params)
    return Function(U).assign(z.subfunctions[0])


# ---------------------------------------------------------------------------
#   Function spaces
# ---------------------------------------------------------------------------
def _function_spaces(mesh, mesh_b, scheme, weak_bcs):
    """Mixed space for the requested scheme, with U and P handed back too.

    ``asf`` carries (u, p, eta, R, omega), the reparametrisation solving for
    eta in place of grad^perp(omega); under weak boundary conditions it gains
    the traction multiplier ``mult`` on the boundary submesh and two real
    multipliers, for the circulation of omega and the mean of p.
    """
    U = VectorFunctionSpace(mesh, "CG", 2, variant="alfeld")
    P = FunctionSpace(mesh, "DG", 1, variant="alfeld")
    if scheme != "asf":
        return U * P, U, P

    W = FunctionSpace(mesh, "CG", 3, variant="alfeld")
    if not weak_bcs:
        return U * P * U * P * W, U, P

    M = VectorFunctionSpace(mesh_b, "CG", 2)   # trace of U on the boundary
    Real = FunctionSpace(mesh, "R", 0)
    return U * P * U * P * W * M * Real * Real, U, P


def _name_fields(z, scheme, weak_bcs):
    """Label the subfunctions for VTK output."""
    names = ["velocity", "bernoulli_pressure"]
    if scheme == "asf":
        names = ["velocity", "bernoulli_pressure", "eta", "R", "vorticity"]
        if weak_bcs:
            names += ["traction_multiplier", "vorticity_multiplier",
                      "pressure_multiplier"]
    for field, name in zip(z.subfunctions, names):
        field.rename(name)


def _constant_nullspace(Z, mesh, *components):
    """Constant nullspace in the given components of ``Z``."""
    comm = mesh.comm
    return MixedVectorSpaceBasis(Z, [
        VectorSpaceBasis(constant=True, comm=comm) if i in components else Z.sub(i)
        for i in range(len(Z))
    ])


# ---------------------------------------------------------------------------
#   Residuals
# ---------------------------------------------------------------------------
def _residual_slip_asf(z, tests, u_old, mesh, Re, dt, theta, imex):
    """Enstrophy-stable scheme with free slip imposed strongly (sec:scheme)."""
    Z = z.function_space()
    dx = Measure("dx", domain=mesh)
    u, p, eta, R, omega = split(z)
    v, q, zeta, S, chi = tests
    u_mid = theta * u + (1 - theta) * u_old
    u_visc = u_old if imex != "none" else u_mid

    F = (
        # Vorticity recovery via eta
        inner(grad(omega), grad(chi)) + inner(eta, rotate(grad(chi)))
        + inner(eta, zeta) - inner(R, div(zeta)) - 2 * inner(dev(u_visc), dev(zeta))
        - inner(div(eta), S)
        # Momentum and incompressibility
        + 1 / dt * inner(u - u_old, v) - inner(p, div(v))
        + inner(omega * rotate(u_old if imex == "full" else u_mid), v)
        - inner(div(u), q)
    ) * dx
    if Re != "infty":
        F += 2 / Re * inner(dev(u_mid), dev(v)) * dx

    bcs = [
        DirichletBC(Z.sub(0).sub(1), 0.0, "on_boundary"),   # u . n = 0
        DirichletBC(Z.sub(2).sub(1), 0.0, "on_boundary"),   # eta . n = 0
        DirichletBC(Z.sub(4), 0.0, "on_boundary"),          # omega = 0
    ]
    return F, bcs, _constant_nullspace(Z, mesh, 1, 3)


def _residual_dirichlet_asf(z, tests, u_old, u_BC, mesh, dC,
                            Re, dt, theta, imex):
    """Enstrophy-stable scheme with the data imposed variationally (sec:bcs)."""
    dx = Measure("dx", domain=mesh)
    ds = Measure("ds", domain=mesh)
    nrm = FacetNormal(mesh)

    u, p, eta, R, omega, mult, lam, lam_p = split(z)
    v, q, zeta, S, chi, rho, mu, mu_p = tests
    u_mid = theta * u + (1 - theta) * u_old

    traction = inner(mult, zeta) * dC
    momentum_boundary = (
        inner(p, dot(v, nrm)) * ds
      - 1 / Re * inner(mult, v) * dC
    )

    F = (
        # Vorticity recovery via eta
        inner(grad(omega), grad(chi)) + inner(eta, rotate(grad(chi)))
        + inner(eta, zeta) - inner(R, div(zeta)) - 2 * inner(dev(u_mid if imex == "none" else u_old), dev(zeta))
        - inner(div(eta), S)
        # Momentum and incompressibility
        + 1 / dt * inner(u - u_old, v) - inner(p, div(v))
        + inner(omega * rotate(u_old if imex == "full" else u_mid), v)
        - inner(div(u), q) + inner(lam_p, q) + inner(p, mu_p)
        # Circulation constraint
        + inner(lam, chi) + inner(omega + div(rotate(u)), mu)
    ) * dx + traction + momentum_boundary
    if Re != "infty":
        F += 2 / Re * inner(dev(u_mid), dev(v)) * dx

    # Velocity data, tested against the trace space.
    F += inner(u - u_BC, rho) * dC

    return F, []


def _standard_form(z, tests, u_old, mesh, Re, dt, theta, imex):
    """Shared residual of the unstabilised scheme; only the data differ."""
    dx = Measure("dx", domain=mesh)
    u, p = split(z)
    v, q = tests
    u_mid = theta * u + (1 - theta) * u_old
    u_adv = u_old if imex in ("partial", "full") else u_mid
    u_rot = u_old if imex == "full" else u_mid

    F = (
        1 / dt * inner(u - u_old, v) - inner(p, div(v))
        - inner(div(rotate(u_adv)) * rotate(u_rot), v)
        - inner(div(u), q)
    ) * dx
    if Re != "infty":
        F += 2 / Re * inner(dev(u_mid), dev(v)) * dx
    return F


def _residual_slip_standard(z, tests, u_old, u_BC, mesh, Re, dt, theta, imex):
    """Unstabilised scheme with free slip, lifted strongly."""
    Z = z.function_space()
    F = _standard_form(z, tests, u_old, mesh, Re, dt, theta, imex)
    bcs = [DirichletBC(Z.sub(0).sub(1), 0.0, "on_boundary")]
    return F, bcs, _constant_nullspace(Z, mesh, 1)


def _residual_dirichlet_standard(z, tests, u_old, u_BC, mesh, Re, dt, theta, imex):
    """Unstabilised scheme with the full velocity lifted strongly."""
    Z = z.function_space()
    F = _standard_form(z, tests, u_old, mesh, Re, dt, theta, imex)
    bcs = [DirichletBC(Z.sub(0), u_BC, "on_boundary")]
    return F, bcs, _constant_nullspace(Z, mesh, 1)


# ---------------------------------------------------------------------------
#   Diagnostics
# ---------------------------------------------------------------------------
def _diagnostics(z, u_old, mesh, scheme, weak_bcs, theta, periodic):
    """Quantities of interest for one record, as (values, log message)."""
    dx = Measure("dx", domain=mesh)
    ds = Measure("ds", domain=mesh)
    nrm = FacetNormal(mesh)

    u, p = split(z)[0], split(z)[1]
    values = {
        "energy": assemble(0.5 * inner(u, u) * dx),
        "enstrophy": assemble(inner(dev(u), dev(u)) * dx),
        "divergence": assemble(inner(div(u), div(u)) * dx),
    }
    msg = (
        f"Energy: {values['energy']:.4e} | "
        f"Enstrophy: {values['enstrophy']:.4e} | "
        f"Divergence: {values['divergence']:.4e}"
    )
    if scheme != "asf":
        return values, msg

    eta, omega = split(z)[2], split(z)[4]
    u_mid = theta * u + (1 - theta) * u_old
    values |= {
        # Nonzero under Dirichlet data, from the harmonic component of eta
        "omega_error": assemble(inner(rotate(grad(omega)) + eta,
                                      rotate(grad(omega)) + eta) * dx),
        # ||curl omega||^2, at the end of the step
        "dissipation": assemble(inner(grad(omega), grad(omega)) * dx),
        # int(omega - curl u), which the circulation constraint fixes to zero
        "circulation": assemble((omega + div(rotate(u))) * dx),
    }
    msg += (
        f" | Omega error: {values['omega_error']:.4e} | "
        f"||curl omega||^2: {values['dissipation']:.4e} | "
        f"int(omega - curl u): {values['circulation']:.4e}"
    )
    if not weak_bcs:
        return values, msg

    values |= {
        "vorticity_multiplier": float(z.subfunctions[6]),
        "enstrophy_flux": assemble(inner(p, dot(eta, nrm)) * ds),
        "advection": assemble(inner(omega * rotate(u_mid), eta) * dx),
        "div_eta": assemble(inner(div(eta), div(eta)) * dx),
        "harmonic": assemble(eta[1] * dx) if periodic == "x" else 0.0,
    }
    msg += (
        f" | lambda: {values['vorticity_multiplier']:.4e} | "
        f"enstrophy flux: {values['enstrophy_flux']:.4e} | "
        f"advection: {values['advection']:.4e} | "
        f"||div eta||^2: {values['div_eta']:.4e} | "
        f"harmonic: {values['harmonic']:.4e}"
    )
    return values, msg


# ---------------------------------------------------------------------------
#   Driver
# ---------------------------------------------------------------------------
def kelvin_helmholtz_2d(prob_params=None, mesh_params=None,
                        disc_params=None, save_params=None,
                        solver_params=None, check_dof=False):
    """Run a 2D Kelvin-Helmholtz instability simulation.

    Parameters
    ----------
    prob_params : dict
        - ``Re`` (float): Reynolds number (default 1e6)
        - ``delta`` (float): (half) number of cells in shear layer (default 5)
        - ``epsilon`` (float): perturbation amplitude (default 0.01)
        - ``BCs`` (str): ``"slip"`` imposes u.n, eta.n and omega strongly;
          ``"dirichlet"`` imposes the full velocity variationally
    mesh_params : dict
        - ``n`` (int): mesh resolution per direction (default 50)
        - ``periodic`` (str): ``"x"`` (default) or ``"none"``
    disc_params : dict
        - ``dt`` (float): timestep (default 0.1)
        - ``T`` (float): final time (default 10.0)
        - ``theta`` (float): implicit midpoint parameter (default 0.5)
        - ``scheme`` (str): ``"asf"`` or ``"standard"`` (default ``"asf"``)
        - ``imex`` (str): explicit handling of the advective term, one of
          ``"none"``, ``"partial"`` or ``"full"`` (default ``"none"``)
    save_params : dict
        - ``save_vtk`` (bool): enable VTK output (default True). Needs the
          ``vtk`` package, which a default Firedrake install does not carry
        - ``save_freq`` (int): write every N steps (default 1)
        - ``output_dir`` (str): output directory
        - ``download`` (bool): download output as zip for Colab (default False)

    Notes
    -----
    ``BCs="dirichlet"`` puts a boundary ``Submesh`` and "R" blocks in one mixed
    space on a periodic mesh, which needs the Firedrake patches in
    ``patches/``.
    """
    prob_params = prob_params or {}
    mesh_params = mesh_params or {}
    disc_params = disc_params or {}
    save_params = save_params or {}
    solver_params = solver_params or {}

    # ----------------------
    #    Parse parameters
    # ----------------------
    Re = prob_params.get("Re", 1e6)
    if Re != "infty": Re = Constant(Re)
    delta_width = prob_params.get("delta", 5)
    epsilon = prob_params.get("epsilon", 0.01)
    BCs = prob_params.get("BCs", "dirichlet").lower()
    if BCs not in {"slip", "dirichlet"}:
        raise ValueError(f"Unknown BCs '{BCs}'. Use 'dirichlet' or 'slip'.")

    n = mesh_params.get("n", 50)
    periodic = mesh_params.get("periodic", "x").lower()
    if periodic not in {"x", "none"}:
        raise ValueError(f"Unknown periodicity '{periodic}'. Use 'x' or 'none'.")

    dt = Constant(disc_params.get("dt", 0.1))
    T = disc_params.get("T", 10.0)
    theta = Constant(disc_params.get("theta", 0.5))
    scheme = disc_params.get("scheme", "asf").lower()
    if scheme not in {"asf", "standard"}:
        raise ValueError(f"Unknown scheme '{scheme}'. Use 'asf' or 'standard'.")
    imex = disc_params.get("imex", "none")
    if imex not in {"none", "partial", "full"}:
        raise ValueError(f"Invalid IMEX option '{imex}'. Use 'none', 'partial' or 'full'.")

    # The unstabilised scheme always lifts its data strongly.
    weak_bcs = (scheme == "asf" and BCs == "dirichlet")

    save_vtk = save_params.get("save_vtk", True)
    save_freq = save_params.get("save_freq", 1)
    recorder = DataRecorder(
        output_dir=save_params.get(
            "output_dir", f"output/kelvin_helmholtz_2d/{BCs}/{scheme}"),
    )
    download = save_params.get("download", False)

    recorder.save_params(prob=prob_params, mesh=mesh_params, disc=disc_params,
                         resolved=dict(scheme=scheme, BCs=BCs))

    # ----------
    #    Mesh
    # ----------
    if periodic == "x":
        mesh = PeriodicUnitSquareMesh(n, n, direction="x", quadrilateral=False)
    else:
        mesh = UnitSquareMesh(n, n)
    x, y = SpatialCoordinate(mesh)

    mesh_b = dC = None
    if weak_bcs:
        mesh_b = Submesh(mesh, mesh.topological_dimension - 1, "on_boundary")
        dC = Measure("dx", domain=mesh_b,
                     intersect_measures=[Measure("ds", domain=mesh)])

    # -------------------------
    #    Spaces and unknowns
    # -------------------------
    Z, U, P = _function_spaces(mesh, mesh_b, scheme, weak_bcs)
    if check_dof:
        parprint(RED % f"DoF check: {[Z_.dim() for Z_ in Z]} | Total: {Z.dim()}")
        return

    z = Function(Z)
    tests = TestFunctions(Z)
    _name_fields(z, scheme, weak_bcs)
    u_sol = z.subfunctions[0]

    # -----------------------
    #    Initial condition
    # -----------------------
    u_target = as_vector([
        _shear_profile(y, n, delta_width),
        epsilon * y * (1 - y) * (sin(2*pi*x) + sin(4*pi*x) - cos(4*pi*x)),
    ])
    # Wall data: the shear profile itself, with no flux through the wall.
    u_BC = as_vector([_shear_profile(y, n, delta_width), Constant(0.0)])

    u_old = _initial_condition(U, P, u_target, u_BC,
                               DEFAULT_SOLVER_PARAMS | solver_params)
    u_sol.assign(u_old)

    # --------------
    #    Residual
    # --------------
    if weak_bcs:
        F, bcs = _residual_dirichlet_asf(z, tests, u_old, u_BC, mesh, dC, Re, dt, theta, imex)
        nullspace = None
        solver_params = real_block_params(Z, DEFAULT_SOLVER_PARAMS | solver_params)
    else:
        residual = {
            ("slip", "asf"): lambda: _residual_slip_asf(
                z, tests, u_old, mesh, Re, dt, theta, imex),
            ("slip", "standard"): lambda: _residual_slip_standard(
                z, tests, u_old, u_BC, mesh, Re, dt, theta, imex),
            ("dirichlet", "standard"): lambda: _residual_dirichlet_standard(
                z, tests, u_old, u_BC, mesh, Re, dt, theta, imex),
        }[(BCs, scheme)]
        F, bcs, nullspace = residual()
        solver_params = LU_PARAMS | DEFAULT_SOLVER_PARAMS | solver_params

    # ---------------
    #    Time loop
    # ---------------
    if save_vtk:
        fields = (z.subfunctions[0], z.subfunctions[1])
        if scheme == "asf":
            fields += (z.subfunctions[4],)
        recorder.setup_vtk("fields", *fields)
        recorder.write_vtk(time=0.0)

    def record(t_val, step_val):
        values, msg = _diagnostics(z, u_old, mesh, scheme, weak_bcs, theta,
                                   periodic)
        parprint(RED % msg)
        recorder.record(time=t_val, step=step_val, **values)

    record(0.0, 0)

    t = 0.0
    num_steps = round(T / float(dt))
    for step in tqdm(range(1, num_steps + 1), desc="Kelvin-Helmholtz (2D)"):
        t += float(dt)
        parprint(GREEN % f"Solving for t = {t:.4e}:")
        solve(F == 0, z, bcs=bcs, nullspace=nullspace,
              solver_parameters=solver_params)
        record(t, step)
        if save_vtk and step % save_freq == 0:
            recorder.write_vtk(time=t)
        u_old.assign(u_sol)

    recorder.save_json()
    if download: recorder.download_zip()


def main():
    """CLI entry point with default parameters."""
    kelvin_helmholtz_2d()


if __name__ == "__main__":
    main()
