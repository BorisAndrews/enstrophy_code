"""2D turbulent wake behind an obstacle.

Uses the penalty method with a de Rham complex (CG3 / BDM2 / DG1).
Compares the enstrophy-stable scheme ('asf'), MEEVC, and a classical
unstabilised discretisation ('standard').
"""

from firedrake import *
from tqdm import tqdm
from enstrophy_stable_ns.meshes import (
    rect_with_hole_2d,
    rect_with_hole_2d_netgen,
    _check_periodic_direction,
)
from enstrophy_stable_ns.common import (
    parprint,
    DataRecorder, DEFAULT_SOLVER_PARAMS, LU_PARAMS,
    dev, rotate,
    boundary_submesh, constant_nullspace,
    h1_sym_build, friction_build,
    midpoint_function, time_derivative,
)


# ---------------------------------------------------------------------------
#   Discrete spaces
# ---------------------------------------------------------------------------
def _function_spaces(mesh, mesh_b, has_vorticity, strong_bcs):
    """Mixed space for the requested scheme."""
    spaces = []
    if has_vorticity:
        spaces.append(FunctionSpace(mesh, "CG", 3))
    V = FunctionSpace(mesh, "BDM", 2)
    spaces.append(V)
    spaces.append(FunctionSpace(mesh, "DG", 1))
    if not strong_bcs:
        spaces.append(FunctionSpace(mesh_b, "DG", 2))
    return MixedFunctionSpace(spaces)


def _name_fields(z, has_vorticity, strong_bcs, bernoulli=False):
    """Label the subfunctions for VTK output."""
    names = (
        (["vorticity"] if has_vorticity else []) \
      + ["velocity", "bernoulli_pressure" if bernoulli else "pressure"] \
      + ([] if strong_bcs else ["viscous_normal_stress"])
    )
    for field, name in zip(z.subfunctions, names):
        field.rename(name)


# ---------------------------------------------------------------------------
#   Other
# ---------------------------------------------------------------------------
SCHEME_MAT_SOLVER = {
    "asf"      : "mumps",
    "meevc"    : "umfpack",
    "standard" : "mumps",
}


def _curl_s(w):
    """Scalar-to-vector curl."""
    return - rotate(grad(w))


# ---------------------------------------------------------------------------
#   Driver
# ---------------------------------------------------------------------------
def vortex_street_2d(prob_params=None, mesh_params=None,
                     disc_params=None, save_params=None,
                     solver_params=None, check_dof=False):
    """Run a 2D turbulent wake simulation.

    Parameters
    ----------
    prob_params : dict
        Problem parameters:
        - ``Re`` (float): Reynolds number (default 1e4)
        - ``gamma_inlet`` (float): friction on the inlet (default 5e2)
        - ``gamma_obstacle`` (float): friction on the obstacle (default 1e2)
        - ``outlet`` (str): ``"slip"`` (default) prescribes ``u.n``;
          ``"traction"`` prescribes the pressure ``p_out`` weakly (the
          do-nothing condition, ill-posed under backflow; not with ``"meevc"``)
        - ``forcing`` (float): streamwise body force (default 5e-2 under x
          periodicity, otherwise None)
        - ``p_out`` (float): outlet pressure for ``"traction"`` (default 0)

    mesh_params : dict
        Passed to ``rect_with_hole_2d``. Keys include ``r``, ``left``,
        ``right``, ``updown``, ``h_fine``, ``h_coarse``, ``obstacle`` and
        ``periodic_direction`` (default ``"x"``, gluing the inlet to the
        outlet; ``"y"`` glues the walls).

    disc_params : dict
        Discretisation parameters:
        - ``scheme`` (str): ``"standard"``, ``"meevc"`` or ``"asf"``
          (default ``"asf"``)
        - ``imex`` (str): ``"none"`` (default), ``"partial"`` or ``"full"``,
          lagging one or both factors of the advective term
        - ``stages`` (int): 1, or 2 for BDF2 with ``theta=1`` (default 1)
        - ``aux_advection`` (bool): advect with the auxiliary vorticity rather
          than ``curl u`` (default True; ignored by ``"standard"``)
        - ``strong_bcs`` (bool): impose the vorticity boundary conditions in
          ``"asf"`` strongly (default True)
        - ``sigma`` (float): penalty parameter (default 5e1)
        - ``dt`` (float): timestep (default 5e-2)
        - ``T`` (float): final time (default 5e1)
        - ``theta`` (float): implicit midpoint parameter (default 0.5)

    save_params : dict
        - ``save_vtk`` (bool): enable VTK output (default True)
        - ``save_freq`` (int): write every N steps (default 2)
        - ``output_dir`` (str): output directory
          (default "output/vortex_street_2d/<scheme>")
        - ``download`` (bool): download output as zip for Colab (default False)

    solver_params : dict
        PETSc/Firedrake nonlinear solver parameters. The direct solver
        defaults per scheme (``SCHEME_MAT_SOLVER``).
    """
    prob_params = prob_params or {}
    mesh_params = mesh_params or {}
    disc_params = disc_params or {}
    save_params = save_params or {}
    solver_params = solver_params or {}

    # ----------------------
    #    Parse parameters
    # ----------------------
    # Problem parameters
    Re = Constant(prob_params.get("Re", 1e4))
    gamma_inlet = Constant(prob_params.get("gamma_inlet", 5e2))
    gamma_obstacle = Constant(prob_params.get("gamma_obstacle", 1e2))
    outlet = str(prob_params.get("outlet", "slip")).lower()
    if outlet not in {"slip", "traction"}:
        raise ValueError(f"Unknown outlet '{outlet}'. Use 'slip' or 'traction'.")
    p_out = Constant(prob_params.get("p_out", 0.0))
    forcing = prob_params.get("forcing", None)

    # Mesh parameters
    mesh_params.setdefault("periodic_direction", "x")
    r_obs = mesh_params.get("r", 0.5)
    updown = mesh_params.get("updown", 1.0)
    periodic_x, periodic_y = _check_periodic_direction(mesh_params["periodic_direction"])

    # Discretisation parameters
    scheme = disc_params.get("scheme", "asf").lower()
    if scheme not in {"standard", "meevc", "asf"}:
        raise ValueError(f"Unknown scheme '{scheme}'. Use 'standard', 'meevc' or 'asf'.")
    imex = disc_params.get("imex", "none").lower()
    if imex not in {"none", "partial", "full"}:
        raise ValueError(f"Unknown imex '{imex}'. Use 'none', 'partial' or 'full'.")
    stages = disc_params.get("stages", 1)
    if stages not in {1, 2}:
        raise ValueError(
            "1 and 2 stage methods implemented only. "
            f"Staged '{stages}' not yet available."
        )
    aux_advection = disc_params.get("aux_advection", True)
    strong_bcs = disc_params.get("strong_bcs", True)
    bernoulli = disc_params.get("bernoulli", False)
    if bernoulli:
        raise ValueError("Bernoulli pressure not yet implemented")
    sigma = Constant(disc_params.get("sigma", 5e1))
    dt = Constant(disc_params.get("dt", 5e-2))
    T = disc_params.get("T", 5e1)
    theta = Constant(disc_params.get("theta", 0.5))

    # Save parameters
    save_vtk = save_params.get("save_vtk", True)
    save_freq = round(save_params.get("save_freq", 2))
    download = save_params.get("download", False)

    # ------------------------
    #    Process parameters
    # ------------------------
    # Problem parameters
    traction_outlet = (outlet == "traction")
    if (not traction_outlet and float(p_out) != 0.0):
        Warning(
            "Outlet pressure specified without traction conditions on the outlet. "
            "Outlet pressure will be ignored."
        )

    # Discretisation parameters
    if (aux_advection and scheme == "standard"):
        Warning(
            "Auxiliary advection requested but scheme set to standard. "
            "Defaulting to no auxiliary advection."
        )
        aux_advection = False
    if traction_outlet and scheme == "meevc":
        Warning(
            "Scheme 'meevc' does not support outlet='traction': its vorticity "
            "equation has no boundary term to carry the prescribed traction. "
            "Defaulting to slip outlet."
        )
        traction_outlet = False
    if traction_outlet and periodic_x:
        Warning(
            "Outlet traction requested with periodic_direction='x', which "
            "glues the outlet to the inlet and leaves no outlet to prescribe "
            "it on. Defaulting to periodic in x."
        )
        traction_outlet = False
    has_vorticity = (scheme in {"meevc", "asf"})
    if forcing is not None:
        forcing = Constant(forcing)
    if periodic_x and (forcing is None):
        forcing = Constant(5e-2)
    if ("strong_bcs" in disc_params) and (scheme == "meevc"):
        Warning(
            f"Choice of strong vs. weak BCs (strong_bcs = {strong_bcs}) "
            f"unavailable for scheme = meevc"
        )
        strong_bcs = True

    # Solver parameters
    user_solver = dict(solver_params)
    mat_solver = user_solver.pop(
        "pc_factor_mat_solver_type", SCHEME_MAT_SOLVER[scheme])
    solver_params = (LU_PARAMS | DEFAULT_SOLVER_PARAMS | user_solver
                     | {"pc_factor_mat_solver_type": mat_solver})

    # ---------------------
    #    Set up recorder
    # ---------------------
    recorder = DataRecorder(
        output_dir=save_params.get(
            "output_dir", f"output/vortex_street_2d/{scheme}"),
    )
    recorder.save_params(
        prob=prob_params,
        mesh=mesh_params,
        disc=disc_params,
        resolved=dict(
            scheme=scheme,
            forcing=(float(forcing) if forcing is not None else None),
            mat_solver_type=solver_params.get("pc_factor_mat_solver_type"),
            Re=float(Re),
            gamma_obstacle=float(gamma_obstacle),
            sigma=float(sigma),
            dt=float(dt),
            T=T,
            theta=float(theta),
            stages=stages,
            imex=imex,
            periodic_direction=mesh_params["periodic_direction"],
            aux_advection=aux_advection,
            strong_bcs=strong_bcs,
            bernoulli=bernoulli,
            outlet=outlet
        )
    )

    # ---------------------------
    #    Function spaces
    # ---------------------------
    # Mesh
    curved = mesh_params.pop("curved", False)
    mesh = (rect_with_hole_2d_netgen if curved else rect_with_hole_2d)(**mesh_params)
    x_c, y_c = SpatialCoordinate(mesh)
    n = FacetNormal(mesh)
    h = CellVolume(mesh) ** 0.5 if curved else CellDiameter(mesh)

    # Measures
    dx = Measure("dx", domain=mesh)
    ds = Measure("ds", domain=mesh)
    dS = Measure("dS", domain=mesh)

    # Boundary ids: 1 = inlet, 2 = outlet, 3 = walls, 4 = obstacle.
    INLET, OUTLET, WALL, OBSTACLE = ((2, 3, (1, 4), 5) if curved else (1, 2, 3, 4))
    weak_ids = ([] if periodic_x else [INLET]) + [OBSTACLE]
    friction_ids = ([] if periodic_x else [(INLET, gamma_inlet)]) + [(OBSTACLE, gamma_obstacle)]
    strong_ids = ([] if periodic_y else [WALL]) + ([] if (periodic_x or traction_outlet) else [OUTLET])

    # Boundary mesh
    mesh_b = dC = None
    if not strong_bcs:
        mesh_b, dC = boundary_submesh(mesh, weak_ids, ds)

    # Function spaces
    Z = _function_spaces(mesh, mesh_b, has_vorticity, strong_bcs)

    # DoF check
    if check_dof:
        parprint(RED % f"DoF check: {[Z_.dim() for Z_ in Z]} | Total: {Z.dim()}")
        return

    # ---------------
    #    Functions
    # ---------------
    # Component indices
    i_w = (0 if has_vorticity else None)  # omega
    i_u = (1 if has_vorticity else 0)  # u
    i_p = i_u + 1  # p
    i_s = i_p + 1  # stress multiplier

    # Solutions
    z = Function(Z)
    fields = split(z)
    if has_vorticity:
        w = fields[i_w]
    u, p = fields[i_u], fields[i_p]
    if not strong_bcs:
        s = fields[i_s]
    _name_fields(z, has_vorticity, strong_bcs, bernoulli)

    # Tests
    tests = TestFunctions(Z)
    if has_vorticity:
        c = tests[i_w]
    v, q = tests[i_u], tests[i_p]
    if not strong_bcs:
        r = tests[i_s]

    # Outputs
    if has_vorticity:
        w_sol = z.subfunctions[i_w]
    u_sol, p_sol = z.subfunctions[i_u], z.subfunctions[i_p]
    if not strong_bcs:
        s_sol = z.subfunctions[i_s]

    # Previous-step values
    z_old = [Function(Z) for _ in range(stages)]
    if has_vorticity:
        w_old = [z_old[stage_].subfunctions[i_w] for stage_ in range(stages)]
    u_old, p_old = [z_old[stage_].subfunctions[i_u] for stage_ in range(stages)], [z_old[stage_].subfunctions[i_p] for stage_ in range(stages)]
    if not strong_bcs:
        s_old = [z_old[stage_].subfunctions[i_s] for stage_ in range(stages)]

    # Midpoint values (for momentum equation)
    if has_vorticity:
        w_mid = midpoint_function(w_old, w, theta)
    u_mid, p_mid = midpoint_function(u_old, u, theta), midpoint_function(p_old, p, theta)
    if not strong_bcs:
        s_mid = midpoint_function(s_old, s, theta)

    # -------------------------
    #    Boundary conditions
    # -------------------------
    # Construct u_BC
    off_obstacle = conditional(lt(x_c**2 + y_c**2, ((updown + r_obs) / 2)**2), 0.0, 1.0)
    u_BC = off_obstacle * as_vector([1.0, 0.0])

    # Initialise
    bcs = []

    # Walls & outlet
    omega_bcs = (  # omega
        [DirichletBC(Z.sub(i_w), 0, index) for index in strong_ids]
        if has_vorticity else []
    )
    bcs.extend(omega_bcs)
    for index in strong_ids:  # u
        bcs.append(DirichletBC(Z.sub(i_u), u_BC, index))

    # Inlet & obstacle
    if strong_bcs:
        if has_vorticity:  # omega
            if INLET in weak_ids:
                bcs.append(EquationBC(
                    inner(w + 2 * gamma_inlet * dot(u, rotate(n)), c) * ds(INLET) == 0,
                    z, INLET, V=Z.sub(i_w), bcs=omega_bcs
                ))
            if OBSTACLE in weak_ids:
                bcs.append(EquationBC(
                    inner(w + 2 * (gamma_obstacle + 1 / r_obs) * dot(u, rotate(n)), c) * ds(OBSTACLE) == 0,
                    z, OBSTACLE, V=Z.sub(i_w), bcs=omega_bcs)
                )
        for index in weak_ids:  # u
            bcs.append(DirichletBC(Z.sub(i_u), u_BC, index))

    # ---------------
    #    Nullspace
    # ---------------
    if traction_outlet:
        nullspace = None
    else:
        nullspace = constant_nullspace(Z, mesh, i_p)

    # ----------------------------
    #    Utility inner products
    # ----------------------------
    h1_sym = h1_sym_build(n, h, sigma, dx, dS)
    far_field = conditional(ge(x_c**2 + y_c**2, (2 * r_obs)**2), 1.0, 0.0)
    h1_sym_far = h1_sym_build(n, h, sigma, dx, dS, mask=far_field)
    friction = friction_build(friction_ids, n, ds)

    # -----------------
    #    Data output
    # -----------------
    # Traction needs true pressure, so undo Bernoulli
    p_true = (p_mid - 1/2 * dot(u_mid, u_mid)) if bernoulli else p_mid
    stress = 2 / Re * dev(u_mid) - p_true * Identity(2)

    def record_data(t_val, step_val):
        energy = 0.5 * assemble(inner(u, u) * dx)
        enstrophy = assemble(h1_sym(u, u) + friction(u, u))
        enstrophy_int = assemble(h1_sym(u, u))
        enstrophy_far = assemble(h1_sym_far(u, u))
        drag = - assemble(dot(as_vector([1, 0]), dot(stress, n)) * ds(OBSTACLE))
        lift = - assemble(dot(as_vector([0, 1]), dot(stress, n)) * ds(OBSTACLE))
        backflow = assemble(conditional(lt(dot(u, n), 0), - dot(u, n), 0) * ds(OUTLET))
        data = dict(
            time=t_val, step=step_val, energy=energy,
            enstrophy=enstrophy, enstrophy_int=enstrophy_int, enstrophy_far=enstrophy_far, drag=drag, lift=lift,
            backflow=backflow
        )
        msg = (
            f"Energy: {energy:.4e} | Enstrophy: {enstrophy:.4e} | "
            f"Enstrophy (int): {enstrophy_int:.4e} | "
            f"Enstrophy (far): {enstrophy_far:.4e} | "
            f"Drag: {drag:.4e} | Lift: {lift:.4e} | Backflow: {backflow:.4e}"
        )
        parprint(RED % msg)
        recorder.record(**data)

    # -------------------------------
    #    Initial condition updater
    # -------------------------------
    def _update_vals():
        for stage_ in range(stages-1):
            if has_vorticity:
                w_old[stage_].assign(w_old[stage_+1])
            u_old[stage_].assign(u_old[stage_+1])
            p_old[stage_].assign(p_old[stage_+1])
            if not strong_bcs:
                s_old[stage_].assign(s_old[stage_+1])
        if has_vorticity:
            w_old[stages-1].assign(w_sol)
        u_old[stages-1].assign(u_sol)
        p_old[stages-1].assign(p_sol)
        if not strong_bcs:
            s_old[stages-1].assign(s_sol)

    # --------------------------
    #    Initial Stokes solve
    # --------------------------
    if not periodic_x:
        # Dissipation, friction & weakly-enforced BCs
        if scheme == "meevc":
            F = 1 / Re * inner(_curl_s(w), v) * dx
        else:
            F = 2 / Re * (
                h1_sym(u, v)
              + friction(u, v)
            )
        if not strong_bcs:
            F -= (
                inner(s - p, dot(v, n))
              + inner(dot(u - u_BC, n), r)
            ) * dC

        # Pressure & incompressibility
        F -= (
            inner(p, div(v))
          + inner(div(u), q)
        ) * dx
        if traction_outlet:
            F += inner(p_out, dot(v, n)) * ds(OUTLET)

        # Vorticity reconstruction
        if scheme == "asf":
            F += 1 / Re * inner(_curl_s(w), _curl_s(c)) * dx
            F -= 2 / Re * (
                h1_sym(u, _curl_s(c))
              + friction(u, _curl_s(c))
            )
            F += inner(p, dot(_curl_s(c), n)) * ds
            if traction_outlet:
                F -= inner(p_out, dot(_curl_s(c), n)) * ds(OUTLET)
            if not strong_bcs:
                F += inner(s - p, dot(_curl_s(c), n)) * dC
        elif scheme == "meevc":
            F += inner(w, c) * dx
            F -= inner(u, _curl_s(c)) * dx

        # Solve
        parprint(GREEN % "Setting up ICs for t = 0:")
        solve(
            F == 0,
            z,
            bcs=bcs,
            nullspace=nullspace,
            solver_parameters=solver_params
        )

        # Update initial data
        for _ in range(stages):
            _update_vals()

    # Output
    record_data(0.0, 0)
    if save_vtk:
        fields_out = ((w_sol,) if has_vorticity else ()) + (u_sol, p_sol)
        recorder.setup_vtk("velocity", *fields_out)
        recorder.write_vtk(time=0.0)

    # ------------------------------
    #    Full Navier-Stokes solve
    # ------------------------------
    # Transient
    u_dt = time_derivative(u_old, u, theta)
    F = 1 / dt * inner(u_dt, v) * dx
    
    # Advection
    u_adv = midpoint_function(
        u_old, u, theta,
        implicit=(False if imex == "full" else True)
    )
    u_flux = midpoint_function(
        u_old, u, theta,
        implicit=(True if imex == "none" else False)
    )
    if aux_advection:
        w_flux = midpoint_function(
            w_old, w, theta,
            implicit=(True if imex == "none" else False)
        )
        F += inner(w_flux * rotate(u_adv), v) * dx
    else:
        F -= inner(div(rotate(u_flux)) * rotate(u_adv), v) * dx
        F += inner(
            dot(rotate(u_flux("+") - u_flux("-")), n("+")) * rotate(avg(u_adv)),
            avg(v)
        ) * dS
    F -= inner(1/2 * dot(u_flux, u_adv), div(v)) * dx

    # Dissipation, friction & weakly-enforced BCs
    if scheme == "meevc":
        F += 1 / Re * inner(_curl_s(w_mid), v) * dx
    else:
        F += 2 / Re * (
            h1_sym(u_mid, v)
          + friction(u_mid, v)
        )
        if not strong_bcs:
            F -= (
                inner(s_mid, dot(v, n))
              + inner(dot(u - u_BC, n), r)
            ) * dC

    # Pressure & incompressibility
    F -= (
        inner(p_mid, div(v))
      + inner(div(u), q)
    ) * dx
    if not strong_bcs:
        F += inner(p_mid, dot(v, n)) * ds  
    if traction_outlet:
        F += inner(p_out + 1/2 * dot(u_flux, u_adv), dot(v, n)) * ds(OUTLET)

    # Vorticity reconstruction
    if scheme == "asf":
        F += 1 / Re * inner(_curl_s(w), _curl_s(c)) * dx
        F -= 2 / Re * (
            h1_sym(u, _curl_s(c))
          + friction(u, _curl_s(c))
        )
        if not strong_bcs:
            F += inner(s + 1/2 * dot(u, u), dot(_curl_s(c), n)) * dC
        if traction_outlet:
            F -= inner(p_out, dot(_curl_s(c), n)) * ds(OUTLET)
    elif scheme == "meevc":
        F += inner(w, c) * dx
        F -= inner(u, _curl_s(c)) * dx

    # Forcing
    if forcing is not None:
        F -= inner(forcing, v[0]) * dx

    # Time loop
    try:
        t = 0.0
        for step in tqdm(range(1, round(T / float(dt)) + 1), desc="Vortex street (2D)"):
            t += float(dt)
            parprint(GREEN % f"Solving for t = {t:.4e}:")
            solve(
                F == 0,
                z,
                bcs=bcs,
                nullspace=nullspace,
                solver_parameters=solver_params
            )

            record_data(t, step)
            if save_vtk and step % save_freq == 0:
                recorder.write_vtk(time=t)
        
            _update_vals()
    finally:
        recorder.save_json()

    if download:
        recorder.download_zip()


def main():
    """CLI entry point with default parameters."""
    vortex_street_2d()


if __name__ == "__main__":
    main()
