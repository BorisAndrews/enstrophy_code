"""3D turbulent wake behind a spherical obstacle.

Uses the penalty method with a de Rham complex (CG / N1curl / BDM / DG).
Compares the enstrophy-stable scheme ('asf'), MEEVC, and a classical
unstabilised discretisation ('standard').
"""

import json
import os
import numpy as np
from firedrake import *
from firedrake.exceptions import ConvergenceError
from tqdm import tqdm
from enstrophy_stable_ns.meshes import (
    rect_with_ball_3d,
    _check_periodic_direction,
)
from enstrophy_stable_ns.harmonic import harmonic_1forms
from enstrophy_stable_ns.common import (
    parprint,
    DataRecorder, DEFAULT_SOLVER_PARAMS, LU_PARAMS, real_block_params,
    dev,
    constant_nullspace,
    h1_sym_build, friction_build,
    midpoint_function, time_derivative,
)

# ---------------------------------------------------------------------------
#   Discrete spaces
# ---------------------------------------------------------------------------
def _function_spaces(mesh, k, has_vorticity, has_alpha, n_harm=0,
                     complex="hybrid"):
    """Mixed space for the requested scheme.

    ``complex`` is ``"hybrid"``, ``CG_{k+1} -> NED1_{k+1} -> BDM_k -> DG_{k-1}``,
    or ``"standard"``, ``CG_{k+2} -> NED2_{k+1} -> BDM_k -> DG_{k-1}``.
    """
    if complex not in {"hybrid", "standard"}:
        raise ValueError(f"Unknown complex '{complex}'. "
                         "Use 'hybrid' or 'standard'.")
    hybrid = (complex == "hybrid")
    spaces = []
    if has_alpha:
        spaces.append(FunctionSpace(mesh, "CG", k + 1 if hybrid else k + 2))
    if has_vorticity:
        spaces.append(FunctionSpace(
            mesh, "N1curl" if hybrid else "N2curl", k + 1))   # omega
    spaces.append(FunctionSpace(mesh, "BDM", k))              # u
    spaces.append(FunctionSpace(mesh, "DG", k - 1))           # p
    spaces += [FunctionSpace(mesh, "R", 0) for _ in range(n_harm)]
    return MixedFunctionSpace(spaces)


def _name_fields(z, has_vorticity, has_alpha):
    """Label the subfunctions for VTK output."""
    names = (
        (["alpha"] if has_alpha else [])
      + (["vorticity"] if has_vorticity else [])
      + ["velocity", "pressure"]
    )
    for field, name in zip(z.subfunctions, names):
        field.rename(name)

# ---------------------------------------------------------------------------
#   Driver
# ---------------------------------------------------------------------------
def vortex_street_3d(prob_params=None, mesh_params=None,
                     disc_params=None, save_params=None,
                     solver_params=None, check_dof=False):
    """Run a 3D turbulent wake simulation.

    Parameters
    ----------
    prob_params : dict
        Problem parameters:
        - ``Re`` (float): Reynolds number (default 5e2)
        - ``gamma_inlet`` (float): friction on the inlet (default 2e2)
        - ``gamma_obstacle`` (float): friction on the obstacle (default 2e1)
        - ``outlet`` (str): ``"slip"`` (default) or ``"traction"``, as in 2D
        - ``p_out`` (float): outlet pressure for ``"traction"`` (default 0)
        - ``forcing`` (float): streamwise body force (default 2e-3 under x
          periodicity, otherwise None)

    mesh_params : dict
        Passed to ``rect_with_ball_3d``. Keys include ``r``, ``left``,
        ``right``, ``updown``, ``h_fine``, ``h_coarse`` and
        ``periodic_direction`` (``"none"`` or ``"x"``, default ``"x"``).

    disc_params : dict
        Discretisation parameters:
        - ``k`` (int): polynomial order (default 2)
        - ``scheme`` (str): ``"standard"``, ``"meevc"`` or ``"asf"``
          (default ``"asf"``)
        - ``complex`` (str): ``"hybrid"`` (default) or ``"standard"``
        - ``imex`` (str): ``"none"`` (default), ``"partial"`` or ``"full"``
        - ``stages`` (int): 1, or 2 for BDF2 with ``theta=1`` (default 2)
        - ``aux_advection`` (bool): advect with the auxiliary vorticity rather
          than ``curl u`` (default True; ignored by ``"standard"``)
        - ``sigma`` (float): interior penalty parameter (default 5e1)
        - ``sigma_vort_scale`` (float): the ratio sigma_bar / sigma of the
          reduced penalty in the vorticity reconstruction (default 0.3)
        - ``epsilon`` (float): the delta stabilisation; ``None`` imposes the
          constraints with Lagrange multipliers instead (default 1e-5)
        - ``stokes_ic`` (bool): start from the steady Stokes flow rather than
          from rest (default True)
        - ``dt`` (float): timestep, or initial timestep if adaptive
          (default 1e-1)
        - ``adaptive`` (bool): choose the timestep from Newton's iteration
          count (default True), within ``dt_min`` (default 1e-4) and
          ``dt_max`` (default 2e-1)
        - ``extrapolate`` (bool): extrapolate the previous two steps for
          Newton's initial guess (default True)
        - ``T`` (float): final time (default 5e1)
        - ``theta`` (float): time-stepping parameter (default 1.0)

    save_params : dict
        - ``save_vtk`` (bool): enable VTK output (default True)
        - ``save_freq`` (int): write every N steps (default 2)
        - ``checkpoint_freq`` (int): checkpoint every N steps, 0 for never
          (default 5)
        - ``restart`` (bool): resume from ``checkpoint.npz`` in the output
          directory if present (default False)
        - ``output_dir`` (str): output directory
          (default "output/vortex_street_3d/<scheme>")
        - ``download`` (bool): download output as zip for Colab (default False)

    solver_params : dict
        PETSc/Firedrake nonlinear solver parameters.
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
    Re = Constant(prob_params.get("Re", 5e2))
    gamma_inlet = Constant(prob_params.get("gamma_inlet", 2e2))
    gamma_obstacle = Constant(prob_params.get("gamma_obstacle", 2e1))
    outlet = str(prob_params.get("outlet", "slip")).lower()
    if outlet not in {"slip", "traction"}:
        raise ValueError(f"Unknown outlet '{outlet}'. Use 'slip' or 'traction'.")
    p_out = Constant(prob_params.get("p_out", 0.0))
    forcing = prob_params.get("forcing", None)

    # Mesh parameters
    mesh_params.setdefault("periodic_direction", "x")
    r_obs = mesh_params.get("r", 0.5)
    updown = mesh_params.get("updown", 1.0)
    # Defaults of rect_with_ball_3d, for the flux diagnostic
    left = mesh_params.get("left", 1.0)
    right = mesh_params.get("right", 5.0)
    periodic = _check_periodic_direction(
        mesh_params["periodic_direction"], dim=3)
    periodic_x, periodic_y, periodic_z = periodic
    if periodic_y or periodic_z:
        raise NotImplementedError(
            "Periodicity in y or z not yet implemented."
        )

    # Discretisation parameters
    stokes_ic = disc_params.get("stokes_ic", True)
    k = disc_params.get("k", 2)
    scheme = disc_params.get("scheme", "asf").lower()
    if scheme not in {"standard", "meevc", "asf"}:
        raise ValueError(f"Unknown scheme '{scheme}'. Use 'standard', 'meevc' or 'asf'.")
    imex = disc_params.get("imex", "none").lower()
    if imex not in {"none", "partial", "full"}:
        raise ValueError(f"Unknown imex '{imex}'. Use 'none', 'partial' or 'full'.")
    stages = disc_params.get("stages", 2)
    if stages not in {1, 2}:
        raise ValueError(
            "1 and 2 stage methods implemented only. "
            f"Staged '{stages}' not yet available."
        )
    aux_advection = disc_params.get("aux_advection", True)
    strong_bcs = disc_params.get("strong_bcs", True)
    complex = disc_params.get("complex", "hybrid")
    extrapolate = disc_params.get("extrapolate", True)
    # Extrapolating Newton's initial guess needs two previous steps, even for
    # a one-stage method
    n_old = max(stages, 2) if extrapolate else stages
    adaptive = disc_params.get("adaptive", True)
    # Newton iteration cap under adaptivity, just above the shrink threshold,
    # so that an expensive step is rejected rather than ground out
    snes_max_it = disc_params.get("snes_max_it", 6)
    epsilon = disc_params.get("epsilon", 1e-5)
    # Whether the delta stabilisation carries its right-hand side delta (u, curl chi)
    epsilon_rhs = disc_params.get("epsilon_rhs", True)
    # Scalings of the penalty and friction coefficients in the vorticity
    # reconstruction only (sigma_bar / sigma and gamma_bar / gamma)
    sigma_vort_scale = disc_params.get("sigma_vort_scale", 0.3)
    gamma_vort_scale = disc_params.get("gamma_vort_scale", 1.0)
    dt_min = disc_params.get("dt_min", 1e-4)
    dt_max = disc_params.get("dt_max", 2e-1)
    if not strong_bcs:
        raise NotImplementedError(
            "Weakly imposed boundary conditions not yet implemented."
        )
    sigma = Constant(disc_params.get("sigma", 5e1))
    penalty_degree = disc_params.get("penalty_degree", None)
    dt = Constant(disc_params.get("dt", 1e-1))
    T = disc_params.get("T", 5e1)
    theta = Constant(disc_params.get("theta", 1.0))

    # Save parameters
    save_vtk = save_params.get("save_vtk", True)
    save_freq = round(save_params.get("save_freq", 2))
    checkpoint_freq = round(save_params.get("checkpoint_freq", 5))
    restart = save_params.get("restart", False)
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
    has_vorticity = (scheme in {"meevc", "asf"})
    # The delta term makes the vorticity block nonsingular, replacing both the
    # alpha multiplier and the harmonic constraint
    has_alpha = (scheme == "asf" and epsilon is None)
    if forcing is not None:
        forcing = Constant(forcing)
    if periodic_x and (forcing is None):
        forcing = Constant(2e-3)

    # ---------------------
    #    Set up recorder
    # ---------------------
    recorder = DataRecorder(
        output_dir=save_params.get(
            "output_dir", f"output/vortex_street_3d/{scheme}"),
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
            n_old=n_old,
            imex=imex,
            periodic_direction=mesh_params["periodic_direction"],
            complex=complex,
            extrapolate=extrapolate,
            epsilon=epsilon,
            epsilon_rhs=epsilon_rhs,
            stokes_ic=stokes_ic,
            snes_max_it=snes_max_it,
            sigma_vort_scale=sigma_vort_scale,
            gamma_vort_scale=gamma_vort_scale,
            penalty_degree=penalty_degree,
            aux_advection=aux_advection,
            strong_bcs=strong_bcs,
            outlet=outlet
        )
    )

    # ---------------------------
    #    Function spaces
    # ---------------------------
    # Mesh
    mesh = rect_with_ball_3d(**mesh_params)
    x_c, y_c, z_c = SpatialCoordinate(mesh)
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)

    # Measures
    dx = Measure("dx", domain=mesh)
    ds = Measure("ds", domain=mesh)
    dS = Measure("dS", domain=mesh)

    # Boundary ids: 1 = inlet, 2 = outlet, 3 = walls, 4 = obstacle.
    INLET, OUTLET, WALL, OBSTACLE = 1, 2, 3, 4
    weak_ids = ([] if periodic_x else [INLET]) + [OBSTACLE]
    friction_ids = ([] if periodic_x else [(INLET, gamma_inlet)]) + [(OBSTACLE, gamma_obstacle)]
    strong_ids = ([] if (periodic_y or periodic_z) else [WALL]) + ([] if (periodic_x or traction_outlet) else [OUTLET])

    # Harmonic 1-forms: constrained without delta, otherwise diagnostic only
    harm_diag = []
    if scheme == "asf":
        harm_diag = harmonic_1forms(mesh, [OBSTACLE], degree=k + 1)
        parprint(RED % f"Harmonic 1-forms: {len(harm_diag)}"
                       f"{' (diagnostic only; eps regularised)' if epsilon is not None else ''}")
    harm = harm_diag if epsilon is None else []
    # Function spaces
    Z = _function_spaces(mesh, k, has_vorticity, has_alpha, n_harm=len(harm),
                         complex=complex)

    # Solver parameters
    solver_params = {"snes_rtol": 1e-6} | solver_params
    omega_shift = solver_params.pop("omega_shift", 1e-3 / float(Re))
    pressure_shift = solver_params.pop("pressure_shift", 1e-6)
    if harm:
        solver_params = real_block_params(
            Z, DEFAULT_SOLVER_PARAMS | solver_params, regularised=True)
    else:
        # MUMPS ignores pc_factor_shift_type; ICNTL(24) detects null pivots
        solver_params = (LU_PARAMS | {"mat_mumps_icntl_24": 1}
                         | DEFAULT_SOLVER_PARAMS | solver_params)

    # DoF check
    if check_dof:
        parprint(RED % f"DoF check: {[Z_.dim() for Z_ in Z]} | Total: {Z.dim()}")
        return

    # ---------------
    #    Functions
    # ---------------
    # Component indices
    i_a = (0 if has_alpha else None)  # alpha
    i_w = ((1 if has_alpha else 0) if has_vorticity else None)  # omega
    i_u = (1 if has_alpha else 0) + (1 if has_vorticity else 0)  # u
    i_p = i_u + 1  # p
    i_h = (i_p + 1 if scheme == "asf" else None)  # harmonic multipliers

    # Shifts for RegularisedBlockPC, small relative to their blocks
    appctx = dict(
        omega_shift=Constant(omega_shift),
        pressure_shift=Constant(pressure_shift),
        omega_index=i_w,
        pressure_index=i_p,
    ) if harm else {}

    # Solutions
    z = Function(Z)
    fields = split(z)
    if has_alpha:
        a = fields[i_a]
    if has_vorticity:
        w = fields[i_w]
    u, p = fields[i_u], fields[i_p]
    lam_h = fields[i_h:i_h + len(harm)] if harm else []
    _name_fields(z, has_vorticity, has_alpha)

    # Tests
    tests = TestFunctions(Z)
    if has_alpha:
        b = tests[i_a]
    if has_vorticity:
        c = tests[i_w]
    v, q = tests[i_u], tests[i_p]
    mu_h = tests[i_h:i_h + len(harm)] if harm else []

    # Outputs
    if has_alpha:
        a_sol = z.subfunctions[i_a]
    if has_vorticity:
        w_sol = z.subfunctions[i_w]
    u_sol, p_sol = z.subfunctions[i_u], z.subfunctions[i_p]

    # Previous-step values
    z_old = [Function(Z) for _ in range(n_old)]
    if has_vorticity:
        w_old = [z_old[stage_].subfunctions[i_w] for stage_ in range(n_old)]
    u_old, p_old = [z_old[stage_].subfunctions[i_u] for stage_ in range(n_old)], [z_old[stage_].subfunctions[i_p] for stage_ in range(n_old)]

    # The most recent `stages` of them, which the time discretisation uses
    w_sch = w_old[n_old - stages:] if has_vorticity else None
    u_sch, p_sch = u_old[n_old - stages:], p_old[n_old - stages:]

    # Midpoint values (for momentum equation)
    if has_vorticity:
        w_mid = midpoint_function(w_sch, w, theta)
    u_mid, p_mid = midpoint_function(u_sch, u, theta), midpoint_function(p_sch, p, theta)

    # -------------------------
    #    Boundary conditions
    # -------------------------
    # Construct u_BC
    off_obstacle = conditional(lt(x_c**2 + y_c**2 + z_c**2, ((updown + r_obs) / 2)**2), 0.0, 1.0)
    u_BC = off_obstacle * as_vector([1.0, 0.0, 0.0])

    # Initialise
    bcs = []

    # Walls & outlet
    omega_bcs = (  # omega
        [DirichletBC(Z.sub(i_w), Constant((0.0, 0.0, 0.0)), index) for index in strong_ids]
        if has_vorticity else []
    )
    bcs.extend(omega_bcs)
    for index in strong_ids:  # u
        bcs.append(DirichletBC(Z.sub(i_u), u_BC, index))

    # Inlet & obstacle
    if has_vorticity:  # omega
        if INLET in weak_ids:
            bcs.append(EquationBC(
                inner(cross(w, n) + 2 * gamma_inlet * u, cross(c, n)) * ds(INLET) == 0,
                z, INLET, V=Z.sub(i_w), bcs=omega_bcs
            ))
        bcs.append(EquationBC(
            inner(cross(w, n) + 2 * (gamma_obstacle + 1 / r_obs) * u, cross(c, n)) * ds(OBSTACLE) == 0,
            z, OBSTACLE, V=Z.sub(i_w), bcs=omega_bcs)
        )
    for index in weak_ids:  # u
        bcs.append(DirichletBC(Z.sub(i_u), u_BC, index))

    # Everywhere (alpha)
    if has_alpha:
        bcs.append(DirichletBC(Z.sub(i_a), Constant(0.0), "on_boundary"))

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
    h1_sym = h1_sym_build(n, h, sigma, dx, dS, facet_degree=penalty_degree)
    far_field = conditional(ge(x_c**2 + y_c**2 + z_c**2, (2 * r_obs)**2), 1.0, 0.0)
    h1_sym_far = h1_sym_build(n, h, sigma, dx, dS, mask=far_field,
                              facet_degree=penalty_degree)
    friction = friction_build(friction_ids, n, ds, degree=penalty_degree)
    # The same two forms with scaled coefficients, for the vorticity reconstruction
    h1_sym_vort = h1_sym_build(n, h, Constant(float(sigma) * sigma_vort_scale),
                               dx, dS, facet_degree=penalty_degree)
    friction_vort = friction_build(
        [(ind, gam * gamma_vort_scale) for (ind, gam) in friction_ids],
        n, ds, degree=penalty_degree)

    def _vorticity_block(u_state):
        """The omega (and alpha) subsystem, shared by the Stokes and transient solves."""
        G = inner(curl(w), curl(c)) * dx
        G -= 2 * (
            h1_sym_vort(u_state, curl(c))
          + friction_vort(u_state, curl(c))
        )
        if epsilon is not None:
            G += Constant(epsilon) * inner(w, c) * dx
            if epsilon_rhs:
                G -= Constant(epsilon) * inner(u_state, curl(c)) * dx
        else:
            G += (
                inner(grad(a), c)
              + inner(w, grad(b))
            ) * dx
        if traction_outlet:
            G -= Re * inner(p_out, dot(curl(c), n)) * ds(OUTLET)
        if harm:
            G += Re * _harmonic_terms(w)
        return G

    def _harmonic_terms(w_state):
        """Lagrange multiplier rows enforcing (omega, h_i) = 0."""
        G = 0
        for h_i, lam_i, mu_i in zip(harm, lam_h, mu_h):
            G += inner(lam_i * h_i, c) * dx
            G += inner(inner(w_state, h_i), mu_i) * dx
        return G

    # -----------------
    #    Data output
    # -----------------
    stress = 2 / Re * dev(u_mid) - p_mid * Identity(3)

    def record_data(t_val, step_val):
        energy = 0.5 * assemble(inner(u, u) * dx)
        enstrophy = assemble(h1_sym(u, u) + friction(u, u))
        enstrophy_int = assemble(h1_sym(u, u))
        enstrophy_far = assemble(h1_sym_far(u, u))
        # Broken curl of the velocity, integrated cellwise
        curl_norm = assemble(inner(curl(u), curl(u)) * dx)
        drag = - assemble(dot(as_vector([1, 0, 0]), dot(stress, n)) * ds(OBSTACLE))
        lift_y = - assemble(dot(as_vector([0, 1, 0]), dot(stress, n)) * ds(OBSTACLE))
        lift_z = - assemble(dot(as_vector([0, 0, 1]), dot(stress, n)) * ds(OBSTACLE))
        backflow = assemble(conditional(lt(dot(u, n), 0), - dot(u, n), 0) * ds(OUTLET))
        # Cross-section flux, independent of the section as u is divergence-free
        flux = assemble(u[0] * dx) / (left + right)
        data = dict(
            time=t_val, step=step_val, energy=energy, flux=flux,
            enstrophy=enstrophy, enstrophy_int=enstrophy_int, enstrophy_far=enstrophy_far,
            curl_norm=curl_norm, drag=drag, lift_y=lift_y, lift_z=lift_z,
            backflow=backflow
        )
        msg = (
            f"Energy: {energy:.4e} | Enstrophy: {enstrophy:.4e} | "
            f"Enstrophy (int): {enstrophy_int:.4e} | "
            f"Enstrophy (far): {enstrophy_far:.4e} | "
            f"Drag: {drag:.4e} | Lift(y): {lift_y:.4e} | Lift(z): {lift_z:.4e} | "
            f"Backflow: {backflow:.4e} | Flux: {flux:.4e} | "
            f"||curl u||^2: {curl_norm:.4e}"
        )
        if harm_diag:
            # Harmonic component of omega, raw and normalised by |omega| |h|
            w_norm = sqrt(abs(assemble(inner(w, w) * dx)))
            comps, rels = [], []
            for h_i in harm_diag:
                c = assemble(inner(w, h_i) * dx)
                h_norm = sqrt(abs(assemble(inner(h_i, h_i) * dx)))
                comps.append(abs(c))
                rels.append(abs(c) / (w_norm * h_norm) if w_norm * h_norm else 0.0)
            data.update(harmonic_residual=max(comps),
                        harmonic_component=max(rels))
            msg += (f" | (omega,h): {max(comps):.3e}"
                    f" | harm/|omega|: {max(rels):.3e}")
        if has_vorticity:
            # MEEVC dissipates ||omega||^2 / Re
            vorticity_norm = assemble(inner(w, w) * dx)
            data.update(vorticity_norm=vorticity_norm)
            msg += f" | ||omega||^2: {vorticity_norm:.4e}"
        if has_alpha:
            alpha_norm = sqrt(abs(assemble(inner(a, a) * dx)))
            data.update(alpha_norm=alpha_norm)
            msg += f" | ||alpha||: {alpha_norm:.4e}"
        parprint(RED % msg)
        recorder.record(**data)

    # -------------------------------
    #    Initial condition updater
    # -------------------------------
    def _update_vals():
        for stage_ in range(n_old - 1):
            if has_vorticity:
                w_old[stage_].assign(w_old[stage_ + 1])
            u_old[stage_].assign(u_old[stage_ + 1])
            p_old[stage_].assign(p_old[stage_ + 1])
        if has_vorticity:
            w_old[n_old - 1].assign(w_sol)
        u_old[n_old - 1].assign(u_sol)
        p_old[n_old - 1].assign(p_sol)

    # --------------------------
    #    Initial Stokes solve
    # --------------------------
    if stokes_ic:
        # Dissipation, friction & weakly-enforced BCs
        if scheme == "meevc":
            F = 1 / Re * inner(curl(w), v) * dx
        else:
            F = 2 / Re * (
                h1_sym(u, v)
              + friction(u, v)
            )

        # Pressure & incompressibility
        F -= (
            inner(p, div(v))
          + inner(div(u), q)
        ) * dx
        if traction_outlet:
            F += inner(p_out, dot(v, n)) * ds(OUTLET)

        # Forcing
        if forcing is not None:
            F -= inner(forcing, v[0]) * dx

        # Vorticity reconstruction
        if scheme == "asf":
            F += _vorticity_block(u)
        elif scheme == "meevc":
            F += (
                inner(w, c)
              - inner(u, curl(c))
            ) * dx

        # Solve
        parprint(GREEN % "Setting up ICs for t = 0:")
        solve(
            F == 0,
            z,
            bcs=bcs,
            nullspace=nullspace,
            solver_parameters=solver_params,
            appctx=appctx
        )

        # Update initial data
        for _ in range(n_old):
            _update_vals()

    # ------------------------------
    #    Full Navier-Stokes solve
    # ------------------------------
    # Transient, with the step ratio of variable-step BDF2
    ratio = Constant(1.0)
    u_dt = time_derivative(u_sch, u, theta,
                           ratio=(ratio if (adaptive and stages == 2) else None))
    F = 1 / dt * inner(u_dt, v) * dx

    # Advection
    u_adv = midpoint_function(
        u_sch, u, theta,
        implicit=(False if imex == "full" else True)
    )
    u_flux = midpoint_function(
        u_sch, u, theta,
        implicit=(True if imex == "none" else False)
    )
    if aux_advection:
        w_flux = midpoint_function(
            w_sch, w, theta,
            implicit=(True if imex == "none" else False)
        )
        F += inner(cross(w_flux, u_adv), v) * dx
    else:
        F += inner(cross(curl(u_flux), u_adv), v) * dx
        F -= inner(
            cross(avg(u_adv), cross(u_flux("+") - u_flux("-"), n("+"))),
            avg(v)
        ) * dS
    F -= inner(1/2 * dot(u_flux, u_adv), div(v)) * dx

    # Dissipation, friction & weakly-enforced BCs
    if scheme == "meevc":
        F += 1 / Re * inner(curl(w_mid), v) * dx
    else:
        F += 2 / Re * (
            h1_sym(u_mid, v)
          + friction(u_mid, v)
        )

    # Pressure & incompressibility
    F -= (
        inner(p_mid, div(v))
      + inner(div(u), q)
    ) * dx
    if traction_outlet:
        F += inner(p_out + 1/2 * dot(u_flux, u_adv), dot(v, n)) * ds(OUTLET)

    # Vorticity reconstruction
    if scheme == "asf":
        F += _vorticity_block(u)
    elif scheme == "meevc":
        F += (
            inner(w, c)
          - inner(u, curl(c))
        ) * dx

    # Forcing
    if forcing is not None:
        F -= inner(forcing, v[0]) * dx

    # ----------------------
    #    Checkpointing
    # ----------------------
    # Raw dof arrays, valid as the mesh is regenerated deterministically;
    # written to a temporary file and renamed, so never half-written
    ckpt = os.path.join(recorder.output_dir, "checkpoint.npz")

    def _save_checkpoint(t_val, step_val):
        arrays = {f"z{i}": f.dat.data_ro.copy()
                  for i, f in enumerate(z.subfunctions)}
        for stage_ in range(n_old):
            for i, f in enumerate(z_old[stage_].subfunctions):
                arrays[f"o{stage_}_{i}"] = f.dat.data_ro.copy()
        np.savez(ckpt + ".tmp.npz", t=t_val, step=step_val, **arrays)
        os.replace(ckpt + ".tmp.npz", ckpt)

    step_0, t_0 = 0, 0.0
    if restart and os.path.exists(ckpt):
        d = np.load(ckpt)
        for i, f in enumerate(z.subfunctions):
            f.dat.data[:] = d[f"z{i}"]
        for stage_ in range(n_old):
            for i, f in enumerate(z_old[stage_].subfunctions):
                f.dat.data[:] = d[f"o{stage_}_{i}"]
        step_0, t_0 = int(d["step"]), float(d["t"])
        # Keep the diagnostics recorded before the checkpoint
        prior = os.path.join(recorder.output_dir, "data.json")
        if os.path.exists(prior):
            with open(prior) as fh:
                recorder.data = [r for r in json.load(fh) if r["time"] < t_0]
            parprint(RED % f"Recovered {len(recorder.data)} earlier records")
        parprint(RED % f"Restarted from checkpoint at t = {t_0:.4e} "
                       f"(step {step_0})")

    # Output
    if step_0 == 0:
        record_data(0.0, 0)
    if save_vtk:
        fields_out = ((a_sol,) if has_alpha else ()) + ((w_sol,) if has_vorticity else ()) + (u_sol, p_sol)
        # A resumed run gets its own .pvd, so earlier frames are not overwritten
        recorder.setup_vtk("fields" if step_0 == 0 else f"fields_from_{step_0}",
                           *fields_out)
        recorder.write_vtk(time=t_0)

    # ------------------------------------
    #    Newton-count timestep adaptivity
    # ------------------------------------
    # Grow the step while Newton is cheap, shrink it when it is not, and
    # reject and retry a failed solve. Growth is capped below Grigorieff's
    # zero-stability bound r <= 1 + sqrt(2) for variable-step BDF2.
    dt0 = float(dt)
    dt_lo = dt_min if dt_min is not None else dt0 / 32
    dt_hi = dt_max if dt_max is not None else dt0 * 8
    GROW, SHRINK, REJECT = 1.4, 0.7, 0.5
    ITS_GROW, ITS_SHRINK = 2, 4

    def _adaptive_loop():
        """Time stepping with the step chosen from Newton's iteration count."""
        problem = NonlinearVariationalProblem(F, z, bcs=bcs)
        solver = NonlinearVariationalSolver(
            problem, nullspace=nullspace,
            solver_parameters=solver_params | {"snes_max_it": snes_max_it},
            appctx=appctx)

        z_backup = Function(Z)
        t_now, step_now, dt_prev = t_0, step_0, dt0
        dt_next = dt0
        pbar = tqdm(total=float(T), desc="Vortex street (3D, adaptive)")
        pbar.update(t_0)

        while t_now < T - 1e-12:
            dt_try = min(dt_next, T - t_now)
            z_backup.assign(z)
            while True:
                dt.assign(dt_try)
                ratio.assign(dt_try / dt_prev)
                if extrapolate:
                    r_ = dt_try / dt_prev
                    if has_vorticity:
                        w_sol.assign((1 + r_) * w_old[-1] - r_ * w_old[-2])
                    u_sol.assign((1 + r_) * u_old[-1] - r_ * u_old[-2])
                    p_sol.assign((1 + r_) * p_old[-1] - r_ * p_old[-2])
                try:
                    solver.solve()
                    its = solver.snes.getIterationNumber()
                    break
                except ConvergenceError:
                    z.assign(z_backup)
                    dt_try *= REJECT
                    if dt_try < dt_lo:
                        parprint(RED % f"  dt below {dt_lo:.3e}; giving up")
                        raise
                    parprint(RED % f"  step rejected, retrying at dt = {dt_try:.4e}")

            t_now += dt_try
            step_now += 1
            record_data(t_now, step_now)
            if save_vtk and step_now % save_freq == 0:
                recorder.write_vtk(time=t_now)
            _update_vals()
            if checkpoint_freq and step_now % checkpoint_freq == 0:
                _save_checkpoint(t_now, step_now)
                recorder.save_json()

            if its <= ITS_GROW:
                dt_next = min(dt_try * GROW, dt_hi)
            elif its >= ITS_SHRINK:
                dt_next = max(dt_try * SHRINK, dt_lo)
            else:
                dt_next = dt_try
            parprint(RED % f"  its={its}  dt {dt_try:.4e} -> {dt_next:.4e}")
            dt_prev = dt_try
            pbar.update(dt_try)
        pbar.close()

    # Time loop
    try:
        t = t_0
        if adaptive:
            _adaptive_loop()
        for step in ([] if adaptive else tqdm(range(step_0 + 1, round(T / float(dt)) + 1), desc="Vortex street (3D)")):
            t += float(dt)
            if extrapolate:
                # Alpha and the multipliers are not extrapolated
                if has_vorticity:
                    w_sol.assign(2 * w_old[-1] - w_old[-2])
                u_sol.assign(2 * u_old[-1] - u_old[-2])
                p_sol.assign(2 * p_old[-1] - p_old[-2])
            parprint(GREEN % f"Solving for t = {t:.4e}:")
            solve(
                F == 0,
                z,
                bcs=bcs,
                nullspace=nullspace,
                solver_parameters=solver_params,
                appctx=appctx
            )

            record_data(t, step)
            if save_vtk and step % save_freq == 0:
                recorder.write_vtk(time=t)

            _update_vals()
            if checkpoint_freq and step % checkpoint_freq == 0:
                _save_checkpoint(t, step)
                recorder.save_json()
    finally:
        recorder.save_json()

    if download:
        recorder.download_zip()


def main():
    """CLI entry point with default parameters."""
    vortex_street_3d()


if __name__ == "__main__":
    main()
