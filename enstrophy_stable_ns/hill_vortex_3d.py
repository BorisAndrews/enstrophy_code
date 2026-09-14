"""3D Hill spherical vortex on a fully periodic unit cube.

Uses the H(grad curl)-free ('charlie') reparametrisation with
Alfeld-split elements. Compares:
  - 'asf': our enstrophy-stable scheme
  - 'rebholz': helicity-stable (Rebholz) scheme
  - 'standard': omega = curl(u) directly
"""

from firedrake import *
from scipy import special
from tqdm import tqdm
import numpy as np

from enstrophy_stable_ns.common import (
    dev,
    parprint,
    DataRecorder, DEFAULT_SOLVER_PARAMS,
)


# ---------------------------------------------------------------------------
#   Hill spherical vortex (analytical, in Cartesian coordinates)
# ---------------------------------------------------------------------------
# Bessel function parameters
bessel_J_root = 5.7634591968945506
bessel_J_root_threehalves = bessel_J(3/2, bessel_J_root)

# (r, theta, phi) components of Hill vortex
def hill_r(r, theta, radius):
    rho = r / radius
    return 2 * (
        bessel_J(3/2, bessel_J_root*rho) / rho**(3/2)
      - bessel_J_root_threehalves
    ) * cos(theta)

def hill_theta(r, theta, radius):
    rho = r / radius
    return (
        bessel_J_root * bessel_J(5/2, bessel_J_root*rho) / rho**(1/2)
      + 2 * bessel_J_root_threehalves
      - 2 * bessel_J(3/2, bessel_J_root*rho) / rho**(3/2)
    ) * sin(theta)

def hill_phi(r, theta, radius):
    rho = r / radius
    return bessel_J_root * (
        bessel_J(3/2, bessel_J_root*rho) / rho**(3/2)
      - bessel_J_root_threehalves
    ) * rho * sin(theta)

# Hill vortex (Cartesian)
def hill_vortex(vec, radius):
    (x, y, z) = vec

    # Cylindrical/spherical coordinates
    r_cyl = sqrt(x**2 + y**2)  # Cylindrical radius
    r_sph = sqrt(x**2 + y**2 + z**2)  # Spherical radius
    theta = conditional(  # Spherical angle
        le(r_cyl, 1e-13),
        0,
        pi/2 - atan(z/r_cyl)
    )

    return conditional(  # If we're outside the vortex...
        ge(r_sph, radius),
        as_vector([0, 0, 0]),
        conditional(  # If we're at the origin...
            le(r_sph, 1e-13),
            as_vector([0, 0, 2*((bessel_J_root/2)**(3/2)/special.gamma(5/2) - bessel_J_root_threehalves)]),
            conditional(  # If we're on the z axis...
                le(r_cyl, 1e-13),
                as_vector([0, 0, hill_r(r_sph, 0, radius)]),
                as_vector(  # Else...
                    hill_r(r_sph, theta, radius) * np.array([x, y, z]) / r_sph
                  + hill_theta(r_sph, theta, radius) * np.array([x*z, y*z, -r_cyl**2]) / r_sph / r_cyl
                  + hill_phi(r_sph, theta, radius) * np.array([-y, x, 0]) / r_cyl
                )
            )
        )
    )


# ---------------------------------------------------------------------------
#   Main simulation
# ---------------------------------------------------------------------------
def hill_vortex_3d(prob_params=None, mesh_params=None,
                   disc_params=None, save_params=None,
                   solver_params=None, check_dof=False):
    """Run a 3D Hill spherical vortex simulation on a periodic cube.

    Parameters
    ----------
    prob_params : dict
        Problem parameters:
        - ``Re`` (float): Reynolds number (default 1e4)
        - ``vortex_radius`` (float): Hill vortex radius (default 0.25)
    mesh_params : dict
        Mesh parameters:
        - ``n`` (int): mesh elements per direction (default 5)
    disc_params : dict
        Discretisation parameters:
        - ``k`` (int): polynomial order of velocity space (default 3)
        - ``k_ic`` (int): polynomial order of velocity space for ICs only (default k)
        - ``dt`` (float): timestep (default 1e-3)
        - ``T`` (float): final time (default 5e-1)
        - ``theta`` (float): implicit midpoint parameter (default 0.5)
        - ``scheme`` (str): ``"asf"``, ``"rebholz"``, or ``"standard"``
          (default ``"asf"``)
    save_params : dict
        - ``save_vtk`` (bool): enable VTK output (default True)
        - ``save_freq`` (int): write every N steps (default 1)
        - ``output_dir`` (str): output directory
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
    Re = Constant(prob_params.get("Re", 1e4))
    vortex_radius = prob_params.get("vortex_radius", 0.25)

    n = mesh_params.get("n", 5)
    lvls = mesh_params.get("lvls", 0)
    
    k = disc_params.get("k", 3)
    k_ic = disc_params.get("k_ic", k)
    dt = Constant(disc_params.get("dt", 1e-3))
    T = disc_params.get("T", 5e-1)
    theta = Constant(disc_params.get("theta", 0.5))
    scheme = disc_params.get("scheme", "asf").lower()
    if scheme not in {"asf", "rebholz", "standard"}:
        raise ValueError(f"Unknown scheme '{scheme}'. "
                         "Use 'asf', 'rebholz', or 'standard'.")

    save_vtk = save_params.get("save_vtk", True)
    save_freq = save_params.get("save_freq", 1)
    recorder = DataRecorder(
        output_dir=save_params.get(
            "output_dir", f"output/hill_vortex_3d/theta_{float(theta):g}/{scheme}"),
    )
    download = save_params.get("download", False)

    solver_params = DEFAULT_SOLVER_PARAMS | solver_params

    # ---------------
    #    Mesh
    # ---------------
    mesh = PeriodicUnitCubeMesh(n, n, n)
    x, y, z = SpatialCoordinate(mesh)

    # --------------------------
    #    Function spaces
    # --------------------------
    # Nullspaces
    nsp_const = VectorSpaceBasis(constant=True, comm=mesh.comm)
    nsp_none = VectorSpaceBasis([], comm=mesh.comm)

    # All spaces use variant="alfeld" to build on the Alfeld split
    if scheme == "asf":
        # Charlie reparametrisation (eq:charlie_3d):
        # Unknowns: (alpha, omega, eta, R, u, P, lambda)
        A = FunctionSpace(mesh, "CG", k + 2, variant="alfeld")     # scalar alpha
        W = FunctionSpace(mesh, "N2curl", k + 1, variant="alfeld") # vector omega
        V = VectorFunctionSpace(mesh, "CG", k, variant="alfeld")   # vector eta, u
        Q = FunctionSpace(mesh, "DG", k - 1, variant="alfeld")     # scalar R, P
        R = FunctionSpace(mesh, "R", 0)                            # Lagrange multiplier lambda

        Z = MixedFunctionSpace([A, W, V, Q, V, Q, R, R, R])

        z_fn = Function(Z)
        alpha, omega, eta, R, u, P, lam_x, lam_y, lam_z = split(z_fn)
        lam = as_vector([lam_x, lam_y, lam_z])
        (alpha_sol, omega_sol,
         eta_sol, R_sol,
         u_sol, P_sol) = z_fn.subfunctions[:6]

        tests = TestFunctions(Z)
        beta, chi, zeta, S, v_vec, Q_test, mu_x, mu_y, mu_z = tests
        mu = as_vector([mu_x, mu_y, mu_z])

        nullspace = MixedVectorSpaceBasis(Z, [nsp_const, nsp_none, nsp_none, nsp_const, nsp_none, nsp_const, nsp_none, nsp_none, nsp_none])

        # Rename for output
        alpha_sol.rename("alpha")
        omega_sol.rename("vorticity")
        eta_sol.rename("eta")
        u_sol.rename("velocity")
        P_sol.rename("bernoulli_pressure")

    elif scheme == "rebholz":
        # Helicity-stable (Rebholz): omega in same space as u (vector CG)
        # Unknowns: (omega, u, P)
        V = VectorFunctionSpace(mesh, "CG", k, variant="alfeld")
        Q = FunctionSpace(mesh, "DG", k - 1, variant="alfeld")
        Z = MixedFunctionSpace([V, Q, V, Q])

        z_fn = Function(Z)
        omega, alpha, u, P = split(z_fn)
        omega_sol, alpha_sol, u_sol, P_sol = z_fn.subfunctions

        tests = TestFunctions(Z)
        chi, beta, v_vec, Q_test = tests

        nullspace = MixedVectorSpaceBasis(Z, [nsp_none, nsp_const, nsp_none, nsp_const])

        omega_sol.rename("vorticity")
        u_sol.rename("velocity")
        P_sol.rename("bernoulli_pressure")

    else:  # standard
        # omega = curl(u), no auxiliary variable
        # Unknowns: (u, P)
        V = VectorFunctionSpace(mesh, "CG", k, variant="alfeld")
        Q = FunctionSpace(mesh, "DG", k - 1, variant="alfeld")
        Z = MixedFunctionSpace([V, Q])

        z_fn = Function(Z)
        u, P = split(z_fn)
        u_sol, P_sol = z_fn.subfunctions

        tests = TestFunctions(Z)
        v_vec, Q_test = tests

        nullspace = MixedVectorSpaceBasis(Z, [nsp_none, nsp_const])

        u_sol.rename("velocity")
        P_sol.rename("bernoulli_pressure")

    if check_dof:
        parprint(RED % f"DoF check: {[Z_.dim() for Z_ in Z]} | Total: {Z.dim()}")
        return

    # Previous-step velocity
    V_prev = VectorFunctionSpace(mesh, "CG", k, variant="alfeld")
    u_old = Function(V_prev)

    # ---------------------------------
    #    Initial condition
    # ---------------------------------
    # Function spaces (potentially lower order)
    V_ic = VectorFunctionSpace(mesh, "CG", k_ic, variant="alfeld")
    Q_ic = FunctionSpace(mesh, "DG", k_ic-1, variant="alfeld")
    Z_ic = MixedFunctionSpace([V_ic, Q_ic])

    # Functions (incl. target function)
    u_vortex = interpolate(
        hill_vortex((x - 0.5, y - 0.5, z - 0.5), vortex_radius),
        V_ic
    )
    z_fn_ic = Function(Z_ic)
    u_ic, p_ic = split(z_fn_ic)
    u_ic_sol, _ = z_fn_ic.subfunctions
    v_ic, q_ic = TestFunctions(Z_ic)

    # Residual for Stokes projection
    F_ic = (
        inner(u_ic - u_vortex, v_ic)
      - inner(p_ic, div(v_ic))
      - inner(div(u_ic), q_ic)
    ) * dx

    # Nullspace
    nullspace_ic = MixedVectorSpaceBasis(Z_ic, [nsp_none, nsp_const])

    # Solve projection
    parprint(GREEN % "Setting up initial condition:")
    solve(F_ic == 0, z_fn_ic, nullspace=nullspace_ic, solver_parameters=solver_params)

    # Normalise to unit energy
    sqrt_u_energy = sqrt(assemble(0.5 * inner(u_ic, u_ic) * dx))
    u_mean = as_vector([assemble(u_ic[i] * dx) for i in range(3)])
    u_old.interpolate((u_ic_sol - u_mean) / sqrt_u_energy)
    u_sol.assign(u_old)

    # --------------------------
    #    Build residual
    # --------------------------
    u_mid = theta * u + (1 - theta) * u_old

    if scheme == "asf":
        # Charlie reparametrisation (eq:charlie_3d)
        # (1) (curl omega, curl chi) - (eta, curl chi) + (grad alpha, chi) = 0
        eps = Constant(1e-8)
        vorticity_recovery = (
            inner(curl(omega), curl(chi))
            + eps * inner(omega, chi)
            - inner(eta, curl(chi))
            + inner(grad(alpha), chi)
        ) * dx

        # (2) (omega, grad beta) = 0  (adjoint div-free constraint)
        omega_divfree = inner(omega, grad(beta)) * dx

        # (3) (eta, zeta) - 2 (dev(u_mid), dev(zeta)) - (R, div zeta) = 0
        eta_eq = (
            inner(eta, zeta)
            - 2 * inner(dev(u_mid), dev(zeta))
            - inner(R, div(zeta))
        ) * dx

        # (4) - (div eta, S) = 0
        eta_div = - inner(div(eta), S) * dx

        # (5) 1/dt*(u - u_old, v) - (cross(u_mid, omega), v) + 1/Re*(eta, v) - (P, div v) = 0
        momentum = (
            1 / dt * inner(u - u_old, v_vec)
            - inner(cross(u_mid, omega), v_vec)
            + 1 / Re * inner(eta, v_vec)
            - inner(P, div(v_vec))
        ) * dx

        # (6) - (div u, Q) = 0
        incomp = - inner(div(u), Q_test) * dx

        # (7) Lagrange multiplier for zero mean on omega
        mean_constraint = (
            inner(lam, chi)
            + inner(omega, mu)
        ) * dx

        F = vorticity_recovery + omega_divfree + eta_eq + eta_div + momentum + incomp + mean_constraint

    elif scheme == "rebholz":
        vorticity_eq = (
            inner(omega, chi)
            - 0.5 * inner(u_mid, curl(chi))
            - 0.5 * inner(curl(u_mid), chi)
            - inner(alpha, div(chi))
        ) * dx

        omega_divfree = - inner(div(omega), beta) * dx

        momentum = (
            1 / dt * inner(u - u_old, v_vec)
            - inner(cross(u_mid, omega), v_vec)
            + 2 / Re * inner(dev(u_mid), dev(v_vec))
            - inner(P, div(v_vec))
        ) * dx

        incomp = - inner(div(u), Q_test) * dx

        F = vorticity_eq + omega_divfree + momentum + incomp

    else:  # standard
        # omega = curl(u) directly; standard velocity-pressure formulation
        momentum = (
            1 / dt * inner(u - u_old, v_vec)
            - inner(cross(u_mid, curl(u_mid)), v_vec)
            + 2 / Re * inner(dev(u_mid), dev(v_vec))
            - inner(P, div(v_vec))
        ) * dx

        incomp = - inner(div(u), Q_test) * dx

        F = momentum + incomp

    # ----------------
    #    VTK setup
    # ----------------
    if save_vtk:
        recorder.setup_vtk("fields", u_sol, P_sol)

    # ----------------
    #    Data output
    # ----------------
    def record_data(t_val, step_val):
        energy = 0.5 * assemble(inner(u, u) * dx)
        enstrophy = assemble(inner(dev(u), dev(u)) * dx)
        helicity = 0.5 * assemble(inner(u, curl(u)) * dx)
        div_err = assemble(inner(div(u), div(u)) * dx)
        if scheme == "asf":
            alpha_err = assemble(inner(grad(alpha), grad(alpha)) * dx)
            omega_err = assemble(inner(curl(omega) - eta, curl(omega) - eta) * dx)
            parprint(RED % f"Energy: {energy:.4e} | Enstrophy: {enstrophy:.4e} | "
                           f"Helicity: {helicity:.4e} | ||div u||^2: {div_err:.4e} | "
                           f"||grad(alpha)||^2: {alpha_err:.4e} | ||curl(omega) - eta||^2: {omega_err:.4e}")
            recorder.record(time=t_val, step=step_val, energy=energy,
                            enstrophy=enstrophy, helicity=helicity,
                            divergence_error=div_err, alpha_error=alpha_err,
                            omega_error=omega_err)
        else:
            parprint(RED % f"Energy: {energy:.4e} | Enstrophy: {enstrophy:.4e} | "
                           f"Helicity: {helicity:.4e} | ||div u||^2: {div_err:.4e}")
            recorder.record(time=t_val, step=step_val, energy=energy,
                            enstrophy=enstrophy, helicity=helicity,
                            divergence_error=div_err)
    
    record_data(0.0, 0)
    if save_vtk:
        recorder.write_vtk()

    # ----------------
    #    Time loop
    # ----------------
    t = 0.0
    step = 0
    num_steps = round(T / float(dt))
    for step in tqdm(range(1, num_steps + 1), desc="Hill vortex 3D"):
        t += float(dt)
        parprint(GREEN % f"Solving for t = {t:.4e}:")
        if scheme == "asf":
            # No solver parameters, so that Firedrake eliminates the Real
            # blocks itself; the explicit elimination stalls here.
            solve(F == 0, z_fn, nullspace=nullspace)
        else:
            solve(F == 0, z_fn, nullspace=nullspace, solver_parameters=solver_params)
        record_data(t, step)
        if save_vtk and step % save_freq == 0:
            recorder.write_vtk()
        u_old.assign(u_sol)
        recorder.save_json()

    if download: recorder.download_zip()


def main():
    """CLI entry point with default parameters."""
    hill_vortex_3d()


if __name__ == "__main__":
    main()
