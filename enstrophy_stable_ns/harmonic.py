"""Harmonic 1-forms of the discrete de Rham complex."""

from firedrake import *


def harmonic_1forms(mesh, flux_markers, degree=1, markers="on_boundary",
                    solver_parameters=None):
    """Basis for h^1_0, one linear solve per element.
    
    ``flux_markers`` gives one boundary marker per harmonic form.
    """
    from enstrophy_stable_ns.common import real_block_params

    flux_markers = list(flux_markers)
    d = len(flux_markers)
    A = FunctionSpace(mesh, "CG", degree)
    W = FunctionSpace(mesh, "N1curl", degree)
    R = FunctionSpace(mesh, "R", 0)
    Z = MixedFunctionSpace([A, W] + [R] * d)

    ds_ = Measure("ds", domain=mesh)
    n = FacetNormal(mesh)
    areas = [assemble(Constant(1.0) * ds_(m)) for m in flux_markers]
    for m, ar in zip(flux_markers, areas):
        if ar <= 0:
            raise ValueError(
                f"Flux marker {m} has zero area: it is glued by periodicity or "
                "absent, so it cannot normalise a harmonic form."
            )

    basis = []
    for i in range(d):
        z = Function(Z)
        fields, tests = split(z), TestFunctions(Z)
        sig, w = fields[0], fields[1]
        tau, chi = tests[0], tests[1]
        lam, mu = fields[2:], tests[2:]

        F = (inner(sig, tau) - inner(w, grad(tau))
             + inner(grad(sig), chi) + inner(curl(w), curl(chi))) * dx
        for j, (m, ar) in enumerate(zip(flux_markers, areas)):
            target = 1.0 if j == i else 0.0
            F += inner(lam[j], dot(chi, n)) * ds_(m)
            F += (inner(dot(w, n), mu[j])
                  - inner(Constant(target / ar), mu[j])) * ds_(m)

        bcs = []
        if markers is not None:
            bcs = [DirichletBC(Z.sub(0), 0.0, markers),
                   DirichletBC(Z.sub(1),
                               Constant((0.0,) * mesh.geometric_dimension),
                               markers)]
        solve(F == 0, z, bcs=bcs,
              solver_parameters=real_block_params(Z, solver_parameters or {}))
        f = Function(W)
        f.assign(z.subfunctions[1])
        basis.append(f)
    return basis


def harmonic_residuals(basis, mesh, markers="on_boundary"):
    """Diagnostics: how harmonic each returned form actually is."""
    n = FacetNormal(mesh)
    ds_ = Measure("ds", domain=mesh)
    out = []
    for h in basis:
        cn = sqrt(abs(assemble(inner(curl(h), curl(h)) * dx)))
        hn = sqrt(abs(assemble(inner(h, h) * dx)))
        tn = sqrt(abs(assemble(inner(cross(h, n), cross(h, n)) * ds_)))
        out.append((cn, hn, tn))
    return out
