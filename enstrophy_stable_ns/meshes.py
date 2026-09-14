"""Mesh generation for enstrophy-stable Navier-Stokes simulations."""

import tempfile
import os
import gmsh
from firedrake import Mesh


# ---------------------------------------------------------------------------
#   Splits
# ---------------------------------------------------------------------------
def alfeld_split(mesh):
    """Apply Alfeld (barycentric) refinement to a simplicial mesh.

    Each simplex is subdivided into sub-simplices by connecting all
    vertices to the barycentre, yielding a mesh on which certain
    macro-element de Rham complexes are exact.

    Parameters
    ----------
    mesh : firedrake.Mesh
        Input simplicial mesh (triangles in 2D, tetrahedra in 3D).

    Returns
    -------
    firedrake.Mesh
        Alfeld-refined mesh.
    """
    from petsc4py.PETSc import DMPlexTransform, DMPlexTransformType

    dm = mesh.topology_dm
    tr = DMPlexTransform().create(comm=mesh.comm)
    tr.setType(DMPlexTransformType.REFINEALFELD)
    tr.setDM(dm)
    tr.setUp()
    return Mesh(tr.apply(dm))


# ---------------------------------------------------------------------------
#   Periodicity
# ---------------------------------------------------------------------------
def _check_periodic_direction(periodic_direction, dim=2):
    """Validate ``periodic_direction`` and split it into one flag per axis.

    Accepts ``"none"``, ``"all"``, any non-repeating subset of the axis letters
    for the dimension (``"x"``, ``"y"``, ``"xz"``, ...) and, in 2D only,
    ``"both"`` as an alias for ``"xy"``.
    """
    periodic_direction = str(periodic_direction).lower()
    axes = "xyz"[:dim]

    if periodic_direction == "none":
        letters = ""
    elif periodic_direction == "all":
        letters = axes
    elif periodic_direction == "both" and dim == 2:
        letters = axes
    elif (periodic_direction
          and set(periodic_direction) <= set(axes)
          and len(set(periodic_direction)) == len(periodic_direction)):
        letters = periodic_direction
    else:
        raise ValueError(
            f"Unknown periodic_direction '{periodic_direction}' in {dim}D. "
            f"Use 'none', 'all'"
            + (", 'both'" if dim == 2 else "")
            + f", or a combination of {tuple(axes)}."
        )

    flags = tuple(axis in letters for axis in axes)
    if all(flags):
        raise NotImplementedError(
            f"Periodicity in every direction leaves no boundary on which omega "
            f"is set strongly, so its level is free: the circulation has to be "
            f"fixed with a Lagrange multiplier, which is not yet implemented."
        )
    return flags


# ---------------------------------------------------------------------------
#   2D
# ---------------------------------------------------------------------------
def rect_with_hole_2d(r=0.5, left=1.0, right=5.0, updown=1.0,
                      h_fine=0.05, h_coarse=0.2, obstacle="circle",
                      periodic_direction="none"):
    """Generate a 2D rectangular mesh with a hole (obstacle).

    Parameters
    ----------
    r : float
        Obstacle radius / half-width.
    left : float
        Distance from left boundary to obstacle centre.
    right : float
        Distance from obstacle centre to right boundary.
    updown : float
        Distance from obstacle centre to top/bottom boundaries.
    h_fine : float
        Target mesh size near the obstacle.
    h_coarse : float
        Target mesh size in the far field.
    obstacle : str
        Shape of the obstacle: "circle", "semicircle", "diamond", "wedge", "square", or None.
    periodic_direction : str
        One of "none", "x" (inlet and outlet identified) or "y" (top and
        bottom identified). An identified pair keeps its physical group.

    Returns
    -------
    firedrake.Mesh
        Physical groups: inlet(1), outlet(2), wall(3), hole(4), surface(5).
    """
    periodic_x, periodic_y = _check_periodic_direction(periodic_direction)

    gmsh.initialize()
    gmsh.model.add("rect_with_hole")

    xmin, xmax = -left, right
    ymin, ymax = -updown, updown

    # Rectangle
    p1 = gmsh.model.geo.addPoint(xmin, ymin, 0, h_coarse)
    p2 = gmsh.model.geo.addPoint(xmax, ymin, 0, h_coarse)
    p3 = gmsh.model.geo.addPoint(xmax, ymax, 0, h_coarse)
    p4 = gmsh.model.geo.addPoint(xmin, ymax, 0, h_coarse)

    l_bottom = gmsh.model.geo.addLine(p1, p2)
    l_right = gmsh.model.geo.addLine(p2, p3)
    l_top = gmsh.model.geo.addLine(p3, p4)
    l_left = gmsh.model.geo.addLine(p4, p1)

    rect_loop = gmsh.model.geo.addCurveLoop([l_bottom, l_right, l_top, l_left])

    # Hole (centred at origin)
    hole_curves = []
    if obstacle == "circle":
        c = gmsh.model.geo.addPoint(0, 0, 0, h_coarse)
        cp = gmsh.model.geo.addPoint(r, 0, 0, h_coarse)
        cn = gmsh.model.geo.addPoint(-r, 0, 0, h_coarse)
        ct = gmsh.model.geo.addPoint(0, r, 0, h_coarse)
        cb = gmsh.model.geo.addPoint(0, -r, 0, h_coarse)
        a1 = gmsh.model.geo.addCircleArc(cp, c, ct)
        a2 = gmsh.model.geo.addCircleArc(ct, c, cn)
        a3 = gmsh.model.geo.addCircleArc(cn, c, cb)
        a4 = gmsh.model.geo.addCircleArc(cb, c, cp)
        hole_curves = [a1, a2, a3, a4]
    elif obstacle == "semicircle":
        c = gmsh.model.geo.addPoint(0, 0, 0, h_coarse)
        cp = gmsh.model.geo.addPoint(0, 0, 0, h_coarse)
        cn = gmsh.model.geo.addPoint(-r, 0, 0, h_coarse)
        ct = gmsh.model.geo.addPoint(0, r, 0, h_coarse)
        cb = gmsh.model.geo.addPoint(0, -r, 0, h_coarse)
        a1 = gmsh.model.geo.addLine(cp, ct)
        a2 = gmsh.model.geo.addCircleArc(ct, c, cn)
        a3 = gmsh.model.geo.addCircleArc(cn, c, cb)
        a4 = gmsh.model.geo.addLine(cb, cp)
        hole_curves = [a1, a2, a3, a4]
    elif obstacle == "diamond":
        cp = gmsh.model.geo.addPoint(r, 0, 0, h_coarse)
        cn = gmsh.model.geo.addPoint(-r, 0, 0, h_coarse)
        ct = gmsh.model.geo.addPoint(0, r, 0, h_coarse)
        cb = gmsh.model.geo.addPoint(0, -r, 0, h_coarse)
        a1 = gmsh.model.geo.addLine(cp, ct)
        a2 = gmsh.model.geo.addLine(ct, cn)
        a3 = gmsh.model.geo.addLine(cn, cb)
        a4 = gmsh.model.geo.addLine(cb, cp)
        hole_curves = [a1, a2, a3, a4]
    elif obstacle == "wedge":
        cp = gmsh.model.geo.addPoint(0, 0, 0, h_coarse)
        cn = gmsh.model.geo.addPoint(-r, 0, 0, h_coarse)
        ct = gmsh.model.geo.addPoint(0, r, 0, h_coarse)
        cb = gmsh.model.geo.addPoint(0, -r, 0, h_coarse)
        a1 = gmsh.model.geo.addLine(cp, ct)
        a2 = gmsh.model.geo.addLine(ct, cn)
        a3 = gmsh.model.geo.addLine(cn, cb)
        a4 = gmsh.model.geo.addLine(cb, cp)
        hole_curves = [a1, a2, a3, a4]
    elif obstacle == "square":
        cp = gmsh.model.geo.addPoint(r, r, 0, h_coarse)
        cn = gmsh.model.geo.addPoint(-r, -r, 0, h_coarse)
        ct = gmsh.model.geo.addPoint(-r, r, 0, h_coarse)
        cb = gmsh.model.geo.addPoint(r, -r, 0, h_coarse)
        a1 = gmsh.model.geo.addLine(cp, ct)
        a2 = gmsh.model.geo.addLine(ct, cn)
        a3 = gmsh.model.geo.addLine(cn, cb)
        a4 = gmsh.model.geo.addLine(cb, cp)
        hole_curves = [a1, a2, a3, a4]

    if obstacle is not None:
        hole_loop = gmsh.model.geo.addCurveLoop(hole_curves)
        surface = gmsh.model.geo.addPlaneSurface([rect_loop, hole_loop])
    else:
        surface = gmsh.model.geo.addPlaneSurface([rect_loop])

    gmsh.model.geo.synchronize()

    # Periodicity
    if periodic_x:
        gmsh.model.mesh.setPeriodic(
            1, [l_right], [l_left],
            [1, 0, 0, left + right,
             0, 1, 0, 0,
             0, 0, 1, 0,
             0, 0, 0, 1],
        )
    if periodic_y:
        gmsh.model.mesh.setPeriodic(
            1, [l_top], [l_bottom],
            [1, 0, 0, 0,
             0, 1, 0, 2 * updown,
             0, 0, 1, 0,
             0, 0, 0, 1],
        )

    # Physical groups
    gmsh.model.addPhysicalGroup(1, [l_left], tag=1)
    gmsh.model.setPhysicalName(1, 1, "inlet")
    gmsh.model.addPhysicalGroup(1, [l_right], tag=2)
    gmsh.model.setPhysicalName(1, 2, "outlet")
    gmsh.model.addPhysicalGroup(1, [l_top, l_bottom], tag=3)
    gmsh.model.setPhysicalName(1, 3, "wall")
    if obstacle is not None:
        gmsh.model.addPhysicalGroup(1, hole_curves, tag=4)
        gmsh.model.setPhysicalName(1, 4, "hole")
    gmsh.model.addPhysicalGroup(2, [surface], tag=5)

    # Mesh refinement near obstacle
    if obstacle is not None:
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "CurvesList", hole_curves)
        gmsh.model.mesh.field.setNumber(1, "Sampling", 100)

        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", h_fine)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", h_coarse)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 1.0 * r)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 2.0 * r)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)
    else:
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)

    # Generate and write mesh
    gmsh.model.mesh.generate(2)

    tmpfile = tempfile.NamedTemporaryFile(suffix=".msh", delete=False)
    tmpfile.close()
    gmsh.write(tmpfile.name)
    gmsh.finalize()

    mesh = Mesh(tmpfile.name)
    os.unlink(tmpfile.name)

    if periodic_x or periodic_y:
        import numpy as np
        Lx, Ly = left + right, 2 * updown
        mesh.topology_dm.setPeriodicity(
            np.array([Lx / 2 if periodic_x else 0.0,
                      Ly / 2 if periodic_y else 0.0]),      # maxCell
            np.array([-left if periodic_x else 0.0,
                      -updown if periodic_y else 0.0]),     # Lstart
            np.array([Lx if periodic_x else -1.0,
                      Ly if periodic_y else -1.0]),         # L
        )

    return mesh

def rect_with_hole_2d_netgen(r=0.5, left=1.0, right=5.0, updown=1.0,
                             h_fine=0.05, h_coarse=0.2, degree=3):
    """The domain of ``rect_with_hole_2d`` with a curved obstacle (Netgen).

    ``degree`` is the geometric order. Not used in the manuscript.
    """
    raise NotImplementedError(
        "Curved (Netgen) meshes are shelved: the vorticity recovery fails to "
        "converge on them. Use rect_with_hole_2d instead."
    )

    from netgen.occ import WorkPlane, OCCGeometry
    from firedrake import Mesh

    face = (
        WorkPlane().MoveTo(-left, -updown).Rectangle(left + right, 2 * updown).Face()
      - WorkPlane().MoveTo(0, 0).Circle(r).Face()
    )
    ngmesh = OCCGeometry(face, dim=2).GenerateMesh(maxh=h_coarse, minh=h_fine)
    mesh = Mesh(ngmesh)
    return Mesh(mesh.curve_field(degree)) if degree > 1 else mesh


# ---------------------------------------------------------------------------
#   3D
# ---------------------------------------------------------------------------
def rect_with_ball_3d(r=0.5, left=1.0, right=5.0, updown=1.0,
                      h_fine=0.2, h_coarse=1.0,
                      periodic_direction="none"):
    """Generate a 3D box mesh with a spherical hole (obstacle).

    Parameters
    ----------
    r : float
        Sphere radius.
    left : float
        Distance from left boundary (x=-left) to sphere centre.
    right : float
        Distance from sphere centre to right boundary (x=right).
    updown : float
        Distance from sphere centre to each of the y/z boundaries.
    h_fine : float
        Target mesh size near the obstacle.
    h_coarse : float
        Target mesh size in the far field.
    periodic_direction : str
        "none", or a combination of "x", "y", "z" naming the axes whose
        opposing faces are identified.

    Returns
    -------
    firedrake.Mesh
        Physical groups: inlet(1), outlet(2), wall(3), obstacle(4), volume(5).
    """
    periodic_x, periodic_y, periodic_z = _check_periodic_direction(
        periodic_direction, dim=3
    )

    gmsh.initialize()
    gmsh.model.add("rect_with_ball")

    xmin, xmax = -left, right
    ymin, ymax = -updown, updown
    zmin, zmax = -updown, updown

    # Box (using OpenCASCADE for boolean operations)
    box = gmsh.model.occ.addBox(xmin, ymin, zmin,
                                xmax - xmin, ymax - ymin, zmax - zmin)
    sphere = gmsh.model.occ.addSphere(0, 0, 0, r)

    # Boolean difference: box minus sphere
    result = gmsh.model.occ.cut([(3, box)], [(3, sphere)])
    gmsh.model.occ.synchronize()

    # Classify the surfaces geometrically; box faces are kept apart by axis so
    # that periodicity can pair them up
    surfaces = gmsh.model.getEntities(dim=2)
    faces = {"xmin": [], "xmax": [], "ymin": [], "ymax": [],
             "zmin": [], "zmax": []}
    obstacle_surfs = []

    for dim_tag in surfaces:
        tag = dim_tag[1]
        com = gmsh.model.occ.getCenterOfMass(2, tag)
        bb = gmsh.model.occ.getBoundingBox(2, tag)
        # bb = (xmin, ymin, zmin, xmax, ymax, zmax)
        if (bb[3] - bb[0] <= 2 * r + 0.01 and
            bb[4] - bb[1] <= 2 * r + 0.01 and
            bb[5] - bb[2] <= 2 * r + 0.01 and
            abs(com[0]) < r + 0.01 and
            abs(com[1]) < r + 0.01 and
            abs(com[2]) < r + 0.01):
            obstacle_surfs.append(tag)
        else:
            for key, (axis, value) in (
                ("xmin", (0, xmin)), ("xmax", (0, xmax)),
                ("ymin", (1, ymin)), ("ymax", (1, ymax)),
                ("zmin", (2, zmin)), ("zmax", (2, zmax)),
            ):
                if abs(com[axis] - value) < 0.01:
                    faces[key].append(tag)
                    break

    inlet_surfs = faces["xmin"]
    outlet_surfs = faces["xmax"]
    wall_surfs = (faces["ymin"] + faces["ymax"]
                + faces["zmin"] + faces["zmax"])

    # Periodicity
    for flag, source, target, shift in (
        (periodic_x, "xmin", "xmax", (left + right, 0.0, 0.0)),
        (periodic_y, "ymin", "ymax", (0.0, 2 * updown, 0.0)),
        (periodic_z, "zmin", "zmax", (0.0, 0.0, 2 * updown)),
    ):
        if flag:
            gmsh.model.mesh.setPeriodic(
                2, faces[target], faces[source],
                [1, 0, 0, shift[0],
                 0, 1, 0, shift[1],
                 0, 0, 1, shift[2],
                 0, 0, 0, 1],
            )

    # Physical groups
    if inlet_surfs:
        gmsh.model.addPhysicalGroup(2, inlet_surfs, tag=1)
        gmsh.model.setPhysicalName(2, 1, "inlet")
    if outlet_surfs:
        gmsh.model.addPhysicalGroup(2, outlet_surfs, tag=2)
        gmsh.model.setPhysicalName(2, 2, "outlet")
    if wall_surfs:
        gmsh.model.addPhysicalGroup(2, wall_surfs, tag=3)
        gmsh.model.setPhysicalName(2, 3, "wall")
    if obstacle_surfs:
        gmsh.model.addPhysicalGroup(2, obstacle_surfs, tag=4)
        gmsh.model.setPhysicalName(2, 4, "obstacle")

    volumes = gmsh.model.getEntities(dim=3)
    vol_tags = [v[1] for v in volumes]
    gmsh.model.addPhysicalGroup(3, vol_tags, tag=5)
    gmsh.model.setPhysicalName(3, 5, "volume")

    # Mesh refinement near obstacle
    if obstacle_surfs:
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "SurfacesList", obstacle_surfs)
        gmsh.model.mesh.field.setNumber(1, "Sampling", 100)

        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", h_fine)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", h_coarse)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 1.0 * r)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 3.0 * r)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)

    # Generate mesh
    gmsh.model.mesh.generate(3)

    tmpfile = tempfile.NamedTemporaryFile(suffix=".msh", delete=False)
    tmpfile.close()
    gmsh.write(tmpfile.name)
    gmsh.finalize()

    mesh = Mesh(tmpfile.name)
    os.unlink(tmpfile.name)

    if periodic_x or periodic_y or periodic_z:
        import numpy as np
        Lx, Lyz = left + right, 2 * updown
        mesh.topology_dm.setPeriodicity(
            np.array([Lx / 2 if periodic_x else 0.0,
                      Lyz / 2 if periodic_y else 0.0,
                      Lyz / 2 if periodic_z else 0.0]),     # maxCell
            np.array([-left if periodic_x else 0.0,
                      -updown if periodic_y else 0.0,
                      -updown if periodic_z else 0.0]),     # Lstart
            np.array([Lx if periodic_x else -1.0,
                      Lyz if periodic_y else -1.0,
                      Lyz if periodic_z else -1.0]),        # L
        )

    return mesh
