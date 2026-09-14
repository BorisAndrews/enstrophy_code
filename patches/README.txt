Firedrake patches required by this package.

These patches modify Firedrake's source, and so, unlike the rest of this
repository, are licensed under Firedrake's licence, the GNU Lesser General
Public License v3.0 or later.

Both patch Firedrake's own installed source, not this package. They are NOT
reapplied by a Firedrake reinstall or upgrade, so re-run apply_patches.sh
after either.

    ./patches/apply_patches.sh            (apply; skips any already applied)
    ./patches/apply_patches.sh --check    (dry run)

Verify with

    python -c "from firedrake.formmanipulation import drop_zero_blocks; print('ok')"


firedrake-real-block-submesh.patch
    REQUIRED for the 3D vortex street with the enstrophy-stable ("asf")
    scheme. That scheme carries a Real-space Lagrange multiplier for the
    harmonic 1-form, so Firedrake eliminates it with a matrix-free Schur
    complement; the sub-blocks it extracts keep structurally-zero terms that
    still name another mesh, and the form compiler rejects them with
    MismatchingDomainError. The patch adds formmanipulation.drop_zero_blocks
    and calls it from ImplicitMatrixContext and PCSNESBase.form.

    Without it, every 3D "asf" run fails at setup.

firedrake-submesh-periodic.patch
    REQUIRED for the 2D vortex street, which imposes the inlet and obstacle
    conditions on a Submesh of a periodic mesh. DMPlexFilter marks the
    submesh as having localized coordinates but never writes them, and
    DMLocalizeCoordinates will not fill them in because the flag it keys off
    is already set. Compounding that, DMSetPeriodicity sizes maxCell/Lstart/L
    by topological dimension, so on a submesh (tdim < cdim) they are too
    short to index. The patch adds mesh._localize_submesh_coordinates.

    Without it, 2D runs on a periodic mesh segfault. No-op for
    non-periodic meshes.

    The 3D driver uses no Submesh, so 3D-only users do not strictly need
    this one. Applying both is simpler and harmless.
