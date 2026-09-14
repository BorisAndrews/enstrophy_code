#!/usr/bin/env bash
#
# Apply the Firedrake patches this package needs, to the Firedrake in the
# currently active virtual environment.
#
#   ./patches/apply_patches.sh          # apply
#   ./patches/apply_patches.sh --check  # dry run, change nothing
#
# Both patches are against Firedrake's installed source tree, not this
# package. Re-installing or upgrading Firedrake silently drops them, so
# re-run this afterwards.

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

# Importing firedrake also prints PETSc's unused-option warnings on stdout,
# so tag the line we want rather than taking the whole output.
SITE="$(python -c 'import firedrake, os
print("SITEDIR=" + os.path.dirname(os.path.dirname(firedrake.__file__)))' 2>/dev/null \
        | sed -n 's/^SITEDIR=//p' | head -1)"
if [ -z "$SITE" ] || [ ! -d "$SITE/firedrake" ]; then
    echo "error: cannot import firedrake. Activate the Firedrake venv first." >&2
    exit 1
fi
echo "Firedrake found at: $SITE/firedrake"

ARGS=(-p1 -d "$SITE")
[ "${1:-}" = "--check" ] && ARGS+=(--dry-run)

for patch in firedrake-real-block-submesh.patch firedrake-submesh-periodic.patch; do
    echo
    echo "== $patch"
    if patch "${ARGS[@]}" -R --dry-run --force < "$DIR/$patch" > /dev/null 2>&1; then
        echo "   already applied, skipping"
        continue
    fi
    patch "${ARGS[@]}" < "$DIR/$patch"
done

echo
echo "Done. Verify with:"
echo "  python -c 'from firedrake.formmanipulation import drop_zero_blocks; print(\"ok\")'"
