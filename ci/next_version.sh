#!/usr/bin/env bash
# Prints the version of the next release: the highest X.Y.Z tag of the remote plus 0.0.1.
# The same version is used for the Docker image tag and the GitHub release tag.
# Without any X.Y.Z tag, it continues from INITIAL_VERSION, the last version released by hand.
#
# Usage: ci/next_version.sh [remote]   (default: origin)
set -euo pipefail

INITIAL_VERSION="${INITIAL_VERSION:-1.1.0}"

latest=$(git ls-remote --tags --refs "${1:-origin}" \
    | sed 's#.*refs/tags/##' \
    | { grep -E '^[0-9]+\.[0-9]+\.[0-9]+$' || true; } \
    | sort -V | tail -n 1)

IFS=. read -r major minor patch <<< "${latest:-$INITIAL_VERSION}"
echo "$major.$minor.$((patch + 1))"
