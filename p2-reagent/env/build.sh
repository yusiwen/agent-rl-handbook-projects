#!/usr/bin/env bash
# Build the Project 2 training image and print exactly what was built.
#
#   bash p2-reagent/env/build.sh
#
# Idempotent: Docker reuses the layers, so a second run takes seconds and reports the same id.
# Set ARLH_TRAIN_IMAGE to build under another tag.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
image="${ARLH_TRAIN_IMAGE:-arlh-pytorch:26.09}"

docker build -f "$here/Dockerfile" -t "$image" "$here"
docker image inspect "$image" \
  --format 'built {{.RepoTags}} id={{.Id}} size={{.Size}} created={{.Created}}'
