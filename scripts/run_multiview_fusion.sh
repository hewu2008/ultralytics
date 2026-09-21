#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

python "$REPO_DIR/tools/multiview_fusion.py" "$REPO_DIR/assets/multiview-fusion.yaml" "$@"