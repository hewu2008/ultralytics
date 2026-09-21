#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
SCENE_DIR="$REPO_DIR/runs/multiview/scene_boxes"

rm -rf "$SCENE_DIR"
python "$REPO_DIR/tools/synth_multiview_scene.py" "$SCENE_DIR" "$@"