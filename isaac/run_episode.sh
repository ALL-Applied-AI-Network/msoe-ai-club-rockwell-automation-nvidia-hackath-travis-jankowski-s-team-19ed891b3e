#!/usr/bin/env bash
# One closed-loop sim episode: twin camera -> SmartGrasp perception and plan -> twin physics.
#   ISAAC=/isaac-sim isaac/run_episode.sh <layout seed> <new run dir> <detector spec>
#   e.g. isaac/run_episode.sh 7 runs/episodes/seed7 yolo:artifacts/models/sim_v1/weights/best.pt
# Score many episodes with: uv run python -m smartgrasp.evaluate episodes runs/episodes/*
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SEED="${1:?layout seed}"; RUN="${2:?run directory}"; DETECTOR="${3:?detector spec}"
ISAAC="${ISAAC:-/isaac-sim}"
# scene_config.py: seed moves the 2 red + 2 blue cubes and all 6 yaws; yellows keep calibrated XY.
export CUBE_LAYOUT_SEED="$SEED"
mkdir -p "$RUN"
[ -z "$(ls -A "$RUN")" ] || { echo "Use an empty run directory: $RUN" >&2; exit 1; }  # no stale reports
"$ISAAC/python.sh" "$ROOT/fanuc_er4ia_sim/capture_scene.py" --out "$RUN" --scale 1
uv run --project "$ROOT" python -m smartgrasp.deploy --image "$RUN/world_camera.png" --calib "$RUN/camera.json" \
    --detector "$DETECTOR" --out "$RUN"
# sort_scene.py exits early when the plan is incomplete; evaluate counts that as a failed episode.
"$ISAAC/python.sh" "$ROOT/fanuc_er4ia_sim/sort_scene.py" --run "$RUN" --physics-only || true
[ -f "$RUN/execution_report.json" ] || echo "no execution report: see $RUN and the sort_scene log" >&2
uv run --project "$ROOT" python -m smartgrasp.evaluate episodes "$RUN"
