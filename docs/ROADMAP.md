# Roadmap: scoped tasks for the team

Each task is 2-4 hours, pairs by default, and ends in something measurable.
"Done when" is the acceptance check. Numbers come from `smartgrasp.evaluate`, never from eyeballing a video.

## Sprint 1 (Oct 19-25): environments working, baseline measured

| # | Task | Done when |
|---|---|---|
| 1 | **First Isaac frames.** On Brev (Isaac Sim 6.0.1 launchable) or a Spark, run `uv run python fanuc_er4ia_sim/make_textures.py`, then `$ISAAC/python.sh isaac/generate_data.py --out data/raw/isaac_try --frames 20` and `uv run python -m smartgrasp.dataset data/raw/isaac_try data/bev/isaac_try`, and open a few `data/bev/.../images` next to their labels. | Labels sit on the cube tops in 20/20 frames. Note the GPU, runtime and credits used. |
| 2 | **Real synthetic dataset.** Generate 3-5k frames and train on a GPU (`--device 0`). | `model_card.json` and an eval JSON on the Isaac test split, uploaded to the team's Hugging Face org (see README). |
| 3 | **Baseline numbers.** Evaluate `yolo`, `hsv` and `vlm --limit 20` on the same Isaac test split. | One table in the README with recall, xy/yaw error and graspable rate for all three. |
| 4 | **Closed-loop sim.** Run `isaac/run_episode.sh` for 20 seeds. | `evaluate episodes` output: full-plan rate, first-attempt grasp success, placement success. |
| 5 | **CSI calibration kit.** Print ArUco markers, make 10 mm blocks, and do a dry run of the calibration steps on a desk camera ([CELL_TEST.md](CELL_TEST.md)). | `calibrate` reports max error under 2 mm on the desk setup. |
| 6 | **Questions to CSI.** Send the list in [CELL_TEST.md](CELL_TEST.md) section 1 and book a session. | Answers recorded in `docs/CELL_TEST.md`; date booked. |

## Sprint 2 (Oct 26-Nov 1): improve one thing at a time, validate on hardware

Change one factor per experiment, re-run task 3 or 4, and keep the result only if the number improves.

| # | Extension | Measure against the baseline |
|---|---|---|
| 7 | **Sim-to-real randomization.** Cube colour tint and roughness, mat texture, small camera-pose jitter, the arm parked in view like `capture_scene.py`, sensor noise and blur. | Recall and graspable rate on the **real** held-out set (task 9). |
| 8 | **In-context learning track.** Compare zero-shot with `vlm:<dataset>` few-shot. Then use the VLM's own box centre for XY and the colour blob only for yaw: today `_fit` throws the VLM's position away. Compare models with `VLM_MODEL`: Gemma 4, `nvidia/nemotron-3.5-super-vl-120b-a12b`, and a self-hosted Qwen3.8 or Cosmos3-Nano served by vLLM on the GB10 or Brev (`VLM_BASE_URL`). | Same perception table, plus latency and failed requests per image. |
| 9 | **Real evaluation set.** 30+ CSI frames, labelled in CVAT, evaluation only. | Every detector scored on it; the sim-to-real gap is stated. |
| 10 | **Recovery after a failed grasp.** Check `pickup_verified` in `sort_scene.py`. On failure, re-capture and re-plan that cube inside the pick loop. `plan_stack` refuses a frame with anything near the stack site, so extend it to start from the levels already built. A wrist-camera check is the stretch version. | Recovery rate in `evaluate episodes` (the baseline reports none: it never retries). |
| 11 | **Supervised hardware session.** Follow [CELL_TEST.md](CELL_TEST.md) section 4. | `evaluate trials` headline numbers against the baseline. |
| 12 | **Video and one-pager** (from Oct 29). | 5-minute video showing real hardware, a baseline comparison and failure areas. |

Stretch, only once the core demo works: custom fingers for larger objects (caps, tubes), sorting rules beyond the stack task, and a learned policy (SmolVLA / GR00T) trained on scripted twin rollouts, in sim only.
