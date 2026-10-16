# Instructions for coding agents

Read `README.md` (pipeline) first, and `CONTEXT.md` (goals, rules, team) if it is present.
`CONTEXT.md` is the mentor's local briefing: never commit or push it (it is git-ignored).
User instructions take precedence over this file.

## Layout

- `smartgrasp/`: host package. Python 3.12, `uv`. One module per stage; each runs with `uv run python -m smartgrasp.<module> --help`.
- `isaac/`: runs only under Isaac Sim's interpreter (`$ISAAC/python.sh`). Never under the host venv.
- `fanuc_er4ia_sim/`: the organisers' calibrated twin. Read its `AGENTS.md` before editing it. Keep edits there minimal and mark them `SmartGrasp:`.
- `tests/`: pytest suite plus `fake_twin.py`, an OpenCV stand-in for Isaac that writes the same files.

## Invariants

- **Train on synthetic data only.** The event guide says real line data is there to look at, not to train on: teams generate training data with the digital twin. Real frames go only to the held-out test split (`dataset --real`), for evaluation.
- **Ground truth is for evaluation.** It never feeds a detector, a plan or a motion command.
- **One perception space.** Every detector takes a BEV image (`geometry.Bev`) and returns `Detection`s with corners in BEV pixels. Add new detectors in `detectors.py` behind `load_detector`.
- **The twin's validator is the contract for plans.** A full six-cube `deploy.plan_stack` must keep passing `fanuc_er4ia_sim/sort_scene.validate_plan` (`tests/test_smartgrasp.py`).
- **No code here commands the real robot.** `cell_plan.csv` is a dry run for a human. Do not add controller communication without the mentor's explicit approval.
- **Never commit** `data/`, `artifacts/`, `runs/`, weights, `.env` or API keys.
- **Less code is better.** Prefer a parameter or a small function over a new module or framework.

## Verify before claiming success

```bash
make test        # SmartGrasp pytest + the twin's 33 host tests
make smoke       # end-to-end on the fake twin (~35 min on a Mac)
```

- Isaac Sim can exit 0 after an exception. Check `execution_report.json` and the logs, not the exit code.
- Host tests prove nothing about GPU, Isaac or real-camera behaviour. Say exactly what ran and what is still unverified.
