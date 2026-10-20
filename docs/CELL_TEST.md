# Testing on the CSI cell

The organisers run the real FANUC ER-4iA as a remote test service at
[hackathon.sidclab.org](https://hackathon.sidclab.org). Our policy runs on a team laptop. Each
step, the cell sends it the table camera image and the arm's pose, `smartgrasp/policy.py`
returns fingertip targets, and the cell plays them. The cell checks every target against its
limits, and a CSI operator sets up, supervises and scores every rollout. A run is 10 rollouts
in a 30-minute slot, booked in the site's Calendar tab. The site's FAQ answers most questions
about the cell: frames, cameras, gripper, objects and scoring.

## Rules that do not bend

- Submit a request or book a slot only with the mentor's OK.
- The arm moves only through the organisers' client. No controller changes, PLC edits or camera moves. Printed fingers need the plant manager **and** a mentor, and go in `custom_hardware` in `submission.yaml`, which CSI reviews with the request.
- Real data is for evaluation only (event guide: real data is there to look at, not to train on). That includes the recorded rollouts.
- Log every rollout, not only the successes.

## 1. Set up the client (once per laptop)

From this repo's folder:

```bash
SG=$PWD
git clone https://github.com/SIDCLab/PhysicalAI-Hackathon-2026 ../PhysicalAI-Hackathon-2026
cd ../PhysicalAI-Hackathon-2026
echo "from smartgrasp.policy import Policy  # noqa: F401" > policy.py
uv run --project "$SG" run.py --mock
```

Run every `run.py` command from the client folder with `uv run --project "$SG"`, so it uses
this repo's environment. `--checkpoint` picks the detector: `yolo:<weights>` (default
`artifacts/models/sim_v1/weights/best.pt`), `hsv` or `vlm`. `git pull` the client before each slot.

What the policy does each rollout:

1. Warps the 1920×1080 table image to the top-down view. The homography is fitted to the
   organisers' calibration (`calibration/cameras.json`, within 0.2 mm), so we need no ArUco
   calibration of our own.
2. Detects the cubes and plans the tower with `deploy.plan_frame`. It writes `table.jpg`,
   `bev.png`, `overlay.png` and the plan files to `runs/cell/<trial id>/`.
3. Streams `cell_plan` as `tcp` targets: straight moves at `SPEED_MM_S` (80 mm/s), one target
   every `DT` (0.1 s), holding `DWELL_S` (1 s) after each gripper switch.
4. Plans once per rollout. It holds the arm still until it has a plan and after the plan has
   played. A failed grasp is not retried.

## 2. Check before every slot

1. `make test` passes, and the code and model are frozen (commit, `model_card.json`).
2. `uv run --project "$SG" run.py --mock` ends with **✓ Ready for the real robot**. The mock
   checks formats, limits and timing only: its images have no cubes, so the arm holds still.
3. Dry-run a real frame (from an earlier rollout's `runs/cell/*/table.jpg`) and check that every
   box in `overlay.png` sits on a cube top:
   ```bash
   uv run python -m smartgrasp.deploy --image table.jpg --calib ../PhysicalAI-Hackathon-2026/calibration/cameras.json \
       --detector yolo:artifacts/models/sim_v1/weights/best.pt --out runs/check
   ```

## 3. Submit and book (mentor's OK)

1. Fill in `submission/submission.yaml`. Model type `Hybrid` for YOLO or HSV plus the planner,
   `VLM + BM` for `--checkpoint vlm`. The task: stack yellow, yellow, blue, blue, red, red cubes
   (10 mm) into a tower at x 280, y 100 mm.
2. Fill in `submission/placement.yaml` with each rollout's cube positions. Keep every cube
   inside x 180-520 mm and y -280-300 mm, within 540 mm of the robot base axis, and at least
   45 mm from the tower site.
3. `uv run --project "$SG" run.py --submit` prints a request id. Pick a slot with it in the
   Calendar tab. `--submit` writes `.request.json` to the client folder: it holds the request's
   secret, so never share or commit it.
4. `uv run --project "$SG" run.py --status` shows whether CSI approved it.

## 4. Run during the slot

```bash
uv run --project "$SG" run.py
```

Start it before the slot and leave it running. It waits for approval, then for the operator,
and runs every rollout by itself. Come to the lab to watch if you can. Afterwards:

- `run.py --status` prints the private results link: scores, cycle times and videos.
- `run.py --download` gets the recorded LeRobotDataset (three cameras and the arm, 10 Hz).

## 5. Score it

The operator scores each rollout as a whole: success or failure, time to success, faults and
interventions. For our headline numbers, also log each pick from the videos in a copy of
[trials_template.csv](trials_template.csv): one row per grasp attempt, with the rollout in
`notes`. Then:

```bash
uv run python -m smartgrasp.evaluate trials trials_day1.csv
```

Compare against a baseline measured the same way, such as `--checkpoint hsv`.

## 6. Build the held-out real evaluation set

Each rollout's `runs/cell/<trial id>/table.jpg` is a full-size real frame taken before the arm
moves. Collect them, warp them and label them:

```bash
mkdir -p data/raw/csi_day1
for d in runs/cell/*/; do cp "$d/table.jpg" "data/raw/csi_day1/$(basename "$d").jpg"; done
uv run python -m smartgrasp.dataset data/raw/csi_day1 data/bev/csi_day1 \
    --calib ../PhysicalAI-Hackathon-2026/calibration/cameras.json --real
```

Label `data/bev/csi_day1/images/test` in CVAT. Draw one rotated box per cube top face. Create
the CVAT labels in `data.yaml` order (0 cube_yellow, 1 cube_blue, 2 cube_red, 3 cap, 4 tube),
and check that the exported ids match: the class id, not the name, decides the colour. Export as
"Ultralytics YOLO Oriented Bounding Boxes" and copy the `.txt` files into `labels/test/`. Then
compare every detector on the same frames:

```bash
uv run python -m smartgrasp.evaluate perception data/bev/csi_day1 --detector yolo:<weights>
uv run python -m smartgrasp.evaluate perception data/bev/csi_day1 --detector hsv
```

## Still to ask CSI

1. How many 10 mm cubes of each colour the cell has, and how `placement.yaml` names a cube's colour.
2. Whether 80 mm/s in 8 mm steps, 0.1 s apart, suits the cell in `tcp` mode, or whether a longer `DT` is better.
3. Whether the operator can leave the tower site clear and keep the arm out of the table camera's view when a rollout starts.
