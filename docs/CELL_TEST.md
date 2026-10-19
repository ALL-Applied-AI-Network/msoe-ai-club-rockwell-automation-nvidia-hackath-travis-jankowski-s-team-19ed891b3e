# Testing at the CSI cell

SmartGrasp never commands the robot. At CSI it turns one camera frame into a
pick-and-place plan that a person reviews, and the CSI operator runs it through
whatever interface the cell already uses. Book time in `#data-and-testbed-requests`
(tag Shamar Webster) with this plan, the attendees and the duration.

## Rules that do not bend

- No arm commands, controller changes, PLC edits or printed fingers without the plant manager **and** a mentor.
- Never move or re-aim a camera. Calibrate around the cameras as they are.
- Real frames are for evaluation only (event guide: real data is there to look at, not to train on). `--real` datasets go to the test split only.
- Log every attempt, not only the successes.

## 1. Questions to settle with CSI before the visit

1. Controller model and software version (R-30iB Mate Plus or R-50iA?), and which options are installed (R912 Remote Motion Interface, J519 Stream Motion, S636).
2. How the cell is driven today (teach pendant programs, PLC over EtherNet/IP, RMI, ROS 2), and what CSI wants from us: a reviewed XYZWPR table, position registers, or a TP program.
3. Active UFRAME and UTOOL. Is UTOOL1 exactly (0, 0, 223, 0, 0, 0)? Do the fingers close along tool Y? (`cell_plan.csv` assumes both.)
4. Gripper I/O: which outputs open and close the SMC MHZL2-16D2, and any jaw sensors.
5. Cameras: model, resolution, lens, and how to grab frames (USB, GigE, RTSP, iRVision). Which overhead camera is closest to the twin's table camera?
6. Safety: speed override for student runs, who holds the E-stop, agreed home pose, DCS limits.
7. Which objects are really on the table (cube size and colours, caps, tubes).
8. Can we have the twin's mat artwork (`assets/mat_print.png`, left out of the starter package)? If not, photograph the mat flat and square so [#35](https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-travis-jankowski-s-team-19ed891b3e/issues/35) can texture it.

## 2. Calibrate the table camera (about 30 minutes)

The model works in a top-down view of the plane 10 mm above the mat (cube-top
height). Calibration finds the homography from camera pixels to robot XY on that plane.

1. Print 6-9 ArUco markers (`DICT_4X4_50`, ids 0-8, about 30 mm) and glue each on a flat 10 mm-high block.
2. Spread the blocks over the work area (x 0.20-0.52 m, y -0.26-0.30 m in the robot base frame), keeping each within 0.54 m of the base axis: the far corners are out of top-down reach.
3. With the operator: jog the TCP to each marker centre and read X, Y from the pendant in UF0. UF0 shares the robot base's XY, so in metres x = X/1000 and y = Y/1000. Write `markers.csv`:
   ```csv
   id,x,y
   0,0.250,-0.200
   ```
4. Remove the robot from the view, grab one frame and run:
   ```bash
   uv run python -m smartgrasp.calibrate --image frame.png --points markers.csv --aruco --out calib/table_cam.json
   ```
   Aim for `max_mm` under 1-2 mm. Larger usually means a typo, a marker not at 10 mm, or lens distortion.
5. Check it: `uv run python -m smartgrasp.deploy --image frame.png --calib calib/table_cam.json --detector hsv --out runs/check`, then open `runs/check/bev.png`. The table should look flat and square, with blocks where you placed them.

## 3. Collect a held-out real evaluation set

```bash
uv run python -m smartgrasp.dataset data/raw/csi_day1 data/bev/csi_day1 --calib calib/table_cam.json --real
```
Label `data/bev/csi_day1/images/test` in CVAT. Draw one rotated box per cube top face. Create the CVAT labels in `data.yaml` order (0 cube_yellow, 1 cube_blue, 2 cube_red, 3 cap, 4 tube), and check that the exported ids match: the class id, not the name, decides the colour. Export as "Ultralytics YOLO Oriented Bounding Boxes" and copy the `.txt` files into `labels/test/`. Then compare every detector on the same frames:
```bash
uv run python -m smartgrasp.evaluate perception data/bev/csi_day1 --detector yolo:<weights>
uv run python -m smartgrasp.evaluate perception data/bev/csi_day1 --detector hsv
```

## 4. Supervised trial session

1. Freeze the code (git commit) and the model (`model_card.json`). Record both in the trial log.
2. Before the session, run `deploy` on saved frames. Confirm it prints the expected number of picks, and that every box in `overlay.png` sits on a cube top.
3. For each trial:
   - Grab a frame and run `deploy`.
   - Review `overlay.png` and the printed table with the operator. The operator executes, or skips if anything looks wrong.
4. Log every attempt in a copy of [trials_template.csv](trials_template.csv). A retry of the same trial gets `attempt` 2, 3, and so on.
5. Score the session: `uv run python -m smartgrasp.evaluate trials trials_day1.csv`. This prints the three headline numbers: first-attempt grasp success, recovery rate and cycle time.

Compare against a baseline measured the same way, such as the `hsv` detector or the taught program.
