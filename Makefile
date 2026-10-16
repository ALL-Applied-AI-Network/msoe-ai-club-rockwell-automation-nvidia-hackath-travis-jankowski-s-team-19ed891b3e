# Shortcuts for the commands in README.md. Each line is also runnable on its own.
MODEL := artifacts/models/fake/weights/best.pt

test:
	uv run pytest
	cd fanuc_er4ia_sim && uv run python -m unittest test_sort_planning test_cube_perception test_cube_pose test_runtime_config

# Whole pipeline on a laptop with the OpenCV fake twin (no GPU, ~35 min on an M-series Mac).
# Proves the plumbing, not sim-to-real accuracy: use Isaac renders for real models.
smoke:
	uv run python tests/fake_twin.py data/raw/fake --frames 400
	uv run python -m smartgrasp.dataset data/raw/fake data/bev/fake
	uv run python -m smartgrasp.train data/bev/fake --name fake --epochs 30
	uv run python -m smartgrasp.evaluate perception data/bev/fake --detector yolo:$(MODEL) --save artifacts/eval/fake_yolo.json
	uv run python -m smartgrasp.evaluate perception data/bev/fake --detector hsv --save artifacts/eval/fake_hsv.json
	uv run python tests/fake_twin.py data/raw/fake_stack --frames 1 --stack
	uv run python -m smartgrasp.deploy --image data/raw/fake_stack/000000.jpg --calib data/raw/fake_stack/camera.json \
		--detector yolo:$(MODEL) --out runs/fake_stack

.PHONY: test smoke
