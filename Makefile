# Shortcuts for the commands in README.md. Each line is also runnable on its own.
MODEL := artifacts/models/fake/weights/best.pt

# Style gate (rules in AGENTS.md): lint, formatting, types; the comment rule runs in pytest.
lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run ty check

# Apply the formatter and Ruff's safe fixes; run before make lint.
format:
	uv run ruff format .
	uv run ruff check --fix .

test: lint
	uv run pytest
	cd fanuc_er4ia_sim && uv run python -m unittest test_sort_planning test_cube_perception test_cube_pose test_runtime_config

# Whole pipeline on the OpenCV fake twin (no GPU, ~35 min on a Mac): proves plumbing, not accuracy.
smoke:
	uv run python tests/fake_twin.py data/raw/fake --frames 400
	uv run python -m smartgrasp.dataset data/raw/fake data/bev/fake
	uv run python -m smartgrasp.train data/bev/fake --name fake --epochs 30
	uv run python -m smartgrasp.evaluate perception data/bev/fake --detector yolo:$(MODEL) --save artifacts/eval/fake_yolo.json
	uv run python -m smartgrasp.evaluate perception data/bev/fake --detector hsv --save artifacts/eval/fake_hsv.json
	uv run python tests/fake_twin.py data/raw/fake_stack --frames 1 --stack
	uv run python -m smartgrasp.deploy --image data/raw/fake_stack/000000.jpg --calib data/raw/fake_stack/camera.json \
		--detector yolo:$(MODEL) --out runs/fake_stack

.PHONY: format lint test smoke
