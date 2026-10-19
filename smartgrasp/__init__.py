"""SmartGrasp: find, grasp and stack cubes with a FANUC ER-4iA.

The cube detector is trained on synthetic data rendered from the digital twin.

Pipeline (one module per stage, each runnable with `python -m smartgrasp.<stage>`):
    isaac/generate_data.py  render randomized frames from the twin (Isaac Sim)
    dataset    frames -> top-down (BEV) images + oriented-box labels
    train      fine-tune YOLO26-OBB, export ONNX, write a model card
    evaluate   perception metrics, sim episode metrics, real-cell trial metrics
    deploy     image -> detections -> plan for the twin and a dry-run plan for the real cell
    calibrate  fit a real camera's homography from touched-off points
"""
