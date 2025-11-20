import os
from typing import Any, Dict, List, Tuple, Optional

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO


_CACHED_MODEL: Optional[YOLO] = None


def resolve_device() -> str:
	"""Pick CUDA if available."""
	return "cuda" if torch.cuda.is_available() else "cpu"


def get_model() -> YOLO:
	"""Load YOLO model once."""
	global _CACHED_MODEL
	if _CACHED_MODEL is not None:
		return _CACHED_MODEL

	model_path = os.getenv("MODEL_PATH", "yolov11m_finetune.pt")
	if not os.path.exists(model_path):
		raise FileNotFoundError(f"Model weights not found at: {model_path}")

	model = YOLO(model_path)
	device = resolve_device()
	model.to(device)
	_CACHED_MODEL = model
	return model


def to_rgb_array(image: Any) -> np.ndarray:
	"""Ensure numpy RGB uint8."""
	if image is None:
		raise ValueError("No image provided")

	if isinstance(image, Image.Image):
		img = image.convert("RGB")
		return np.array(img, dtype=np.uint8)

	arr = np.asarray(image)
	if arr.ndim == 2:
		arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
	elif arr.shape[2] == 4:
		# Drop alpha
		arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
	# Assume already RGB otherwise
	return arr.astype(np.uint8)


def resize_long_side(image_rgb: np.ndarray, max_long_side: int = 1280) -> np.ndarray:
	"""Limit long side for faster inference, keep aspect ratio."""
	h, w = image_rgb.shape[:2]
	long_side = max(h, w)
	if long_side <= max_long_side:
		return image_rgb
	scale = max_long_side / float(long_side)
	new_w = int(round(w * scale))
	new_h = int(round(h * scale))
	return cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def preprocess_image(image: Any) -> np.ndarray:
	"""Minimal preprocessing for robustness."""
	rgb = to_rgb_array(image)
	rgb = resize_long_side(rgb, max_long_side=int(os.getenv("MAX_LONG_SIDE", "1280")))
	return rgb


def run_inference(
	image: Any,
	conf_threshold: float = 0.25,
	iou_threshold: float = 0.45,
) -> Tuple[np.ndarray, str]:
	"""Run YOLO and return annotated image + detections text."""
	model = get_model()
	input_rgb = preprocess_image(image)

	# Ultralytics handles letterbox/normalization internally
	results = model.predict(
		source=input_rgb,
		conf=conf_threshold,
		iou=iou_threshold,
		verbose=False,
	)
	res = results[0]

	# Annotated image from Ultralytics is BGR
	annotated_bgr = res.plot()
	annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

	detections: List[Dict[str, Any]] = []
	names = res.names if hasattr(res, "names") and res.names else (model.names if hasattr(model, "names") else {})
	if res.boxes is not None and len(res.boxes) > 0:
		for box in res.boxes:
			xyxy = box.xyxy[0].tolist()
			x_min, y_min, x_max, y_max = [round(float(v), 2) for v in xyxy]
			cls_id = int(box.cls[0].item()) if box.cls is not None else -1
			conf = float(box.conf[0].item()) if box.conf is not None else 0.0
			cls_name = names.get(cls_id, str(cls_id))
			detections.append(
				{
					"class": cls_name,
					"confidence": round(conf, 4),
					"x_min": x_min,
					"y_min": y_min,
					"x_max": x_max,
					"y_max": y_max,
				}
			)

	# Format detections as readable text
	if detections:
		lines = ["Detections found:\n"]
		for i, det in enumerate(detections, 1):
			lines.append(
				f"{i}. {det['class']} (confidence: {det['confidence']:.2%})\n"
				f"   Bounding box: ({det['x_min']}, {det['y_min']}) to ({det['x_max']}, {det['y_max']})"
			)
		detections_text = "\n\n".join(lines)
	else:
		detections_text = "No detections found."

	return annotated_rgb, detections_text


def example_images(limit: int = 3) -> List[str]:
	"""Collect a few example images if present."""
	candidates: List[str] = []
	for split in ("test", "valid", "train"):
		base = os.path.join("data", "images", split)
		if not os.path.isdir(base):
			continue
		for name in os.listdir(base):
			if name.lower().endswith((".jpg", ".jpeg", ".png")):
				candidates.append(os.path.join(base, name))
				if len(candidates) >= limit:
					return candidates
	return candidates


def build_interface() -> gr.Blocks:
	with gr.Blocks(title="Brain Tumor Detection (YOLO)") as demo:
		gr.Markdown(
			"## Brain Tumor Detection\n"
			"Upload an MRI/CT slice. The model will detect tumor classes and draw boxes.\n"
			"Model: `best.pt` (override via `MODEL_PATH`)."
		)
		with gr.Row():
			with gr.Column(scale=1):
				image_in = gr.Image(type="numpy", label="Upload Image", sources=["upload", "clipboard"])
				conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence threshold")
				iou = gr.Slider(0.2, 0.95, value=0.45, step=0.05, label="NMS IoU threshold")
				run_btn = gr.Button("Run Detection", variant="primary")
			with gr.Column(scale=1):
				image_out = gr.Image(type="numpy", label="Annotated Image")
				detections_out = gr.Textbox(label="Detections", lines=10, interactive=False)
		run_btn.click(
			fn=run_inference,
			inputs=[image_in, conf, iou],
			outputs=[image_out, detections_out],
			api_name=False,
		)

		examples = example_images()
		if examples:
			gr.Examples(
				examples=examples,
				inputs=[image_in],
				examples_per_page=min(3, len(examples)),
				label="Examples (from data/images/* if available)",
			)
	return demo


if __name__ == "__main__":
	# E.g. PORT=7860 MODEL_PATH=best.pt python app/inference_ui.py
	ui = build_interface()
	ui.launch(
		server_name=os.getenv("HOST", "127.0.0.1"),
		server_port=int(os.getenv("PORT", "7860")),
		show_error=True,
		share=True,
	)


