# BrainTumor — YOLO-based Brain Tumor Detection

A compact repository for preparing, training, and running inference for brain tumor detection on MRI/CT images (YOLO-style detector pipeline). This repo contains dataset preparation helpers, training scripts, and a simple Gradio inference UI.

Key files
- `configs/yolo11.yaml` — data/config for training
- `scripts/train_yolo11.py` — training entrypoint
- `app/inference_ui.py` — small Gradio app for local inference
- `scripts/prepare_brain_tumor_dataset.py` — (if present) data conversion/prep helpers

Quick facts
- Dataset: Brain tumor MRI dataset (public sources; see scripts/notes)
- Model: YOLO-style detector (configurable in `scripts/train_yolo11.py`)
- Environment: Conda environment `gemma` is recommended

Repository structure
```
.
├── app/                   # inference UI and utilities
├── configs/               # model and data configs
├── scripts/               # data prep and training scripts
├── data/                  # (gitignored) dataset root — keep local
├── runs_brain_tumor/      # training outputs (gitignored)
└── README.md
```

Requirements
- Python 3.8+ (prefer 3.9/3.10)
- GPU recommended for training (CUDA/cuDNN)
- Activate environment:
```
conda activate gemma
pip install -r requirements.txt
```

Data placement (important)
- The repository ignores the `data/` folder to avoid committing large datasets. Place your dataset under `data/` locally.
- Example recommended layout:
```
data/
  brain_tumor_prepared/
    images/
      train/
      val/
      test/
    labels/
      train/
      val/
      test/
    dataset_summary.json
```

Preparing the dataset
- If you have raw source files (e.g., from Kaggle/Roboflow), use the provided scripts (if available) or adapt your own converter to produce the `images/` and `labels/` structure above. The repository previously used a prep script that produced YOLO-format labels.

Training
- Basic example (adjust flags in `scripts/train_yolo11.py` or pass CLI args):
```
python scripts/train_yolo11.py \
  --data configs/yolo11.yaml \
  --model yolo11m.pt \
  --project runs_brain_tumor \
  --name yolo11_brain_tumor \
  --epochs 120 \
  --batch 24 \
  --img-size 640
```
- Check `runs_brain_tumor/{name}` for checkpoints, metrics, and tensorboard logs.

Inference (local UI)
- Run the Gradio-based inference UI:
```
conda activate gemma
pip install -r requirements.txt
python app/inference_ui.py
```
- By default the app prints a local URL (e.g., `http://127.0.0.1:7860`). Override defaults with environment variables:
  - `MODEL_PATH` — path to model weights (default `best.pt`)
  - `HOST`, `PORT` — bind address/port
  - `MAX_LONG_SIDE` — resize long side before passing to the model
