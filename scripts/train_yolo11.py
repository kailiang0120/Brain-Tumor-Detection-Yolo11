#!/usr/bin/env python3
"""Train and evaluate a YOLOv11 object detector on the merged brain tumor dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
from ultralytics import YOLO

DEFAULT_DATA_CONFIG = Path("configs/yolo11.yaml")
DEFAULT_MODEL = "yolo11m.pt"
DEFAULT_PROJECT = "runs_brain_tumor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv11 on brain tumor MRI dataset."
    )
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA_CONFIG, help="Path to YOLO data YAML"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Ultralytics model checkpoint or YAML"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--img-size", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--lr0", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device spec (auto, cpu, cuda:0, etc.)",
    )
    parser.add_argument(
        "--project", default=DEFAULT_PROJECT, help="Ultralytics project directory"
    )
    parser.add_argument("--name", default="yolo11_brain_tumor", help="Experiment name")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint in project/name",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--patience", type=int, default=50, help="Early stopping patience (epochs)"
    )
    parser.add_argument(
        "--val-only",
        action="store_true",
        help="Skip training and run validation on existing weights",
    )
    return parser.parse_args()


def ensure_paths(args: argparse.Namespace) -> None:
    if not args.data.exists():
        raise FileNotFoundError(f"Data config not found: {args.data}")
    args.data = args.data.resolve()


def train(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_paths(args)
    if args.device != "cpu":
        if not torch.cuda.is_available():
            print("CUDA unavailable; falling back to CPU")
            args.device = "cpu"
        else:
            visible = torch.cuda.device_count()
            if visible == 0:
                print("CUDA visible devices = 0; falling back to CPU")
                args.device = "cpu"

    model = YOLO(args.model)

    fit_params = dict(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch,
        lr0=args.lr0,
        device=args.device,
        project=args.project,
        name=args.name,
        seed=args.seed,
        patience=args.patience,
        resume=args.resume,
    )

    if args.val_only:
        metrics = model.val(
            data=str(args.data),
            imgsz=args.img_size,
            project=args.project,
            name=args.name,
            device=args.device,
        )
        print(metrics)
        return metrics

    results = model.train(**fit_params)
    print(results)
    metrics = model.val(
        data=str(args.data),
        imgsz=args.img_size,
        project=args.project,
        name=f"{args.name}_eval",
        device=args.device,
    )
    print(metrics)
    return metrics


if __name__ == "__main__":
    metrics = train(parse_args())
    print("Training completed. Key metrics:", metrics)
