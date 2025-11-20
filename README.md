# Brain Tumor Detection with YOLOv11

This project demonstrates a fine-tuned **YOLOv11** model for detecting brain tumors in MRI/CT scan images. It includes a training notebook and an interactive web-based inference interface built with **Gradio**.

## 🌟 Features

- **State-of-the-art Detection**: Utilizes the YOLOv11 architecture fine-tuned for medical imaging.
- **Interactive UI**: User-friendly web interface to upload images and visualize detection results instantly.
- **Adjustable Parameters**: Real-time control over Confidence and NMS IoU thresholds.
- **GPU Support**: Automatically detects and utilizes CUDA-enabled GPUs for faster inference.

## 🛠️ Installation

### Prerequisites

- Python 3.9 or higher
- [Optional] CUDA-enabled GPU for faster training and inference

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/kailiang0120/Brain-Tumor-Detection-Yolo11.git
   cd Brain-Tumor-Detection-Yolo11
   ```

2. **Install dependencies**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Running the Inference UI

To start the web application for tumor detection:

```bash
python app/inference_ui.py
```

After running the command, access the interface in your browser at:
**http://127.0.0.1:7860**

### Training

If you wish to retrain or fine-tune the model on your own dataset, refer to the Jupyter notebook located in:
`notebooks/yolov11.ipynb`

## 📂 Project Structure

- **`app/`**: Contains the source code for the Gradio inference application.
- **`configs/`**: Configuration files for the model and training.
- **`data/`**: Directory for storing dataset images (train/val/test).
- **`notebooks/`**: Jupyter notebooks used for data exploration and model training.
- **`yolov11m_finetune.pt`**: The pre-trained/fine-tuned model weights.
- **`requirements.txt`**: List of Python dependencies.

## 🔧 Configuration

The inference app supports environment variables for easy configuration:

- `MODEL_PATH`: Path to the `.pt` model file (default: `yolov11m_finetune.pt`).
- `PORT`: Port to run the Gradio app on (default: `7860`).
- `HOST`: Host address (default: `127.0.0.1`).

Example:
```bash
MODEL_PATH=my_custom_model.pt PORT=8080 python app/inference_ui.py
```
