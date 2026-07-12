# 🎭 Deepfake Detection — ViT-B/16

Detects whether an image or video contains a deepfake, using a fine-tuned Vision Transformer (ViT-B/16). Trained on FaceForensics++, Celeb-DF-v2(reel images only), and StyleGAN3 faces(reel images only). Served through a simple Flask web app.

## How it works

- Trained as a 7-class classifier (Real + 6 deepfake generation methods), collapsed to a **Real/Fake** verdict at inference.
- Faces are auto-detected and cropped (Haar cascade) before being passed to the model.
- Video is analyzed by sampling frames and averaging the fake-probability across them.

## Project files

```
app.py                  # Flask app — upload an image/video, get a verdict
main_deepfake.ipynb     # Training & evaluation notebook (Kaggle)
```

## Setup

```bash
pip install flask torch torchvision opencv-python numpy pillow werkzeug scikit-learn seaborn matplotlib tqdm grad-cam
```

Point the app at your trained checkpoint (`.pth` file):

```bash
export VIT_MODEL_PATH="/path/to/latest_best_vit_7cls_epoch8.pth"
```

Run it:

```bash
python app.py
```

Then open `http://localhost:5000` in your browser.

## API

`POST /predict` — form-data with a `file` (image or video) and optional `threshold` (default `0.5`).

```json
{
  "verdict": "FAKE",
  "fake_probability": 87.42,
  "threshold": 0.5,
  "file_type": "image"
}
```

Supported formats: `.jpg .jpeg .png .webp` (images), `.mp4 .avi .mov .mkv` (video).

## Notes

- The trained checkpoint (`.pth`).
- `main_deepfake.ipynb` contains the full training pipeline and paths reference Kaggle (`/kaggle/input/...`) — update these if running elsewhere.
- Detection quality depends on face-detection accuracy and how closely inputs resemble the training data; don't treat verdicts as definitive.
