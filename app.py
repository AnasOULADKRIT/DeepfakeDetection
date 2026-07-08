"""
Deepfake Detection Flask App
Model: ViT-B/16 (Anas's model) — image + video
"""

import os, gc, time, json, base64
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as tv_models


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
UPLOAD_FOLDER   = Path("static/uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
ALLOWED_IMAGE   = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_VIDEO   = {".mp4", ".avi", ".mov", ".mkv"}
MAX_CONTENT_MB  = 500

# Update this path to point at your saved .pth file
VIT_MODEL_PATH  = os.environ.get("VIT_MODEL_PATH",  "C:\\licence AI\\S2\\Projet_Transversaux\\deepfake_app\\models\\latest_best_vit_7cls_epoch8.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────
# ViT model definition (matches Anas's notebook)
# ─────────────────────────────────────────────
CATEGORIES = [
    "Real",              # 0
    "DeepFakeDetection", # 1
    "Deepfakes",         # 2
    "Face2Face",         # 3
    "FaceShifter",       # 4
    "FaceSwap",          # 5
    "NeuralTextures",    # 6
]
NUM_CLASSES = len(CATEGORIES)
VIT_DROPOUT = 0.35

def build_vit(num_classes=NUM_CLASSES, dropout=VIT_DROPOUT):
    model = tv_models.vit_b_16(weights=None)
    in_f = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.LayerNorm(in_f),
        nn.Linear(in_f, 512),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(512, num_classes),
    )
    return model

# ─────────────────────────────────────────────
# Model loader (lazy, cached)
# ─────────────────────────────────────────────
_models: dict = {}

def load_vit():
    if "vit" not in _models:
        if not Path(VIT_MODEL_PATH).exists():
            raise FileNotFoundError(
                f"ViT checkpoint not found at '{VIT_MODEL_PATH}'. "
                "Set VIT_MODEL_PATH env var or copy the file there.")
        print("Loading ViT model …")
        model = build_vit()
        ckpt  = torch.load(VIT_MODEL_PATH, map_location=DEVICE, weights_only=False)
        # handles both raw state-dict and wrapped checkpoints
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        model.load_state_dict(sd, strict=False)
        model.to(DEVICE).eval()
        _models["vit"] = model
        print("ViT loaded.")
    return _models["vit"]

# ─────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

vit_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ─────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────

def predict_image_vit(image_path: str, threshold: float = 0.5) -> dict:
    if not os.path.exists(image_path):
        return {"error": f"File not found: {image_path}"}
        
    model = load_vit()
    img_pil = Image.open(image_path).convert("RGB")
    w, h = img_pil.size
    
    # ── 1. Détection et recadrage automatique du visage ──
    try:
        img_np = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            x, y, face_w, face_h = faces[0]  # On prend le premier visage trouvé
            
            x_min = max(0, int(x))
            y_min = max(0, int(y))
            x_max = min(w, int(x + face_w))
            y_max = min(h, int(y + face_h))
            
            img_pil = img_pil.crop((x_min, y_min, x_max, y_max))
            
    except Exception as e:
        print(f"Vision preprocessing warning: {e}. Using full image instead.")

    # ── 2. Inférence avec votre modèle ViT (et correction du batch) ──
    with torch.no_grad():
        t      = vit_tfm(img_pil).unsqueeze(0).to(DEVICE)
        
        # Le [0] ici est CRITIQUE pour extraire l'image du batch et éviter les prédictions bloquées à 0%
        probs  = F.softmax(model(t), dim=1)[0]
        
        # Somme des probabilités des classes 1 à 6 (les classes de Deepfakes)
        fake_p = probs[1:].sum().item()
        
    verdict = "FAKE" if fake_p >= threshold else "REAL"
    return {"verdict": verdict, "fake_probability": round(fake_p * 100, 2),
            "threshold": threshold}

def predict_video_vit(video_path: str, frame_interval: int = 10,
                      threshold: float = 0.5) -> dict:
    model = load_vit()
    cap   = cv2.VideoCapture(video_path)
    fake_probs, frame_count, analyzed = [], 0, 0
    model.eval()
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    m  = int(w * 0.1)
                    y1, y2 = max(0, y - m), min(frame.shape[0], y + h + m)
                    x1, x2 = max(0, x - m), min(frame.shape[1], x + w + m)
                    face = Image.fromarray(
                        cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
                    t      = vit_tfm(face).unsqueeze(0).to(DEVICE)
                    probs  = F.softmax(model(t), dim=1)[0]
                    fake_probs.append(probs[1:].sum().item())
                    analyzed += 1
            frame_count += 1
    cap.release()
    if analyzed == 0:
        return {"error": "No faces detected in video."}
    avg_p   = float(np.mean(fake_probs))
    verdict = "FAKE" if avg_p >= threshold else "REAL"
    return {"verdict": verdict,
            "fake_probability": round(avg_p * 100, 2),
            "frames_analyzed": analyzed,
            "threshold": threshold}


# ─────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024
app.config["UPLOAD_FOLDER"]      = str(UPLOAD_FOLDER)


def allowed_file(filename: str):
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_IMAGE | ALLOWED_VIDEO, ext


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request."}), 400

    file   = request.files["file"]
    method = "vit"
    try:
        threshold = float(request.form.get("threshold", 0.5))
    except (TypeError, ValueError):
        threshold = 0.5
    threshold = min(max(threshold, 0.0), 1.0)

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    ok, ext = allowed_file(file.filename)
    if not ok:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    filename = secure_filename(file.filename)
    save_path = str(UPLOAD_FOLDER / filename)
    file.save(save_path)

    is_image = ext in ALLOWED_IMAGE
    is_video = ext in ALLOWED_VIDEO

    try:
        t0 = time.time()
        result = None

        if is_image:
            result = predict_image_vit(save_path, threshold=threshold)
        elif is_video:
            result = predict_video_vit(save_path, threshold=threshold)

        if result is None:
            return jsonify({"error": "Could not run inference for this input."}), 400

        result["inference_time_s"] = round(time.time() - t0, 2)
        result["file_type"] = "image" if is_image else "video"
        result["method"]    = method
        return jsonify(result)

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500
    finally:
        # clean up uploaded file to save disk space
        try:
            os.remove(save_path)
        except Exception:
            pass


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
