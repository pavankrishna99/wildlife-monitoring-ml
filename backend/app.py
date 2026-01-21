from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import uuid
from PIL import Image
from torchvision import transforms, models

from megadetector_loader import load_megadetector

# =========================================================
# CONFIG
# =========================================================
DEVICE = "cpu"
CONF_THRESHOLD = 0.6
NOVELTY_THRESHOLD = 0.007806
CROP_PADDING = 0.15

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))

MD_PATH = os.path.join(MODEL_DIR, "md_v5a.0.0.pt")
STAGE2_PATH = os.path.join(MODEL_DIR, "stage2_species_finetuned.pth")
AE_PATH = os.path.join(MODEL_DIR, "stage3_autoencoder.pth")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# =========================================================
# LOAD MODELS
# =========================================================
md_model = load_megadetector(MD_PATH, device=DEVICE)
if md_model is None:
    raise RuntimeError("MegaDetector failed to load")

ckpt = torch.load(STAGE2_PATH, map_location=DEVICE)
class_names = ckpt["class_names"]

classifier = models.resnet18(pretrained=False)
classifier.fc = nn.Linear(classifier.fc.in_features, len(class_names))
classifier.load_state_dict(ckpt["model_state"])
classifier.eval()

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, 2, 1, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

ae = AutoEncoder()
ae.load_state_dict(torch.load(AE_PATH, map_location=DEVICE))
ae.eval()

# =========================================================
# TRANSFORMS
# =========================================================
clf_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

ae_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================================================
# FLASK
# =========================================================
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

# =========================================================
# HELPERS
# =========================================================
def crop_with_padding(img, box):
    w, h = img.size
    x1, y1, x2, y2 = map(int, box)
    bw, bh = x2 - x1, y2 - y1
    pw, ph = int(bw * CROP_PADDING), int(bh * CROP_PADDING)
    return img.crop((
        max(0, x1 - pw),
        max(0, y1 - ph),
        min(w, x2 + pw),
        min(h, y2 + ph)
    ))

# =========================================================
# PIPELINE
# =========================================================
def run_pipeline(image_path):
    image = Image.open(image_path).convert("RGB")
    detections = md_model.detect_animals(image, conf_threshold=0.7)
    animals = [d for d in detections if int(d[5]) == 0]

    orig_np = np.array(image)
    orig_bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)

    # Unique filename for THIS request
    out_name = f"result_{uuid.uuid4().hex}.jpg"
    out_path = os.path.join(RESULT_DIR, out_name)

    # -------- EMPTY FRAME --------
    if not animals:
        cv2.putText(orig_bgr, "EMPTY FRAME", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.imwrite(out_path, orig_bgr)
        return {
            "status": "empty",
            "species": "none",
            "confidence": 0.0,
            "novelty_score": 0.0,
            "image_url": f"/results/{out_name}"
        }

    # -------- BEST ANIMAL --------
    best = max(animals, key=lambda d: (d[2]-d[0])*(d[3]-d[1]))
    x1, y1, x2, y2 = map(int, best[:4])

    crop = crop_with_padding(image, best[:4])

    # -------- STAGE 2 --------
    x = clf_tfms(crop).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(classifier(x), dim=1)
        conf, idx = probs.max(dim=1)

    confidence = conf.item()
    species = class_names[idx.item()]

    # -------- STAGE 3 --------
    x_ae = ae_tfms(crop).unsqueeze(0)
    with torch.no_grad():
        recon = ae(x_ae)
        recon_error = torch.mean((x_ae - recon) ** 2).item()

    status = "known" if (confidence >= CONF_THRESHOLD and recon_error <= NOVELTY_THRESHOLD) else "unknown"
    label = species if status == "known" else "Unknown"

    # -------- DRAW OUTPUT --------
    cv2.rectangle(orig_bgr, (x1, y1), (x2, y2), (0,255,0), 3)
    cv2.putText(orig_bgr, label, (x1, max(y1-10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imwrite(out_path, orig_bgr)

    return {
        "status": status,
        "species": label.lower() if status == "known" else "unknown",
        "confidence": confidence,
        "novelty_score": recon_error,
        "image_url": f"/results/{out_name}"
    }

# =========================================================
# ROUTES
# =========================================================
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(path)

    try:
        result = run_pipeline(path)
    finally:
        os.remove(path)

    return jsonify(result)

@app.route("/results/<path:filename>")
def results(filename):
    return send_from_directory(RESULT_DIR, filename)

# =========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
