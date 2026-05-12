from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io
import base64
from PIL import Image
import numpy as np
import cv2
import torch
from torchvision import transforms, models
import torch.nn as nn
import os
import json
import random
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
model = None
model_arch = None
class_names = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_FILES = {
    'resnet18':    'resnet18.pth',
    'resnet50':    'resnet50.pth',
    'mobilenetv2': 'mobilenetv2.pth',
}

def build_model(arch, num_classes):
    if arch == 'resnet18':
        m = models.resnet18(pretrained=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == 'resnet50':
        m = models.resnet50(pretrained=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == 'mobilenetv2':
        m = models.mobilenet_v2(pretrained=False)
        m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return m

def get_target_layer(arch):
    if arch in ('resnet18', 'resnet50'):
        return model.layer4[-1]
    elif arch == 'mobilenetv2':
        return model.features[-1]
    return None

def pick_best_model():
    """Read metrics.json to find the best model, fallback to first available .pth."""
    if os.path.exists('metrics.json'):
        try:
            with open('metrics.json', 'r') as f:
                data = json.load(f)
            best = data.get('best_model')
            if best and os.path.exists(MODEL_FILES.get(best, '')):
                return best, MODEL_FILES[best]
        except Exception:
            pass

    # Fallback: use first available
    for arch, path in MODEL_FILES.items():
        if os.path.exists(path):
            return arch, path

    return None, None

def load_model():
    global model, model_arch, class_names
    class_names_path = "class_names.txt"

    arch, model_path = pick_best_model()

    if arch and model_path and os.path.exists(class_names_path):
        try:
            print(f"Loading {arch.upper()} from {model_path}...")
            with open(class_names_path, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]

            model = build_model(arch, len(class_names))
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            model_arch = arch
            print(f"Model loaded successfully: {arch.upper()}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    else:
        print("No model file found. Using Mock Model.")
        return False

# Initialize model
model_loaded = load_model()
if model_loaded:
    print(f"Server initialized with classes: {class_names}")
else:
    print("WARNING: Server initialized in MOCK MODE")

def apply_clahe(pil_image):
    """Apply CLAHE to the L channel of the image in LAB colour space."""
    img_array = np.array(pil_image)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced)

def compute_gradcam(input_tensor, target_class_idx):
    target_layer = get_target_layer(model_arch)
    if target_layer is None:
        return None
    with GradCAMPlusPlus(model=model, target_layers=[target_layer]) as cam:
        targets = [ClassifierOutputTarget(target_class_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    return grayscale_cam[0]

def overlay_heatmap(pil_image, cam):
    """Overlay Grad-CAM heatmap onto the model-input crop and return as base64 PNG."""
    # Replicate the exact same transform the model saw: Resize(256) then CenterCrop(224)
    # Using simple resize(224,224) causes spatial misalignment with the heatmap
    w, h = pil_image.size
    scale = 256 / min(w, h)
    rw, rh = int(w * scale), int(h * scale)
    resized = pil_image.resize((rw, rh), Image.BILINEAR)
    left = (rw - 224) // 2
    top  = (rh - 224) // 2
    cropped = resized.crop((left, top, left + 224, top + 224))
    img = np.array(cropped).astype(np.uint8)

    # Resize CAM to image size and apply jet colormap
    cam_resized = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend original + heatmap
    overlay = (0.55 * img + 0.45 * heatmap).astype(np.uint8)

    # Encode to base64
    overlay_pil = Image.fromarray(overlay)
    buffer = io.BytesIO()
    overlay_pil.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

# Preprocessing transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.get("/")
def read_root():
    return {"message": "Feline Skin Disease Detector API"}

@app.get("/metrics")
def get_metrics():
    metrics_path = "metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return {"error": "Metrics not found. Run evaluate.py first."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')

    if model_loaded and model:
        # Apply CLAHE preprocessing
        image = apply_clahe(image)

        inputs = transform(image).unsqueeze(0).to(device)

        # Run inference (with grad enabled for Grad-CAM)
        outputs = model(inputs)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0].detach()

        top_prob, top_catid = torch.topk(probabilities, 1)
        confidence = top_prob.item()
        label = class_names[top_catid.item()]

        all_probs = [
            {"label": class_names[i], "confidence": float(probabilities[i])}
            for i in range(len(class_names))
        ]
        all_probs.sort(key=lambda x: x["confidence"], reverse=True)

        # Compute Grad-CAM heatmap
        gradcam_base64 = None
        try:
            cam = compute_gradcam(inputs, top_catid.item())
            gradcam_base64 = overlay_heatmap(image, cam)
        except Exception as e:
            print(f"Grad-CAM failed: {e}")

        return {
            "prediction": label,
            "confidence": float(confidence),
            "all_probabilities": all_probs,
            "gradcam_image": gradcam_base64,
            "mode": "real_model"
        }
    else:
        # Mock Inference
        mock_classes = ["ringworm", "scabies", "flea", "healthy"]
        probs = [random.random() for _ in mock_classes]
        total = sum(probs)
        probs = [p / total for p in probs]
        all_probs = sorted(
            [{"label": mock_classes[i], "confidence": probs[i]} for i in range(4)],
            key=lambda x: x["confidence"], reverse=True
        )
        label = all_probs[0]["label"]
        confidence = all_probs[0]["confidence"]

        return {
            "prediction": label,
            "confidence": confidence,
            "all_probabilities": all_probs,
            "gradcam_image": None,
            "mode": "mock_model"
        }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
