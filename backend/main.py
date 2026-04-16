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
class_names = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model():
    global model, class_names
    model_path = "model.pth"
    class_names_path = "class_names.txt"

    if os.path.exists(model_path) and os.path.exists(class_names_path):
        try:
            print(f"Loading model from {model_path}...")
            with open(class_names_path, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]

            model = models.resnet18(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(class_names))

            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            print("Model loaded successfully.")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    else:
        print("Model file not found. Using Mock Model.")
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
    """Compute Grad-CAM heatmap for the target class using ResNet18's last conv layer."""
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['value'] = output

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0]

    # Hook into the last conv layer of ResNet18 (layer4[-1])
    target_layer = model.layer4[-1]
    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass with gradients enabled
    model.zero_grad()
    output = model(input_tensor)

    # Backprop for the target class
    class_score = output[0, target_class_idx]
    class_score.backward()

    # Compute weighted activation map
    grads = gradients['value'][0]        # [C, H, W]
    acts  = activations['value'][0]      # [C, H, W]
    weights = grads.mean(dim=(1, 2))     # Global average pool → [C]

    cam = torch.zeros(acts.shape[1:], device=device)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = torch.relu(cam)
    cam = cam.detach().cpu().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()

    fwd_handle.remove()
    bwd_handle.remove()

    return cam

def overlay_heatmap(pil_image, cam):
    """Overlay Grad-CAM heatmap onto the original image and return as base64 PNG."""
    img = np.array(pil_image.resize((224, 224))).astype(np.uint8)

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
