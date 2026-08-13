from fastapi import FastAPI, UploadFile, File, HTTPException
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

# Gate-keeper blocklist: reject if top-1 ImageNet class is a wild cat or dog breed
# with high confidence. Blocklist is used instead of an allowlist because close-up
# skin/fur photos don't score on domestic cat classes (281-285) — ImageNet sees fur
# texture, not a recognisable cat face — causing false rejections on the actual dataset.
# Wild cats: 286-293 (cougar, lynx, leopard, snow leopard, jaguar, lion, tiger, cheetah)
# Dog breeds: 151-268
BLOCKED_IMAGENET_INDICES = set(range(151, 269)) | set(range(286, 294))
gate_model = None

# Stage-2 flea/ringworm specialist: fires only when stage-1's top-2 predicted
# classes are exactly {flea, ringworm}, the pair the 4-class model confuses most.
STAGE2_PAIR = {'flea', 'ringworm'}
STAGE2_MODEL_FILE = 'models/mobilenetv2_stage2_flea_ringworm.pth'
STAGE2_CLASS_NAMES_FILE = 'models/stage2_class_names.txt'
stage2_model = None
stage2_class_names = []

MODEL_FILES = {
    'resnet18':    'models/resnet18_clahe.pth',
    'resnet50':    'models/resnet50_clahe.pth',
    'mobilenetv2': 'models/mobilenetv2_clahe.pth',
}

def load_gate_model():
    global gate_model
    try:
        try:
            gate_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        except AttributeError:
            gate_model = models.mobilenet_v2(pretrained=True)
        gate_model.eval()
        print("Gate-keeper model (ImageNet MobileNetV2) loaded.")
    except Exception as e:
        print(f"Gate-keeper model failed to load: {e}")

def load_stage2_model():
    global stage2_model, stage2_class_names
    try:
        if os.path.exists(STAGE2_MODEL_FILE) and os.path.exists(STAGE2_CLASS_NAMES_FILE):
            with open(STAGE2_CLASS_NAMES_FILE, 'r') as f:
                stage2_class_names = [line.strip() for line in f.readlines()]
            m = build_model('mobilenetv2', len(stage2_class_names))
            m.load_state_dict(torch.load(STAGE2_MODEL_FILE, map_location=device))
            m.to(device)
            m.eval()
            stage2_model = m
            print(f"Stage-2 flea/ringworm specialist loaded: {stage2_class_names}")
        else:
            print("Stage-2 model files not found. Running without flea/ringworm specialist.")
    except Exception as e:
        print(f"Stage-2 model failed to load: {e}")

def is_feline_image(pil_image, block_threshold=0.50):
    """Return False only when top-1 ImageNet class is a wild cat or dog with >= block_threshold confidence.

    Allowlist approach (require high cat-class probability) fails on close-up skin photos:
    ImageNet sees fur texture rather than a recognisable cat, so real disease images get
    incorrectly rejected. Blocklist approach only rejects clear non-feline animals
    (tiger at 90%, dog at 80%) while accepting ambiguous close-up skin images.
    """
    if gate_model is None:
        return True  # fail open if gate model unavailable
    with torch.no_grad():
        tensor = transform(pil_image).unsqueeze(0)  # runs on CPU
        out = gate_model(tensor)
        probs = torch.softmax(out, dim=1)[0]
    top1_idx = torch.argmax(probs).item()
    top1_prob = probs[top1_idx].item()
    return not (top1_idx in BLOCKED_IMAGENET_INDICES and top1_prob >= block_threshold)

def compute_normalized_entropy(probabilities):
    """Entropy normalised to [0, 1]. Values near 1 mean the model is maximally uncertain."""
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9))
    max_entropy = torch.log(torch.tensor(float(len(class_names))))
    return (entropy / max_entropy).item()

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

def get_target_layers(arch):
    # layer2 (28×28) for fine detail, layer3 (14×14) for mid-level patterns,
    # layer4 (7×7) for semantic context.  layer1 excluded — its 56×56 grid
    # produces a dotted/pixelated pattern after upsampling.
    if arch == 'resnet18':
        return [model.layer2[-1], model.layer3[-1], model.layer4[-1]]
    elif arch == 'resnet50':
        return [model.layer2[-1], model.layer3[-1], model.layer4[-1]]
    elif arch == 'mobilenetv2':
        return [model.features[-5], model.features[-3], model.features[-1]]
    return []

def pick_best_model():
    """Read metrics.json to find the best model, fallback to first available .pth."""
    if os.path.exists('models/metrics.json'):
        try:
            with open('models/metrics.json', 'r') as f:
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
    class_names_path = "models/class_names.txt"

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

# Initialize models
model_loaded = load_model()
if model_loaded:
    print(f"Server initialized with classes: {class_names}")
else:
    print("WARNING: Server initialized in MOCK MODE")
load_gate_model()
load_stage2_model()

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
    target_layers = get_target_layers(model_arch)
    if not target_layers:
        return None
    with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
        targets = [ClassifierOutputTarget(target_class_idx)]
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets,
            eigen_smooth=True,
        )
    # Gaussian blur smooths out the block-grid artefacts that appear when
    # multiple feature-map scales are aggregated and upsampled.
    raw = grayscale_cam[0]
    smoothed = cv2.GaussianBlur(raw, (0, 0), sigmaX=3)
    # Re-normalise so the peak stays at 1.0 after blurring.
    smoothed = smoothed / (smoothed.max() + 1e-8)
    return smoothed

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

    cam_resized = cv2.resize(cam, (224, 224))

    # --- Colour guide: anomaly + pink/red skin detector ------------------
    img_f32 = img.astype(np.float32) / 255.0

    # 1) Two-scale spatial anomaly (colour deviation from local neighbourhood).
    m_fine   = cv2.blur(img_f32, (21, 21))
    m_coarse = cv2.blur(img_f32, (81, 81))
    anom = np.maximum(
        np.mean((img_f32 - m_fine)   ** 2, axis=2),
        np.mean((img_f32 - m_coarse) ** 2, axis=2),
    )
    anom = cv2.GaussianBlur(anom.astype(np.float32), (11, 11), 0)
    anom = anom / (anom.max() + 1e-8)

    # 2) Pink/red skin detector (hue-based).
    # Inflamed tissue (ringworm, flea bites, scabies) has hue near 0°/360° (red-pink).
    # Orange fur sits at H 8-25 in OpenCV; white fur has near-zero saturation.
    # → Only pink/red pixels with some saturation and brightness qualify.
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    h_ch = img_hsv[:, :, 0]           # OpenCV: 0-180
    s_ch = img_hsv[:, :, 1] / 255.0
    v_ch = img_hsv[:, :, 2] / 255.0
    # Bright inflamed tissue: pink/red hue, reasonably bright (ringworm, exposed skin)
    bright_pinkred = (((h_ch < 9) | (h_ch > 163)) & (s_ch > 0.20) & (v_ch > 0.25)).astype(np.float32)
    # Dark red/brown spots: flea bite dots on orange fur — darker, slightly wider hue
    dark_spots = ((h_ch < 14) & (s_ch > 0.15) & (v_ch > 0.10) & (v_ch < 0.58)).astype(np.float32)
    pinkred = np.maximum(bright_pinkred, dark_spots)
    pinkred = cv2.GaussianBlur(pinkred, (11, 11), 0)
    pinkred = pinkred / (pinkred.max() + 1e-8)

    # 3) Combined guide: max of spatial anomaly and pink/red skin signal.
    guide = np.maximum(anom, pinkred)
    guide = guide / (guide.max() + 1e-8)

    # Stronger blend: high-guide regions fully boosted, low-guide suppressed harder.
    cam_guided = cam_resized * (0.25 + 0.75 * guide)
    cam_guided /= (cam_guided.max() + 1e-8)
    # --------------------------------------------------------------------

    # Threshold: drop bottom 20 % only — keep most signal so dim spots aren't lost.
    thresh = float(np.percentile(cam_guided, 20))
    cam_clean = np.where(cam_guided >= thresh, cam_guided, 0.0)
    if cam_clean.max() > 0:
        cam_clean /= cam_clean.max()

    # Small close: merges adjacent cells within the same lesion without
    # expanding outward into the surrounding fur.
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary  = (cam_clean > 0).astype(np.uint8)
    closed  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)
    cam_filled = cam_clean * closed.astype(np.float32)
    if cam_filled.max() > 0:
        cam_filled /= cam_filled.max()

    # Mild gamma so dim lesion areas stay visible alongside the brightest spot.
    cam_final = np.power(cam_filled, 0.55)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_final), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Adaptive blend: high-activation pixels show more heatmap, background stays natural.
    alpha = (0.30 + 0.55 * cam_final[:, :, np.newaxis]).clip(0, 1)
    overlay = ((1 - alpha) * img + alpha * heatmap).astype(np.uint8)

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
    metrics_path = "models/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return {"error": "Metrics not found. Run training/evaluate.py first."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')

    # --- Guardrail 1: ImageNet gate-keeper (domestic cat check) ---
    if not is_feline_image(image):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "not_feline",
                "message": "No domestic cat detected in this image. Please upload a clear, close-up photo of a cat's skin.",
            },
        )

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

        # --- Guardrail 2: entropy-based out-of-distribution guard ---
        normalized_entropy = compute_normalized_entropy(probabilities)
        is_ood = normalized_entropy > 0.85

        all_probs = [
            {"label": class_names[i], "confidence": float(probabilities[i])}
            for i in range(len(class_names))
        ]
        all_probs.sort(key=lambda x: x["confidence"], reverse=True)

        # --- Stage 2: flea/ringworm specialist ---
        # Fires only when stage-1 is genuinely torn between flea and ringworm
        # (its two most-confused classes), and redistributes their combined
        # probability mass according to the specialist's binary verdict.
        stage2_applied = False
        target_class_idx = top_catid.item()
        if stage2_model is not None and {all_probs[0]["label"], all_probs[1]["label"]} == STAGE2_PAIR:
            with torch.no_grad():
                stage2_probs = torch.nn.functional.softmax(stage2_model(inputs), dim=1)[0]
            s2_lookup = {stage2_class_names[i]: stage2_probs[i].item() for i in range(len(stage2_class_names))}

            combined_mass = sum(e["confidence"] for e in all_probs if e["label"] in STAGE2_PAIR)
            denom = max(s2_lookup["flea"] + s2_lookup["ringworm"], 1e-8)
            for entry in all_probs:
                if entry["label"] in STAGE2_PAIR:
                    entry["confidence"] = combined_mass * (s2_lookup[entry["label"]] / denom)
            all_probs.sort(key=lambda x: x["confidence"], reverse=True)

            label = all_probs[0]["label"]
            confidence = all_probs[0]["confidence"]
            target_class_idx = class_names.index(label)
            stage2_applied = True

        # Compute Grad-CAM heatmap
        gradcam_base64 = None
        try:
            cam = compute_gradcam(inputs, target_class_idx)
            gradcam_base64 = overlay_heatmap(image, cam)
        except Exception as e:
            print(f"Grad-CAM failed: {e}")

        return {
            "prediction": label,
            "confidence": float(confidence),
            "is_ood": is_ood,
            "all_probabilities": all_probs,
            "gradcam_image": gradcam_base64,
            "mode": "real_model",
            "stage2_applied": stage2_applied,
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
            "is_ood": False,
            "all_probabilities": all_probs,
            "gradcam_image": None,
            "mode": "mock_model",
        }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
