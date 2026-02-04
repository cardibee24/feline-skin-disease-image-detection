from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import os
import random

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev
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
            # Load class names
            with open(class_names_path, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
            
            # Initialize model architecture (must match training)
            # Assuming ResNet18 as in train.py
            model = models.resnet18(pretrained=False) # No need to download weights, we load ours
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

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    
    if model_loaded and model:
        # Real Inference
        inputs = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, top_catid = torch.topk(probabilities, 1)
            
            confidence = top_prob.item()
            label = class_names[top_catid.item()]
            
        return {
            "prediction": label,
            "confidence": float(confidence),
            "mode": "real_model"
        }
    else:
        # Mock Inference
        # Returns a random result for demonstration
        mock_diseases = ["Ringworm", "Flea Allergy", "Mange", "Healthy", "Ear Mites"]
        label = random.choice(mock_diseases)
        confidence = random.uniform(0.70, 0.99)
        
        return {
            "prediction": label,
            "confidence": confidence,
            "mode": "mock_model"
        }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
