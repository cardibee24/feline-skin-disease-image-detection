import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np
import json
import os

MODELS = {
    'resnet18':    'resnet18.pth',
    'resnet50':    'resnet50.pth',
    'mobilenetv2': 'mobilenetv2.pth',
}

def build_model(model_name, num_classes):
    if model_name == 'resnet18':
        m = models.resnet18(pretrained=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        m = models.resnet50(pretrained=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif model_name == 'mobilenetv2':
        m = models.mobilenet_v2(pretrained=False)
        m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    return m

def evaluate_single(model_name, model_path, val_loader, class_names, device):
    print(f"\nEvaluating {model_name.upper()} from {model_path} ...")
    model = build_model(model_name, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    accuracy = float(np.mean(all_preds == all_labels))
    report   = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    # ROC-AUC (one-vs-rest, macro)
    try:
        roc_auc = float(roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro'))
    except Exception:
        roc_auc = None

    per_class = {}
    for cls in class_names:
        per_class[cls] = {
            "precision": round(report[cls]["precision"], 4),
            "recall":    round(report[cls]["recall"],    4),
            "f1_score":  round(report[cls]["f1-score"],  4),
            "support":   int(report[cls]["support"]),
        }

    result = {
        "overall_accuracy": round(accuracy, 4),
        "macro_precision":  round(report["macro avg"]["precision"], 4),
        "macro_recall":     round(report["macro avg"]["recall"],    4),
        "macro_f1":         round(report["macro avg"]["f1-score"],  4),
        "roc_auc":          round(roc_auc, 4) if roc_auc is not None else None,
        "per_class":        per_class,
    }

    print(f"  Accuracy : {accuracy*100:.2f}%")
    print(f"  Macro F1 : {result['macro_f1']}")
    print(f"  ROC-AUC  : {result['roc_auc']}")

    return result

def evaluate_all(data_dir='dataset', class_names_path='class_names.txt'):
    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), val_transform)
    val_loader  = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    device      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    comparison = {}
    best_model = None
    best_f1    = -1

    for model_name, model_path in MODELS.items():
        if not os.path.exists(model_path):
            print(f"Skipping {model_name} — {model_path} not found.")
            continue
        result = evaluate_single(model_name, model_path, val_loader, class_names, device)
        comparison[model_name] = result
        if result["macro_f1"] > best_f1:
            best_f1    = result["macro_f1"]
            best_model = model_name

    if not comparison:
        print("No trained models found. Train at least one model first.")
        return

    # Build final metrics.json — keep best model's metrics at top level for backward compat
    best = comparison[best_model]
    metrics = {
        # Top-level: best model's metrics (used by About modal existing code)
        "overall_accuracy": best["overall_accuracy"],
        "macro_precision":  best["macro_precision"],
        "macro_recall":     best["macro_recall"],
        "macro_f1":         best["macro_f1"],
        "roc_auc":          best["roc_auc"],
        "per_class":        best["per_class"],
        "best_model":       best_model,
        "epochs":           25,
        "dataset_split":    "80% Train / 20% Validation",
        # Full comparison table
        "comparison":       comparison,
    }

    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*40}")
    print(f"Best model: {best_model.upper()} (Macro F1: {best_f1:.4f})")
    print(f"Metrics saved to metrics.json")
    print(f"{'='*40}")

if __name__ == '__main__':
    evaluate_all()
