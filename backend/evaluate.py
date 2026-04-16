import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import json
import os

def evaluate_model(data_dir='dataset', model_path='model.pth', class_names_path='class_names.txt'):
    # Load class names
    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    # Validation transform (same as training)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall accuracy
    accuracy = float(np.mean(all_preds == all_labels))

    # Per-class metrics
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    per_class = {}
    for cls in class_names:
        per_class[cls] = {
            "precision": round(report[cls]["precision"], 4),
            "recall":    round(report[cls]["recall"], 4),
            "f1_score":  round(report[cls]["f1-score"], 4),
            "support":   int(report[cls]["support"]),
        }

    metrics = {
        "overall_accuracy": round(accuracy, 4),
        "macro_precision":  round(report["macro avg"]["precision"], 4),
        "macro_recall":     round(report["macro avg"]["recall"], 4),
        "macro_f1":         round(report["macro avg"]["f1-score"], 4),
        "per_class":        per_class,
        "model":            "ResNet18",
        "epochs":           10,
        "dataset_split":    "80% Train / 20% Validation",
    }

    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nOverall Accuracy : {accuracy*100:.2f}%")
    print(f"Macro Precision  : {metrics['macro_precision']}")
    print(f"Macro Recall     : {metrics['macro_recall']}")
    print(f"Macro F1-Score   : {metrics['macro_f1']}")
    print("\nPer-class metrics:")
    for cls, m in per_class.items():
        print(f"  {cls:12s} — Precision: {m['precision']:.4f}  Recall: {m['recall']:.4f}  F1: {m['f1_score']:.4f}")
    print("\nMetrics saved to metrics.json")

if __name__ == '__main__':
    evaluate_model()
