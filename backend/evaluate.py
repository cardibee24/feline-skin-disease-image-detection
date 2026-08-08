import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import numpy as np
import json
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from clahe_transform import CLAHETransform

ARCHS = ['resnet18', 'resnet50', 'mobilenetv2']

EVAL_OUTPUT_DIR = 'evaluation_results'

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

def save_confusion_matrix_image(cm, class_names, file_tag):
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center', fontsize=11,
                    color='white' if cm[i, j] > thresh else 'black')

    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_title(f'Confusion Matrix — {file_tag.upper()}', fontsize=13, pad=12)
    plt.tight_layout()

    path = os.path.join(EVAL_OUTPUT_DIR, f'confusion_matrix_{file_tag}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Confusion matrix image saved to {path}")

def evaluate_single(model_name, model_path, val_loader, val_dataset, class_names, device, file_tag=None):
    file_tag = file_tag or model_name
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

    try:
        roc_auc = float(roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro'))
    except Exception:
        roc_auc = None

    # Confusion matrix — rows=true label, cols=predicted label
    cm = confusion_matrix(all_labels, all_preds)
    save_confusion_matrix_image(cm, class_names, file_tag)
    cm_data = {
        "labels": class_names,
        "matrix": cm.tolist(),
    }

    # Misclassified samples — val_loader uses shuffle=False so index order is preserved
    image_paths = [s[0] for s in val_dataset.samples]
    misclassified = []
    for i in range(len(all_preds)):
        if all_preds[i] != all_labels[i]:
            misclassified.append({
                "file":            os.path.basename(image_paths[i]),
                "true_label":      class_names[all_labels[i]],
                "predicted_label": class_names[all_preds[i]],
                "confidence":      round(float(all_probs[i][all_preds[i]]), 4),
            })

    misclassified_path = os.path.join(EVAL_OUTPUT_DIR, f'misclassified_{file_tag}.json')
    with open(misclassified_path, 'w') as f:
        json.dump(misclassified, f, indent=2)
    print(f"  Misclassified: {len(misclassified)}/{len(all_labels)} saved to {misclassified_path}")

    per_class = {}
    for cls in class_names:
        per_class[cls] = {
            "precision": round(report[cls]["precision"], 4),
            "recall":    round(report[cls]["recall"],    4),
            "f1_score":  round(report[cls]["f1-score"],  4),
            "support":   int(report[cls]["support"]),
        }

    result = {
        "overall_accuracy":    round(accuracy, 4),
        "macro_precision":     round(report["macro avg"]["precision"], 4),
        "macro_recall":        round(report["macro avg"]["recall"],    4),
        "macro_f1":            round(report["macro avg"]["f1-score"],  4),
        "roc_auc":             round(roc_auc, 4) if roc_auc is not None else None,
        "per_class":           per_class,
        "confusion_matrix":    cm_data,
        "misclassified_count": len(misclassified),
        "total_val_samples":   len(all_labels),
    }

    print(f"  Accuracy : {accuracy*100:.2f}%")
    print(f"  Macro F1 : {result['macro_f1']}")
    print(f"  ROC-AUC  : {result['roc_auc']}")
    print(f"\n  Confusion matrix (rows=true, cols=predicted):")
    header = "           " + "  ".join(f"{c[:8]:>8}" for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {class_names[i][:8]:>8} : " + "  ".join(f"{v:>8}" for v in row))

    return result

def evaluate_all(data_dir='dataset', class_names_path='class_names.txt', use_clahe=False):
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    clahe_step = [CLAHETransform()] if use_clahe else []
    val_transform = transforms.Compose(clahe_step + [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), val_transform)
    val_loader  = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    device      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    suffix = '_clahe' if use_clahe else ''
    MODELS = {arch: f'{arch}{suffix}.pth' for arch in ARCHS}

    comparison = {}
    best_model = None
    best_f1    = -1

    for model_name, model_path in MODELS.items():
        if not os.path.exists(model_path):
            print(f"Skipping {model_name} — {model_path} not found.")
            continue
        result = evaluate_single(model_name, model_path, val_loader, val_dataset, class_names, device,
                                  file_tag=f'{model_name}{suffix}')
        comparison[model_name] = result
        if result["macro_f1"] > best_f1:
            best_f1    = result["macro_f1"]
            best_model = model_name

    if not comparison:
        print("No trained models found. Train at least one model first.")
        return

    best = comparison[best_model]
    metrics = {
        "overall_accuracy":    best["overall_accuracy"],
        "macro_precision":     best["macro_precision"],
        "macro_recall":        best["macro_recall"],
        "macro_f1":            best["macro_f1"],
        "roc_auc":             best["roc_auc"],
        "per_class":           best["per_class"],
        "confusion_matrix":    best["confusion_matrix"],
        "misclassified_count": best["misclassified_count"],
        "total_val_samples":   best["total_val_samples"],
        "best_model":          best_model,
        "epochs":              25,
        "dataset_split":       "80% Train / 20% Validation",
        "comparison":          comparison,
    }

    metrics_path = f'metrics{suffix}.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*40}")
    print(f"Best model: {best_model.upper()} (Macro F1: {best_f1:.4f})")
    print(f"Metrics saved to {metrics_path}")
    print(f"{'='*40}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clahe', action='store_true',
                        help='Evaluate the CLAHE-trained models (*_clahe.pth) with matching CLAHE preprocessing, writing metrics_clahe.json')
    args = parser.parse_args()
    evaluate_all(use_clahe=args.clahe)
