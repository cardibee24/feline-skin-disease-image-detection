import torch
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import numpy as np
import json
import os
from clahe_transform import CLAHETransform
from stage2_dataset import BinarySubsetDataset
from evaluate import build_model, evaluate_single, save_confusion_matrix_image, EVAL_OUTPUT_DIR

STAGE1_ARCH = 'mobilenetv2'
STAGE1_MODEL_PATH = 'mobilenetv2_clahe.pth'
STAGE2_MODEL_PATH = 'mobilenetv2_stage2_flea_ringworm.pth'
STAGE2_CLASS_NAMES_PATH = 'stage2_class_names.txt'
CLASS_NAMES_PATH = 'class_names.txt'


def build_val_transform():
    return transforms.Compose([
        CLAHETransform(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def evaluate_binary_sanity(data_dir, device):
    """Standalone accuracy of the stage-2 specialist on the flea/ringworm binary subset."""
    val_transform = build_val_transform()
    base_val = datasets.ImageFolder(os.path.join(data_dir, 'val'), val_transform)
    binary_val = BinarySubsetDataset(base_val, 'flea', 'ringworm')
    val_loader = torch.utils.data.DataLoader(binary_val, batch_size=32, shuffle=False, num_workers=0)

    result = evaluate_single('mobilenetv2', STAGE2_MODEL_PATH, val_loader, binary_val,
                              list(binary_val.classes), device, file_tag='stage2_binary')

    with open('metrics_stage2_binary.json', 'w') as f:
        json.dump(result, f, indent=2)
    print("Saved metrics_stage2_binary.json")
    return result


def evaluate_pipeline(data_dir, device):
    """End-to-end two-stage accuracy on the full 4-class validation set,
    compared against the stage-1-only baseline."""
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    with open(STAGE2_CLASS_NAMES_PATH, 'r') as f:
        stage2_class_names = [line.strip() for line in f.readlines()]

    val_transform = build_val_transform()
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    stage1 = build_model(STAGE1_ARCH, len(class_names))
    stage1.load_state_dict(torch.load(STAGE1_MODEL_PATH, map_location=device))
    stage1.to(device)
    stage1.eval()

    stage2 = build_model('mobilenetv2', len(stage2_class_names))
    stage2.load_state_dict(torch.load(STAGE2_MODEL_PATH, map_location=device))
    stage2.to(device)
    stage2.eval()

    flea_idx = class_names.index('flea')
    ring_idx = class_names.index('ringworm')
    s2_flea_idx = stage2_class_names.index('flea')
    s2_ring_idx = stage2_class_names.index('ringworm')

    all_preds = []
    all_labels = []
    all_probs = []
    trigger_count = 0
    trigger_correct_before = 0
    trigger_correct_after = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            s1_probs = torch.nn.functional.softmax(stage1(inputs), dim=1)
            top2_vals, top2_idx = torch.topk(s1_probs, 2, dim=1)
            top1_idx, top2nd_idx = top2_idx[:, 0], top2_idx[:, 1]
            is_pair = (
                ((top1_idx == flea_idx) & (top2nd_idx == ring_idx)) |
                ((top1_idx == ring_idx) & (top2nd_idx == flea_idx))
            )

            s2_probs = torch.nn.functional.softmax(stage2(inputs), dim=1)

            combined_mass = s1_probs[:, flea_idx] + s1_probs[:, ring_idx]
            s2_flea = s2_probs[:, s2_flea_idx]
            s2_ring = s2_probs[:, s2_ring_idx]
            denom = (s2_flea + s2_ring).clamp(min=1e-8)
            new_flea = combined_mass * (s2_flea / denom)
            new_ring = combined_mass * (s2_ring / denom)

            final_probs = s1_probs.clone()
            final_probs[:, flea_idx] = torch.where(is_pair, new_flea, s1_probs[:, flea_idx])
            final_probs[:, ring_idx] = torch.where(is_pair, new_ring, s1_probs[:, ring_idx])

            final_preds = torch.argmax(final_probs, dim=1)
            s1_preds = torch.argmax(s1_probs, dim=1)

            trigger_count += int(is_pair.sum().item())
            trigger_correct_before += int(((s1_preds == labels) & is_pair).sum().item())
            trigger_correct_after += int(((final_preds == labels) & is_pair).sum().item())

            all_preds.extend(final_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(final_probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = float(np.mean(all_preds == all_labels))
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    try:
        roc_auc = float(roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro'))
    except Exception:
        roc_auc = None

    cm = confusion_matrix(all_labels, all_preds)
    save_confusion_matrix_image(cm, class_names, 'stage2_pipeline')
    cm_data = {"labels": class_names, "matrix": cm.tolist()}

    image_paths = [s[0] for s in val_dataset.samples]
    misclassified = []
    for i in range(len(all_preds)):
        if all_preds[i] != all_labels[i]:
            misclassified.append({
                "file": os.path.basename(image_paths[i]),
                "true_label": class_names[all_labels[i]],
                "predicted_label": class_names[all_preds[i]],
                "confidence": round(float(all_probs[i][all_preds[i]]), 4),
            })
    misclassified_path = os.path.join(EVAL_OUTPUT_DIR, 'misclassified_stage2_pipeline.json')
    with open(misclassified_path, 'w') as f:
        json.dump(misclassified, f, indent=2)
    print(f"  Misclassified: {len(misclassified)}/{len(all_labels)} saved to {misclassified_path}")

    per_class = {}
    for cls in class_names:
        per_class[cls] = {
            "precision": round(report[cls]["precision"], 4),
            "recall": round(report[cls]["recall"], 4),
            "f1_score": round(report[cls]["f1-score"], 4),
            "support": int(report[cls]["support"]),
        }

    result = {
        "overall_accuracy": round(accuracy, 4),
        "macro_precision": round(report["macro avg"]["precision"], 4),
        "macro_recall": round(report["macro avg"]["recall"], 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "per_class": per_class,
        "confusion_matrix": cm_data,
        "misclassified_count": len(misclassified),
        "total_val_samples": len(all_labels),
        "trigger_count": trigger_count,
        "trigger_correct_before": trigger_correct_before,
        "trigger_correct_after": trigger_correct_after,
    }

    baseline_path = 'metrics_clahe.json'
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
        result["baseline_stage1_only"] = {
            "overall_accuracy": baseline.get("overall_accuracy"),
            "per_class": {
                "flea": baseline.get("per_class", {}).get("flea"),
                "ringworm": baseline.get("per_class", {}).get("ringworm"),
            },
        }

    with open('metrics_stage2_pipeline.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*40}")
    print(f"Two-stage pipeline accuracy: {accuracy*100:.2f}%")
    print(f"Trigger count: {trigger_count} / {len(all_labels)}")
    print(f"Trigger correct before stage-2: {trigger_correct_before}  after: {trigger_correct_after}")
    print(f"Saved metrics_stage2_pipeline.json")
    print(f"{'='*40}")

    return result


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    evaluate_binary_sanity('dataset', device)
    evaluate_pipeline('dataset', device)
