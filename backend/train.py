import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import argparse

def build_model(model_name, num_classes):
    """Build and return a model with replaced final layer."""
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'mobilenetv2':
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model

HARD_CLASSES = {'flea', 'ringworm'}

def train_model(model_name='resnet18', data_dir='dataset', num_epochs=25, batch_size=32, learning_rate=0.001):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                   shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assign 1.5x loss weight to flea and ringworm to penalise their confusion harder
    class_weights = torch.tensor(
        [1.5 if c in HARD_CLASSES else 1.0 for c in class_names],
        dtype=torch.float
    ).to(device)

    print(f"\n{'='*40}")
    print(f"Training: {model_name.upper()}")
    print(f"Device  : {device}")
    print(f"Classes : {class_names}")
    print(f"Weights : {dict(zip(class_names, class_weights.tolist()))}")
    print(f"{'='*40}\n")

    model = build_model(model_name, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}', flush=True)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        exp_lr_scheduler.step()
        print()

    print(f'Best val Acc ({model_name}): {best_acc:.4f}')
    model.load_state_dict(best_model_wts)

    save_path = f'{model_name}.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}\n")

    # Save class names (same for all models)
    with open('class_names.txt', 'w') as f:
        for c in class_names:
            f.write(c + '\n')

if __name__ == '__main__':
    if not os.path.exists('dataset/train') or not os.path.exists('dataset/val'):
        print("Error: dataset/train or dataset/val not found.")
        print("Run: python prepare_dataset.py --source raw_data --output dataset")
        exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='all',
                        choices=['resnet18', 'resnet50', 'mobilenetv2', 'all'],
                        help='Model to train (default: all)')
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()

    models_to_train = ['resnet18', 'resnet50', 'mobilenetv2'] if args.model == 'all' else [args.model]

    for m in models_to_train:
        train_model(model_name=m, num_epochs=args.epochs)

    print("All training complete.")
