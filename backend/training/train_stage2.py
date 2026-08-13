import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import argparse
from clahe_transform import CLAHETransform
from stage2_dataset import BinarySubsetDataset
from train import build_model

STAGE2_CLASSES = ('flea', 'ringworm')


def train_stage2(data_dir='dataset', num_epochs=20, batch_size=32, learning_rate=0.001):
    data_transforms = {
        'train': transforms.Compose([
            CLAHETransform(),
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
            CLAHETransform(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        x: BinarySubsetDataset(
            datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]),
            label_a=STAGE2_CLASSES[0], label_b=STAGE2_CLASSES[1]
        )
        for x in ['train', 'val']
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                        shuffle=(x == 'train'), num_workers=4)
        for x in ['train', 'val']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*40}")
    print(f"Training: MOBILENETV2 (stage-2 flea/ringworm specialist)")
    print(f"Device  : {device}")
    print(f"Classes : {list(STAGE2_CLASSES)}")
    print(f"Train/Val sizes: {dataset_sizes}")
    print(f"{'='*40}\n")

    model = build_model('mobilenetv2', 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
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

    print(f'Best val Acc (stage-2 flea/ringworm): {best_acc:.4f}')
    model.load_state_dict(best_model_wts)

    os.makedirs('models', exist_ok=True)
    save_path = 'models/mobilenetv2_stage2_flea_ringworm.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    with open('models/stage2_class_names.txt', 'w') as f:
        for c in STAGE2_CLASSES:
            f.write(c + '\n')
    print("Class names saved to models/stage2_class_names.txt\n")


if __name__ == '__main__':
    if not os.path.exists('dataset/train') or not os.path.exists('dataset/val'):
        print("Error: dataset/train or dataset/val not found.")
        print("Run: python training/prepare_dataset.py --source raw_data --output dataset")
        exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    train_stage2(num_epochs=args.epochs)
    print("Stage-2 training complete.")
