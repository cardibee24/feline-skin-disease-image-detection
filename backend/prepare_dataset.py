
import os
import shutil
import random
import argparse
from pathlib import Path

def prepare_dataset(source_dir, output_dir='dataset', split_ratio=0.8):
    """
    Splits images from source_dir into train/val folders in output_dir.
    
    Expected source_dir structure:
    source_dir/
        class_1/
            img1.jpg
            ...
        class_2/
            img1.jpg
            ...
            
    Creates:
    output_dir/
        train/
            class_1/
            class_2/
        val/
            class_1/
            class_2/
    """
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return

    # Create train and val directories
    train_dir = output_path / 'train'
    val_dir = output_path / 'val'
    
    if output_path.exists():
        print(f"Warning: Output directory '{output_dir}' already exists. Merging/Overwriting...")
    
    # Get all class folders (subdirectories in source)
    classes = [d for d in source_path.iterdir() if d.is_dir()]
    
    if not classes:
        print(f"No class folders found in '{source_dir}'.")
        print("Please organize your images into subfolders named after their class (e.g., 'ringworm', 'healthy').")
        return

    print(f"Found classes: {[c.name for c in classes]}")
    
    for class_dir in classes:
        class_name = class_dir.name
        
        # Create destination folders for this class
        os.makedirs(train_dir / class_name, exist_ok=True)
        os.makedirs(val_dir / class_name, exist_ok=True)
        
        # Get all images in the class folder
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']:
            images.extend(list(class_dir.glob(ext)))
            
        # Shuffle images
        random.shuffle(images)
        
        # Split point
        split_idx = int(len(images) * split_ratio)
        
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        print(f"Processing '{class_name}': {len(train_images)} train, {len(val_images)} val")
        
        # Copy files
        for img in train_images:
            shutil.copy2(img, train_dir / class_name / img.name)
            
        for img in val_images:
            shutil.copy2(img, val_dir / class_name / img.name)
            
    print(f"\nDataset prepared successfully in '{output_dir}'!")
    print(f"Train path: {train_dir}")
    print(f"Val path:   {val_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare dataset for training.')
    parser.add_argument('--source', type=str, required=True, help='Path to the folder containing raw images organized by class')
    parser.add_argument('--output', type=str, default='dataset', help='Path to the output dataset folder (default: dataset)')
    parser.add_argument('--split', type=float, default=0.8, help='Train/Val split ratio (default: 0.8)')
    
    args = parser.parse_args()
    
    prepare_dataset(args.source, args.output, args.split)
