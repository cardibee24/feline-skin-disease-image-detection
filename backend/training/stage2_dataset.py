import torch


class BinarySubsetDataset(torch.utils.data.Dataset):
    """Wraps a 4-class ImageFolder dataset, keeping only two named classes
    and remapping their labels to 0/1 (label_a -> 0, label_b -> 1)."""

    def __init__(self, base_dataset, label_a='flea', label_b='ringworm'):
        self.base_dataset = base_dataset
        self.classes = [label_a, label_b]

        orig_idx_a = base_dataset.class_to_idx[label_a]
        orig_idx_b = base_dataset.class_to_idx[label_b]
        remap = {orig_idx_a: 0, orig_idx_b: 1}

        self.indices = [
            i for i, (_, label) in enumerate(base_dataset.samples)
            if label in remap
        ]
        self.remap = remap
        self.samples = [
            (base_dataset.samples[i][0], remap[base_dataset.samples[i][1]])
            for i in self.indices
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, orig_label = self.base_dataset[self.indices[idx]]
        return image, self.remap[orig_label]
