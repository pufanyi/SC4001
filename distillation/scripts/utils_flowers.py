"""
Utility functions for the Flowers102 dataset.
These will help load the data, get label mappings,
and normalize label text for comparison.
"""

from datasets import load_dataset


def load_flowers102():
    """
    Load the Oxford Flowers-102 dataset (pufanyi version from Hugging Face)
    and return the train/val/test splits along with label maps.
    """
    dataset = load_dataset("pufanyi/flowers102")
    train_ds = dataset["train"]
    val_ds = dataset["validation"]
    test_ds = dataset["test"]

    # The key is 'label' in this dataset (not 'labels')
    id2label = train_ds.features["label"].names
    label2id = {name: i for i, name in enumerate(id2label)}

    print("✅ Loaded Flowers102 dataset:")
    print(f"Train: {len(train_ds)} | Validation: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"Number of classes: {len(id2label)}")

    return train_ds, val_ds, test_ds, id2label, label2id


def normalize_label(name: str) -> str:
    """
    Normalize a label for text comparison.
    Converts underscores and hyphens to spaces, lowercases, strips spaces.
    """
    return name.replace("_", " ").replace("-", " ").strip().lower()


# Run a quick demo if this file is executed directly
if __name__ == "__main__":
    train_ds, val_ds, test_ds, id2label, label2id = load_flowers102()
    example = train_ds[0]
    label_id = example["label"]
    label_raw = id2label[label_id]
    label_norm = normalize_label(label_raw)
    print("\nExample label:")
    print(f"Raw: {label_raw}")
    print(f"Normalized: {label_norm}")
