from datasets import load_dataset


def main():
    # 1️⃣ Load the dataset from Hugging Face
    dataset = load_dataset("pufanyi/flowers102")

    # 2️⃣ Split into train/validation/test
    train_ds = dataset["train"]
    val_ds = dataset["validation"]
    test_ds = dataset["test"]

    print("✅ Dataset loaded successfully!")
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")

    # 3️⃣ Show the structure of one sample
    example = train_ds[0]
    print("\nExample sample keys:", example.keys())

    # The 'image' key contains a PIL image
    image = example["image"]
    print("Image type:", type(image))
    print("Image size:", image.size)

    # The 'labels' key gives a numeric class id
    label_id = example["label"]
    id2label = train_ds.features["label"].names
    label_name = id2label[label_id]
    print(f"Label ID: {label_id}, Label name: {label_name}")

    # (Optional) Display the image if you are in Jupyter or VSCode interactive mode
    # image.show()


if __name__ == "__main__":
    main()
