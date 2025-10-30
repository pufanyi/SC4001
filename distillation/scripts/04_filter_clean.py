import json
import os

INPUT_FILE = "distillation/outputs/distilled_train_raw.jsonl"
OUTPUT_FILE = "distillation/outputs/distilled_train_clean.jsonl"


def load_jsonl(path):
    """Load all lines from a JSONL file."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data, path):
    """Save list of dicts as JSONL."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_good_sample(row):
    """Check if a sample passes filtering rules."""
    # Rule 1: must have Gemini reasoning text
    if not row.get("teacher_reasoning"):
        return False
    # Rule 2: reasoning length should be reasonable
    if len(row["teacher_reasoning"]) < 40:
        return False
    # Rule 3: must be correct (Gemini prediction matches ground truth)
    if not row.get("match"):
        return False
    # Rule 4: avoid weird or empty predictions
    pred = row.get("teacher_prediction_norm", "")
    if not pred or len(pred.split()) > 5:
        return False
    return True


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        print("Make sure you've run 03_distill_train.py first.")
        return

    print(f"🔍 Loading data from {INPUT_FILE} ...")
    data = load_jsonl(INPUT_FILE)
    print(f"Total samples: {len(data)}")

    print("🧹 Filtering...")
    clean_data = [row for row in data if is_good_sample(row)]
    print(
        f"✅ Kept {len(clean_data)} clean samples ("
        f"{len(clean_data) / len(data) * 100:.2f}%)"
    )

    print(f"💾 Saving to {OUTPUT_FILE} ...")
    save_jsonl(clean_data, OUTPUT_FILE)
    print("✅ Done!")


if __name__ == "__main__":
    main()
