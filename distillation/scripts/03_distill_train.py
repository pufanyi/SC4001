import json
import os
import sys
import time
import uuid

import google.generativeai as genai
from datasets import load_dataset
from tqdm import tqdm

from distillation.scripts.utils_flowers import normalize_label  # noqa: E402

# ─────────────────────────────────────────
# Make sure we can import our own utilities
# ─────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ─────────────────────────────────────────
# Config: output + resume support
# ─────────────────────────────────────────
OUTPUT_DIR = "distillation/outputs"
FINAL_JSONL = os.path.join(OUTPUT_DIR, "distilled_train_raw.jsonl")
PROGRESS_JSON = os.path.join(OUTPUT_DIR, "progress.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────
# Helper: load & save progress
# ─────────────────────────────────────────
def load_progress():
    """
    Returns the next index we should start from.
    If no progress file exists, we start from 0.
    """
    if not os.path.exists(PROGRESS_JSON):
        return 0

    try:
        with open(PROGRESS_JSON, encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("next_index", 0))
    except Exception:
        return 0


def save_progress(next_index: int):
    """
    Writes the next index we should process into progress.json.
    """
    with open(PROGRESS_JSON, "w", encoding="utf-8") as f:
        json.dump({"next_index": next_index}, f, ensure_ascii=False)


# ─────────────────────────────────────────
# Helper: key rotation for Gemini
# ─────────────────────────────────────────
def load_api_keys():
    """
    Load keys from environment variable GEMINI_KEYS.
    Fallback: try GEMINI_API_KEY (single key).
    Fallback2: hardcode (not recommended to commit!).
    """
    # Preferred: multiple keys as comma-separated string
    if os.getenv("GEMINI_KEYS"):
        keys = [k.strip() for k in os.getenv("GEMINI_KEYS").split(",") if k.strip()]
        if keys:
            return keys

    # Fallback: single key via GEMINI_API_KEY
    if os.getenv("GEMINI_API_KEY"):
        return [os.getenv("GEMINI_API_KEY").strip()]

    # LAST RESORT (dev only): hardcode here if you really must
    # return ["key1", "key2", "key3"]
    raise RuntimeError(
        "No API keys found. Set GEMINI_KEYS='k1,k2,...' or GEMINI_API_KEY='k1'"
    )


API_KEYS = load_api_keys()
current_key_index = 0


def init_gemini_client():
    """
    Configure genai with the current key and return a model handle.
    """
    global current_key_index
    key = API_KEYS[current_key_index]
    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-2.5-pro")
    return model


model = init_gemini_client()


def rotate_key_and_retry():
    """
    Move to the next API key, wait a moment, and re-init the model.
    """
    global current_key_index, model
    current_key_index = (current_key_index + 1) % len(API_KEYS)
    print(f"\n⚠️  Quota hit. Switching to API key index {current_key_index} ...")
    # small sleep helps cool off both the model and avoid hammering
    time.sleep(10)
    model = init_gemini_client()


# ─────────────────────────────────────────
# Helper: clean Gemini output for JSON
# ─────────────────────────────────────────
def clean_json_text(text: str) -> str:
    text = text.strip()
    # Handle ```json ... ``` and ``` ... ``` fences
    if text.startswith("```"):
        lines = text.splitlines()
        # drop first fence line (``` or ```json)
        if len(lines) >= 1 and lines[0].startswith("```"):
            lines = lines[1:]
        # drop trailing fence line if present
        if len(lines) >= 1 and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


# ─────────────────────────────────────────
# Prompt for Gemini (Version A)
# ─────────────────────────────────────────
PROMPT_TEXT = """
You are a botanist that classifies flowers.

Look at the provided image of a single flower or a small bunch of flowers.

Your job:
1. Carefully describe the visible characteristics: petal color, petal shape,
   petal arrangement, center of the flower, leaves/stem structure, and overall
   shape.
2. Use these observations to reason step by step about which flower it is.
3. Make a final conclusion about the most likely flower type.

Return ONLY valid JSON with EXACTLY these fields:
{
  "question": "Try to identify the type of this flower. Give me your thinking "
              "process and reasoning, then finally conclude.",
  "reasoning": "Your detailed thinking process and reasoning in 2-6 sentences. "
               "Mention visual evidence.",
  "final_answer": "Your final predicted flower type as a short common name "
                  "(only the flower name, no extra words)."
}

Rules:
- Output MUST be valid JSON.
- Do not include any text before or after the JSON.
- You MUST give your best guess even if you are not 100% sure.
- Do not say 'I am not sure' or 'I cannot tell'. Always choose the most likely
  flower type.
"""


# ─────────────────────────────────────────
# Main distillation logic
# ─────────────────────────────────────────
def main():
    # 1. Load dataset
    full_dataset = load_dataset("pufanyi/flowers102")
    train_ds = full_dataset["train"]
    id2label = train_ds.features["label"].names
    total = len(train_ds)
    print(f"Loaded train split with {total} samples")

    # 2. Figure out where to resume
    start_index = load_progress()
    print(f"Resuming from index {start_index}")

    # 3. Open output JSONL in append mode
    # If this is a fresh run (start_index == 0 and file exists), you can choose:
    # - either delete it manually before running
    # - or keep appending (will duplicate if you rerun from 0)
    out_f = open(FINAL_JSONL, "a", encoding="utf-8")

    # 4. Iterate
    for idx in tqdm(range(start_index, total), desc="Distilling train split"):
        row = train_ds[idx]
        image = row["image"]  # PIL image
        label_id = row["label"]  # numeric label
        gold_label_raw = id2label[label_id]  # e.g. "wallflower"
        gold_label_norm = normalize_label(gold_label_raw)

        # ---- 4a. Call Gemini with retries and key rotation ----
        raw_text = None
        for _attempt in range(len(API_KEYS)):
            try:
                response = model.generate_content([PROMPT_TEXT, image])
                raw_text = response.text
                break  # success
            except Exception as e:
                msg = str(e)
                # Quota or rate limit? -> rotate key, retry
                if "429" in msg or "quota" in msg.lower() or "rate" in msg.lower():
                    rotate_key_and_retry()
                    continue
                # Other error? -> log & skip this sample
                print(f"[idx {idx}] Gemini call failed (non-quota): {e}")
                raw_text = None
                break

        if not raw_text:
            # couldn't get output from any key, skip this index but continue with next
            continue

        # ---- 4b. Clean + JSON parse ----
        cleaned = clean_json_text(raw_text)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"[idx {idx}] JSON parse failed: {e}")
            # Debug? You can uncomment this:
            # print("CLEANED:\n", cleaned)
            continue

        teacher_reasoning = parsed.get("reasoning", "").strip()
        teacher_pred_raw = parsed.get("final_answer", "").strip()
        teacher_pred_norm = normalize_label(teacher_pred_raw)
        match = teacher_pred_norm == gold_label_norm

        # ---- 4c. Build the record for this sample ----
        record = {
            "uid": str(uuid.uuid4()),
            "split": "train",
            "image_index": idx,
            "gold_label_raw": gold_label_raw,
            "gold_label_norm": gold_label_norm,
            "question": parsed.get("question", ""),
            "teacher_reasoning": teacher_reasoning,
            "teacher_prediction_raw": teacher_pred_raw,
            "teacher_prediction_norm": teacher_pred_norm,
            "match": match,
        }

        # ---- 4d. Write this record as one line of JSONL immediately ----
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_f.flush()  # force write to disk, safer if you kill the run

        # ---- 4e. Update progress so we can resume later ----
        # next_index means: we've finished idx, next time start at idx+1
        save_progress(idx + 1)

    out_f.close()
    print(f"✅ Done. Distilled data written to {FINAL_JSONL}")
    print(f"   Progress saved to {PROGRESS_JSON}")


if __name__ == "__main__":
    main()
