import json
import os
import sys

import google.generativeai as genai
from datasets import load_dataset

# 🔌 Make sure we can import our own repo modules
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from distillation.scripts.utils_flowers import normalize_label


# --------------------------
# Helper: strip code fences if Gemini returns ```json ... ```
# --------------------------
def clean_json_text(text: str) -> str:
    text = text.strip()
    # If model wrapped output like ```json ... ``` then remove those fences
    if text.startswith("```"):
        # remove leading fence
        # e.g. ```json\n{ ... }\n```
        # step 1: drop first line up to first newline
        lines = text.splitlines()
        # remove first line (``` or ```json)
        if len(lines) >= 1 and lines[0].startswith("```"):
            lines = lines[1:]
        # if last line is ``` remove it
        if len(lines) >= 1 and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


# --------------------------
# 1️⃣ Configure Gemini 2.5 Pro
# --------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-pro")

# --------------------------
# 2️⃣ Define the teacher prompt
# --------------------------
PROMPT_TEXT = """
You are a botanist that classifies flowers.

Look at the provided image of a single flower or a small bunch of flowers.

Your job:
1. Carefully describe the visible characteristics: petal color, petal shape, petal arrangement, center of the flower, leaves/stem structure, and overall shape.
2. Use these observations to reason step by step about which flower it is.
3. Make a final conclusion about the most likely flower type.

Return ONLY valid JSON with EXACTLY these fields:
{
  "question": "Try to identify the type of this flower. Give me your thinking process and reasoning, then finally conclude.",
  "reasoning": "Your detailed thinking process and reasoning in 2-6 sentences. Mention visual evidence.",
  "final_answer": "Your final predicted flower type as a short common name (only the flower name, no extra words)."
}

Rules:
- Output MUST be valid JSON.
- Do not include any text before or after the JSON.
- You MUST give your best guess even if you are not 100% sure.
- Do not say 'I am not sure' or 'I cannot tell'. Always choose the most likely flower type.
"""

# --------------------------
# 3️⃣ Load dataset
# --------------------------
dataset = load_dataset("pufanyi/flowers102")["train"]
id2label = dataset.features["label"].names

# Pick one example to test
sample_index = 0  # you can change this to test other samples
example = dataset[sample_index]
image = example["image"]
label_id = example["label"]
gold_label_raw = id2label[label_id]
gold_label_norm = normalize_label(gold_label_raw)

print(f"🖼️  Sample index: {sample_index}")
print(f"Ground truth label: {gold_label_raw}")

# --------------------------
# 4️⃣ Send to Gemini
# --------------------------
try:
    response = model.generate_content([PROMPT_TEXT, image])
    raw_text = response.text
    print("\nRaw output from Gemini:\n", raw_text)
except Exception as e:
    print("❌ Gemini call failed:", e)
    sys.exit(1)

# --------------------------
# 5️⃣ Clean + parse Gemini JSON
# --------------------------
cleaned_text = clean_json_text(raw_text)

try:
    parsed = json.loads(cleaned_text)
except json.JSONDecodeError:
    print("\n❌ Failed to parse JSON after cleanup.")
    print("Cleaned text was:\n", cleaned_text)
    sys.exit(1)

reasoning = parsed.get("reasoning", "").strip()
pred_raw = parsed.get("final_answer", "").strip()
pred_norm = normalize_label(pred_raw)
match = pred_norm == gold_label_norm

# --------------------------
# 6️⃣ Display results
# --------------------------
print("\nParsed reasoning:\n", reasoning)
print(f"\nGemini prediction (raw): {pred_raw}")
print(f"Gemini prediction (norm): {pred_norm}")
print(f"Ground truth (raw): {gold_label_raw}")
print(f"Ground truth (norm): {gold_label_norm}")
print(f"Match: {match}")
