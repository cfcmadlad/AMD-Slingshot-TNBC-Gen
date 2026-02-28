import os
import re
from datasets import load_dataset
from PIL import Image

# --- Configuration ---
OUTPUT_DIR = "/content/drive/MyDrive/AMD_TNBC/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TNBC_KEYWORDS = [
    "triple negative", "invasive ductal", "basal-like",
    "breast cancer", "carcinoma", "necrosis", "mitotic"
]

# --- Load and Filter Dataset ---
dataset = load_dataset("flaviagiammarino/path-vqa", split="train")

def is_tnbc_relevant(sample):
    question = sample.get("question", "").lower()
    answer = sample.get("answer", "").lower()
    return any(kw in question or kw in answer for kw in TNBC_KEYWORDS)

filtered = dataset.filter(is_tnbc_relevant)
print(f"Total samples after filtering: {len(filtered)}")

# --- Caption Cleaning ---
def clean_caption(question, answer):
    question = question.strip().rstrip("?").lower()
    answer = answer.strip().lower()
    if answer in ["yes", "no"]:
        polarity = "showing" if answer == "yes" else "not showing"
        return f"Histopathology slide of triple-negative breast cancer {polarity} {question}."
    else:
        return f"Histopathology slide of triple-negative breast cancer demonstrating {answer}."

# --- Save Images and Captions ---
for idx, sample in enumerate(filtered):
    img = sample["image"].convert("RGB").resize((512, 512), Image.LANCZOS)
    img.save(os.path.join(OUTPUT_DIR, f"tnbc_{idx:04d}.png"))

    caption = clean_caption(sample.get("question", ""), sample.get("answer", ""))
    with open(os.path.join(OUTPUT_DIR, f"tnbc_{idx:04d}.txt"), "w") as f:
        f.write(caption)

print(f"Saved and captioned {idx + 1} samples to {OUTPUT_DIR}")